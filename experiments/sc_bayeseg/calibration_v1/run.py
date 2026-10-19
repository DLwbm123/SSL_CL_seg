"""Bounded generator calibration followed by joint unfreezing; create only."""
import copy
import json
import os
import time
import traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset, collate_labeled
from lcrseg.engine.trainer import Trainer, TrainerState, seed_everything
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from care_seg_study_v1.run import write, append, save
from sc_bayeseg.model import Model
from sc_bayeseg.run import snapshot, restore, same
from sc_bayeseg.paired_lr_v1.run import evaluate as _evaluate
from sc_bayeseg.jointwrite_v1.run import Adapter, posterior_swap


def generator(name):
    return name.startswith(('anatomy.','structure.'))


def evaluate(model,image,y):
    rng=capture_rng_state()
    try:return _evaluate(model,image,y)
    finally:restore_rng_state(rng)


def density(model,image,y,nl=2):
    rng=capture_rng_state();rows=[]
    try:
        with torch.no_grad():
            for draw in range(8):
                seed_everything(90163+draw);model.draw()
                try:
                    bound,terms,_=model.bound(image,y);kl=float(model.kl())
                    row={k:float(-v.mean()/128**2) for k,v in terms.items()}
                    row.update(logq=float((sum(terms.values())-bound).mean()/128**2),
                               negative_bound=float(-bound.mean()/128**2),kl=kl,
                               total=float(-bound.mean()/128**2+kl/(nl*128**2)))
                    assert all(torch.isfinite(torch.tensor(v)) for v in row.values());rows.append(row)
                finally:model.clear_draw()
        return {k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}
    finally:
        restore_rng_state(rng)


def run(settings):
    root=Path(settings['output'])
    with (root/'RUN_STARTED.json').open('x') as f:json.dump(dict(time=time.time()),f)
    write(root/'CONTEXT.json',settings)
    costs=dict(diagnostic=0,engineering=0);active={};started=time.time();prior_cost=0
    try:
        prior_attempt=Path(settings['prior_attempt'])
        history=[json.loads(v) for v in (prior_attempt/'physical_updates.jsonl').read_text().splitlines()]
        assert [v['physical'] for v in history]==[1,2]
        assert all(v['kind']=='engineering' for v in history)
        assert json.loads((prior_attempt/'FAILURE.json').read_text())['costs']==dict(diagnostic=0,engineering=2)
        assert json.loads((prior_attempt/'EXIT.json').read_text())['exit_code']==1
        prior_cost=len(history);assert prior_cost==settings['prior_actual_updates']==2
        write(root/'CUMULATIVE_BUDGET.json',dict(prior_actual_updates=prior_cost,new_run_cap=1204,cumulative_cap=1206,prior_ledger_continuous=True))
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        origin=Path(settings['origin']);previous=Path(settings['previous'])
        assert json.loads((previous/'EXIT.json').read_text())['exit_code']==0 and not (previous/'FAILURE.json').exists()
        assert json.loads((previous/'OVERLAY_REVIEW.json').read_text())['status']=='PASS_VISUAL_SCREEN'
        payload=torch.load(origin/'L3'/'final.pt',map_location='cpu',weights_only=False)
        assert payload['cursor']==1800 and payload['population']==dict(labeled=230,unlabeled=0)
        data=H5LabeledDataset(settings['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        assert len(data)==230
        sel=json.loads((previous/'PRIVATE_SELECTION.json').read_text());assert len(set(sel['identities']))==4
        write(root/'PRIVATE_SELECTION.json',sel)
        lb=collate_labeled([data[i] for i in sel['train_indices']]).to('cuda')
        lb4=collate_labeled([data[i] for i in sel['indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False)
        y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        seed_everything(163);model=Model().cuda();method=Adapter(model,'joint');method.begin_site('Siemens',None,3000)
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=0)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        trainer=Trainer(method,optimizer=optimizer,scheduler=scheduler,device='cuda',amp=False,grad_clip_norm=1.)
        restore(payload,method,trainer)
        assert same(payload['model'],model.state_dict()) and same(payload['trainer'],trainer.state_dict())
        method.nl=2
        original=json.loads((origin/'L3'/'metrics.jsonl').read_text().splitlines()[-1])
        baseline=evaluate(model,image,y);assert all(baseline[k]==original[k] for k in baseline)
        write(root/'RESTORE.json',dict(status='PASS',source_cursor=1800,model_equal=True,trainer_equal=True,baseline_equal=True))
        prior={k:v.clone() for k,v in model.state_dict().items() if 'prior_' in k or 'initial_' in k}
        results={};swaps={};norms=[];phase=['engineering'];name=['K'];local={}
        def guard(opt,args,kwargs):
            kind=phase[0]
            assert prior_cost+sum(costs.values())<1206
            assert sum(costs.values())<1204 and costs[kind]<(4 if kind=='engineering' else 1200)
            assert local[kind]<(2 if kind=='engineering' else 600)
            assert all(g['lr']==3e-4 for g in opt.param_groups)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if name[0]=='K':assert all(p.grad is None for n,p in model.named_parameters() if not generator(n))
        def receipt(opt,args,kwargs):
            kind=phase[0];costs[kind]+=1;local[kind]+=1
            append(root/'physical_updates.jsonl',dict(physical=sum(costs.values()),cumulative_physical=prior_cost+sum(costs.values()),kind=kind,kind_physical=costs[kind],stage=name[0],local=local[kind],time=time.time()))
        pre=optimizer.register_step_pre_hook(guard);post=optimizer.register_step_post_hook(receipt)
        original_clip=torch.nn.utils.clip_grad_norm_
        def clip(parameters,max_norm,*args,**kwargs):
            v=original_clip(parameters,max_norm,*args,**kwargs);norms.append(float(v));return v
        torch.nn.utils.clip_grad_norm_=clip
        try:
            for stage in ['K','T']:
                name[0]=stage;phase[0]='engineering';local.clear();local.update(diagnostic=0,engineering=0)
                out=root/stage;out.mkdir(exist_ok=False);cursor=1800 if stage=='K' else 2400
                for n,p in model.named_parameters():p.requires_grad_(stage=='T' or generator(n))
                optimizer.zero_grad(set_to_none=True)
                initial=snapshot(method,trainer,cursor);save(out/'initial.pt',initial)
                frozen={n:p.detach().clone() for n,p in model.named_parameters() if not generator(n)}
                old_state=copy.deepcopy(optimizer.state_dict()['state'])
                stage_baseline=evaluate(model,image,y)
                append(out/'metrics.jsonl',dict(added_step=0,cumulative_step=cursor,**stage_baseline))
                write(out/'DENSITY_INITIAL.json',density(model,image,y))
                active.update(out=out,method=method,trainer=trainer,cursor=cursor,stage=stage)
                def step(index):
                    return trainer.train_step(lb,None,state=TrainerState(global_step=index,site_step=index))
                a=step(cursor);expected=snapshot(method,trainer,cursor+1);restore(initial,method,trainer)
                b=step(cursor);observed=snapshot(method,trainer,cursor+1)
                comparison=dict(loss_equal=torch.equal(a.total_loss,b.total_loss),model_equal=same(expected['model'],observed['model']),trainer_equal=same(expected['trainer'],observed['trainer']),rng_equal={k:same(v,observed['rng'][k]) for k,v in expected['rng'].items()})
                write(out/'GOLDEN_COMPARISON.json',comparison)
                assert comparison['loss_equal'] and same(expected,observed)
                restore(initial,method,trainer)
                write(out/'QUALIFICATION.json',dict(status='PASS',engineering_updates=2,full_state_golden_equal=True,phi_frozen=stage=='K'))
                phase[0]='diagnostic';start_norm=len(norms)
                for added in range(1,601):
                    r=step(cursor+added-1);active['cursor']=cursor+added
                    if added==1 or added%25==0:
                        row=dict(stage=stage,added_step=added,cumulative_step=cursor+added,loss=float(r.total_loss.detach()),gradient_norm_before_clip=norms[-1],**r.scalars)
                        append(out/'telemetry.jsonl',row);write(root/'STATUS.json',dict(status='RUNNING',costs=costs.copy(),**row))
                    if added in (25,100,300,600):append(out/'metrics.jsonl',dict(added_step=added,cumulative_step=cursor+added,**evaluate(model,image,y)))
                    if added%100==0:save(out/'latest.pt',snapshot(method,trainer,cursor+added))
                assert local==dict(diagnostic=600,engineering=2)
                assert all(torch.equal(v,model.state_dict()[k]) for k,v in prior.items())
                metric=json.loads((out/'metrics.jsonl').read_text().splitlines()[-1])
                if stage=='K':
                    assert all(torch.equal(v,dict(model.named_parameters())[k]) for k,v in frozen.items())
                    assert all(same(v,optimizer.state_dict()['state'][k]) for k,v in old_state.items())
                    assert all(metric[k]==stage_baseline[k] for k in baseline)
                    write(out/'FROZEN_AUDIT.json',dict(status='PASS',all_phi_unchanged=True,original_adam_states_unchanged=True,predictions_equal=True))
                save(out/'final.pt',snapshot(method,trainer,cursor+600))
                write(out/'DENSITY_FINAL.json',density(model,image,y))
                metric.update(status='COMPLETE',updates=local.copy(),main_pass=all(v>=.8 for row in metric['stochastic_dice'] for v in row),
                              clipping_fraction=sum(v>1 for v in norms[start_norm:])/600,gradient_norm_max=max(norms[start_norm:]))
                write(out/'RESULTS.json',metric);results[stage]=metric
                # The swap utility reseeds; preserve the uninterrupted training RNG across diagnostics.
                saved_rng=snapshot(method,trainer,cursor+600)
                write(root/'STATUS.json',dict(status='POSTERIOR_SWAP',stage=stage,costs=costs.copy()))
                swaps[stage]=posterior_swap(out,payload,saved_rng,lb4,True)
                restore(saved_rng,method,trainer)
                active.clear()
        finally:
            torch.nn.utils.clip_grad_norm_=original_clip;pre.remove();post.remove()
        assert costs==dict(diagnostic=1200,engineering=4)
        complete=dict(status='COMPLETE',started=started,finished=time.time(),costs=costs,prior_actual_updates=prior_cost,cumulative_actual_updates=prior_cost+sum(costs.values()),results=results,swaps=swaps,calls=model.calls.copy())
        write(root/'RESULTS.json',complete);write(root/'STATUS.json',complete)
    except BaseException:
        failure=dict(status='FAILED',costs=costs.copy(),prior_actual_updates=prior_cost,cumulative_actual_updates=prior_cost+sum(costs.values()),traceback=traceback.format_exc())
        if active:
            save(active['out']/'failure_partial.pt',snapshot(active['method'],active['trainer'],active['cursor']));failure['partial_saved']=True
        write(root/'FAILURE.json',failure);write(root/'STATUS.json',failure);raise


if __name__=='__main__':
    settings=json.loads(Path(os.environ['TASK_SETTINGS']).read_text());root=Path(settings['output'])
    if (root/'RUN_STARTED.json').exists():raise FileExistsError('create-only training already started')
    code=1
    try:run(settings);code=0
    finally:write(root/'EXIT.json',dict(exit_code=code,time=time.time()))
