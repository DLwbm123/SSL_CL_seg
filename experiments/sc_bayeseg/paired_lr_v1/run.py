"""Resume one exact source checkpoint twice; change only the effective LR."""
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
from care_seg_study_v1.run import write, append, save
from sc_bayeseg.model import Model
from sc_bayeseg.run import snapshot, restore, same
from sc_bayeseg.diagnostic_v1.run import DiagnosticMethod, dice, logits


def evaluate(model,image,y):
    rng=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
    try:
        seed_everything(50163)
        with torch.no_grad():
            probabilities=[logits(model.proposal(image)[0]).softmax(1) for _ in range(4)]
            p=torch.stack(probabilities).mean(0)
            direct=logits(model.proposal(image,sample=False)[0]).softmax(1)
            result={}
            for mode,prob in [('stochastic',p),('direct',direct)]:
                pred=prob.argmax(1)
                result[mode+'_dice']=dice(pred,y)
                result[mode+'_ce']=float(F.nll_loss(prob.clamp_min(1e-30).log(),y,reduction='none').mean())
                result[mode+'_background_fraction']=float((pred==0).float().mean())
                result[mode+'_recall']=[[float(((pred[i]==k)&(y[i]==k)).sum()/(y[i]==k).sum()) for k in (1,2,3)] for i in range(len(y))]
            return result
    finally:
        torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda)


def run(settings):
    root=Path(settings['output'])
    # The output already contains only private overlay preparation and its review receipt.
    assert json.loads((root/'OVERLAY_REVIEW.json').read_text())['status']=='PASS_VISUAL_SCREEN'
    with (root/'RUN_STARTED.json').open('x') as f:json.dump(dict(started=time.time()),f)
    write(root/'CONTEXT.json',settings)
    costs=dict(diagnostic=0,engineering=0);active={};started=time.time()
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        old=Path(settings['origin']);payload=torch.load(old/'D'/'final.pt',map_location='cpu',weights_only=False)
        assert payload['cursor']==600 and payload['population']==dict(labeled=230,unlabeled=0)
        assert {int(v['step']) for v in payload['trainer']['optimizer_state']['state'].values()}=={600}
        dataset=H5LabeledDataset(settings['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        selection=json.loads((root/'PRIVATE_SELECTION.json').read_text())
        lb=collate_labeled([dataset[i] for i in selection['indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False)
        y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        original_metric=json.loads((old/'D'/'metrics.jsonl').read_text().splitlines()[-1])
        results={}
        for name,rate in [('L1',1e-4),('L3',3e-4)]:
            out=root/name;out.mkdir(exist_ok=False)
            seed_everything(163);model=Model().cuda();method=DiagnosticMethod(model,230,'mean_ce')
            method.begin_site('Siemens',None,1800)
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=0)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
            trainer=Trainer(method,optimizer=optimizer,scheduler=scheduler,device='cuda',amp=False,grad_clip_norm=1.)
            restore(payload,method,trainer)
            assert same(payload['model'],model.state_dict())
            assert same(payload['trainer']['optimizer_state']['state'],optimizer.state_dict()['state'])
            assert same(payload['trainer'],trainer.state_dict())
            # Preserve Adam history, epoch and scheduling function; change its constant LR consistently.
            for group in optimizer.param_groups:group['lr']=rate;group['initial_lr']=rate
            scheduler.base_lrs=[rate for _ in optimizer.param_groups]
            scheduler._last_lr=[rate for _ in optimizer.param_groups]
            baseline=evaluate(model,image,y)
            assert all(baseline[k]==original_metric[k] for k in baseline if not k.endswith('_recall'))
            write(out/'RESTORE.json',dict(status='PASS',source_cursor=600,source_adam_steps=[600],model_equal=True,
                                          adam_moments_equal=True,optimizer_and_scheduler_equal_before_lr_change=True,
                                          baseline_predictions_equal=True,lr=rate,scheduler_base_lrs=scheduler.base_lrs))
            append(out/'metrics.jsonl',dict(added_step=0,cumulative_step=600,**baseline))
            initial=snapshot(method,trainer,600)
            prior={k:v.clone() for k,v in model.state_dict().items() if 'prior_' in k or 'initial_' in k}
            local=dict(diagnostic=0,engineering=0);phase=['engineering'];norms=[]
            active.update(method=method,trainer=trainer,out=out,cursor=600)
            def guard(opt,args,kwargs):
                assert sum(costs.values())<2404
                key=phase[0]
                assert costs[key]<(4 if key=='engineering' else 2400)
                assert local[key]<(2 if key=='engineering' else 1200)
                assert all(g['lr']==rate for g in opt.param_groups)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            def receipt(opt,args,kwargs):
                key=phase[0];local[key]+=1;costs[key]+=1
                append(root/'physical_updates.jsonl',dict(physical=sum(costs.values()),kind=key,kind_physical=costs[key],
                       arm=name,local=local[key],source_cursor=600,time=time.time()))
            pre=optimizer.register_step_pre_hook(guard);post=optimizer.register_step_post_hook(receipt)
            original_clip=torch.nn.utils.clip_grad_norm_
            def clip(parameters,max_norm,*args,**kwargs):
                value=original_clip(parameters,max_norm,*args,**kwargs);norms.append(float(value));return value
            torch.nn.utils.clip_grad_norm_=clip
            try:
                def step(index):
                    result=trainer.train_step(lb,None,state=TrainerState(global_step=index,site_step=index))
                    assert optimizer.param_groups[0]['lr']==rate and scheduler.get_last_lr()==[rate]
                    return result
                first=step(600);expected=snapshot(method,trainer,601);restore(initial,method,trainer)
                second=step(600)
                assert torch.equal(first.total_loss,second.total_loss) and same(expected,snapshot(method,trainer,601))
                restore(initial,method,trainer)
                write(out/'QUALIFICATION.json',dict(status='PASS',engineering_updates=2,full_state_golden_equal=True))
                phase[0]='diagnostic'
                for added in range(1,1201):
                    result=step(600+added-1);active['cursor']=600+added
                    if added==1 or added%25==0:
                        row=dict(arm=name,added_step=added,cumulative_step=600+added,loss=float(result.total_loss.detach()),
                                 gradient_norm_before_clip=norms[-1],**result.scalars)
                        append(out/'telemetry.jsonl',row);write(root/'STATUS.json',dict(status='RUNNING',costs=costs.copy(),**row))
                    if added in (25,100,300,600,900,1200):append(out/'metrics.jsonl',dict(added_step=added,cumulative_step=600+added,**evaluate(model,image,y)))
                    if added%100==0:save(out/'latest.pt',snapshot(method,trainer,600+added))
                assert local==dict(diagnostic=1200,engineering=2)
                assert all(torch.equal(v,model.state_dict()[k]) for k,v in prior.items())
                save(out/'final.pt',snapshot(method,trainer,1800))
                metric=json.loads((out/'metrics.jsonl').read_text().splitlines()[-1])
                metric.update(status='COMPLETE',lr=rate,updates=local,prior_unchanged=True,calls=model.calls.copy(),
                              main_pass=all(v>=.8 for row in metric['stochastic_dice'] for v in row),
                              direct_pass=all(v>=.8 for row in metric['direct_dice'] for v in row),
                              clipping_fraction=sum(v>1 for v in norms)/len(norms),gradient_norm_min=min(norms),gradient_norm_max=max(norms))
                write(out/'RESULTS.json',metric);results[name]=metric
            finally:
                torch.nn.utils.clip_grad_norm_=original_clip;pre.remove();post.remove()
            active.clear();del model,method,trainer;torch.cuda.empty_cache()
        assert costs==dict(diagnostic=2400,engineering=4)
        complete=dict(status='COMPLETE',started=started,finished=time.time(),costs=costs,results=results)
        write(root/'RESULTS.json',complete);write(root/'STATUS.json',complete)
    except BaseException:
        failure=dict(status='FAILED',costs=costs,traceback=traceback.format_exc())
        if active:
            save(active['out']/'failure_partial.pt',snapshot(active['method'],active['trainer'],active['cursor']))
            failure['partial_saved']=True
        write(root/'FAILURE.json',failure);write(root/'STATUS.json',failure)
        raise


if __name__=='__main__':
    settings=json.loads(Path(os.environ['TASK_SETTINGS']).read_text());root=Path(settings['output'])
    if (root/'RUN_STARTED.json').exists():raise FileExistsError('create-only training already started')
    code=1
    try:run(settings);code=0
    finally:write(root/'EXIT.json',dict(exit_code=code,time=time.time()))
