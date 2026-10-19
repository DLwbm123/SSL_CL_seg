"""Qualified full-view inference: frozen-semantic calibration and paired C/J."""
import copy,json,os,time,traceback
from dataclasses import replace
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import Trainer,TrainerState,seed_everything
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.split_v1.model import Model,semantic
from sc_bayeseg.jointwrite_v1.run import Adapter
from sc_bayeseg.run import snapshot,restore,same
from sc_bayeseg.guard_v1.run import frame
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage


def density(model,batch):
    rng=capture_rng_state();rows=[]
    try:
        with torch.no_grad():
            for seed in (50163,80163):
                seed_everything(seed);draws=[]
                for _ in range(2):
                    model.draw()
                    try:
                        bound,terms,_=model.bound(batch.image,batch.label);kl=model.kl()
                        row=dict(joint=float(-bound.mean()/128**2+kl/(2*128**2)),kl=float(kl/(2*128**2)),
                                 logq=float((sum(terms.values())-bound).mean()/128**2),
                                 **{k:float(-v.mean()/128**2) for k,v in terms.items()})
                        assert all(torch.isfinite(torch.tensor(v)) for v in row.values());draws.append(row)
                    finally:model.clear_draw()
                rows.append(dict(seed=seed,draws=draws))
        return rows
    finally:restore_rng_state(rng)


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time()
    costs=dict(optimizer=0,main=0,golden=0);results={};active={};paired={}
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        fit=json.loads((Path(s['fit'])/'RESULTS.json').read_text())
        assert fit['arms']['fullview']['training_semantic_qualified']
        repair=json.loads((Path(s['fit'])/'cursor_repair_v1/RECEIPT.json').read_text());assert repair['status']=='PASS'
        payload=torch.load(s['checkpoint'],map_location='cpu',weights_only=False)
        assert payload['cursor']==6775 and payload['population']==dict(labeled=2,unlabeled=0)
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());assert len(set(sel['identities']))==4
        assert not set(sel['train_indices'])&set(sel['heldout_indices'])
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda')
        lb4=collate_labeled([ds[i] for i in sel['train_indices']+sel['heldout_indices']]).to('cuda')
        image=F.interpolate(lb4.image,(128,128),mode='bilinear',align_corners=False)
        y=F.interpolate(lb4.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        valid=F.interpolate(lb.valid_mask.float(),(128,128),mode='nearest').bool()
        batch=replace(lb,image=torch.cat([flip(image[:2],k) for k in range(4)]),
                      label=torch.cat([flip(y[:2],k) for k in range(4)]),valid_mask=torch.cat([flip(valid,k) for k in range(4)]),
                      case_id=lb.case_id*4,patient_id=lb.patient_id*4,site=lb.site*4,slice_index=lb.slice_index*4)
        assert len(batch.image)==8 and set(batch.patient_id)==set(lb.patient_id)

        def arm(name,source,steps):
            out=root/name;out.mkdir(exist_ok=False);base=source['cursor']
            seed_everything(163);model=Model().cuda();method=Adapter(model,'ce' if name=='C' else 'joint');method.nl=2
            method.begin_site('Siemens',None,base+steps)
            opt=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=0)
            scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
            trainer=Trainer(method,optimizer=opt,scheduler=scheduler,device='cuda',amp=False,grad_clip_norm=1.)
            restore(source,method,trainer)
            assert same(source,snapshot(method,trainer,base))
            named=list(model.named_parameters());assert len(named)==100
            enabled=lambda n: (not semantic(n)) if name=='K' else (semantic(n) if name=='C' else True)
            for n,p in named:p.requires_grad_(enabled(n))
            frozen={n:v.detach().clone() for n,v in model.state_dict().items() if not enabled(n) or 'prior_' in n or 'initial_' in n}
            frozen_adam={i:copy.deepcopy(opt.state_dict()['state'].get(i)) for i,(n,p) in enumerate(named) if not enabled(n)}
            phase=['golden'];index=[0];local=dict(main=0,golden=0);goldens=[];rngs=[]
            def pre(opt,args,kwargs):
                assert costs['optimizer']<812 and local[phase[0]]<(steps if phase[0]=='main' else 4)
                assert opt.param_groups[0]['lr']==3e-4
                assert all(p.grad is None or (enabled(n) and torch.isfinite(p.grad).all()) for n,p in named)
            def post(opt,args,kwargs):
                costs['optimizer']+=1;costs[phase[0]]+=1;local[phase[0]]+=1
                append(root/'physical_updates.jsonl',dict(physical=costs['optimizer'],kind=phase[0],arm=name,local=local.copy(),main_step=index[0]+1,source_cursor=base))
                active['cursor']=base+index[0]+1
            before_hook=opt.register_step_pre_hook(pre);after_hook=opt.register_step_post_hook(post)
            def update(i):
                index[0]=i
                return trainer.train_step(batch,None,state=TrainerState(global_step=base+i,site_step=base+i))
            def evaluate(step):
                rng=capture_rng_state()
                row=dict(step=step,cursor=base+step,
                         groups={part:frame(model,image[ix],y[ix])['metrics'] for part,ix in [('train',slice(0,2)),('heldout',slice(2,4))]},
                         train_views={str(k):frame(model,flip(image[:2],k),flip(y[:2],k))['metrics'] for k in range(4)})
                if step in (0,steps):row['density']=density(model,batch)
                assert same(rng,capture_rng_state());append(out/'metrics.jsonl',row);return row
            def frozen_check():
                assert all(torch.equal(v,model.state_dict()[n]) for n,v in frozen.items())
                assert all(same(v,opt.state_dict()['state'].get(i)) for i,v in frozen_adam.items())
            storage(s);save(out/'S0000.pt',snapshot(method,trainer,base));first_metrics=evaluate(0)
            active.update(out=out,method=method,trainer=trainer,cursor=base,phase=name)
            try:
                for i in range(steps):
                    rng=capture_rng_state();rngs.append(copy.deepcopy(rng))
                    if name=='J':assert same(rng,paired['C'][i])
                    if i in (0,steps-1):
                        phase[0]='golden';before=snapshot(method,trainer,base+i)
                        a=update(i);expected=snapshot(method,trainer,base+i+1);restore(before,method,trainer)
                        b=update(i);assert torch.equal(a.total_loss,b.total_loss) and same(expected,snapshot(method,trainer,base+i+1))
                        restore(before,method,trainer);goldens.append(dict(main_step=i+1,full_state_and_loss_equal=True));write(out/'GOLDEN.json',goldens)
                    phase[0]='main';r=update(i)
                    append(out/'steps.jsonl',dict(step=i+1,cursor=base+i+1,loss=float(r.total_loss.detach()),scalars=r.scalars))
                    if (i+1)%25==0:
                        frozen_check();storage(s);save(out/f'S{i+1:04}.pt',snapshot(method,trainer,base+i+1))
                        write(root/'STATUS.json',dict(status='RUNNING',arm=name,step=i+1,costs=costs,space=storage(s)))
                    if (i+1)%(300 if name=='K' else 25)==0:last_metrics=evaluate(i+1)
                frozen_check();assert local==dict(main=steps,golden=4) and len(goldens)==2
                if name=='K':
                    assert same(first_metrics['groups'],last_metrics['groups']) and same(first_metrics['train_views'],last_metrics['train_views'])
                qualified=all(v>=.8 for view in last_metrics['train_views'].values() for row in view['stochastic']['dice'] for v in row)
                final=snapshot(method,trainer,base+steps);assert final['trainer']['scheduler_state']['last_epoch']==base+steps
                assert method.extra_reverse_calls==0
                receipt=dict(status='COMPLETE',costs=local,source_cursor=base,final_cursor=base+steps,
                             initial=first_metrics,final=last_metrics,training_semantic_qualified=qualified,
                             frozen_model_and_adam_pass=True,prior_initial_unchanged=True,golden=goldens,extra_reverse=0)
                write(out/'RESULTS.json',receipt);results[name]=receipt
                storage(s);save(out/'PRIVATE_RNG.pt',rngs);paired[name]=rngs
                active.clear();return final
            finally:before_hook.remove();after_hook.remove()

        calibrated=arm('K',payload,600);arm('C',calibrated,100)
        passed=results['C']['training_semantic_qualified']
        write(root/'BRANCH_DECISION.json',dict(C_pass=passed,J_scheduled=passed,basis='terminal 24 training view Dice >= .8'))
        if passed:arm('J',calibrated,100)
        assert costs==dict(optimizer=812 if passed else 708,main=800 if passed else 700,golden=12 if passed else 8)
        final=dict(status='COMPLETE',costs=costs,arms=results,C_J_RNG_paired=passed,J_run=passed,
                   started=started,finished=time.time(),space=storage(s))
        write(root/'RESULTS.json',final);write(root/'STATUS.json',final);write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:
            storage(s);save(active['out']/'failure_partial.pt',snapshot(active['method'],active['trainer'],active['cursor']))
        failure=dict(status='FAILED',costs=costs,traceback=traceback.format_exc())
        write(root/'FAILURE.json',failure);write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
