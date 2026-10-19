"""Frozen inference, paired unlabeled-style versus observed-label proposals."""
import copy,json,os,traceback
from pathlib import Path
from dataclasses import replace
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import Trainer,TrainerState
from lcrseg.engine.checkpoint import capture_rng_state
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.jointwrite_v1.run import Adapter
from sc_bayeseg.run import snapshot,restore,same
from sc_bayeseg.guard_v1.run import frame
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage
from sc_bayeseg.window_v1.label_chart import LabelChartModel
from sc_bayeseg.window_v1.view_joint import density


def generator(n):return n.startswith(('anatomy.','structure.'))


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(optimizer=0,main=0,golden=0,extra_reverse=0,evaluation_bound=0);active={};results={};rng_reference=[]
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        audit=json.loads((Path(s['label_chart'])/'RESULTS.json').read_text());assert audit['status']=='COMPLETE' and audit['all_model_tensors_unchanged']
        assert json.loads((Path(s['label_chart'])/'EXIT.json').read_text())['exit_code']==0
        source=torch.load(s['readout_states']['G'],map_location='cpu',weights_only=False);base=source['cursor'];assert base==7475
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());assert len(set(sel['identities']))==4
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda');lb4=collate_labeled([ds[i] for i in sel['train_indices']+sel['heldout_indices']]).to('cuda')
        image=F.interpolate(lb4.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb4.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        batch=replace(lb,image=torch.cat([flip(image[:2],k) for k in range(4)]),label=torch.cat([flip(y[:2],k) for k in range(4)]),valid_mask=torch.ones((8,1,128,128),device='cuda',dtype=torch.bool),
                      case_id=lb.case_id*4,patient_id=lb.patient_id*4,site=lb.site*4,slice_index=lb.slice_index*4)
        for family in ('U','L'):
            out=root/family;out.mkdir(exist_ok=False);model=LabelChartModel().cuda();model.chart_enabled=family=='L'
            method=Adapter(model,'joint');method.nl=2;method.begin_site('Siemens',None,base+600)
            opt=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=0);scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
            trainer=Trainer(method,optimizer=opt,scheduler=scheduler,device='cuda',amp=False,grad_clip_norm=1.)
            restore(source,method,trainer);assert same(source,snapshot(method,trainer,base))
            named=list(model.named_parameters());train_names={n for n,p in named if generator(n)};assert len(train_names)==24 and len(named)==100
            for n,p in named:p.requires_grad_(n in train_names)
            frozen={n:v.clone() for n,v in source['model'].items() if n not in train_names}
            frozen_adam={i:copy.deepcopy(opt.state_dict()['state'][i]) for i,(n,p) in enumerate(named) if n not in train_names}
            phase=['golden'];index=[0];local=dict(main=0,golden=0);goldens=[];paired=[]
            def pre(opt,a,kw):
                assert costs['optimizer']<1208 and local[phase[0]]<(600 if phase[0]=='main' else 4)
                assert opt.param_groups[0]['lr']==3e-4 and all(p.grad is None or (n in train_names and torch.isfinite(p.grad).all()) for n,p in named)
            def post(opt,a,kw):
                costs['optimizer']+=1;costs[phase[0]]+=1;local[phase[0]]+=1;active['cursor']=base+index[0]+1
                append(root/'physical_updates.jsonl',dict(physical=costs['optimizer'],arm=family,phase=phase[0],main_step=index[0]+1,source_cursor=base))
            prehook=opt.register_step_pre_hook(pre);posthook=opt.register_step_post_hook(post)
            def check_frozen():
                assert all(torch.equal(v,model.state_dict()[n].cpu()) for n,v in frozen.items())
                assert all(same(v,opt.state_dict()['state'][i]) for i,v in frozen_adam.items())
            def update(i):
                index[0]=i;active['cursor']=base+i
                return trainer.train_step(batch,None,state=TrainerState(global_step=base+i,site_step=base+i))
            def evaluate(step):
                rng=capture_rng_state();row=dict(step=step,groups={part:frame(model,image[ix],y[ix])['metrics'] for part,ix in [('train',slice(0,2)),('heldout',slice(2,4))]})
                if step in (0,600):
                    row['own_bound']=density(model,batch);costs['evaluation_bound']+=4
                    model.chart_enabled=False
                    try:row['common_U_bound']=density(model,batch);costs['evaluation_bound']+=4
                    finally:model.chart_enabled=family=='L'
                    append(root/'evaluation_suites.jsonl',dict(index=costs['evaluation_bound']//8,arm=family,step=step,bound_calls=8))
                assert same(rng,capture_rng_state());append(out/'metrics.jsonl',row);return row
            active.update(out=out,method=method,trainer=trainer,cursor=base)
            storage(s);save(out/'S0000.pt',source);first=evaluate(0)
            try:
                for i in range(600):
                    rng=capture_rng_state();paired.append(copy.deepcopy(rng))
                    if family=='L':assert same(rng,rng_reference[i])
                    if i in (0,599):
                        phase[0]='golden';before=snapshot(method,trainer,base+i)
                        a=update(i);expected=snapshot(method,trainer,base+i+1);restore(before,method,trainer)
                        b=update(i);assert torch.equal(a.total_loss,b.total_loss) and same(expected,snapshot(method,trainer,base+i+1))
                        restore(before,method,trainer);goldens.append(dict(step=i+1,full_state_and_loss_equal=True));write(out/'GOLDEN.json',goldens)
                    phase[0]='main';r=update(i)
                    append(out/'steps.jsonl',dict(step=i+1,loss=float(r.total_loss.detach()),scalars=r.scalars,chart_stats=model.chart_stats if family=='L' else None))
                    if (i+1)%25==0:check_frozen();write(root/'STATUS.json',dict(status='RUNNING',arm=family,step=i+1,costs=costs,space=storage(s)))
                    if (i+1)%100==0:storage(s);save(out/f'S{i+1:04}.pt',snapshot(method,trainer,base+i+1))
                    if (i+1)%300==0:last=evaluate(i+1)
                check_frozen();assert same(first['groups'],last['groups']) and local==dict(main=600,golden=4)
                result=dict(status='COMPLETE',costs=local,initial=first,final=last,all_direct_predictions_unchanged=True,frozen_76_model_and_adam=True,source_cursor=base,final_cursor=base+600,golden=goldens)
                write(out/'RESULTS.json',result);results[family]=result
                storage(s);save(out/'PRIVATE_RNG.pt',paired)
                if family=='U':rng_reference=paired
                active.clear()
            finally:prehook.remove();posthook.remove()
        assert costs==dict(optimizer=1208,main=1200,golden=8,extra_reverse=0,evaluation_bound=32)
        result=dict(status='COMPLETE',costs=costs,arms=results,paired_RNG_600=True,prior_initial_unchanged=True,space=storage(s))
        write(root/'RESULTS.json',result);write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:storage(s);save(active['out']/'failure_partial.pt',snapshot(active['method'],active['trainer'],active['cursor']))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
