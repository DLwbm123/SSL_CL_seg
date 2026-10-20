"""One anatomy posterior coordinate update with fixed normalized spatial qR."""
import copy,json,os,traceback
from pathlib import Path
from dataclasses import replace
import torch
from torch.nn import functional as F
from lcrseg.contracts import MethodStepOutput
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import Trainer,TrainerState,seed_everything
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.method import Method
from sc_bayeseg.run import snapshot,restore,same
from sc_bayeseg.split_v1.model import semantic
from sc_bayeseg.diagnostic_v1.run import dice
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.common import storage
from sc_bayeseg.window_v1.fullview import flip


def active_parameter(name):return name.startswith('anatomy.')


class LabeledMethod(Method):
    def training_step(self,lb,ub,global_step,site_step):
        assert self.nl==2 and self.nu==0 and ub is None and lb.valid_mask.all()
        self.model.draw()
        try:
            b,t,d=self.model.bound(lb.image,lb.label,self.residual_q);scale=self.model.size**2;kl=self.model.kl()/(self.nl*scale)
            loss=-b.mean()/scale+kl
            stats={k:float(-v.mean().detach()/scale) for k,v in t.items()}
            stats.update(logq=float(d['logq'].mean().detach()/scale),kl=float(kl.detach()),normalization_error=d['normalization_error'],residual_KL=float(d['residual_KL'].mean().detach()/scale))
            return MethodStepOutput(loss,dict(labeled_bound=-b.mean()/scale,posterior_kl=kl),stats)
        finally:self.model.clear_draw()


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(optimizer=0,main=0,golden=0,extra_reverse=0,evaluation_bound=0);active={};hooks=[]
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        qroot=Path(s['spatial_residual']);assert json.loads((qroot/'ZERO_RESULT.json').read_text())['status']=='PASS' and json.loads((qroot/'EXIT.json').read_text())['exit_code']==0
        p=torch.load(Path(s['discrete_train'])/'S0600.pt',map_location='cpu',weights_only=False);base=p['cursor'];assert base==8075
        qp=torch.load(qroot/'S0500.pt',map_location='cpu',weights_only=False);assert qp['step']==500
        p=copy.deepcopy(p);p['rng']=qp['rng'];qcpu=tuple(t.clone() for t in qp['q']);qfixed=tuple(t.cuda() for t in qcpu)
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());assert len(set(sel['identities']))==4
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda');lb4=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda')
        image=F.interpolate(lb4.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb4.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        batch=replace(lb,image=torch.cat([flip(image[:2],k) for k in range(4)]),label=torch.cat([flip(y[:2],k) for k in range(4)]),valid_mask=torch.ones((8,1,128,128),device='cuda',dtype=torch.bool),
                      case_id=lb.case_id*4,patient_id=lb.patient_id*4,site=lb.site*4,slice_index=lb.slice_index*4)
        model=SpatialResidualModel().cuda();method=LabeledMethod(model,2,0);method.residual_q=qfixed;method.begin_site('Siemens',None,base+600)
        opt=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=0);scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        trainer=Trainer(method,optimizer=opt,scheduler=scheduler,device='cuda',amp=False,grad_clip_norm=1.)
        restore(p,method,trainer);assert same(p,snapshot(method,trainer,base))
        named=list(model.named_parameters());assert len(named)==100 and sum(semantic(n) for n,v in named)==40 and sum(active_parameter(n) for n,v in named)==12
        for n,v in named:v.requires_grad_(active_parameter(n))
        frozen={n:v for n,v in p['model'].items() if not active_parameter(n) or 'prior_' in n or 'initial_' in n}
        frozen_adam={i:copy.deepcopy(opt.state_dict()['state'][i]) for i,(n,v) in enumerate(named) if not active_parameter(n)}
        phase=['golden'];index=[0];goldens=[]
        def pre(opt,a,kw):
            assert costs['optimizer']<604 and opt.param_groups[0]['lr']==3e-4
            assert all((v.grad is None) if not active_parameter(n) else (v.grad is not None and torch.isfinite(v.grad).all()) for n,v in named)
        def post(opt,a,kw):
            costs['optimizer']+=1;costs[phase[0]]+=1;active['cursor']=base+index[0]+1
            append(root/'physical_updates.jsonl',dict(physical=costs['optimizer'],phase=phase[0],main_step=index[0]+1,source_cursor=base))
        hooks=[opt.register_step_pre_hook(pre),opt.register_step_post_hook(post)]
        def frozen_check():
            assert all(torch.equal(v,model.state_dict()[n].cpu()) for n,v in frozen.items())
            assert same(qcpu,qfixed)
            assert all(same(v,opt.state_dict()['state'][i]) for i,v in frozen_adam.items())
        def update(i):
            index[0]=i;active['cursor']=base+i
            return trainer.train_step(batch,None,state=TrainerState(global_step=base+i,site_step=base+i))
        def evaluate(step):
            rng=capture_rng_state();row=dict(step=step,recognition={})
            with torch.no_grad():
                for name,ix in [('train',slice(0,2))]:
                    logits=model.recognition_logits(image[ix]);row['recognition'][name]=dict(dice=dice(logits.argmax(1),y[ix]),ce=float(F.cross_entropy(logits,y[ix],reduction='none').mean()))
                values=[]
                for seed in (50163,80163):
                    seed_everything(seed);draws=[]
                    for _ in range(2):
                        model.draw()
                        try:
                            for _ in range(4):
                                b,t,d=model.bound(batch.image,batch.label,qfixed);kl=model.kl()/(2*128**2)
                                draws.append(dict(joint=float(-b.mean()/128**2+kl),kl=float(kl),residual_KL=float(d['residual_KL'].mean()/128**2),logq=float(d['logq'].mean()/128**2),terms={k:float(-v.mean()/128**2) for k,v in t.items()}))
                                costs['evaluation_bound']+=1;append(root/'evaluation_bound_calls.jsonl',dict(index=costs['evaluation_bound'],step=step,seed=seed))
                        finally:model.clear_draw()
                    values.append(dict(seed=seed,draws=draws))
                row['new_model_bound']=values
                for layer in model.bayes_layers():layer.sampled=(layer.mean_w,layer.mean_b)
                try:
                    h,_=model.labeled_proposal(batch.image,batch.label,sample=False);logits=model.spatial_logits(h['c'],qfixed[0]);d=dice(logits.argmax(1),batch.label)
                    row['conditional_reconstruction']=dict(CE=float(F.cross_entropy(logits,batch.label,reduction='none').mean()),all_view_Dice=d,qualified24=all(v>=.8 for r in d for v in r))
                finally:model.clear_draw()
            restore_rng_state(rng);append(root/'metrics.jsonl',row);return row
        def checkpoint(cursor):
            state=snapshot(method,trainer,cursor);state['residual_q']=qcpu;state['residual_local_step']=500;return state
        active.update(method=method,trainer=trainer,cursor=base);storage(s);save(root/'S0000.pt',checkpoint(base));first=evaluate(0)
        for i in range(600):
            if i in (0,599):
                phase[0]='golden';before=snapshot(method,trainer,base+i)
                a=update(i);expected=snapshot(method,trainer,base+i+1);restore(before,method,trainer)
                b=update(i);assert torch.equal(a.total_loss,b.total_loss) and same(expected,snapshot(method,trainer,base+i+1))
                restore(before,method,trainer);goldens.append(dict(main_step=i+1,full_state_and_loss_equal=True));write(root/'GOLDEN.json',goldens)
            phase[0]='main';r=update(i);append(root/'steps.jsonl',dict(step=i+1,loss=float(r.total_loss.detach()),scalars=r.scalars))
            if (i+1)%25==0:frozen_check();write(root/'STATUS.json',dict(status='RUNNING',step=i+1,costs=costs,space=storage(s)))
            if (i+1)%100==0:storage(s);save(root/f'S{i+1:04}.pt',checkpoint(base+i+1))
            if (i+1)%300==0:last=evaluate(i+1)
        frozen_check();assert same(first['recognition'],last['recognition'])
        assert costs==dict(optimizer=604,main=600,golden=4,extra_reverse=0,evaluation_bound=48)
        result=dict(status='COMPLETE',costs=costs,initial=first,final=last,golden=goldens,nongenerator_88_model_adam_frozen=True,fixed_residual_posterior=True,no_heldout_or_U=True,prior_initial_frozen=True,
                    recognition_predictions_unchanged=True,source_cursor=base,final_cursor=base+600,space=storage(s))
        write(root/'RESULTS.json',result);write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:storage(s);save(root/'failure_partial.pt',dict(**snapshot(active['method'],active['trainer'],active['cursor']),residual_q=qcpu,residual_local_step=500))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise
    finally:
        for h in hooks:h.remove()


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
