"""Paired deterministic labeled categorical reconstruction, no heldout."""
import copy,json,os,traceback
from pathlib import Path
from dataclasses import replace
import torch
from torch.nn import functional as F
from lcrseg.contracts import MethodStepOutput
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import Trainer,TrainerState
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.method import Method
from sc_bayeseg.run import snapshot,restore,same
from sc_bayeseg.diagnostic_v1.run import dice
from sc_bayeseg.window_v1.discrete_labeled import DiscreteLabeledModel
from sc_bayeseg.window_v1.common import storage
from sc_bayeseg.window_v1.fullview import flip


def trainable(name,arm):
    return (name.startswith('anatomy.') and ('.mean_w' in name or '.mean_b' in name)) or (arm=='code' and name.startswith('code_encoder.'))


def logits_for(model,image,y):
    for layer in model.bayes_layers():layer.sampled=(layer.mean_w,layer.mean_b)
    try:
        h,_=model.labeled_proposal(image,y,sample=False);a=model.anatomy_mean(h['c'])
        return torch.cat([torch.zeros_like(a[:,:1]),a],1)
    finally:model.clear_draw()


class ReconMethod(Method):
    def training_step(self,lb,ub,global_step,site_step):
        assert ub is None and self.nl==2 and self.nu==0
        logits=logits_for(self.model,lb.image,lb.label);loss=F.cross_entropy(logits,lb.label,reduction='none').mean()
        return MethodStepOutput(loss,dict(categorical_reconstruction=loss),dict(CE=float(loss.detach())))


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(optimizer=0,main=0,golden=0,extra_reverse=0);active={};hooks=[]
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        src=Path(s['discrete_train']);p=torch.load(src/'S0600.pt',map_location='cpu',weights_only=False);base=p['cursor'];assert base==8075
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda');im=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        batch=replace(lb,image=torch.cat([flip(im,k) for k in range(4)]),label=torch.cat([flip(y,k) for k in range(4)]),valid_mask=torch.ones((8,1,128,128),device='cuda',dtype=torch.bool),case_id=lb.case_id*4,patient_id=lb.patient_id*4,site=lb.site*4,slice_index=lb.slice_index*4)
        model=DiscreteLabeledModel().cuda();method=ReconMethod(model,2,0);method.begin_site('Siemens',None,base+600)
        opt=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=0);scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        trainer=Trainer(method,optimizer=opt,scheduler=scheduler,device='cuda',amp=False,grad_clip_norm=1.)
        named=list(model.named_parameters());assert len(named)==100
        index=[0];phase=['golden'];arm=['fixed'];paired=[];goldens=[];results={}
        def pre(opt,a,kw):
            assert costs['optimizer']<1208 and opt.param_groups[0]['lr']==3e-4
            assert all((v.grad is not None and torch.isfinite(v.grad).all()) if trainable(n,arm[0]) else v.grad is None for n,v in named)
        def post(opt,a,kw):
            costs['optimizer']+=1;costs[phase[0]]+=1;active['cursor']=base+index[0]+1
            append(root/'physical_updates.jsonl',dict(physical=costs['optimizer'],arm=arm[0],phase=phase[0],main_step=index[0]+1,source_cursor=base))
        hooks=[opt.register_step_pre_hook(pre),opt.register_step_post_hook(post)]
        def update(i):
            index[0]=i;active['cursor']=base+i
            return trainer.train_step(batch,None,state=TrainerState(global_step=base+i,site_step=base+i))
        def evaluate(out,step):
            rng=capture_rng_state()
            with torch.no_grad():
                logits=logits_for(model,batch.image,batch.label);nll=F.cross_entropy(logits,batch.label,reduction='none');d=dice(logits.argmax(1),batch.label)
                row=dict(step=step,CE=float(nll.mean()),class_NLL=[float(nll[batch.label==k].mean()) for k in range(4)],all_view_Dice=d,predicted_fraction=[float((logits.argmax(1)==k).float().mean()) for k in range(4)],qualified24=all(v>=.8 for row in d for v in row))
            assert same(rng,capture_rng_state());append(out/'metrics.jsonl',row);return row
        for name in ('fixed','code'):
            arm[0]=name;out=root/name;out.mkdir(exist_ok=False)
            for n,v in named:v.requires_grad_(True)
            restore(p,method,trainer);assert same(p,snapshot(method,trainer,base))
            for n,v in named:v.requires_grad_(trainable(n,name))
            frozen={n:v for n,v in p['model'].items() if n not in {k for k,_ in named if trainable(k,name)}}
            frozen_adam={i:copy.deepcopy(opt.state_dict()['state'][i]) for i,(n,v) in enumerate(named) if not trainable(n,name)}
            def frozen_check():
                assert all(torch.equal(v,model.state_dict()[n].cpu()) for n,v in frozen.items())
                assert all(same(v,opt.state_dict()['state'][i]) for i,v in frozen_adam.items())
            active.update(out=out,method=method,trainer=trainer,cursor=base);storage(s);save(out/'S0000.pt',snapshot(method,trainer,base));first=evaluate(out,0)
            for i in range(600):
                if i in (0,599):
                    phase[0]='golden';before=snapshot(method,trainer,base+i)
                    a=update(i);expected=snapshot(method,trainer,base+i+1);restore(before,method,trainer)
                    b=update(i);assert torch.equal(a.total_loss,b.total_loss) and same(expected,snapshot(method,trainer,base+i+1))
                    restore(before,method,trainer);goldens.append(dict(arm=name,main_step=i+1,full_state_and_loss_equal=True));write(root/'GOLDEN.json',goldens)
                rng=capture_rng_state()
                if name=='fixed':paired.append(rng)
                else:assert same(paired[i],rng)
                append(out/'rng_pairs.jsonl',dict(step=i+1,paired=name=='code'))
                phase[0]='main';r=update(i);append(out/'steps.jsonl',dict(step=i+1,loss=float(r.total_loss.detach())))
                if (i+1)%25==0:frozen_check();write(root/'STATUS.json',dict(status='RUNNING',arm=name,step=i+1,costs=costs,space=storage(s)))
                if (i+1)%100==0:storage(s);save(out/f'S{i+1:04}.pt',snapshot(method,trainer,base+i+1))
                if (i+1)%300==0:last=evaluate(out,i+1)
            frozen_check();results[name]=dict(initial=first,final=last,active_parameters=sum(trainable(n,name) for n,v in named),frozen_model_adam_pass=True,prior_initial_frozen=True)
        assert costs==dict(optimizer=1208,main=1200,golden=8,extra_reverse=0)
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,arms=results,golden=goldens,paired_rng_steps=600,no_heldout_or_U=True,source_cursor=base,final_cursor=base+600,space=storage(s)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:storage(s);save(active['out']/'failure_partial.pt',snapshot(active['method'],active['trainer'],active['cursor']))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise
    finally:
        for h in hooks:h.remove()


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
