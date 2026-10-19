"""Finite semantic-step gate followed by bounded constrained joint updates."""
import copy,json,os,traceback
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import Trainer,TrainerState
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.split_v1.model import Model,semantic
from sc_bayeseg.jointwrite_v1.run import Adapter
from sc_bayeseg.run import snapshot,restore,same
from sc_bayeseg.adam_audit_v1.run import candidate
from sc_bayeseg.geometry_v1.run import flat
from sc_bayeseg.geometry_v1.geometry import solve
from sc_bayeseg.guard_v1.run import frame
from sc_bayeseg.window_v1.common import storage
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.view_geometry import constraints
from sc_bayeseg.window_v1.view_joint import density


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False)
    costs=dict(optimizer=0,main=0,golden=0,extra_reverse=0,pure_adam=0,candidates=0);active={};hooks=[];oldclip=torch.nn.utils.clip_grad_norm_
    def reverse(*a,**kw):
        assert costs['extra_reverse']<6865;costs['extra_reverse']+=1
        append(root/'reverse_calls.jsonl',dict(index=costs['extra_reverse']));return torch.autograd.grad(*a,**kw)
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        geo=json.loads((Path(s['view_geometry'])/'RESULTS.json').read_text());assert geo['states']['0']['geometry']['strict']['strict_witness']
        p=torch.load(Path(s['view_joint'])/'K/S0600.pt',map_location='cpu',weights_only=False);base=p['cursor'];assert base==7375
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());assert len(set(sel['identities']))==4
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda');lb4=collate_labeled([ds[i] for i in sel['train_indices']+sel['heldout_indices']]).to('cuda')
        image=F.interpolate(lb4.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb4.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        batch=replace(lb,image=torch.cat([flip(image[:2],k) for k in range(4)]),label=torch.cat([flip(y[:2],k) for k in range(4)]),
                      valid_mask=torch.ones((8,1,128,128),device='cuda',dtype=torch.bool),case_id=lb.case_id*4,patient_id=lb.patient_id*4,site=lb.site*4,slice_index=lb.slice_index*4)
        model=Model().cuda();method=Adapter(model,'joint');method.nl=2;method.begin_site('Siemens',None,base+100)
        named=list(model.named_parameters());params=[v for n,v in named];flags=[semantic(n) for n,v in named];sp=[v for v,f in zip(params,flags) if f]
        opt=torch.optim.Adam(params,lr=3e-4,weight_decay=0);scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        trainer=Trainer(method,optimizer=opt,scheduler=scheduler,device='cuda',amp=False,grad_clip_norm=1.)
        restore(p,method,trainer);assert same(p,snapshot(method,trainer,base))
        prior={n:v.clone() for n,v in p['model'].items() if 'prior_' in n or 'initial_' in n}
        phase=['preflight'];index=[0];stash={}
        def scores(rng):
            after=capture_rng_state();out=[]
            try:
                with torch.no_grad():
                    for noise in ('original',50163,80163):
                        if noise=='original':restore_rng_state(rng)
                        else:
                            from lcrseg.engine.trainer import seed_everything
                            seed_everything(noise)
                        r=method.training_step(batch,None,0,0);out.append(float(r.total_loss))
                return out
            finally:restore_rng_state(after)
        def values():
            with torch.no_grad():return [float(v) for v in constraints(model,batch.image,batch.label)[1]]
        grid=[(b,a) for b in (1.,.25,0.) for a in (1.,.25,.0625,.015625,.00390625,.0009765625,.000244140625)]+[(b,0.) for b in (1.,.25,0.)]
        def search(old,delta,d,valid,v0,j0):
            radius=float(np.linalg.norm(flat([v for v,f in zip(delta,flags) if f],sp)));chosen=None;trials=[]
            for beta,alpha in (grid if valid else grid[-3:]):
                with torch.no_grad():
                    pos=0
                    for i,(n,v) in enumerate(named):
                        source=old['model'][n].to(v.device)
                        if flags[i]:
                            dd=torch.as_tensor(d[pos:pos+v.numel()].reshape(v.shape),device=v.device,dtype=v.dtype);v.copy_(source+alpha*radius*dd);pos+=v.numel()
                        else:v.copy_(source+beta*delta[i])
                vv=values();jj=scores(old['rng']);semok=all(a<=b+1e-6 for a,b in zip(vv,v0));jok=all(a<=b+1e-5 for a,b in zip(jj,j0)) and sum(jj)/3<sum(j0)/3-1e-5
                assert costs['candidates']<2520;costs['candidates']+=1
                row=dict(index=costs['candidates'],phase=phase[0],step=index[0],alpha=alpha,beta=beta,semantic_pass=semok,joint_pass=jok,vector=vv,joint=jj)
                append(root/'candidates.jsonl',row);trials.append(row)
                if semok and jok:chosen=dict(alpha=alpha,beta=beta,vector=vv,joint=jj);break
            if chosen is None:model.load_state_dict(old['model']);chosen=dict(alpha=0.,beta=0.,vector=v0,joint=j0)
            semnorm=sum(float((v.detach().cpu()-old['model'][n]).double().square().sum()) for n,v in named if semantic(n))**.5
            return dict(**chosen,rollback=not any(v['semantic_pass'] and v['joint_pass'] for v in trials),semantic_delta_norm=semnorm,radius=radius,trials=len(trials))
        # Gate uses the already certified private direction, with one fresh raw gradient.
        reference=np.load(Path(s['view_geometry'])/'PRIVATE_GEOMETRY.npz')
        v0=values();assert np.allclose(v0,geo['states']['0']['baseline'],rtol=0,atol=1e-7)
        j0=scores(p['rng']);restore_rng_state(p['rng']);r=method.training_step(batch,None,base,base)
        full=reverse(r.total_loss,params,allow_unused=True);nn=torch.linalg.vector_norm(torch.stack(torch._foreach_norm(full,2)),2)
        clipped=torch._foreach_mul(full,(1/(nn+1e-6)).clamp(max=1.));op=p['trainer']['optimizer_state'];group=op['param_groups'][0]
        with torch.no_grad():delta,*_=candidate(params,clipped,[op['state'][i] for i in group['params']],group)
        costs['pure_adam']=1;decision=search(p,delta,reference['S0_strict'],True,v0,j0)
        gate=decision['semantic_delta_norm']>0 and not decision['rollback'];restore(p,method,trainer)
        assert same(p,snapshot(method,trainer,base));write(root/'ZERO_GATE.json',dict(pass_gate=gate,decision=decision,costs=costs.copy(),complete_state_restored=True))
        if not gate:
            write(root/'RESULTS.json',dict(status='COMPLETE',training_started=False,costs=costs,gate=decision,space=storage(s)));write(root/'EXIT.json',dict(exit_code=0));return
        out=root/'G';out.mkdir(exist_ok=False);storage(s);save(out/'S0000.pt',p)
        def before(opt,a,kw):
            assert costs['optimizer']<104 and all(v.grad is not None and torch.isfinite(v.grad).all() for v in params)
            assert opt.param_groups[0]['lr']==3e-4
        def after(opt,a,kw):
            costs['optimizer']+=1;costs[phase[0]]+=1
            append(root/'physical_updates.jsonl',dict(physical=costs['optimizer'],phase=phase[0],step=index[0],source_cursor=base))
        hooks=[opt.register_step_pre_hook(before),opt.register_step_post_hook(after)]
        def clip(pp,maxnorm,*a,**kw):
            pp=list(pp);stash['raw']=flat([v.grad for v,f in zip(params,flags) if f],sp)
            return oldclip(pp,maxnorm,*a,**kw)
        torch.nn.utils.clip_grad_norm_=clip
        def advance(k):
            index[0]=k;active['cursor']=base+k-1;old=snapshot(method,trainer,base+k-1);j0=scores(old['rng'])
            _,vals=constraints(model,batch.image,batch.label);v0=[float(v.detach()) for v in vals]
            C=np.stack([flat(reverse(v,sp,allow_unused=True,retain_graph=i<63),sp) for i,v in enumerate(vals)])
            auxiliary=[]
            for seed in (50163,80163):
                from lcrseg.engine.trainer import seed_everything
                seed_everything(seed);rr=method.training_step(batch,None,base+k-1,base+k-1)
                auxiliary.append(flat(reverse(rr.total_loss,sp,allow_unused=True),sp))
            restore_rng_state(old['rng']);r=trainer.train_step(batch,None,state=TrainerState(global_step=base+k-1,site_step=base+k-1));active['cursor']=base+k
            delta=[v.detach()-old['model'][n].to(v.device) for n,v in named];postrng=capture_rng_state();opstate=copy.deepcopy(trainer.state_dict())
            geometry,dirs=solve(C,np.stack([stash['raw']]+auxiliary));valid=geometry['strict']['strict_witness']
            decision=search(old,delta,dirs['strict'],valid,v0,j0)
            assert same(postrng,capture_rng_state()) and same(opstate,trainer.state_dict())
            assert all(torch.equal(v,model.state_dict()[n].cpu()) for n,v in prior.items())
            return dict(step=k,loss=float(r.total_loss.detach()),joint_before=j0,vector_before=v0,strict_witness=valid,**decision)
        def evaluate(k):
            row=dict(step=k,groups={part:frame(model,image[ix],y[ix])['metrics'] for part,ix in [('train',slice(0,2)),('heldout',slice(2,4))]},
                     train_views={str(v):frame(model,flip(image[:2],v),flip(y[:2],v))['metrics'] for v in range(4)})
            if k in (0,100):row['density']=density(model,batch)
            append(out/'metrics.jsonl',row);return row
        evaluate(0);rows=[];goldens=[];active.update(cursor=base)
        for k in range(1,101):
            phase[0]='golden' if k in (1,100) else 'main'
            if k in (1,100):
                before_state=snapshot(method,trainer,base+k-1)
                a=advance(k);expected=snapshot(method,trainer,base+k);restore(before_state,method,trainer)
                b=advance(k);assert same(a,b) and same(expected,snapshot(method,trainer,base+k))
                restore(before_state,method,trainer);goldens.append(dict(step=k,full_state_and_decision_equal=True));write(out/'GOLDEN.json',goldens)
            phase[0]='main';row=advance(k);rows.append(row);append(out/'steps.jsonl',row)
            storage(s);save(out/f'S{k:04}.pt',snapshot(method,trainer,base+k))
            if k%25==0:last=evaluate(k)
            write(root/'STATUS.json',dict(status='RUNNING',step=k,costs=costs,semantic_nonzero=sum(v['semantic_delta_norm']>0 for v in rows),space=storage(s)))
        assert costs['optimizer']==104 and costs['main']==100 and costs['golden']==4 and costs['extra_reverse']==6865
        result=dict(status='COMPLETE',training_started=True,costs=costs,nonzero_semantic_steps=sum(v['semantic_delta_norm']>0 for v in rows),
                    rollbacks=sum(v['rollback'] for v in rows),semantic_path=sum(v['semantic_delta_norm'] for v in rows),final=last,golden=goldens,space=storage(s))
        write(root/'RESULTS.json',result);write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active and 'trainer' in locals():storage(s);save(root/'failure_partial.pt',snapshot(method,trainer,active['cursor']))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise
    finally:
        torch.nn.utils.clip_grad_norm_=oldclip
        for h in hooks:h.remove()


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
