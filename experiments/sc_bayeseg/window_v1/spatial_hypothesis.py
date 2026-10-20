"""Symmetric GT/BG local Gaussian inference under one updated generator."""
import json,os,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.run import same
from sc_bayeseg.diagnostic_v1.run import dice
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.common import storage
from sc_bayeseg.window_v1.fullview import flip


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(local_reverse=0,bound=0,candidate=0,accepted=0,decoder=0,optimizer=0,model_applications=0);active={}
    def forbid(*a,**kw):raise AssertionError('optimizer forbidden')
    torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        src=Path(s['spatial_generator']);ledger=(src/'physical_updates.jsonl').read_text();assert len(ledger.splitlines())==604
        payload=torch.load(src/'S0600.pt',map_location='cpu',weights_only=False);assert payload['cursor']==8675
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda');im=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);gt=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        image=im;y=gt
        model=SpatialResidualModel().cuda();model.load_state_dict(payload['model']);model.requires_grad_(False);initial_rng=capture_rng_state();zero=torch.zeros((2,16,16,16),device='cuda');q=(zero.clone(),zero.clone())
        def counted_bound(q,kind):
            b,t,d=model.bound(image,y,q);costs['bound']+=1;assert costs['bound']<=72200
            append(root/'bound_calls.jsonl',dict(index=costs['bound'],hypothesis=name,kind=kind))
            if kind!='candidate' or torch.isfinite(b).all():assert d['normalization_error']<1e-6
            return b,t,d
        initial_script_rng=capture_rng_state();original_gt=gt;results={};pairs=[]
        previous=Path(s['spatial_recouple'])/'local_updates.jsonl';previous_text=previous.read_text()
        def saa(q,kind,grad=False):
            values=[];parts=[]
            with torch.set_grad_enabled(grad):
                for _ in range(2):
                    model.draw()
                    try:
                        for _ in range(4):
                            b,t,d=counted_bound(q,kind);values.append(b/128**2)
                            parts.append({**{k:v.detach()/128**2 for k,v in t.items()},'negative_logq':-d['logq'].detach()/128**2,'residual_KL':d['residual_KL'].detach()/128**2})
                    finally:model.clear_draw()
                return torch.stack(values).mean(0),{k:torch.stack([v[k] for v in parts]).mean(0).cpu().tolist() for k in parts[0]}
        def evaluate(step):
            rng=capture_rng_state();flows=[]
            for seed in (50163,80163):
                seed_everything(seed);v,t=saa(q,'readout');flows.append(dict(seed=seed,ELBO=v.cpu().tolist(),parts=t,global_Theta_KL_constant=float(model.kl()/(2*128**2))))
            restore_rng_state(rng)
            row=dict(step=step,flows=flows,mu_RMS=float(q[0].square().mean().sqrt()),logsd_quantiles=torch.quantile(q[1].detach().flatten(),torch.tensor([.05,.5,.95],device='cuda')).tolist())
            assert same(rng,capture_rng_state());append(out/'metrics.jsonl',row);return row
        def checkpoint(step):
            storage(s);save(out/f'S{step:04}.pt',dict(step=step,q=tuple(v.detach().cpu() for v in q),rng=capture_rng_state(),costs=costs.copy()))
        for name in ('GT','BG'):
            out=root/name;out.mkdir(exist_ok=False);y=original_gt if name=='GT' else torch.zeros_like(original_gt);q=(zero.clone(),zero.clone());restore_rng_state(payload['rng'])
            if name=='BG':
                assert same(capture_rng_state(),torch.load(root/'GT/S0000.pt',map_location='cpu',weights_only=False)['rng']);pairs.append(0);append(root/'RNG_PAIRS.jsonl',dict(step=0,exact=True))
            active.update(out=out,q=q,step=0);checkpoint(0);first=evaluate(0)
            for step in range(1,501):
                q=tuple(v.detach().requires_grad_() for v in q);start_rng=capture_rng_state();before,_=saa(q,'gradient',True);after_rng=capture_rng_state()
                g=torch.autograd.grad(-before.sum(),q);assert all(torch.isfinite(v).all() for v in g);costs['local_reverse']+=1
                append(root/'reverse_calls.jsonl',dict(index=costs['local_reverse'],hypothesis=name,step=step,norms=[float(v.norm()) for v in g]))
                chosen=None
                for alpha in (1.,.25,.0625,.015625,.00390625,.0009765625,.000244140625,0.):
                    candidate=tuple((v-.05*alpha*256*dv).detach() for v,dv in zip(q,g));restore_rng_state(start_rng);value,_=saa(candidate,'candidate')
                    finite=bool(torch.isfinite(value).all());ok=finite and bool((value>=before.detach()-1e-6).all()) and float(value.mean())>float(before.detach().mean())+1e-5
                    costs['candidate']+=1;assert costs['candidate']<=8000
                    append(root/'candidates.jsonl',dict(index=costs['candidate'],hypothesis=name,step=step,alpha=alpha,accepted=ok,finite=finite))
                    if ok:chosen=(candidate,value,alpha);break
                restore_rng_state(after_rng)
                if chosen is not None:q,value,alpha=chosen;costs['accepted']+=1;append(root/'local_updates.jsonl',dict(index=costs['accepted'],hypothesis=name,step=step,alpha=alpha))
                else:q=tuple(v.detach() for v in q);value=before.detach();alpha=0.
                append(out/'steps.jsonl',dict(step=step,alpha=alpha,accepted=chosen is not None,before=before.detach().cpu().tolist(),after=value.cpu().tolist()))
                active.update(q=q,step=step)
                if step%25==0 and name=='BG':
                    old=torch.load(root/'GT'/f'S{step:04}.pt',map_location='cpu',weights_only=False);assert same(capture_rng_state(),old['rng']);pairs.append(step);append(root/'RNG_PAIRS.jsonl',dict(step=step,exact=True))
                if step%25==0:checkpoint(step);write(root/'STATUS.json',dict(status='RUNNING',hypothesis=name,step=step,costs=costs,space=storage(s)))
                if step%100==0:last=evaluate(step)
            assert same(payload['model'],model.state_dict()) and all(v.grad is None for v in model.parameters());results[name]=dict(initial=first,final=last,model_unchanged=True)
        assert costs['local_reverse']==1000 and len(pairs)==21
        assert (src/'physical_updates.jsonl').read_text()==ledger and previous.read_text()==previous_text
        restore_rng_state(initial_script_rng);assert same(initial_script_rng,capture_rng_state())
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,hypotheses=results,RNG_pairs=21,model_generator604_recouple_history_unchanged=True,no_heldout_or_U=True,source_cursor=8675,model_optimizer=0,space=storage(s)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:storage(s);save(active['out']/'failure_partial.pt',dict(step=active['step'],q=tuple(v.detach().cpu() for v in active['q']),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
