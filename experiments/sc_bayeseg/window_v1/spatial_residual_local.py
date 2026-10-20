"""Zero admission followed by frozen-model local Gaussian spatial VI."""
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
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(zero_reverse=0,local_reverse=0,bound=0,candidate=0,accepted=0,decoder=0,optimizer=0,model_applications=0);active={}
    def forbid(*a,**kw):raise AssertionError('optimizer forbidden')
    torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        src=Path(s['discrete_train']);ledger=(src/'physical_updates.jsonl').read_text();assert len(ledger.splitlines())==604
        payload=torch.load(src/'S0600.pt',map_location='cpu',weights_only=False);assert payload['cursor']==8075
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda');im=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);gt=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        image=torch.cat([flip(im,k) for k in range(4)]);y=torch.cat([flip(gt,k) for k in range(4)])
        model=SpatialResidualModel().cuda();model.load_state_dict(payload['model']);model.requires_grad_(False);initial_rng=capture_rng_state();zero=torch.zeros((8,16,16,16),device='cuda');q=(zero.clone(),zero.clone())
        def counted_bound(q,kind):
            b,t,d=model.bound(image,y,q);costs['bound']+=1;assert costs['bound']<=36200
            append(root/'bound_calls.jsonl',dict(index=costs['bound'],kind=kind))
            if kind!='candidate' or torch.isfinite(b).all():assert d['normalization_error']<1e-6
            return b,t,d
        # Exact conditional-density reduction at R=0, not equality of different marginal models.
        model.draw()
        try:
            with torch.no_grad():
                h,_=model.labeled_proposal(image,y);j,t,p=model.log_joint(image,h,y);j2,t2,p2=model.conditional_joint(image,h,zero)
                assert torch.equal(j,j2) and all(torch.equal(t[k],t2[k]) for k in t) and torch.equal(p,p2)
                costs['bound']+=1;append(root/'bound_calls.jsonl',dict(index=costs['bound'],kind='zero_conditional_parity'))
        finally:model.clear_draw()
        assert torch.equal(model.residual_kl(q),torch.zeros(8,device='cuda'))
        a=(torch.full_like(zero,.125),torch.full_like(zero,-.2));ref=torch.distributions.kl_divergence(torch.distributions.Normal(a[0],a[1].exp()),torch.distributions.Normal(torch.zeros_like(zero),torch.ones_like(zero))).flatten(1).sum(1)
        err=float((ref-model.residual_kl(a)).abs().max());assert err<1e-3
        for k in range(10):
            qi=tuple(v.clone().requires_grad_(k>=8) for v in q);model.draw()
            try:
                with torch.set_grad_enabled(k>=8):
                    b,t,d=counted_bound(qi,'zero');assert torch.isfinite(b).all()
                    if k>=8:
                        g=torch.autograd.grad(-b.sum()/128**2,qi);assert all(torch.isfinite(v).all() and v.norm()>0 for v in g)
                        costs['zero_reverse']+=1;append(root/'zero_reverse_calls.jsonl',dict(index=costs['zero_reverse'],norms=[float(v.norm()) for v in g]))
            finally:model.clear_draw()
        assert same(payload['model'],model.state_dict()) and all(v.grad is None for v in model.parameters());restore_rng_state(initial_rng)
        write(root/'ZERO_RESULT.json',dict(status='PASS',zero_residual_original_joint_exact=True,prior_posterior_KL_zero=True,KL_reference_max_error=err,finite10=True,reverse=2,optimizer=0,model_unchanged=True))
        # Fresh source RNG per logical step; replay only within the same step's finite candidates.
        restore_rng_state(payload['rng']);q=tuple(v.clone() for v in q)
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
            for layer in model.bayes_layers():layer.sampled=(layer.mean_w,layer.mean_b)
            try:
                with torch.no_grad():
                    h,_=model.labeled_proposal(image,y,sample=False);l=model.spatial_logits(h['c'],q[0]);nll=F.cross_entropy(l,y,reduction='none');d=dice(l.argmax(1),y)
                    row=dict(step=step,flows=flows,conditional_mean_CE=float(nll.mean()),class_NLL=[float(nll[y==k].mean()) for k in range(4)],all_view_Dice=d,qualified24=all(v>=.8 for r in d for v in r),mu_RMS=float(q[0].square().mean().sqrt()),logsd_quantiles=torch.quantile(q[1].detach().flatten(),torch.tensor([.05,.5,.95],device='cuda')).tolist());costs['decoder']+=1
            finally:model.clear_draw()
            assert same(rng,capture_rng_state());append(root/'metrics.jsonl',row);return row
        def checkpoint(step):
            storage(s);save(root/f'S{step:04}.pt',dict(step=step,q=tuple(v.detach().cpu() for v in q),rng=capture_rng_state(),costs=costs.copy()))
        active.update(q=q,step=0);checkpoint(0);first=evaluate(0)
        for step in range(1,501):
            q=tuple(v.detach().requires_grad_() for v in q);start_rng=capture_rng_state();before,_=saa(q,'gradient',True);after_rng=capture_rng_state()
            g=torch.autograd.grad(-before.sum(),q);assert all(torch.isfinite(v).all() for v in g);costs['local_reverse']+=1
            append(root/'reverse_calls.jsonl',dict(index=costs['local_reverse'],step=step,norms=[float(v.norm()) for v in g]))
            chosen=None
            for alpha in (1.,.25,.0625,.015625,.00390625,.0009765625,.000244140625,0.):
                candidate=tuple((v-.05*alpha*256*dv).detach() for v,dv in zip(q,g));restore_rng_state(start_rng);value,_=saa(candidate,'candidate')
                finite=bool(torch.isfinite(value).all());ok=finite and bool((value>=before.detach()-1e-6).all()) and float(value.mean())>float(before.detach().mean())+1e-5
                costs['candidate']+=1;assert costs['candidate']<=4000
                append(root/'candidates.jsonl',dict(index=costs['candidate'],step=step,alpha=alpha,accepted=ok,finite=finite))
                if ok:chosen=(candidate,value,alpha);break
            restore_rng_state(after_rng)
            if chosen is not None:q,value,alpha=chosen;costs['accepted']+=1;append(root/'local_updates.jsonl',dict(index=costs['accepted'],step=step,alpha=alpha))
            else:q=tuple(v.detach() for v in q);value=before.detach();alpha=0.
            append(root/'steps.jsonl',dict(step=step,alpha=alpha,accepted=chosen is not None,before=before.detach().cpu().tolist(),after=value.cpu().tolist()))
            active.update(q=q,step=step)
            if step%25==0:checkpoint(step);write(root/'STATUS.json',dict(status='RUNNING',step=step,costs=costs,space=storage(s)))
            if step%100==0:last=evaluate(step)
        assert costs['local_reverse']==500 and costs['decoder']==6 and same(payload['model'],model.state_dict()) and all(v.grad is None for v in model.parameters())
        assert (src/'physical_updates.jsonl').read_text()==ledger
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,initial=first,final=last,model_source604_unchanged=True,no_heldout_or_U=True,source_cursor=8075,model_optimizer=0,space=storage(s)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:storage(s);save(root/'failure_partial.pt',dict(step=active['step'],q=tuple(v.detach().cpu() for v in active['q']),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
