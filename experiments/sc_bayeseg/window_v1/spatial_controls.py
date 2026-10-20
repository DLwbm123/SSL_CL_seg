"""Same-source M/S controls, completed V reused without rerunning."""
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
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(local_reverse=0,bound=0,candidate=0,accepted=0,decoder=0,mean_forward=0,optimizer=0,model_applications=0);active={}
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
            append(root/'bound_calls.jsonl',dict(index=costs['bound'],kind=kind,arm=name))
            if kind!='candidate' or torch.isfinite(b).all():assert d['normalization_error']<1e-6
            return b,t,d
        refroot=Path(s['spatial_residual']);assert json.loads((refroot/'ZERO_RESULT.json').read_text())['status']=='PASS'
        reference=json.loads((refroot/'RESULTS.json').read_text());reference_ledger=(refroot/'local_updates.jsonl').read_text();assert len(reference_ledger.splitlines())==500
        results={};rng_pairs=[]
        def saa(q,kind,grad=False,regular=True):
            values=[];parts=[]
            with torch.set_grad_enabled(grad):
                for _ in range(2):
                    model.draw()
                    try:
                        for _ in range(4):
                            b,t,d=counted_bound(q,kind);values.append((b+(0 if regular else d['residual_KL']))/128**2)
                            parts.append({**{k:v.detach()/128**2 for k,v in t.items()},'negative_logq':-d['logq'].detach()/128**2,'residual_KL':d['residual_KL'].detach()/128**2})
                    finally:model.clear_draw()
                return torch.stack(values).mean(0),{k:torch.stack([v[k] for v in parts]).mean(0).cpu().tolist() for k in parts[0]}
        def mean_logits(q):
            for layer in model.bayes_layers():layer.sampled=(layer.mean_w,layer.mean_b)
            try:
                h,_=model.labeled_proposal(image,y,sample=False)
                value=model.spatial_logits(h['c'],q[0]);costs['mean_forward']+=1;assert costs['mean_forward']<=4512
                append(root/'mean_forward_calls.jsonl',dict(index=costs['mean_forward'],arm=name))
                return value
            finally:model.clear_draw()
        def objective(q,kind,grad=False):
            if name=='S':return saa(q,kind,grad,regular=False)
            with torch.set_grad_enabled(grad):
                nll=F.cross_entropy(mean_logits(q),y,reduction='none');return -nll.mean((1,2)),{}
        def evaluate(step):
            rng=capture_rng_state();flows=[]
            for seed in (50163,80163):
                seed_everything(seed);v,t=saa(q,'readout');flows.append(dict(seed=seed,ELBO=v.cpu().tolist(),parts=t,global_Theta_KL_constant=float(model.kl()/(2*128**2))))
            restore_rng_state(rng)
            for layer in model.bayes_layers():layer.sampled=(layer.mean_w,layer.mean_b)
            try:
                with torch.no_grad():
                    l=mean_logits(q);nll=F.cross_entropy(l,y,reduction='none');d=dice(l.argmax(1),y)
                    row=dict(step=step,flows=flows,conditional_mean_CE=float(nll.mean()),class_NLL=[float(nll[y==k].mean()) for k in range(4)],all_view_Dice=d,qualified24=all(v>=.8 for r in d for v in r),mu_RMS=float(q[0].square().mean().sqrt()),logsd_quantiles=torch.quantile(q[1].detach().flatten(),torch.tensor([.05,.5,.95],device='cuda')).tolist());costs['decoder']+=1
            finally:model.clear_draw()
            assert same(rng,capture_rng_state());append(out/'metrics.jsonl',row);return row
        def checkpoint(step):
            storage(s);save(out/f'S{step:04}.pt',dict(step=step,q=tuple(v.detach().cpu() for v in q),rng=capture_rng_state(),costs=costs.copy()))
        for name in ('M','S'):
            out=root/name;out.mkdir(exist_ok=False);q=(zero.clone(),zero.clone());restore_rng_state(payload['rng'])
            active.update(out=out,q=q,step=0)
            if name=='S':
                assert same(capture_rng_state(),torch.load(refroot/'S0000.pt',map_location='cpu',weights_only=False)['rng']);rng_pairs.append(0);append(root/'RNG_PAIRS.jsonl',dict(step=0,exact=True))
            checkpoint(0);first=evaluate(0);assert same(first['flows'],reference['initial']['flows'])
            for step in range(1,501):
                q=tuple(v.detach().requires_grad_(name=='S' or j==0) for j,v in enumerate(q));start_rng=capture_rng_state();before,_=objective(q,'gradient',True);after_rng=capture_rng_state()
                g=torch.autograd.grad(-before.sum(),q) if name=='S' else (torch.autograd.grad(-before.sum(),q[0])[0],torch.zeros_like(q[1]));assert all(torch.isfinite(v).all() for v in g);costs['local_reverse']+=1
                append(root/'reverse_calls.jsonl',dict(index=costs['local_reverse'],arm=name,step=step,norms=[float(v.norm()) for v in g]))
                chosen=None
                for alpha in (1.,.25,.0625,.015625,.00390625,.0009765625,.000244140625,0.):
                    candidate=tuple((v-.05*alpha*256*dv).detach() for v,dv in zip(q,g));restore_rng_state(start_rng);value,_=objective(candidate,'candidate')
                    finite=bool(torch.isfinite(value).all());ok=finite and bool((value>=before.detach()-1e-6).all()) and float(value.mean())>float(before.detach().mean())+1e-5
                    costs['candidate']+=1;assert costs['candidate']<=8000
                    append(root/'candidates.jsonl',dict(index=costs['candidate'],arm=name,step=step,alpha=alpha,accepted=ok,finite=finite))
                    if ok:chosen=(candidate,value,alpha);break
                restore_rng_state(after_rng)
                if chosen is not None:q,value,alpha=chosen;costs['accepted']+=1;append(root/'local_updates.jsonl',dict(index=costs['accepted'],arm=name,step=step,alpha=alpha))
                else:q=tuple(v.detach() for v in q);value=before.detach();alpha=0.
                append(out/'steps.jsonl',dict(step=step,alpha=alpha,accepted=chosen is not None,before=before.detach().cpu().tolist(),after=value.cpu().tolist()))
                active.update(q=q,step=step)
                if name=='M':assert torch.count_nonzero(q[1])==0
                if step%25==0 and name=='S':
                    original=torch.load(refroot/f'S{step:04}.pt',map_location='cpu',weights_only=False);assert same(capture_rng_state(),original['rng']);rng_pairs.append(step);append(root/'RNG_PAIRS.jsonl',dict(step=step,exact=True))
                if step%25==0:checkpoint(step);write(root/'STATUS.json',dict(status='RUNNING',arm=name,step=step,costs=costs,space=storage(s)))
                if step%100==0:last=evaluate(step)
            assert same(payload['model'],model.state_dict()) and all(v.grad is None for v in model.parameters())
            results[name]=dict(initial=first,final=last,model_unchanged=True,logsd_frozen_zero=name=='M')
        assert costs['local_reverse']==1000 and costs['decoder']==12 and len(rng_pairs)==21
        assert (src/'physical_updates.jsonl').read_text()==ledger and (refroot/'local_updates.jsonl').read_text()==reference_ledger
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,arms=results,reused_V=reference['final'],reference_costs_not_new=reference['costs'],S_V_RNG_pairs=21,model_source604_V500_unchanged=True,no_heldout_or_U=True,model_optimizer=0,space=storage(s)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:storage(s);save(active['out']/'failure_partial.pt',dict(step=active['step'],q=tuple(v.detach().cpu() for v in active['q']),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
