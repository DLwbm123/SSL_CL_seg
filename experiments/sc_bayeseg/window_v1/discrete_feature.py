"""Frozen code tangent versus nonnegative 4x4 feature diagnostic."""
import json,os,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.checkpoint import capture_rng_state
from care_seg_study_v1.run import write,append
from sc_bayeseg.run import same
from sc_bayeseg.diagnostic_v1.run import dice
from sc_bayeseg.window_v1.discrete_labeled import DiscreteLabeledModel
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(forward=0,reverse=0,candidate=0,optimizer=0,model_applications=0,local_updates=0)
    def forbid(*a,**kw):raise AssertionError('optimizer forbidden')
    torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        src=Path(s['discrete_recon']);local=Path(s['code_local']);ledger=(src/'physical_updates.jsonl').read_text();local_ledger=(local/'local_updates.jsonl').read_text();assert len(ledger.splitlines())==1208 and len(local_ledger.splitlines())==200
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda');gt=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long();y=torch.cat([flip(gt,k) for k in range(4)])
        model=DiscreteLabeledModel().cuda();model.requires_grad_(False);rng=capture_rng_state();results={}
        def from_h(h):return model.anatomy[2](F.relu(model.anatomy[1](F.interpolate(h,(128,128),mode='bilinear',align_corners=False))))
        def count(arm,kind):
            costs['forward']+=1;assert costs['forward']<=22;append(root/'forward_calls.jsonl',dict(index=costs['forward'],arm=arm,kind=kind))
        def stats(a):
            logits=torch.cat([torch.zeros_like(a[:,:1]),a],1);nll=F.cross_entropy(logits,y,reduction='none')
            return dict(CE=float(nll.mean()),per_view_CE=nll.mean((1,2)).tolist(),class_NLL=[float(nll[y==k].mean()) for k in range(4)],all_view_Dice=dice(logits.argmax(1),y)),nll
        for arm in ('fixed','code'):
            p=torch.load(src/arm/'S0600.pt',map_location='cpu',weights_only=False);model.load_state_dict(p['model'])
            for layer in model.bayes_layers():layer.sampled=(layer.mean_w,layer.mean_b)
            c0=torch.load(local/arm/'S000.pt',map_location='cuda',weights_only=False);c=torch.load(local/arm/'S100.pt',map_location='cuda',weights_only=False)
            with torch.no_grad():
                pre=model.anatomy[0](c);h=F.relu(pre).reshape(8,16,4,4);h0=F.relu(model.anatomy[0](c0)).reshape_as(h)
                count(arm,'h_parity');a=from_h(h);initial,_=stats(a);count(arm,'code_parity');assert torch.equal(a,model.anatomy_mean(c))
            h=h.detach().requires_grad_();count(arm,'gradient');_,nll=stats(from_h(h));grads=[]
            for k in range(5):
                loss=nll.mean() if k==4 else nll[y==k].mean();g=torch.autograd.grad(loss,h,retain_graph=k<4)[0];assert torch.isfinite(g).all();grads.append(g.detach())
                costs['reverse']+=1;append(root/'reverse_calls.jsonl',dict(index=costs['reverse'],arm=arm,term=k))
            g=grads[-1];projected=[];ranks=[];fractions=[];residuals=[]
            weight=model.anatomy[0].mean_w.detach().double().cpu()
            for i in range(8):
                J=(pre[i].detach().cpu()>0).double()[:,None]*weight;u,sv,v=torch.linalg.svd(J,full_matrices=False);rank=int((sv>sv.max()*1e-10).sum());Q=u[:,:rank]
                gi=g[i].flatten().double().cpu();proj=Q@(Q.T@gi);assert torch.linalg.norm(Q.T@(gi-proj))<1e-10
                residual=float(torch.linalg.norm(Q@(Q.T@proj)-proj));assert residual<1e-10
                projected.append(proj.to(device='cuda',dtype=h.dtype).reshape_as(h[i]));ranks.append(rank);fractions.append(float(proj.norm()/gi.norm().clamp_min(1e-30)));residuals.append(residual)
            full=-g.clone();full[(h.detach()==0)&(full<0)]=0;tangent=-torch.stack(projected);tangent[h.detach()==0]=0
            directions={}
            for name,d in [('full_h',full),('code_tangent_h',tangent)]:
                rms=d.square().mean((1,2,3),keepdim=True).sqrt();directions[name]=torch.where(rms>0,d/rms.clamp_min(1e-30),torch.zeros_like(d))
            observed=(h.detach()-h0).square().mean((1,2,3),keepdim=True).sqrt();radius=observed.clone()
            for d in directions.values():
                bound=torch.where(d<0,h.detach()/(-d).clamp_min(1e-30),torch.full_like(d,float('inf'))).flatten(1).min(1).values[:,None,None,None]
                radius=torch.minimum(radius,bound*.5)
            candidates=[]
            for name,d in directions.items():
                for alpha in (1.,.25,.0625,.015625):
                    delta=alpha*radius*d;candidate=h.detach()+delta;assert candidate.min()>=-1e-7
                    with torch.no_grad():count(arm,'candidate');row,_=stats(from_h(candidate))
                    costs['candidate']+=1;append(root/'candidates.jsonl',dict(index=costs['candidate'],arm=arm,direction=name,alpha=alpha))
                    candidates.append(dict(direction=name,alpha=alpha,metrics=row,feature_RMS=delta.square().mean((1,2,3)).sqrt().tolist(),first_order_total_CE=float((g*delta).sum())))
            model.clear_draw();assert same(p['model'],model.state_dict()) and all(v.grad is None for v in model.parameters())
            results[arm]=dict(initial=initial,rank=ranks,total_gradient_code_fraction=fractions,projection_idempotence_error=residuals,observed_radius=observed.flatten().tolist(),effective_radius=radius.flatten().tolist(),per_class_h_gradient_norm=[float(v.norm()) for v in grads],candidates=candidates,model_unchanged=True,exact_h_code_parity=True)
        assert costs['forward']==22 and costs['reverse']==10 and costs['candidate']==16
        assert same(rng,capture_rng_state()) and (src/'physical_updates.jsonl').read_text()==ledger and (local/'local_updates.jsonl').read_text()==local_ledger
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,arms=results,model_RNG_source_ledgers_unchanged=True,no_heldout_or_U=True,space=storage(s)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
