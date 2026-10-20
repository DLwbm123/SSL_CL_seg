"""Paired spatial residual resolution, frozen decoder and common baseline."""
import json,os,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.checkpoint import capture_rng_state
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.run import same
from sc_bayeseg.diagnostic_v1.run import dice
from sc_bayeseg.window_v1.discrete_labeled import DiscreteLabeledModel
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(reverse=0,candidate=0,forward=0,accepted=0,optimizer=0,model_applications=0);active={}
    def forbid(*a,**kw):raise AssertionError('optimizer forbidden')
    torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        src=Path(s['discrete_recon']);ledger=(src/'physical_updates.jsonl').read_text();assert len(ledger.splitlines())==1208
        local=Path(s['feature_local']);history=(local/'local_updates.jsonl').read_text();assert len(history.splitlines())==200;prior=json.loads((local/'RESULTS.json').read_text());sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda');im=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);gt=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        image=torch.cat([flip(im,k) for k in range(4)]);y=torch.cat([flip(gt,k) for k in range(4)])
        model=DiscreteLabeledModel().cuda();model.requires_grad_(False);rng=capture_rng_state();results={}
        def score(c,arm,step,kind):
            feature=(base+F.interpolate(c,(128,128),mode='bilinear',align_corners=False)).clamp_min(0);a=model.anatomy[2](F.relu(model.anatomy[1](feature)));logits=torch.cat([torch.zeros_like(a[:,:1]),a],1);ce=F.cross_entropy(logits,y,reduction='none');v=ce.mean((1,2))
            costs['forward']+=1;assert costs['forward']<=18010
            append(root/'forward_calls.jsonl',dict(index=costs['forward'],arm=arm,step=step,kind=kind))
            return v,logits,ce
        def evaluate(c,arm,step,out):
            with torch.no_grad():
                v,l,nll=score(c,arm,step,'evaluation');d=dice(l.argmax(1),y)
                row=dict(step=step,full_feature_RMS=float(((base+F.interpolate(c,(128,128),mode='bilinear',align_corners=False)).clamp_min(0)-base).square().mean().sqrt()),CE=float(v.mean()),per_view_CE=v.tolist(),class_NLL=[float(nll[y==k].mean()) for k in range(4)],all_view_Dice=d,qualified24=all(x>=.8 for r in d for x in r))
            append(out/'metrics.jsonl',row);return row
        for arm,resolution in [('r4',4),('r16',16)]:
            out=root/arm;out.mkdir(exist_ok=False);p=torch.load(src/'code'/'S0600.pt',map_location='cpu',weights_only=False);assert p['cursor']==8675
            model.load_state_dict(p['model'])
            for layer in model.bayes_layers():layer.sampled=(layer.mean_w,layer.mean_b)
            with torch.no_grad():
                h=torch.load(local/'code'/'S100.pt',map_location='cuda',weights_only=False);base=F.interpolate(h,(128,128),mode='bilinear',align_corners=False).detach();c=torch.zeros((8,16,resolution,resolution),device='cuda')
            start=c.clone();initial=evaluate(c,arm,0,out);assert abs(initial['CE']-prior['arms']['code']['final']['CE'])<1e-7
            storage(s);save(out/'S0000.pt',c.cpu());accepted=0;active.update(out=out,c=c,step=0)
            for step in range(1,1001):
                c=c.detach().requires_grad_();v,_,_=score(c,arm,step,'gradient');g=torch.autograd.grad(v.sum(),c)[0];assert torch.isfinite(g).all()
                costs['reverse']+=1;assert costs['reverse']<=2000
                append(root/'reverse_calls.jsonl',dict(index=costs['reverse'],arm=arm,step=step,gradient_norm=float(g.norm())))
                chosen=None
                for alpha in (1.,.25,.0625,.015625,.00390625,.0009765625,.000244140625,0.):
                    candidate=(c-.05*alpha*resolution**2*g).detach()
                    with torch.no_grad():after,_,_=score(candidate,arm,step,'candidate')
                    ok=bool(torch.isfinite(after).all()) and bool((after<=v.detach()+1e-6).all()) and float(after.mean())<float(v.detach().mean())-1e-7
                    costs['candidate']+=1;assert costs['candidate']<=16000
                    append(root/'candidates.jsonl',dict(index=costs['candidate'],arm=arm,step=step,alpha=alpha,accepted=ok))
                    if ok:chosen=(candidate,after,alpha);break
                if chosen is not None:
                    c,after,alpha=chosen;costs['accepted']+=1;accepted+=1
                    append(root/'local_updates.jsonl',dict(index=costs['accepted'],arm=arm,step=step,alpha=alpha))
                else:c=c.detach();after=v.detach();alpha=0.
                append(out/'steps.jsonl',dict(step=step,alpha=alpha,accepted=chosen is not None,before=v.detach().tolist(),after=after.tolist(),residual_RMS=float((F.interpolate(c,(128,128),mode='bilinear',align_corners=False)).square().mean().sqrt())))
                active.update(c=c,step=step)
                if step%25==0:storage(s);save(out/f'S{step:04}.pt',c.cpu())
                if step%250==0:last=evaluate(c,arm,step,out);write(root/'STATUS.json',dict(status='RUNNING',arm=arm,step=step,costs=costs,space=storage(s)))
            model.clear_draw();assert same(p['model'],model.state_dict()) and all(v.grad is None for v in model.parameters())
            results[arm]=dict(initial=initial,final=last,accepted=accepted,residual_RMS=float((F.interpolate(c,(128,128),mode='bilinear',align_corners=False)).square().mean().sqrt()),model_unchanged=True,initial_parity=True);active.clear()
        assert same(rng,capture_rng_state()) and (src/'physical_updates.jsonl').read_text()==ledger and (local/'local_updates.jsonl').read_text()==history
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,arms=results,model_RNG_source1208_and_featurelocal200_unchanged=True,no_heldout_or_U=True,space=storage(s)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:storage(s);save(active['out']/'failure_partial.pt',dict(step=active['step'],c=active['c'].detach().cpu(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
