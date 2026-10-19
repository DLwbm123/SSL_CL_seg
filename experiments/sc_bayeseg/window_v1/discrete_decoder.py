"""Training-only categorical decoder information and gradient audit; no updates."""
import json,os,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from care_seg_study_v1.run import write,append
from sc_bayeseg.run import same
from sc_bayeseg.diagnostic_v1.run import dice
from sc_bayeseg.window_v1.discrete_labeled import DiscreteLabeledModel
from sc_bayeseg.window_v1.common import storage


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(forward=0,reverse=0,optimizer=0,parameter_applications=0)
    def forbid(*args,**kwargs):raise AssertionError('optimizer forbidden')
    torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        src=Path(s['discrete_train']);states={str(k):torch.load(src/f'S{k:04}.pt',map_location='cpu',weights_only=False)['model'] for k in (0,600)}
        ledger=(src/'physical_updates.jsonl').read_text();assert len(ledger.splitlines())==604
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda');image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);gt=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        model=DiscreteLabeledModel().cuda();model.requires_grad_(False);rng=capture_rng_state()
        frequencies=torch.bincount(gt.flatten(),minlength=4).float()/gt.numel();entropy=float(-(frequencies*frequencies.clamp_min(1e-30).log()).sum());results={};gradients={}
        def metrics(logits):
            p=logits.softmax(1);nll=F.cross_entropy(logits,gt,reduction='none');pred=logits.argmax(1)
            return dict(CE=float(nll.mean()),class_NLL=[float(nll[gt==k].mean()) for k in range(4)],predicted_fraction=[float((pred==k).float().mean()) for k in range(4)],mean_probability=p.mean((0,2,3)).tolist(),GT_Dice_given_labels=dice(pred,gt))
        for theta in ('0','600'):
            sd=states['600'].copy()
            for k in sd:
                if k.startswith(('anatomy.','structure.')):sd[k]=states[theta][k]
            model.load_state_dict(sd);before_rng=capture_rng_state();codes={};probs={}
            for label,z in [('GT',gt),('BG',torch.zeros_like(gt))]:
                name=f'theta{theta}_{label}';rows=[]
                for draw in range(9):
                    if draw==0:
                        for layer in model.bayes_layers():layer.sampled=(layer.mean_w,layer.mean_b)
                    else:
                        # Matching each GT/BG draw includes the code Gaussian stream.
                        seed_everything(50163+draw-1);model.draw()
                    try:
                        with torch.no_grad():
                            h,_=model.labeled_proposal(image,z,sample=draw>0);c=h['c'];a=model.anatomy_mean(c);logits=torch.cat([torch.zeros_like(a[:,:1]),a],1)
                            rows.append(metrics(logits));costs['forward']+=1;append(root/'forward_calls.jsonl',dict(index=costs['forward'],cell=name,draw=draw))
                            if draw==0:codes[label]=c.detach();probs[label]=logits.softmax(1).detach()
                        if draw==0 and label=='GT':
                            params=[v for n,v in model.named_parameters() if n.startswith('anatomy.') and ('.mean_w' in n or '.mean_b' in n)]
                            for v in params:v.requires_grad_(True)
                            c=c.detach().requires_grad_();a=model.anatomy_mean(c);logits=torch.cat([torch.zeros_like(a[:,:1]),a],1);nll=F.cross_entropy(logits,gt,reduction='none');vectors=[];summary=[]
                            for k in range(5):
                                loss=nll.mean() if k==4 else nll[gt==k].mean();g=torch.autograd.grad(loss,[c]+params,retain_graph=k<4);assert all(torch.isfinite(v).all() for v in g)
                                vector=torch.cat([v.detach().flatten() for v in g[1:]]);vectors.append(vector)
                                summary.append(dict(loss=float(loss.detach()),code_gradient_norm=float(g[0].norm()),anatomy_gradient_norm=float(vector.norm())))
                                costs['reverse']+=1;append(root/'reverse_calls.jsonl',dict(index=costs['reverse'],theta=theta,term=k))
                            v=torch.stack(vectors);gram=v@v.T;norm=v.norm(dim=1);cos=gram/(norm[:,None]*norm[None,:]).clamp_min(1e-30)
                            gradients[theta]=dict(terms=summary,anatomy_gradient_cosine=cos.cpu().tolist())
                            for v in params:v.requires_grad_(False)
                    finally:model.clear_draw()
                results[name]=dict(mean=rows[0],samples=rows[1:])
            results['theta'+theta+'_sensitivity']=dict(code_GT_BG_RMS=float((codes['GT']-codes['BG']).square().mean().sqrt()),probability_GT_BG_RMS=float((probs['GT']-probs['BG']).square().mean().sqrt()))
            assert same(sd,model.state_dict()) and all(v.grad is None for v in model.parameters());restore_rng_state(before_rng)
        restore_rng_state(rng);assert same(rng,capture_rng_state());assert (src/'physical_updates.jsonl').read_text()==ledger
        assert costs['forward']==36 and costs['reverse']==10
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,training_class_fraction=frequencies.cpu().tolist(),constant_frequency_CE=entropy,cells=results,gradients=gradients,all_model_and_RNG_unchanged=True,no_heldout_or_U=True,source604_ledger_unchanged=True,space=storage(s)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
