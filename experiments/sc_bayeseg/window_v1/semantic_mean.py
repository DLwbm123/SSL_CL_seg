"""Frozen-nuisance semantic mean-only local ELBO ascent."""
import json,os,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.split_v1.model import Model
from sc_bayeseg.guard_v1.run import dice
from sc_bayeseg.window_v1.common import storage


def metrics(p,y):
    out={}
    for name,ix in [('train',slice(0,2)),('heldout',slice(2,4))]:
        pp,yy=p[ix],y[ix];pred=pp.argmax(1);ce=-pp.clamp_min(1e-30).log().gather(1,yy[:,None])[:,0]
        out[name]=dict(dice=dice(pred,yy),ce=float(ce.mean()),class2_ce=float(ce[yy==2].mean()),class2_confusion=[int(((yy==2)&(pred==c)).sum()) for c in range(4)])
    return out


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False)
    costs=dict(model_optimizer=0,model_parameter_updates=0,local_reverse=0,local_updates=0,candidates=0,elbo_evaluations=0,predictive_batches=0)
    active={};private={};groups={}
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());assert len(set(sel['identities']))==4 and not set(sel['train_indices'])&set(sel['heldout_indices'])
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']+sel['heldout_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False)
        y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        local=torch.load(s['states']['terminal'],map_location='cpu',weights_only=False)
        for post in ['terminal','prior']:
            sd=local['model'].copy()
            if post=='prior':
                source=torch.load(s['anchor'],map_location='cpu',weights_only=False)['model']
                for key in sd:
                    if key.startswith(('anatomy.','structure.')):
                        prefix,leaf=key.rsplit('.',1)
                        if leaf in ['mean_w','mean_b','logsd_w','logsd_b']:
                            assert torch.equal(source[prefix+'.prior_'+leaf],source[prefix+'.initial_'+leaf]);sd[key]=source[prefix+'.prior_'+leaf]
                        else:sd[key]=source[key]
            model=Model().cuda();model.load_state_dict(sd);model.requires_grad_(False)
            before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            shifts={k:tuple(torch.zeros(shape,device='cuda',requires_grad=True) for _ in range(2)) for k,shape in dict(x=(4,1,128,128),m=(4,1,128,128),ell=(4,3,128,128),c=(4,16)).items()}
            def evaluate(state,grad=False,seed=50163):
                costs['elbo_evaluations']+=1;seed_everything(seed);values=[]
                with torch.set_grad_enabled(grad):
                    try:
                        for _ in range(2):
                            model.draw()
                            for _ in range(4):
                                h,logq=model.proposal(image,state);joint,_,_=model.log_joint(image,h);values.append((joint-logq)/128**2)
                        value=torch.stack(values).mean(0);assert torch.isfinite(value).all();return value
                    finally:model.clear_draw()
            def predict(state):
                costs['predictive_batches']+=1;seed_everything(50163)
                with torch.no_grad():
                    h,_=model.proposal(image,state,sample=False);mean=torch.cat([torch.zeros_like(h['ell'][:,:1]),h['ell']],1).softmax(1);weights=[];probs=[]
                    try:
                        for _ in range(2):
                            model.draw()
                            for _ in range(4):
                                h,logq=model.proposal(image,state);joint,_,p=model.log_joint(image,h);weights.append(joint-logq);probs.append(p)
                        w=torch.stack(weights).softmax(0);p=(w[:,:,None,None,None]*torch.stack(probs)).sum(0)
                        return dict(mean=mean.cpu(),SNIS=p.cpu()),dict(ess_per_image=[float(v) for v in 1/w.square().sum(0)],max_weight_per_image=[float(v) for v in w.max(0).values])
                    finally:model.clear_draw()
            baseline=evaluate(shifts).detach();assert torch.equal(baseline,evaluate(shifts).detach())
            cross_initial=evaluate(shifts,seed=80163).detach();initial_pred,initial_diag=predict(shifts)
            save(root/(post+'_S00.pt'),{k:[v.detach().cpu() for v in pair] for k,pair in shifts.items()})
            trace=[]
            for step in range(1,26):
                active=dict(posterior=post,step=step,shifts=shifts)
                costs['local_reverse']+=1;append(root/'reverse_calls.jsonl',dict(index=costs['local_reverse'],posterior=post,step=step))
                value=evaluate(shifts,grad=True);old=value.detach();flat=[v for pair in shifts.values() for v in pair]
                active_grad=torch.autograd.grad(-value.sum(),[shifts['ell'][0],shifts['c'][0]])
                grad=[torch.zeros_like(v) for v in flat];grad[4],grad[6]=active_grad
                direction=[-.05*g*(128**2 if v.ndim==4 else 1) for v,g in zip(flat,grad)]
                chosen=0.;after=old;selected=shifts
                for alpha in [1.,.25,.0625,.015625,.00390625,.0009765625,.000244140625,0.]:
                    costs['candidates']+=1;ix=costs['candidates'];append(root/'candidate_evaluations.jsonl',dict(index=ix,posterior=post,step=step,alpha=alpha))
                    candidate=[(v+alpha*d).detach().requires_grad_() for v,d in zip(flat,direction)]
                    trial={k:tuple(candidate[2*i:2*i+2]) for i,k in enumerate(shifts)}
                    score=evaluate(trial).detach();accepted=bool((score>=old-1e-6).all() and score.mean()>old.mean()+1e-5)
                    append(root/'candidate_outcomes.jsonl',dict(index=ix,per_image=[float(v) for v in score],accepted=accepted))
                    if accepted:chosen=alpha;after=score;selected=trial;break
                if chosen:
                    costs['local_updates']+=1;append(root/'local_updates.jsonl',dict(index=costs['local_updates'],posterior=post,step=step,alpha=chosen))
                labels=[k+'_'+part for k in shifts for part in ['mean','logsd']]
                row=dict(step=step,alpha=chosen,elbo_before=[float(v) for v in old],elbo_after=[float(v) for v in after],blocks={k:dict(gradient_rms=float(g.square().mean().sqrt()),delta_rms=float((chosen*d).square().mean().sqrt()),shift_rms=float(a.square().mean().sqrt())) for k,g,d,a in zip(labels,grad,direction,[v for pair in selected.values() for v in pair])})
                trace.append(row);append(root/(post+'_steps.jsonl'),row);shifts=selected
                write(root/'STATUS.json',dict(phase='LOCAL_INFERENCE',posterior=post,step=step,costs=costs))
            final=evaluate(shifts).detach();assert torch.equal(final,after)
            cross_final=evaluate(shifts,seed=80163).detach();final_pred,final_diag=predict(shifts)
            assert all(torch.count_nonzero(v)==0 for k,pair in shifts.items() for i,v in enumerate(pair) if not (k in ['ell','c'] and i==0))
            save(root/(post+'_S25.pt'),{k:[v.detach().cpu() for v in pair] for k,pair in shifts.items()})
            assert all(torch.equal(v.cpu(),before[k]) for k,v in model.state_dict().items())
            private[post]=dict(initial=initial_pred,final=final_pred)
            groups[post]=dict(trace=trace,initial_elbo=[float(v) for v in baseline],final_elbo=[float(v) for v in final],cross_initial_elbo=[float(v) for v in cross_initial],cross_final_elbo=[float(v) for v in cross_final],initial_diagnostics=initial_diag,final_diagnostics=final_diag,model_unchanged=True)
        # No label-dependent scoring until both optimizations have ended.
        for post in groups:
            for state in ['initial','final']:groups[post][state+'_metrics']={mode:metrics(p,y.cpu()) for mode,p in private[post][state].items()}
        assert costs['local_reverse']==50 and costs['local_updates']<=50 and costs['candidates']<=400
        storage(s);save(root/'PRIVATE_PREDICTIONS.pt',private)
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,groups=groups,space=storage(s),no_label_in_optimization=True,all_models_unchanged=True,only_ell_c_mean_shift=True));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        save(root/'PRIVATE_PARTIAL.pt',dict(active=active,costs=costs))
        write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
