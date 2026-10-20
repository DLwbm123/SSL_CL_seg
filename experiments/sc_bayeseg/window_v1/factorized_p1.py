"""Bounded local A/B inference; no global model optimization."""
import json, math, os, traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset, collate_labeled
from lcrseg.engine.trainer import seed_everything
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from care_seg_study_v1.run import write, append, save
from sc_bayeseg.run import same
from sc_bayeseg.model import sar, student_log_prob, normal_log_prob
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.common import storage


def entropy(q):
    return (q[2]+q[3]+math.log(2*math.pi*math.e)).flatten(1).sum(1)


def self_check():
    q=[torch.tensor([[[[v]]]],dtype=torch.float64) for v in (0.,0.,-.7,-.2,.4)]
    l=torch.stack([q[2].exp().flatten()[0],torch.tensor(0.,dtype=torch.float64),q[4].flatten()[0],q[3].exp().flatten()[0]]).reshape(2,2)
    expected=torch.distributions.MultivariateNormal(torch.zeros(2,dtype=torch.float64),scale_tril=l).entropy()
    assert abs(float(expected-entropy(q)))<1e-12
    independent=sum(torch.distributions.Normal(torch.tensor(0.),q[i].exp()).entropy() for i in (2,3))
    assert abs(float(independent-entropy(q)))<1e-12
    return dict(joint_entropy_error=abs(float(expected-entropy(q))),zero_correlation_entropy_parity=True,cholesky_positive=True)


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False)
    costs=dict(reverse=0,candidate=0,accepted=0,density=0,optimizer=0,model_applications=0);active={}
    try:
        storage(s);check=self_check();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        src=Path(s['spatial_generator']);hist=Path(s['spatial_hypothesis']);ledger=(src/'physical_updates.jsonl').read_text();previous=(hist/'reverse_calls.jsonl').read_text()
        assert len(ledger.splitlines())==604 and len(previous.splitlines())==1000
        p=torch.load(src/'S0600.pt',map_location='cpu',weights_only=False);assert p['cursor']==8675
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text())
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda');image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);gt=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        model=SpatialResidualModel().cuda();model.load_state_dict(p['model']);model.requires_grad_(False);original_rng=capture_rng_state();results={};N=128**2
        def density(q,kind,fz=None):
            if block=='A':
                cm,cl,rm,rl=q;c=cm+cl.exp()*torch.randn_like(cm);r=rm+rl.exp()*torch.randn_like(rm)
                logits=model.spatial_logits(c,r);cat=-F.cross_entropy(logits,y,reduction='none').sum((-2,-1))
                kc=.5*(cm.square()+(2*cl).exp()-1-2*cl).flatten(1).sum(1);kr=model.residual_kl((rm,rl))
                terms=dict(categories=cat,negative_KLc=-kc,negative_KLR=-kr)
            else:
                mx,mm,lx,lm,b=q;e1=torch.randn_like(mx);e2=torch.randn_like(mm)
                x=mx+lx.exp()*e1;m=mm+b*e1+lm.exp()*e2
                terms=dict(image=student_log_prob(image-x-m,.1),shape=student_log_prob(sar(x-fz),.2)+model.logdet,appearance=normal_log_prob(m,torch.zeros_like(m),torch.full_like(m,math.log(.5))),joint_entropy=entropy(q))
            costs['density']+=1;assert costs['density']<=130000
            append(root/'density_calls.jsonl',dict(index=costs['density'],branch=key,kind=kind))
            return sum(terms.values())/N,{k:v.detach()/N for k,v in terms.items()}
        def saa(q,kind,draws=2,grad=False):
            vals=[];terms=[]
            with torch.set_grad_enabled(grad):
                for _ in range(draws):
                    model.draw()
                    try:
                        fz=model.structure_mean(F.one_hot(y,4).permute(0,3,1,2).to(image.dtype)) if block=='B' else None
                        for _ in range(4):
                            value,t=density(q,kind,fz);vals.append(value);terms.append(t)
                    finally:model.clear_draw()
            return torch.stack(vals).mean(0),{k:torch.stack([t[k] for t in terms]).mean(0).cpu().tolist() for k in terms[0]}
        def evaluate(q):
            rng=capture_rng_state();seed_everything(80163);v,t=saa(q,'evaluation',32);restore_rng_state(rng)
            assert torch.isfinite(v).all();return dict(value=v.cpu().tolist(),parts=t)
        for block in ('A','B'):
            for init in (0,1):
                for name in ('GT','BG'):
                    key=f'{block}_init{init}_{name}';out=root/key;out.mkdir();y=gt if name=='GT' else torch.zeros_like(gt)
                    with torch.no_grad():
                        if block=='A':
                            onehot=F.one_hot(y,4).permute(0,3,1,2).to(image.dtype)
                            _,_,cm,cl=model.gaussian(model.code_encoder(onehot[:,1:]),False)
                            old=torch.load(hist/name/'S0500.pt',map_location='cpu',weights_only=False)['q']
                            q=(cm,cl,old[0].to('cuda'),old[1].to('cuda')) if init==0 else (torch.zeros_like(cm),torch.zeros_like(cl),torch.zeros_like(old[0],device='cuda'),torch.zeros_like(old[1],device='cuda'))
                        else:
                            _,_,mx,lx=model.gaussian(model.physical_encoder(image),False);_,_,mm,lm=model.gaussian(model.appearance_encoder(image),False)
                            if init==1:
                                for layer in model.bayes_layers():layer.sampled=(layer.mean_w,layer.mean_b)
                                try:mx=model.structure_mean(F.one_hot(y,4).permute(0,3,1,2).to(image.dtype));mm=image-mx
                                finally:model.clear_draw()
                            q=(mx,mm,lx,lm,torch.zeros_like(mx))
                    q=tuple(v.detach().clone() for v in q);q0=tuple(v.clone() for v in q);restore_rng_state(p['rng']);first=evaluate(q);steps=[]
                    def checkpoint(step):
                        storage(s);save(out/f'S{step:04}.pt',dict(step=step,q=tuple(v.detach().cpu() for v in q),rng=capture_rng_state(),costs=costs.copy(),model_cursor=8675))
                        if name=='BG':
                            reference=torch.load(root/f'{block}_init{init}_GT'/f'S{step:04}.pt',map_location='cpu',weights_only=False)
                            assert same(reference['rng'],capture_rng_state());append(root/'RNG_PAIRS.jsonl',dict(branch=f'{block}_init{init}',step=step,exact=True))
                    checkpoint(0)
                    for step in range(1,201):
                        active.update(out=out,q=q,step=step-1);q=tuple(v.detach().requires_grad_() for v in q);rng=capture_rng_state();before,_=saa(q,'gradient',grad=True);after=capture_rng_state()
                        g=torch.autograd.grad(-before.sum(),q);assert all(torch.isfinite(v).all() for v in g);costs['reverse']+=1
                        append(root/'reverse_calls.jsonl',dict(index=costs['reverse'],branch=key,step=step))
                        scales=(1,1,256,256) if block=='A' else (N,)*5;chosen=None
                        for alpha in (1.,.25,.0625,.015625,.00390625,.0009765625,.000244140625,0.):
                            candidate=tuple((v-.05*alpha*scale*dv).detach() for v,dv,scale in zip(q,g,scales));restore_rng_state(rng);value,_=saa(candidate,'candidate')
                            finite=bool(torch.isfinite(value).all());ok=finite and bool((value>=before.detach()-1e-6).all()) and float(value.mean())>float(before.detach().mean())+1e-5
                            costs['candidate']+=1;assert costs['candidate']<=12800
                            append(root/'candidates.jsonl',dict(index=costs['candidate'],branch=key,step=step,alpha=alpha,finite=finite,accepted=ok))
                            if ok:chosen=(candidate,value,alpha);break
                        restore_rng_state(after)
                        if chosen:q,value,alpha=chosen;costs['accepted']+=1;append(root/'local_updates.jsonl',dict(index=costs['accepted'],branch=key,step=step))
                        else:q=tuple(v.detach() for v in q);value=before.detach();alpha=0.
                        row=dict(step=step,alpha=alpha,accepted=chosen is not None,before=before.detach().cpu().tolist(),after=value.cpu().tolist());steps.append(row);append(out/'steps.jsonl',row)
                        active.update(q=q,step=step)
                        if step%25==0:checkpoint(step);write(root/'STATUS.json',dict(status='RUNNING',branch=key,step=step,costs=costs))
                    last=evaluate(q);results[key]=dict(initial=first,final=last,accepted=sum(t['accepted'] for t in steps),last25_mean_same_step_gain=sum(sum(a-b for a,b in zip(t['after'],t['before']))/2 for t in steps[-25:])/25,field_RMS_displacement=[float((a-b).square().mean().sqrt()) for a,b in zip(q,q0)])
                    if block=='B':
                        rho=q[4]/(q[4].square()+(2*q[3]).exp()).sqrt()
                        results[key]['image_stats']=dict(correlation_quantiles=torch.quantile(rho.flatten(),torch.tensor([.05,.5,.95],device='cuda')).cpu().tolist(),initial_m_mean_RMS=float(q0[1].square().mean().sqrt()),final_m_mean_RMS=float(q[1].square().mean().sqrt()),x_plus_m_mean_displacement_RMS=float((q[0]+q[1]-q0[0]-q0[1]).square().mean().sqrt()))
        assert costs['reverse']==1600 and same(p['model'],model.state_dict()) and all(v.grad is None for v in model.parameters())
        assert (src/'physical_updates.jsonl').read_text()==ledger and (hist/'reverse_calls.jsonl').read_text()==previous
        assert len((root/'RNG_PAIRS.jsonl').read_text().splitlines())==36
        for filename,count in [('reverse_calls.jsonl','reverse'),('candidates.jsonl','candidate'),('local_updates.jsonl','accepted'),('density_calls.jsonl','density')]:
            entries=[json.loads(t) for t in (root/filename).read_text().splitlines()];assert [t['index'] for t in entries]==list(range(1,costs[count]+1))
        restore_rng_state(original_rng);assert same(original_rng,capture_rng_state())
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,checks=check,branches=results,model_rng_grad_source_history_unchanged=True,no_heldout_U=True,RNG_pairs=36,model_cursor=8675,space=storage(s)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:storage(s);save(active['out']/'failure_partial.pt',dict(step=active['step'],q=tuple(v.detach().cpu() for v in active['q']),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
