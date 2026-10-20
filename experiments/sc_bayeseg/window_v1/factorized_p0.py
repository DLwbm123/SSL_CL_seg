"""Frozen image-factor diagnosis; Gaussian reference is a different model."""
import json, math, os, traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset, collate_labeled
from lcrseg.engine.trainer import seed_everything
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from care_seg_study_v1.run import write, append
from sc_bayeseg.run import same
from sc_bayeseg.model import sar, student_log_prob, normal_log_prob
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.common import storage


def covariance(size, device):
    k=torch.arange(size, dtype=torch.float64, device=device)
    a=1-.9*(torch.cos(2*math.pi*k/size)[:,None]+torch.cos(2*math.pi*k/size)[None,:])/2
    return .04/a.square()+.26


def gaussian_score(image, mean):
    residual=(image-mean).double(); n=image.shape[-1]**2
    c=covariance(image.shape[-1], image.device)
    power=torch.fft.fft2(residual, norm='ortho').abs().square()
    quad=(power/c).flatten(1).sum(1)
    logp=-.5*(quad+c.log().sum()+n*math.log(2*math.pi))
    return logp, quad


def numeric_check():
    # Independent dense reference validates FFT normalization and SAR eigenvalues.
    size=8; n=size**2
    identity=torch.eye(n,dtype=torch.float64)
    a=sar(identity.reshape(n,1,size,size)).reshape(n,n).T
    inv=torch.linalg.inv(a); c=.04*inv@inv.T+.26*identity
    r=torch.linspace(-1,1,n,dtype=torch.float64).reshape(1,1,size,size)
    fast,quad=gaussian_score(r,torch.zeros_like(r))
    densequad=r.flatten()@torch.linalg.solve(c,r.flatten())
    dense=-.5*(densequad+torch.linalg.slogdet(c).logabsdet+n*math.log(2*math.pi))
    assert abs(float(fast)-float(dense))<1e-10
    assert abs(float(quad)-float(densequad))<1e-10
    fft_sar=torch.fft.ifft2(torch.fft.fft2(r)*(1-.9*(torch.cos(2*math.pi*torch.arange(size,dtype=torch.float64)/size)[:,None]+torch.cos(2*math.pi*torch.arange(size,dtype=torch.float64)/size)[None,:])/2)).real
    assert torch.allclose(sar(r),fft_sar,atol=1e-12,rtol=0)
    return dict(dense_fft_loglik_error=abs(float(fast-dense)),dense_fft_quad_error=abs(float(quad-densequad)))


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False)
    costs=dict(structure_forward=0,image_bound=0,theta_draw=0,reverse=0,optimizer=0,local_updates=0)
    try:
        space=storage(s); checks=numeric_check()
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        source=Path(s['spatial_generator']);history=Path(s['spatial_hypothesis'])/'reverse_calls.jsonl'
        ledger=(source/'physical_updates.jsonl').read_text();old_history=history.read_text()
        assert len(ledger.splitlines())==604 and len(old_history.splitlines())==1000
        payload=torch.load(source/'S0600.pt',map_location='cpu',weights_only=False);assert payload['cursor']==8675
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text())
        assert len(sel['train_indices'])==2
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False)
        gt=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        labels=dict(GT=gt,BG=torch.zeros_like(gt),shift=torch.roll(gt,(32,32),(-2,-1)))
        for cls in (1,2,3):labels[f'delete{cls}']=torch.where(gt==cls,0,gt)
        labels['other']=gt.flip(0)
        assert torch.equal(torch.stack([(gt==k).sum((-2,-1)) for k in range(4)]),torch.stack([(labels['shift']==k).sum((-2,-1)) for k in range(4)]))
        model=SpatialResidualModel().cuda();model.load_state_dict(payload['model']);model.requires_grad_(False)
        script_rng=capture_rng_state();seed_everything(50163)
        rows=[]
        with torch.no_grad():
            for draw in range(33):
                if draw==0:
                    for module in (model.anatomy,model.structure):
                        for layer in module:layer.sampled=(layer.mean_w,layer.mean_b)
                else:model.draw();costs['theta_draw']+=1
                try:
                    means={}
                    for name,z in labels.items():
                        onehot=F.one_hot(z,4).permute(0,3,1,2).to(image.dtype)
                        means[name]=model.structure_mean(onehot);costs['structure_forward']+=1
                        append(root/'forward_calls.jsonl',dict(index=costs['structure_forward'],draw=draw,candidate=name))
                    values={name:[] for name in labels};parts={name:[] for name in labels}
                    for _ in range(4):
                        x,lqx,_,_=model.gaussian(model.physical_encoder(image),True)
                        m,lqm,_,_=model.gaussian(model.appearance_encoder(image),True)
                        common=dict(image=student_log_prob(image-x-m,.1),appearance=normal_log_prob(m,torch.zeros_like(m),torch.full_like(m,math.log(.5))),negative_logq=-lqx-lqm)
                        for name in labels:
                            shape=student_log_prob(sar(x-means[name]),.2)+model.logdet
                            terms={**common,'shape':shape};b=sum(terms.values())/128**2
                            assert torch.isfinite(b).all();values[name].append(b.double())
                            parts[name].append({k:(v/128**2).cpu().tolist() for k,v in terms.items()})
                            costs['image_bound']+=1;append(root/'bound_calls.jsonl',dict(index=costs['image_bound'],draw=draw,candidate=name))
                    result={}
                    for name in labels:
                        lp,quad=gaussian_score(image,means[name]);sep,sepquad=gaussian_score(means[name],means['GT'])
                        result[name]=dict(gaussian_logp=lp.cpu().tolist(),gaussian_per_pixel=(lp/128**2).cpu().tolist(),residual_quad_per_pixel=(quad/128**2).cpu().tolist(),F_minus_GT_RMS=(means[name]-means['GT']).square().flatten(1).mean(1).sqrt().cpu().tolist(),F_minus_GT_mahalanobis_per_pixel=(sepquad/128**2).cpu().tolist(),student_fixed_proposal_B=torch.stack(values[name]).mean(0).cpu().tolist(),student_parts=parts[name])
                        assert torch.isfinite(lp).all()
                    row=dict(draw=draw,weight_mode='mean' if draw==0 else 'posterior',candidates=result)
                    rows.append(row);append(root/'scores.jsonl',row)
                finally:model.clear_draw()
                write(root/'STATUS.json',dict(status='RUNNING',draw=draw,costs=costs))
        restore_rng_state(script_rng)
        assert same(payload['model'],model.state_dict()) and all(v.grad is None for v in model.parameters())
        assert same(script_rng,capture_rng_state())
        assert (source/'physical_updates.jsonl').read_text()==ledger and history.read_text()==old_history
        assert costs==dict(structure_forward=231,image_bound=924,theta_draw=32,reverse=0,optimizer=0,local_updates=0)
        for filename,key in [('forward_calls.jsonl','structure_forward'),('bound_calls.jsonl','image_bound')]:
            entries=[json.loads(v) for v in (root/filename).read_text().splitlines()]
            assert [v['index'] for v in entries]==list(range(1,costs[key]+1))
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,checks=checks,model_rng_grad_history_unchanged=True,no_heldout_U=True,source_cursor=8675,space=storage(s),rows=rows))
        write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
