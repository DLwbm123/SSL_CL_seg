"""Convex source-centred occupancy drift with matched quadratic curvature."""
import torch
from torch.nn import functional as F
from ..apos_v2.math import area,lift,project,pixel

def mixture(prior,pi):
    pi=pi.double();mu=prior['mu'].double();f=prior['factor'].double();sigma=prior['sigma2'].double()
    mean=pi@mu
    within=(pi.sqrt()[:,:,None,None]*f[None]).permute(0,2,1,3).flatten(2)
    between=(pi.sqrt()[:,:,None]*(mu[None]-mean[:,None])).transpose(1,2)
    return torch.cat((within,between),-1),(pi@sigma).clamp_min(1e-6)
def penalty(r,d,kind):
    t=r.square()/d
    return .5*t if kind=='Q' else t/(torch.sqrt(1+t)+1)
def precision(r,d,kind):return torch.ones_like(r)/d if kind=='Q' else 1/(d*torch.sqrt(1+r.square()/d))
def latent(u,a0,f,D):
    rank=f.shape[-1];H=torch.eye(rank,device=u.device,dtype=u.dtype)[None]+f.transpose(-1,-2)@(D[...,None]*f)
    b=(f*(D*(u-a0))[...,None]).sum(-2)
    return torch.cholesky_solve(b[...,None],torch.linalg.cholesky(H))[...,0]
def energy(lp,lq,u,a0,z,f,d,w,kind):
    r=(u-a0).flatten(1)-torch.einsum('bnr,br->bn',f,z)
    M=u.shape[-1]**2
    return (lp.exp()*(lp-lq)).sum(1).mean()+(.5*(w*(area(lp.exp()[:,1:],u.shape[-1]).double()-u).square()).flatten(1).sum(1)/M+(penalty(r,d,kind).sum(1)+.5*z.square().sum(1))/M).mean()
def solve(lq,lq0,w,pi,prior,kind,calls=None):
    assert kind in ('Q','R');f,d=mixture(prior,pi);grid=int((d.shape[1]//3)**.5)
    a0=area(lq0.exp()[:,1:],grid).double();u=area(lq.exp()[:,1:],grid).double();w=w.double();z=torch.zeros(len(lq),f.shape[-1],device=lq.device,dtype=torch.double)
    lp=lq;maxres=0.;trace=[]
    with torch.no_grad():initial=float(energy(lp,lq,u,a0,z,f,d,w,kind))
    previous=initial
    for _ in range(8):
        lp,res=pixel(lq,u,w,calls);maxres=max(maxres,float(res.detach()))
        r=(u-a0).flatten(1)-torch.einsum('bnr,br->bn',f,z);D=precision(r,d,kind)
        if calls:calls.component(len(lq))
        z=latent(u.flatten(1),a0.flatten(1),f,D)
        centre_shape=a0+torch.einsum('bnr,br->bn',f,z).reshape_as(u);dd=D.reshape_as(u)
        h=w+dd;centre=(w*area(lp.exp()[:,1:],grid).double()+dd*centre_shape)/h;u=project(centre,h)
        with torch.no_grad():value=float(energy(lp,lq,u,a0,z,f,d,w,kind))
        assert value<=previous+1e-6*max(1,abs(previous)),('energy increase',kind,previous,value)
        trace.append(value);previous=value
    lp,res=pixel(lq,u,w,calls);maxres=max(maxres,float(res.detach()))
    with torch.no_grad():
        value=float(energy(lp,lq,u,a0,z,f,d,w,kind));assert value<=previous+1e-6*max(1,abs(previous))
        rr=(u-a0).flatten(1)-torch.einsum('bnr,br->bn',f,z);D=precision(rr,d,kind)
        znew=latent(u.flatten(1),a0.flatten(1),f,D)
        zr=float(((znew-z).abs()/(1+z.abs().amax(1,keepdim=True))).max())
        g=w*(u-area(lp.exp()[:,1:],grid).double())+(D*rr).reshape_as(u);h=w+D.reshape_as(u)
        ur=float((project(u-g/h,h)-u).abs().max());stationary=max(float(res),zr,ur)
        assert stationary<=1e-4,('full stationary residual',kind,stationary,zr,ur)
        diff=lp.exp()-lq.exp();flip=(lp.argmax(1)!=lq.argmax(1));raw=lq.argmax(1)
        stats=dict(initial_energy=initial,final_energy=value,max_pixel_residual=maxres,stationarity_residual=stationary,latent_residual=zr,occupancy_residual=ur,probability_change_max=float(diff.abs().max()),probability_change_mean=float(diff.abs().mean()),argmax_flip_fraction=float(flip.float().mean()),class_flip_counts=[int((flip&(raw==k)).sum()) for k in range(4)],W_mean=float(w.mean()),W_max=float(w.max()),standardized_residual_mean=float((rr.abs()/d.sqrt()).mean()),robust_to_quadratic_influence_ratio_mean=float((1/torch.sqrt(1+rr.square()/d)).mean()),energy_trace=trace+[value])
    return lp,stats

def check(device):
    torch.manual_seed(31);dtype=torch.double
    lq=F.log_softmax(torch.randn(2,4,8,8,device=device,dtype=dtype),1).detach().requires_grad_();lq0=lq.detach().clone()
    prior=dict(mu=torch.zeros(8,12,device=device,dtype=dtype),factor=torch.randn(8,12,8,device=device,dtype=dtype)*.005,sigma2=torch.full((8,12),.03,device=device,dtype=dtype))
    pi=torch.full((2,8),.125,device=device,dtype=dtype);w=torch.full((2,3,2,2),.5,device=device,dtype=dtype);notes=[]
    for kind in ('Q','R'):
        p,s=solve(lq,lq0,w,pi,prior,kind);assert float((p.exp()-lq0.exp()).abs().max())<=5e-7 and torch.equal(p.argmax(1),lq0.argmax(1));p.square().mean().backward(retain_graph=True);assert torch.isfinite(lq.grad).all();notes.append(s)
        shifted=F.log_softmax(lq+torch.linspace(-.1,.1,4,device=device,dtype=dtype)[None,:,None,None],1);p,s=solve(shifted,lq0,w,pi,prior,kind);p.square().mean().backward(retain_graph=True);notes.append(s)
    stress=dict(mu=prior['mu'],factor=torch.zeros_like(prior['factor']),sigma2=torch.full_like(prior['sigma2'],1e-6))
    for kind in ('Q','R'):
        p,ss=solve(shifted,lq0,w,pi,stress,kind);p.square().mean().backward(retain_graph=True);notes.append(ss)
    r=torch.tensor([-.5,-.01,0.,.01,.5],device=device,dtype=dtype,requires_grad=True);d=torch.full_like(r,.1);v=penalty(r,d,'R');g=torch.autograd.grad(v.sum(),r)[0];eps=1e-6;fd=(penalty(r.detach()+eps,d,'R')-penalty(r.detach()-eps,d,'R'))/(2*eps);assert float((fd-g).abs().max())<1e-8
    assert bool((g.abs()<=r.detach().abs()/d+1e-10).all())
    return dict(status='PASS',device=device,synthetic_reverse=7,finite_difference_max_error=float((fd-g).abs().max()),checks=notes)
