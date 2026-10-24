"""Nonnegative generalized energy, exact conditional blocks and fixed pixel solve."""
import torch
from torch.nn import functional as F

def area(x,g):return F.avg_pool2d(x,x.shape[-1]//g)
def lift(x,size):return F.interpolate(x,size=size,mode='nearest')

def project(c,h):
    # Weighted projection onto three positive occupancies whose sum is <=1.
    positive=c.clamp_min(0);need=positive.sum(1)>1
    out=positive;done=~need
    for bits in range(1,8):
        mask=torch.tensor([(bits>>i)&1 for i in range(3)],device=c.device,dtype=torch.bool)[None,:,None,None]
        eta=((c*mask).sum(1)-1)/((1/h)*mask).sum(1)
        raw=c-eta[:,None]/h
        valid=(eta>=-1e-12)&torch.where(mask,raw>=-1e-11,raw<=1e-11).all(1)&~done
        out=torch.where(valid[:,None],raw.clamp_min(0),out);done=done|valid
    assert bool(done.all()),'weighted simplex active set'
    assert bool((out>=0).all()) and float(out.sum(1).max())<=1+1e-9
    return out

def pixel(logq,u,w,calls=None):
    g=u.shape[-1];v=torch.zeros_like(u)
    if calls:calls.pixel(len(logq)*16)
    for _ in range(16):
        corr=torch.cat([torch.zeros_like(v[:,:1]),v],1)
        lp=F.log_softmax(logq-lift(corr,logq.shape[-2:]).to(logq),1)
        v=w*(area(lp.exp()[:,1:],g).double()-u)
    corr=torch.cat([torch.zeros_like(v[:,:1]),v],1)
    lp=F.log_softmax(logq-lift(corr,logq.shape[-2:]).to(logq),1)
    residual=(v-w*(area(lp.exp()[:,1:],g).double()-u)).abs().max()
    assert float(residual.detach())<=1e-4,'pixel fixed point residual'
    return lp,residual

def component(u,z,mu,factor,sigma,kind):
    r=u.flatten(1)[:,None]-mu[None]-torch.einsum('knr,bkr->bkn',factor,z)
    penalty=.5*r.square()/sigma[None] if kind=='Q' else 2.5*torch.log1p(r.square()/(2*sigma[None]))
    M=u.shape[-1]**2
    e=(penalty.sum(-1)+.5*z.square().sum(-1))/M
    d=1/sigma[None] if kind=='Q' else 5/(2*sigma[None]+r.square())
    return e,d

def energy(lp,logq,u,z,rho,w,pi,mu,factor,sigma,kind):
    p=lp.exp();kl=(p*(lp-logq)).sum(1).mean()
    coupling=.5*(w*(area(p[:,1:],u.shape[-1]).double()-u).square()).sum((1,2,3))/u.shape[-1]**2
    e,_=component(u,z,mu,factor,sigma,kind)
    categorical=(rho*(rho.clamp_min(1e-300).log()-pi.log())).sum(-1)
    return kl+((rho*e).sum(-1)+coupling+categorical).mean()

def solve(logq,w,pi,prior,kind,calls=None):
    assert kind in ('Q','R');mu=prior['mu'].double();factor=prior['factor'].double();sigma=prior['sigma2'].double()
    grid=int((mu.shape[1]//3)**.5);u=area(logq.exp()[:,1:],grid).double()
    w=w.double();pi=pi.double();rho=pi;z=torch.zeros(len(logq),8,8,device=logq.device,dtype=torch.float64)
    lp=logq;energies=[];maxres=0.
    with torch.no_grad():initial=float(energy(lp,logq,u,z,rho,w,pi,mu,factor,sigma,kind))
    previous=initial
    for _ in range(8):
        lp,res=pixel(logq,u,w,calls);maxres=max(maxres,float(res.detach()))
        _,d=component(u,z,mu,factor,sigma,kind)
        if calls:calls.component(len(logq)*8)
        H=torch.eye(8,device=logq.device,dtype=torch.float64)+factor[None].transpose(-1,-2)@(d[...,None]*factor[None])
        L=torch.linalg.cholesky(H)
        b=(factor[None]*(d*(u.flatten(1)[:,None]-mu[None]))[...,None]).sum(-2)
        z=torch.cholesky_solve(b[...,None],L)[...,0]
        means=mu[None]+torch.einsum('knr,bkr->bkn',factor,z)
        precision=(rho[...,None]*d).sum(1).reshape(-1,3,grid,grid)
        target=(rho[...,None]*d*means).sum(1).reshape_as(u)
        h=w+precision;centre=(w*area(lp.exp()[:,1:],grid).double()+target)/h
        u=project(centre,h)
        e,_=component(u,z,mu,factor,sigma,kind)
        rho=F.softmax(pi.log()-e,-1)
        with torch.no_grad():value=float(energy(lp,logq,u,z,rho,w,pi,mu,factor,sigma,kind))
        assert value<=previous+1e-6*max(1,abs(previous)),('energy increase',kind,previous,value)
        energies.append(value);previous=value
    lp,res=pixel(logq,u,w,calls);maxres=max(maxres,float(res.detach()))
    with torch.no_grad():value=float(energy(lp,logq,u,z,rho,w,pi,mu,factor,sigma,kind))
    assert value<=previous+1e-6*max(1,abs(previous))
    return lp,dict(initial_energy=initial,final_energy=value,energy_trace=energies+[value],max_pixel_residual=maxres)

def check(device):
    torch.manual_seed(19);dtype=torch.float64
    c=torch.randn(3,3,2,2,device=device,dtype=dtype,requires_grad=True);h=torch.rand_like(c)+.1
    u=project(c,h);assert float(u.sum(1).max())<=1+1e-9
    # KKT: positive coordinates share multiplier; inactive gradients satisfy dual inequality.
    eta=torch.where(u>1e-10,h*(c-u),torch.zeros_like(u)).amax(1)
    assert bool(torch.where(u>1e-10,(h*(c-u)-eta[:,None]).abs()<1e-8,h*c<=eta[:,None]+1e-8).all())
    q=F.log_softmax(torch.randn(2,4,8,8,device=device,dtype=dtype),1).detach().requires_grad_()
    prior=dict(mu=torch.full((8,12),.1,device=device,dtype=dtype),factor=torch.randn(8,12,8,device=device,dtype=dtype)*.005,sigma2=torch.full((8,12),.03,device=device,dtype=dtype))
    notes=[]
    for kind in ('Q','R'):
        p,s=solve(q,torch.full((2,3,2,2),.5,device=device,dtype=dtype),torch.full((2,8),.125,device=device,dtype=dtype),prior,kind)
        assert torch.allclose(p.exp().sum(1),torch.ones_like(p[:,0]),atol=1e-8)
        (p.square().mean()).backward(retain_graph=True);assert torch.isfinite(q.grad).all();notes.append(s)
    stress=dict(mu=prior['mu'],factor=torch.zeros_like(prior['factor']),sigma2=torch.full_like(prior['sigma2'],1e-6))
    for kind in ('Q','R'):
        lp,ss=solve(q,torch.ones(2,3,2,2,device=device,dtype=dtype),torch.full((2,8),.125,device=device,dtype=dtype),stress,kind)
        lp.square().mean().backward(retain_graph=True);assert torch.isfinite(q.grad).all();notes.append(ss)
    v=torch.arange(64,dtype=dtype,device=device).reshape(1,1,8,8);a=area(v,2)
    assert torch.equal(lift(a,(8,8))[:,:,0:4,0:4],a[:,:,0:1,0:1].expand(1,1,4,4))
    return dict(status='PASS',device=device,checks=notes,synthetic_reverse=4)
