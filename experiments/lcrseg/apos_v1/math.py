"""Low-rank conditional Gaussian and finite Student-t variational posterior."""
import math
import torch
from torch.nn import functional as F
LOG2PI=math.log(2*math.pi)


def lowrank_nll(value,mean,diagonal,factor):
    # batch prefixes arbitrary; last axes N,R. Cov = diag(diagonal)+factor factor^T.
    delta=value-mean; inv=diagonal.reciprocal()
    rank=factor.shape[-1]
    H=torch.eye(rank,device=value.device,dtype=value.dtype)+factor.transpose(-1,-2)@(factor*inv[...,None])
    L=torch.linalg.cholesky(H)
    b=(factor*(delta*inv)[...,None]).sum(-2)
    sol=torch.cholesky_solve(b[...,None],L)[...,0]
    quad=(delta.square()*inv).sum(-1)-(b*sol).sum(-1)
    logdet=diagonal.log().sum(-1)+2*L.diagonal(dim1=-2,dim2=-1).log().sum(-1)
    return .5*(quad+logdet+value.shape[-1]*LOG2PI)


def conditional(t,w,mu,factor,tau2,omega):
    D=tau2/omega;s=D+w.reciprocal();v=t-mu
    rank=factor.shape[-1]
    H=torch.eye(rank,device=t.device,dtype=t.dtype)+factor.transpose(-1,-2)@(factor/s[...,None])
    chol=torch.linalg.cholesky(H)
    z=torch.cholesky_solve((factor*(v/s)[...,None]).sum(-2)[...,None],chol)[...,0]
    Fz=(factor*z[...,None,:]).sum(-1);a=D/s
    r=a*(v-Fz);m=mu+Fz+r
    # Q = F L^-T, so Q Q^T = F H^-1 F^T.
    Q=torch.linalg.solve_triangular(chol,factor.transpose(-1,-2),upper=False).transpose(-1,-2)
    fv=Q.square().sum(-1);diag=D*(1-a)
    Cfactor=(1-a)[...,None]*Q
    rsecond=r.square()+diag+a.square()*fv
    evidence=-lowrank_nll(t,mu,s,factor)
    return m,diag,Cfactor,rsecond,evidence


def posterior(t,w,logpi,mu,factor,tau2,robust):
    t=t.double()[:,None];w=w.double()[:,None]
    mu=mu.double()[None];factor=factor.double()[None];tau2=tau2.double()[None]
    omega=torch.ones_like(t-mu);qa=qb=None
    if robust:
        for _ in range(3):
            _,_,_,rsecond,_=conditional(t,w,mu,factor,tau2,omega)
            qa=torch.full_like(omega,2.5);qb=(4+rsecond/tau2)/2;omega=qa/qb
    m,diagonal,Cfactor,_,evidence=conditional(t,w,mu,factor,tau2,omega)
    if robust:
        elog=torch.digamma(qa)-qb.log()
        p=torch.distributions.Gamma(torch.full_like(qa,2.),torch.full_like(qb,2.))
        q=torch.distributions.Gamma(qa,qb)
        evidence=evidence+.5*(elog-omega.log()).sum(-1)-torch.distributions.kl_divergence(q,p).sum(-1)
    logweights=F.log_softmax(logpi.double()+evidence,dim=-1)
    return m,diagonal,Cfactor,logweights


def check(device):
    g=torch.Generator(device=device).manual_seed(713)
    dtype=torch.float64
    t=torch.randn(2,1,7,generator=g,device=device,dtype=dtype,requires_grad=True)
    w=torch.rand(2,1,7,generator=g,device=device,dtype=dtype)+1
    mu=torch.randn(1,3,7,generator=g,device=device,dtype=dtype)
    f=torch.randn(1,3,7,2,generator=g,device=device,dtype=dtype)*.2
    tau=torch.rand(1,3,7,generator=g,device=device,dtype=dtype)+.1
    omega=torch.ones(2,3,7,device=device,dtype=dtype)
    m,d,c,r,e=conditional(t,w,mu,f,tau,omega)
    K=f@f.transpose(-1,-2)+torch.diag_embed(tau)
    C=torch.linalg.inv(torch.linalg.inv(K)+torch.diag_embed(w.expand(2,3,7)))
    mm=(C@(torch.linalg.solve(K,mu[...,None])+w[...,None]*t[...,None]))[...,0]
    assert torch.allclose(m,mm,atol=1e-10,rtol=1e-9)
    assert torch.allclose(C,torch.diag_embed(d)+c@c.transpose(-1,-2),atol=1e-10,rtol=1e-9)
    val=torch.randn(2,3,7,generator=g,device=device,dtype=dtype)
    n=lowrank_nll(val,m,d,c)
    ref=-torch.distributions.MultivariateNormal(m,covariance_matrix=C).log_prob(val)
    assert torch.allclose(n,ref,atol=1e-9,rtol=1e-9)
    n.sum().backward();assert torch.isfinite(t.grad).all() and t.grad.norm()>0
    tm=t.detach()[:,0].requires_grad_();ww=w[:,0]
    rr=posterior(tm,ww,torch.zeros(2,3,device=device),mu[0],f[0],tau[0],True)
    assert all(torch.isfinite(a).all() for a in rr)
    assert torch.allclose(rr[-1].exp().sum(-1),torch.ones(2,device=device,dtype=dtype))
    (rr[0].square().mean()+rr[1].mean()).backward();assert torch.isfinite(tm.grad).all()
    # r second moment includes latent covariance even at posterior mean zero.
    V=K+torch.diag_embed(w.reciprocal().expand(2,3,7))
    rmean=(torch.diag_embed(tau)@torch.linalg.solve(V,(t-mu)[...,None]))[...,0]
    rcov=torch.diag_embed(tau)-torch.diag_embed(tau)@torch.linalg.solve(V,torch.diag_embed(tau))
    assert torch.allclose(r,rmean.square()+rcov.diagonal(dim1=-2,dim2=-1),atol=1e-10,rtol=1e-9)
    gaussian=posterior(tm.detach(),ww,torch.zeros(2,3,device=device),mu[0],f[0],tau[0],False)
    assert (rr[0]-gaussian[0]).norm()>1e-6
    return dict(status='PASS',device=device,mean_max_error=float((m-mm).abs().max()),covariance_max_error=float((C-torch.diag_embed(d)-c@c.transpose(-1,-2)).abs().max()),nll_max_error=float((n-ref).abs().max()),synthetic_reverse=2)
