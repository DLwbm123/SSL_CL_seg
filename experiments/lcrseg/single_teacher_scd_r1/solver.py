"""R1's single analytic-bracket repair; no network forward or fallback."""
import math
import torch
from torch.nn import functional as F

BLOCK=65536
ITERATIONS=64
TOL=1e-9

def boundary_target(q,minimum_support):
    r=torch.where(minimum_support,q,0.)
    return r/r.sum(-1,keepdim=True)

def shifted_constraint(p,a):
    """Algebraic a.p-min(a), compensated around the largest probability."""
    m=a.min(-1).values
    d=a-m[:,None]
    k=p.argmax(-1)
    other=p.clone();other.scatter_(1,k[:,None],0.)
    sum_error=(p.gather(1,k[:,None]).squeeze(1)-1.)+other.sum(-1)
    v=(d*p).sum(-1)+m*sum_error
    return d,v

@torch.no_grad()
def project(p,q,y,reliable):
    if p.dtype!=torch.float64 or q.dtype!=torch.float64 or p.shape!=q.shape or p.ndim!=2:
        raise ValueError("FP64 equal-shape class vectors required")
    if y.shape!=p.shape[:1] or reliable.shape!=y.shape:
        raise ValueError("reference shape")
    if not (torch.isfinite(p).all() and torch.isfinite(q).all() and (p>=0).all() and (q>0).all()):
        raise FloatingPointError("invalid probability support")
    if not (torch.allclose(p.sum(-1),torch.ones_like(p[:,0]),atol=1e-12,rtol=0) and
            torch.allclose(q.sum(-1),torch.ones_like(q[:,0]),atol=1e-12,rtol=0)):
        raise FloatingPointError("nonunit probabilities")
    a=p-F.one_hot(y,p.shape[-1]).double()
    conflict=reliable & (((p-q)*a).sum(-1)<0)
    r=q.clone();logeta=torch.full_like(p[:,0],-torch.inf);residual=torch.zeros_like(p[:,0])
    boundary=torch.zeros_like(reliable);analytic=torch.zeros_like(reliable)
    certificate_gap=torch.zeros_like(p[:,0]);v_difference=torch.zeros_like(p[:,0])
    ix=conflict.nonzero().flatten()
    for start in range(0,ix.numel(),BLOCK):
        j=ix[start:start+BLOCK]
        scale=a[j].abs().amax(-1)
        if (scale==0).any():raise FloatingPointError("constant direction classified conflicting")
        aa=a[j]/scale[:,None];d,v=shifted_constraint(p[j],aa)
        old_v=(aa*p[j]).sum(-1)-aa.min(-1).values
        v_difference[j]=(v-old_v).abs()
        if (v<0).any() or not torch.isfinite(v).all():
            raise FloatingPointError("negative/nonfinite shifted threshold; no clipping permitted")
        logq=q[j].log();zero=d==0;logd=d.log()
        logq0=torch.logsumexp(logq.masked_fill(~zero,-torch.inf),-1)
        edge=v==0
        if edge.any():
            rr=boundary_target(q[j][edge],zero[edge])
            r[j[edge]]=rr;logeta[j[edge]]=torch.inf;boundary[j[edge]]=True
        use=~edge
        if not use.any():continue
        k=j[use];dd=d[use];ld=logd[use];lq=logq[use];vv=v[use];lv=vv.log()
        initial=torch.logsumexp(lq+ld,-1)-torch.logsumexp(lq,-1)
        active=initial>lv
        if not active.any():continue
        k=k[active];dd=dd[active];ld=ld[active];lq=lq[active];vv=vv[active];lv=lv[active]
        delta=dd.masked_fill(dd==0,torch.inf).min(-1).values
        numerator=torch.logsumexp(lq+ld,-1)-logq0[use][active]-lv+math.log(2.)
        if (numerator<=0).any():raise FloatingPointError("invalid analytic numerator")
        log_hi=numerator.log()-delta.log()
        hi=torch.logaddexp(torch.zeros_like(log_hi),log_hi)
        lo=torch.zeros_like(hi)
        if not torch.isfinite(hi).all():raise FloatingPointError("analytic interval outside finite log coordinate")
        def at(t):
            # t=log1p(eta_scaled); positive infinite penalties mean exact FP64 underflow,
            # while every argmin component retains finite logq, so normalization exists.
            le=torch.where(t>40.,t,torch.expm1(t).log())
            penalty=(le[:,None]+ld).exp()
            lw=lq-penalty
            norm=torch.logsumexp(lw,-1)
            moment=torch.logsumexp(lw+ld,-1)-norm
            return lw-norm[:,None],moment,le
        if (at(hi)[1]>lv).any():raise FloatingPointError("analytic feasible-end certificate failed")
        for _ in range(ITERATIONS):
            mid=lo+(hi-lo)/2
            bad=at(mid)[1]>lv
            lo=torch.where(bad,mid,lo);hi=torch.where(bad,hi,mid)
        lr,lmoment,le=at(hi);rr=lr.exp()
        error=((rr*dd).sum(-1)-vv).abs()
        if (lmoment>lv).any() or (error>TOL).any() or not torch.isfinite(rr).all():
            raise FloatingPointError("projector certificate/residual failed")
        if not torch.allclose(rr.sum(-1),torch.ones_like(vv),atol=1e-12,rtol=0):
            raise FloatingPointError("target lost simplex support")
        r[k]=rr;logeta[k]=le-scale[use][active].log();residual[k]=error
        certificate_gap[k]=lmoment-lv;analytic[k]=True
    return r,conflict,dict(log_eta=logeta,residual=residual,boundary=boundary,analytic=analytic,
        certificate_log_moment_gap=certificate_gap,compensated_v_difference=v_difference,iterations=ITERATIONS)
