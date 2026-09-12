"""Adam's actual increment corrected using current-response VJPs, before native EMA."""
import math
from unittest.mock import patch
import torch
from torch.nn import functional as F
from experiments.lcrseg.ams_seq_transfer_v0_1 import core as parent
from experiments.lcrseg.ams_seq_transfer_v0_1.core import *
from experiments.lcrseg.lctx_weight_memory_v0_1.core import LAYERS,configure,hash_state

ARMS=('DPR_U','RESPONSE_U','DPR_L')
MAIN='DPR_U'

def parameters(model):
    named=dict(model.named_parameters())
    if tuple(n for n,p in named.items() if p.requires_grad)!=LAYERS:raise ValueError('F_CONV whitelist')
    return [named[n] for n in LAYERS]

def vector(ps,gradient=False):return torch.cat([(p.grad if gradient else p).detach().reshape(-1) for p in ps])

@torch.no_grad()
def assign(ps,v):
    offset=0
    for p in ps:
        p.copy_(v[offset:offset+p.numel()].view_as(p));offset+=p.numel()
    assert offset==v.numel()

def fixed_state(model):return {n:v for n,v in model.state_dict().items() if n not in LAYERS}

def strength(epoch):return min(1.,max(0.,(epoch-20)/20))

def response_input(batch):
    # Capability boundary: query functions reject labels rather than silently ignoring them.
    if set(batch)!={'image','geometry'}:raise PermissionError('response input must contain image and geometry only')
    validate_batch(batch,False)
    return batch

class DeterministicAdaptivePool(torch.autograd.Function):
    """Exact adaptive bins, native forward; deterministic disjoint-bin or separable VJP."""
    @staticmethod
    def forward(ctx,x):
        ctx.shape=x.shape[-2:]
        return F.adaptive_avg_pool2d(x,(32,32))
    @staticmethod
    def backward(ctx,grad):
        h,w=ctx.shape
        if h%32==0 and w%32==0:
            # Formal384: native non-overlapping12x12 averages, no atomic scatter.
            return (grad/((h//32)*(w//32))).repeat_interleave(h//32,-2).repeat_interleave(w//32,-1)
        def weights(n):
            i=torch.arange(32,device=grad.device);start=(i*n)//32;end=((i+1)*n+31)//32
            pos=torch.arange(n,device=grad.device)
            return ((pos[None]>=start[:,None])&(pos[None]<end[:,None])).to(grad)/(end-start)[:,None]
        return torch.einsum('ih,bcij,jw->bchw',weights(h),grad,weights(w))

def contrast(logits,valid):
    if valid.dtype!=torch.bool or valid.shape!=logits.shape[:1]+logits.shape[2:]:raise ValueError('geometry mask')
    frac=F.adaptive_avg_pool2d(valid[:,None].to(logits),(32,32))
    pooled=DeterministicAdaptivePool.apply(logits*valid[:,None])/frac.clamp_min(1e-12)
    C=logits.new_tensor([[-1/math.sqrt(2),1/math.sqrt(2),0],[-1/math.sqrt(6),-1/math.sqrt(6),2/math.sqrt(6)]])
    return torch.einsum('kc,bchw->bkhw',C,pooled),frac>0

def response(model,batch,key,counts,stream):
    response_input(batch)
    z,_=fwd(model,batch['image'],False,key,counts,stream)
    return contrast(z,batch['geometry'])

def jacobian(model,batch,key,counts):
    ps=parameters(model);before=vector(ps,True).clone()
    chi,valid=response(model,batch,key,counts,'response_query');den=(2*valid.sum()).double().sqrt()
    rows=[]
    for j in range(2):
        gen=torch.Generator(device=chi.device).manual_seed(stable_seed('DPR_V0_1',*key,j))
        signs=torch.randint(0,2,chi.shape,device=chi.device,generator=gen)*2-1
        phi=(signs*chi*valid).double().sum()/den.clamp_min(1)
        grads=torch.autograd.grad(phi,ps,create_graph=False,retain_graph=j==0)
        counts['response_VJP']+=1
        rows.append(torch.cat([g.detach().reshape(-1) for g in grads]))
    if not torch.equal(vector(ps,True),before):raise RuntimeError('response VJP contaminated supervised .grad')
    R=torch.stack(rows).double();norm=R.norm()
    if not torch.isfinite(R).all():raise FloatingPointError('nonfinite response VJP')
    return R/norm if norm>1e-30 else torch.zeros_like(R),float(norm)

@torch.no_grad()
def correct(d0,g,J,lam,preserve=True):
    d0,g,J=d0.double(),g.double(),J.double()
    if d0.ndim!=1 or g.shape!=d0.shape or J.ndim!=2 or J.shape[1]!=len(g):raise ValueError('correction shape')
    if not 0<=lam<=1 or not all(bool(torch.isfinite(x).all()) for x in (d0,g,J)):raise FloatingPointError('invalid correction inputs')
    if lam==0 or g.norm()<=1e-30 or J.norm()<=1e-30:return d0.clone(),1.
    A=J
    if preserve:
        e=g/g.norm();A=J-(J@e)[:,None]*e
    S=torch.eye(J.shape[0],dtype=torch.float64,device=J.device)+lam*(A@A.T)
    return d0-lam*A.T@torch.linalg.solve(S,J@d0),float(torch.linalg.cond(S))

@torch.no_grad()
def roundoff_budget(target64):
    # Independent of the observed residual: largest adjacent FP32 spacing / 2,
    # plus the FP64 addition rounding bound. No small-update relative tolerance.
    q=target64.float();up=torch.nextafter(q,torch.full_like(q,float('inf'))).double()
    down=torch.nextafter(q,torch.full_like(q,-float('inf'))).double()
    return .5*torch.maximum(up-q.double(),q.double()-down)+torch.finfo(torch.float64).eps*target64.abs()+2**-149

@torch.no_grad()
def numerical_row(theta,raw,g,J,d,lam,preserve,condition,Rnorm,actual):
    theta,raw,g,actual=theta.double(),raw.double(),g.double(),actual.double()
    d0=raw-theta;applied=actual-theta;eps=roundoff_budget(theta+d)
    # FP64 reductions and solve have a separate pre-frozen algebra allowance.
    algebra=1e-12*max(float(g.norm()*d0.norm()),1e-30)
    objective0=.5*lam*(J@d0).square().sum()
    objective=.5*(d-d0).square().sum()+.5*lam*(J@d).square().sum()
    progress=float(abs(g@(d-d0)));applied_residual=float(abs(g@(applied-d0)))
    budget=float(g.abs()@eps)+algebra
    jr0=float((J@d0).norm());jrd=float((J@d).norm());jra=float((J@applied).norm())
    allowance=1e-12*max(float(d0.square().sum()),1e-30)
    checks=dict(finite=all(bool(torch.isfinite(v).all()) for v in (actual,d)),
                objective_nonincrease=float(objective-objective0)<=allowance,
                algebra_progress=(not preserve or progress<=algebra),
                applied_progress=(not preserve or applied_residual<=budget),
                cast_within_budget=bool(((actual-(theta+d)).abs()<=eps).all()),
                sampled_response_nonincrease=jrd<=jr0+1e-12*max(float(d0.norm()),1e-30),
                correction_bound=float((d-d0).norm())<=math.sqrt(lam)*jr0+1e-12*max(float(d0.norm()),1e-30))
    return dict(checks=checks,lambda_response=lam,g_norm=float(g.norm()),d0_norm=float(d0.norm()),
        correction_ratio=float((d-d0).norm()/d0.norm().clamp_min(1e-30)),g_dot_d0=float(g@d0),g_dot_dstar=float(g@d),g_dot_applied=float(g@applied),
        Jd0_norm=jr0,Jdstar_norm=jrd,Japplied_norm=jra,condition_number=condition,raw_response_norm=Rnorm,
        algebra_progress_residual=progress,algebra_progress_budget=algebra,applied_progress_residual=applied_residual,applied_progress_budget=budget,
        applied_rounding_norm=float((applied-d).norm()),rounding_vector_budget_norm=float(eps.norm()),objective_difference=float(objective-objective0),
        zero_probe=int(Rnorm<=1e-30),zero_task_gradient=int(g.norm()<=1e-30),raw_non_descent=int(g@d0>=0))

def step(model,ema,opt,l,probe,task,epoch,index,counts,patients,diagnostic=False):
    if task['arm'] not in ARMS:raise PermissionError('unregistered arm')
    lam=strength(epoch);ps=parameters(model);original=opt.step;detail={};transient={}
    if lam==0 and probe is not None:raise PermissionError('warmup response access')
    def proposal(*args,**kwargs):
        theta=vector(ps).clone();g=vector(ps,True).double()
        if lam:
            J,Rnorm=jacobian(model,probe,(task['seed'],task['domain'],epoch,index),counts)
        else:J=torch.zeros(2,g.numel(),dtype=torch.float64,device=g.device);Rnorm=0.
        original(*args,**kwargs);raw=vector(ps).clone();d0=raw.double()-theta.double()
        d,condition=correct(d0,g,J,lam,task['arm']!='RESPONSE_U')
        # A genuine identity path: no response query and no parameter copy for warmup/degeneracies.
        if lam and g.norm()>1e-30 and Rnorm>1e-30:assign(ps,(theta.double()+d).float())
        detail.update(numerical_row(theta,raw,g,J,d,lam,task['arm']!='RESPONSE_U',condition,Rnorm,vector(ps)))
        if not lam:detail.update(zero_probe=0,zero_task_gradient=int(g.norm()<=1e-30))
        if diagnostic:transient.update(theta=theta.cpu(),raw=raw.cpu())
    # Parent executes the supervised backward first, then this Adam hook, then its unchanged EMA.
    with patch.object(opt,'step',proposal):
        row=parent.step(model,ema,opt,l,None,'T_LCTX',task['seed'],task['domain'],epoch,index,counts,patients)
    row.update(detail,pseudo_labels=0,response_images=0 if probe is None else len(probe['image']),
               response_VJP=2 if lam else 0,EMA_updates=1)
    return row,transient

@torch.no_grad()
def supervised_value(model,l,task,epoch,index,counts):
    """Read-only native LCTX forward, same mix/noise/source collection; no extra backward."""
    mc=parent.mathcore;key=(task['seed'],task['domain'],epoch,index)
    if epoch>20:
        donor={k:l[k].flip(0) for k in ('image','geometry')};ix,_=mc.pairing(len(l['image']),len(donor['image']),l['image'].device)
        mask,_=mc.rectangles(len(l['image']),*l['image'].shape[-2:],key,l['image'].device);ux=noisy(donor['image'],key)[ix]
        za,_=fwd(model,torch.where(mask[:,None],l['image'],ux),False,key,counts,'diagnostic_supervised_A')
        zb,_=fwd(model,torch.where(mask[:,None],ux,l['image']),False,key,counts,'diagnostic_supervised_B')
        logp,_=mc.collect_sources(za.log_softmax(1),zb.log_softmax(1),mask)
    else:
        z,_=fwd(model,l['image'],False,key,counts,'diagnostic_supervised_L');logp=z.log_softmax(1)
    ce,dice=mc.supervised_parts(logp,l['label']);return float(ce+dice)

@torch.no_grad()
def actual_diagnostic(model,l,probe,task,epoch,index,counts,transient,row):
    ps=parameters(model);corrected=vector(ps).clone();values={};key=(task['seed'],task['domain'],epoch,index)
    try:
        for name,v in [('before',transient['theta']),('raw',transient['raw']),('corrected',corrected)]:
            assign(ps,v.to(corrected));chi,valid=response(model,probe,key,counts,'diagnostic_response')
            values[name]=(chi.detach().double(),supervised_value(model,l,task,epoch,index,counts))
        old=values['before'][0];mask=valid.expand_as(old);den=mask.sum().clamp_min(1)
        return dict(raw_response_RMS=float(((values['raw'][0]-old).square()*mask).sum().div(den).sqrt()),
                    corrected_response_RMS=float(((values['corrected'][0]-old).square()*mask).sum().div(den).sqrt()),
                    before_loss=values['before'][1],raw_loss_delta=values['raw'][1]-values['before'][1],corrected_loss_delta=values['corrected'][1]-values['before'][1],
                    first_order_raw=row['g_dot_d0'],first_order_algebra_corrected=row['g_dot_dstar'],first_order_applied=row['g_dot_applied'],
                    scope='same current training tensors; descriptive only; no accept/reject or tuning')
    finally:assign(ps,corrected)
