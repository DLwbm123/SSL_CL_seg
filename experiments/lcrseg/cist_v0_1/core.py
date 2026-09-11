"""Frozen source, one shared transport, current-image context; no teacher or pseudo labels."""
import contextlib
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from experiments.lcrseg.ams_seq_transfer_v0_1 import core as parent
from experiments.lcrseg.ams_seq_transfer_v0_1.core import *
from experiments.lcrseg.lctx_weight_memory_v0_1.core import hash_state
from experiments.lcrseg.ssl_anchored_mix_v0_1 import telemetry

ARMS=('SCALE_LU','AFFINE_LU','ISO_GLOBAL_LU','ISO_COND_L','ISO_COND_LU','ISO_RANDOM_LU')
MAIN='ISO_COND_LU'
ISO64_LIMIT=1e-10
DISTANCE32_LIMIT=2e-5

def count(label,amount=1):
    if telemetry.ACTIVE is not None:telemetry.ACTIVE.counts[label]+=amount

def canonical(q):
    index=q.abs().argmax(0);sign=q[index,torch.arange(q.shape[1])].sign()
    return q*sign

def readout_basis(kernel,seed,domain):
    k=kernel.detach().cpu().double()
    if k.shape!=(3,16,3,3) or not torch.isfinite(k).all():raise ValueError('source readout geometry/finite')
    pi=torch.eye(3,dtype=torch.float64)-torch.ones(3,3,dtype=torch.float64)/3
    g=sum(k[:,:,i,j].T@pi@k[:,:,i,j] for i in range(3) for j in range(3));g=(g+g.T)/2
    eig,q=torch.linalg.eigh(g);eig=eig.flip(0);q=canonical(q.flip(1))
    tol=16*torch.finfo(torch.float64).eps*float(eig[0])
    if eig[0]<=0 or int((eig>tol).sum())<8:raise ValueError('degenerate source readout')
    gen=torch.Generator().manual_seed(stable_seed('CIST_V0_1',seed,domain,'random_basis'))
    qr,_=torch.linalg.qr(torch.randn(16,16,dtype=torch.float64,generator=gen));qr=canonical(qr)
    t=dict(V=q[:,:8].contiguous(),N=q[:,8:].contiguous(),V_random=qr[:,:8].contiguous(),N_random=qr[:,8:].contiguous(),G_R=g,eigenvalues=eig,kernel=k)
    gap=float((eig[7]-eig[8])/eig[0]);orth=float((q.T@q-torch.eye(16,dtype=q.dtype)).norm())
    assert orth<1e-10
    return t,dict(rank=8,channel_dimension=16,eigenvalues=eig.tolist(),relative_cut_gap=gap,near_degenerate_cut=gap<=1e-8,
                  numerical_rank=int((eig>tol).sum()),orthogonality_F=orth,sign_rule='largest absolute entry positive in every column',
                  tensor_hashes={n:hash_state({n:v}) for n,v in t.items()})

def cayley(m):
    a=(m.double()-m.double().transpose(-1,-2))/2
    eye=torch.eye(8,device=m.device,dtype=torch.float64)
    if not torch.isfinite(a).all():raise FloatingPointError('nonfinite Cayley input')
    q=torch.linalg.solve(eye-a,eye+a)
    if not torch.isfinite(q).all():raise FloatingPointError('nonfinite Cayley output')
    return q,a

class Transport(nn.Module):
    def __init__(self,tensors,arm,seed,domain):
        super().__init__()
        if arm not in ARMS:raise PermissionError('unknown transport arm')
        self.arm=arm;self.journal=None
        suffix='_random' if arm=='ISO_RANDOM_LU' else ''
        self.register_buffer('V',tensors['V'+suffix].float().clone());self.register_buffer('N',tensors['N'+suffix].float().clone())
        if arm=='SCALE_LU':self.scale=nn.Parameter(torch.zeros(8));self.bias=nn.Parameter(torch.zeros(8))
        else:
            with rng(torch.device('cpu'),'CIST_V0_1',seed,domain,'controller_init'):
                self.mlp=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,72))
            nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)
        assert sum(p.numel() for p in self.parameters())==(16 if arm=='SCALE_LU' else 2920)

    def forward(self,h):
        if h.ndim!=4 or h.shape[1]!=16 or h.dtype!=torch.float32:raise ValueError('FP32 B16HW source features required')
        x=h.flatten(2);v,n=self.V,self.N;s=v.T@x;z=n.T@x
        context=torch.cat((z.mean(-1),torch.sqrt(z.var(-1,unbiased=False)+1e-6)),1)
        eye=torch.eye(8,device=h.device,dtype=torch.float32)
        if self.arm=='SCALE_LU':
            q=torch.diag_embed(1+self.scale).expand(len(h),-1,-1);bias=self.bias.expand(len(h),-1);orth64=None
        else:
            raw=self.mlp(torch.zeros_like(context) if self.arm=='ISO_GLOBAL_LU' else context)
            m=raw[:,:64].reshape(-1,8,8);bias=raw[:,64:]
            if self.arm=='AFFINE_LU':q=eye+m;orth64=None
            else:
                if self.journal:self.journal('PRE_SOLVE',dict(M=m.detach().cpu()))
                q64,a=cayley(m);orth64=(q64.transpose(-1,-2)@q64-torch.eye(8,device=h.device,dtype=torch.float64)).norm(dim=(-2,-1))
                if self.journal:self.journal('POST_SOLVE',dict(Q=q64.detach().cpu(),orthogonality64_max=float(orth64.detach().max())))
                if orth64.detach().max()>ISO64_LIMIT:raise FloatingPointError('Cayley orthogonality failure')
                q=q64.float();count('cayley_solves',len(h))
        transformed=q@s+bias[:,:,None]
        out=(x+v@((q-eye)@s+bias[:,:,None])).reshape_as(h)
        if not torch.isfinite(out).all():raise FloatingPointError('nonfinite transported features')
        count('controller_forward');count('controller_images',len(h))
        return out,dict(S=s,transformed=transformed,Q=q,b=bias,context=context,orthogonality64=orth64)

class Student(nn.Module):
    def __init__(self,core,transport):
        super().__init__();self.core=core.requires_grad_(False).eval();self.transport=transport

    @torch.no_grad()
    def features(self,x):
        if x.ndim!=4 or x.shape[1]!=3 or any(n%8 for n in x.shape[-2:]):raise ValueError('source input geometry')
        m=self.core;e1=m.enc1(x);e2=m.enc2(m.pool(e1));e3=m.enc3(m.pool(e2));b=m.bottleneck(m.pool(e3))
        h=m.decoder.dec1(m.decoder.dec2(m.decoder.dec3(b,e3),e2),e1)
        count('model_forward');count('image_forward',len(x));count('frozen_core_forward');count('frozen_core_images',len(x))
        return h

    def readout(self,h,shape):
        count('readout_forward');count('readout_images',len(h))
        return F.interpolate(self.core.decoder.conv_logit(h,stochastic=False),size=shape,mode='bilinear',align_corners=True)

    def forward(self,x,*,stochastic_classifier=False):
        if stochastic_classifier:raise PermissionError('frozen raw linear readout only')
        h=self.features(x);hp,detail=self.transport(h)
        return self.readout(hp,x.shape[-2:]),(h,hp,detail)

def appearance(x,key):
    # eps_c is an independently sampled per-image RGB offset, matching the plan's c-only index.
    out=[]
    for i,image in enumerate(x):
        gen=torch.Generator(device=x.device).manual_seed(stable_seed('CIST_V0_1',*key,'appearance',i))
        gamma=.8+.4*torch.rand((),device=x.device,generator=gen)
        gain=.9+.2*torch.rand(3,1,1,device=x.device,generator=gen)
        noise=.02*torch.randn(3,1,1,device=x.device,generator=gen)
        out.append((gain*image.pow(gamma)+noise).clamp(0,1))
    return torch.stack(out)

def view_loss(a,b,valid):
    if valid.dtype!=torch.bool or valid.shape!=a['S'].shape[:1]+(a['S'].shape[-1],):raise ValueError('image geometry mask only')
    mask=valid[:,None];den=8*valid.sum(1)
    if (den==0).any():raise ValueError('no geometrically valid pixels')
    energy=(.5*((a['S'].square()+b['S'].square())*mask).sum((1,2))/den).detach()+1e-4
    per=((a['transformed']-b['transformed']).square()*mask).sum((1,2))/den
    return (per/energy).mean(),dict(view_error_before=float((((a['S']-b['S']).square()*mask).sum((1,2))/den/energy).detach().mean()),view_error_after=float((per/energy).detach().mean()),energy_min=float(energy.min()))

@torch.no_grad()
def mechanism(h,hp,detail):
    # Deterministic widely separated pixel pairs; denominator is source pair energy, not update norm.
    x=h.flatten(2);y=hp.flatten(2);p=x.shape[-1];ix=torch.linspace(0,p-1,32,device=x.device).long();jx=(ix+p//2)%p
    before=(x[:,:,ix]-x[:,:,jx]).double().square().sum(1);after=(y[:,:,ix]-y[:,:,jx]).double().square().sum(1)
    floor=1e-6*x.double().square().sum(1).mean(1,keepdim=True)+1e-12
    distortion=(after-before).abs()/torch.maximum(before,floor)
    q=detail['Q'].detach();bias=detail['b'].detach();orth=detail['orthogonality64']
    return dict(distance_relative_max=float(distortion.max()),distance_relative_mean=float(distortion.mean()),
                Q_norm=float(q.norm(dim=(-2,-1)).mean()),b_norm=float(bias.norm(dim=-1).mean()),
                Q_batch_variance=float((q.double()-q.double().mean(0)).square().sum()/len(h)) if len(h)>1 else None,
                b_batch_variance=float((bias.double()-bias.double().mean(0)).square().sum()/len(h)) if len(h)>1 else None,images=len(h),
                orthogonality64_max=None if orth is None else float(orth.max()))

def labeled_loss(student,l,seed,domain,epoch,index,counts,patients):
    parent.mathcore.validate_batch(l,True);key=(seed,domain,epoch,index);coords=[]
    if epoch>20:
        donor=parent.donor_batch(l,patients);mask,coords=parent.mathcore.rectangles(len(l['image']),*l['image'].shape[-2:],key,l['image'].device)
        x=parent.mathcore.noisy(donor['image'],key)
        za,_=fwd(student,torch.where(mask[:,None],l['image'],x),False,key,counts,'student_mix_A')
        zb,_=fwd(student,torch.where(mask[:,None],x,l['image']),False,key,counts,'student_mix_B')
        logl,_=parent.mathcore.collect_sources(za.log_softmax(1),zb.log_softmax(1),mask)
    else:
        z,_=fwd(student,l['image'],False,key,counts,'student_l');logl=z.log_softmax(1)
    ce,dice=parent.mathcore.supervised_parts(logl,l['label'])
    return ce,dice,coords

def step(student,opt,l,u,task,epoch,index,counts,patients):
    opt.zero_grad(set_to_none=True);ce,dice,coords=labeled_loss(student,l,task['seed'],task['domain'],epoch,index,counts,patients)
    view=ce*0;info=dict(view_error_before=0.,view_error_after=0.,energy_min=None);diag=None
    if task['arm']!='ISO_COND_L' and epoch>20:
        if u is None or set(u)!={'image','geometry'}:raise PermissionError('current U image/geometry only')
        ha=student.features(u['image']);hb=student.features(appearance(u['image'],(task['seed'],task['domain'],epoch,index)))
        hpa,a=student.transport(ha);hpb,b=student.transport(hb)
        view,info=view_loss(a,b,u['geometry'].flatten(1));diag=mechanism(ha,hpa,a)
    elif u is not None:raise PermissionError('L-only or warmup cannot receive U')
    lam=min(1.,max(0.,(epoch-20)/20));loss=ce+dice+lam*view
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite objective')
    loss.backward();counts['backward']+=1
    if any(p.grad is not None for p in student.core.parameters()):raise RuntimeError('frozen core gradient')
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student.transport.parameters()):raise FloatingPointError('nonfinite controller gradient')
    first_grad=None if task['arm']=='SCALE_LU' else float(student.transport.mlp[0].weight.grad.norm())
    opt.step();counts['optimizer_steps']+=1
    return dict(loss=float(loss.detach()),CE=float(ce.detach()),GT_Dice=float(dice.detach()),L_view=float(view.detach()),lambda_view=lam,
                anchor_label_records=len(l['image']),labeled_source_scoring_multiplicity=1,donor_context_records=len(l['image']) if coords else 0,
                donor_extra_image_reads=0,U_image_records=0 if u is None else len(u['image']),pseudo_label_records=0,EMA_updates=0,
                mix_rectangles=coords,first_layer_weight_gradient_norm=first_grad,view=info,mechanism=diag)

@contextlib.contextmanager
def training_access(domain,arm):
    original=CurrentData.__init__;stage=DOMAINS.index(domain)+1
    def init(self,data,actual_stage,role,**kw):
        allowed=('train_labeled',) if arm=='ISO_COND_L' else ('train_labeled','train_unlabeled')
        if actual_stage!=stage or kw.get('domain',stage)!=stage or kw.get('purpose','train')!='train' or role not in allowed:raise PermissionError('current-domain role boundary')
        return original(self,data,actual_stage,role,**kw)
    with patch.object(CurrentData,'__init__',init):yield
