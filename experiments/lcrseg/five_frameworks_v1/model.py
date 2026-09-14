import copy
import torch
from torch import nn
from .kernels import StageSubspaceAdapter, channel_basis


class Model(nn.Module):
    def __init__(self, parent, family, ratio=.5, previous=None, initialize_basis=True):
        super().__init__()
        self.parent,self.family=parent,family
        self.rank_ratio=ratio;self.role="student"
        kernel=parent.effective_readout_kernel_at_entry()
        d=kernel.shape[1]
        if d<2:
            raise ValueError('incompatible feature width < 2')
        self.subspace = family in ('F1','F3','F4','F5')
        self.dense = family.startswith(('B3','B4'))
        self.spectral={}
        if self.subspace:
            k=max(1,min(d-1,round(d*ratio)))
            if initialize_basis:
                all_q,all_s=channel_basis(kernel,d,previous)
            else:
                all_q=torch.eye(d).to(kernel);all_s=kernel.new_zeros(d)
            q,s=all_q[:,:k],all_s[:k]
            self.sidecar=StageSubspaceAdapter(q,previous)
            # All diagnostics computed from native C F_prev in the same coordinate.

            self.spectral={'rank':k,'width':d,'eigensolves':int(initialize_basis),'proxy_degenerate':bool(all_s.max()==0),
                           'energy':float(s.sum()/all_s.sum().clamp_min(1e-30)),
                           'cut_gap':float(all_s[k-1]-all_s[k]),
                           'coordinate':'raw_post_nonlinearity_before_F_prev'}
            self.register_buffer('spectrum',s)
        elif self.dense:
            self.sidecar=StageSubspaceAdapter(torch.eye(d).to(kernel),previous)
        else:
            if previous is not None and not torch.equal(previous,torch.eye(d).to(previous)):
                raise ValueError('parent-only/F2 cannot inherit a learned feature transform')
            self.sidecar=None

    def checked_mode(self,mode):
        mode=self.role if mode is None else mode
        if mode not in ('student','teacher','eval'):raise ValueError('invalid bridge mode')
        if mode=='student' and not self.training:raise ValueError('student mode requires training module mode')
        if mode in ('teacher','eval') and (self.training or any(p.requires_grad for p in self.parameters())):
            raise ValueError('teacher/eval requires frozen parameters and evaluation module mode')
        return mode

    def configure_training(self):
        self.train(); self.role='student'
        self.parent.configure_stage_training()
        if self.sidecar is not None:
            self.sidecar.previous.requires_grad_(False);self.sidecar.r.requires_grad_(True)

    @classmethod
    def for_resume(cls,parent,family,ratio=.5,previous=None):
        return cls(parent,family,ratio,previous,initialize_basis=False)

    def parts(self,x,detach_parent=False,scale=None,mode=None):
        mode=self.checked_mode(mode)
        h=self.parent.features(x,mode)
        if detach_parent: h=h.detach()
        if self.sidecar is None: return h,h,h
        out,before=self.sidecar(h,backward_scale=scale,return_pre_previous=True)
        return h,before,out

    def forward(self,x,detach_parent=False,scale=None,mode=None):
        mode=self.checked_mode(mode)
        _,_,h=self.parts(x,detach_parent,scale,mode)
        return self.parent.native_readout(h,x.shape[-2:],mode)

    def u_parameters(self):
        groups=self.parent.parameter_groups()
        if self.family in ('F1','F3','F4','F5'): return [self.sidecar.r]
        if self.family=='F2': return groups['output_factors']
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def teacher(self):
        ema=copy.deepcopy(self).eval().requires_grad_(False)
        ema.role="teacher"
        if hasattr(ema.parent,"update_dense_ema"):
            ema.parent.stage_exit();ema.eval().requires_grad_(False)
        if self.sidecar is not None:
            # One immutable F_prev allocation shared by student/current EMA.
            ema.sidecar.previous=self.sidecar.previous
        return ema

    @torch.no_grad()
    def deploy(self):
        parent=copy.deepcopy(self.parent)
        parent.stage_exit()
        kernel=parent.effective_readout_kernel_at_entry()
        transform=(self.sidecar.seal() if self.sidecar is not None else
                   torch.eye(kernel.shape[1]).to(kernel))
        return Deployment(parent,transform).eval().requires_grad_(False)


class Deployment(nn.Module):
    """One sealed student with one fixed dxd transform. No teacher/Q/R/cache."""
    def __init__(self,parent,transform):
        super().__init__();self.parent=parent
        self.transform=nn.Parameter(transform.detach().clone(),requires_grad=False)

    def forward(self,x):
        h=self.parent.features(x,'eval')
        h=torch.einsum('ij,bjhw->bihw',self.transform,h)
        return self.parent.native_readout(h,x.shape[-2:],'eval')
