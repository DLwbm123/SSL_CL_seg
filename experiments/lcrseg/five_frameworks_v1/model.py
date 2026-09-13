import copy
import torch
from torch import nn
from .kernels import StageSubspaceAdapter, channel_basis


class Model(nn.Module):
    def __init__(self, parent, family, ratio=.5, previous=None):
        super().__init__()
        self.parent,self.family=parent,family
        kernel=parent.effective_readout_kernel_at_entry()
        d=kernel.shape[1]
        if d<2:
            raise ValueError('incompatible feature width < 2')
        self.subspace = family in ('F1','F3','F4','F5')
        self.dense = family.startswith(('B3','B4'))
        self.spectral={}
        if self.subspace:
            k=max(1,min(d-1,round(d*ratio)))
            q,s=channel_basis(kernel,k,previous)
            self.sidecar=StageSubspaceAdapter(q,previous)
            # All diagnostics computed from native C F_prev in the same coordinate.
            _, all_s=channel_basis(kernel,d,previous)
            self.spectral={'rank':k,'width':d,'proxy_degenerate':bool(all_s.max()==0),
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

    def parts(self,x,detach_parent=False,scale=None):
        h=self.parent.features(x,'student')
        if detach_parent: h=h.detach()
        if self.sidecar is None: return h,h,h
        out,before=self.sidecar(h,backward_scale=scale,return_pre_previous=True)
        return h,before,out

    def forward(self,x,detach_parent=False,scale=None):
        _,_,h=self.parts(x,detach_parent,scale)
        return self.parent.native_readout(h,x.shape[-2:],'student')

    def u_parameters(self):
        groups=self.parent.parameter_groups()
        if self.family in ('F1','F3','F4','F5'): return [self.sidecar.r]
        if self.family=='F2': return groups['output_factors']
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def teacher(self):
        ema=copy.deepcopy(self).eval().requires_grad_(False)
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
