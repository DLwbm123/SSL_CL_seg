"""Reject R consistency gradients that oppose current supervised R gradients."""
from pathlib import Path
import torch
from ..f5_u_strength_v6.run import StrengthTrainer
from ..five_frameworks_v1.gate import digest

def reject(labeled,unlabeled):
    if labeled.shape!=unlabeled.shape:raise ValueError('gradient shapes differ')
    if not torch.isfinite(labeled).all() or not torch.isfinite(unlabeled).all():raise FloatingPointError('nonfinite rejection input')
    dot=(labeled.double()*unlabeled.double()).sum()
    if not torch.isfinite(dot):raise FloatingPointError('nonfinite gradient dot')
    blocked=bool(dot<0)
    return torch.zeros_like(unlabeled) if blocked else unlabeled,dict(blocked=blocked,dot_before=float(dot))

class GatedTrainer(StrengthTrainer):
    def __init__(self,*args,gate_enabled=True,**kwargs):
        self.gate_enabled=gate_enabled;super().__init__(*args,**kwargs)
        self.telemetry.update(R_gate_calls=0,R_blocked_U_calls=0,R_effective_nonzero_U_calls=0)
    @classmethod
    def for_resume(cls,model,provider,options=None,cwmi=None,execution=None,*,gate_enabled=True):
        return cls(model,provider,options,cwmi,initialize=False,execution=execution,gate_enabled=gate_enabled)
    def loss_contract(self):
        return {**super().loss_contract(),'gate_enabled':self.gate_enabled,'gate_equation':'R U=0 if gL dot gU<0; otherwise unchanged','gate_implementation':digest(Path(__file__).read_text())}
    def transform_gradients(self,grads):
        super().transform_gradients(grads)
        if not self.gate_enabled:return
        p=self.model.sidecar.r;l,u=grads[id(p)]['L'],grads[id(p)]['U']
        if u is None:return
        if l is None:l=torch.zeros_like(p)
        filtered,stats=reject(l,u)
        if stats['blocked']:p.grad=l.clone()
        self.module_stats.update(**stats,R_effective_U_norm=float(filtered.norm()))
        self.telemetry['R_gate_calls']+=1;self.telemetry['R_blocked_U_calls']+=int(stats['blocked']);self.telemetry['R_effective_nonzero_U_calls']+=int(bool(filtered.norm()>0))
