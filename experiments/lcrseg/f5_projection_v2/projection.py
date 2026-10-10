"""One-sided PCGrad adaptation on the existing F5 R gradient only."""
from pathlib import Path
import torch
from ..f5_module_pilots_v1.modules import PilotTrainer
from ..five_frameworks_v1.gate import digest


def project(labeled,unlabeled):
    if labeled.shape!=unlabeled.shape:raise ValueError('gradient shapes differ')
    if not torch.isfinite(labeled).all() or not torch.isfinite(unlabeled).all():
        raise FloatingPointError('nonfinite gradient projection input')
    l,u=labeled.double(),unlabeled.double();dot=(l*u).sum();norm=l.square().sum()
    conflict=bool(dot<0 and norm>0)
    filtered=unlabeled-(dot/norm).to(labeled)*labeled if conflict else unlabeled
    after=(l*filtered.double()).sum()
    tolerance=1e-6*(l.norm()*u.norm()).clamp_min(1e-20)
    if conflict and after < -tolerance:raise FloatingPointError('projection violated raw-gradient half-space')
    return filtered,dict(projected=conflict,dot_before=float(dot),dot_after=float(after))


class ProjectionTrainer(PilotTrainer):
    def __init__(self,*args,project_enabled=True,**kwargs):
        self.project_enabled=project_enabled
        super().__init__(*args,**kwargs)
        self.telemetry.setdefault('projection_calls',0);self.telemetry.setdefault('conflicting_projection_calls',0)

    @classmethod
    def for_resume(cls,model,provider,options=None,cwmi=None,execution=None,*,arm='F5',project_enabled=True):
        return cls(model,provider,options,cwmi,initialize=False,execution=execution,arm=arm,project_enabled=project_enabled)

    def loss_contract(self):
        return {**super().loss_contract(),'projection_enabled':self.project_enabled,
                'projection_equation':'gU-min(0,gL.gU)/norm(gL)^2*gL; R only',
                'projection_implementation':digest(Path(__file__).read_text())}

    def transform_gradients(self,gradients):
        if not self.project_enabled:return
        p=self.model.sidecar.r;pair=gradients[id(p)];l,u=pair['L'],pair['U']
        if u is None:return
        if l is None:l=torch.zeros_like(p)
        filtered,stats=project(l,u)
        p.grad=l+filtered
        self.module_stats=stats
        self.telemetry['projection_calls']+=1
        self.telemetry['conflicting_projection_calls']+=int(stats['projected'])
