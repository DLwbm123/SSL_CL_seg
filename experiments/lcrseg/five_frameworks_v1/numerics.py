"""Finite-state transaction checks; no loss-improvement or efficacy gate."""
import math
import torch


def finite(value,name):
    if isinstance(value,torch.Tensor):
        if not bool(torch.isfinite(value).all()):raise FloatingPointError('nonfinite '+name)
    elif isinstance(value,dict):
        for k,v in value.items():finite(v,name+'.'+str(k))
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):finite(v,name+'.'+str(i))
    elif isinstance(value,float) and not math.isfinite(value):raise FloatingPointError('nonfinite '+name)


def ema_value(a,b):
    return a.detach().clone().mul_(.99).add_(b,alpha=.01)


@torch.no_grad()
def validate_commit(trainer,pending):
    m,t=trainer.model,trainer.ema
    finite(m.state_dict(),'student')
    finite(trainer.optimizer.state_dict(),'optimizer')
    finite(trainer.scheduler.state_dict(),'scheduler')
    finite(trainer.scaler.state_dict(),'scaler')
    for a in m.parent.adapters:finite(a.effective_weight(),'student effective adapter')
    if m.sidecar is not None:finite(m.sidecar.effective(),'student effective F')
    if getattr(trainer,'native',False):
        m.parent.update_dense_ema(t.parent,validate_only=True)
    else:
        # Validate one EMA tensor at a time, not a third full model copy.
        for a,b in zip(t.parameters(),m.parameters()):
            finite(a if a is b else (ema_value(a,b) if b.requires_grad else b),'candidate EMA parameter')
        for a,b in zip(t.buffers(),m.buffers()):finite(b,'candidate EMA buffer')
        for a,b in zip(t.parent.adapters,m.parent.adapters):
            finite(b.base+ema_value(a.b,b.b)@ema_value(a.a,b.a),'candidate EMA effective adapter')
    if m.sidecar is not None:
        q=m.sidecar.q;r=ema_value(t.sidecar.r,m.sidecar.r)
        finite(m.sidecar.previous@(torch.eye(q.shape[0]).to(q)+q@r@q.T),'candidate EMA effective F')
    if pending:
        import copy
        candidate=copy.deepcopy(trainer.prototypes)
        finite(pending,'pending current-L prototypes')
        candidate.commit(*pending)
        finite(candidate.values,'candidate prototype cache')
