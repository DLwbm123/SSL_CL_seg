"""V13 reference with Adam displacement alignment; optimizer state stays native."""
from pathlib import Path
import torch
from ..f5_class_reference_v13.model import ClassReferenceModel,ClassReferenceTrainer
from ..f5_gua_candidate.alignment import align_displacement
from ..five_frameworks_v1.gate import digest


def dot_stats(direction,reference):
    dot=sum((a.double()*b.double()).sum() for a,b in zip(direction,reference))
    scale=(sum(a.double().square().sum() for a in direction)*sum(b.double().square().sum() for b in reference)).sqrt()
    return float(dot),float(dot/scale) if scale>0 else 0.


def adam_metric(optimizer,params):
    if not isinstance(optimizer,torch.optim.Adam):raise TypeError('GUA qualification binds native Adam')
    groups={id(p):g for g in optimizer.param_groups for p in g['params']}
    out=[]
    for p in params:
        g=groups[id(p)];s=optimizer.state[p]
        if not s or float(s['step'])<=0:raise RuntimeError('missing committed Adam proposal state')
        v=s['max_exp_avg_sq'] if g.get('amsgrad',False) else s['exp_avg_sq']
        out.append((v.double()/(1-g['betas'][1]**float(s['step']))).sqrt()+g['eps'])
    return out


class GUAModel(ClassReferenceModel):
    def __init__(self,*args,gua=True,**kwargs):
        super().__init__(*args,class_reference=True,**kwargs)
        self.gua=gua


class GUATrainer(ClassReferenceTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gua_reference=None
        self.telemetry.update(gua_calls=0,gua_projected_calls=0,gua_raw_safe_proposal_conflicts=0,
                              gua_actual_negative_dot_calls=0,gua_actual_cosine_below_tolerance_calls=0)

    def loss_contract(self):
        return {**super().loss_contract(),'gua':self.model.gua,
                'gua_formula':'V13 raw-gradient guard retained; weighted nearest Adam descent displacement against same balanced entry reference; M=sqrt(vhat)+eps; moments unchanged',
                'gua_implementation':digest(Path(__file__).read_text()),
                'gua_solver':digest(Path(__file__).parents[1].joinpath('f5_gua_candidate/alignment.py').read_text()),
                'gua_update_hook':digest(Path(__file__).parents[1].joinpath('five_frameworks_v1/train_stage.py').read_text())}

    def transform_gradients(self,grads):
        self.gua_reference=None
        if self.model.gua and self.entry_reference_grads is not None:
            params=[p for p in self.model.parameters() if p.requires_grad]
            self.gua_reference=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,self.entry_reference_grads)]
        super().transform_gradients(grads)
        self.module_stats.update(gua=self.model.gua,gua_applied=False)

    @torch.no_grad()
    def transform_update(self,before):
        if self.gua_reference is None:return
        params=[p for p in self.model.parameters() if p.requires_grad]
        reference=self.gua_reference
        raw=[torch.zeros_like(p) if p.grad is None else p.grad for p in params]
        raw_dot,raw_cos=dot_stats(raw,reference)
        proposal=[before[id(p)]-p for p in params]
        corrected,stats=align_displacement(proposal,[reference],adam_metric(self.optimizer,params))
        if stats['projected']:
            for p,d in zip(params,corrected):p.copy_(before[id(p)]-d)
        if any(not torch.isfinite(p).all() for p in params):raise FloatingPointError('nonfinite aligned parameters')
        actual=[before[id(p)]-p for p in params]
        actual_dot,actual_cos=dot_stats(actual,reference)
        self.telemetry['gua_calls']+=1
        self.telemetry['gua_projected_calls']+=int(stats['projected'])
        self.telemetry['gua_raw_safe_proposal_conflicts']+=int(raw_cos>=-1e-6 and stats['dots_before'][0]<0)
        self.telemetry['gua_actual_negative_dot_calls']+=int(actual_dot<0)
        self.telemetry['gua_actual_cosine_below_tolerance_calls']+=int(actual_cos < -1e-6)
        self.module_stats.update(gua_applied=True,gua_projected=stats['projected'],gua_raw_dot=raw_dot,
                                gua_raw_cosine=raw_cos,gua_proposal_dot=stats['dots_before'][0],
                                gua_aligned_dot=stats['dots_after'][0],gua_actual_dot=actual_dot,gua_actual_cosine=actual_cos)
        self.gua_reference=None
