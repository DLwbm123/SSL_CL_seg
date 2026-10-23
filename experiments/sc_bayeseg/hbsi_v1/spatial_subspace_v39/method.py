"""Fixed spatial mask subspace risk."""
import json
import torch
from ..semantic_alignment_v23.method import Objective as BaseObjective
from ..semantic_evidence_v22.method import sample_semantics
from ..posterior_teacher_v25.method import parameter_groups,AUDIT_STEPS
from .reference import subspace_risk,check_fixture


def auxiliary(s,x,reference):
    assert x.shape==(1,1,128,128)
    rng=torch.Generator().manual_seed(39000000+101*s.steps);probs=[]
    for _ in range(2):
        theta=s.q.rsample(rng);a=s.current.rsample(theta,rng)
        for _ in range(2):probs.append(sample_semantics(s,x,theta,a,rng)[0])
    p=torch.stack(probs).mean(0);foreground=p[:,1]
    risk=subspace_risk(foreground,reference)
    return risk,dict(subspace_residual_mse=risk.detach(),foreground_mean=foreground.detach().mean(),foreground_square_mean=foreground.detach().square().mean(),minimum_probability=p.detach().min())



class Objective:
    def __init__(self,state,refiner,use_u,reference,audit_path=None):
        self.reference=reference;self.s=state;self.counter=refiner.counter;self.use_u=use_u
        self.base=BaseObjective(state,refiner,False);self.audit_path=audit_path;self.pending=None

    def estimate(self,L,U,counts,generator,sampling_weights=None):
        s=self.s;assert U is not None and 'observations' not in U
        if self.use_u:s.check_batch(U,'U')
        else:assert U['role']=='appearance_donor_L' and U['stage']==s.active['id']
        base,record=self.base.estimate(L,None,counts,generator,sampling_weights)
        aux,stats=auxiliary(s,U['x'],self.reference);loss=base+aux
        if not torch.isfinite(loss):raise FloatingPointError('spatial subspace risk')
        self.counter.add('subspace_global_draws',2);self.counter.add('subspace_semantic_forwards',4*len(U['x']))
        if self.audit_path is not None and s.steps in AUDIT_STEPS:
            groups=parameter_groups(s);params=[p for group in groups.values() for p in group]
            gb=torch.autograd.grad(base,params,retain_graph=True,allow_unused=True)
            gu=torch.autograd.grad(aux,params,retain_graph=True,allow_unused=True)
            row=dict(step=s.steps,auxiliary_pool='U' if self.use_u else 'L',extra_reverse_calls=2,groups={});offset=0
            for name,pp in groups.items():
                b=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(pp,gb[offset:offset+len(pp)])])
                v=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(pp,gu[offset:offset+len(pp)])]);offset+=len(pp)
                nb,nv=float(b.norm()),float(v.norm())
                row['groups'][name]=dict(base_gradient_norm=nb,auxiliary_gradient_norm=nv,auxiliary_over_base=nv/nb if nb else None,cosine=float(torch.dot(b,v)/(b.norm()*v.norm())) if nb and nv else None)
            self.pending=(row,{name:[p.detach().clone() for p in pp] for name,pp in groups.items()})
        return loss,dict(record,scaled=loss.detach(),base_objective=base.detach(),subspace_risk=aux.detach(),auxiliary_pool_U=loss.new_tensor(float(self.use_u)),**stats)

    def after_update(self,step):
        if self.pending is None:return
        row,before=self.pending;assert step==row['step']+1
        for name,params in parameter_groups(self.s).items():
            row['groups'][name]['actual_combined_Adam_displacement']=float(torch.cat([(p.detach()-q).reshape(-1) for p,q in zip(params,before[name])]).norm())
        with self.audit_path.open('a') as f:f.write(json.dumps(row)+'\n')
        self.pending=None
