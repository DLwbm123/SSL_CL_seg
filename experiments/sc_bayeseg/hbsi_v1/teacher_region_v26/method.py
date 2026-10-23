"""Fixed posterior targets: discrete CE and region risk, without weight search."""
import json
import torch
import torch.nn.functional as F
from ..semantic_alignment_v23.method import Objective as BaseObjective
from ..paired_semantic_v19.method import balanced_risk
from ..posterior_teacher_v25.method import teacher_risk,student_probability,parameter_groups,AUDIT_STEPS


def region_risk(target,student,arm):
    assert arm in ('H','R')
    # Argmax of the same cached posterior mean, with no confidence cutoff or teacher update.
    hard=F.one_hot(target.detach().argmax(1),2).permute(0,3,1,2).to(student)
    ce,details=teacher_risk(hard,student,torch.ones_like(hard[:,0]))
    full=balanced_risk(student,hard).mean()
    details.update(U_hard_CE=ce,U_foreground_Dice=full-ce)
    return (ce if arm=='H' else full),details


class Objective:
    def __init__(self,state,refiner,arm,audit_path=None):
        self.s=state;self.base=BaseObjective(state,refiner,False);self.arm=arm
        self.counter=refiner.counter;self.audit_path=audit_path;self.pending=None

    def estimate(self,L,U,counts,generator,sampling_weights=None):
        s=self.s;base,record=self.base.estimate(L,None,counts,generator,sampling_weights)
        u=base*0;details={}
        if self.arm in ('H','R'):
            s.check_batch(U,'U');target=U['teacher_mean'].to(L['x'].device)
            w=torch.ones_like(target[:,0])
            student=student_probability(s,U['x']);u,details=region_risk(target,student,self.arm)
            self.counter.add('local_mc_samples',s.config.global_mc*s.config.local_mc*len(U['x']))
            self.counter.add('global_mc_samples',s.config.global_mc)
        loss=base+u
        if not torch.isfinite(loss):raise FloatingPointError('fixed teacher risk')
        if self.audit_path is not None and s.steps in AUDIT_STEPS:
            groups=parameter_groups(s);params=[p for group in groups.values() for p in group]
            gb=torch.autograd.grad(base,params,retain_graph=True,allow_unused=True)
            gu=torch.autograd.grad(u,params,retain_graph=True,allow_unused=True) if self.arm in ('H','R') else [None]*len(params)
            row=dict(step=s.steps,arm=self.arm,extra_reverse_calls=2 if self.arm in ('H','R') else 1,groups={});offset=0
            for name,pp in groups.items():
                b=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(pp,gb[offset:offset+len(pp)])])
                v=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(pp,gu[offset:offset+len(pp)])]);offset+=len(pp)
                nb,nv=float(b.norm()),float(v.norm())
                row['groups'][name]=dict(base_gradient_norm=nb,U_gradient_norm=nv,U_over_base=nv/nb if nb else None,cosine=float(torch.dot(b,v)/(b.norm()*v.norm())) if nb and nv else None)
            self.pending=(row,{name:[p.detach().clone() for p in pp] for name,pp in groups.items()})
        return loss,dict(record,scaled=loss.detach(),U_risk=u.detach(),**{k:v.detach() for k,v in details.items()})

    def after_update(self,step):
        if self.pending is None:return
        row,before=self.pending;assert step==row['step']+1
        for name,params in parameter_groups(self.s).items():
            row['groups'][name]['actual_combined_Adam_displacement']=float(torch.cat([(p.detach()-q).reshape(-1) for p,q in zip(params,before[name])]).norm())
        with self.audit_path.open('a') as f:f.write(json.dumps(row)+'\n')
        self.pending=None
