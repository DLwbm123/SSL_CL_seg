"""Same-sample full/task diagnostic with sparse gradient and actual-step audit."""
import torch
from ..observation_block_v43.method import Objective as Components


class Objective(Components):
    def __init__(self,state,refiner,arm,audits):
        super().__init__(state,refiner);self.arm=arm;self.audits=audits;self.pending=None

    def estimate(self,L,U,counts,generator,sampling_weights=None):
        assert U is None and sampling_weights is None
        _,v=self.components(L,counts,generator,preflight=True)
        full=v['full_objective_same_samples'];task=v['task']
        loss=full if self.arm=='Full' else task
        if self.s.steps in (0,99,199):
            params=[p for g in self.s.optimizer.param_groups for p in g['params'] if p.requires_grad]
            gt=torch.autograd.grad(task,params,allow_unused=True,retain_graph=True)
            go=torch.autograd.grad(full-task,params,allow_unused=True,retain_graph=True)
            rows=[];start=0
            for name,g in zip(('q','eta','phi'),self.s.optimizer.param_groups):
                ps=[p for p in g['params'] if p.requires_grad];n=len(ps)
                t=[torch.zeros_like(p) if x is None else x for p,x in zip(ps,gt[start:start+n])]
                o=[torch.zeros_like(p) if x is None else x for p,x in zip(ps,go[start:start+n])]
                tn=sum(float(x.square().sum()) for x in t)**.5;on=sum(float(x.square().sum()) for x in o)**.5
                dot=sum(float(x.mul(y).sum()) for x,y in zip(t,o))
                row=dict(group=name,task_norm=tn,other_norm=on,cosine=dot/(tn*on) if tn*on else None)
                rows.append((row,ps,[p.detach().clone() for p in ps],t,o));start+=n
            self.pending=(self.s.steps,rows)
        return loss,{**{k:x.detach() for k,x in v.items()},'scaled':loss.detach()}

    def after_step(self,step):
        if self.pending is None:return
        before,rows=self.pending;assert step==before+1
        report=[]
        for row,ps,old,t,o in rows:
            delta=[p.detach()-b for p,b in zip(ps,old)]
            row.update(adam_delta_norm=sum(float(d.square().sum()) for d in delta)**.5,
                task_dot_delta=sum(float(a.mul(d).sum()) for a,d in zip(t,delta)),
                other_dot_delta=sum(float(a.mul(d).sum()) for a,d in zip(o,delta)))
            report.append(row)
        self.audits.append(dict(step_before=before,groups=report));self.pending=None
