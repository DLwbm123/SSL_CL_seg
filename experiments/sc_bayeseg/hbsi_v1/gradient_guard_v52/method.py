"""Project only the nonsupervised gradient that opposes the labeled task."""
import math
import torch
from ..observation_block_v43.method import Objective as Components


class Objective(Components):
    def __init__(self,state,refiner,project=True):
        super().__init__(state,refiner)
        self.project=project;self.handles=[];self.pending=None;self.audits=[]
        self.stats={name:dict(steps=0,opposed=0,cosine_sum=0.,cosine_valid=0,alpha_sum=0.) for name in ('q','eta','phi')}

    def clear_hooks(self):
        for handle in self.handles:handle.remove()
        self.handles=[]

    def estimate(self,L,U,counts,generator,sampling_weights=None):
        assert U is None and sampling_weights is None
        self.clear_hooks()
        _,parts=self.components(L,counts,generator,preflight=True)
        full=parts['full_objective_same_samples'];task=parts['task']
        if self.project:
            groups=[[p for p in g['params'] if p.requires_grad] for g in self.s.optimizer.param_groups]
            params=[p for group in groups for p in group]
            gt=torch.autograd.grad(task,params,allow_unused=True,retain_graph=True)
            go=torch.autograd.grad(full-task,params,allow_unused=True,retain_graph=True)
            self.expected=[];before=self.s.steps;audits=[];offset=0
            for name,group in zip(('q','eta','phi'),groups):
                n=len(group);t=[torch.zeros_like(p) if z is None else z.detach() for p,z in zip(group,gt[offset:offset+n])]
                o=[torch.zeros_like(p) if z is None else z.detach() for p,z in zip(group,go[offset:offset+n])]
                dot=sum(float(a.mul(b).sum()) for a,b in zip(t,o));t2=sum(float(a.square().sum()) for a in t);o2=sum(float(a.square().sum()) for a in o)
                alpha=max(0.,-dot/t2) if t2>0 else 0.
                row=self.stats[name];row['steps']+=1;row['opposed']+=int(alpha>0);row['alpha_sum']+=alpha
                if t2>0 and o2>0:row['cosine_valid']+=1;row['cosine_sum']+=dot/math.sqrt(t2*o2)
                for p,a,b in zip(group,t,o):
                    correction=(alpha*a).detach()
                    if alpha:self.handles.append(p.register_hook(lambda grad,c=correction:grad+c))
                    self.expected.append((p,(a+b+correction).detach()))
                if before in (0,199,399,799):audits.append(dict(group=name,task_norm=math.sqrt(t2),other_norm=math.sqrt(o2),cosine=dot/math.sqrt(t2*o2) if t2*o2>0 else None,alpha=alpha,task_gradient=t,parameters=group,previous=[p.detach().clone() for p in group]))
                offset+=n
            if audits:self.pending=(before,audits)
        return full,{k:v.detach() for k,v in parts.items() if k!='full_objective_same_samples'}|{'scaled':full.detach()}

    def after_step(self,step):
        if self.pending is None:return
        before,rows=self.pending;assert step==before+1
        public=[]
        for row in rows:
            delta=[p.detach()-prev for p,prev in zip(row['parameters'],row['previous'])]
            public.append(dict(group=row['group'],task_norm=row['task_norm'],other_norm=row['other_norm'],cosine=row['cosine'],alpha=row['alpha'],task_dot_adam_delta=sum(float(g.mul(d).sum()) for g,d in zip(row['task_gradient'],delta))))
        self.audits.append(dict(step_before=before,groups=public));self.pending=None

    def summary(self):
        self.clear_hooks()
        return dict(groups={k:dict(steps=v['steps'],opposed=v['opposed'],mean_cosine=v['cosine_sum']/v['cosine_valid'] if v['cosine_valid'] else None,mean_alpha=v['alpha_sum']/v['steps'] if v['steps'] else None) for k,v in self.stats.items()},adam_audits=self.audits)
