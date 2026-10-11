"""Nearest merged gradient in the entry and current-L half-space intersection."""
from pathlib import Path
import torch
from ..f5_class_reference_v13.model import ClassReferenceModel,ClassReferenceTrainer
from ..five_frameworks_v1.gate import digest

def dual_halfspace(merged,entry,labeled):
    if len(merged)!=len(entry) or len(merged)!=len(labeled) or any(a.shape!=b.shape or a.shape!=c.shape for a,b,c in zip(merged,entry,labeled)):
        raise ValueError('dual reference parameter grids must match')
    if not merged:return merged,dict(projected=False,dot_before=0.,dot_after=0.,reference_norm=0.,L_dot_before=0.,L_dot_after=0.,V13_L_dot_after=0.,active_constraints=0)
    if any(not torch.isfinite(x).all() for x in merged+entry+labeled):raise FloatingPointError('nonfinite dual reference gradient')
    def dot(a,b):return sum((x.double()*y.double()).sum() for x,y in zip(a,b))
    norms=[dot(r,r).sqrt() for r in (entry,labeled)]
    if any(not torch.isfinite(n) for n in norms):raise FloatingPointError('overflow dual reference norm')
    refs=[[x.double()/n for x in r] for r,n in zip((entry,labeled),norms) if n>0]
    gnorm=dot(merged,merged).sqrt()
    if not torch.isfinite(gnorm):raise FloatingPointError('overflow merged gradient norm')
    tolerance=1e-10*max(float(gnorm),1.)
    dots=torch.stack([dot(merged,r) for r in refs]) if refs else gnorm.new_empty(0)
    gram=torch.stack([torch.stack([dot(a,b) for b in refs]) for a in refs]) if refs else gnorm.new_empty((0,0))
    candidates=[dots.new_zeros(len(refs))]
    for i in range(len(refs)):
        c=dots.new_zeros(len(refs));c[i]=(-dots[i]/gram[i,i]).clamp_min(0);candidates.append(c)
    if len(refs)==2:
        determinant=gram[0,0]*gram[1,1]-gram[0,1].square()
        if determinant>0:
            c=torch.stack([(-gram[1,1]*dots[0]+gram[0,1]*dots[1])/determinant,(gram[0,1]*dots[0]-gram[0,0]*dots[1])/determinant])
            if torch.isfinite(c).all() and (c>=0).all():candidates.append(c)
    feasible=[c for c in candidates if (dots+gram@c>=-tolerance).all()]
    if not feasible:raise FloatingPointError('dual reference intersection solve lost numerical feasibility')
    coefficients=min(feasible,key=lambda c:float(c@gram@c))
    out=[(a.double()+sum((c*r[i] for c,r in zip(coefficients,refs)),torch.zeros_like(a,dtype=torch.double))).to(a.dtype) for i,a in enumerate(merged)]
    entry_before=dot(merged,entry);entry_square=norms[0].square()
    old_L_dot=dot(merged,labeled)-(entry_before/entry_square*dot(entry,labeled) if entry_before<0 and entry_square>0 else 0.)
    return out,dict(projected=bool((coefficients>0).any()),dot_before=float(dot(merged,entry)),dot_after=float(dot(out,entry)),reference_norm=float(norms[0]),L_dot_before=float(dot(merged,labeled)),L_dot_after=float(dot(out,labeled)),V13_L_dot_after=float(old_L_dot),active_constraints=int((coefficients>0).sum()))

class DualReferenceModel(ClassReferenceModel):
    def __init__(self,*args,dual_reference=True,**kwargs):
        super().__init__(*args,class_reference=True,**kwargs)
        self.dual_reference=dual_reference

class DualReferenceTrainer(ClassReferenceTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.telemetry.update(dual_guard_calls=0,dual_projected_calls=0,dual_both_active_calls=0,V13_negative_L_direction_calls=0)

    def loss_contract(self):
        return {**super().loss_contract(),'dual_reference':self.model.dual_reference,'dual_projection':'nearest raw merged A/B/R gradient with dot(entry)>=0 and dot(current supervised L)>=0; no new loss/VJP; Adam unconstrained','dual_reference_implementation':digest(Path(__file__).read_text())}

    def project_entry_gradients(self,merged,reference,grads,params):
        if not self.model.dual_reference:return super().project_entry_gradients(merged,reference,grads,params)
        labeled=[torch.zeros_like(p) if grads[id(p)]['L'] is None else grads[id(p)]['L'] for p in params]
        out,stats=dual_halfspace(merged,reference,labeled)
        self.telemetry['dual_guard_calls']+=1;self.telemetry['dual_projected_calls']+=int(stats['projected'])
        self.telemetry['dual_both_active_calls']+=int(stats['active_constraints']==2)
        self.telemetry['V13_negative_L_direction_calls']+=int(stats['V13_L_dot_after']<0)
        return out,stats

    def transform_gradients(self,grads):
        super().transform_gradients(grads)
        self.module_stats.update(dual_reference=self.model.dual_reference)
