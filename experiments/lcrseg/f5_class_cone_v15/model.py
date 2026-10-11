"""Separate present-class entry gradient constraints on the V13 base."""
from itertools import combinations
from pathlib import Path
import torch
from ..f5_class_reference_v13.model import ClassReferenceModel,ClassReferenceTrainer
from ..f5_anchor_guard_v10.model import AnchorGuardTrainer
from ..five_frameworks_v1.kernels import masked_kl
from ..five_frameworks_v1.gate import digest

def class_cone(merged,references):
    if len(references)>3 or any(len(r)!=len(merged) or any(a.shape!=b.shape for a,b in zip(merged,r)) for r in references):
        raise ValueError('present-class reference grids must match, at most3')
    if any(not torch.isfinite(x).all() for x in merged+[x for r in references for x in r]):raise FloatingPointError('nonfinite class cone gradient')
    def dot(a,b):return sum((x.double()*y.double()).sum() for x,y in zip(a,b))
    if not merged:return merged,dict(projected=False,dots_before=[],dots_after=[],active_constraints=0,aggregate_safe_class_conflict=False)
    norms=[dot(r,r).sqrt() for r in references]
    if any(not torch.isfinite(n) for n in norms):raise FloatingPointError('overflow class reference norm')
    refs=[[x.double()/n for x in r] for r,n in zip(references,norms) if n>0]
    norm=dot(merged,merged).sqrt()
    if not torch.isfinite(norm):raise FloatingPointError('overflow merged gradient norm')
    if not refs:return merged,dict(projected=False,dots_before=[0.]*len(references),dots_after=[0.]*len(references),active_constraints=0,aggregate_safe_class_conflict=False)
    dots=torch.stack([dot(merged,r) for r in refs]);gram=torch.stack([torch.stack([dot(a,b) for b in refs]) for a in refs])
    tolerance=1e-10*max(float(norm),1.);candidates=[dots.new_zeros(len(refs))]
    for size in range(1,len(refs)+1):
        for active in combinations(range(len(refs)),size):
            ix=torch.tensor(active,device=dots.device);block=gram[ix][:,ix]
            value=-(torch.linalg.pinv(block,hermitian=True)@dots[ix])
            if torch.isfinite(value).all() and (value>=-tolerance).all():
                coef=dots.new_zeros(len(refs));coef[ix]=value.clamp_min(0);candidates.append(coef)
    feasible=[c for c in candidates if (dots+gram@c>=-tolerance).all()]
    if not feasible:raise FloatingPointError('class cone lost numerical feasibility')
    coefficients=min(feasible,key=lambda c:float(c@gram@c))
    out=[(a.double()+sum((c*r[i] for c,r in zip(coefficients,refs)),torch.zeros_like(a,dtype=torch.double))).to(a.dtype) for i,a in enumerate(merged)]
    before=[float(dot(merged,r)) for r in references];after=[float(dot(out,r)) for r in references]
    return out,dict(projected=bool((coefficients>0).any()),dots_before=before,dots_after=after,active_constraints=int((coefficients>0).sum()),aggregate_safe_class_conflict=sum(before)>=0 and any(x<0 for x in before))

class ClassConeModel(ClassReferenceModel):
    def __init__(self,*args,class_cone=True,**kwargs):
        super().__init__(*args,class_reference=True,**kwargs);self.class_cone=class_cone

class ClassConeTrainer(ClassReferenceTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.entry_class_losses=None;self.entry_class_grads=None;self.entry_class_ids=[]
        self.telemetry.update(class_cone_calls=0,class_cone_projected_calls=0,class_cone_multiple_active_calls=0,class_cone_aggregate_safe_conflict_calls=0)

    def loss_contract(self):
        return {**super().loss_contract(),'class_cone':self.model.class_cone,'class_cone_formula':'nearest merged A/B/R raw gradient constrained against each present entry class KL gradient; mean used for diagnostic compatibility only; Adam unconstrained','class_cone_implementation':digest(Path(__file__).read_text())}

    def entry_reference_objective(self,logits,anchor,support):
        if not self.model.class_cone:return super().entry_reference_objective(logits,anchor,support)
        labels=anchor.detach().argmax(1);counts=[int((support & (labels==c)).sum()) for c in range(logits.shape[1])]
        self.entry_class_ids=[c for c,n in enumerate(counts) if n]
        self.entry_class_losses=[masked_kl(logits,anchor,support & (labels==c)) for c in self.entry_class_ids]
        for name,n in zip(('background','rim','cup'),counts):self.telemetry['reference_'+name+'_pixels']+=n
        self.telemetry['reference_balanced_calls']+=1;self.telemetry['reference_present_class_calls']+=len(self.entry_class_ids);self.last['reference_class_pixels']=counts
        return torch.stack(self.entry_class_losses).mean() if self.entry_class_losses else masked_kl(logits,anchor,support)

    def gradient_audit(self,labeled,unlabeled):
        if not self.model.class_cone:return super().gradient_audit(labeled,unlabeled)
        AnchorGuardTrainer.gradient_audit(self,labeled,unlabeled);self.entry_reference_grads=None;self.entry_class_grads=None
        if self.model.entry_gradient and unlabeled is not None:
            params=[p for p in self.model.parameters() if p.requires_grad]
            self.entry_class_grads=[torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True) for loss in self.entry_class_losses]
            self.telemetry['entry_reference_vjps']+=len(self.entry_class_grads)
            self.entry_reference_grads=tuple(sum((torch.zeros_like(p) if r[i] is None else r[i] for r in self.entry_class_grads),torch.zeros_like(p))/max(len(self.entry_class_grads),1) for i,p in enumerate(params))
            self.entry_reference_loss=None;self.entry_class_losses=None

    def project_entry_gradients(self,merged,reference,grads,params):
        if not self.model.class_cone:return super().project_entry_gradients(merged,reference,grads,params)
        refs=[[torch.zeros_like(p) if g is None else g for p,g in zip(params,r)] for r in self.entry_class_grads]
        out,stats=class_cone(merged,refs);self.entry_class_grads=None
        self.telemetry['class_cone_calls']+=1;self.telemetry['class_cone_projected_calls']+=int(stats['projected']);self.telemetry['class_cone_multiple_active_calls']+=int(stats['active_constraints']>1);self.telemetry['class_cone_aggregate_safe_conflict_calls']+=int(stats['aggregate_safe_class_conflict'])
        stats.update(dot_before=float(sum((a.double()*b.double()).sum() for a,b in zip(merged,reference))),dot_after=float(sum((a.double()*b.double()).sum() for a,b in zip(out,reference))),reference_norm=float(sum(x.double().square().sum() for x in reference).sqrt()),class_ids=self.entry_class_ids)
        return out,stats

    def transform_gradients(self,grads):
        super().transform_gradients(grads);self.module_stats.update(class_cone=self.model.class_cone)
