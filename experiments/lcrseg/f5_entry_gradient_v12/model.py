"""Current-U entry-reference half-space guard on V10 merged gradients."""
from pathlib import Path
import torch
from ..f5_anchor_guard_v10.model import AnchorGuardModel,AnchorGuardTrainer,guarded_mask
from ..five_frameworks_v1.gate import digest

def halfspace(merged,reference):
    if len(merged)!=len(reference) or any(a.shape!=b.shape for a,b in zip(merged,reference)):
        raise ValueError('merged/reference parameter grids must match')
    if not merged:return merged,dict(projected=False,dot_before=0.,dot_after=0.,reference_norm=0.)
    if any(not torch.isfinite(x).all() for x in merged+reference):raise FloatingPointError('nonfinite entry gradient')
    dot=sum((a.double()*b.double()).sum() for a,b in zip(merged,reference))
    norm=sum(b.double().square().sum() for b in reference)
    if not torch.isfinite(dot) or not torch.isfinite(norm):raise FloatingPointError('overflow entry gradient dot/norm')
    coefficient=dot/norm if dot<0 and norm>0 else dot.new_zeros(())
    out=[(a.double()-coefficient*b.double()).to(a.dtype) for a,b in zip(merged,reference)]
    after=sum((a.double()*b.double()).sum() for a,b in zip(out,reference))
    return out,dict(projected=bool(dot<0 and norm>0),dot_before=float(dot),dot_after=float(after),reference_norm=float(norm.sqrt()))

class EntryGradientModel(AnchorGuardModel):
    def __init__(self,*args,entry_gradient=True,**kwargs):
        super().__init__(*args,anchor_guard=True,**kwargs)
        self.entry_gradient=entry_gradient

class EntryGradientTrainer(AnchorGuardTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.entry_reference_loss=None;self.entry_reference_grads=None
        self.telemetry.update(entry_reference_vjps=0,entry_guard_calls=0,entry_projected_calls=0,entry_reference_pixels=0)

    def loss_contract(self):
        return {**super().loss_contract(),'entry_gradient':self.model.entry_gradient,
                'projection':'merged A/B/R L+U raw gradient -> dot>=0 against confident original-PAS frozen-entry KL reference; zero-reference no-op; Adam unconstrained',
                'entry_gradient_implementation':digest(Path(__file__).read_text())}

    def entry_reference_objective(self,logits,anchor,support):
        return self.fine_kl(logits,anchor,support)

    def kl_objective(self,images,logits,target,valid):
        if not self.model.entry_gradient:return super().kl_objective(images,logits,target,valid)
        with torch.no_grad():
            anchor=self.model.entry_anchor(images).softmax(1)
            kept=guarded_mask(anchor,target,valid,self.options['PAS_confidence'])
            support=valid & (anchor.max(1).values>=self.options['PAS_confidence'])
        self.entry_reference_loss=self.entry_reference_objective(logits,anchor,support)
        count=int((valid & ~kept).sum())
        self.telemetry['teacher_full_forwards']+=1;self.telemetry['anchor_full_forwards']+=1
        self.telemetry['anchor_candidate_pixels']+=int(valid.sum());self.telemetry['anchor_rejected_pixels']+=count
        self.telemetry['anchor_gate_calls']+=1;self.telemetry['anchor_rejection_calls']+=int(count>0)
        self.telemetry['entry_reference_pixels']+=int(support.sum())
        self.last['anchor_gate']=dict(candidate_pixels=int(valid.sum()),rejected_pixels=count,entry_reference_pixels=int(support.sum()))
        return self.fine_kl(logits,target,kept)

    def gradient_audit(self,labeled,unlabeled):
        super().gradient_audit(labeled,unlabeled)
        self.entry_reference_grads=None
        if self.model.entry_gradient and unlabeled is not None:
            params=[p for p in self.model.parameters() if p.requires_grad]
            self.entry_reference_grads=torch.autograd.grad(self.entry_reference_loss,params,allow_unused=True,retain_graph=True)
            self.entry_reference_loss=None
            self.telemetry['entry_reference_vjps']+=1

    def project_entry_gradients(self,merged,reference,grads,params):
        return halfspace(merged,reference)

    def transform_gradients(self,grads):
        super().transform_gradients(grads)
        stats=dict(projected=False,dot_before=0.,dot_after=0.,reference_norm=0.)
        if self.entry_reference_grads is not None:
            params=[p for p in self.model.parameters() if p.requires_grad]
            merged=[torch.zeros_like(p) if p.grad is None else p.grad for p in params]
            reference=[torch.zeros_like(p) if g is None else g for p,g in zip(params,self.entry_reference_grads)]
            projected,stats=self.project_entry_gradients(merged,reference,grads,params)
            for p,g in zip(params,projected):p.grad=g
            self.telemetry['entry_guard_calls']+=1;self.telemetry['entry_projected_calls']+=int(stats['projected'])
            self.entry_reference_grads=None
        self.module_stats.update(entry_gradient=self.model.entry_gradient,**{'entry_'+k:v for k,v in stats.items()})
