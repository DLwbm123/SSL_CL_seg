"""Frozen stage-entry prediction guard on the positive joint-KL F5 base."""
import copy
from pathlib import Path
import torch
from ..f5_joint_kl_bridge_v9.model import JointKLModel,JointKLTrainer
from ..five_frameworks_v1.model import Deployment
from ..five_frameworks_v1.semantics import tensor_fingerprint
from ..five_frameworks_v1.gate import digest

def guarded_mask(anchor,target,valid,threshold):
    if anchor.shape!=target.shape or valid.shape!=target.shape[:1]+target.shape[2:]:
        raise ValueError('anchor/EMA/PAS grids must match')
    if not 0<=threshold<=1:raise ValueError('invalid anchor confidence threshold')
    if not torch.isfinite(anchor).all() or not torch.isfinite(target).all():raise FloatingPointError('nonfinite anchor/EMA probabilities')
    confident=anchor.detach().max(1).values>=threshold
    conflict=anchor.detach().argmax(1)!=target.detach().argmax(1)
    return valid & ~(confident & conflict)

class AnchorGuardModel(JointKLModel):
    def __init__(self,*args,anchor_guard=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.anchor_guard=anchor_guard
        parent=copy.deepcopy(self.parent);parent.stage_exit()
        self.entry_anchor=Deployment(parent,self.sidecar.previous).eval().requires_grad_(False)

    def configure_training(self):
        super().configure_training()
        self.entry_anchor.eval().requires_grad_(False)

    def teacher(self):
        teacher=super().teacher()
        teacher.entry_anchor=self.entry_anchor
        return teacher

class AnchorGuardTrainer(JointKLTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.anchor_fingerprint=tensor_fingerprint(self.model.entry_anchor.state_dict())
        self.telemetry.update(anchor_full_forwards=0,anchor_candidate_pixels=0,anchor_rejected_pixels=0,anchor_gate_calls=0,anchor_rejection_calls=0)

    def loss_contract(self):
        return {**super().loss_contract(), 'anchor_guard':self.model.anchor_guard,
                'anchor_source':self.model.parent.stage_source,'anchor_entry_fingerprint':self.anchor_fingerprint,
                'anchor_threshold':self.options['PAS_confidence'],
                'gate':'KL PAS pixels rejected only if confident immutable entry teacher disagrees with EMA class; SWD unchanged',
                'anchor_implementation':digest(Path(__file__).read_text())}

    @torch.no_grad()
    def kl_admission(self,images,target,valid):
        if not self.model.anchor_guard:return valid
        anchor=self.model.entry_anchor(images).softmax(1)
        kept=guarded_mask(anchor,target,valid,self.options['PAS_confidence'])
        candidates=int(valid.sum());rejected=int((valid & ~kept).sum())
        self.telemetry['teacher_full_forwards']+=1
        self.telemetry['anchor_full_forwards']+=1
        self.telemetry['anchor_candidate_pixels']+=candidates
        self.telemetry['anchor_rejected_pixels']+=rejected
        self.telemetry['anchor_gate_calls']+=1
        self.telemetry['anchor_rejection_calls']+=int(rejected>0)
        self.last['anchor_gate']=dict(candidate_pixels=candidates,rejected_pixels=rejected)
        return kept

    def transform_gradients(self,grads):
        super().transform_gradients(grads)
        if self.model.entry_anchor.training or any(p.requires_grad or p.grad is not None for p in self.model.entry_anchor.parameters()):
            raise PermissionError('immutable entry anchor must remain frozen and in eval mode')
        self.module_stats.update(anchor_guard=self.model.anchor_guard,**self.last.get('anchor_gate',{}))
