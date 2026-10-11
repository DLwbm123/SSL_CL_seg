"""Restore rejected entry knowledge through the same KL objective."""
from pathlib import Path
import torch
from ..f5_anchor_guard_v10.model import AnchorGuardModel,AnchorGuardTrainer,guarded_mask
from ..five_frameworks_v1.gate import digest

def routed_target(anchor,target,valid,threshold):
    kept=guarded_mask(anchor,target,valid,threshold)
    restored=valid & ~kept
    return torch.where(restored[:,None],anchor.detach(),target.detach()),restored

class AnchorRestoreModel(AnchorGuardModel):
    def __init__(self,*args,anchor_restore=True,**kwargs):
        super().__init__(*args,anchor_guard=True,**kwargs)
        self.anchor_restore=anchor_restore

class AnchorRestoreTrainer(AnchorGuardTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.telemetry.update(anchor_restored_pixels=0,anchor_restore_calls=0)

    def loss_contract(self):
        return {**super().loss_contract(),'anchor_restore':self.model.anchor_restore,
                'KL_target_routing':'confident entry/EMA conflict on original PAS support -> entry soft target; otherwise EMA; average original PAS support',
                'restore_implementation':digest(Path(__file__).read_text())}

    def kl_objective(self,images,logits,target,valid):
        if not self.model.anchor_restore:return super().kl_objective(images,logits,target,valid)
        with torch.no_grad():
            anchor=self.model.entry_anchor(images).softmax(1)
            routed,restored=routed_target(anchor,target,valid,self.options['PAS_confidence'])
        candidates=int(valid.sum());count=int(restored.sum())
        self.telemetry['teacher_full_forwards']+=1
        self.telemetry['anchor_full_forwards']+=1
        self.telemetry['anchor_candidate_pixels']+=candidates
        self.telemetry['anchor_rejected_pixels']+=count
        self.telemetry['anchor_gate_calls']+=1
        self.telemetry['anchor_rejection_calls']+=int(count>0)
        self.telemetry['anchor_restored_pixels']+=count
        self.telemetry['anchor_restore_calls']+=int(count>0)
        self.last['anchor_gate']=dict(candidate_pixels=candidates,rejected_pixels=count,restored_pixels=count)
        return self.fine_kl(logits,routed,valid)

    def transform_gradients(self,grads):
        super().transform_gradients(grads)
        self.module_stats.update(anchor_restore=self.model.anchor_restore)
