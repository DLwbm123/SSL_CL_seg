"""Present-class balanced entry reference; retain the V10/V12 training routes."""
from pathlib import Path
import torch
from ..f5_entry_gradient_v12.model import EntryGradientModel,EntryGradientTrainer
from ..five_frameworks_v1.kernels import masked_kl
from ..five_frameworks_v1.gate import digest

def class_reference(logits,target,valid):
    labels=target.detach().argmax(1)
    counts=[int((valid & (labels==c)).sum()) for c in range(logits.shape[1])]
    terms=[masked_kl(logits,target,valid & (labels==c)) for c,n in enumerate(counts) if n]
    value=torch.stack(terms).mean() if terms else masked_kl(logits,target,valid)
    return value,counts

class ClassReferenceModel(EntryGradientModel):
    def __init__(self,*args,class_reference=True,**kwargs):
        super().__init__(*args,entry_gradient=True,**kwargs)
        self.class_reference=class_reference

class ClassReferenceTrainer(EntryGradientTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.telemetry.update(reference_background_pixels=0,reference_rim_pixels=0,reference_cup_pixels=0,reference_balanced_calls=0,reference_present_class_calls=0)

    def loss_contract(self):
        return {**super().loss_contract(),'class_reference':self.model.class_reference,
                'reference_reduction':'equal mean of per-entry-argmax-class pixel mean KL over present confident original-PAS classes; absent classes omitted',
                'class_reference_implementation':digest(Path(__file__).read_text())}

    def entry_reference_objective(self,logits,anchor,support):
        if not self.model.class_reference:return super().entry_reference_objective(logits,anchor,support)
        value,counts=class_reference(logits,anchor,support)
        for name,count in zip(('background','rim','cup'),counts):self.telemetry['reference_'+name+'_pixels']+=count
        self.telemetry['reference_balanced_calls']+=1
        self.telemetry['reference_present_class_calls']+=sum(n>0 for n in counts)
        self.last['reference_class_pixels']=counts
        return value

    def transform_gradients(self,grads):
        super().transform_gradients(grads)
        self.module_stats.update(class_reference=self.model.class_reference,reference_class_pixels=self.last.get('reference_class_pixels',[0,0,0]))
