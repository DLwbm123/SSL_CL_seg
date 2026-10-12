"""CaRE hard entropy gate on the V13 class-balanced entry reference."""
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from ..f5_class_reference_v13.model import ClassReferenceModel,ClassReferenceTrainer
from ..f5_anchor_guard_v10.model import guarded_mask
from ..five_frameworks_v1.gate import digest
from ..five_frameworks_v1.kernels import _finite,_mask


def care_reference(logits,teacher_logits,support):
    if logits.shape!=teacher_logits.shape or logits.ndim!=4 or logits.shape[1]<2:
        raise ValueError('aligned BCHW logits with at least two classes required')
    _finite(logits,'student logits');_finite(teacher_logits,'teacher logits');_mask(support,logits)
    logq=F.log_softmax(logits,dim=1);q=logq.exp()
    with torch.no_grad():
        logp=F.log_softmax(teacher_logits.detach().to(logits),dim=1);p=logp.exp()
        hp=-(p*logp).sum(1);hq=-(q.detach()*logq.detach()).sum(1)
        forward=hp<=hq
        labels=p.argmax(1)
        floor=math.log(torch.finfo(logits.dtype).tiny)
        bounded_logp=logp.clamp_min(floor)
    fkl=(p*(logp-logq)).sum(1)
    rkl=(q*(logq-bounded_logp)).sum(1)
    pixel=torch.where(forward,fkl,rkl)
    counts=[];terms=[]
    for c in range(logits.shape[1]):
        selected=support & (labels==c);n=int(selected.sum());counts.append(n)
        if n:terms.append(pixel[selected].mean())
    value=torch.stack(terms).mean() if terms else logits.sum()*0
    _finite(value,'CaRE reference')
    return value,dict(class_pixels=counts,forward_pixels=int((support & forward).sum()),
                      reverse_pixels=int((support & ~forward).sum()),
                      teacher_log_floor_values=int(((logp<floor) & support[:,None]).sum()))


class CaREModel(ClassReferenceModel):
    def __init__(self,*args,care=True,**kwargs):
        super().__init__(*args,class_reference=True,**kwargs);self.care=care


class CaRETrainer(ClassReferenceTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.telemetry.update(care_calls=0,care_forward_pixels=0,care_reverse_pixels=0,
                              care_reverse_calls=0,care_teacher_log_floor_values=0)

    def loss_contract(self):
        return {**super().loss_contract(),'care':self.model.care,
                'care_formula':'p=own entry softmax,q=student softmax,T1; detached gate H(p)<=H(q) selects KL(p||q), else KL(q||p); reverse teacher log floor log(dtype.tiny); original support and equal present-class means',
                'care_scope':'reference-only hard CaRE divergence; no Revival/BALD/soft gate; V13 raw halfspace and main losses unchanged',
                'care_implementation':digest(Path(__file__).read_text())}

    def kl_objective(self,images,logits,target,valid):
        if not self.model.care:return super().kl_objective(images,logits,target,valid)
        with torch.no_grad():
            entry_logits=self.model.entry_anchor(images);anchor=entry_logits.softmax(1)
            kept=guarded_mask(anchor,target,valid,self.options['PAS_confidence'])
            support=valid & (anchor.max(1).values>=self.options['PAS_confidence'])
        self.entry_reference_loss,stats=care_reference(logits,entry_logits,support)
        counts=stats['class_pixels']
        for name,count in zip(('background','rim','cup'),counts):self.telemetry['reference_'+name+'_pixels']+=count
        self.telemetry['reference_balanced_calls']+=1
        self.telemetry['reference_present_class_calls']+=sum(n>0 for n in counts)
        self.telemetry['care_calls']+=1
        for key in ('forward_pixels','reverse_pixels','teacher_log_floor_values'):self.telemetry['care_'+key]+=stats[key]
        self.telemetry['care_reverse_calls']+=int(stats['reverse_pixels']>0)
        self.last['reference_class_pixels']=counts;self.last['care']=stats
        count=int((valid & ~kept).sum())
        self.telemetry['teacher_full_forwards']+=1;self.telemetry['anchor_full_forwards']+=1
        self.telemetry['anchor_candidate_pixels']+=int(valid.sum());self.telemetry['anchor_rejected_pixels']+=count
        self.telemetry['anchor_gate_calls']+=1;self.telemetry['anchor_rejection_calls']+=int(count>0)
        self.telemetry['entry_reference_pixels']+=int(support.sum())
        self.last['anchor_gate']=dict(candidate_pixels=int(valid.sum()),rejected_pixels=count,entry_reference_pixels=int(support.sum()))
        return self.fine_kl(logits,target,kept)

    def transform_gradients(self,grads):
        super().transform_gradients(grads)
        self.module_stats.update(care=self.model.care,**self.last.get('care',{}))
