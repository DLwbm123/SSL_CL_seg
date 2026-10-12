"""SARD spatial importance on the V13 class-balanced entry reference only."""
from pathlib import Path
import torch
import torch.nn.functional as F
from ..f5_class_reference_v13.model import ClassReferenceModel,ClassReferenceTrainer
from ..f5_anchor_guard_v10.model import guarded_mask
from ..five_frameworks_v1.gate import digest
from ..five_frameworks_v1.kernels import _finite,_prob


@torch.no_grad()
def structure_score(features):
    if features.ndim!=4 or not features.is_floating_point() or min(features.shape)<1:
        raise ValueError('expected nonempty BCHW floating teacher features')
    if not torch.isfinite(features).all():raise FloatingPointError('nonfinite teacher features')
    f=features.detach();channels=f.shape[1]
    # Replicate borders are an explicit adaptation choice; Sobel coefficients are unscaled.
    k=f.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3)
    padded=F.pad(f,(1,1,1,1),mode='replicate')
    dx=F.conv2d(padded,k.expand(channels,1,3,3),groups=channels)
    dy=F.conv2d(padded,k.transpose(-1,-2).expand(channels,1,3,3),groups=channels)
    a=dx.square().sum(1);b=(dx*dy).sum(1);c=dy.square().sum(1)
    edge=((a-c).square()+4*b.square()).clamp_min(0).sqrt()
    corner=(a*c-b.square()).clamp_min(0).sqrt()
    padded=F.pad(f,(7,7,7,7),mode='replicate')
    mean=F.avg_pool2d(padded,15,stride=1)
    density=torch.zeros_like(edge)
    # Stream the exact 225 Euclidean distances instead of materializing a BCHW225 tensor.
    for y in range(15):
        for x in range(15):
            density.add_(torch.linalg.vector_norm(padded[:,:,y:y+f.shape[2],x:x+f.shape[3]]-mean,dim=1))
    score=2*edge+corner+density/225
    if not torch.isfinite(score).all():raise FloatingPointError('nonfinite structure score')
    return score


def structured_reference(logits,target,support,score):
    if logits.shape!=target.shape or support.shape!=logits.shape[:1]+logits.shape[2:] or score.shape!=support.shape:
        raise ValueError('reference logits, probabilities, support and spatial score must align')
    if support.dtype!=torch.bool:raise TypeError('reference support must be boolean')
    if not torch.isfinite(score).all() or (score<0).any():raise ValueError('structure weights must be finite and nonnegative')
    _finite(logits,'logits');_prob(target,'target')
    q=target.detach().to(logits);s=score.detach().to(logits);labels=q.argmax(1)
    if not torch.allclose(q.sum(1),torch.ones_like(q[:,0]),atol=1e-5,rtol=1e-5):
        raise ValueError('target must sum to one across classes')
    per_pixel=(torch.xlogy(q,q)-q*F.log_softmax(logits,dim=1)).sum(1)
    counts=[];terms=[];fallback=0;concentrations=[]
    for c in range(logits.shape[1]):
        selected=support & (labels==c);n=int(selected.sum());counts.append(n)
        if not n:continue
        weights=s[selected];mass=weights.sum()
        if mass>0:
            weights=weights/mass;terms.append((per_pixel[selected]*weights).sum())
            concentrations.append(float(weights.square().sum()*n))
        else:
            terms.append(per_pixel[selected].mean());fallback+=1;concentrations.append(1.)
    value=torch.stack(terms).mean() if terms else logits.sum()*0
    return value,dict(class_pixels=counts,uniform_fallback_classes=fallback,
                      weight_concentration=concentrations)


class SARDModel(ClassReferenceModel):
    def __init__(self,*args,sard=True,**kwargs):
        super().__init__(*args,class_reference=True,**kwargs)
        self.sard=sard


class SARDTrainer(ClassReferenceTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.telemetry.update(sard_calls=0,sard_nonuniform_calls=0,sard_uniform_fallback_classes=0,
                              sard_score_pixels=0,sard_neighbor_vector_distances=0)

    def loss_contract(self):
        return {**super().loss_contract(),'sard':self.model.sard,
                'sard_formula':'detached post-prefix entry features; unscaled Sobel/replicate; E=lambda1-lambda2,C=sqrt(det J),D=mean225 Euclidean distance to replicate15x15 mean; S=2E+C+D; per-present-class normalized S-weighted entry KL; zero-mass class uniform fallback',
                'sard_scope':'reference only; original main losses, V13 raw-gradient halfspace, Adam and deployment unchanged; no feature matching or denoising head',
                'sard_implementation':digest(Path(__file__).read_text())}

    def kl_objective(self,images,logits,target,valid):
        if not self.model.sard:return super().kl_objective(images,logits,target,valid)
        with torch.no_grad():
            entry=self.model.entry_anchor
            features=entry.parent.features(images,'eval')
            features=torch.einsum('ij,bjhw->bihw',entry.transform,features)
            anchor=entry.parent.native_readout(features,images.shape[-2:],'eval').softmax(1)
            score=structure_score(features)
            kept=guarded_mask(anchor,target,valid,self.options['PAS_confidence'])
            support=valid & (anchor.max(1).values>=self.options['PAS_confidence'])
        self.entry_reference_loss,stats=structured_reference(logits,anchor,support,score)
        counts=stats['class_pixels']
        for name,count in zip(('background','rim','cup'),counts):self.telemetry['reference_'+name+'_pixels']+=count
        self.telemetry['reference_balanced_calls']+=1
        self.telemetry['reference_present_class_calls']+=sum(n>0 for n in counts)
        self.telemetry['sard_calls']+=1
        self.telemetry['sard_score_pixels']+=score.numel()
        self.telemetry['sard_neighbor_vector_distances']+=225*score.numel()
        self.telemetry['sard_nonuniform_calls']+=int(any(v>1+1e-6 for v in stats['weight_concentration']))
        self.telemetry['sard_uniform_fallback_classes']+=stats['uniform_fallback_classes']
        self.last['reference_class_pixels']=counts;self.last['sard']=stats
        count=int((valid & ~kept).sum())
        self.telemetry['teacher_full_forwards']+=1;self.telemetry['anchor_full_forwards']+=1
        self.telemetry['anchor_candidate_pixels']+=int(valid.sum());self.telemetry['anchor_rejected_pixels']+=count
        self.telemetry['anchor_gate_calls']+=1;self.telemetry['anchor_rejection_calls']+=int(count>0)
        self.telemetry['entry_reference_pixels']+=int(support.sum())
        self.last['anchor_gate']=dict(candidate_pixels=int(valid.sum()),rejected_pixels=count,entry_reference_pixels=int(support.sum()))
        return self.fine_kl(logits,target,kept)

    def transform_gradients(self,grads):
        super().transform_gradients(grads)
        self.module_stats.update(sard=self.model.sard,**self.last.get('sard',{}))
