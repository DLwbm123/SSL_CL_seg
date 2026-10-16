"""Reuse MRI stem/forward/merge semantics, replace only the sidecar operator."""
from ..main_style_ssl_v2.model import Parent, MainModel as BaseModel
from .protection import OrthogonalSidecar
import copy
from pathlib import Path
import torch
from ..main_style_ssl_v1.model import MainTrainer as V1Trainer, weighted_kl
from ..main_style_ssl_v2.model import independent_reference
from ..five_frameworks_v1.gate import digest
from .memory import History
from .protocol import VERSION, MEMORY_WEIGHT
from .protection import elastic_displacement, spectrum


class MainModel(BaseModel):
    def __init__(self, parent, parent_map=(0,1,1,1), previous=None, initialize_basis=True):
        if tuple(parent_map)!=(0,1,1,1) or not parent.mri:
            raise ValueError('this candidate is M&Ms only')
        super().__init__(parent,parent_map,previous,initialize_basis)
        if self.sidecar is not None:
            self.sidecar=OrthogonalSidecar(self.sidecar.q,self.sidecar.previous)


class MainTrainer(V1Trainer):
    """Candidate hooks on the existing StageTrainer update/commit loop."""
    def __init__(self,*args,history=(),**kwargs):
        super().__init__(*args,**kwargs)
        self.history_lineage=[s['domain'] for s in history]
        self.history=History(history,self.history_lineage)
        self.history.terms(self.model.native_values())
        self.precision_counts=torch.stack((torch.ones(4),torch.full((4,),2.))).double()
        self.guard_totals=dict(calls=0,zero_displacement=0,corrected=0,residual_conflicts=0)
        self.protect_loss=None;self.entry_grads=None;self.labeled_grads=None;self.calibration=None

    def loss_contract(self):
        return dict(version=VERSION,history_lineage=self.history_lineage,
                    summary='fixed final native/F per-domain/class',memory_weight=MEMORY_WEIGHT,
                    reference='independent entry geometry/confidence; training-L precision',
                    implementation=digest({p.name:p.read_text(encoding='utf-8') for p in sorted(Path(__file__).parent.glob('*.py'))}))

    def losses(self):
        self.protect_loss=None;self.current_l_images=None;self.calibration=None
        return super().losses()

    def student_images(self,images,donor):
        if self.current_l_images is None:self.current_l_images=images.detach()
        return super().student_images(images,donor)

    def labeled_addition(self,logp,labels,mix_mask):
        hierarchy=super().labeled_addition(logp,labels,mix_mask)
        penalty=self.history.penalty(self.model.native_values())
        self.proxy_before={k:float(v.detach()) for k,v in self.history.terms(self.model.native_values()).items()}
        if self.step%100==0:
            with torch.no_grad():prediction=self.model.entry_anchor(self.current_l_images).argmax(1)
            valid=labels!=255
            counts=torch.stack([((prediction==c)&valid).sum() for c in range(4)]).cpu().double()
            correct=torch.stack([((prediction==c)&(labels==c)).sum() for c in range(4)]).cpu().double()
            self.calibration=(counts,correct);self.telemetry['teacher_full_forwards']+=1
        return (penalty*0 if hierarchy is None else hierarchy)+MEMORY_WEIGHT*penalty

    def pseudo_admission(self,target,geometry,pas_valid):
        self.u_geometry=geometry.detach()
        return super().pseudo_admission(target,geometry,pas_valid)

    def kl_objective(self,images,logits,target,valid):
        with torch.no_grad():q=self.model.entry_anchor(images).softmax(1)
        self.telemetry['teacher_full_forwards']+=1
        reliability=(self.precision_counts[0]/self.precision_counts[1]).to(logits)
        loss,_,stats=weighted_kl(logits,target,valid,q)
        self.protect_loss,counts,support=independent_reference(logits,q,self.u_geometry,reliability,torch.ones_like(self.u_geometry,dtype=logits.dtype))
        stats.update(reference_class_pixels=counts,entry_training_L_precision=reliability.tolist(),
                     reference_conflict_pixels=int((support&(q.argmax(1)!=target.argmax(1))).sum()),
                     no_foreground_reference=not any(counts[1:]))
        self.last['pseudo']=stats
        return loss

    def gradient_audit(self,labeled,unlabeled):
        params=[p for p in self.model.parameters() if p.requires_grad]
        self.entry_grads=None
        if self.protect_loss is not None:
            self.entry_grads=torch.autograd.grad(self.protect_loss,params,allow_unused=True,retain_graph=True)
            self.telemetry['entry_reference_vjps']+=1

    def transform_gradients(self,grads):
        params=[p for p in self.model.parameters() if p.requires_grad]
        self.labeled_grads=[torch.zeros_like(p) if grads[id(p)]['L'] is None else grads[id(p)]['L'].detach() for p in params]
        for teacher in (self.ema,self.model.entry_anchor):
            if teacher.training or any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
                raise PermissionError('teacher must remain frozen')

    def transform_update(self,before):
        params=[p for p in self.model.parameters() if p.requires_grad]
        terms=self.history.terms(self.model.native_values());refs=[]
        proposal={k:float(v.detach()) for k,v in terms.items()}
        for name,value in terms.items():
            gradients=torch.autograd.grad(value,params,allow_unused=True,retain_graph=True)
            refs.append((name,[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,gradients)],.5/len(terms)))
        refs.append(('combined_L',self.labeled_grads,.25))
        if self.entry_grads is not None:refs.append(('entry',[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,self.entry_grads)],.25))
        stats=elastic_displacement(params,before,refs)
        self.model.sidecar.valid()
        accepted={k:float(v.detach()) for k,v in self.history.terms(self.model.native_values()).items()}
        stats.update(proxy_before=self.proxy_before,proxy_proposal=proposal,proxy_accepted=accepted,
                     G=spectrum(self.model.sidecar.matrices()[0]),F=spectrum(self.model.sidecar.effective()),
                     R_norm=float(self.model.sidecar.r.detach().norm()))
        self.last['elastic_guard']=stats
        self.guard_totals['calls']+=1;self.guard_totals['zero_displacement']+=int(stats['zero_displacement'])
        self.guard_totals['corrected']+=int(stats['rounding_error']>0 or stats['continuous_retention'] is not None and stats['continuous_retention']<1-1e-12)
        self.guard_totals['residual_conflicts']+=stats['residual_conflicts']
        for k,v in self.guard_totals.items():self.telemetry['mnms_guard_'+k]=v

    def update(self,*args,**kwargs):
        result=super().update(*args,**kwargs)
        skipped=kwargs.get('skip',args[0] if args else False)
        if not skipped and self.calibration is not None:
            counts,correct=self.calibration
            self.precision_counts[0]+=correct;self.precision_counts[1]+=counts
        self.calibration=None
        return result
