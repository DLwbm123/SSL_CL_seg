import copy
from pathlib import Path
import torch
from torch.nn import functional as F
from ..five_frameworks_v1.model import Model,Deployment
from ..five_frameworks_v1.native_parent import NativeLRParent
from ..five_frameworks_v1.train_stage import StageTrainer
from ..five_frameworks_v1.gate import digest
from ..five_frameworks_v1.semantics import tensor_fingerprint
from ..f5_hierarchy_contrast_v27.model import HierarchyProtoTrainer
from ..f5_class_reference_v13.model import class_reference
from ..f5_entry_gradient_v12.model import halfspace


def style(images,donor,limits):
    if images.shape!=donor.shape:raise ValueError('style requires paired same-site geometry')
    mu=images.mean((-2,-1),keepdim=True);std=images.std((-2,-1),keepdim=True,unbiased=False)
    dm=donor.detach().mean((-2,-1),keepdim=True);ds=donor.detach().std((-2,-1),keepdim=True,unbiased=False)
    return ((images-mu)/(std+1e-6)*(.8*std+.2*ds)+.8*mu+.2*dm).clamp(*limits)


def weighted_kl(logits,q,valid,anchor=None):
    if logits.shape!=q.shape or valid.shape!=q.shape[:1]+q.shape[2:] or valid.dtype!=torch.bool:
        raise ValueError('pseudo-label probability/mask grids must match')
    q=q.detach();valid=valid.detach().clone();labels=q.argmax(1);confidence=q.max(1).values
    valid &= confidence>.7
    conflict=torch.zeros_like(valid) if anchor is None else ((anchor.detach().max(1).values>=.7)&(anchor.detach().argmax(1)!=labels))
    valid &= ~((labels==0)&conflict)
    counts=torch.stack([(valid&(labels==c)).sum() for c in range(q.shape[1])]).to(q)
    frequency=counts/counts.sum().clamp_min(1)
    weights=frequency.clamp_min(1e-12).rsqrt().clamp(max=5)[labels].detach()
    taper=((confidence-.7)/.3).clamp(0,1).detach()
    soft=torch.where((labels>0)&conflict,.5,1.).to(q).detach()
    per=(torch.xlogy(q,q)-q*logits.log_softmax(1)).sum(1)
    # The soft/confidence factors deliberately do not cancel in the denominator.
    loss=(per*valid*weights*taper*soft).sum()/(valid*weights).sum().clamp_min(1)
    return loss,valid,dict(class_pixels=counts.int().tolist(),foreground_conflicts=int(((labels>0)&conflict&valid).sum()),
                           rejected_background_conflicts=int(((labels==0)&conflict).sum()),confidence_mean=float(taper[valid].mean()) if valid.any() else 0.)


class Parent(NativeLRParent):
    def __init__(self,native,seed,stage_source,source=False):
        self.source_dense=source
        mono=native.enc1.block[0].weight.shape[1]==1
        super().__init__(native,seed,stage_source,adapt=not source,minimum_input_width=9 if mono else 27,require_full_rank=False)
    def configure_stage_training(self):
        if not self.source_dense:return super().configure_stage_training()
        self.train();self.requires_grad_(True)
        self.native.decoder.conv_logit.sigma.requires_grad_(False)
        self.native.decoder.conv_logit.grad_update.requires_grad_(False)
    def optimizer_groups(self,opt):
        if not self.source_dense:return super().optimizer_groups(opt)
        return [dict(params=[p for p in self.parameters() if p.requires_grad],lr=opt['lr'],name='dense_source')]
    def constraint_loss(self):
        if self.source_dense:return self.native.enc1.parameters().__next__().sum()*0
        return super().constraint_loss()
    def semantic_metadata(self):
        return {**super().semantic_metadata(),'source_dense':self.source_dense,'classes':self.effective_readout_kernel_at_entry().shape[0]}
    def supervised(self,logp,y):
        classes=logp.shape[1];v=y!=255
        if y.dtype!=torch.long or not (((y>=0)&(y<classes))|~v).all():raise ValueError('invalid class labels')
        ce=(-(logp.gather(1,y.masked_fill(~v,0)[:,None])[:,0])*v).sum()/v.sum().clamp_min(1)
        p=logp.exp().double()[:,1:]*v[:,None];target=torch.stack([y==c for c in range(1,classes)],1).double()*v[:,None]
        dice=(2*(p*target).sum((2,3))+1e-5)/(p.sum((2,3))+target.sum((2,3))+1e-5)
        return ce+((1-dice.mean(1))*v.flatten(1).any(1)).mean()


class MainModel(Model):
    def __init__(self,parent,parent_map,previous=None,initialize_basis=True):
        self.source_dense=parent.source_dense
        super().__init__(parent,'MAIN_SOURCE' if self.source_dense else 'F5',.25,previous,initialize_basis)
        # Eigensolve and identity resume factories otherwise use different Q strides.
        if self.sidecar is not None:self.sidecar.q=self.sidecar.q.contiguous()
        self.parent_map=tuple(parent_map);self.hierarchy=True;self.alignment=False
        self.hierarchy_weight=.1;self.hierarchy_temperature=.1;self.hierarchy_margin=.25
        self._capture_hierarchy=False;self._labeled_features=[]
        self.entry_anchor=None
        if not self.source_dense:
            entry=copy.deepcopy(parent);entry.stage_exit()
            self.entry_anchor=Deployment(entry,self.sidecar.previous).eval().requires_grad_(False)
    def configure_training(self):
        super().configure_training()
        if self.entry_anchor is not None:self.entry_anchor.eval().requires_grad_(False)
    def parts(self,*args,**kwargs):
        parts=super().parts(*args,**kwargs)
        if self._capture_hierarchy and self.role=='student':self._labeled_features.append(parts[2])
        return parts
    def u_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
    def detach_u_parent(self,*,clean=False):return clean
    def teacher(self):
        teacher=super().teacher();teacher.entry_anchor=self.entry_anchor
        return teacher


class MainTrainer(StageTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.hierarchy_stats={};self.capture_qualification_terms=False;self.qualification_terms=None
        self.entry_reference_loss=None;self.entry_reference_grads=None
        self.anchor_fingerprint=None if self.model.entry_anchor is None else tensor_fingerprint(self.model.entry_anchor.state_dict())
        self.telemetry.update({key:0 for key in ('hierarchy_calls','hierarchy_coarse_calls','hierarchy_parts_calls','hierarchy_alignment_calls','hierarchy_aligned_edges','hierarchy_valid_pixels','hierarchy_seam_excluded_pixels','hierarchy_labeled_feature_views','entry_reference_vjps','entry_projected_calls')})
    labeled_addition=HierarchyProtoTrainer.labeled_addition
    def loss_contract(self):
        return dict(version='MAIN_STYLE_SSL_V1',source=self.model.source_dense,parent_map=self.model.parent_map,
                    style_alpha=.2,foreground_conflict_weight=.5,class_weight_cap=5,confidence=.7,background_cosine=.5,
                    reference='confident EMA/entry agreement; present-class balanced',anchor=self.anchor_fingerprint,
                    source_SWD='absent: no target R at source',
                    implementation=digest({p.name:p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))}))
    def student_images(self,images,donor):
        return style(images,donor,(0,1) if self.provider.dataset=='fundus' else (-1,1))
    def pseudo_admission(self,target,geometry,pas_valid):
        return (geometry&(target.max(1).values>.7)&((target.argmax(1)>0)|pas_valid)).detach()
    def losses(self):
        self.model._capture_hierarchy=True;self.model._labeled_features=[]
        try:return super().losses()
        finally:self.model._capture_hierarchy=False;self.model._labeled_features=[]
    def kl_objective(self,images,logits,target,valid):
        anchor=None
        if self.model.entry_anchor is not None:
            with torch.no_grad():anchor=self.model.entry_anchor(images).softmax(1)
            self.telemetry['teacher_full_forwards']+=1
        loss,kept,stats=weighted_kl(logits,target,valid,anchor)
        if anchor is not None:
            support=kept&(anchor.max(1).values>=.7)&(anchor.argmax(1)==target.argmax(1))
            self.entry_reference_loss,counts=class_reference(logits,anchor,support)
            stats['reference_class_pixels']=counts
        self.last['pseudo']=stats
        return loss
    def gradient_audit(self,labeled,unlabeled):
        self.entry_reference_grads=None
        if unlabeled is not None and self.entry_reference_loss is not None:
            params=[p for p in self.model.parameters() if p.requires_grad]
            self.entry_reference_grads=torch.autograd.grad(self.entry_reference_loss,params,allow_unused=True,retain_graph=True)
            self.entry_reference_loss=None;self.telemetry['entry_reference_vjps']+=1
    def transform_gradients(self,grads):
        params=[p for p in self.model.parameters() if p.requires_grad]
        if self.entry_reference_grads is not None:
            merged=[torch.zeros_like(p) if p.grad is None else p.grad for p in params]
            refs=[torch.zeros_like(p) if g is None else g for p,g in zip(params,self.entry_reference_grads)]
            projected,stats=halfspace(merged,refs)
            for p,g in zip(params,projected):p.grad=g
            self.last['entry_projection']=stats;self.telemetry['entry_projected_calls']+=int(stats['projected'])
            self.entry_reference_grads=None
        for teacher in (self.ema,self.model.entry_anchor):
            if teacher is not None and (teacher.training or any(p.requires_grad or p.grad is not None for p in teacher.parameters())):
                raise PermissionError('teacher changed trainability/mode/received gradient')
