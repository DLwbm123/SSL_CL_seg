"""Current-U entry dense preservation; unchanged V21 labeled and V13 U engine."""
from pathlib import Path
import math
import torch
from ..f5_hierarchy_proto_v21.model import HierarchyProtoModel as V21Model,HierarchyProtoTrainer as V21Trainer
from ..f5_hierarchy_dense_v25.dense import dense_terms
from ..five_frameworks_v1.model import Deployment
from ..five_frameworks_v1.gate import digest


class CapturingDeployment(Deployment):
    def forward(self,x):
        h=self.parent.features(x,'eval');h=torch.einsum('ij,bjhw->bihw',self.transform,h)
        logits=self.parent.native_readout(h,x.shape[-2:],'eval')
        if self.capture_dense:
            self.dense_features=h.detach();self.dense_probabilities=logits.detach().softmax(1)
        return logits


class HierarchyProtoModel(V21Model):
    def __init__(self,*args,dense=True,dense_weight=.1,**kwargs):
        super().__init__(*args,**kwargs)
        if not math.isfinite(dense_weight) or dense_weight<=0:raise ValueError('positive finite dense weight required')
        self.dense=dense;self.dense_weight=dense_weight
        self.entry_anchor.__class__=CapturingDeployment
        self.entry_anchor.capture_dense=False;self.entry_anchor.dense_features=None;self.entry_anchor.dense_probabilities=None


class HierarchyProtoTrainer(V21Trainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.telemetry.update(dense_calls=0,dense_leaf_regions=0,dense_parent_regions=0,dense_valid_pixels=0,dense_zero_pixels=0,dense_positive_calls=0,dense_student_full_forwards=0)
        self.dense_stats={}

    def loss_contract(self):
        return {**super().loss_contract(),'dense':self.model.dense,'dense_weight':self.model.dense_weight,
                'dense_scope':'original current-U only; frozen-entry argmax over ORIGINAL PAS valid & entry-confidence>=PAS_confidence; nearest feature-grid support, ignore255; region-balanced dense 1-cosine on same original-U images; one additional live A/B/R clean student forward; no additional teacher forward; .1 inside original lambdaU/ramp, inactive during original U warmup; original KL/reference and SWD R-only unchanged',
                'dense_implementation':digest([Path(__file__).read_text(),Path(__file__).parents[1].joinpath('f5_hierarchy_dense_v25/dense.py').read_text()])}

    def losses(self):
        self.dense_stats={}
        try:return super().losses()
        finally:
            anchor=self.model.entry_anchor
            anchor.capture_dense=False;anchor.dense_features=None;anchor.dense_probabilities=None

    def kl_objective(self,images,logits,target,valid):
        if not self.model.dense:return super().kl_objective(images,logits,target,valid)
        anchor=self.model.entry_anchor
        try:
            anchor.capture_dense=True
            main=super().kl_objective(images,logits,target,valid)
            anchor.capture_dense=False
            teacher=anchor.dense_features;probabilities=anchor.dense_probabilities
            if teacher is None or probabilities is None:raise RuntimeError('same original-U entry reference forward must supply features/probabilities')
            with torch.no_grad():
                support=valid & (probabilities.max(1).values>=self.options['PAS_confidence'])
                labels=self.model.parent.output_to_feature(probabilities.argmax(1),teacher.shape[-2:],True)
                kept=self.model.parent.output_to_feature(support,teacher.shape[-2:],True)
                labels=torch.where(kept,labels,torch.full_like(labels,255))
            student=self.model.parts(images,detach_parent=False)[2]
            self.telemetry['student_full_forwards']+=1;self.telemetry['dense_student_full_forwards']+=1
            terms,stats=dense_terms(student,teacher,labels,self.model.parent_map)
            self.dense_stats={**stats,'dense_loss':float(terms['dense_loss'].detach()),'original_reference_support_pixels':int(support.sum()),'dense_main_KL':float(main.detach()),'dense_original_U_aligned':True}
            self.telemetry['dense_calls']+=1;self.telemetry['dense_positive_calls']+=int(float(terms['dense_loss'].detach())>0)
            for k in ('dense_leaf_regions','dense_parent_regions','dense_valid_pixels','dense_zero_pixels'):self.telemetry[k]+=stats[k]
            if self.capture_qualification_terms:self.qualification_terms={**(self.qualification_terms or {}),**terms}
            return main+self.model.dense_weight*terms['dense_loss']
        finally:
            anchor.capture_dense=False;anchor.dense_features=None;anchor.dense_probabilities=None

    def transform_gradients(self,grads):
        super().transform_gradients(grads)
        self.module_stats.update(dense=self.model.dense,dense_stats=self.dense_stats)
