"""A current-L hierarchy regularizer on the unchanged V13 loss/update engine."""
from pathlib import Path
import math
import torch
from torch.nn import functional as F
from ..f5_class_reference_v13.model import ClassReferenceModel,ClassReferenceTrainer
from ..five_frameworks_v1.kernels import collect_sources
from ..five_frameworks_v1.gate import digest
from ..f5_hierarchy_proto_v21.hierarchy import hierarchy_terms,validate_mapping
from ..five_frameworks_v1.model import Deployment
from .orientation import orientation_terms


class CapturingDeployment(Deployment):
    def forward(self,x):
        h=self.parent.features(x,'eval');h=torch.einsum('ij,bjhw->bihw',self.transform,h)
        if self.capture_orientation:self.orientation_features=h.detach()
        return self.parent.native_readout(h,x.shape[-2:],'eval')


class HierarchyProtoModel(ClassReferenceModel):
    def __init__(self,*args,parent_map,hierarchy_weight,temperature,margin,hierarchy=True,orientation=True,orientation_weight=.1,**kwargs):
        super().__init__(*args,class_reference=True,**kwargs)
        validate_mapping(parent_map)
        if len(parent_map)!=self.parent.effective_readout_kernel_at_entry().shape[0]:raise ValueError('ancestry must cover every actual output class')
        if not math.isfinite(hierarchy_weight) or hierarchy_weight<=0:raise ValueError('positive finite hierarchy weight required')
        if not math.isfinite(temperature) or temperature<=0 or not math.isfinite(margin) or not 0<=margin<=2:raise ValueError('invalid hierarchy temperature/margin')
        self.parent_map=tuple(parent_map);self.hierarchy_weight=hierarchy_weight
        self.hierarchy_temperature=temperature;self.hierarchy_margin=margin;self.hierarchy=hierarchy
        self._capture_hierarchy=False;self._labeled_features=[];self._labeled_inputs=[]
        if not math.isfinite(orientation_weight) or orientation_weight<=0:raise ValueError('positive finite orientation weight required')
        self.orientation=orientation;self.orientation_weight=orientation_weight
        self.entry_anchor.__class__=CapturingDeployment
        self.entry_anchor.capture_orientation=False;self.entry_anchor.orientation_features=None

    def parts(self,x,detach_parent=False,scale=None,mode=None):
        result=super().parts(x,detach_parent,scale,mode)
        if self._capture_hierarchy and (self.role if mode is None else mode)=='student':
            self._labeled_features.append(result[2]);self._labeled_inputs.append(x.detach())
        return result


class HierarchyProtoTrainer(ClassReferenceTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.telemetry.update(hierarchy_calls=0,hierarchy_coarse_calls=0,hierarchy_parts_calls=0,
                              hierarchy_alignment_calls=0,hierarchy_aligned_edges=0,hierarchy_valid_pixels=0,
                              hierarchy_seam_excluded_pixels=0,hierarchy_labeled_feature_views=0)
        self.telemetry.update(orientation_calls=0,orientation_leaf_nodes=0,orientation_parent_nodes=0,orientation_positive_calls=0,orientation_teacher_full_forwards=0)
        self.orientation_stats={};self.hierarchy_stats={};self.capture_qualification_terms=False;self.qualification_terms=None

    def loss_contract(self):
        return {**super().loss_contract(),'hierarchy':self.model.hierarchy,'orientation':self.model.orientation,'orientation_weight':self.model.orientation_weight,
                'orientation_scope':'current-L GT prototypes; same mixed views/seam exclusion; frozen entry teacher no U labels; class-balanced unit leaf and multi-child-parent prototype orientation 1-cosine; existing student L views, additional teacher view per L view',
                'hierarchy_parent_map':self.model.parent_map,'hierarchy_weight':self.model.hierarchy_weight,
                'hierarchy_temperature':self.model.hierarchy_temperature,'hierarchy_margin':self.model.hierarchy_margin,
                'hierarchy_scope':'current L only; existing post-transform LCTX features collected using the same mask, 3x3 feature seam excluded; detached batch prototype targets; coarse and sibling class-balanced contrast plus live child-to-detached-parent hinge; no bank/head/U label/teacher/deployment change',
                'hierarchy_implementation':digest([Path(__file__).read_text(),Path(__file__).with_name('orientation.py').read_text(),Path(__file__).parents[1].joinpath('f5_hierarchy_relation_v23/relation.py').read_text()])}

    def losses(self):
        self.model._capture_hierarchy=self.model.hierarchy
        self.model._labeled_features=[];self.model._labeled_inputs=[];self.orientation_stats={};self.hierarchy_stats={};self.qualification_terms=None
        try:return super().losses()
        finally:
            self.model._capture_hierarchy=False;self.model._labeled_features=[];self.model._labeled_inputs=[]
            self.model.entry_anchor.capture_orientation=False;self.model.entry_anchor.orientation_features=None

    def labeled_addition(self,logp,labels,mix_mask):
        if not self.model.hierarchy:return None
        views=self.model._labeled_features;self.model._capture_hierarchy=False
        expected=1 if mix_mask is None else 2
        if len(views)!=expected:raise RuntimeError('current-L feature capture does not match LCTX views')
        h=views[0];mapped=self.model.parent.output_to_feature(labels,h.shape[-2:],True)
        excluded=0
        if mix_mask is not None:
            mask=self.model.parent.output_to_feature(mix_mask,h.shape[-2:],True)
            h=collect_sources(views[0],views[1],mask)[0]
            hi=F.max_pool2d(mask[:,None].float(),3,stride=1,padding=1)
            lo=-F.max_pool2d(-mask[:,None].float(),3,stride=1,padding=1)
            seam=(hi!=lo)[:,0];excluded=int(((mapped!=255)&seam).sum())
            mapped=torch.where(seam,torch.full_like(mapped,255),mapped)
        terms,stats=hierarchy_terms(h,mapped,self.model.parent_map,self.model.hierarchy_temperature,self.model.hierarchy_margin)
        self.hierarchy_stats={**stats,**{'loss_'+k:float(v.detach()) for k,v in terms.items()},
                              'seam_excluded_pixels':excluded,'labeled_feature_views':len(views)}
        self.telemetry['hierarchy_calls']+=1
        for term in terms:self.telemetry['hierarchy_'+term+'_calls']+=int(float(terms[term].detach())>0)
        for key in ('aligned_edges','valid_pixels','seam_excluded_pixels','labeled_feature_views'):
            self.telemetry['hierarchy_'+key]+=self.hierarchy_stats[key]
        addition=self.model.hierarchy_weight*sum(terms.values())
        if self.model.orientation:
            old=[];anchor=self.model.entry_anchor
            with torch.no_grad():
                anchor.capture_orientation=True
                for x in self.model._labeled_inputs:
                    anchor(x);old.append(anchor.orientation_features);anchor.orientation_features=None
                    self.telemetry['teacher_full_forwards']+=1;self.telemetry['orientation_teacher_full_forwards']+=1
                anchor.capture_orientation=False
            if len(old)!=expected:raise RuntimeError('entry teacher must see each exact current-L view once')
            teacher=old[0] if mix_mask is None else collect_sources(old[0],old[1],mask)[0]
            rel,stats=orientation_terms(h,teacher,mapped,self.model.parent_map)
            self.orientation_stats={**stats,**{k:float(v.detach()) for k,v in rel.items()}}
            self.telemetry['orientation_calls']+=1
            self.telemetry['orientation_positive_calls']+=int(float(sum(rel.values()).detach())>0)
            for k in ('orientation_leaf_nodes','orientation_parent_nodes'):self.telemetry[k]+=stats[k]
            addition=addition+self.model.orientation_weight*sum(rel.values());terms={**terms,**rel}
        if self.capture_qualification_terms:self.qualification_terms=terms
        return addition

    def transform_gradients(self,grads):
        super().transform_gradients(grads)
        self.module_stats.update(hierarchy=self.model.hierarchy,hierarchy_stats=self.hierarchy_stats,orientation_stats=self.orientation_stats)
