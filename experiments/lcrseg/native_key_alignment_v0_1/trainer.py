"""B2 loss adapter; optimizer, gradients, EMA and commit lifecycle stay inherited.

Only CPU generated-provider execution is enabled in this code-preparation release.
"""
import math
import torch
from ..five_frameworks_v1.train_stage import StageTrainer
from ..five_frameworks_v1.recipes import collected_forward,unique_indices,SyntheticCurrentDomain
from ..five_frameworks_v1.kernels import masked_kl
from ..five_frameworks_v1.numerics import finite
from .alignment import LAYER,binding,coordinate_basis,capture_input,alignment_loss


class KeyAlignmentTrainer(StageTrainer):
    def __init__(self,model,provider,options,arm,execution,alignment_weight=None):
        if alignment_weight not in (None,0.):raise ValueError('only zero-weight qualification override is allowed')
        if (model.family!='B2_PARENT_PAS_KL' or model.sidecar is not None
                or not isinstance(provider,SyntheticCurrentDomain)
                or next(model.parameters()).device.type!='cpu'
                or execution.bindings.get('study_id')!='NATIVE_KEY_ALIGNMENT_V0_1'
                or execution.bindings.get('execution_scope')!='cpu_synthetic'):
            raise PermissionError('CODE_READY_FOR_REVIEW: real/CUDA execution disabled')
        super().__init__(model,provider,options,execution=execution)
        self.arm=arm;self.align_weight=(0. if arm=='C0' else .05) if alignment_weight is None else alignment_weight
        self.align_layer,v=binding(self.model.parent)
        self.align_basis=coordinate_basis(arm,v,provider)
        self.alignment_cost={'clean_U_forward_attempts':0,'clean_U_forwards':0,'diagnostic_vjps':0,'invalid_steps':0,'active_steps':0}

    def losses(self):
        # This is the original B2 branch, using its unchanged helper operations.
        opt=self.options;m=self.model;t=self.ema
        x,y,patients=self.provider.labeled(self.cursor)
        if len(set(patients))!=2 or len(patients)!=2:raise ValueError('LCTX requires two distinct patients')
        total=opt['total_steps'];warmup=math.ceil(opt['warmup_fraction']*total);active=self.step>=warmup
        def forward(v,s=None):
            self.telemetry['student_full_forwards']+=1
            return m(v,detach_parent=False,scale=s)
        if active:logp,_=collected_forward(forward,x,x.flip(0),self.rng('LCTX'))
        else:logp=forward(x).log_softmax(1)
        supervised=m.parent.supervised(logp,y);constraint=m.parent.constraint_loss();labeled=supervised+constraint
        finite((supervised,constraint,labeled),'active supervised/constraint losses')
        self.last={'labeled_loss':float(labeled.detach()),'active_U':False}
        if not active:return labeled,None,None
        with capture_input(t.parent.native.get_submodule(LAYER)) as reference:
            p,s,_,_,_=self.prototypes.estimate(t,x,y)
        if len(reference)!=1:raise ValueError('expected exactly one current teacher-L capture')
        pending=(p,s);self.telemetry['teacher_full_forwards']+=1
        u,geometry,ids=self.provider.unlabeled(self.cursor)
        indices=unique_indices(ids);u=u[indices];geometry=geometry[indices]
        if len(u)>len(x):raise ValueError('U batch exceeds paired L context')
        with torch.no_grad():
            _,_,h_u=t.parts(u,mode='teacher')
            q=t.parent.native_readout(h_u,u.shape[-2:],'teacher').softmax(1)
        self.telemetry['teacher_full_forwards']+=1
        valid,_=self.prototypes.admission(t,q,h_u,geometry,opt['PAS_confidence'],opt['PAS_cosine'])
        logu,_=collected_forward(forward,u,x[:len(u)],self.rng('UL'))
        kl=masked_kl(logu,q,valid)
        ramp=min(1.,(self.step-warmup+1)/max(1,math.ceil(opt['U_ramp_fraction']*total)))
        # Capture exactly one extra clean-U feature forward, with upstream graph.
        self.alignment_cost['clean_U_forward_attempts']+=1
        with torch.set_grad_enabled(self.align_weight!=0),capture_input(self.align_layer) as current:
            m.parts(u,detach_parent=False)
        if len(current)!=1:raise ValueError('expected exactly one clean-U capture')
        self.telemetry['student_full_forwards']+=1;self.telemetry['extra_clean_U_forwards']+=1
        self.alignment_cost['clean_U_forwards']+=1;self.alignment_cost['active_steps']+=1
        if self.align_weight:
            align,stats=alignment_loss(current[0],reference[0],y,q.argmax(1),valid,geometry,
                                       self.align_basis,self.provider,self.cursor)
            self.alignment_cost['invalid_steps']+=int(stats['invalid_step'])
        else:align=None;stats={'no_op':True,'weighted_loss':0.}
        unlabeled=kl*opt['lambda_U']*ramp
        if align is not None:unlabeled=unlabeled+ramp*self.align_weight*align
        stats.update(ramp=ramp,weighted_loss=0. if align is None else float((ramp*self.align_weight*align).detach()))
        self.last.update(active_U=True,accepted=int(valid.sum()),missing_support=(~self.prototypes.support).tolist(),
                         unlabeled_loss=float(unlabeled.detach()),KL=float(kl.detach()),alignment=stats)
        if self.step+1 in {math.ceil(total*f) for f in (.25,.5,.75,1.)}:
            self.last['gradient_diagnostics']=self.gradient_diagnostics(labeled,kl*ramp,align,ramp)
        return labeled,unlabeled,pending

    def gradient_diagnostics(self,labeled,kl,align,ramp):
        """Read-only extra VJPs at frozen steps; never changes optimizer rules."""
        params=[p for p in self.model.parameters() if p.requires_grad]
        vectors={};norms={}
        for name,loss in [('supervised',labeled),('KL',kl),('alignment',None if align is None else align*ramp*self.align_weight)]:
            if loss is None:vectors[name]=None;norms[name]=0.;continue
            self.alignment_cost['diagnostic_vjps']+=1
            grads=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
            vectors[name]=torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(params,grads)])
            norms[name]=float(vectors[name].norm())
        angles={}
        for name in ('supervised','KL'):
            a,b=vectors['alignment'],vectors[name]
            cosine=None if a is None or norms['alignment']==0 or norms[name]==0 else float((a@b)/(a.norm()*b.norm()))
            angles[name]={'cosine':cosine,'angle_degrees':None if cosine is None else math.degrees(math.acos(max(-1.,min(1.,cosine))))}
        return dict(norms=norms,alignment_vs=angles,extra_vjps=sum(v is not None for v in vectors.values()))
