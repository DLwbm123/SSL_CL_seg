"""B2 loss adapter; optimizer, gradients, EMA and commit lifecycle stay inherited.

Production is gated by fresh study authority; this delivery executes CPU tests only.
"""
import math
import torch
from ..five_frameworks_v1.train_stage import StageTrainer
from ..five_frameworks_v1.recipes import collected_forward,unique_indices,SyntheticCurrentDomain
from ..five_frameworks_v1.kernels import masked_kl
from ..five_frameworks_v1.numerics import finite
from .alignment import LAYER,binding,coordinate_basis,capture_input,alignment_loss


class KeyAlignmentTrainer(StageTrainer):
    def __init__(self,model,provider,options,arm,execution,alignment_weight=None,
                 initialize=True,node=None,restored_basis=None):
        from .authority import validate_runtime
        if type(self) is not KeyAlignmentTrainer:raise PermissionError('exact alignment trainer required')
        if alignment_weight not in (None,0.):raise ValueError('only zero qualification override')
        weight=(0. if arm=='C0' else .05) if alignment_weight is None else alignment_weight
        validate_runtime(model,provider,options,arm,execution,node,weight)
        self.arm,self.align_weight,self.node=arm,weight,node
        self.align_layer,v=binding(model.parent)
        self.align_basis=(restored_basis.detach().clone() if restored_basis is not None else
                          coordinate_basis(arm,v,provider))
        from ..five_frameworks_v1.semantics import tensor_fingerprint
        self._entry_basis_hash=tensor_fingerprint({'basis':self.align_basis}) if self.align_basis is not None else None
        if arm=='C3' and not torch.equal(self.align_basis,v):raise ValueError('C3 must use existing entry V')
        super().__init__(model,provider,options,initialize=initialize,execution=execution)
        self.alignment_cost={'clean_U_forward_attempts':0,'clean_U_forwards':0,'diagnostic_vjps':0,'invalid_steps':0,'active_steps':0}
        self.diagnostics={};self.support_summary={'committed_active_steps':0,'valid_steps':0,'classes':{}}

    def semantic_record(self):
        from ..five_frameworks_v1.semantics import tensor_fingerprint
        from .protocol import STUDY,ARMS
        current=tensor_fingerprint({'basis':self.align_basis}) if self.align_basis is not None else None
        if current!=self._entry_basis_hash:raise ValueError('entry auxiliary basis mutated')
        return dict(study_id=STUDY,arm=self.arm,options=self.options,layer=LAYER,
            coordinate=ARMS[self.arm]['coordinate'],dimension=ARMS[self.arm]['D'],
            lambda_align=self.align_weight,dimension_scale='D/8',key_sha256=tensor_fingerprint({'V':binding(self.model.parent)[1]}),
            basis_sha256=current,prefix=self.provider.stage_source,
            namespaces=['random_key_entry','pixels/L/class','pixels/U/class','directions/D'])

    def update(self,*args,**kwargs):
        self.semantic_record()
        before=self.step
        result=super().update(*args,**kwargs)
        if self.step==before:return result
        if 'gradient_diagnostics' in self.last:
            import copy
            self.diagnostics[str(self.step)]={**copy.deepcopy(self.last['gradient_diagnostics']),
                'alignment':{k:copy.deepcopy(v) for k,v in self.last['alignment'].items() if k!='positions'}}
        if self.last.get('active_U'):
            s=self.support_summary;s['committed_active_steps']+=1
            a=self.last['alignment'];s['valid_steps']+=int(a.get('valid_classes',0)>0)
            for cls,row in a.get('classes',{}).items():
                target=s['classes'].setdefault(cls,{'steps':0,'valid_steps':0,'n_sum':0,'PAS_coverage_sum':0.})
                target['steps']+=1;target['valid_steps']+=int(row['valid']);target['n_sum']+=row['n'];target['PAS_coverage_sum']+=row['PAS_coverage']
        return result

    def losses(self):
        from .authority import validate_runtime
        validate_runtime(self.model,self.provider,self.options,self.arm,self.execution,self.node,self.align_weight)
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
