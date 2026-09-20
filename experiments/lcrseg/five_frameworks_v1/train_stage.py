"""Unified stage runner. Synthetic use is available; real binding is unimplemented."""
import math
import torch
from torch.nn import functional as F
from .kernels import masked_kl, masked_jml1, gradient_seed_basis, uncertainty_scales, InsufficientProbeSupport
from .losses import convex_shape, repair_target, class_swd
from .reliability import CurrentPrototypes
from .recipes import generator, collected_forward, unique_indices, mixed_feature_scales


from .semantics import resolve_options, tensor_fingerprint
from .numerics import finite, validate_commit

NO_U={'B0_PARENT_LCTX','B3_PARENT_LCTX_DENSEG'}


def split_gradients(model,labeled,unlabeled):
    params=[p for p in model.parameters() if p.requires_grad]
    gl=torch.autograd.grad(labeled,params,allow_unused=True)
    allowed=model.u_parameters()
    gu=torch.autograd.grad(unlabeled,allowed,allow_unused=True) if unlabeled is not None else [None]*len(allowed)
    by_id={id(p):g for p,g in zip(allowed,gu)}
    result={}
    for p,a in zip(params,gl):
        b=by_id.get(id(p))
        p.grad=None if a is None and b is None else (torch.zeros_like(p) if a is None else a)+(0 if b is None else b)
        result[id(p)]={'L':a,'U':b}
    return result


class StageTrainer:
    def __init__(self,model,provider,options=None,cwmi=None,initialize=True,execution=None):
        self.native=not getattr(model.parent,'synthetic',False)
        if self.native:
            from .native_parent import NativeLRParent
            if not isinstance(model.parent,NativeLRParent) or execution is None:
                raise RuntimeError('real parent/controller requires bound execution capability')
            execution.validate()
        self.execution=execution
        self.model,self.provider=model,provider
        self.options=resolve_options(options);self.cwmi=cwmi
        entry=dict(model.parent.state_dict())
        if model.sidecar is not None:entry['__F_prev__']=model.sidecar.previous
        self.entry_fingerprint=tensor_fingerprint(entry)
        self.model.configure_training()
        self.requires_restore=False;self.physical_optimizer_updates=0
        self.step=0;self.cursor=0;self.epoch=0
        self.telemetry={'student_full_forwards':0,'teacher_full_forwards':0,'probe_full_forwards':0,
                        'probe_vjps':0,'readout_only_forwards':0,'coordinate_vjps':0,
                        'synthetic_optimizer_updates':0,'formal_optimizer_updates':0,
                        'extra_clean_U_forwards':0,'skipped_updates':0}
        self.probe={'scale':1.,'complete':False,'fallback':[]}
        if initialize:
            self.model.parent.stage_entry()
            self.prepare_stage()
        self.ema=model.teacher()
        self.prototypes=CurrentPrototypes(3,model.parent.d)
        if initialize and model.family not in NO_U:
            x,y,_=provider.labeled(0,'prototype_initialization')
            p,s,*_=self.prototypes.estimate(self.ema,x,y)
            self.prototypes.commit(p,s);self.telemetry['teacher_full_forwards']+=1
        groups=model.parent.optimizer_groups(self.options)
        if model.sidecar is not None:
            groups.append({'params':[model.sidecar.r],'lr':self.options.get('lr',.01)*self.options.get('feature_lr_multiplier',1.),
                           'weight_decay':self.options.get('weight_decay',0.)*self.options.get('feature_weight_decay_multiplier',1.),'name':'R'})
        self.optimizer=torch.optim.Adam(groups,weight_decay=self.options.get('weight_decay',0.))
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,
            (lambda step:max(0.,1-step/self.options['total_steps'])**.9) if self.native else (lambda _:1.))
        self.scaler=(torch.amp.GradScaler('cpu',enabled=False) if hasattr(torch.amp,'GradScaler')
                     else torch.cuda.amp.GradScaler(enabled=False))
        self.last={}

    @classmethod
    def for_resume(cls,model,provider,options=None,cwmi=None,execution=None):
        """No stage_entry, source read, probe, prototype initialization or reset."""
        return cls(model,provider,options,cwmi,initialize=False,execution=execution)

    def structural(self,p,y):
        if self.model.family=='F2':
            if self.cwmi is None:raise RuntimeError('F2 requires the locked CWMI backend')
            return self.cwmi(p,y)
        return convex_shape(p,y!=255)

    def prepare_stage(self):
        if self.model.family not in ('F2','F4'):return
        self.probe["complete"]=False
        ratios=[]
        # These gradients are with respect to the SAME final feature h, neither
        # optimizer gradients nor real-data qualification updates.
        for i in range(8):
            x,y,_=self.provider.labeled(i,'scale_calibration')
            h=self.model.parent.features(x).detach().requires_grad_(True)
            readout_h=h if self.model.sidecar is None else self.model.sidecar(h)
            logits=self.model.parent.native_readout(readout_h,x.shape[-2:])
            sup=self.model.parent.supervised(logits.log_softmax(1),y)
            structure=self.structural(logits.softmax(1),y)
            finite((sup,structure),"calibration losses")
            gs,=torch.autograd.grad(sup,h,retain_graph=True)
            gt,=torch.autograd.grad(structure,h)
            finite((gs,gt),"calibration gradients")
            if gt.norm()>0:ratios.append(float(gs.norm()/gt.norm()))
            self.telemetry['probe_full_forwards']+=1;self.telemetry['probe_vjps']+=2
        self.probe['scale']=min(1e3,max(1e-3,float(torch.quantile(torch.tensor(ratios,dtype=torch.float64),.5)))) if ratios else 1.
        self.probe['zero_structure']=not bool(ratios)
        if self.model.family=='F2':
            adapters=self.model.parent.adapters
            selected=list(range(len(adapters)))
            if self.options.get('probe_layers','last_two_parent_adapters')=='last_two_parent_adapters':selected=selected[-2:]
            for i in selected:
                if not adapters[i].new or adapters[i].b.count_nonzero():
                    raise ValueError('F2 may initialize only NEW zero-increment adapters')
            probes={i:[] for i in selected}
            for n in range(8):
                x,y,_=self.provider.labeled(n,'direction_probes')
                weights={i:adapters[i].effective_weight().detach().requires_grad_(True) for i in selected}
                h=self.model.parent.features(x,overrides=weights)
                logits=self.model.parent.native_readout(h,x.shape[-2:])
                # Direction objective is scaled STRUCTURE ONLY, as specified.
                objective=self.options.get('lambda_structure',.1)*self.probe['scale']*self.structural(logits.softmax(1),y)
                finite(objective,"direction objective")
                grads=torch.autograd.grad(objective,list(weights.values()))
                for i,g in zip(selected,grads):probes[i].append(g)
                self.telemetry['probe_full_forwards']+=1;self.telemetry['probe_vjps']+=1
            replacements={};fallbacks=[]
            for i in selected:
                adapter=adapters[i]
                try:
                    q=gradient_seed_basis(probes[i],adapter.a.shape[0],adapter.free_projector)
                    replacements[i]=q.T*(adapter.a.norm()/q.norm().clamp_min(1e-12))
                    finite(replacements[i],'candidate probe A')
                except InsufficientProbeSupport as e:
                    fallbacks.append({'layer':i,'reason':str(e),'type':type(e).__name__})
            # No selected A is changed until every candidate/constraint validated.
            with torch.no_grad():
                for i,value in replacements.items():adapters[i].a.copy_(value)
            self.probe['fallback']=fallbacks
        self.probe['complete']=True

    def fine_kl(self, logits, target, valid):
        return masked_kl(logits, target, valid)

    def losses(self):
        opt=self.options;m=self.model;t=self.ema
        x,y,patients=self.provider.labeled(self.cursor)
        if len(set(patients))!=2 or len(patients)!=2:raise ValueError('LCTX requires two distinct patients')
        total=opt.get('total_steps',5);warmup=math.ceil(opt["warmup_fraction"]*total)
        active=self.step>=warmup
        def forward(v,s=None,detach=False):
            self.telemetry['student_full_forwards']+=1
            return m(v,detach_parent=detach,scale=s)
        if active:
            logp,_=collected_forward(lambda v,s:forward(v,s),x,x.flip(0),self.rng('LCTX'))
        else:logp=forward(x).log_softmax(1)
        supervised=m.parent.supervised(logp,y);constraint=m.parent.constraint_loss()
        finite((supervised,constraint),'active supervised/constraint losses')
        labeled=supervised+constraint
        if m.family=='F2':
            structure=self.structural(logp.exp(),y);finite(structure,'active CWMI loss')
            labeled=labeled+opt.get('lambda_structure',.1)*self.probe['scale']*structure
        finite(labeled,'combined labeled loss')
        self.last={'labeled_loss':float(labeled.detach()),'active_U':False}
        if m.family in NO_U or not active:return labeled,None,None
        p,s,before_l,h_l,lclasses=self.prototypes.estimate(t,x,y)
        pending=(p,s);self.telemetry['teacher_full_forwards']+=1
        u,geometry,ids=self.provider.unlabeled(self.cursor)
        indices=unique_indices(ids);u=u[indices];geometry=geometry[indices]
        # Deduplicating before forward gives each physical source exactly one vote.
        if len(u)>len(x):raise ValueError('U batch exceeds the paired current-L context batch')
        donor=x[:len(u)]
        with torch.no_grad():
            _,before_u,h_u=t.parts(u,mode="teacher")
            q=t.parent.native_readout(h_u,u.shape[-2:],'teacher').softmax(1)
        self.telemetry['teacher_full_forwards']+=1
        valid,_=self.prototypes.admission(t,q,h_u,geometry,opt.get('PAS_confidence',.7),opt.get('PAS_cosine',.5))
        if m.family.startswith('B1'):valid=geometry&(q.max(1).values>opt.get('PAS_confidence',.7))
        scales=None
        if m.family=='F3':
            valid=geometry
            with torch.no_grad():ql=t.parent.native_readout(h_l,x.shape[-2:],'teacher').softmax(1)
            self.telemetry['readout_only_forwards']+=1
            uu=self.prototypes.uncertainty(t,q,h_u,opt.get('prototype_uncertainty_mix',0.))
            ul=self.prototypes.uncertainty(t,ql,h_l,opt.get('prototype_uncertainty_mix',0.))[:len(u)]
            def scales(mask):
                return mixed_feature_scales(m.parent,mask,uu,ul,h_u.shape[-2:],m.spectrum,opt.get('kappa',.5))
        detach=m.family in ('F1','F3','F4','F5')
        logu,_=collected_forward(lambda v,s:forward(v,s,detach),u,donor,self.rng('UL'),scales)
        target=q
        if m.family=='F4':
            target,stats=repair_target(t,u,geometry,opt.get('lambda_shape',.1),self.probe['scale'],
                                      opt.get('inner_steps',3),opt.get('trust_fraction',.05),cached=(before_u,q))
            self.telemetry['teacher_full_forwards']+=stats['full_forwards']
            self.telemetry['coordinate_vjps']+=stats['coordinate_vjps']
            self.telemetry['readout_only_forwards']+=stats['readout_only_forwards']
            self.last['repair']=stats
        unlabeled=self.fine_kl(logu,target,valid)
        if m.family=='F1':unlabeled=unlabeled+opt.get('lambda_JML',.25)*masked_jml1(logu.exp(),target,valid)
        if m.family=='F5':
            _,before_clean,_=m.parts(u,detach_parent=True)
            self.telemetry['student_full_forwards']+=1;self.telemetry['extra_clean_U_forwards']+=1
            qu=m.sidecar.q
            zu=torch.einsum('dk,bdhw->bkhw',qu,before_clean)
            zl=torch.einsum('dk,bdhw->bkhw',qu,before_l)
            uc=m.parent.output_to_feature(q.argmax(1),zu.shape[-2:],True)
            uv=m.parent.output_to_feature(valid,zu.shape[-2:],True)
            swd,counts=class_swd(zu,zl,uc,lclasses,uv,self.rng('swd_sampling'))
            unlabeled=unlabeled+opt.get('lambda_SWD',.05)*swd
            self.last['swd_counts']=counts
        ramp=min(1.,(self.step-warmup+1)/max(1,math.ceil(opt["U_ramp_fraction"]*total)))
        unlabeled=unlabeled*opt.get('lambda_U',.5)*ramp
        self.last.update(active_U=True,accepted=int(valid.sum()),missing_support=(~self.prototypes.support).tolist(),
                         unlabeled_loss=float(unlabeled.detach()))
        return labeled,unlabeled,pending

    def rng(self,stream):
        p=self.provider
        return generator(p.seed,p.order,p.stage,self.cursor,stream)

    def update(self,skip=False,fault=None,physical=None):
        if self.requires_restore:raise RuntimeError('uncommitted failed step requires checkpoint restore')
        try:return self._update(skip,fault,physical)
        except Exception:
            self.requires_restore=True
            raise

    def _update(self,skip=False,fault=None,physical=None):
        self.optimizer.zero_grad(set_to_none=True)
        labeled,unlabeled,pending=self.losses()
        finite((labeled,unlabeled),"active L/U losses")
        finite(labeled if unlabeled is None else labeled+unlabeled,"combined objective")
        grads=split_gradients(self.model,labeled,unlabeled)
        if skip:
            self.optimizer.zero_grad(set_to_none=True);self.telemetry['skipped_updates']+=1
            return grads
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise FloatingPointError('nonfinite merged gradient; no state committed')
        def inject(point):
            if fault==point:raise RuntimeError('injected '+point)
        inject('before_optimizer')
        before={id(p):p.detach().clone() for p in self.model.parameters() if p.requires_grad}
        try:
            self.optimizer.step()
        finally:
            # Physical invocation may have partially updated state even if it raises.
            self.physical_optimizer_updates+=1
            if physical:physical(self.step+1)
        inject('after_optimizer');self.model.parent.apply_constraints()
        self.scheduler.step()
        validate_commit(self,pending)
        inject('before_ema')
        with torch.no_grad():
            if self.native:
                self.model.parent.update_dense_ema(self.ema.parent)
                if self.model.sidecar is not None:self.ema.sidecar.r.mul_(.99).add_(self.model.sidecar.r,alpha=.01)
            else:
                for a,b in zip(self.ema.parameters(),self.model.parameters()):
                    if a is b:continue
                    if b.requires_grad:a.mul_(.99).add_(b,alpha=.01)
                    else:a.copy_(b)
                for a,b in zip(self.ema.buffers(),self.model.buffers()):a.copy_(b)
        inject('after_ema')
        if pending:self.prototypes.commit(*pending)
        self.step+=1;self.cursor+=1
        scope=self.execution.bindings.get('execution_scope','synthetic') if self.native else 'synthetic'
        key={'formal':'formal_optimizer_updates','smoke':'real_smoke_optimizer_updates'}.get(scope,'synthetic_optimizer_updates')
        self.telemetry[key]=self.telemetry.get(key,0)+1
        self.last['actual_update_norms']={name:float((p.detach()-before[id(p)]).norm()) for name,p in self.model.named_parameters() if id(p) in before}
        return grads
