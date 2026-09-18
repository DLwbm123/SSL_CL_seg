"""Independent B2 adapter: explicit heads/EMA/U whitelist and atomic risk commit."""
import copy
import math
import torch
from ..five_frameworks_v1.train_stage import StageTrainer, split_gradients
from ..five_frameworks_v1.recipes import collected_forward, unique_indices
from ..five_frameworks_v1.kernels import masked_kl, collect_sources
from ..five_frameworks_v1.numerics import finite, validate_commit, ema_value
from .protocol import ARMS, METHOD, GEOMETRY, STUDY, digest
from .core import brier, selector, parent_loss, coverage, shadow_counts


class AGMSTrainer(StageTrainer):
    def __init__(self, model, provider, options, arm, execution, initialize=True, node=None):
        from .authority import validate_runtime
        validate_runtime(model, provider, options, arm, execution, node)
        self.arm, self.node = arm, node
        self.risk = torch.full((3,), .25, device=next(model.parameters()).device)
        self.risk_updates = 0
        self._pending_risk = None
        self.diagnostics = {}
        self.support_summary = {'active_steps': 0, 'empty_coarse': 0, 'fine': 0, 'coarse': 0, 'geometry': 0}
        self.extra_cost = {'diagnostic_vjps': 0, 'student_aux_readouts': 0, 'teacher_aux_readouts': 0, 'extra_backbone_forwards': 0}
        super().__init__(model, provider, options, initialize=initialize, execution=execution)
        if model.aux:
            self.optimizer.add_param_group(dict(params=list(model.aux.parameters()), lr=.001, initial_lr=.001, weight_decay=4e-5, name='aux_L_only'))
            # Rebind the original polynomial schedule to all three groups.
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step:max(0., 1-step/self.options['total_steps'])**.9)
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        actual = {id(p) for g in self.optimizer.param_groups for p in g['params']}
        if expected != actual or any(id(p) in {id(a) for a in model.aux.parameters()} for p in model.u_parameters()):
            raise ValueError('optimizer/gradient whitelist mismatch')

    def semantic_record(self):
        return dict(study_id=STUDY, arm=self.arm, options=self.options, method=METHOD, geometry=GEOMETRY,
                    arm_options=ARMS[self.arm], prefix=self.provider.stage_source,
                    aux_schema={n:list(p.shape) for n,p in self.model.aux.named_parameters()},
                    namespace='AGMS_CL_V0_1/aux/seed/order/stage; arm excluded',
                    parent='NATIVE_LR_SRC_A_3DOMAIN_V1', data=self.provider.semantic_metadata())

    def losses(self):
        from .authority import validate_runtime
        validate_runtime(self.model, self.provider, self.options, self.arm, self.execution, self.node)
        self._pending_risk = None
        if self.arm == 'A0':return super().losses()
        opt, m, t = self.options, self.model, self.ema
        x, y, patients = self.provider.labeled(self.cursor)
        if len(patients) != 2 or len(set(patients)) != 2:raise ValueError('two distinct L sources required')
        total = opt['total_steps'];warmup = math.ceil(opt['warmup_fraction'] * total);active = self.step >= warmup
        aux_views = []
        def forward(v, scale=None):
            self.telemetry['student_full_forwards'] += 1
            with m.capture() as features:out = m(v, detach_parent=False, scale=scale)
            aux_views.append(m.read_aux(features, v.shape[-2:]))
            self.extra_cost['student_aux_readouts'] += len(m.aux)
            return out
        if active:
            logp, mask = collected_forward(forward, x, x.flip(0), self.rng('LCTX'))
            auxiliary = [collect_sources(a, b, mask)[0] for a,b in zip(*aux_views)] if m.aux else []
        else:
            logp = forward(x).log_softmax(1);auxiliary = aux_views[0]
        supervised = m.parent.supervised(logp, y);constraint = m.parent.constraint_loss()
        ds = torch.stack([m.parent.supervised(p, y) for p in auxiliary]).mean() if auxiliary else supervised * 0
        labeled = supervised + constraint
        if m.aux:labeled = labeled + METHOD['lambda_DS'] * ds
        self.last = dict(labeled_loss=float(labeled.detach()), active_U=False,
                         losses={'supervised':float(supervised.detach()), 'DS':float(ds.detach()), 'constraint':float(constraint.detach())})
        if not active:return labeled, None, None
        with t.capture() as features:
            p, s, _, h_l, _ = self.prototypes.estimate(t, x, y)
        pending = (p, s);self.telemetry['teacher_full_forwards'] += 1
        with torch.no_grad():
            ql = t.parent.native_readout(h_l, x.shape[-2:], 'teacher').softmax(1)
            qls = [ql] + [a.exp() for a in t.read_aux(features, x.shape[-2:])]
        self.telemetry['readout_only_forwards'] += 1
        self.extra_cost['teacher_aux_readouts'] += len(t.aux)
        if t.aux:
            r = brier(qls, y)
            if r is not None:self._pending_risk = (.9 * self.risk + .1 * r).detach()
        u, geometry, ids = self.provider.unlabeled(self.cursor)
        indices = unique_indices(ids);u, geometry = u[indices], geometry[indices]
        if len(u) > len(x):raise ValueError('U exceeds paired L batch')
        with torch.no_grad(), t.capture() as features:
            _, _, h_u = t.parts(u, mode='teacher')
            q = t.parent.native_readout(h_u, u.shape[-2:], 'teacher').softmax(1)
            qs = [q] + [a.exp() for a in t.read_aux(features, u.shape[-2:])]
        self.telemetry['teacher_full_forwards'] += 1
        self.extra_cost['teacher_aux_readouts'] += len(t.aux)
        fine, _ = self.prototypes.admission(t, q, h_u, geometry, opt['PAS_confidence'], opt['PAS_cosine'])
        # UL uses the original main-only forward; auxiliary parameters get no U graph.
        def forward_u(v, scale=None):
            self.telemetry['student_full_forwards'] += 1
            return m(v, detach_parent=False, scale=scale)
        logu, _ = collected_forward(forward_u, u, x[:len(u)], self.rng('UL'))
        kl = masked_kl(logu, q, fine)
        coarse, stats = selector(qs, geometry, fine, self.arm, self.risk)
        if not ARMS[self.arm]['H']:coarse = coarse & False
        h = parent_loss(logu, coarse, geometry)
        ramp = min(1., (self.step-warmup+1)/max(1, math.ceil(opt['U_ramp_fraction']*total)))
        unlabeled = kl * opt['lambda_U'] * ramp
        if ARMS[self.arm]['H']:unlabeled = unlabeled + ramp * .5 * h
        self.last.update(active_U=True, accepted=int(fine.sum()), missing_support=(~self.prototypes.support).tolist(),
                         unlabeled_loss=float(unlabeled.detach()), coverage=coverage(fine, coarse, geometry, stats),
                         risk=self.risk.tolist(), alpha=stats['alpha'].tolist(), risk_updates=self.risk_updates,
                         losses={**self.last['losses'], 'KL':float(kl.detach()), 'H':float(h.detach()),
                                 'weighted_DS':float((METHOD['lambda_DS']*ds).detach()), 'weighted_KL':float((kl*opt['lambda_U']*ramp).detach()),
                                 'weighted_H':float((ramp*.5*h).detach()), 'ramp':ramp})
        finite(self._pending_risk, 'pending risk')
        if self.step+1 in {math.ceil(total*f) for f in (.25,.5,.75,1.)}:
            with torch.no_grad():
                gl = y != 255
                fl, _ = self.prototypes.admission(t, ql, h_l, gl, opt['PAS_confidence'], opt['PAS_cosine'])
                cl, sl = selector(qls, gl, fl, self.arm, self.risk)
                uniform, _ = selector(qls, gl, fl, 'A4' if t.aux else 'A1', self.risk)
                adaptive, _ = selector(qls, gl, fl, 'A5' if t.aux else 'A1', self.risk)
            self.last['diagnostic'] = self.gradient_diagnostics(supervised, kl, ds, h, ramp)
            self.last['diagnostic'].update(coverage=self.last['coverage'], risk=self.risk.tolist(), alpha=stats['alpha'].tolist(),
                 losses=copy.deepcopy(self.last['losses']), shadow_L=shadow_counts(qls,y,gl,fl,cl),
                 uniform_risk_disagreement=int((uniform != adaptive).sum()),
                 L_errors=[dict(fine_error=float((p.argmax(1)[gl] != y[gl]).float().mean()) if gl.any() else None,
                                brier=None if brier([p],y) is None else float(brier([p],y)[0])) for p in qls])
        return labeled, unlabeled, pending

    def gradient_diagnostics(self, supervised, kl, ds, h, ramp):
        named = [(n,p) for n,p in self.model.named_parameters() if p.requires_grad]
        vectors, norms = {}, {}
        for name, loss, enabled, factor in [('supervised',supervised,True,1.),('KL',kl,True,ramp),
                                            ('DS',ds,ARMS[self.arm]['M'],METHOD['lambda_DS']),('H',h,ARMS[self.arm]['H'],ramp*.5)]:
            grads = torch.autograd.grad(loss, [p for _,p in named], allow_unused=True, retain_graph=True)
            self.extra_cost['diagnostic_vjps'] += 1
            groups = {}
            for group, predicate in [('all_AB',lambda n:n.startswith('parent.')),('aux_heads',lambda n:n.startswith('aux.')),
                                     ('aux_upstream_AB',lambda n:n.startswith('parent.') and 'decoder.dec1.' not in n)]:
                terms = [torch.zeros_like(p).flatten() if g is None else g.detach().flatten() for (n,p),g in zip(named,grads) if predicate(n)]
                v = torch.cat(terms) if terms else loss.new_zeros(1)
                groups[group] = dict(raw_norm=float(v.norm()), weighted_norm=float(v.norm()*factor))
                if group == 'all_AB':vectors[name] = v
            norms[name] = dict(status='measured' if enabled else 'not_applicable', groups=groups)
        angles = {}
        for a in vectors:
            for b in vectors:
                if a >= b:continue
                va,vb=vectors[a],vectors[b];den=va.norm()*vb.norm()
                angles[a+'__'+b]=None if den==0 else float((va@vb/den).clamp(-1,1))
        return dict(extra_vjps=4, gradients=norms, cosine=angles)

    def update(self, *args, **kwargs):
        self.semantic_record()
        before = self.step
        try:
            result = super().update(*args, **kwargs)
        finally:
            if self.step == before:self._pending_risk = None
        if self.step != before and self.last.get('active_U') and self.arm != 'A0':
            s = self.support_summary;s['active_steps'] += 1
            for r in self.last['coverage']:
                for k in ('fine','coarse','geometry'):s[k] += r[k]
            s['empty_coarse'] += int(not any(r['coarse'] for r in self.last['coverage']))
            if 'diagnostic' in self.last:self.diagnostics[str(self.step)] = copy.deepcopy(self.last['diagnostic'])
        return result

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
        aux_pending = [ema_value(a,b) for a,b in zip(self.ema.aux.parameters(),self.model.aux.parameters())]
        finite(aux_pending, 'candidate auxiliary EMA')
        finite(self._pending_risk, 'candidate risk')
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
        with torch.no_grad():
            for a,b in zip(self.ema.aux.parameters(),aux_pending):a.copy_(b)
            if self._pending_risk is not None:
                self.risk.copy_(self._pending_risk);self.risk_updates += 1
        self._pending_risk = None
        self.step+=1;self.cursor+=1
        scope=self.execution.bindings.get('execution_scope','synthetic') if self.native else 'synthetic'
        key={'formal':'formal_optimizer_updates','smoke':'real_smoke_optimizer_updates'}.get(scope,'synthetic_optimizer_updates')
        self.telemetry[key]=self.telemetry.get(key,0)+1
        self.last['actual_update_norms']={name:float((p.detach()-before[id(p)]).norm()) for name,p in self.model.named_parameters() if id(p) in before}
        return grads
