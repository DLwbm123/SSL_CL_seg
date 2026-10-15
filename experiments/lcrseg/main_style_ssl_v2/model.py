"""V2 hooks on the unchanged StageTrainer; no second training loop."""
import torch
from torch.nn import functional as F
from ..main_style_ssl_v1.model import Parent as V1Parent, MainModel as V1Model, MainTrainer as V1Trainer, weighted_kl
from ..five_frameworks_v1.native_parent import lr
from ..five_frameworks_v1.kernels import collect_sources
from ..five_frameworks_v1.gate import digest
from .protocol import VERSION, GUARD, MEMORY
from .protection import BoundedSidecar, constrain_displacement, spectral_stats
from .memory import SiteMemory, finite_tree


def independent_reference(logits, q, geometry, reliability, evidence):
    if logits.shape != q.shape or geometry.shape != q.shape[:1]+q.shape[2:] or geometry.dtype != torch.bool:
        raise ValueError('independent reference geometry mismatch')
    if reliability.shape != (q.shape[1],) or evidence.shape != geometry.shape:
        raise ValueError('independent reference weight shape mismatch')
    q=q.detach();labels=q.argmax(1);support=geometry.detach()&(q.max(1).values>=.7)
    per_pixel=(torch.xlogy(q,q)-q*logits.log_softmax(1)).sum(1)
    terms=[];counts=[]
    for c in range(q.shape[1]):
        mask=support&(labels==c);counts.append(int(mask.sum()))
        if mask.any():
            terms.append(reliability[c].detach()*(per_pixel*mask*evidence.detach()).sum()/mask.sum())
    loss=torch.stack(terms).mean() if terms else logits.sum()*0
    return loss,counts,support


class ObservedConv(lr.LowRankConv):
    def forward(self, x):
        w = self.effective()
        if self.capture:
            self.views.append(w)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Parent(V1Parent):
    def __init__(self, native, seed, stage_source, source=False):
        self.mri = native.enc1.block[0].weight.shape[1] == 1
        super().__init__(native, seed, stage_source, source)
        for a in self.adapters:
            # Instance-local subclass: state keys and existing merge/EMA semantics are unchanged.
            a.layer.__class__ = ObservedConv
            a.layer.capture = False
            a.layer.views = []

    def configure_stage_training(self):
        super().configure_stage_training()
        if not self.source_dense and self.mri:
            self.adapters[0].a.requires_grad_(False)
            self.adapters[0].b.requires_grad_(False)

    def optimizer_groups(self, options):
        groups = super().optimizer_groups(options)
        for group in groups:
            group['params'] = [p for p in group['params'] if p.requires_grad]
        return groups

    def native_values(self):
        values = {}
        for name, p in self.native.named_parameters():
            module, _, key = name.rpartition('.')
            layer = self.native.get_submodule(module)
            if isinstance(layer, lr.LowRankConv):
                if key != 'weight':
                    continue
                p = layer.effective()
            if key in ('sigma', 'grad_update') or '.sigma.' in name:
                continue
            values[name] = p
        return values

    def semantic_metadata(self):
        return {**super().semantic_metadata(), 'version':VERSION,
                'mri_target_first_conv_frozen':self.mri and not self.source_dense}


class MainModel(V1Model):
    def __init__(self, parent, parent_map, previous=None, initialize_basis=True):
        super().__init__(parent, parent_map, previous, initialize_basis)
        if self.sidecar is not None:
            self.sidecar = BoundedSidecar(self.sidecar.q, self.sidecar.previous)
        self.source_f_views = []
        self.observe = False

    def parts(self, *args, **kwargs):
        h, before, out = super().parts(*args, **kwargs)
        if self.source_dense and self.observe and self.role == 'student':
            f = torch.eye(h.shape[1], device=h.device, dtype=h.dtype, requires_grad=True)
            out = torch.einsum('ij,bjhw->bihw', f, out)
            self.source_f_views.append(f)
            if self._capture_hierarchy:
                self._labeled_features[-1] = out
        return h, before, out

    def native_values(self):
        values = self.parent.native_values()
        values['__F__'] = (self.sidecar.effective() if self.sidecar is not None else
                           torch.eye(self.parent.d, device=next(self.parameters()).device))
        return values


class MainTrainer(V1Trainer):
    def __init__(self, *args, history=(), **kwargs):
        super().__init__(*args, **kwargs)
        c = len(self.model.parent_map)
        self.memory = SiteMemory(c, self.model.parent.d, history)
        self.memory.assert_coordinates(self.model.native_values())
        self.history_lineage = [h['domain'] for h in history]
        self.protect_loss = None
        self.protect_grads = None
        self.labeled_grads = None
        self.memory_loss = None
        self.draft = None
        self.bootstrap = False
        self.current_l_images = None
        self.guard_totals = dict(calls=0, damped=0, cone_zero=0, zero_displacement=0)

    def loss_contract(self):
        from pathlib import Path
        inherited=super().loss_contract()
        inherited['reference']='superseded by V2 independent support and actual-displacement guard'
        return dict(version=VERSION, inherited=inherited, guard=GUARD, memory=MEMORY,
                    history_lineage=self.history_lineage,
                    reference='independent geometry/entry confidence; training-L precision and soft historical evidence',
                    implementation=digest({p.name:p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))}))

    def student_images(self, images, donor):
        if self.current_l_images is None:
            self.current_l_images = images.detach()
        return super().student_images(images, donor)

    def losses(self):
        m = self.model
        observe = self.bootstrap or self.step % MEMORY['stride'] == 0
        m.observe = observe; m.source_f_views = []
        for a in m.parent.adapters:
            a.layer.capture = observe; a.layer.views = []
        if m.sidecar is not None:
            m.sidecar.capture = observe; m.sidecar.views = []
        self.draft = None; self.current_l_images = None; self.protect_loss = None
        try:
            return super().losses()
        finally:
            m.observe = False; m.source_f_views = []
            for a in m.parent.adapters:
                a.layer.capture = False; a.layer.views = []
            if m.sidecar is not None:
                m.sidecar.capture = False; m.sidecar.views = []

    def labeled_addition(self, logp, labels, mix_mask):
        m = self.model
        views = list(m._labeled_features)
        hierarchy = super().labeled_addition(logp, labels, mix_mask)
        self.memory_loss = self.memory.penalty(m.native_values())
        if m.observe:
            native_views = {}
            for name, p in m.parent.native.named_parameters():
                module, _, key = name.rpartition('.'); layer = m.parent.native.get_submodule(module)
                if isinstance(layer, ObservedConv):
                    if key == 'weight':
                        native_views[name] = [v for v in layer.views if v.requires_grad]
                elif p.requires_grad:
                    native_views[name] = [p]
            native_views['__F__'] = (m.source_f_views if m.sidecar is None else m.sidecar.views)
            items = [(n,v) for n, vs in native_views.items() for v in vs if v.requires_grad]
            h = views[0]
            mapped = m.parent.output_to_feature(labels, h.shape[-2:], True)
            if mix_mask is not None:
                mask = m.parent.output_to_feature(mix_mask, h.shape[-2:], True)
                h = collect_sources(views[0], views[1], mask)[0]
                hi = F.max_pool2d(mask[:,None].float(),3,1,1)
                lo = -F.max_pool2d(-mask[:,None].float(),3,1,1)
                mapped = mapped.masked_fill((hi != lo)[:,0],255)
            classes = []
            for c in range(len(m.parent_map)):
                pixels = labels == c
                fm = mapped == c; den = fm.sum((-2,-1)); present = den > 0
                if not pixels.any() or not present.any():
                    continue
                objective = -(logp[:,c]*pixels).sum()/pixels.sum()
                grads = torch.autograd.grad(objective, [v for _,v in items], allow_unused=True, retain_graph=True)
                combined = {}
                for (name,_), g in zip(items, grads):
                    if g is not None:
                        combined[name] = combined.get(name,0)+g.detach()
                means = (h.detach()*fm[:,None]).sum((-2,-1))/den.clamp_min(1)[:,None]
                classes.append((c, combined, means[present]))
                self.telemetry['memory_class_vjps'] = self.telemetry.get('memory_class_vjps',0)+1
            calibration = None
            if m.entry_anchor is not None:
                with torch.no_grad():
                    prediction = m.entry_anchor(self.current_l_images).argmax(1)
                    counts = torch.stack([((prediction == c)&(labels != 255)).sum() for c in range(len(m.parent_map))]).float()
                    correct = torch.stack([((prediction == c)&(labels == c)).sum() for c in range(len(m.parent_map))]).float()
                calibration = counts, correct
                self.telemetry['teacher_full_forwards'] += 1
            self.draft = dict(classes=classes, calibration=calibration)
            if not finite_tree(self.draft):
                raise FloatingPointError('nonfinite training-L memory observation')
        m.observe = False
        for a in m.parent.adapters:
            a.layer.capture = False
        if m.sidecar is not None:
            m.sidecar.capture = False
        return (hierarchy if hierarchy is not None else self.memory_loss*0)+MEMORY['weight']*self.memory_loss

    def pseudo_admission(self, target, geometry, pas_valid):
        self.u_geometry = geometry.detach()
        return super().pseudo_admission(target, geometry, pas_valid)

    def kl_objective(self, images, logits, target, valid):
        anchor = self.model.entry_anchor
        if anchor is None:
            loss, _, stats = weighted_kl(logits, target, valid)
            self.last['pseudo'] = stats
            return loss
        with torch.no_grad():
            h = anchor.parent.features(images, 'eval')
            h = torch.einsum('ij,bjhw->bihw', anchor.transform, h)
            q = anchor.parent.native_readout(h, images.shape[-2:], 'eval').softmax(1)
            labels = q.argmax(1); support = self.u_geometry & (q.max(1).values >= .7)
            hf = anchor.parent.feature_to_output(h, images.shape[-2:])
            evidence = self.memory.evidence(hf, labels)
            reliability = self.memory.reliability(logits.device)
        self.telemetry['teacher_full_forwards'] += 1
        loss, _, stats = weighted_kl(logits, target, valid, q)
        self.protect_loss,counts,support = independent_reference(logits,q,self.u_geometry,reliability,evidence)
        stats.update(reference_class_pixels=counts,
                     reference_conflict_pixels=int((support&(labels != target.argmax(1))).sum()),
                     no_foreground_reference=not any(counts[1:]),
                     entry_training_L_precision=reliability.tolist())
        self.telemetry['no_foreground_reference_batches']=self.telemetry.get('no_foreground_reference_batches',0)+int(not any(counts[1:]))
        self.telemetry['reference_conflict_pixels']=self.telemetry.get('reference_conflict_pixels',0)+stats['reference_conflict_pixels']
        self.last['pseudo'] = stats
        return loss

    def gradient_audit(self, labeled, unlabeled):
        params = [p for p in self.model.parameters() if p.requires_grad]
        protection = MEMORY['weight']*self.memory_loss
        if self.protect_loss is not None:
            protection = protection+self.protect_loss
        self.protect_grads = (torch.autograd.grad(protection, params, allow_unused=True, retain_graph=True)
                              if protection.requires_grad else (None,)*len(params))
        self.telemetry['entry_reference_vjps'] += int(self.protect_loss is not None)

    def transform_gradients(self, grads):
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.labeled_grads = [torch.zeros_like(p) if grads[id(p)]['L'] is None else grads[id(p)]['L'].detach() for p in params]
        for teacher in (self.ema,self.model.entry_anchor):
            if teacher is not None and (teacher.training or any(p.requires_grad or p.grad is not None for p in teacher.parameters())):
                raise PermissionError('teacher must remain frozen')

    def transform_update(self, before):
        params = [p for p in self.model.parameters() if p.requires_grad]
        refs = [torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,self.protect_grads)]
        sidecar = self.model.sidecar
        stats = constrain_displacement(params, before, [refs,self.labeled_grads],
                                       lambda:sidecar is None or sidecar.valid())
        if sidecar is not None:
            stats.update(G=spectral_stats(sidecar.matrices()[0]), F=spectral_stats(sidecar.matrices()[1]))
        self.last['actual_proposal_guard'] = stats
        self.guard_totals['calls'] += 1
        self.guard_totals['damped'] += int(stats['scale'] < 1)
        for key in ('cone_zero','zero_displacement'):
            self.guard_totals[key] += int(stats[key])
        for key,value in self.guard_totals.items():
            self.telemetry['v2_guard_'+key]=value

    def update(self, *args, **kwargs):
        result = super().update(*args, **kwargs)
        skipped = kwargs.get('skip', args[0] if args else False)
        if not skipped and self.draft is not None:
            try:
                self.memory.commit(self.draft)
            except Exception:
                self.requires_restore = True
                raise
        self.draft = None
        return result
