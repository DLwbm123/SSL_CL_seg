"""Independent CARE-Seg V1 integration with the shared continual engine."""
import copy
import torch
from torch.utils.data import Subset
from ..base import ContinualSegMethod, freeze_model
from ...contracts import MethodStepOutput
from .core import DecisionTree
from .statistics import TemporalRisk
from .controller import refresh, resize

DEFAULTS = dict(fit_pixels_per_patient=2048, grid=[32, 32], block_steps=100, ema=.99, minimum_patients=2,
                em_iterations=25, repair_bound=.5, repair_steps=50, repair_lr=.1,
                repair_radius=.2, risk_ridge=1., maximum_risk=.15, risk_margin=.02,
                temporal_momentum=.7, lambda_care=1., geometry='normalized2d', volumes={})


def resolve_config(config=None):
    result = copy.deepcopy(DEFAULTS)
    result.update(config or {})
    for key in ('tree', 'roles'):
        if key not in result:
            raise ValueError(f'CARE requires explicit {key}')
    if result['geometry'] not in ('normalized2d', 'physical3d'):
        raise ValueError('unknown geometry')
    if len(result['grid']) != 2 or min(result['grid']) < 2 or result['block_steps'] < 1:
        raise ValueError('invalid grid/block size')
    if not 0 <= result['ema'] < 1 or not 0 <= result['maximum_risk'] <= 1 or result['risk_margin'] < 0:
        raise ValueError('invalid teacher/risk settings')
    return result


class CARESegMethod(ContinualSegMethod):
    method_name = 'care_seg_v1'
    method_version = '1.0'

    def __init__(self, model, *, config=None):
        super().__init__(model, config=resolve_config(config))
        self.tree = DecisionTree(self.num_classes, self.config['tree'])
        self.ema_model = freeze_model(copy.deepcopy(model))
        self.datasets = None
        self.binding = None
        self.maps, self.fitted, self.accounting = {}, [], {}
        self.last_refresh = -1
        self.ema_updates = 0
        self.temporal = TemporalRisk(self.config['temporal_momentum'])
        self.pending = None

    def prepare_datasets(self, labeled, unlabeled, site_id):
        if unlabeled is None or not len(unlabeled):
            raise ValueError('CARE requires an image-only U split')
        role = self.config['roles'][site_id]
        a, b = set(role['A']), set(role['B'])
        lp = [s.row.get('patient_id') or s.row['case_id'] for s in labeled.samples]
        up = [s.row.get('patient_id') or s.row['case_id'] for s in unlabeled.samples]
        if not a or len(b) < self.config['minimum_patients'] or a & b or (a | b) != set(lp) or (a | b) & set(up):
            raise ValueError('patient roles must partition visible L and be disjoint from U')
        for dataset in (labeled, unlabeled):
            for s in dataset.samples:
                meta = dataset._metadata(s)
                if meta['site'] != site_id:
                    raise ValueError('future/other site in controller data')
        ai = [i for i, p in enumerate(lp) if p in a]
        bi = [i for i, p in enumerate(lp) if p in b]
        canonical_l, canonical_u = copy.copy(labeled), copy.copy(unlabeled)
        canonical_l.transform = canonical_u.transform = None
        self.datasets = (Subset(canonical_l, ai), Subset(canonical_l, bi), canonical_u)
        self.binding = (site_id, tuple(sorted(a)), tuple(sorted(b)),
                        tuple((s.row['case_id'], s.slice_index) for s in unlabeled.samples))
        return Subset(labeled, ai), unlabeled

    def begin_site(self, site_id, previous_checkpoint, total_steps):
        super().begin_site(site_id, previous_checkpoint, total_steps)
        if previous_checkpoint is not None:
            self._make_old_model()
        self.ema_model = freeze_model(copy.deepcopy(self.model))
        self.maps, self.fitted, self.accounting = {}, [], {}
        self.last_refresh, self.ema_updates = -1, 0
        self.temporal = TemporalRisk(self.config['temporal_momentum'])
        self.pending = None

    def before_training_step(self, site_step):
        if self.datasets is None:
            raise RuntimeError('prepare patient roles before training')
        block = site_step // self.config['block_steps']
        if block == self.last_refresh:
            return
        self.ema_model.eval()
        if self.old_model is not None:
            self.old_model.eval()
        self.maps, self.fitted, self.accounting = refresh(
            self.tree, self.config, self.ema_model, self.old_model, self.datasets, self.temporal)
        self.last_refresh = block

    def aligned_entries(self, batch):
        entries = []
        device = batch.strong_image.device
        for case, index, record in zip(batch.case_id, batch.slice_index, batch.geometry_record, strict=True):
            if set(record) - {'hflip', 'vflip', 'cutout', 'cutout_box'}:
                raise ValueError('unsupported geometry transform')
            current = []
            for entry in self.maps[(case, index)]:
                output = {}
                for key, value in entry.items():
                    if value is None:
                        output[key] = None
                        continue
                    scalar = value.ndim == 2
                    value = resize(value[None] if scalar else value, batch.strong_image.shape[-2:], 'nearest')
                    if scalar:
                        value = value[0]
                    if record.get('hflip'):
                        value = value.flip(-1)
                    if record.get('vflip'):
                        value = value.flip(-2)
                    output[key] = value.to(device)
                current.append(output)
            entries.append(current)
        return entries

    @torch.no_grad()
    def drift(self, image, entries):
        was_training = self.model.training
        self.model.eval()
        try:
            p = self.model(image).logits.float().softmax(1)
        finally:
            self.model.train(was_training)
        numerator = p.new_zeros(())
        support = p.new_zeros(())
        for i, decisions in enumerate(entries):
            for node, entry in zip(self.tree.nodes, decisions, strict=True):
                if entry['historical'] is None:
                    continue
                q, _ = self.tree.read(p[i], node)
                h, _ = self.tree.read(entry['historical'], node)
                keep = entry['keep']
                numerator += ((h*(h.clamp_min(1e-8).log()-q.clamp_min(1e-8).log())).sum(0)*keep).sum()
                support += keep.sum()
        return float(numerator/support.clamp_min(1)), float(support/(p.shape[0]*p.shape[2]*p.shape[3]*len(self.tree.nodes)))

    def training_step(self, labeled_batch, unlabeled_batch, global_step, site_step):
        if self.last_refresh != site_step // self.config['block_steps']:
            raise RuntimeError('controller must refresh before training step')
        output = self.model(labeled_batch.image)
        losses = self._supervised_losses(output, labeled_batch)
        care = output.logits.sum()*0
        scalars = dict(self.accounting)
        self.pending = None
        if unlabeled_batch is not None:
            entries = self.aligned_entries(unlabeled_batch)
            # Fixed weak-view K is held unchanged across the complete optimizer update.
            before, coverage = self.drift(unlabeled_batch.weak_image, entries)
            logits = self.model(unlabeled_batch.strong_image).logits
            care = self.tree.loss(logits, entries, unlabeled_batch.strong_valid_mask)
            scalars.update(drift_before=before, drift_coverage=coverage)
            self.pending = (unlabeled_batch.weak_image.detach(), entries, scalars, before)
        losses['loss_assim'] = care
        return MethodStepOutput(losses['loss_sup']+self.config['lambda_care']*care,
                                self._all_loss_keys(output.logits, losses), scalars)

    @torch.no_grad()
    def after_optimizer_step(self):
        if self.pending is not None:
            image, entries, scalars, before = self.pending
            after, _ = self.drift(image, entries)
            scalars.update(drift_after=after, drift_delta=after-before)
            self.pending = None
        for target, source in zip(self.ema_model.parameters(), self.model.parameters(), strict=True):
            target.lerp_(source, 1-self.config['ema'])
        for target, source in zip(self.ema_model.buffers(), self.model.buffers(), strict=True):
            target.copy_(source)
        self.ema_updates += 1

    def method_state_dict(self):
        state = super().method_state_dict()
        state['method_statistics']['care'] = copy.deepcopy(dict(config=self.config, binding=self.binding,
            ema=self.ema_model.state_dict(), maps=self.maps, fitted=self.fitted,
            accounting=self.accounting, temporal=self.temporal, last_refresh=self.last_refresh,
            ema_updates=self.ema_updates))
        return state

    def load_method_state_dict(self, state):
        saved = state['method_statistics']['care']
        if saved['config'] != self.config or saved['binding'] != self.binding:
            raise ValueError('checkpoint config or patient binding changed')
        super().load_method_state_dict(state)
        self.ema_model.load_state_dict(saved['ema'])
        for key in ('maps', 'fitted', 'accounting', 'temporal', 'last_refresh', 'ema_updates'):
            setattr(self, key, copy.deepcopy(saved[key]))
        self.pending = None
