import copy
import hashlib
import os
import tempfile
from pathlib import Path
import torch
from .config import Config
from .conditional_memory import ConditionalDomainMemory
from .global_posterior import GlobalPosterior, FixedProjection
from .generator import GenerativeModel, layout
from .recognition import Recognition
from .evidence import validate_evidence


def digest(obj):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(str((v.dtype, tuple(v.shape))).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode()); visit(v[k])
        elif isinstance(v, (tuple, list)):
            for w in v: visit(w)
        else:
            h.update(repr(v).encode())
    visit(obj)
    return h.hexdigest()


class StreamState:
    schema = 2

    def __init__(self, config):
        self.config = config
        self.generator = torch.Generator().manual_seed(config.seed)
        self.q = GlobalPosterior(layout(config), config, self.generator)
        self.projection = FixedProjection(self.q, config.projection_dim, config.projection_seed)
        self.model = GenerativeModel(config, self.q)
        # Constructor initialization must not advance caller RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.phi = Recognition(config, self.projection)
        self.memories = {}
        self.receipts = {}
        self.active = None
        self.prior = None
        self.current = None
        self.optimizer = None
        self.steps = 0
        self.runtime = None

    def to_device(self, device):
        for module in (self.q, self.projection, self.model, self.phi, self.prior,
                       self.current, *self.memories.values()):
            if module is not None:
                module.to(device)
        if self.optimizer is not None:
            for values in self.optimizer.state.values():
                for key, value in values.items():
                    if isinstance(value, torch.Tensor) and key != 'step':
                        values[key] = value.to(device)
        return self

    def begin_task(self, stage_id, manifest_hash, evidence=None):
        if self.active is not None or stage_id in self.receipts or not stage_id:
            raise ValueError('active or duplicate stage')
        if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
            raise ValueError('manifest SHA256 required')
        if evidence is None:
            # Legacy algebra fixtures have no record manifest. Real training always supplies event identities.
            if any(r.get('manifest_hash') == manifest_hash for r in self.receipts.values()):
                raise ValueError('legacy manifest already consumed; explicit event identities required')
        else:
            events = validate_evidence(evidence)
            if any('evidence' not in r or r['evidence'] is None for r in self.receipts.values()):
                raise ValueError('historical evidence identities unavailable; continuation needs audited migration')
            if any(events.intersection(validate_evidence(r['evidence'])) for r in self.receipts.values()):
                raise ValueError('observation evidence already consumed')
        self.prior = self.q.frozen_copy()
        if not self.receipts:
            with torch.no_grad():
                from .distributions import inverse_positive
                self.prior.mean.zero_()
                self.prior.raw_scale.copy_(inverse_positive(self.q.prior_sd, self.config.scale_floor))
        self.prior_hash = digest(self.prior.state_dict())
        self.projection_hash = self.projection.fingerprint()
        self.current = ConditionalDomainMemory(self.projection, self.config.channels, self.config.scale_floor)
        self.active = {'id': stage_id, 'manifest_hash': manifest_hash, 'evidence': copy.deepcopy(evidence)}
        self.steps = 0
        self.optimizer = torch.optim.Adam([
            {'params': list(self.q.parameters()), 'name': 'global-generative'},
            {'params': list(self.current.parameters()), 'name': 'current-domain-variational'},
            {'params': list(self.phi.parameters()), 'name': 'amortizer'}], lr=self.config.outer_lr)
        self.runtime = None

    def assert_active(self):
        if self.active is None:
            raise RuntimeError('no active task')
        if digest(self.prior.state_dict()) != self.prior_hash or self.projection.fingerprint() != self.projection_hash:
            raise RuntimeError('frozen prior/projection changed')

    def check_batch(self, batch, role):
        if batch.get('stage') != self.active['id'] or batch.get('role') != 'train' + role:
            raise PermissionError('only current stage training observations')
        if role == 'U' and any(k in batch for k in ('observations', 'label', 'labels', 'y', 'label_path')):
            raise PermissionError('U batch has a label capability')

    def finalize_task(self, cost=None):
        self.assert_active()
        stage = self.active['id']
        self.memories[stage] = copy.deepcopy(self.current).requires_grad_(False)
        self.receipts[stage] = dict(self.active, steps=self.steps, projection=self.projection_hash,
                                    posterior=digest(self.q.state_dict()), cost=cost or {})
        self.active = None
        self.current = None
        self.optimizer = None
        self.prior = None
        return self.receipts[stage]

    def payload(self):
        return dict(schema=self.schema, config=self.config.to_dict(), q=self.q.state_dict(),
                    phi=self.phi.state_dict(), model=self.model.state_dict(), projection=self.projection.state_dict(),
                    projection_hash=self.projection.fingerprint(), memories={k: v.state_dict() for k, v in self.memories.items()},
                    receipts=self.receipts, active=self.active, steps=self.steps, rng=self.generator.get_state(), runtime=self.runtime,
                    prior=None if self.prior is None else self.prior.state_dict(),
                    current=None if self.current is None else self.current.state_dict(),
                    optimizer=None if self.optimizer is None else self.optimizer.state_dict())

    def save(self, path):
        path = Path(path)
        fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as f:
                torch.save(self.payload(), f)
                f.flush()
                os.fsync(f.fileno())
            # Atomic, create-only publication; an interrupted save cannot corrupt the last checkpoint.
            os.link(temporary, path)
        finally:
            os.unlink(temporary)

    @classmethod
    def load(cls, path):
        p = torch.load(path, map_location='cpu', weights_only=True)
        if p['schema'] not in (1, cls.schema):
            raise ValueError('checkpoint schema')
        state = cls(Config(**p['config']))
        if p['projection_hash'] != state.projection.fingerprint():
            raise ValueError('projection metadata mismatch')
        for name in ('q', 'phi', 'model', 'projection'):
            getattr(state, name).load_state_dict(p[name], strict=True)
        if state.projection.fingerprint() != p['projection_hash']:
            raise ValueError('projection content mismatch')
        state.receipts = p['receipts']
        for k, v in p['memories'].items():
            memory = ConditionalDomainMemory(copy.deepcopy(state.projection), state.config.channels, state.config.scale_floor)
            memory.load_state_dict(v, strict=True)
            state.memories[k] = memory.requires_grad_(False)
        if p['active'] is not None:
            state.begin_task(p['active']['id'], p['active']['manifest_hash'], p['active'].get('evidence'))
            state.prior.load_state_dict(p['prior'], strict=True)
            state.prior_hash = digest(state.prior.state_dict())
            state.current.load_state_dict(p['current'], strict=True)
            state.optimizer.load_state_dict(p['optimizer'])
            state.steps = p['steps']
        state.generator.set_state(p['rng'])
        state.runtime = p.get('runtime')
        return state
