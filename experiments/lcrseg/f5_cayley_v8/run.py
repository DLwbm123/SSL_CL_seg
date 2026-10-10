"""One Cayley sidecar change; reuse the existing training/readout engine."""
import os
from pathlib import Path
import torch
from ..f5_u_strength_v6.run import StrengthTrainer, engine
from ..f5_module_pilots_v1.modules import make_model as original_model
from ..five_frameworks_v1.native_data import NativeCurrentDomain
from ..five_frameworks_v1.native_parent import NativeLRParent
from ..five_frameworks_v1.recipes import generator
from ..five_frameworks_v1.gate import digest
from .module import CayleyAdapter
from . import protocol


def make_model(parent, options, previous, rng, *, enabled=True):
    model = original_model(parent, 'F5', options, previous, rng)
    model.sidecar = CayleyAdapter(model.sidecar.q, model.sidecar.previous, cayley_enabled=enabled)
    return model


class CayleyTrainer(StrengthTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model.sidecar, CayleyAdapter):
            raise TypeError('actual Cayley sidecar required')

    def loss_contract(self):
        return {**super().loss_contract(),
                'cayley_enabled': self.model.sidecar.cayley_enabled,
                'equation': 'S=(R-R.T)/2; O=solve(I-S,I+S); G=I+Q@(O-I)@Q.T',
                'module_implementation': digest(Path(__file__).with_name('module.py').read_text()),
                'trainer_implementation': digest(Path(__file__).read_text())}

    def transform_gradients(self, grads):
        super().transform_gradients(grads)
        with torch.no_grad():
            f = self.model.sidecar.effective()
            eye = torch.eye(f.shape[0], device=f.device, dtype=f.dtype)
            error = float((f.T @ f - eye).norm())
            if self.model.sidecar.cayley_enabled and error > 2e-4:
                raise FloatingPointError('own Cayley cumulative transform lost orthogonality')
            self.module_stats.update(cayley_enabled=self.model.sidecar.cayley_enabled,
                                     cumulative_orthogonality_error=error)


def construct(config, j, stage, device, permit, previous=None, native=None, source_id=None, allow_u=True):
    if native is None:
        native, sr = engine.source(config, j['seed'], device)
        source_id = dict(node_id=sr['node_id'], student_hash=sr['student_hash'], seed=j['seed'], domain='REFUGE')
    provider = NativeCurrentDomain(config['data'], j['seed'], j['order'], stage, source_id, device, permit, allow_u=allow_u)
    options = {**protocol.OPTIONS, 'total_steps': protocol.STEPS[provider.domain]}
    model = make_model(NativeLRParent(native, j['seed'], source_id), options, previous,
                       generator(j['seed'], j['order'], stage, 0, 'pilot_adapter_initialization')).to(device)
    return CayleyTrainer(model, provider, options, execution=permit)


def bind():
    for name in ('plan', 'jobs', 'CAPS', 'ARMS', 'compare'):
        setattr(engine, name, getattr(protocol, name))
    engine.construct = construct


def main():
    bind()
    config = protocol.read(os.environ['EXEC_CONFIG'])
    if os.environ.get('EXEC_MODE') == 'qualify':
        from .tests import qualify
        qualify(config)
    else:
        engine.main()
