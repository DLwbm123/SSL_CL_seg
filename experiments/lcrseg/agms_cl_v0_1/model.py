"""Training-only heads; inherited native main forward and deployment unchanged."""
from contextlib import contextmanager
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from ..five_frameworks_v1.model import Model
from .protocol import ARMS


class AGMSModel(Model):
    def __init__(self, parent, arm, seed, order, stage, initialize_heads=True):
        if arm not in ARMS:
            raise ValueError('unknown arm')
        super().__init__(parent, 'B2_PARENT_PAS_KL')
        self.arm = arm
        self.aux = nn.ModuleList()
        if ARMS[arm]['M']:
            # fork_rng protects the original CPU stream; heads constructed on CPU.
            key = f'AGMS_CL_V0_1/aux/{seed}/{order}/{stage}'
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % (2**63-1))
                self.aux.extend(nn.Conv2d(c, 3, 1, bias=True) for c in (64, 32))
            if not initialize_heads:
                # Values must be loaded before construction of the resume trainer.
                with torch.no_grad():
                    for p in self.aux.parameters():p.fill_(float('nan'))

    def configure_training(self):
        super().configure_training()
        self.aux.requires_grad_(True)

    def u_parameters(self):
        g = self.parent.parameter_groups()
        return g['input_factors'] + g['output_factors']

    @contextmanager
    def capture(self):
        values = {}
        handles = []
        if self.aux:
            for name in ('dec3', 'dec2'):
                def hook(module, args, output, name=name):values[name] = output
                handles.append(getattr(self.parent.native.decoder, name).register_forward_hook(hook))
        try:yield values
        finally:
            for h in handles:h.remove()

    def read_aux(self, values, shape):
        return [F.interpolate(head(values[name]), size=shape, mode='bilinear', align_corners=True).log_softmax(1)
                for name, head in zip(('dec3', 'dec2'), self.aux)]
