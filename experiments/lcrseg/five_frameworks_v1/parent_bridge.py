"""Explicit bridge protocol and synthetic implementation; NOT original KI."""
from typing import Protocol
import torch
from torch import nn
from torch.nn import functional as F


class ParentBindingRequired(RuntimeError):
    pass


class ParentBridge(Protocol):
    def features(self, x, mode): ...
    def native_readout(self, h, output_shape, mode): ...
    def effective_readout_kernel_at_entry(self): ...
    def parameter_groups(self): ...
    def supervised(self, logp, labels): ...
    def configure_stage_training(self): ...
    def semantic_metadata(self): ...
    def stage_entry(self): ...
    def stage_exit(self): ...
    def constraint_loss(self): ...
    def apply_constraints(self): ...
    def output_to_feature(self, x, feature_shape, categorical=False): ...
    def feature_to_output(self, x, output_shape): ...
    def optimizer_groups(self, options): ...


def bind_real_parent(binding):
    # No data loader or tensor deserialization is reachable from an unbound bridge.
    raise ParentBindingRequired('PARENT_BINDING_REQUIRED: one original KI command or run_id')


class ToyAdapter(nn.Module):
    def __init__(self, d, rank):
        super().__init__()
        self.register_buffer('base', torch.eye(d))
        self.a = nn.Parameter(torch.randn(rank, d) * .1)
        self.b = nn.Parameter(torch.zeros(d, rank))
        self.new = True
        self.free_projector = None  # Toy has no hard/soft orthogonality constraint.

    def forward(self, x, effective=None):
        w = self.base + self.b @ self.a if effective is None else effective
        return F.relu(torch.einsum('ij,bjhw->bihw', w, x))

    @torch.no_grad()
    def seal(self):
        self.base.add_(self.b @ self.a)
        self.b.zero_()
        self.new = False


class SyntheticParentBridge(nn.Module):
    """Spatial identity feature grid; 3x3 padding-0 affine readout + resize.

    Output-to-feature correspondence is explicitly identity in normalized full
    image coordinates. This is a synthetic convention, not a verified mapping
    for the unknown real network. Readout cropping is NOT silently overridden.
    """
    synthetic = True
    def __init__(self, d=4, rank=2):
        super().__init__()
        self.d = d
        self.stem = nn.Conv2d(3, d, 1)
        self.adapters = nn.ModuleList([ToyAdapter(d, rank) for _ in range(3)])
        self.readout = nn.Conv2d(d, 3, 3, padding=0, bias=True)
        self.stem.requires_grad_(False)
        self.readout.requires_grad_(False)

    def features(self, x, mode='student', overrides=None):
        h = F.relu(self.stem(x))
        for i, adapter in enumerate(self.adapters):
            h = adapter(h, None if overrides is None else overrides.get(i))
        return h

    def native_readout(self, h, output_shape, mode='student'):
        return F.interpolate(self.readout(h), size=output_shape,
                             mode='bilinear', align_corners=True)

    def native(self, x):
        return self.native_readout(self.features(x), x.shape[-2:])

    def effective_readout_kernel_at_entry(self):
        return self.readout.weight.detach()

    def parameter_groups(self):
        return {'input_factors': [a.a for a in self.adapters],
                'output_factors': [a.b for a in self.adapters],
                'other_allowed': [],
                'frozen': [p for p in self.parameters() if not p.requires_grad]}

    def optimizer_groups(self, options):
        # Toy optimizer only: real optimizer cannot be inferred from this fixture.
        lr = options.get('lr', .01) * options.get('parent_lr_multiplier', 1.)
        groups = self.parameter_groups()
        return [{'params': groups['input_factors'], 'lr': lr, 'name': 'A'},
                {'params': groups['output_factors'],
                 'lr': lr * options.get('lr_B_over_A', 1.), 'name': 'B'}]

    def supervised(self, logp, labels):
        valid = labels != 255
        safe = labels.masked_fill(~valid, 0)
        ce = -(logp.gather(1, safe[:, None]).squeeze(1) * valid).sum() / valid.sum().clamp_min(1)
        probs = logp.exp()
        target = F.one_hot(safe, 3).permute(0, 3, 1, 2).to(probs)
        p, y = probs[:, 1:] * valid[:, None], target[:, 1:] * valid[:, None]
        dice = 1 - (2*(p*y).sum((-2,-1)) + 1e-6) / (p.sum((-2,-1))+y.sum((-2,-1))+1e-6)
        active = valid.flatten(1).any(1)
        return ce + (dice.mean(1)*active).sum()/active.sum().clamp_min(1)

    def constraint_loss(self):
        return sum(a.a.sum()*0 for a in self.adapters)

    def apply_constraints(self):
        pass  # Explicitly unconstrained synthetic parent, not a KI constraint stub.

    def configure_stage_training(self):
        # Restore legal flags/modes without touching values, new flags or probes.
        self.train()
        self.requires_grad_(False)
        for adapter in self.adapters:
            adapter.a.requires_grad_(True); adapter.b.requires_grad_(True)
        self.stem.eval(); self.readout.eval()

    def semantic_metadata(self):
        from .semantics import tensor_fingerprint
        return {'bridge':type(self).__module__+'.'+type(self).__qualname__,
                'synthetic':True,'d':self.d,'rank':[a.a.shape[0] for a in self.adapters],
                'delta':'B @ A','constraint':'none_synthetic',
                'free_projectors':[None if a.free_projector is None else tensor_fingerprint({'P':a.free_projector}) for a in self.adapters],
                'readout':{'kernel':list(self.readout.kernel_size),'padding':list(self.readout.padding),
                           'bias':self.readout.bias is not None,'align_corners':True},
                'loss':'toy_CE_Dice_foreground_smooth_1e-6',
                'geometry':'normalized_full_image_coordinate_v1'}

    @torch.no_grad()
    def stage_entry(self):
        for a in self.adapters:
            if a.b.count_nonzero():
                raise ValueError('cannot enter a stage with an unsealed active adapter')
            a.new = True
            a.a.requires_grad_(True); a.b.requires_grad_(True)

    @torch.no_grad()
    def stage_exit(self):
        for a in self.adapters:
            a.seal()

    def output_to_feature(self, value, feature_shape, categorical=False):
        shape = value.shape
        x = value[:, None] if value.ndim == 3 else value
        mode = 'nearest' if categorical else 'bilinear'
        y = F.interpolate(x.float(), size=feature_shape, mode=mode,
                          **({} if categorical else {'align_corners': True}))
        return (y[:, 0] if len(shape) == 3 else y).to(value.dtype)

    def feature_to_output(self, value, output_shape):
        return F.interpolate(value, size=output_shape, mode='bilinear', align_corners=True)
