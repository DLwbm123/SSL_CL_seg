"""Registered same-draw gradient precision; native scoring is never replaced."""
from contextlib import contextmanager
import copy
from unittest.mock import patch

import numpy as np
import torch

from di_dmpa_gate1.feature_extraction import seed_after_load, state_hash
from di_dmpa_jascl.checkpoint import capture_rng_state
from . import binding as b


def rng_hash():
    state = capture_rng_state(); npstate = state['numpy']
    return b.H(dict(python=state['python'], numpy=[npstate[0], npstate[1].tolist(), *npstate[2:]],
        torch_cpu=b.tensor_hash(state['torch_cpu']), torch_cuda=[b.tensor_hash(x) for x in state['torch_cuda']]))


@contextmanager
def replay_draw(draw):
    calls = 0
    b.require(draw.dtype == torch.float32 and not draw.requires_grad, 'source draw must be detached float32')

    def replay(value, *args, **kwargs):
        nonlocal calls
        b.require(calls == 0 and not args and not kwargs and value.shape == draw.shape and value.dtype == torch.float64, 'unexpected reference Gaussian call')
        calls += 1
        return draw.to(device=value.device, dtype=value.dtype)

    with patch.object(torch, 'randn_like', replay):
        yield
    b.require(calls == 1, 'reference did not consume exactly one captured draw')


def compare(native, reference):
    native = np.asarray(native, np.float64).reshape(-1); reference = np.asarray(reference, np.float64).reshape(-1)
    b.require(native.shape == reference.shape and native.size > 0, 'reference comparison geometry')
    b.finite(native, reference)
    nn = float(np.linalg.norm(native)); rn = float(np.linalg.norm(reference)); delta = native-reference
    return dict(native_sha256=b.array_hash(native), reference_sha256=b.array_hash(reference),
        max_abs_error=float(np.abs(delta).max()), native_l2_norm=nn, reference_l2_norm=rn,
        relative_l2=None if rn == 0 else float(np.linalg.norm(delta))/rn,
        cosine=None if nn == 0 or rn == 0 else float(np.clip(np.dot(native, reference)/(nn*rn), -1, 1)))


def comparable(row):
    if row['native_l2_norm'] == row['reference_l2_norm'] == 0:
        return True
    return bool(row['relative_l2'] is not None and row['cosine'] is not None and
                row['relative_l2'] <= 1e-3 and row['cosine'] >= .9999)


def attach_gradient_student(models, p):
    mode = p.get('diagnostic_precision', 'float32_native')
    b.require(mode in ('float32_native', 'float64_shadow'), 'unknown diagnostic precision')
    b.require('gradient_student' not in models, 'duplicate gradient receiver')
    if mode == 'float32_native':
        return
    b.require(p.get('_precision_contract_verified') is True, 'unregistered precision mode')
    source = models['student']
    b.require(all(not m.training for m in source.modules()), 'shadow requires frozen eval mode')
    before = state_hash(source.state_dict()); rng = rng_hash()
    shadow = copy.deepcopy(source).double().requires_grad_(True)
    expected = {k: v.double() if v.is_floating_point() else v for k, v in source.state_dict().items()}
    b.require(state_hash(shadow.state_dict()) == state_hash(expected), 'shadow conversion changed values')
    b.require(all(x.data_ptr() != y.data_ptr() for x, y in zip(source.parameters(), shadow.parameters())), 'shadow aliases original')
    b.require(before == state_hash(source.state_dict()) and rng == rng_hash(), 'shadow construction changed source/RNG')
    models['gradient_student'] = shadow


def student_forward(models, images, seed):
    seed_after_load(seed)
    source = models['student']; shadow = models.get('gradient_student')
    if shadow is None:
        logits, features = source(images, stochastic_classifier=True)
        return logits, features, logits, None
    original = torch.randn_like; draws = []

    def capture(value, *args, **kwargs):
        b.require(not draws and value is source.decoder.conv_logit.mu.weight and
                  value.dtype == torch.float32 and not args and not kwargs, 'unexpected native classifier draw')
        draw = original(value)
        b.finite(draw); b.require(not draw.requires_grad, 'Gaussian unexpectedly differentiable')
        draws.append(draw.detach().clone())
        return draw

    with patch.object(torch, 'randn_like', capture):
        logits, features = source(images, stochastic_classifier=True)
    b.require(len(draws) == 1 and logits.dtype == features.dtype == torch.float32, 'native forward changed')
    rng_before = rng_hash()
    with replay_draw(draws[0]):
        gradient_logits, gradient_features = shadow(images.double(), stochastic_classifier=True)
    b.finite(gradient_logits, gradient_features)
    b.require(gradient_logits.dtype == gradient_features.dtype == torch.float64 and rng_before == rng_hash(), 'shadow precision/RNG changed')
    receipt = dict(seed=seed, native_gaussian_sha256=b.tensor_hash(draws[0]),
        shadow_gaussian_sha256=b.tensor_hash(draws[0].double()), gaussian_shape=list(draws[0].shape),
        native_logits_sha256=b.tensor_hash(logits), gradient_logits_sha256=b.tensor_hash(gradient_logits),
        rng_before_shadow=rng_before, rng_after_shadow=rng_hash(), native_forwards=1, shadow_forwards=1)
    return logits, features, gradient_logits, receipt
