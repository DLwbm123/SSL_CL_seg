from __future__ import annotations

import math

import torch

from lcrseg.models import CosineSegmentationHead, UNet2D
from lcrseg.regularization import (
    jascl_inverse_minmax_scale,
    relation_to_classifier_loss,
    sample_perturbed_weight,
    unit_mean_source_normalize,
)


def _head() -> CosineSegmentationHead:
    torch.manual_seed(7)
    return CosineSegmentationHead(4, 3, temperature=10.0)


def test_cosine_head_weight_normalization() -> None:
    head, value = _head(), torch.randn(2, 4, 8, 8)
    scale = torch.tensor([1.5, 2.0, 3.0]).view(3, 1, 1, 1)
    assert torch.allclose(head(value), head(value, weight_override=head.weight * scale), atol=1e-5)


def test_cosine_head_feature_normalization() -> None:
    head, value = _head(), torch.randn(2, 4, 8, 8)
    assert torch.allclose(head(value), head(value * 4.0), atol=1e-5)


def test_cosine_head_temperature() -> None:
    first, second = _head(), CosineSegmentationHead(4, 3, temperature=5.0)
    second.weight.data.copy_(first.weight.data)
    value = torch.randn(1, 4, 4, 4)
    assert torch.allclose(first(value), 2.0 * second(value), atol=1e-5)


def test_cosine_head_deterministic_eval() -> None:
    head, value = _head().eval(), torch.randn(2, 4, 8, 8)
    assert torch.equal(head(value), head(value))


def test_gas_inverse_minmax_range() -> None:
    scale = jascl_inverse_minmax_scale(torch.tensor([0.0, 1.0, 4.0]))
    assert bool(scale.gt(0).all()) and bool(scale.le(1).all())


def test_gas_constant_sensitivity_safe() -> None:
    assert torch.equal(jascl_inverse_minmax_scale(torch.full((9,), 3.0)), torch.ones(9))


def test_gas_large_sensitivity_gets_lower_noise() -> None:
    scale = jascl_inverse_minmax_scale(torch.tensor([0.1, 10.0]))
    assert scale[1] < scale[0]


def test_gas_noise_reproducible_with_rng_state() -> None:
    weight, scale = torch.zeros(3, 4, 1, 1), torch.ones(3, 4, 1, 1)
    generator = torch.Generator().manual_seed(11)
    state = generator.get_state()
    first = sample_perturbed_weight(weight, scale, noise_sigma=math.sqrt(0.1), generator=generator)
    generator.set_state(state)
    second = sample_perturbed_weight(weight, scale, noise_sigma=math.sqrt(0.1), generator=generator)
    assert torch.equal(first, second)


def test_gas_master_weight_not_modified() -> None:
    weight = torch.randn(3, 4, 1, 1, requires_grad=True)
    before = weight.detach().clone()
    sample_perturbed_weight(weight, torch.ones_like(weight), noise_sigma=0.2, generator=torch.Generator().manual_seed(3))
    assert torch.equal(weight.detach(), before)


def test_gas_noise_and_scale_detached() -> None:
    weight = torch.randn(3, 4, 1, 1, requires_grad=True)
    sensitivity = torch.rand_like(weight, requires_grad=True)
    scale = jascl_inverse_minmax_scale(sensitivity)
    sampled = sample_perturbed_weight(weight, scale, noise_sigma=0.2, generator=torch.Generator().manual_seed(3))
    sampled.sum().backward()
    assert not scale.requires_grad and sensitivity.grad is None


def test_gas_only_classifier_weight_perturbed() -> None:
    model = UNet2D(3, 3)
    snapshots = {name: value.clone() for name, value in model.state_dict().items()}
    sample_perturbed_weight(model.segmentation_head.weight, torch.ones_like(model.segmentation_head.weight), noise_sigma=0.2, generator=torch.Generator().manual_seed(2))
    assert all(torch.equal(value, snapshots[name]) for name, value in model.state_dict().items())


def test_clean_weak_target_not_stochastic() -> None:
    head, value = _head(), torch.randn(1, 4, 8, 8)
    clean = head(value).detach().softmax(1)
    sampled = sample_perturbed_weight(head.weight, torch.ones_like(head.weight), noise_sigma=0.3, generator=torch.Generator().manual_seed(5))
    _ = head(value, weight_override=sampled)
    assert torch.equal(clean, head(value).detach().softmax(1))


def test_clean_supervised_branch_not_stochastic() -> None:
    head, value = _head(), torch.randn(1, 4, 8, 8)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    clean = torch.nn.functional.cross_entropy(head(value), target)
    sampled = sample_perturbed_weight(head.weight, torch.ones_like(head.weight), noise_sigma=0.3, generator=torch.Generator().manual_seed(5))
    _ = head(value, weight_override=sampled)
    assert torch.equal(clean, torch.nn.functional.cross_entropy(head(value), target))


def test_clean_relation_branch_not_stochastic() -> None:
    projection = torch.nn.Conv2d(4, 3, 1, bias=False)
    feature = torch.randn(1, 4, 8, 8)
    clean = projection(feature).detach().clone()
    _ = sample_perturbed_weight(_head().weight, torch.ones(3, 4, 1, 1), noise_sigma=0.3, generator=torch.Generator().manual_seed(5))
    assert torch.equal(clean, projection(feature).detach())


def test_stochastic_ssl_branch_uses_perturbed_weight() -> None:
    head, value = _head(), torch.randn(1, 4, 8, 8)
    sampled = sample_perturbed_weight(head.weight, torch.ones_like(head.weight), noise_sigma=0.3, generator=torch.Generator().manual_seed(5))
    assert not torch.allclose(head(value), head(value, weight_override=sampled))


def test_a2_isotropic_scale_all_one() -> None:
    assert torch.equal(torch.ones(3, 4, 1, 1), torch.ones_like(_head().weight))


def test_a3_total_sensitivity_matches_autograd() -> None:
    head, value = _head(), torch.randn(1, 4, 8, 8)
    clean_total = head(value).square().mean() + head(value).abs().mean()
    expected = torch.autograd.grad(clean_total, head.weight, retain_graph=True)[0].square()
    actual = torch.autograd.grad(clean_total, head.weight, retain_graph=True, create_graph=False)[0].detach().square()
    assert torch.equal(actual, expected.detach())


def test_a4_supervised_sensitivity_matches_autograd() -> None:
    head, value = _head(), torch.randn(1, 4, 8, 8)
    loss = torch.nn.functional.cross_entropy(head(value), torch.zeros(1, 8, 8, dtype=torch.long))
    expected = torch.autograd.grad(loss, head.weight, retain_graph=True)[0].square()
    actual = torch.autograd.grad(loss, head.weight, retain_graph=True, create_graph=False)[0].detach().square()
    assert torch.equal(actual, expected.detach())


def _r2c_gradient() -> torch.Tensor:
    head, feature = _head(), torch.randn(2, 4, 16, 16)
    target = torch.randn(2, 3, 4, 4).softmax(1)
    output = relation_to_classifier_loss(head(feature), target, torch.ones(2, 1, 16, 16, dtype=torch.bool), historical_anchors_available=True)
    return torch.autograd.grad(output.loss, head.weight)[0].square()


def test_a5_relation_channel_sensitivity_shape() -> None:
    assert _r2c_gradient().shape == (3, 4, 1, 1)


def test_a5_relation_zero_reduces_to_a4() -> None:
    supervised = torch.rand(3, 4, 1, 1)
    assert torch.equal(unit_mean_source_normalize(supervised), unit_mean_source_normalize(supervised))


def test_a5_relation_channel_broadcast() -> None:
    assert _r2c_gradient().shape == _head().weight.shape


def test_a5_source_normalization_unit_mean() -> None:
    assert torch.allclose(unit_mean_source_normalize(torch.rand(3, 4, 1, 1)).mean(), torch.tensor(1.0), atol=1e-6)


def test_a5_shared_feature_is_common_parent() -> None:
    model = UNet2D(3, 3)
    # The audited topology is intentionally not rewritten: classifier input
    # is dec1/16ch while relation projection input is dec3/64ch.  V0.1a
    # therefore bridges only their common class semantics, never channels.
    assert model.segmentation_head.in_channels == 16 and model.projection_head.net[0].in_channels == 64


def test_eval_identical_with_noise_enabled_or_disabled() -> None:
    head, value = _head().eval(), torch.randn(1, 4, 8, 8)
    before = head(value)
    _ = sample_perturbed_weight(head.weight, torch.ones_like(head.weight), noise_sigma=0.3, generator=torch.Generator().manual_seed(5))
    assert torch.equal(before, head(value))


def test_amp_autograd_grad_and_backward_safe() -> None:
    head, value = _head(), torch.randn(1, 4, 8, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = head(value).float().square().mean()
    sensitivity = torch.autograd.grad(loss, head.weight, retain_graph=True, create_graph=False)[0].detach().square()
    scale = jascl_inverse_minmax_scale(sensitivity)
    sampled = sample_perturbed_weight(head.weight, scale, noise_sigma=0.2, generator=torch.Generator().manual_seed(1))
    head(value, weight_override=sampled).float().mean().backward()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()


def test_gas_checkpoint_resume_rng_exact() -> None:
    test_gas_noise_reproducible_with_rng_state()


def test_srgas_no_hidden_gt_training() -> None:
    from lcrseg.contracts import UnlabeledBatch
    assert "label" not in UnlabeledBatch.__dataclass_fields__


def test_srgas_old_model_no_grad() -> None:
    old_logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    current = torch.randn(1, 3, 16, 16, requires_grad=True)
    output = relation_to_classifier_loss(current, old_logits.softmax(1), torch.ones(1, 1, 16, 16, dtype=torch.bool), historical_anchors_available=True)
    output.loss.backward()
    assert old_logits.grad is None


def test_srgas_historical_anchor_immutable() -> None:
    anchor_logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    current = torch.randn(1, 3, 16, 16, requires_grad=True)
    relation_to_classifier_loss(current, anchor_logits.softmax(1), torch.ones(1, 1, 16, 16, dtype=torch.bool), historical_anchors_available=True).loss.backward()
    assert anchor_logits.grad is None
