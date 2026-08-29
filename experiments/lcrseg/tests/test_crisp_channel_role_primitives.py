from __future__ import annotations

import io

import torch

from lcrseg.losses.channel_role_consistency import (
    invariant_feature_consolidation,
    plastic_feature_consistency,
)
from lcrseg.representation.channel_roles import (
    FEATURE_LAYERS,
    ChannelRoleState,
    build_channel_role_state,
    case_equal_mean,
    content_relevance_case,
    continuous_channel_roles,
    hard_rank_roles,
    jaccard,
    quartile_indices,
    spearman_correlation,
    stable_half_assignment,
    style_sensitivity_case,
    uniform_half_roles,
)


def test_content_relevance_is_squared_activation_gradient_and_case_equal() -> None:
    feature_a = torch.tensor([[[[1.0, 2.0]]], [[[2.0, 1.0]]]])
    gradient_a = torch.tensor([[[[3.0, 4.0]]], [[[5.0, 6.0]]]])
    feature_b = torch.ones_like(feature_a) * 2.0
    gradient_b = torch.ones_like(feature_a) * 3.0
    first = content_relevance_case(feature_a, gradient_a)
    second = content_relevance_case(feature_b, gradient_b)
    assert torch.allclose(first, (feature_a * gradient_a).square().mean(dim=(0, 2, 3)))
    assert torch.equal(case_equal_mean([first, second]), torch.stack([first, second]).mean(dim=0))


def test_style_sensitivity_uses_centered_spatial_unit_maps() -> None:
    clean = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [0.0, 1.0]]]])
    style = torch.tensor([[[[3.0, 2.0], [1.0, 0.0]], [[0.0, 2.0], [0.0, 2.0]]]])
    score = style_sensitivity_case(clean, style)
    assert score.shape == (2,)
    assert float(score[0]) > 0.0
    assert float(score[1]) < 1.0e-12  # affine scaling disappears after centering/L2 normalization


def test_continuous_roles_and_registered_controls_are_exact() -> None:
    content = torch.tensor([0.0, 1.0, 3.0, 2.0])
    style = torch.tensor([0.0, 3.0, 1.0, 2.0])
    alpha, beta, zero = continuous_channel_roles(content, style)
    assert alpha[0] == beta[0] == 0.5 and bool(zero[0])
    assert torch.all((alpha >= 0) & (alpha <= 1))
    assert torch.allclose(alpha + beta, torch.ones_like(alpha), atol=1.0e-7, rtol=0)
    hard_alpha, hard_beta = hard_rank_roles(alpha)
    assert int(hard_alpha.sum()) == 3  # ceil(0.60 * 4)
    assert torch.equal(hard_alpha + hard_beta, torch.ones_like(alpha))
    uniform_alpha, uniform_beta = uniform_half_roles(alpha)
    assert torch.equal(uniform_alpha, torch.full_like(alpha, 0.5))
    assert torch.equal(uniform_alpha, uniform_beta)


def test_split_ranking_and_overlap_are_deterministic() -> None:
    items = [("p3", "c3"), ("p1", "c1"), ("p2", "c2"), ("p4", "c4")]
    first = stable_half_assignment(items)
    second = stable_half_assignment(list(reversed(items)))
    assert first == second and set(first.values()) == {"A", "B"}
    left = torch.tensor([0.1, 0.3, 0.2, 0.9])
    right = torch.tensor([0.2, 0.4, 0.1, 0.8])
    assert spearman_correlation(left, right) > 0.0
    assert jaccard(quartile_indices(left, largest=True), quartile_indices(right, largest=True)) == 1.0


def _features(*, requires_grad: bool) -> dict[str, torch.Tensor]:
    return {
        "dec3": torch.randn((2, 4, 5, 5), requires_grad=requires_grad),
        "dec1": torch.randn((2, 3, 7, 7), requires_grad=requires_grad),
    }


def test_ifc_stops_old_gradient_and_pfc_updates_both_current_branches() -> None:
    weights = {"dec3": torch.tensor([0.8, 0.6, 0.4, 0.2]), "dec1": torch.tensor([0.7, 0.5, 0.3])}
    current, previous = _features(requires_grad=True), _features(requires_grad=True)
    ifc = invariant_feature_consolidation(current, previous, weights)
    ifc.loss.backward()
    assert all(value.grad is not None for value in current.values())
    assert all(value.grad is None for value in previous.values())

    weak, strong = _features(requires_grad=True), _features(requires_grad=True)
    valid = torch.ones((2, 1, 9, 9), dtype=torch.bool)
    valid[:, :, :2, :3] = False
    pfc = plastic_feature_consistency(weak, strong, weights, valid)
    pfc.loss.backward()
    assert all(value.grad is not None for value in weak.values())
    assert all(value.grad is not None for value in strong.values())


def test_dead_or_spatially_constant_channels_have_finite_zero_subgradient() -> None:
    weak = {
        "dec3": torch.ones((2, 4, 5, 5), requires_grad=True),
        "dec1": torch.zeros((2, 3, 7, 7), requires_grad=True),
    }
    strong = {
        "dec3": torch.ones((2, 4, 5, 5), requires_grad=True),
        "dec1": torch.zeros((2, 3, 7, 7), requires_grad=True),
    }
    weights = {"dec3": torch.full((4,), 0.5), "dec1": torch.full((3,), 0.5)}
    valid = torch.ones((2, 1, 9, 9), dtype=torch.bool)
    output = plastic_feature_consistency(weak, strong, weights, valid)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    for feature in (*weak.values(), *strong.values()):
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()


def test_role_state_roundtrip_is_tensor_exact_and_not_a_module_parameter() -> None:
    content = {"dec3": torch.tensor([1.0, 2.0]), "dec1": torch.tensor([3.0, 4.0, 5.0])}
    style = {"dec3": torch.tensor([2.0, 1.0]), "dec1": torch.tensor([5.0, 4.0, 3.0])}
    state = build_channel_role_state(
        site_id="RIM_ONE_r3",
        source_checkpoint_sha256="a" * 64,
        labeled_case_ids=["l1", "l2"],
        unlabeled_case_ids=["u1", "u2"],
        style_probe_sha256="b" * 64,
        content_scores=content,
        style_scores=style,
        feature_shapes={"dec3": (1, 2, 4, 4), "dec1": (1, 3, 8, 8)},
    )
    buffer = io.BytesIO()
    torch.save(state.state_dict(), buffer)
    buffer.seek(0)
    restored = ChannelRoleState.from_state_dict(torch.load(buffer, weights_only=False))
    assert tuple(restored.invariant_weights) == FEATURE_LAYERS
    for layer in FEATURE_LAYERS:
        assert torch.equal(state.invariant_weights[layer], restored.invariant_weights[layer])
        assert torch.equal(state.plastic_weights[layer], restored.plastic_weights[layer])
