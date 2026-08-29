from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from lcrseg.transport import (
    TransportState,
    build_case_prototypes,
    estimate_all_class_transport,
    swap_fundus_foreground_deltas,
    transport_anchors,
)


def _paired_batches() -> tuple[object, object]:
    torch.manual_seed(7)
    old = F.normalize(torch.randn(4, 5, 8, 8), dim=1)
    current = F.normalize(old + 0.15 * torch.randn_like(old), dim=1)
    labels = torch.stack(
        [torch.arange(64).reshape(8, 8).remainder(3) for _ in range(4)]
    )
    return (
        build_case_prototypes(old, labels, num_classes=3, minimum_pixels=16),
        build_case_prototypes(current, labels, num_classes=3, minimum_pixels=16),
    )


def test_tarc_case_prototypes_include_background_and_are_normalized() -> None:
    old, _ = _paired_batches()
    assert old.prototypes.shape == (4, 3, 5)
    assert old.valid.all()
    assert torch.allclose(old.prototypes[old.valid].norm(dim=1), torch.ones(12), atol=1e-6)
    assert bool((old.pixel_counts >= 16).all())


def test_tarc_minimum_pixels_and_two_case_requirement() -> None:
    feature = F.normalize(torch.randn(1, 4, 4, 4), dim=1)
    label = torch.zeros((1, 4, 4), dtype=torch.long)
    batch = build_case_prototypes(feature, label, num_classes=3, minimum_pixels=17)
    assert not batch.valid.any()
    estimate = estimate_all_class_transport(batch, batch)
    assert all(not item.valid and item.shrinkage == 0.0 for item in estimate.class_estimates)


def test_tarc_case_balance_not_pixel_balance() -> None:
    # A tiny eligible case and a much larger case remain two equal units after
    # prototype construction; pixel counts do not weight transport.
    old, current = _paired_batches()
    estimate = estimate_all_class_transport(old, current)
    class_id = 0
    displacement = current.prototypes[:, class_id] - old.prototypes[:, class_id]
    assert torch.allclose(estimate.class_estimates[class_id].mean_displacement, displacement.mean(dim=0))


def test_tarc_shrinkage_bounds_and_noisy_shift_shrinks() -> None:
    old, current = _paired_batches()
    estimate = estimate_all_class_transport(old, current)
    for item in (*estimate.class_estimates, estimate.global_estimate):
        assert 0.0 <= item.shrinkage <= 1.0
        assert torch.allclose(item.delta, item.mean_displacement * item.shrinkage)


def test_tarc_transport_is_detached_normalized_and_immutable() -> None:
    anchors = F.normalize(torch.randn(3, 1, 5), dim=2)
    frozen = anchors.clone()
    delta = torch.randn(3, 5) * 0.01
    view = transport_anchors(anchors, delta)
    assert torch.equal(anchors, frozen)
    assert not view.requires_grad
    assert torch.allclose(view[:, 0].norm(dim=1), torch.ones(3), atol=1e-6)


def test_tarc_shift_swap_only_swaps_foreground() -> None:
    delta = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    swapped = swap_fundus_foreground_deltas(delta)
    assert torch.equal(swapped[0], delta[0])
    assert torch.equal(swapped[1], delta[2])
    assert torch.equal(swapped[2], delta[1])


def test_tarc_transport_state_round_trip_and_fixed_epoch_view() -> None:
    anchors = F.normalize(torch.randn(3, 1, 4), dim=2).detach()
    state = TransportState(
        site_id="RIM_ONE_r3",
        site_index=1,
        epoch=3,
        variant="T3",
        transported_anchors=anchors,
        class_deltas=torch.zeros(3, 4),
        global_delta=torch.zeros(4),
        paired_case_counts=torch.tensor([4, 4, 4]),
    )
    restored = TransportState.from_state_dict(state.state_dict())
    assert restored.epoch == 3
    assert torch.equal(restored.transported_anchors, state.transported_anchors)
    with pytest.raises(ValueError):
        TransportState(
            site_id="x",
            site_index=1,
            epoch=0,
            variant="T3",
            transported_anchors=anchors.requires_grad_(),
            class_deltas=torch.zeros(3, 4),
            global_delta=torch.zeros(4),
            paired_case_counts=torch.ones(3, dtype=torch.long),
        )
