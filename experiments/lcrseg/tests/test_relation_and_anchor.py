from __future__ import annotations

import torch
from torch.nn import functional as F

from lcrseg.methods.components.anchor_bank import AnchorBank, background_boundary_mask
from lcrseg.methods.components.relation_field import relation_field


def _bank(*, classes: int = 3, dim: int = 4) -> AnchorBank:
    bank = AnchorBank(classes, dim, min_support_pixels=1, max_pixels_per_class=64, background_boundary_exclusion=0)
    with torch.no_grad():
        bank.anchors.copy_(F.normalize(torch.arange(1, classes * dim + 1, dtype=torch.float32).reshape(classes, 1, dim), dim=-1))
        bank.valid.fill_(True)
    return bank


def test_relation_probability_normalization_invalid_classes_and_temperature() -> None:
    bank = _bank()
    features = torch.randn(2, 4, 3, 5)
    output = relation_field(features, bank, temperature=0.1)
    assert torch.allclose(output.probabilities.sum(dim=1), torch.ones_like(output.top1[:, 0]), atol=1e-5)
    assert torch.isfinite(output.probabilities).all()
    bank.valid[2, 0] = False
    masked = relation_field(features, bank, temperature=0.1)
    assert torch.count_nonzero(masked.probabilities[:, 2]) == 0
    with torch.no_grad():
        bank.valid.fill_(False)
    try:
        relation_field(features, bank, temperature=0.1)
    except RuntimeError as exc:
        assert "no valid" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("all-invalid bank must fail explicitly")
    try:
        relation_field(features, _bank(), temperature=0.0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("non-positive temperature must fail")


def test_relation_scale_invariance() -> None:
    bank = _bank()
    features = torch.randn(1, 4, 2, 3)
    baseline = relation_field(features, bank, temperature=0.2).probabilities
    with torch.no_grad():
        bank.anchors.mul_(7.0)
    scaled = relation_field(features * 11.0, bank, temperature=0.2).probabilities
    assert torch.allclose(baseline, scaled, atol=1e-5)


def test_anchor_is_nonparametric_and_current_old_storage_is_independent() -> None:
    current = _bank()
    old = current.clone()
    current.assert_no_parameters()
    assert not list(current.named_parameters())
    assert current.anchors.data_ptr() != old.anchors.data_ptr()
    with torch.no_grad():
        current.anchors[0, 0, 0] += 0.25
    assert not torch.equal(current.anchors, old.anchors)


def test_anchor_empty_class_update_math_and_support_counts() -> None:
    bank = AnchorBank(3, 2, min_support_pixels=1, max_pixels_per_class=8, momentum=0.5, background_boundary_exclusion=0)
    features = torch.tensor([[[[1.0, 0.0]], [[0.0, 2.0]]]])
    labels = torch.tensor([[[1, 1]]])
    weights = torch.ones((1, 1, 1, 2))
    first = bank.update(features, labels, weights, source="labeled", step=4)
    assert first.updated_pixels == {1: 2}
    assert set(first.skipped_classes) == {0, 2}
    expected = F.normalize(torch.tensor([[0.5, 0.5]]), dim=1)[0]
    assert torch.allclose(bank.anchors[1, 0], expected, atol=1e-6)
    assert bank.counts_labeled[1, 0].item() == 2
    old_empty = bank.anchors[2].clone()
    bank.update(features, labels, weights, source="unlabeled", step=5)
    assert torch.equal(bank.anchors[2], old_empty)
    assert bank.counts_unlabeled[2, 0].item() == 0


def test_background_boundary_exclusion() -> None:
    labels = torch.zeros((1, 7, 7), dtype=torch.long)
    labels[:, 3, 3] = 1
    safe = background_boundary_mask(labels, width=1)
    assert not bool(safe[0, 3, 3])
    assert not bool(safe[0, 2:5, 2:5].any())
    assert bool(safe[0, 0, 0])
