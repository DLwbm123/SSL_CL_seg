from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from lcrseg.losses.stable_feature_maintaining import evidence_coefficient, stable_feature_maintaining
from lcrseg.semantics.anchored_validation import anchored_validation, partition_stable_plastic
from lcrseg.semantics.session_prototypes import SessionPrototypeSet, build_session_prototypes
from scripts.audit_sparc_feasibility import _required_boundary_sizes


@dataclass
class _Batch:
    image: torch.Tensor
    label: torch.Tensor
    valid_mask: torch.Tensor
    case_id: list[str]

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "_Batch":
        del non_blocking
        return _Batch(self.image.to(device), self.label.to(device), self.valid_mask.to(device), self.case_id)


class _IdentityRelationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, image: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(relation_features=image * self.scale, logits=torch.zeros(image.shape[0], 3, *image.shape[-2:]))


def _prototype_set(prototypes: torch.Tensor, valid: torch.Tensor, role: str = "current") -> SessionPrototypeSet:
    return SessionPrototypeSet(
        model_role=role,
        site_id="RIM_ONE_r3",
        epoch_id=1,
        prototypes=prototypes.detach(),
        valid_classes=valid.detach(),
        case_counts=valid.long(),
        pixel_counts=valid.long() * 32,
        class_semantics_sha256="semantics",
        source_checkpoint_sha256="checkpoint",
        source_case_ids_sha256="cases",
    )


def test_sparc_prototype_case_normalize_then_equal_average_and_minimum_pixels() -> None:
    model = _IdentityRelationModel()
    model.train()
    image = torch.zeros(2, 2, 8, 8)
    label = torch.zeros(2, 8, 8, dtype=torch.long)
    label[0, 4:] = 1
    image[0, 0] = 1.0
    image[1, 1] = 1.0
    valid = torch.ones(2, 1, 8, 8)
    batch = _Batch(image, label, valid, ["case-a", "case-b"])
    result = build_session_prototypes(
        model,
        [batch],
        model_role="current",
        site_id="RIM_ONE_r3",
        epoch_id=3,
        num_classes=3,
        class_semantics_sha256="semantics",
        source_checkpoint_sha256="checkpoint",
    )
    expected = torch.tensor([2.0**-0.5, 2.0**-0.5])
    assert torch.allclose(result.prototypes[0], expected, atol=1.0e-6, rtol=0.0)
    assert torch.equal(result.case_counts, torch.tensor([2, 1, 0]))
    assert torch.equal(result.pixel_counts, torch.tensor([96, 32, 0]))
    assert torch.equal(result.valid_classes, torch.tensor([True, True, False]))
    assert torch.equal(result.prototypes[2], torch.zeros(2))
    assert not result.prototypes.requires_grad
    assert model.training
    restored = SessionPrototypeSet.from_state_dict(result.state_dict())
    assert torch.equal(restored.prototypes, result.prototypes)
    assert restored.source_case_ids_sha256 == result.source_case_ids_sha256


def test_sparc_anchored_validation_missing_prototype_and_partition_contract() -> None:
    prototypes = torch.eye(3)
    current_prototypes = prototypes.clone()
    current_prototypes[2].zero_()
    current_set = _prototype_set(current_prototypes, torch.tensor([True, True, False]), "current")
    previous_set = _prototype_set(prototypes, torch.tensor([True, True, True]), "previous")
    logits = torch.tensor(
        [[[[8.0, 0.0], [0.0, 0.0]], [[0.0, 8.0], [0.0, 0.0]], [[0.0, 0.0], [8.0, 8.0]]]],
        requires_grad=True,
    )
    features = torch.zeros(1, 3, 2, 2, requires_grad=True)
    with torch.no_grad():
        features[0, 0, 0, 0] = 1
        features[0, 1, 0, 1] = 1
        features[0, 2, 1] = 1
    relation_valid = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    current = anchored_validation(logits, features, current_set, relation_valid)
    previous = anchored_validation(logits.detach(), features.detach(), previous_set, relation_valid)
    partition = partition_stable_plastic(current, previous, relation_valid)
    assert torch.equal(current.valid, torch.tensor([[[True, True], [False, False]]]))
    assert torch.equal(partition.stable, current.valid)
    assert not bool(partition.plastic.any())
    assert torch.equal(partition.rejected, ~current.valid)
    assert not current.valid.requires_grad
    assert not current.predicted_class.requires_grad


def test_sparc_feature_loss_is_same_layer_targeted_detached_and_localized() -> None:
    current = {
        "dec3": torch.randn(1, 4, 2, 2, requires_grad=True),
        "dec1": torch.randn(1, 2, 4, 4, requires_grad=True),
    }
    previous = {
        "dec3": torch.randn(1, 4, 2, 2, requires_grad=True),
        "dec1": torch.randn(1, 2, 4, 4, requires_grad=True),
    }
    stable = torch.tensor([[[True, False], [False, False]]])
    classes = torch.tensor([[[1, 1], [2, 0]]])
    current_valid = torch.tensor([[[True, True], [True, False]]])
    output = stable_feature_maintaining(current, previous, stable, classes, current_valid)
    assert torch.allclose(output.kappa, torch.tensor(0.25))
    assert not output.kappa.requires_grad
    output.loss.backward()
    assert previous["dec3"].grad is None and previous["dec1"].grad is None
    dec3_grad = current["dec3"].grad.abs().sum(dim=1)[0]
    assert dec3_grad[0, 0] > 0
    assert torch.equal(dec3_grad[~stable[0]], torch.zeros_like(dec3_grad[~stable[0]]))
    dec1_grad = current["dec1"].grad.abs().sum(dim=1)[0]
    assert dec1_grad[:2, :2].sum() > 0
    outside = dec1_grad.clone()
    outside[:2, :2] = 0
    assert torch.equal(outside, torch.zeros_like(outside))


def test_sparc_feature_loss_empty_foreground_is_differentiable_zero() -> None:
    current = {"dec3": torch.randn(1, 4, 2, 2, requires_grad=True), "dec1": torch.randn(1, 2, 4, 4, requires_grad=True)}
    previous = {name: value.detach().clone() for name, value in current.items()}
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    background = torch.zeros(1, 2, 2, dtype=torch.long)
    output = stable_feature_maintaining(current, previous, mask, background, mask)
    assert output.loss.requires_grad
    assert float(output.loss) == 0.0
    assert float(output.kappa) == 0.0
    output.loss.backward()
    assert all(torch.equal(value.grad, torch.zeros_like(value)) for value in current.values())


def test_sparc_evidence_coefficient_is_class_mean_not_pixel_mean() -> None:
    classes = torch.tensor([[[1, 1, 2, 2], [2, 2, 2, 2]]])
    valid = torch.ones_like(classes, dtype=torch.bool)
    stable = torch.zeros_like(valid)
    stable[0, 0, 0] = True  # class 1: 1/2
    stable[0, 0, 2] = True
    stable[0, 1, 0] = True
    stable[0, 1, 1] = True  # class 2: 3/6
    kappa, present = evidence_coefficient(stable, valid, classes)
    assert present == (1, 2)
    assert torch.allclose(kappa, torch.tensor(0.5))


def test_sparc_posthoc_boundary_cache_uses_actual_feature_sizes() -> None:
    relation = torch.empty(1, 128, 96, 96)
    decoder = {
        "dec3": torch.empty(1, 64, 96, 96),
        "dec1": torch.empty(1, 16, 384, 384),
    }
    assert _required_boundary_sizes(relation, decoder) == ((96, 96), (384, 384))
