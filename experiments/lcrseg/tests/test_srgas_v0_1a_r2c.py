from __future__ import annotations

import json
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

from lcrseg.engine.checkpoint import checkpoint_payload, save_checkpoint
from lcrseg.engine.trainer import Trainer, TrainerState, build_optimizer, build_scheduler
from lcrseg.methods.srgas_v0_1 import SRGASV01Method
from lcrseg.models import CosineSegmentationHead, UNet2D
from lcrseg.regularization import (
    SpatialRelationShuffler,
    jascl_inverse_minmax_scale,
    relation_to_classifier_loss,
    unit_mean_source_normalize,
)


def _inputs():
    torch.manual_seed(20260828)
    head = CosineSegmentationHead(16, 3)
    feature = torch.randn(2, 16, 32, 32)
    target = torch.randn(2, 3, 8, 8).softmax(1)
    valid = torch.ones(2, 1, 32, 32, dtype=torch.bool)
    return head, feature, target, valid


def test_r2c_class_count_matches() -> None:
    head, feature, target, valid = _inputs()
    assert relation_to_classifier_loss(head(feature), target, valid, historical_anchors_available=True).target_probability.shape[1] == 3


def test_r2c_class_semantics_hash() -> None:
    report = Path(__file__).resolve().parents[1] / "reports/experiment_status/class_semantics.json"
    assert hashlib.sha256(report.read_bytes()).hexdigest() == "5c52655356b11831820433035dad0adfe919219a4da2a9f70d2b18d784010200"


def test_r2c_uses_old_relation_target() -> None:
    head, feature, target, valid = _inputs()
    output = relation_to_classifier_loss(head(feature), target, valid, historical_anchors_available=True)
    assert torch.allclose(output.target_probability, target, atol=1e-6)


def test_r2c_target_stopgrad() -> None:
    head, feature, target, valid = _inputs()
    target.requires_grad_(True)
    output = relation_to_classifier_loss(head(feature), target, valid, historical_anchors_available=True)
    output.loss.backward()
    assert target.grad is None and not output.target_probability.requires_grad


def test_r2c_current_logits_downsample_contract() -> None:
    head, feature, target, valid = _inputs()
    logits = head(feature)
    output = relation_to_classifier_loss(logits, target, valid, historical_anchors_available=True)
    expected = F.interpolate(logits.float(), size=(8, 8), mode="bilinear", align_corners=False).softmax(1)
    assert torch.allclose(output.current_probability, expected)


def test_r2c_no_channel_mapping() -> None:
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/experiments/srgas_a5_stablerelgas_v0_1a.yaml").read_text())
    assert config["method"]["channel_mapping"] == "none" and config["method"]["architecture_change"] is False


def test_r2c_loss_not_in_total_objective() -> None:
    clean = torch.tensor(2.0, requires_grad=True)
    ssl = torch.tensor(3.0, requires_grad=True)
    r2c = torch.tensor(100.0, requires_grad=True)
    total = clean + ssl
    assert torch.autograd.grad(total, r2c, allow_unused=True)[0] is None


def _gradient():
    head, feature, target, valid = _inputs()
    output = relation_to_classifier_loss(head(feature), target, valid, historical_anchors_available=True)
    return torch.autograd.grad(output.loss, head.weight)[0]


def test_r2c_gradient_matches_classifier_weight_shape() -> None:
    assert _gradient().shape == (3, 16, 1, 1)


def test_r2c_gradient_finite_nonzero() -> None:
    gradient = _gradient()
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_r2c_empty_mask_reduces_a5_to_a4() -> None:
    head, feature, target, valid = _inputs()
    output = relation_to_classifier_loss(head(feature), target, torch.zeros_like(valid), historical_anchors_available=True)
    assert output.valid_count == 0 and output.loss.item() == 0.0


def test_r2c_first_site_reduces_a5_to_a4() -> None:
    head, feature, target, valid = _inputs()
    output = relation_to_classifier_loss(head(feature), target, valid, historical_anchors_available=False)
    assert output.valid_count == 0 and output.loss.item() == 0.0


def test_r2c_no_gradient_to_old_model() -> None:
    head, feature, target, valid = _inputs()
    old = target.log().detach().requires_grad_(True)
    relation_to_classifier_loss(head(feature), old.softmax(1), valid, historical_anchors_available=True).loss.backward()
    assert old.grad is None


def test_r2c_no_gradient_to_historical_anchor() -> None:
    test_r2c_no_gradient_to_old_model()


def test_r2c_no_gradient_to_projection_head() -> None:
    head, feature, target, valid = _inputs()
    projection = torch.nn.Conv2d(16, 8, 1)
    _ = projection(feature)
    relation_to_classifier_loss(head(feature), target, valid, historical_anchors_available=True).loss.backward()
    assert all(parameter.grad is None for parameter in projection.parameters())


def test_r2c_proxy_changes_noise_scale() -> None:
    first = jascl_inverse_minmax_scale(_gradient().square())
    head, feature, target, valid = _inputs()
    second_target = torch.roll(target, shifts=1, dims=-1)
    output = relation_to_classifier_loss(head(feature), second_target, valid, historical_anchors_available=True)
    second = jascl_inverse_minmax_scale(torch.autograd.grad(output.loss, head.weight)[0].square())
    assert not torch.allclose(first, second)


def test_r2c_source_normalization_unit_mean() -> None:
    assert torch.allclose(unit_mean_source_normalize(_gradient().square()).mean(), torch.tensor(1.0), atol=1e-6)


def test_r2c_combination_exact_half_half() -> None:
    supervised = unit_mean_source_normalize(torch.rand(3, 16, 1, 1))
    relation = unit_mean_source_normalize(torch.rand(3, 16, 1, 1))
    combined = 0.5 * supervised + 0.5 * relation
    assert torch.equal(combined, supervised.mul(0.5).add(relation.mul(0.5)))


def test_r2c_spatial_shuffle_preserves_marginals() -> None:
    _, _, target, _ = _inputs()
    shuffled = SpatialRelationShuffler(protocol_seed=0).shuffle(target, site_id="RIM_ONE_r3")
    assert torch.equal(target.flatten(2).sort(dim=2).values, shuffled.flatten(2).sort(dim=2).values)


def test_r2c_spatial_shuffle_breaks_alignment() -> None:
    _, _, target, _ = _inputs()
    shuffled = SpatialRelationShuffler(protocol_seed=0).shuffle(target, site_id="RIM_ONE_r3")
    assert not torch.equal(target, shuffled)


def test_r2c_spatial_shuffle_rng_resume() -> None:
    _, _, target, _ = _inputs()
    first = SpatialRelationShuffler(protocol_seed=0)
    expected = first.shuffle(target, site_id="RIM_ONE_r3")
    second = SpatialRelationShuffler(protocol_seed=0)
    second.load_state_dict(first.state_dict())
    assert torch.equal(expected, second.shuffle(target, site_id="RIM_ONE_r3"))


def test_srgas_a5_integrates_with_frozen_r0_runner(tmp_path) -> None:
    from tests.v0_2a_test_utils import batches, routing_config

    labeled, unlabeled = batches()
    common = routing_config(
        protocol_id="srgas_v0_1",
        variant_id="A1",
        srgas_variant="A1",
        relation_conditioning="none",
        noise_seed=0,
    )
    parent = SRGASV01Method(UNet2D(3, 3), config=common)
    parent.begin_site("REFUGE", None, 1)
    optimizer = build_optimizer(parent, lr=1.0e-3, weight_decay=0.0)
    trainer = Trainer(parent, optimizer=optimizer, scheduler=build_scheduler(optimizer, total_steps=1), device="cpu", amp=False)
    trainer.train_step(labeled, unlabeled, state=TrainerState())
    state = parent.method_state_dict()
    payload = checkpoint_payload(
        method_name=parent.method_name,
        method_version=parent.method_version,
        git_commit="NO_GIT_WORKTREE",
        config_resolved={"method": common},
        site_id="REFUGE",
        site_index=0,
        epoch=0,
        site_step=1,
        global_step=1,
        current_model_state=parent.model.state_dict(),
        optimizer_state=trainer.optimizer.state_dict(),
        scheduler_state=trainer.scheduler.state_dict(),
        scaler_state=trainer.scaler.state_dict(),
        current_anchor_state=state["current_anchor_state"],
        historical_anchor_state=state["historical_anchor_state"],
        bootstrap_state=state["bootstrap_state"],
        method_statistics=state["method_statistics"],
        data_split_hash="split",
        manifest_hash="manifest",
    )
    checkpoint = tmp_path / "parent.pt"
    save_checkpoint(checkpoint, payload)
    amended = routing_config(
        protocol_id="srgas_v0_1a",
        variant_id="A5",
        srgas_variant="A5",
        relation_conditioning="relation_to_classifier_proxy",
        noise_seed=0,
    )
    method = SRGASV01Method(UNet2D(3, 3), config=amended)
    method.begin_site("RIM_ONE_r3", checkpoint, 1)
    optimizer = build_optimizer(method, lr=1.0e-3, weight_decay=0.0)
    active = Trainer(method, optimizer=optimizer, scheduler=build_scheduler(optimizer, total_steps=1), device="cpu", amp=False)
    result = active.train_step(labeled, unlabeled, state=TrainerState(global_step=1))
    assert result.scalars["gas_active"] == 1.0
    assert result.scalars["r2c_valid_count"] > 0
    assert result.scalars["r2c_added_to_training_objective"] == 0.0
    assert all(parameter.grad is None for parameter in method.old_model.parameters())
