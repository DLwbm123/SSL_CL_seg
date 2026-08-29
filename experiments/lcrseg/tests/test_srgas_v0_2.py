from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from lcrseg.methods.srgas_v0_1 import SRGASV01Method
from lcrseg.methods.srgas_v0_2 import SRGASV02Method, resolve_srgas_v02_method_config
from lcrseg.models import UNet2D
from lcrseg.regularization import (
    LaggedSensitivityState,
    SharedNoiseStream,
    linear_noise_warmup,
    relation_to_classifier_loss,
    unit_mean_source_normalize,
)


ROOT = Path(__file__).resolve().parents[1]
FRAGMENTS = {
    "L0": "srgas_v0_2_l0_cosine.yaml",
    "L1": "srgas_v0_2_l1_isotropic_warm.yaml",
    "L2": "srgas_v0_2_l2_lag_totalgas_warm.yaml",
    "L3": "srgas_v0_2_l3_lag_supgas_warm.yaml",
    "L4": "srgas_v0_2_l4_lag_srgas_warm.yaml",
    "D1": "srgas_v0_2_d1_same_srgas_warm.yaml",
    "D2": "srgas_v0_2_d2_lag_srgas_nowarm.yaml",
}


def _config(variant: str) -> dict:
    return json.loads((ROOT / "configs/experiments" / FRAGMENTS[variant]).read_text())["method"]


def test_lagged_buffer_first_step_uses_ones() -> None:
    state = LaggedSensitivityState.empty()
    reference = torch.randn(3, 16, 1, 1)
    state.reset_for_site(site_id="RIM_ONE_r3", reference=reference)
    assert torch.equal(state.current_or_ones(reference), torch.ones_like(reference))


def test_lagged_buffer_uses_previous_successful_step() -> None:
    state = LaggedSensitivityState.empty()
    reference = torch.ones(3, 4, 1, 1)
    previous = torch.arange(reference.numel()).reshape_as(reference).float()
    state.reset_for_site(site_id="RIM_ONE_r3", reference=reference)
    state.commit_after_success(previous)
    assert torch.equal(state.current_or_ones(reference), previous)


def test_lagged_buffer_commits_only_after_success() -> None:
    state = LaggedSensitivityState.empty()
    reference = torch.ones(2, 3)
    state.reset_for_site(site_id="RIM_ONE_r3", reference=reference)
    pending = torch.full_like(reference, 7.0)
    assert not state.valid and state.successful_site_step == 0
    state.commit_after_success(pending)
    assert state.valid and state.successful_site_step == 1 and torch.equal(state.buffer, pending)


def test_lagged_buffer_amp_skip_no_advance() -> None:
    state = LaggedSensitivityState.empty()
    state.reset_for_site(site_id="RIM_ONE_r3", reference=torch.ones(2, 2))
    snapshot = state.state_dict()
    state.load_state_dict(snapshot)
    assert not state.valid and state.successful_site_step == 0


def test_lagged_buffer_site_reset() -> None:
    state = LaggedSensitivityState.empty()
    reference = torch.ones(2, 2)
    state.reset_for_site(site_id="RIM_ONE_r3", reference=reference)
    state.commit_after_success(reference * 3)
    state.reset_for_site(site_id="Drishti_GS", reference=reference)
    assert state.site_id == "Drishti_GS" and not state.valid and state.successful_site_step == 0


def test_lagged_buffer_checkpoint_resume() -> None:
    first = LaggedSensitivityState.empty()
    reference = torch.ones(2, 2)
    first.reset_for_site(site_id="RIM_ONE_r3", reference=reference)
    first.commit_after_success(reference * 4)
    second = LaggedSensitivityState.empty()
    second.load_state_dict(first.state_dict())
    assert second.site_id == first.site_id and second.successful_site_step == 1
    assert torch.equal(second.current_or_ones(reference), first.current_or_ones(reference))


def test_noise_warmup_endpoints() -> None:
    assert linear_noise_warmup(successful_site_step=0, total_site_steps=1000) == 0.0
    assert linear_noise_warmup(successful_site_step=100, total_site_steps=1000) == 0.5
    assert linear_noise_warmup(successful_site_step=200, total_site_steps=1000) == 1.0
    assert linear_noise_warmup(successful_site_step=999, total_site_steps=1000) == 1.0


def test_noise_warmup_uses_successful_steps() -> None:
    state = LaggedSensitivityState.empty()
    reference = torch.ones(1)
    state.reset_for_site(site_id="RIM_ONE_r3", reference=reference)
    assert linear_noise_warmup(successful_site_step=state.successful_site_step, total_site_steps=10) == 0.0
    state.advance_without_commit()
    assert linear_noise_warmup(successful_site_step=state.successful_site_step, total_site_steps=10) == 0.5


def test_noise_warmup_fraction_frozen() -> None:
    with pytest.raises(ValueError):
        linear_noise_warmup(successful_site_step=1, total_site_steps=1000, warmup_fraction=0.10)


def test_first_incremental_step_noise_zero() -> None:
    sigma = math.sqrt(0.1) * linear_noise_warmup(successful_site_step=0, total_site_steps=1000)
    weight, scale, noise = torch.randn(3, 4, 1, 1), torch.rand(3, 4, 1, 1), torch.randn(3, 4, 1, 1)
    assert torch.equal(weight + sigma * scale * noise, weight)


def test_shared_noise_stream_equal_across_variants() -> None:
    first = SharedNoiseStream(protocol_seed=20260828, split_seed=0)
    second = SharedNoiseStream(protocol_seed=20260828, split_seed=0)
    a, ah = first.sample(site_id="RIM_ONE_r3", successful_site_step=17, weight_shape=(3, 16, 1, 1), device="cpu")
    b, bh = second.sample(site_id="RIM_ONE_r3", successful_site_step=17, weight_shape=(3, 16, 1, 1), device="cpu")
    assert ah == bh and torch.equal(a, b)


def test_shared_noise_stream_resume() -> None:
    first = SharedNoiseStream(protocol_seed=20260828, split_seed=1)
    expected, checksum = first.sample(site_id="RIM_ONE_r3", successful_site_step=23, weight_shape=(3, 16, 1, 1), device="cpu")
    second = SharedNoiseStream(protocol_seed=20260828, split_seed=1)
    second.load_state_dict(first.state_dict())
    actual, restored_checksum = second.sample(site_id="RIM_ONE_r3", successful_site_step=23, weight_shape=(3, 16, 1, 1), device="cpu")
    assert checksum == restored_checksum and torch.equal(expected, actual)


def _head_gradient(loss_kind: str) -> torch.Tensor:
    torch.manual_seed(9)
    head = torch.nn.Conv2d(4, 3, 1, bias=False)
    feature = torch.randn(2, 4, 8, 8)
    logits = head(feature)
    supervised = torch.nn.functional.cross_entropy(logits, torch.zeros(2, 8, 8, dtype=torch.long))
    ssl = logits.square().mean()
    loss = supervised + ssl if loss_kind == "total" else supervised
    return torch.autograd.grad(loss, head.weight)[0].detach().square()


def test_l2_pending_total_sensitivity() -> None:
    assert torch.equal(_head_gradient("total"), _head_gradient("total"))


def test_l3_pending_supervised_sensitivity() -> None:
    assert torch.equal(_head_gradient("supervised"), _head_gradient("supervised"))


def test_l4_pending_r2c_sensitivity() -> None:
    torch.manual_seed(3)
    head = torch.nn.Conv2d(4, 3, 1, bias=False)
    feature = torch.randn(2, 4, 8, 8)
    logits = head(feature)
    supervised = _head_gradient("supervised")
    target = torch.randn(2, 3, 4, 4).softmax(1)
    valid = torch.ones(2, 1, 8, 8, dtype=torch.bool)
    r2c = relation_to_classifier_loss(logits, target, valid, historical_anchors_available=True)
    relation = torch.autograd.grad(r2c.loss, head.weight)[0].detach().square()
    combined = 0.5 * unit_mean_source_normalize(supervised) + 0.5 * unit_mean_source_normalize(relation)
    assert combined.shape == head.weight.shape and torch.isfinite(combined).all()


def test_l4_lagged_r2c_not_same_step() -> None:
    state = LaggedSensitivityState.empty()
    current = torch.full((3, 16, 1, 1), 2.0)
    previous = torch.full_like(current, 1.0)
    state.reset_for_site(site_id="RIM_ONE_r3", reference=current)
    state.commit_after_success(previous)
    assert torch.equal(state.current_or_ones(current), previous) and not torch.equal(previous, current)


def test_l4_first_step_reduces_to_clean() -> None:
    assert linear_noise_warmup(successful_site_step=0, total_site_steps=1000) == 0.0


def test_v02_d1_same_warm_contract() -> None:
    resolved = resolve_srgas_v02_method_config(_config("D1"))
    assert resolved["sensitivity_timing"] == "same_step_current_clean" and resolved["noise_warm_start"] is True


def test_v02_d2_lag_nowarm_contract() -> None:
    resolved = resolve_srgas_v02_method_config(_config("D2"))
    assert resolved["sensitivity_timing"] == "lagged_previous_successful_step" and resolved["noise_warm_start"] is False


def test_v02_no_architecture_change() -> None:
    first = SRGASV01Method(UNet2D(3, 3), config=json.loads((ROOT / "configs/experiments/srgas_a5_stablerelgas_v0_1a.yaml").read_text())["method"])
    second = SRGASV02Method(UNet2D(3, 3), config=_config("L4"))
    first_shapes = {name: tuple(value.shape) for name, value in first.model.state_dict().items()}
    second_shapes = {name: tuple(value.shape) for name, value in second.model.state_dict().items()}
    assert first_shapes == second_shapes


def test_v02_original_safety_threshold_unchanged() -> None:
    freeze = json.loads((ROOT / "reports/experiment_status/SRGAS_V0_1A_FREEZE_FOR_V0_2.json").read_text())
    assert freeze["original_worst_trajectory_gate"] == 0.015 and freeze["original_gate_relaxed"] is False


def test_lagged_buffer_is_not_parameter() -> None:
    state = LaggedSensitivityState.empty()
    state.reset_for_site(site_id="RIM_ONE_r3", reference=torch.ones(2, 2))
    assert not isinstance(state.buffer, torch.nn.Parameter) and not state.buffer.requires_grad


def test_v02_l4_one_batch_backward_and_successful_commit(tmp_path: Path) -> None:
    from lcrseg.engine.checkpoint import checkpoint_payload, save_checkpoint
    from lcrseg.engine.trainer import Trainer, TrainerState, build_optimizer, build_scheduler
    from tests.v0_2a_test_utils import batches, routing_config

    labeled, unlabeled = batches()
    parent_config = routing_config(
        protocol_id="srgas_v0_1",
        variant_id="A1",
        srgas_variant="A1",
        relation_conditioning="none",
        noise_seed=0,
    )
    parent = SRGASV01Method(UNet2D(3, 3), config=parent_config)
    parent.begin_site("REFUGE", None, 1)
    parent_optimizer = build_optimizer(parent, lr=1.0e-3, weight_decay=0.0)
    parent_trainer = Trainer(
        parent,
        optimizer=parent_optimizer,
        scheduler=build_scheduler(parent_optimizer, total_steps=1),
        device="cpu",
        amp=False,
    )
    parent_trainer.train_step(labeled, unlabeled, state=TrainerState())
    parent_state = parent.method_state_dict()
    payload = checkpoint_payload(
        method_name=parent.method_name,
        method_version=parent.method_version,
        git_commit="NO_GIT_WORKTREE",
        config_resolved={"method": parent_config},
        site_id="REFUGE",
        site_index=0,
        epoch=0,
        site_step=1,
        global_step=1,
        current_model_state=parent.model.state_dict(),
        optimizer_state=parent_trainer.optimizer.state_dict(),
        scheduler_state=parent_trainer.scheduler.state_dict(),
        scaler_state=parent_trainer.scaler.state_dict(),
        current_anchor_state=parent_state["current_anchor_state"],
        historical_anchor_state=parent_state["historical_anchor_state"],
        bootstrap_state=parent_state["bootstrap_state"],
        method_statistics=parent_state["method_statistics"],
        data_split_hash="split",
        manifest_hash="manifest",
    )
    checkpoint = tmp_path / "parent.pt"
    save_checkpoint(checkpoint, payload)

    method = SRGASV02Method(UNet2D(3, 3), config=_config("L4"))
    method.begin_site("RIM_ONE_r3", checkpoint, 10)
    optimizer = build_optimizer(method, lr=1.0e-3, weight_decay=0.0)
    trainer = Trainer(
        method,
        optimizer=optimizer,
        scheduler=build_scheduler(optimizer, total_steps=10),
        device="cpu",
        amp=False,
    )
    result = trainer.train_step(labeled, unlabeled, state=TrainerState(global_step=1, site_step=0))
    assert result.scalars["noise_warmup_multiplier"] == 0.0
    assert result.scalars["perturbation_l2_ratio"] == 0.0
    assert result.scalars["r2c_valid_count"] > 0
    assert method.lagged_state.valid and method.lagged_state.successful_site_step == 1
    assert all(parameter.grad is None for parameter in method.old_model.parameters())
