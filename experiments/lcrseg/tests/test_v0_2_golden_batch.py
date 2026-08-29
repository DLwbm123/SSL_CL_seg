from __future__ import annotations

import torch

from lcrseg.analysis.golden_v0_2 import golden_payload_v0_2, write_or_verify_v0_2_golden
from lcrseg.contracts import LabeledBatch, UnlabeledBatch
from lcrseg.engine.checkpoint import checkpoint_payload, save_checkpoint
from lcrseg.engine.trainer import Trainer, TrainerState, build_optimizer, build_scheduler
from lcrseg.methods.lcrseg_v0_1 import LCRSegV01Method
from lcrseg.methods.lcrseg_v0_2 import LCRSegV02Method
from lcrseg.models import UNet2D
from tests.conftest import make_synthetic_root


def _batches() -> tuple[LabeledBatch, UnlabeledBatch]:
    torch.manual_seed(20260820)
    image = torch.randn(2, 3, 32, 32)
    label = torch.zeros((2, 32, 32), dtype=torch.long)
    label[:, 3:16, 3:16] = 1
    label[:, 17:29, 17:29] = 2
    labeled = LabeledBatch(
        image=image,
        label=label,
        valid_mask=torch.ones((2, 1, 32, 32), dtype=torch.bool),
        case_id=["l0", "l1"],
        patient_id=["l0", "l1"],
        site=["A", "A"],
        slice_index=[None, None],
    )
    unlabeled = UnlabeledBatch(
        weak_image=image.clone(),
        strong_image=image * 1.1 + 0.02,
        strong_valid_mask=torch.ones((2, 1, 32, 32), dtype=torch.bool),
        case_id=["u0", "u1"],
        patient_id=["u0", "u1"],
        site=["B", "B"],
        slice_index=[None, None],
        geometry_record=[{}, {}],
    )
    return labeled, unlabeled


def _routing_config() -> dict[str, object]:
    return {
        "anchor_bootstrap_steps": 0,
        "anchor_min_support_pixels": 1,
        "anchor_max_pixels_per_class": 128,
        "background_boundary_exclusion": 0,
        "tau_cls": 0.0,
        "tau_anchor": 0.0,
        "delta_anchor": 0.0,
        "tau_spatial": 0.0,
        "assim_ramp_steps": 1,
        "relation_ramp_steps": 1,
        "min_rank_pixels": 1,
    }


def _trainer(method: LCRSegV01Method | LCRSegV02Method, steps: int) -> Trainer:
    optimizer = build_optimizer(method, lr=1.0e-3, weight_decay=0.0)
    return Trainer(method, optimizer=optimizer, scheduler=build_scheduler(optimizer, total_steps=steps), device="cpu", amp=False)


def _previous_checkpoint(tmp_path, labeled: LabeledBatch, unlabeled: UnlabeledBatch):
    first = LCRSegV01Method(UNet2D(3, 3), config=_routing_config())
    first.begin_site("A", None, 1)
    trainer = _trainer(first, 1)
    trainer.train_step(labeled, unlabeled, state=TrainerState(global_step=0, site_step=0, epoch=0))
    state = first.method_state_dict()
    payload = checkpoint_payload(
        method_name=first.method_name,
        method_version=first.method_version,
        git_commit="NO_GIT_WORKTREE",
        config_resolved={"model": {"in_channels": 3, "num_classes": 3}, "method": _routing_config()},
        site_id="A",
        site_index=0,
        epoch=0,
        site_step=1,
        global_step=1,
        current_model_state=first.model.state_dict(),
        optimizer_state=trainer.optimizer.state_dict(),
        scheduler_state=trainer.scheduler.state_dict(),
        scaler_state=trainer.scaler.state_dict(),
        current_anchor_state=state["current_anchor_state"],
        historical_anchor_state=state["historical_anchor_state"],
        bootstrap_state=state["bootstrap_state"],
        method_statistics=state["method_statistics"],
        data_split_hash="synthetic-split",
        manifest_hash="synthetic-manifest",
    )
    path = tmp_path / "previous.pt"
    save_checkpoint(path, payload)
    return path


def test_v0_2_r3_golden_batch_exposes_all_frozen_routing_outputs(tmp_path) -> None:
    labeled, unlabeled = _batches()
    previous = _previous_checkpoint(tmp_path, labeled, unlabeled)
    method = LCRSegV02Method(UNet2D(3, 3), config=_routing_config())
    method.begin_site("B", previous, total_steps=10)
    assert method.old_model is not None and method.old_anchor_bank is not None
    assert method.current_anchor_bank.anchors.data_ptr() != method.old_anchor_bank.anchors.data_ptr()
    old_anchor = method.old_anchor_bank.anchors.detach().clone()

    trainer = _trainer(method, 10)
    result = trainer.train_step(labeled, unlabeled, state=TrainerState(global_step=1, site_step=0, epoch=0))

    assert result.maps is not None
    expected_maps = {
        "current_relation_probability",
        "raw_learnability",
        "admission_mask",
        "raw_compatibility",
        "calibrated_compatibility",
        "rejection_mask",
        "consolidation_weights",
    }
    assert expected_maps.issubset(result.maps)
    assert {"loss_sup", "loss_assim", "loss_relation"}.issubset(result.losses)
    assert torch.isfinite(result.total_loss)
    assert not result.maps["admission_mask"].requires_grad
    assert not result.maps["rejection_mask"].requires_grad
    assert not result.maps["consolidation_weights"].requires_grad
    assert bool(result.maps["consolidation_weights"].ge(0.5).all())
    assert bool(result.maps["consolidation_weights"].le(1.0).all())
    assert all(parameter.grad is None for parameter in method.old_model.parameters())
    assert torch.equal(method.old_anchor_bank.anchors, old_anchor)
    method.assert_old_state_unchanged()

    statistics = method.method_state_dict()["method_statistics"]
    assert "compatibility_calibrator_state" in statistics
    assert statistics["compatibility_calibrator_state"]["status"] == "unavailable"


def test_v0_2_golden_artifact_is_created_then_independently_verified(tmp_path) -> None:
    root = make_synthetic_root(tmp_path)
    labeled, unlabeled = _batches()
    previous = _previous_checkpoint(tmp_path, labeled, unlabeled)
    method = LCRSegV02Method(UNet2D(3, 3), config=_routing_config())
    method.begin_site("B", previous, total_steps=10)
    trainer = _trainer(method, 10)
    trainer.train_step(labeled, unlabeled, state=TrainerState(global_step=1, site_step=0, epoch=0))
    state = method.method_state_dict()
    current = tmp_path / "r3_checkpoint.pt"
    save_checkpoint(
        current,
        checkpoint_payload(
            method_name=method.method_name,
            method_version=method.method_version,
            git_commit="NO_GIT_WORKTREE",
            config_resolved={"model": {"in_channels": 3, "num_classes": 3}, "method": dict(method.config)},
            site_id="B",
            site_index=1,
            epoch=0,
            site_step=1,
            global_step=2,
            current_model_state=method.model.state_dict(),
            optimizer_state=trainer.optimizer.state_dict(),
            scheduler_state=trainer.scheduler.state_dict(),
            scaler_state=trainer.scaler.state_dict(),
            current_anchor_state=state["current_anchor_state"],
            historical_anchor_state=state["historical_anchor_state"],
            bootstrap_state=state["bootstrap_state"],
            method_statistics=state["method_statistics"],
            data_split_hash="synthetic-split",
            manifest_hash="synthetic-manifest",
        ),
    )
    golden_dir = tmp_path / "golden_v0_2"
    first_losses, first_arrays, first_metadata = golden_payload_v0_2(
        root=root,
        checkpoint=current,
        old_checkpoint=previous,
        dataset="fundus",
        site="SITE",
        seed=0,
        device="cpu",
    )
    created = write_or_verify_v0_2_golden(
        output_dir=golden_dir,
        losses=first_losses,
        arrays=first_arrays,
        metadata=first_metadata,
        verify=False,
        atol=1.0e-6,
    )
    assert not created["verified"]

    verified_losses, verified_arrays, verified_metadata = golden_payload_v0_2(
        root=root,
        checkpoint=current,
        old_checkpoint=previous,
        dataset="fundus",
        site="SITE",
        seed=0,
        device="cpu",
    )
    verified = write_or_verify_v0_2_golden(
        output_dir=golden_dir,
        losses=verified_losses,
        arrays=verified_arrays,
        metadata=verified_metadata,
        verify=True,
        atol=1.0e-6,
    )
    assert verified["verified"]
    assert verified["passed"]
    assert all(value == 0.0 for value in verified["array_max_abs_error"].values())
    assert all(value == 0.0 for value in verified["loss_relative_error"].values())
