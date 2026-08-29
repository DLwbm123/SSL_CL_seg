from __future__ import annotations

import torch

from lcrseg.contracts import LabeledBatch, UnlabeledBatch
from lcrseg.engine.checkpoint import checkpoint_payload, save_checkpoint
from lcrseg.engine.trainer import Trainer, TrainerState, build_optimizer, build_scheduler
from lcrseg.methods.lcrseg_v0_1 import LCRSegV01Method
from lcrseg.models import UNet2D


def batches() -> tuple[LabeledBatch, UnlabeledBatch]:
    torch.manual_seed(20260827)
    image = torch.randn(2, 3, 32, 32)
    label = torch.zeros((2, 32, 32), dtype=torch.long)
    label[:, 2:16, 2:16] = 1
    label[:, 17:30, 17:30] = 2
    labeled = LabeledBatch(
        image=image,
        label=label,
        valid_mask=torch.ones((2, 1, 32, 32), dtype=torch.bool),
        case_id=["l0", "l1"], patient_id=["l0", "l1"], site=["A", "A"], slice_index=[None, None],
    )
    unlabeled = UnlabeledBatch(
        weak_image=image.clone(),
        strong_image=image * 1.05 + 0.01,
        strong_valid_mask=torch.ones((2, 1, 32, 32), dtype=torch.bool),
        case_id=["u0", "u1"], patient_id=["u0", "u1"], site=["B", "B"], slice_index=[None, None],
        geometry_record=[{}, {}],
    )
    return labeled, unlabeled


def routing_config(**updates):
    config = {
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
    config.update(updates)
    return config


def trainer(method, steps: int = 4) -> Trainer:
    optimizer = build_optimizer(method, lr=1.0e-3, weight_decay=0.0)
    return Trainer(method, optimizer=optimizer, scheduler=build_scheduler(optimizer, total_steps=steps), device="cpu", amp=False)


def previous_checkpoint(tmp_path):
    labeled, unlabeled = batches()
    method = LCRSegV01Method(UNet2D(3, 3), config=routing_config())
    method.begin_site("A", None, 1)
    active_trainer = trainer(method, 1)
    active_trainer.train_step(labeled, unlabeled, state=TrainerState(global_step=0, site_step=0, epoch=0))
    state = method.method_state_dict()
    payload = checkpoint_payload(
        method_name=method.method_name,
        method_version=method.method_version,
        git_commit="NO_GIT_WORKTREE",
        config_resolved={"method": routing_config()},
        site_id="A", site_index=0, epoch=0, site_step=1, global_step=1,
        current_model_state=method.model.state_dict(),
        optimizer_state=active_trainer.optimizer.state_dict(),
        scheduler_state=active_trainer.scheduler.state_dict(),
        scaler_state=active_trainer.scaler.state_dict(),
        current_anchor_state=state["current_anchor_state"],
        historical_anchor_state=state["historical_anchor_state"],
        bootstrap_state=state["bootstrap_state"],
        method_statistics=state["method_statistics"],
        data_split_hash="split", manifest_hash="manifest",
    )
    path = tmp_path / "previous.pt"
    save_checkpoint(path, payload)
    return path
