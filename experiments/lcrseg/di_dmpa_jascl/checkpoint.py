from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .modeling import assert_complete_classifier_load, classifier_gas_state, restore_gas_state


CHECKPOINT_SCHEMA_VERSION = 2


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_checkpoint(
    *,
    wrapper,
    optimizer: torch.optim.Optimizer,
    scheduler,
    stage_state: dict[str, Any],
    sampler_state: dict[str, Any],
    prototypes: torch.Tensor | None,
    config_hash: str,
    evaluation_matrices: dict[str, Any],
    best_metric: float,
    git_commit: str = "unit_test",
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "student": wrapper.student.state_dict(),
        "ema_teacher": wrapper.teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "stage_state": dict(stage_state),
        "sampler_state": dict(sampler_state),
        "gas_state": classifier_gas_state(wrapper.student),
        "rng_state": capture_rng_state(),
        "prototypes": None if prototypes is None else prototypes.detach().cpu().clone(),
        "config_hash": config_hash,
        "git_commit": git_commit,
        "evaluation_matrices": evaluation_matrices,
        "best_metric": float(best_metric),
    }


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def load_checkpoint(
    path: str | Path,
    *,
    wrapper,
    optimizer: torch.optim.Optimizer,
    scheduler,
    expected_config_hash: str,
    expected_git_commit: str | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("unsupported checkpoint schema")
    if payload.get("config_hash") != expected_config_hash:
        raise RuntimeError("checkpoint/config hash mismatch")
    if expected_git_commit is not None and payload.get("git_commit") != expected_git_commit:
        raise RuntimeError("checkpoint/source commit mismatch")
    assert_complete_classifier_load(payload["student"], wrapper.student)
    assert_complete_classifier_load(payload["ema_teacher"], wrapper.teacher)
    wrapper.student.load_state_dict(payload["student"], strict=True)
    wrapper.teacher.load_state_dict(payload["ema_teacher"], strict=True)
    wrapper.freeze_teacher()
    wrapper.teacher.eval()
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_gas_state(wrapper.student, payload["gas_state"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload
