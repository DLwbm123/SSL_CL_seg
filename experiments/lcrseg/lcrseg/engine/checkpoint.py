"""Atomic, contract-shaped checkpoints outside the frozen data bundle."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA_VERSION = 1
_REQUIRED_KEYS = {
    "schema_version",
    "method_name",
    "method_version",
    "git_commit",
    "config_resolved",
    "site_id",
    "site_index",
    "epoch",
    "site_step",
    "global_step",
    "current_model_state",
    "optimizer_state",
    "scheduler_state",
    "scaler_state",
    "current_anchor_state",
    "historical_anchor_state",
    "bootstrap_state",
    "method_statistics",
    "rng_python",
    "rng_numpy",
    "rng_torch_cpu",
    "rng_torch_cuda",
    "data_split_hash",
    "preprocess_version",
    "manifest_hash",
}


def capture_rng_state() -> dict[str, Any]:
    return {
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch_cpu": torch.get_rng_state(),
        "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(payload: dict[str, Any]) -> None:
    random.setstate(payload["rng_python"])
    np.random.set_state(payload["rng_numpy"])
    torch.set_rng_state(payload["rng_torch_cpu"])
    if torch.cuda.is_available() and payload.get("rng_torch_cuda"):
        torch.cuda.set_rng_state_all(payload["rng_torch_cuda"])


def checkpoint_payload(
    *,
    method_name: str,
    method_version: str,
    git_commit: str,
    config_resolved: dict[str, Any],
    site_id: str,
    site_index: int,
    epoch: int,
    site_step: int,
    global_step: int,
    current_model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    scheduler_state: dict[str, Any] | None,
    scaler_state: dict[str, Any] | None,
    current_anchor_state: dict[str, Any] | None,
    historical_anchor_state: dict[str, Any] | None,
    bootstrap_state: dict[str, Any] | None,
    method_statistics: dict[str, Any] | None,
    data_split_hash: str,
    manifest_hash: str,
    preprocess_version: str = "v1",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method_name": method_name,
        "method_version": method_version,
        "git_commit": git_commit,
        "config_resolved": config_resolved,
        "site_id": site_id,
        "site_index": int(site_index),
        "epoch": int(epoch),
        "site_step": int(site_step),
        "global_step": int(global_step),
        "current_model_state": current_model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state or {},
        "scaler_state": scaler_state or {},
        "current_anchor_state": current_anchor_state or {},
        "historical_anchor_state": historical_anchor_state or {},
        "bootstrap_state": bootstrap_state or {},
        "method_statistics": method_statistics or {},
        "data_split_hash": data_split_hash,
        "preprocess_version": preprocess_version,
        "manifest_hash": manifest_hash,
    }
    payload.update(capture_rng_state())
    return payload


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    missing = _REQUIRED_KEYS.difference(payload)
    if missing:
        raise ValueError(f"checkpoint payload misses required keys: {sorted(missing)}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite stale temporary checkpoint: {temporary}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    missing = _REQUIRED_KEYS.difference(payload)
    if missing:
        raise ValueError(f"checkpoint is incomplete: {sorted(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {payload['schema_version']}")
    return payload
