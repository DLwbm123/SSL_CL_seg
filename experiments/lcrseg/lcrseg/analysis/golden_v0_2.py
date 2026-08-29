"""Fixed R3 golden-batch artifacts for LCR-Seg V0.2.

This module is deliberately separate from the V0.1 golden implementation so
the old regression gate remains byte-for-byte on its existing execution path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..common import sha256_path, write_json
from ..data import H5LabeledDataset, H5UnlabeledDataset, LabeledTransform, WeakStrongTransform, collate_labeled, collate_unlabeled
from ..engine.checkpoint import load_checkpoint
from ..methods.base import model_checksum
from ..methods.lcrseg_v0_2 import LCRSegV02Method
from ..models import UNet2D
from .golden import write_or_verify_golden


_REQUIRED_MAPS = {
    "current_relation_probability",
    "raw_learnability",
    "admission_mask",
    "raw_compatibility",
    "calibrated_compatibility",
    "rejection_mask",
    "consolidation_weights",
}


def _model_from_config(config: dict[str, Any], device: torch.device) -> UNet2D:
    model_config = dict(config["model"])
    return UNet2D(
        int(model_config["in_channels"]),
        int(model_config["num_classes"]),
        base_channels=int(model_config.get("base_channels", 16)),
        relation_dim=int(model_config.get("relation_dim", 128)),
    ).to(device)


def _method(checkpoint: Path, device: torch.device) -> tuple[LCRSegV02Method, dict[str, Any]]:
    payload = load_checkpoint(checkpoint, map_location="cpu")
    if payload["method_name"] != "lcrseg_v0_2" or str(payload["method_version"]) != "0.2":
        raise ValueError("V0.2 golden regression requires an LCR-Seg V0.2 checkpoint")
    config = dict(payload["config_resolved"])
    method = LCRSegV02Method(_model_from_config(config, device), config=dict(config.get("method", {}))).to(device)
    method.model.load_state_dict(payload["current_model_state"], strict=True)
    method.load_method_state_dict(payload)
    method.site_id = str(payload["site_id"])
    method.site_index = int(payload["site_index"])
    method.total_steps = int((payload.get("method_statistics") or {}).get("active_site_total_steps") or max(1, int(payload["site_step"])))
    if not method.current_anchor_bank.all_classes_valid:
        raise RuntimeError("V0.2 golden checkpoint has incomplete current anchors")
    if method.old_model is None or method.old_anchor_bank is None:
        raise RuntimeError("V0.2 R3 golden requires a final incremental-site checkpoint with historical state")
    method.model.eval()
    method.old_model.eval()
    return method, payload


def _validate_old_checkpoint(current_payload: dict[str, Any], old_checkpoint: Path, device: torch.device) -> dict[str, str]:
    old_payload = load_checkpoint(old_checkpoint, map_location="cpu")
    current_config = dict(current_payload["config_resolved"])
    old_model = _model_from_config(current_config, device)
    old_model.load_state_dict(old_payload["current_model_state"], strict=True)
    expected_checksum = str((current_payload.get("method_statistics") or {}).get("old_model_checksum") or "")
    actual_checksum = model_checksum(old_model)
    if not expected_checksum or actual_checksum != expected_checksum:
        raise AssertionError("explicit old checkpoint does not match the frozen V0.2 historical model")
    historical = dict(current_payload.get("historical_anchor_state") or {})
    old_current = dict(old_payload.get("current_anchor_state") or {})
    for key in ("anchors", "valid", "support"):
        if key in historical and key in old_current and not torch.equal(historical[key].cpu(), old_current[key].cpu()):
            raise AssertionError(f"explicit old checkpoint does not match frozen historical anchors ({key})")
    return {
        "old_checkpoint": str(Path(old_checkpoint).resolve()),
        "old_checkpoint_sha256": sha256_path(old_checkpoint),
        "old_model_checksum": actual_checksum,
    }


def golden_payload_v0_2(
    *,
    root: Path,
    checkpoint: Path,
    old_checkpoint: Path,
    dataset: str,
    site: str,
    seed: int,
    device: torch.device | str,
    labeled_batch_size: int = 2,
    unlabeled_batch_size: int = 2,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Evaluate one deterministic, augmentation-free R3 batch without updating state."""

    if dataset != "fundus":
        raise ValueError("the preregistered V0.2 golden batch is defined for Fundus")
    device = torch.device(device)
    method, payload = _method(checkpoint, device)
    old_identity = _validate_old_checkpoint(payload, old_checkpoint, device)
    labeled = H5LabeledDataset(root, seed=seed, dataset=dataset, sites=(site,), transform=LabeledTransform(flip_probability=0.0))
    unlabeled = H5UnlabeledDataset(
        root,
        seed=seed,
        dataset=dataset,
        sites=(site,),
        transform=WeakStrongTransform(
            flip_probability=0.0,
            strong_noise_std=0.0,
            brightness_delta=0.0,
            contrast_delta=0.0,
            cutout_probability=0.0,
        ),
    )
    if not len(labeled) or not len(unlabeled):
        raise RuntimeError("golden batch site has no current-site labeled or unlabeled records")
    labeled_batch = collate_labeled([labeled[index] for index in range(min(labeled_batch_size, len(labeled)))]).to(device)
    unlabeled_batch = collate_unlabeled([unlabeled[index] for index in range(min(unlabeled_batch_size, len(unlabeled)))]).to(device)
    result = method.training_step(
        labeled_batch,
        unlabeled_batch,
        global_step=int(payload["global_step"]),
        site_step=max(0, int(payload["site_step"]) - 1),
    )
    if not result.maps or not _REQUIRED_MAPS.issubset(result.maps):
        missing = sorted(_REQUIRED_MAPS.difference(result.maps or {}))
        raise RuntimeError(f"V0.2 golden batch did not produce required maps: {missing}")
    arrays = {
        "model_logits": method.model(labeled_batch.image).logits.detach().float().cpu().numpy(),
        "relation_probabilities": result.maps["current_relation_probability"].detach().float().cpu().numpy(),
        "raw_learnability": result.maps["raw_learnability"].detach().float().cpu().numpy(),
        "admission_mask": result.maps["admission_mask"].detach().to(torch.uint8).cpu().numpy(),
        "raw_compatibility": result.maps["raw_compatibility"].detach().float().cpu().numpy(),
        "calibrated_compatibility": result.maps["calibrated_compatibility"].detach().float().cpu().numpy(),
        "rejection_mask": result.maps["rejection_mask"].detach().to(torch.uint8).cpu().numpy(),
        "consolidation_weights": result.maps["consolidation_weights"].detach().float().cpu().numpy(),
    }
    losses = {
        "loss_sup": float(result.losses["loss_sup"].detach().cpu()),
        "loss_assim": float(result.losses["loss_assim"].detach().cpu()),
        "loss_relation": float(result.losses["loss_relation"].detach().cpu()),
        "total_loss": float(result.total_loss.detach().cpu()),
    }
    calibrator_state = method.compatibility_calibrator.state_dict()
    metadata = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_path(checkpoint),
        **old_identity,
        "dataset": dataset,
        "site": site,
        "seed": int(seed),
        "site_step": max(0, int(payload["site_step"]) - 1),
        "total_site_steps": int(method.total_steps),
        "site_progress": float(result.scalars["site_progress"]),
        "labeled_case_ids": list(labeled_batch.case_id),
        "unlabeled_case_ids": list(unlabeled_batch.case_id),
        "augmentation": {
            "labeled_flip_probability": 0.0,
            "weak_strong_flip_probability": 0.0,
            "strong_noise_std": 0.0,
            "brightness_delta": 0.0,
            "contrast_delta": 0.0,
            "cutout_probability": 0.0,
        },
        "calibrator_state": calibrator_state,
        "calibrator_status": str(result.scalars["calibrator_status"]),
        "selected_counts_by_class": list(method.v02_statistics.get("selected_counts_by_class", [])),
        "rejected_counts_by_class": list(method.v02_statistics.get("rejected_counts_by_class", [])),
        "valid_counts": {
            "pseudo_valid": int(result.maps["pseudo_valid"].sum()),
            "admission_selected": int(result.maps["admission_mask"].sum()),
            "relation_valid": int(result.maps["relation_valid_mask"].sum()),
            "compatibility_rejected": int(result.maps["rejection_mask"].sum()),
            "current_valid_anchors": int(method.current_anchor_bank.valid.sum()),
            "historical_valid_anchors": int(method.old_anchor_bank.valid.sum()),
        },
    }
    return losses, arrays, {"metadata": metadata, "anchor_state": method.current_anchor_bank.exported_state()}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_or_verify_v0_2_golden(
    *,
    output_dir: Path,
    losses: dict[str, Any],
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    verify: bool,
    atol: float,
) -> dict[str, Any]:
    """Freeze or independently verify arrays, losses, provenance, and calibration state."""

    output_dir = Path(output_dir)
    if verify:
        recorded = json.loads((output_dir / "metadata.json").read_text())
        expected = dict(metadata["metadata"])
        keys = (
            "checkpoint_sha256",
            "old_checkpoint_sha256",
            "old_model_checksum",
            "dataset",
            "site",
            "seed",
            "site_step",
            "total_site_steps",
            "site_progress",
            "augmentation",
            "calibrator_state",
        )
        mismatched = [key for key in keys if _canonical(recorded.get(key)) != _canonical(expected.get(key))]
        if mismatched:
            raise AssertionError(f"V0.2 golden provenance differs for: {mismatched}")
        return write_or_verify_golden(
            output_dir=output_dir,
            losses=losses,
            arrays=arrays,
            metadata=metadata,
            verify=True,
            atol=atol,
        )
    result = write_or_verify_golden(
        output_dir=output_dir,
        losses=losses,
        arrays=arrays,
        metadata=metadata,
        verify=False,
        atol=atol,
    )
    write_json(output_dir / "calibrator_state.json", metadata["metadata"]["calibrator_state"])
    return result
