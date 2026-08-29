"""Fixed, training-manifest-only golden-batch regression artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..common import write_json, write_text
from ..data import H5LabeledDataset, H5UnlabeledDataset, LabeledTransform, WeakStrongTransform, collate_labeled, collate_unlabeled
from ..engine.checkpoint import load_checkpoint
from ..methods.lcrseg_v0_1 import LCRSegV01Method
from ..models import UNet2D


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _method(checkpoint: Path, device: torch.device) -> tuple[LCRSegV01Method, dict[str, Any]]:
    payload = load_checkpoint(checkpoint, map_location="cpu")
    if payload["method_name"] != "lcrseg_v0_1":
        raise ValueError("golden regression is currently defined for lcrseg_v0_1")
    config = payload["config_resolved"]
    model_config = config["model"]
    model = UNet2D(
        int(model_config["in_channels"]),
        int(model_config["num_classes"]),
        base_channels=int(model_config.get("base_channels", 16)),
        relation_dim=int(model_config.get("relation_dim", 128)),
    ).to(device)
    method = LCRSegV01Method(model, config=config.get("method", {})).to(device)
    method.model.load_state_dict(payload["current_model_state"], strict=True)
    method.load_method_state_dict(payload)
    method.site_id = str(payload["site_id"])
    method.site_index = int(payload["site_index"])
    method.total_steps = int((payload.get("method_statistics") or {}).get("active_site_total_steps") or max(1, payload["site_step"]))
    method.model.eval()
    if method.old_model is not None:
        method.old_model.eval()
    return method, payload


@torch.no_grad()
def golden_payload(
    *,
    root: Path,
    checkpoint: Path,
    dataset: str,
    site: str,
    seed: int,
    device: torch.device | str,
    labeled_batch_size: int = 2,
    unlabeled_batch_size: int = 2,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    device = torch.device(device)
    method, checkpoint_payload = _method(checkpoint, device)
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
    labeled_batch = collate_labeled([labeled[index] for index in range(min(labeled_batch_size, len(labeled)))]).to(device)
    unlabeled_batch = collate_unlabeled([unlabeled[index] for index in range(min(unlabeled_batch_size, len(unlabeled)))]).to(device)
    result = method.training_step(
        labeled_batch,
        unlabeled_batch,
        global_step=int(checkpoint_payload["global_step"]),
        site_step=max(0, int(checkpoint_payload["site_step"]) - 1),
    )
    if not result.maps or "current_relation_probability" not in result.maps or "learnability" not in result.maps or "compatibility" not in result.maps:
        raise RuntimeError("golden batch did not produce complete LCR maps")
    arrays = {
        "model_logits": method.model(labeled_batch.image).logits.detach().float().cpu().numpy(),
        "relation_probabilities": result.maps["current_relation_probability"].detach().float().cpu().numpy(),
        "learnability": result.maps["learnability"].detach().float().cpu().numpy(),
        "compatibility": result.maps["compatibility"].detach().float().cpu().numpy(),
    }
    losses = {name: float(value.detach().cpu()) for name, value in result.losses.items()}
    metadata = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "dataset": dataset,
        "site": site,
        "seed": seed,
        "labeled_case_ids": labeled_batch.case_id,
        "unlabeled_case_ids": unlabeled_batch.case_id,
        "valid_counts": {
            "pseudo_valid": int(result.maps["pseudo_valid"].sum()),
            "current_valid_anchors": int(method.current_anchor_bank.valid.sum()),
            "historical_valid_anchors": int(method.old_anchor_bank.valid.sum()) if method.old_anchor_bank is not None else 0,
        },
    }
    return losses, arrays, {"metadata": metadata, "anchor_state": method.current_anchor_bank.exported_state()}


def write_or_verify_golden(
    *,
    output_dir: Path,
    losses: dict[str, Any],
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    verify: bool,
    atol: float,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    paths = {name: output_dir / f"{name}.npy" for name in arrays}
    if verify:
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"golden files are missing: {missing}")
        array_errors = {name: float(np.max(np.abs(np.load(paths[name]) - value))) for name, value in arrays.items()}
        expected_losses = json.loads((output_dir / "losses.json").read_text())
        loss_errors = {
            name: abs(float(expected_losses[name]) - float(value)) / max(1.0, abs(float(expected_losses[name])))
            for name, value in losses.items()
        }
        result = {
            "verified": True,
            "array_max_abs_error": array_errors,
            "loss_relative_error": loss_errors,
            "passed": all(error <= atol for error in array_errors.values()) and all(error <= atol for error in loss_errors.values()),
        }
        if not result["passed"]:
            raise AssertionError(f"golden regression failed: {result}")
        return result
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen golden artifacts: {output_dir}")
    output_dir.mkdir(parents=True)
    for name, value in arrays.items():
        np.save(paths[name], value)
        write_text(output_dir / f"{name}.sha256", _array_hash(value) + "\n")
    torch.save(metadata["anchor_state"], output_dir / "anchor_state.pt")
    write_json(output_dir / "losses.json", losses)
    write_json(output_dir / "valid_counts.json", metadata["metadata"]["valid_counts"])
    write_json(output_dir / "metadata.json", metadata["metadata"])
    return {"verified": False, "output_dir": str(output_dir), "array_hashes": {name: _array_hash(value) for name, value in arrays.items()}}
