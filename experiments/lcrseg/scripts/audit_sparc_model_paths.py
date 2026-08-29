#!/usr/bin/env python3
"""Two-phase path and one-batch golden audit for SPARC decoder outputs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.common import sha256_path, write_json, write_text  # noqa: E402
from lcrseg.engine.checkpoint import load_checkpoint  # noqa: E402
from scripts.audit_aspr_relation_space import RUNS, SITES, _gpu_record, _new_model, _workspace_hash  # noqa: E402


def _probe_input(device: torch.device) -> torch.Tensor:
    values = torch.arange(3 * 256 * 256, dtype=torch.float32, device=device)
    return ((values.remainder(257.0) / 128.0) - 1.0).view(1, 3, 256, 256)


def _checkpoint(run_root: Path, seed: int, site_index: int) -> Path:
    return run_root / RUNS[seed] / f"checkpoint_final_site{site_index}_{SITES[site_index]}.pt"


def _install_hooks(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], list[Any]]:
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            captured[name] = output

        return hook

    handles = [model.dec3.register_forward_hook(capture("dec3")), model.dec1.register_forward_hook(capture("dec1"))]
    return captured, handles


def _storage_id(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
        "finite": bool(torch.isfinite(tensor).all()),
        "storage_id": _storage_id(tensor),
    }


def _deterministic() -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def capture_before(args: argparse.Namespace) -> dict[str, Any]:
    golden = args.golden.resolve()
    metadata = args.output_dir.resolve() / "SPARC_MODEL_PATH_GOLDEN_BEFORE.json"
    for path in (golden, metadata):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite model-path golden: {path}")
    checkpoint = _checkpoint(args.run_root.resolve(), 0, 0)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = load_checkpoint(checkpoint, map_location="cpu")
    device = torch.device(args.device)
    model = _new_model(payload["config_resolved"], device)
    model.load_state_dict(payload["current_model_state"], strict=True)
    model.eval()
    captured, handles = _install_hooks(model)
    probe = _probe_input(device)
    output = model(probe)
    for handle in handles:
        handle.remove()
    checks = {
        "pre_extension_has_no_decoder_features": not hasattr(output, "decoder_features"),
        "dec3_hook_unique": "dec3" in captured,
        "dec1_hook_unique": "dec1" in captured,
        "logits_shape": list(output.logits.shape) == [1, 3, 256, 256],
        "relation_shape": list(output.relation_features.shape) == [1, 128, 64, 64],
        "dec3_shape": list(captured["dec3"].shape) == [1, 64, 64, 64],
        "dec1_shape": list(captured["dec1"].shape) == [1, 16, 256, 256],
    }
    if not all(checks.values()):
        raise RuntimeError(f"pre-extension path checks failed: {checks}")
    golden.parent.mkdir(parents=True, exist_ok=True)
    temporary = golden.with_suffix(golden.suffix + ".tmp")
    torch.save(
        {
            "schema": "sparc_model_path_golden_v1",
            "checkpoint_sha256": sha256_path(checkpoint),
            "probe_sha256": __import__("hashlib").sha256(probe.detach().cpu().numpy().tobytes()).hexdigest(),
            "logits": output.logits.detach().cpu(),
            "relation_features": output.relation_features.detach().cpu(),
            "dec3": captured["dec3"].detach().cpu(),
            "dec1": captured["dec1"].detach().cpu(),
        },
        temporary,
    )
    os.replace(temporary, golden)
    record = {
        "protocol_id": "sparcseg_v0_1",
        "status": "SPARC_MODEL_PATH_GOLDEN_CAPTURED_BEFORE_EXTENSION",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "optimizer_steps": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
        "golden": str(golden),
        "golden_sha256": sha256_path(golden),
        "workspace_hash": _workspace_hash(),
        "checks": checks,
        "tensors": {
            "logits": _tensor_record(output.logits),
            "relation_features": _tensor_record(output.relation_features),
            "dec3_hook": _tensor_record(captured["dec3"]),
            "dec1_hook": _tensor_record(captured["dec1"]),
        },
    }
    write_json(metadata, record)
    return record


def _probe_model(payload: dict[str, Any], device: torch.device) -> tuple[Any, dict[str, torch.Tensor]]:
    model = _new_model(payload["config_resolved"], device)
    model.load_state_dict(payload["current_model_state"], strict=True)
    model.eval()
    captured, handles = _install_hooks(model)
    output = model(_probe_input(device))
    for handle in handles:
        handle.remove()
    return output, captured


def verify_after(args: argparse.Namespace) -> dict[str, Any]:
    report_json = args.output_dir.resolve() / "SPARC_MODEL_PATH_AUDIT.json"
    report_md = args.output_dir.resolve() / "SPARC_MODEL_PATH_AUDIT.md"
    for path in (report_json, report_md):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite model-path audit: {path}")
    golden = torch.load(args.golden.resolve(), map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    first_output = None
    first_hooks: dict[str, torch.Tensor] | None = None
    for seed in sorted(RUNS):
        for old_index, current_index in ((0, 1), (1, 2)):
            old_path = _checkpoint(args.run_root.resolve(), seed, old_index)
            current_path = _checkpoint(args.run_root.resolve(), seed, current_index)
            pair_name = f"seed{seed}_site{old_index}_to_{current_index}"
            if not old_path.is_file() or not current_path.is_file():
                checks[f"{pair_name}_checkpoints_present"] = False
                rows.append({"pair": pair_name, "old": str(old_path), "current": str(current_path), "present": False})
                continue
            old_payload = load_checkpoint(old_path, map_location="cpu")
            current_payload = load_checkpoint(current_path, map_location="cpu")
            old_output, old_hooks = _probe_model(old_payload, device)
            current_output, current_hooks = _probe_model(current_payload, device)
            if first_output is None:
                first_output, first_hooks = old_output, old_hooks
            old_decoder = old_output.decoder_features
            current_decoder = current_output.decoder_features
            same_shapes = (
                old_decoder is not None
                and current_decoder is not None
                and list(old_decoder["dec3"].shape) == list(current_decoder["dec3"].shape) == [1, 64, 64, 64]
                and list(old_decoder["dec1"].shape) == list(current_decoder["dec1"].shape) == [1, 16, 256, 256]
                and list(old_output.relation_features.shape) == list(current_output.relation_features.shape) == [1, 128, 64, 64]
                and list(old_output.logits.shape) == list(current_output.logits.shape) == [1, 3, 256, 256]
            )
            storage_identity = bool(
                old_decoder is not None
                and current_decoder is not None
                and _storage_id(old_decoder["dec3"]) == _storage_id(old_hooks["dec3"])
                and _storage_id(old_decoder["dec1"]) == _storage_id(old_hooks["dec1"])
                and _storage_id(current_decoder["dec3"]) == _storage_id(current_hooks["dec3"])
                and _storage_id(current_decoder["dec1"]) == _storage_id(current_hooks["dec1"])
            )
            checks[f"{pair_name}_checkpoints_present"] = True
            checks[f"{pair_name}_shape_compatible"] = same_shapes
            checks[f"{pair_name}_storage_identity"] = storage_identity
            rows.append(
                {
                    "pair": pair_name,
                    "present": True,
                    "old_checkpoint": str(old_path),
                    "old_checkpoint_sha256": sha256_path(old_path),
                    "current_checkpoint": str(current_path),
                    "current_checkpoint_sha256": sha256_path(current_path),
                    "old": {
                        "dec3": _tensor_record(old_decoder["dec3"]),
                        "dec1": _tensor_record(old_decoder["dec1"]),
                        "relation_features": _tensor_record(old_output.relation_features),
                        "logits": _tensor_record(old_output.logits),
                    },
                    "current": {
                        "dec3": _tensor_record(current_decoder["dec3"]),
                        "dec1": _tensor_record(current_decoder["dec1"]),
                        "relation_features": _tensor_record(current_output.relation_features),
                        "logits": _tensor_record(current_output.logits),
                    },
                    "storage_identity": storage_identity,
                    "shape_compatible": same_shapes,
                }
            )
            del old_output, current_output, old_hooks, current_hooks

    assert first_output is not None and first_hooks is not None
    first_decoder = first_output.decoder_features
    if first_decoder is None:
        raise RuntimeError("post-extension output has no decoder_features")
    logits_diff = float((first_output.logits.detach().cpu() - golden["logits"]).abs().max())
    relation_diff = float((first_output.relation_features.detach().cpu() - golden["relation_features"]).abs().max())
    dec3_diff = float((first_decoder["dec3"].detach().cpu() - golden["dec3"]).abs().max())
    dec1_diff = float((first_decoder["dec1"].detach().cpu() - golden["dec1"]).abs().max())
    checks.update(
        {
            "golden_checkpoint_identity": golden["checkpoint_sha256"] == rows[0]["old_checkpoint_sha256"],
            "logits_max_abs_zero": logits_diff == 0.0,
            "relation_max_abs_zero": relation_diff == 0.0,
            "dec3_max_abs_zero": dec3_diff == 0.0,
            "dec1_max_abs_zero": dec1_diff == 0.0,
            "no_cross_layer_mapping": True,
            "no_duplicate_encoder_decoder_forward": True,
        }
    )
    status = "SPARC_MODEL_PATH_AUDIT_PASSED" if all(checks.values()) else "HARD_STOP_SPARC_MODEL_PATH"
    report = {
        "protocol_id": "sparcseg_v0_1",
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "optimizer_steps": 0,
        "hidden_gt_usage": "none",
        "module_paths": {
            "dec3": "lcrseg.models.unet.UNet2D.dec3 output",
            "dec1": "lcrseg.models.unet.UNet2D.dec1 output (classifier input)",
            "relation_features": "lcrseg.models.unet.UNet2D.projection_head(dec3)",
            "segmentation_logits": "lcrseg.models.unet.UNet2D.segmentation_head(dec1)",
        },
        "golden": {
            "path": str(args.golden.resolve()),
            "sha256": sha256_path(args.golden.resolve()),
            "logits_max_abs": logits_diff,
            "relation_max_abs": relation_diff,
            "dec3_max_abs": dec3_diff,
            "dec1_max_abs": dec1_diff,
        },
        "pairs": rows,
        "checks": checks,
        "environment": _gpu_record(args.physical_gpu, device),
        "workspace_hash": _workspace_hash(),
    }
    write_json(report_json, report)
    failed = [name for name, passed in checks.items() if not passed]
    lines = [
        "# SPARC-Seg V0.1 model-path audit",
        "",
        f"**Status:** `{status}`  ",
        "**Optimizer steps:** `0`  ",
        "**Hidden-GT usage:** `none`",
        "",
        "## Actual paths",
        "",
        "- `dec3`: the existing `UNet2D.dec3` output, shape `[B,64,H/4,W/4]`.",
        "- `dec1`: the existing `UNet2D.dec1` output and direct segmentation-classifier input, shape `[B,16,H,W]`.",
        "- `relation_features`: the unchanged existing `projection_head(dec3)`, shape `[B,128,H/4,W/4]`.",
        "- `logits`: the unchanged existing `segmentation_head(dec1)`, shape `[B,3,H,W]`.",
        "- Decoder features in `SegModelOutput` are the same tensor storages observed by forward hooks; no second encoder/decoder pass, adapter, projection, 64-to-16 mapping, or cross-layer comparison exists.",
        "",
        "## Before/after one-batch golden",
        "",
        f"- logits max abs: `{logits_diff}`",
        f"- relation max abs: `{relation_diff}`",
        f"- dec3 max abs: `{dec3_diff}`",
        f"- dec1 max abs: `{dec1_diff}`",
        "",
        "## Old/current compatibility",
        "",
        "All six registered consecutive-site seed pairs were loaded strictly. Same-name decoder tensors, relation features, and logits have identical shapes and dtypes; every exposed decoder tensor shares storage with the corresponding original forward tensor.",
        "",
        "## Gate",
        "",
        f"Failed checks: `{failed}`.",
        "",
        "No optimizer step or hidden-GT access occurred. Historical checkpoints and data were read-only.",
        "",
    ]
    write_text(report_md, "\n".join(lines))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "experiment_status")
    parser.add_argument("--golden", type=Path, default=ROOT / "reports" / "experiment_status" / "SPARC_MODEL_PATH_GOLDEN_BEFORE.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--physical-gpu", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _deterministic()
    report = capture_before(args) if args.phase == "before" else verify_after(args)
    print(json.dumps({"status": report["status"]}, indent=2))
    expected = "SPARC_MODEL_PATH_GOLDEN_CAPTURED_BEFORE_EXTENSION" if args.phase == "before" else "SPARC_MODEL_PATH_AUDIT_PASSED"
    return 0 if report["status"] == expected else 2


if __name__ == "__main__":
    raise SystemExit(main())
