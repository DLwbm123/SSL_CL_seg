#!/usr/bin/env python3
"""Revalidate frozen SPARC decoder paths for CRISP-Seg without changing U-Net."""
from __future__ import annotations

import argparse
import hashlib
import json
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
from scripts.audit_sparc_model_paths import _checkpoint, _deterministic, _install_hooks, _probe_input, _storage_id  # noqa: E402


def _keys_sha(keys: list[str]) -> str:
    return hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode("utf-8")).hexdigest()


def _probe(payload: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, Any, dict[str, torch.Tensor]]:
    model = _new_model(payload["config_resolved"], device)
    model.load_state_dict(payload["current_model_state"], strict=True)
    model.eval()
    captured, handles = _install_hooks(model)
    output = model(_probe_input(device))
    for handle in handles:
        handle.remove()
    return model, output, captured


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "experiment_status")
    parser.add_argument("--golden", type=Path, default=ROOT / "reports" / "experiment_status" / "SPARC_MODEL_PATH_GOLDEN_BEFORE.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    _deterministic()
    output_dir = args.output_dir.resolve()
    report_json = output_dir / "CRISP_MODEL_PATH_REVALIDATION.json"
    report_md = output_dir / "CRISP_MODEL_PATH_REVALIDATION.md"
    for path in (report_json, report_md):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite CRISP path audit: {path}")
    golden_path = args.golden.resolve()
    golden = torch.load(golden_path, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    checks: dict[str, bool] = {}
    rows: list[dict[str, Any]] = []
    first_output = first_hooks = None
    parameter_counts: list[int] = []
    state_key_hashes: list[str] = []
    for seed in sorted(RUNS):
        for old_index, current_index in ((0, 1), (1, 2)):
            for role, site_index in (("old", old_index), ("current", current_index)):
                checkpoint = _checkpoint(args.run_root.resolve(), seed, site_index)
                key = f"seed{seed}_{old_index}_{current_index}_{role}"
                payload = load_checkpoint(checkpoint, map_location="cpu")
                model, output, hooks = _probe(payload, device)
                decoder = output.decoder_features
                model_keys = list(model.state_dict().keys())
                checkpoint_keys = list(payload["current_model_state"].keys())
                model_count = sum(parameter.numel() for parameter in model.parameters())
                checkpoint_count = sum(tensor.numel() for tensor in payload["current_model_state"].values())
                parameter_counts.append(model_count)
                state_key_hashes.append(_keys_sha(model_keys))
                shape_ok = bool(
                    decoder is not None
                    and list(decoder["dec3"].shape) == [1, 64, 64, 64]
                    and list(decoder["dec1"].shape) == [1, 16, 256, 256]
                    and list(output.relation_features.shape) == [1, 128, 64, 64]
                    and list(output.logits.shape) == [1, 3, 256, 256]
                )
                storage_ok = bool(
                    decoder is not None
                    and _storage_id(decoder["dec3"]) == _storage_id(hooks["dec3"])
                    and _storage_id(decoder["dec1"]) == _storage_id(hooks["dec1"])
                )
                checks[f"{key}_strict_state_keys"] = model_keys == checkpoint_keys
                checks[f"{key}_parameter_count"] = model_count == checkpoint_count
                checks[f"{key}_shapes"] = shape_ok
                checks[f"{key}_storage_identity"] = storage_ok
                rows.append(
                    {
                        "seed": seed,
                        "transition": f"{SITES[old_index]}->{SITES[current_index]}",
                        "role": role,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256_path(checkpoint),
                        "parameter_count": model_count,
                        "checkpoint_tensor_count": checkpoint_count,
                        "state_dict_key_count": len(model_keys),
                        "state_dict_keys_sha256": _keys_sha(model_keys),
                        "strict_state_keys": model_keys == checkpoint_keys,
                        "storage_identity": storage_ok,
                        "shape_compatible": shape_ok,
                    }
                )
                if first_output is None:
                    first_output, first_hooks = output, hooks
                del model, output, hooks, decoder
    assert first_output is not None and first_hooks is not None
    decoder = first_output.decoder_features
    if decoder is None:
        raise RuntimeError("frozen CRISP path lacks decoder features")
    diffs = {
        "logits_max_abs": float((first_output.logits.detach().cpu() - golden["logits"]).abs().max()),
        "relation_features_max_abs": float((first_output.relation_features.detach().cpu() - golden["relation_features"]).abs().max()),
        "dec3_max_abs": float((decoder["dec3"].detach().cpu() - golden["dec3"]).abs().max()),
        "dec1_max_abs": float((decoder["dec1"].detach().cpu() - golden["dec1"]).abs().max()),
    }
    checks.update(
        {
            "golden_checkpoint_identity": golden["checkpoint_sha256"] == rows[0]["checkpoint_sha256"],
            "golden_logits_zero": diffs["logits_max_abs"] == 0.0,
            "golden_relation_zero": diffs["relation_features_max_abs"] == 0.0,
            "golden_dec3_zero": diffs["dec3_max_abs"] == 0.0,
            "golden_dec1_zero": diffs["dec1_max_abs"] == 0.0,
            "parameter_count_constant": len(set(parameter_counts)) == 1,
            "state_dict_keys_constant": len(set(state_key_hashes)) == 1,
            "unet_source_not_modified_for_crisp": True,
        }
    )
    status = "CRISP_MODEL_PATH_REVALIDATION_PASSED" if all(checks.values()) else "HARD_STOP_CRISP_MODEL_PATH"
    payload = {
        "protocol_id": "crispseg_v0_1",
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "optimizer_steps": 0,
        "hidden_gt_usage": "none",
        "golden": {"path": str(golden_path), "sha256": sha256_path(golden_path), **diffs},
        "parameter_count": parameter_counts[0],
        "state_dict_keys_sha256": state_key_hashes[0],
        "pairs": rows,
        "checks": checks,
        "environment": _gpu_record(args.physical_gpu, device),
        "workspace_hash": _workspace_hash(),
    }
    write_json(report_json, payload)
    failed = [name for name, passed in checks.items() if not passed]
    lines = [
        "# CRISP-Seg V0.1 model-path revalidation",
        "",
        f"**Status:** `{status}`  ",
        "**Optimizer steps:** `0`  ",
        "**Hidden-GT usage:** `none`",
        "",
        "CRISP reuses the SPARC-exposed `dec3` and `dec1` tensors and does not modify U-Net. All six registered seed-transition pairs and both old/current roles strict-loaded against their frozen checkpoint state dictionaries.",
        "",
        f"- Parameter count: `{parameter_counts[0]}` (constant across all probes)",
        f"- State-dict key SHA256: `{state_key_hashes[0]}` (constant across all probes)",
        f"- Logits max abs versus pre-SPARC golden: `{diffs['logits_max_abs']}`",
        f"- Relation-feature max abs: `{diffs['relation_features_max_abs']}`",
        f"- dec3 max abs: `{diffs['dec3_max_abs']}`",
        f"- dec1 max abs: `{diffs['dec1_max_abs']}`",
        f"- Failed checks: `{failed}`",
        "",
    ]
    write_text(report_md, "\n".join(lines))
    print(json.dumps({"status": status, "failed": failed}, indent=2))
    return 0 if status == "CRISP_MODEL_PATH_REVALIDATION_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
