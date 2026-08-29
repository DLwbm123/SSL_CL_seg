#!/usr/bin/env python3
"""Audit the frozen relation-space contract before ASPR feasibility work.

This script is intentionally read-only with respect to data and prior runs.  It
loads every preregistered R0 site checkpoint, validates the frozen class
semantics, and probes old/current model feature compatibility.  It does not
construct site memory and performs no optimizer step.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.common import sha256_path, write_json, write_text  # noqa: E402
from lcrseg.engine.checkpoint import load_checkpoint  # noqa: E402
from lcrseg.methods.components.progressive_admission import strict_relation_valid_mask  # noqa: E402
from lcrseg.models import UNet2D  # noqa: E402


RUNS = {
    0: "fundus_seed0_lcrseg_uniform_relation_kd_full200e",
    1: "fundus_seed1_lcrseg_v0_3_r0_legacy_uniform_full200e",
    2: "fundus_seed2_lcrseg_v0_3_r0_legacy_uniform_full200e",
}
SITES = ("REFUGE", "RIM_ONE_r3", "Drishti_GS")
CLASS_ORDER = (
    {"class_id": 0, "class_name": "background"},
    {"class_id": 1, "class_name": "optic_disc_rim"},
    {"class_id": 2, "class_name": "optic_cup"},
)


def _workspace_hash() -> str:
    """Hash implementation inputs while excluding caches and generated reports."""

    digest = hashlib.sha256()
    roots = ("lcrseg", "scripts", "tests", "configs")
    files: list[Path] = []
    for root_name in roots:
        root = ROOT / root_name
        if not root.exists():
            continue
        files.extend(path for path in root.rglob("*") if path.is_file())
    files.extend(
        path
        for path in (ROOT / "AGENTS.md", ROOT / "METHOD_SPEC_V0_1.md", ROOT / "IMPLEMENTATION_CONTRACT_V0_1.md")
        if path.is_file()
    )
    for path in sorted(files):
        relative = path.relative_to(ROOT)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_path(path)))
    return digest.hexdigest()


def _gpu_record(physical_index: int, device: torch.device) -> dict[str, Any]:
    record: dict[str, Any] = {
        "physical_index": physical_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={physical_index}",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        uuid, name, driver = (item.strip() for item in result.stdout.strip().split(",", maxsplit=2))
        record.update(uuid=uuid, name=name, driver=driver)
    except (OSError, subprocess.CalledProcessError, ValueError):
        record.update(uuid="unavailable", name="unavailable", driver="unavailable")
    return record


def _new_model(config: dict[str, Any], device: torch.device) -> UNet2D:
    model_cfg = config["model"]
    return UNet2D(
        int(model_cfg["in_channels"]),
        int(model_cfg["num_classes"]),
        base_channels=int(model_cfg.get("base_channels", 16)),
        relation_dim=int(model_cfg["relation_dim"]),
    ).to(device)


def _anchor_contract(state: dict[str, Any]) -> dict[str, Any]:
    if not state:
        return {"present": False, "shape": [], "valid": [], "all_classes_valid": False}
    anchors = state.get("anchors")
    valid = state.get("valid")
    return {
        "present": anchors is not None and valid is not None,
        "shape": list(anchors.shape) if anchors is not None else [],
        "valid": valid.detach().cpu().tolist() if valid is not None else [],
        "all_classes_valid": bool(valid.all()) if valid is not None else False,
        "finite": bool(torch.isfinite(anchors).all()) if anchors is not None else False,
        "normalized_valid_anchors": bool(
            torch.allclose(
                anchors[valid].float().norm(dim=-1),
                torch.ones_like(anchors[valid].float().norm(dim=-1)),
                atol=1.0e-5,
                rtol=0.0,
            )
        )
        if anchors is not None and valid is not None and bool(valid.any())
        else False,
    }


def _load_semantics() -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    semantics_path = ROOT / "reports" / "experiment_status" / "class_semantics.json"
    label_map_path = ROOT / "configs" / "data" / "fundus_label_map.yaml"
    semantics = json.loads(semantics_path.read_text(encoding="utf-8"))
    label_map = yaml.safe_load(label_map_path.read_text(encoding="utf-8"))
    yaml_order = tuple(
        {"class_id": int(class_id), "class_name": str(class_name)}
        for class_id, class_name in sorted(label_map["class_names"].items(), key=lambda item: int(item[0]))
    )
    yaml_mapping = {str(raw): int(training) for raw, training in label_map["mapping"].items()}
    checks = {
        "label_map_frozen_and_confirmed": label_map.get("status") == "FROZEN_USER_AUTHORIZED_V1"
        and bool(label_map.get("confirmed")),
        "yaml_class_order_exact": yaml_order == CLASS_ORDER,
        "json_class_order_exact": tuple(semantics.get("training_class_order", ())) == CLASS_ORDER,
        "raw_mapping_exact": yaml_mapping == semantics.get("raw_to_training_label") == {"0": 2, "128": 1, "255": 0},
        # The frozen file names the relation axis by reference to the anchor
        # axis rather than duplicating the literal training-axis string.  The
        # semantic bridge is therefore a two-edge contract, not three equal
        # strings.
        "classifier_relation_anchor_axes_agree": semantics.get("segmentation_classifier_class_axis")
        == semantics.get("anchor_bank_class_axis")
        == "training class_id order"
        and semantics.get("relation_distribution_class_axis") == "anchor bank class_id order",
    }
    return semantics, label_map, checks


def audit(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    run_root = args.run_root.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA audit requested but CUDA is unavailable")

    semantics, label_map, semantics_checks = _load_semantics()
    checks: dict[str, bool] = dict(semantics_checks)
    checkpoint_rows: list[dict[str, Any]] = []
    payloads: dict[tuple[int, int], dict[str, Any]] = {}
    expected_dim: int | None = None
    temperatures: set[float] = set()

    for seed, run_name in RUNS.items():
        manifest = data_root / "manifests" / "training" / f"lcrseg_v1_seed{seed}.csv"
        split = data_root / "splits" / f"fundus_seed{seed}.json"
        checks[f"seed{seed}_manifest_present"] = manifest.is_file()
        checks[f"seed{seed}_split_present"] = split.is_file()
        manifest_sha = sha256_path(manifest) if manifest.is_file() else ""
        split_sha = sha256_path(split) if split.is_file() else ""
        for site_index, site_id in enumerate(SITES):
            checkpoint = run_root / run_name / f"checkpoint_final_site{site_index}_{site_id}.pt"
            key = f"seed{seed}_site{site_index}"
            checks[f"{key}_checkpoint_present"] = checkpoint.is_file()
            if not checkpoint.is_file():
                checkpoint_rows.append(
                    {"seed": seed, "site_index": site_index, "site_id": site_id, "path": str(checkpoint), "present": False}
                )
                continue
            payload = load_checkpoint(checkpoint, map_location="cpu")
            payloads[(seed, site_index)] = payload
            config = payload["config_resolved"]
            model_cfg = config["model"]
            method_cfg = config["method"]
            relation_dim = int(model_cfg["relation_dim"])
            expected_dim = relation_dim if expected_dim is None else expected_dim
            temperature = float(method_cfg.get("relation_temperature", 0.1))
            temperatures.add(temperature)
            current_anchor = _anchor_contract(payload["current_anchor_state"])
            historical_anchor = _anchor_contract(payload["historical_anchor_state"])
            row = {
                "seed": seed,
                "site_index": site_index,
                "site_id": site_id,
                "path": str(checkpoint),
                "present": True,
                "sha256": sha256_path(checkpoint),
                "manifest_sha256": manifest_sha,
                "split_sha256": split_sha,
                "payload_manifest_sha256": payload["manifest_hash"],
                "payload_split_sha256": payload["data_split_hash"],
                "method_name": payload["method_name"],
                "relation_dim": relation_dim,
                "relation_temperature": temperature,
                "current_anchor": current_anchor,
                "historical_anchor": historical_anchor,
            }
            checkpoint_rows.append(row)
            checks[f"{key}_site_identity_exact"] = payload["site_id"] == site_id and int(payload["site_index"]) == site_index
            checks[f"{key}_hash_lineage_exact"] = payload["manifest_hash"] == manifest_sha and payload["data_split_hash"] == split_sha
            checks[f"{key}_fundus_model_contract"] = (
                int(model_cfg["in_channels"]) == 3
                and int(model_cfg["num_classes"]) == 3
                and int(model_cfg.get("base_channels", 16)) == 16
                and relation_dim == 128
            )
            checks[f"{key}_current_anchor_contract"] = (
                current_anchor["shape"] == [3, 1, relation_dim]
                and current_anchor["all_classes_valid"]
                and current_anchor["finite"]
                and current_anchor["normalized_valid_anchors"]
            )
            checks[f"{key}_historical_anchor_lifecycle"] = (
                not historical_anchor["present"]
                if site_index == 0
                else historical_anchor["shape"] == [3, 1, relation_dim]
                and historical_anchor["all_classes_valid"]
                and historical_anchor["finite"]
                and historical_anchor["normalized_valid_anchors"]
            )

    checks["all_relation_dimensions_identical"] = expected_dim == 128 and all(
        row.get("relation_dim") == expected_dim for row in checkpoint_rows if row.get("present")
    )
    checks["relation_temperature_frozen"] = temperatures == {0.1}

    dummy = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device=device)
    pair_rows: list[dict[str, Any]] = []
    for seed in RUNS:
        for old_index, current_index in ((0, 1), (1, 2)):
            key = f"seed{seed}_pair{old_index}_{current_index}"
            old_payload = payloads.get((seed, old_index))
            current_payload = payloads.get((seed, current_index))
            if old_payload is None or current_payload is None:
                checks[f"{key}_pairable"] = False
                pair_rows.append({"seed": seed, "old_site_index": old_index, "current_site_index": current_index, "pairable": False})
                continue
            old_model = _new_model(old_payload["config_resolved"], device)
            current_model = _new_model(current_payload["config_resolved"], device)
            old_model.load_state_dict(old_payload["current_model_state"], strict=True)
            current_model.load_state_dict(current_payload["current_model_state"], strict=True)
            old_model.eval()
            current_model.eval()
            with torch.inference_mode():
                old_output = old_model(dummy)
                current_output = current_model(dummy)
            old_shape = list(old_output.relation_features.shape)
            current_shape = list(current_output.relation_features.shape)
            pairable = (
                old_shape == current_shape == [1, 128, 64, 64]
                and list(old_output.logits.shape) == list(current_output.logits.shape) == [1, 3, 256, 256]
                and bool(torch.isfinite(old_output.relation_features).all())
                and bool(torch.isfinite(current_output.relation_features).all())
                and bool(
                    torch.allclose(
                        old_output.relation_features.norm(dim=1),
                        torch.ones((1, 64, 64), device=device),
                        atol=1.0e-5,
                        rtol=0.0,
                    )
                )
                and bool(
                    torch.allclose(
                        current_output.relation_features.norm(dim=1),
                        torch.ones((1, 64, 64), device=device),
                        atol=1.0e-5,
                        rtol=0.0,
                    )
                )
            )
            checks[f"{key}_pairable"] = pairable
            pair_rows.append(
                {
                    "seed": seed,
                    "old_site_index": old_index,
                    "current_site_index": current_index,
                    "old_relation_shape": old_shape,
                    "current_relation_shape": current_shape,
                    "pairable": pairable,
                }
            )
            del old_model, current_model, old_output, current_output

    valid_probe = strict_relation_valid_mask(torch.ones((1, 1, 256, 256), device=device), (64, 64))
    checks["strict_relation_valid_mask_contract"] = (
        list(valid_probe.shape) == [1, 1, 64, 64]
        and valid_probe.dtype == torch.bool
        and bool(valid_probe.all())
        and not valid_probe.requires_grad
    )
    source = inspect.getsource(UNet2D.forward)
    checks["relation_feature_source_is_projection_head_dec3"] = "self.projection_head(dec3)" in source
    checks["relation_grid_is_existing_quarter_resolution"] = all(
        row.get("pairable", False) and row["current_relation_shape"][-2:] == [64, 64] for row in pair_rows
    )

    status = "ASPR_RELATION_SPACE_AUDIT_PASSED" if all(checks.values()) else "HARD_STOP_ASPR_RELATION_SPACE"
    report = {
        "protocol_id": "asprseg_v0_1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "optimizer_steps": 0,
        "hidden_gt_usage": "none",
        "relation_contract": {
            "feature_source": "UNet2D dec3 -> existing ProjectionHead(64, relation_dim)",
            "feature_dimension": expected_dim,
            "probe_input_shape": [1, 3, 256, 256],
            "relation_grid": [64, 64],
            "grid_rule": "existing one-quarter input resolution",
            "class_order": list(CLASS_ORDER),
            "background_id": 0,
            "foreground_ids": [1, 2],
            "relation_temperature": sorted(temperatures),
            "valid_mask": "strict_relation_valid_mask: adaptive-average-pool full-resolution valid mask and require cell mean == 1.0",
            "class_anchor_lifecycle": "site0 current anchors only; incremental checkpoints carry current plus frozen historical class-anchor state",
        },
        "class_semantics": semantics,
        "label_map_status": label_map.get("status"),
        "class_semantics_sha256": sha256_path(ROOT / "reports" / "experiment_status" / "class_semantics.json"),
        "label_map_sha256": sha256_path(ROOT / "configs" / "data" / "fundus_label_map.yaml"),
        "checkpoint_rows": checkpoint_rows,
        "old_current_pairs": pair_rows,
        "checks": checks,
        "environment": _gpu_record(args.physical_gpu, device),
        "workspace_hash": _workspace_hash(),
    }
    if args.supersedes:
        report["audit_correction"] = {
            "supersedes": args.supersedes,
            "reason": "the original audit incorrectly required descriptive axis strings to be literally equal; the frozen semantics define the relation axis by reference to the anchor-bank axis",
            "prior_artifact_preserved": True,
            "data_or_checkpoint_change": False,
        }
    return report


def _markdown(report: dict[str, Any]) -> str:
    failed = [name for name, passed in report["checks"].items() if not passed]
    lines = [
        "# ASPR-Seg V0.1 relation-space audit",
        "",
        f"**Status:** `{report['status']}`  ",
        f"**Optimizer steps:** `{report['optimizer_steps']}`  ",
        f"**Workspace hash:** `{report['workspace_hash']}`",
        "",
        "## Frozen contract",
        "",
        "- Feature source: `UNet2D dec3 -> existing ProjectionHead(64, relation_dim)`.",
        "- Relation dimension: `128`; relation grid: existing one-quarter resolution.",
        "- Fundus class order: `0 background`, `1 optic_disc_rim`, `2 optic_cup`.",
        "- Site-memory foreground IDs, if feasibility later passes: `[1, 2]`.",
        "- Existing relation temperature: `0.1`.",
        "- Existing valid mask: strict full-cell valid pooling on the relation grid.",
        "- Existing class anchors remain class-semantic anchors and are not replaced.",
        "",
        "## Checkpoint coverage",
        "",
        "| Seed | Site | SHA-256 | Relation dim | Current anchors | Historical anchors |",
        "|---:|---|---|---:|---|---|",
    ]
    for row in report["checkpoint_rows"]:
        lines.append(
            "| {seed} | {site_id} | `{sha}` | {dim} | {current} | {historical} |".format(
                seed=row["seed"],
                site_id=row["site_id"],
                sha=row.get("sha256", "missing"),
                dim=row.get("relation_dim", "missing"),
                current="valid" if row.get("current_anchor", {}).get("all_classes_valid") else "invalid/missing",
                historical=(
                    "not applicable"
                    if row["site_index"] == 0 and not row.get("historical_anchor", {}).get("present")
                    else "valid"
                    if row.get("historical_anchor", {}).get("all_classes_valid")
                    else "invalid/missing"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Old/current pairing",
            "",
            "All six consecutive-site model pairs were loaded strictly and probed with the same input tensor. "
            "Their logits and normalized relation-feature grids must match exactly in shape.",
            "",
            "## Gate",
            "",
            f"Failed checks: `{failed}`.",
            "",
            "No data, split, manifest, prior report, prior run, or checkpoint was modified. No optimizer step was executed.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "experiment_status")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--artifact-stem", default="ASPR_RELATION_SPACE_AUDIT")
    parser.add_argument("--supersedes", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if not args.artifact_stem or Path(args.artifact_stem).name != args.artifact_stem:
        raise ValueError("artifact stem must be a simple filename stem")
    json_path = output_dir / f"{args.artifact_stem}.json"
    markdown_path = output_dir / f"{args.artifact_stem}.md"
    for path in (json_path, markdown_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite relation-space audit: {path}")
    report = audit(args)
    write_json(json_path, report)
    write_text(markdown_path, _markdown(report))
    print(json.dumps({"status": report["status"], "json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0 if report["status"] == "ASPR_RELATION_SPACE_AUDIT_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
