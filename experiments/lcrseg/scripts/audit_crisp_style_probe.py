#!/usr/bin/env python3
"""Compile the prerequisite CRISP style-probe audit without optimizer steps."""
from __future__ import annotations

import argparse
import inspect
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
from lcrseg.data.transforms import _flip  # noqa: E402
from lcrseg.representation.style_probe import (  # noqa: E402
    FrozenStyleProbeTransform,
    crisp_style_probe_contract,
)


FROZEN_VALUES = {
    "flip_probability": 0.5,
    "strong_noise_std": 0.03,
    "brightness_delta": 0.1,
    "contrast_delta": 0.1,
    "cutout_probability": 0.5,
    "cutout_fraction": 0.2,
}


def _exercise(dataset: str, channels: int) -> tuple[dict[str, bool], dict[str, Any]]:
    image = torch.linspace(0.0, 1.0, channels * 31 * 29, dtype=torch.float32).reshape(channels, 31, 29)
    image_before = image.clone()
    rng_before = torch.get_rng_state().clone()
    transform = FrozenStyleProbeTransform(protocol_seed=0)
    first = transform(image=image, dataset=dataset, site_id="audit_site", case_id=f"{dataset}_audit_case")
    second = transform(image=image, dataset=dataset, site_id="audit_site", case_id=f"{dataset}_audit_case")
    rng_after = torch.get_rng_state()
    geometry = first["geometry_record"]
    expected_clean = _flip(image, hflip=geometry["hflip"], vflip=geometry["vflip"])
    checks = {
        f"{dataset}_input_immutable": torch.equal(image, image_before),
        f"{dataset}_global_rng_immutable": torch.equal(rng_before, rng_after),
        f"{dataset}_repeat_clean_bitwise": torch.equal(first["clean_image"], second["clean_image"]),
        f"{dataset}_repeat_style_bitwise": torch.equal(first["style_image"], second["style_image"]),
        f"{dataset}_repeat_records_equal": first["geometry_record"] == second["geometry_record"] and first["style_record"] == second["style_record"],
        f"{dataset}_clean_is_exact_shared_geometry": torch.equal(first["clean_image"], expected_clean),
        f"{dataset}_paired_shapes_equal": first["clean_image"].shape == first["style_image"].shape,
        f"{dataset}_style_differs_by_appearance": not torch.equal(first["clean_image"], first["style_image"]),
        f"{dataset}_cutout_disabled": geometry["cutout"] is False and geometry["cutout_box"] is None,
        f"{dataset}_all_pixels_valid": bool(first["style_valid_mask"].all()),
        f"{dataset}_finite": bool(torch.isfinite(first["style_image"]).all()),
    }
    record = {
        "input_shape": list(image.shape),
        "geometry_record": geometry,
        "style_record": first["style_record"],
        "clean_mean": float(first["clean_image"].mean()),
        "style_mean": float(first["style_image"].mean()),
        "paired_mean_absolute_difference": float((first["style_image"] - first["clean_image"]).abs().mean()),
    }
    return checks, record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "lcrseg_v0_2_r0_uniform.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "experiment_status")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    report_json = output_dir / "CRISP_STYLE_PROBE_AUDIT.json"
    report_md = output_dir / "CRISP_STYLE_PROBE_AUDIT.md"
    for path in (report_json, report_md):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite CRISP style audit: {path}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    configured = config.get("transforms", {})
    contract = crisp_style_probe_contract()
    checks: dict[str, bool] = {
        "formal_r0_frozen_values_exact": all(float(configured.get(key, float("nan"))) == value for key, value in FROZEN_VALUES.items()),
        "contract_flip_exact": contract["geometry"]["flip_probability"] == FROZEN_VALUES["flip_probability"],
        "contract_noise_exact": contract["appearance"]["strong_noise_std"] == FROZEN_VALUES["strong_noise_std"],
        "contract_brightness_exact": contract["appearance"]["brightness_delta"] == FROZEN_VALUES["brightness_delta"],
        "contract_contrast_exact": contract["appearance"]["contrast_delta"] == FROZEN_VALUES["contrast_delta"],
        "contract_no_cutout": contract["cutout"] is False,
        "contract_no_new_augmentation": contract["new_augmentation"] is False,
        "contract_same_geometry": contract["geometry"]["paired_views_share_exact_geometry"] is True,
        "contract_case_deterministic": contract["seed_components"] == ["protocol_seed", "site_id", "case_id"],
        "call_interface_has_no_label": "label" not in inspect.signature(FrozenStyleProbeTransform.__call__).parameters,
    }
    probes: dict[str, Any] = {}
    for dataset, channels in (("fundus", 3), ("prostate", 1)):
        dataset_checks, probes[dataset] = _exercise(dataset, channels)
        checks.update(dataset_checks)

    transform_source = ROOT / "lcrseg" / "data" / "transforms.py"
    runner_source = ROOT / "lcrseg" / "engine" / "continual_runner.py"
    style_source = ROOT / "lcrseg" / "representation" / "style_probe.py"
    source_text = transform_source.read_text(encoding="utf-8")
    runner_text = runner_source.read_text(encoding="utf-8")
    checks.update(
        {
            "existing_transform_contains_noise": "torch.randn_like(strong_image) * self.strong_noise_std" in source_text,
            "existing_transform_contains_brightness": "strong_image = strong_image * brightness" in source_text,
            "existing_transform_contains_contrast": "(strong_image - center) * contrast + center" in source_text,
            "existing_transform_uses_shared_geometry": "strong_image = weak_image.clone()" in source_text,
            "runner_defaults_match_noise": 'transforms.get("strong_noise_std", 0.03)' in runner_text,
            "runner_defaults_match_brightness": 'transforms.get("brightness_delta", 0.10)' in runner_text,
            "runner_defaults_match_contrast": 'transforms.get("contrast_delta", 0.10)' in runner_text,
            "audit_optimizer_steps_zero": True,
            "audit_hidden_gt_usage_none": True,
        }
    )
    failed = [name for name, passed in checks.items() if not passed]
    status = "CRISP_STYLE_PROBE_AUDIT_PASSED" if not failed else "HARD_STOP_CRISP_STYLE_PROBE"
    payload = {
        "protocol_id": "crispseg_v0_1",
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "optimizer_steps": 0,
        "hidden_gt_usage": "none",
        "pseudo_label_usage": "none",
        "contract": contract,
        "frozen_config": {
            "path": str(args.config.resolve()),
            "sha256": sha256_path(args.config.resolve()),
            "values": configured,
        },
        "source_files": {
            "existing_transform": {"path": str(transform_source), "sha256": sha256_path(transform_source)},
            "continual_runner": {"path": str(runner_source), "sha256": sha256_path(runner_source)},
            "style_probe": {"path": str(style_source), "sha256": sha256_path(style_source)},
        },
        "dataset_operators": {
            "fundus": {
                "channels": 3,
                "clean": "same deterministic horizontal/vertical flips only",
                "style": "existing scalar contrast, scalar brightness, and elementwise Gaussian noise",
                "color_specific_operator": "none",
            },
            "prostate": {
                "channels": 1,
                "clean": "same deterministic horizontal/vertical flips only",
                "style": "existing scalar contrast, scalar brightness, and elementwise Gaussian noise",
                "color_specific_operator": "not applicable",
            },
        },
        "probes": probes,
        "checks": checks,
        "failed_checks": failed,
    }
    write_json(report_json, payload)
    lines = [
        "# CRISP-Seg V0.1 style-probe audit",
        "",
        f"**Status:** `{status}`  ",
        "**Optimizer steps:** `0`  ",
        "**Hidden-GT usage:** `none`  ",
        "**Pseudo-label usage:** `none`",
        "",
        "The paired role-probe views share one deterministic horizontal/vertical flip. The style view applies only the already registered contrast, brightness, and Gaussian-noise path. Cutout is disabled because it changes spatial support.",
        "",
        f"- Contract SHA256: `{contract['contract_sha256']}`",
        f"- Frozen magnitudes: flip `{FROZEN_VALUES['flip_probability']}`, noise std `{FROZEN_VALUES['strong_noise_std']}`, brightness delta `{FROZEN_VALUES['brightness_delta']}`, contrast delta `{FROZEN_VALUES['contrast_delta']}`",
        "- Fundus: three-channel intensity path; no new hue/saturation/color operator.",
        "- Prostate MRI: the same existing operations on a single intensity channel.",
        "- Case RNG key: `protocol_seed + site_id + case_id`; the global PyTorch RNG is unchanged.",
        f"- Failed checks: `{failed}`",
        "",
    ]
    write_text(report_md, "\n".join(lines))
    print(json.dumps({"status": status, "failed": failed, "contract_sha256": contract["contract_sha256"]}, indent=2))
    return 0 if status == "CRISP_STYLE_PROBE_AUDIT_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
