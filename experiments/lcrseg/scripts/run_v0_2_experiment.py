#!/usr/bin/env python3
"""Run one preregistered LCR-Seg V0.2 configuration without mutable overrides."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.common import sha256_path
from lcrseg.data import DeterministicBatcher, collate_labeled, collate_unlabeled
from lcrseg.engine.continual_runner import ContinualRunner


VARIANT_FLAGS = {
    "lcrseg_v0_2_r0_uniform": (False, False, False),
    "lcrseg_v0_2_r1_learnability_admission": (True, False, False),
    "lcrseg_v0_2_r2_compatibility_reject": (False, True, True),
    "lcrseg_v0_2_r3_asymmetric_full": (True, True, True),
}


def _load_config(path: Path) -> dict[str, Any]:
    # Preregistered YAML files intentionally use JSON syntax, which is valid
    # YAML 1.2 and avoids an additional parser dependency on the frozen host.
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("V0.2 configuration must be a JSON object")
    return value


def _validate_config(config: dict[str, Any], path: Path) -> None:
    stem = path.stem
    if stem not in VARIANT_FLAGS:
        raise ValueError(f"configuration is not one of the preregistered R0-R3 variants: {path}")
    if config.get("data", {}).get("dataset") != "fundus":
        raise ValueError("the preregistered V0.2 runner accepts Fundus configurations only")
    if int(config.get("experiment", {}).get("seed", -1)) != 0:
        raise ValueError("the preregistered V0.2 configurations require seed 0")
    if config.get("data", {}).get("evaluation_role") != "val":
        raise ValueError("the preregistered V0.2 configurations require validation role")
    if int(config.get("training", {}).get("epochs_per_site", -1)) != 200:
        raise ValueError("the preregistered V0.2 configurations require 200 epochs per site")
    method = config.get("method", {})
    if method.get("name") != "lcrseg_v0_2" or str(method.get("version")) != "0.2":
        raise ValueError("configuration must select LCR-Seg V0.2")
    actual = tuple(bool(method.get(key)) for key in ("progressive_admission", "compatibility_calibration", "compatibility_rejection"))
    if actual != VARIANT_FLAGS[stem]:
        raise ValueError(f"{stem} has a non-preregistered routing flag combination: {actual}")


def _plan(runner: ContinualRunner) -> dict[str, Any]:
    """Read frozen manifests only and prove the configured 13,400-step budget."""

    sites: list[dict[str, Any]] = []
    total = 0
    for site_index, site_id in enumerate(runner.site_order):
        labeled_dataset, unlabeled_dataset = runner._datasets((site_id,))
        labeled_batcher = DeterministicBatcher(
            labeled_dataset,
            batch_size=int(runner.config["training"]["labeled_batch_size"]),
            seed=runner.seed,
            namespace=f"v0_2_preflight:{runner.dataset}:{site_id}:labeled",
            collate=collate_labeled,
        )
        unlabeled_batcher = DeterministicBatcher(
            unlabeled_dataset,
            batch_size=int(runner.config["training"]["unlabeled_batch_size"]),
            seed=runner.seed,
            namespace=f"v0_2_preflight:{runner.dataset}:{site_id}:unlabeled",
            collate=collate_unlabeled,
        )
        site_total, steps_per_epoch = runner._total_steps(labeled_batcher, unlabeled_batcher, scope=(site_id,))
        sites.append(
            {
                "site_index": site_index,
                "site_id": site_id,
                "labeled_records": len(labeled_dataset),
                "unlabeled_records": len(unlabeled_dataset),
                "steps_per_epoch": steps_per_epoch,
                "total_steps": site_total,
            }
        )
        total += int(site_total)
    if total != 13400:
        raise AssertionError(f"preregistered V0.2 plan resolves to {total} steps, not 13400")
    return {
        "dataset": runner.dataset,
        "seed": runner.seed,
        "run_name": runner.config["experiment"]["run_name"],
        "manifest_sha256": sha256_path(runner.manifest_path),
        "split_sha256": sha256_path(runner.split_path),
        "total_steps": total,
        "sites": sites,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true", help="Read frozen manifests and verify the fixed budget without training.")
    args = parser.parse_args()

    config = _load_config(args.config)
    _validate_config(config, args.config)
    if args.root is not None:
        config["data"]["data_root"] = str(args.root)
    elif os.environ.get("LCRSEG_DATA_ROOT"):
        config["data"]["data_root"] = os.environ["LCRSEG_DATA_ROOT"]
    if args.run_root is not None:
        config["experiment"]["run_root"] = str(args.run_root)
    elif os.environ.get("LCRSEG_RUN_ROOT"):
        config["experiment"]["run_root"] = os.environ["LCRSEG_RUN_ROOT"]
    if args.device is not None:
        config["experiment"]["device"] = args.device

    runner = ContinualRunner(config)
    if args.validate_only:
        print(json.dumps(_plan(runner), ensure_ascii=False, sort_keys=True))
        return
    summary = runner.run(resume_checkpoint=args.resume)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
