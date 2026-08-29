#!/usr/bin/env python3
"""Run one immutable formal LCR-Seg V0.2a Fundus configuration."""
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


VARIANTS = {
    "lcrseg_v0_2a_r0": ("R0", "legacy_continuous_v01", "uniform_relation"),
    "lcrseg_v0_2a_r1": ("R1", "progressive_admission", "uniform_relation"),
    "lcrseg_v0_2a_r2": ("R2", "legacy_continuous_v01", "calibrated_teacher_rejection"),
    "lcrseg_v0_2a_r3": ("R3", "progressive_admission", "calibrated_teacher_rejection"),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("V0.2a config must be a JSON/YAML object")
    return value


def _validate(config: dict[str, Any], path: Path) -> tuple[str, str, str]:
    if path.stem not in VARIANTS:
        raise ValueError(f"not a registered V0.2a config: {path}")
    expected = VARIANTS[path.stem]
    method = config.get("method", {})
    actual = (method.get("variant_id"), method.get("assimilation_mode"), method.get("consolidation_mode"))
    if actual != expected:
        raise ValueError(f"registered semantics mismatch: expected {expected}, got {actual}")
    if method.get("name") != "lcrseg_v0_2a" or method.get("protocol_id") != "lcrseg_v0_2a":
        raise ValueError("config does not select lcrseg_v0_2a")
    if config.get("data", {}).get("dataset") != "fundus" or int(config.get("experiment", {}).get("seed", -1)) != 0:
        raise ValueError("formal V0.2a configs require Fundus seed 0")
    if config.get("data", {}).get("evaluation_role") != "val":
        raise ValueError("formal V0.2a configs require validation evaluation")
    if int(config.get("training", {}).get("epochs_per_site", -1)) != 200:
        raise ValueError("formal V0.2a configs require 200 epochs per site")
    return expected


def _assert_formal_gpu() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"4", "5", "6", "7"}:
        raise RuntimeError(f"formal V0.2a execution requires one physical GPU in 4-7, got {visible!r}")


def _plan(runner: ContinualRunner) -> dict[str, Any]:
    sites: list[dict[str, Any]] = []
    total = 0
    for site_id in runner.site_order:
        labeled, unlabeled = runner._datasets((site_id,))
        labeled_batcher = DeterministicBatcher(labeled, batch_size=int(runner.config["training"]["labeled_batch_size"]), seed=runner.seed, namespace=f"v0_2a_preflight:{site_id}:labeled", collate=collate_labeled)
        unlabeled_batcher = DeterministicBatcher(unlabeled, batch_size=int(runner.config["training"]["unlabeled_batch_size"]), seed=runner.seed, namespace=f"v0_2a_preflight:{site_id}:unlabeled", collate=collate_unlabeled)
        site_total, steps_per_epoch = runner._total_steps(labeled_batcher, unlabeled_batcher, scope=(site_id,))
        sites.append({"site": site_id, "steps_per_epoch": steps_per_epoch, "total_steps": site_total})
        total += site_total
    if total != 13400:
        raise AssertionError(f"formal V0.2a budget resolved to {total}, not 13400")
    return {
        "protocol_id": "lcrseg_v0_2a",
        "variant_id": runner.config["method"]["variant_id"],
        "total_steps": total,
        "sites": sites,
        "manifest_sha256": sha256_path(runner.manifest_path),
        "split_sha256": sha256_path(runner.split_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    config = _load(args.config)
    variant, _, _ = _validate(config, args.config)
    runner = ContinualRunner(config)
    plan = _plan(runner)
    if args.validate_only:
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return
    if variant == "R0":
        raise RuntimeError("formal R0 is the frozen legacy artifact and must not be rerun or overwritten")
    _assert_formal_gpu()
    print(json.dumps(runner.run(resume_checkpoint=args.resume), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
