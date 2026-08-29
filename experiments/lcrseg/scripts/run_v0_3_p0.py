#!/usr/bin/env python3
"""Run V0.3 P0 from the frozen, exact-bridged R1 REFUGE checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.common import sha256_path
from lcrseg.engine.checkpoint import load_checkpoint
from lcrseg.engine.continual_runner import ContinualRunner
from lcrseg.data import DeterministicBatcher, collate_labeled, collate_unlabeled
from lcrseg.methods.lcrseg_v0_3 import (
    FROZEN_FUNDUS_MANIFEST_HASHES,
    FROZEN_FUNDUS_SPLIT_HASHES,
    FROZEN_V02A_R1_SITE0_SHA256,
    validate_parent_checkpoint_lineage,
)


RUN_NAME = "fundus_seed0_lcrseg_v0_3_p0_progressive_norelation_full200e"


def build_config(parent: Path, parent_run: Path, run_root: Path) -> dict:
    parent = parent.resolve()
    parent_run = parent_run.resolve()
    payload = load_checkpoint(parent, map_location="cpu")
    if parent.parent != parent_run:
        raise ValueError("P0 parent checkpoint must belong to the declared R1 run")
    validate_parent_checkpoint_lineage(
        payload,
        checkpoint_sha256=sha256_path(parent),
        expected_sha256=FROZEN_V02A_R1_SITE0_SHA256,
    )
    config = json.loads((PROJECT_ROOT / "configs/experiments/lcrseg_v0_3_p0.yaml").read_text())
    config["experiment"].update(
        {
            "run_name": RUN_NAME,
            "run_root": str(run_root.resolve()),
            "initial_previous_checkpoint": str(parent),
            "initial_previous_run": str(parent_run),
            "initial_global_step": 8000,
            "inherit_completed_site_artifacts": True,
        }
    )
    config["data"].update({"site_order": ["RIM_ONE_r3", "Drishti_GS"], "site_index_offset": 1})
    return config


def preflight(runner: ContinualRunner) -> dict:
    new_steps = 0
    sites = []
    for site_id in runner.site_order:
        labeled, unlabeled = runner._datasets((site_id,))
        labeled_batcher = DeterministicBatcher(
            labeled,
            batch_size=int(runner.config["training"]["labeled_batch_size"]),
            seed=runner.seed,
            namespace=f"v0_3_p0_preflight:{site_id}:labeled",
            collate=collate_labeled,
        )
        unlabeled_batcher = DeterministicBatcher(
            unlabeled,
            batch_size=int(runner.config["training"]["unlabeled_batch_size"]),
            seed=runner.seed,
            namespace=f"v0_3_p0_preflight:{site_id}:unlabeled",
            collate=collate_unlabeled,
        )
        total, per_epoch = runner._total_steps(labeled_batcher, unlabeled_batcher, scope=(site_id,))
        new_steps += total
        sites.append({"site": site_id, "steps_per_epoch": per_epoch, "total_steps": total})
    manifest = sha256_path(runner.manifest_path)
    split = sha256_path(runner.split_path)
    if manifest != FROZEN_FUNDUS_MANIFEST_HASHES[0] or split != FROZEN_FUNDUS_SPLIT_HASHES[0]:
        raise AssertionError("P0 frozen manifest/split hash mismatch")
    if new_steps != 5400 or 8000 + new_steps != 13_400:
        raise AssertionError("P0 lineage budget does not equal the full 13,400-step protocol")
    return {
        "protocol_id": "lcrseg_v0_3",
        "variant_id": "P0",
        "completed_parent_steps": 8000,
        "new_optimizer_steps": new_steps,
        "equivalent_full_run_steps": 8000 + new_steps,
        "manifest_sha256": manifest,
        "split_sha256": split,
        "sites": sites,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = build_config(args.parent, args.parent_run, args.run_root)
    runner = ContinualRunner(config)
    plan = preflight(runner)
    if args.validate_only:
        print(json.dumps({"plan": plan, "config": runner.config}, ensure_ascii=False, sort_keys=True))
        return
    bridge_path = PROJECT_ROOT / "reports/experiment_status/V0_3_P0_SITE1_BRIDGE.json"
    if not bridge_path.is_file():
        raise FileNotFoundError("P0 cannot run before the site-1 bridge report exists")
    bridge = json.loads(bridge_path.read_text())
    if bridge.get("status") != "PASSED" or bridge.get("parent_checkpoint_sha256") != FROZEN_V02A_R1_SITE0_SHA256:
        raise RuntimeError("P0 site-1 bridge did not pass for the frozen parent checkpoint")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"5", "6", "7"}:
        raise RuntimeError("P0 full continuation must use one declared physical GPU in 5-7")
    print(json.dumps(runner.run(resume_checkpoint=args.resume), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
