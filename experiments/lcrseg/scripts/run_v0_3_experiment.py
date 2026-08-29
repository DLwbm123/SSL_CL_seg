#!/usr/bin/env python3
"""Run one immutable V0.3 Fundus R0/R1 multi-seed configuration."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.common import sha256_path
from lcrseg.data import DeterministicBatcher, collate_labeled, collate_unlabeled
from lcrseg.engine.continual_runner import ContinualRunner
from lcrseg.methods.lcrseg_v0_3 import FROZEN_FUNDUS_MANIFEST_HASHES, FROZEN_FUNDUS_SPLIT_HASHES


RUN_NAMES = {
    (1, "R0"): "fundus_seed1_lcrseg_v0_3_r0_legacy_uniform_full200e",
    (1, "R1"): "fundus_seed1_lcrseg_v0_3_r1_progressive_uniform_full200e",
    (2, "R0"): "fundus_seed2_lcrseg_v0_3_r0_legacy_uniform_full200e",
    (2, "R1"): "fundus_seed2_lcrseg_v0_3_r1_progressive_uniform_full200e",
}


def build_config(variant: str, seed: int, run_root: Path) -> dict:
    variant = variant.upper()
    if (seed, variant) not in RUN_NAMES:
        raise ValueError("formal V0.3 multi-seed runs are restricted to R0/R1 seeds 1 and 2")
    path = PROJECT_ROOT / "configs/experiments" / f"lcrseg_v0_3_{variant.lower()}.yaml"
    config = json.loads(path.read_text())
    config["experiment"].update(
        {"seed": seed, "run_name": RUN_NAMES[(seed, variant)], "run_root": str(run_root.resolve())}
    )
    return config


def preflight(runner: ContinualRunner) -> dict:
    total = 0
    sites = []
    for site_id in runner.site_order:
        labeled, unlabeled = runner._datasets((site_id,))
        labeled_batcher = DeterministicBatcher(
            labeled,
            batch_size=int(runner.config["training"]["labeled_batch_size"]),
            seed=runner.seed,
            namespace=f"v0_3_preflight:{site_id}:labeled",
            collate=collate_labeled,
        )
        unlabeled_batcher = DeterministicBatcher(
            unlabeled,
            batch_size=int(runner.config["training"]["unlabeled_batch_size"]),
            seed=runner.seed,
            namespace=f"v0_3_preflight:{site_id}:unlabeled",
            collate=collate_unlabeled,
        )
        site_total, per_epoch = runner._total_steps(labeled_batcher, unlabeled_batcher, scope=(site_id,))
        total += site_total
        sites.append({"site": site_id, "steps_per_epoch": per_epoch, "total_steps": site_total})
    manifest = sha256_path(runner.manifest_path)
    split = sha256_path(runner.split_path)
    if total != 13_400:
        raise AssertionError(f"V0.3 budget resolved to {total}, not 13400")
    if manifest != FROZEN_FUNDUS_MANIFEST_HASHES[runner.seed]:
        raise AssertionError("training manifest hash differs from the frozen V0.3 preregistration")
    if split != FROZEN_FUNDUS_SPLIT_HASHES[runner.seed]:
        raise AssertionError("Fundus split hash differs from the frozen V0.3 preregistration")
    return {
        "protocol_id": "lcrseg_v0_3",
        "variant_id": runner.config["method"]["variant_id"],
        "seed": runner.seed,
        "manifest_sha256": manifest,
        "split_sha256": split,
        "total_steps": total,
        "sites": sites,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("R0", "R1", "r0", "r1"), required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = build_config(args.variant, args.seed, args.run_root)
    runner = ContinualRunner(config)
    plan = preflight(runner)
    if args.validate_only:
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"4", "5", "6", "7"}:
        raise RuntimeError("formal V0.3 runs require one declared physical RTX 3090 in GPU 4-7")
    print(json.dumps(runner.run(resume_checkpoint=args.resume), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

