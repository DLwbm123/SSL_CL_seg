#!/usr/bin/env python3
"""Run one registered SR-GAS parent, pilot, or full continuation."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.common import sha256_path  # noqa: E402
from lcrseg.engine.checkpoint import load_checkpoint  # noqa: E402
from lcrseg.engine.continual_runner import ContinualRunner  # noqa: E402


CONFIGS = {
    "A1": "srgas_a1_cosine.yaml",
    "A2": "srgas_a2_isotropic.yaml",
    "A3": "srgas_a3_totalgas.yaml",
    "A4": "srgas_a4_supgas.yaml",
    "A5": "srgas_a5_stablerelgas_v0_1a.yaml",
    "A6": "srgas_a6_freeze_control.yaml",
}
PILOT_NAMES = {
    "A1": "fundus_seed0_srgas_a1_pilot1000",
    "A2": "fundus_seed0_srgas_a2_pilot1000",
    "A3": "fundus_seed0_srgas_a3_pilot1000",
    "A4": "fundus_seed0_srgas_a4_pilot1000",
    "A5": "fundus_seed0_srgas_a5_pilot1000",
    "A6": "fundus_seed0_srgas_a6_freeze_pilot1000",
}
FULL_STEMS = {
    "A1": "a1_cosine",
    "A2": "a2_isotropic",
    "A3": "a3_totalgas",
    "A4": "a4_supgas",
    "A5": "a5_stablerelgas",
}


def _load_fragment(variant: str) -> dict:
    return json.loads((PROJECT_ROOT / "configs/experiments" / CONFIGS[variant]).read_text())


def build_config(
    *,
    variant: str,
    stage: str,
    seed: int,
    run_root: Path,
    parent_checkpoint: Path | None,
) -> dict:
    base = json.loads((PROJECT_ROOT / "configs/experiments/srgas_a1_cosine.yaml").read_text())
    base["method"].update(_load_fragment(variant)["method"])
    base["experiment"].update({"seed": seed, "optimization_seed": seed, "run_root": str(run_root.resolve())})
    base["method"]["noise_seed"] = seed
    if stage == "parent":
        if variant != "A1" or seed != 0 or parent_checkpoint is not None:
            raise ValueError("the registered parent is seed-0 A1 without a previous checkpoint")
        base["experiment"]["run_name"] = "fundus_seed0_srgas_a1_cosine_site1"
        base["data"]["site_order"] = ["REFUGE"]
        return base
    if parent_checkpoint is None or not parent_checkpoint.is_file():
        raise FileNotFoundError("pilot/full continuation requires the completed common parent checkpoint")
    payload = load_checkpoint(parent_checkpoint, map_location="cpu")
    if payload["method_name"] != "srgas_v0_1" or payload["site_id"] != "REFUGE" or int(payload["site_index"]) != 0:
        raise ValueError("parent checkpoint is not the registered A1 REFUGE artifact")
    if int(payload["site_step"]) < 1:
        raise ValueError("parent checkpoint has no completed optimizer steps")
    parent_run = parent_checkpoint.resolve().parent
    base["experiment"].update(
        {
            "initial_previous_checkpoint": str(parent_checkpoint.resolve()),
            "initial_previous_run": str(parent_run),
            "initial_global_step": int(payload["global_step"]),
            "inherit_completed_site_artifacts": True,
        }
    )
    base["data"]["site_index_offset"] = 1
    if stage == "pilot":
        if seed != 0:
            raise ValueError("engineering pilots are registered only for seed 0")
        base["experiment"]["run_name"] = PILOT_NAMES[variant]
        base["data"]["site_order"] = ["RIM_ONE_r3"]
        base["training"]["steps_per_site"] = 1000
        base["training"]["checkpoint_interval_steps"] = 250
        base["training"]["pilot_trajectory_interval"] = 50
        return base
    if stage != "full" or variant == "A6":
        raise ValueError("A6 is pilot-only and the stage must be parent/pilot/full")
    base["experiment"]["run_name"] = f"fundus_seed{seed}_srgas_{FULL_STEMS[variant]}_full200e"
    base["data"]["site_order"] = ["RIM_ONE_r3", "Drishti_GS"]
    return base


def preflight(config: dict, *, parent_checkpoint: Path | None) -> dict:
    runner = ContinualRunner(config)
    run_dir = Path(config["experiment"]["run_root"]) / config["experiment"]["run_name"]
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {run_dir}")
    return {
        "protocol_id": config["method"]["protocol_id"],
        "variant": config["method"]["srgas_variant"],
        "stage": "parent" if parent_checkpoint is None else "continuation",
        "seed": runner.seed,
        "run_dir": str(run_dir),
        "site_order": list(runner.site_order),
        "site_index_offset": runner.site_index_offset,
        "manifest_sha256": sha256_path(runner.manifest_path),
        "split_sha256": sha256_path(runner.split_path),
        "parent_checkpoint": str(parent_checkpoint.resolve()) if parent_checkpoint else "",
        "parent_checkpoint_sha256": sha256_path(parent_checkpoint) if parent_checkpoint else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--stage", choices=("parent", "pilot", "full"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = build_config(
        variant=args.variant,
        stage=args.stage,
        seed=args.seed,
        run_root=args.run_root,
        parent_checkpoint=args.parent_checkpoint,
    )
    plan = preflight(config, parent_checkpoint=args.parent_checkpoint)
    if args.validate_only:
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {"5", "6", "7"}:
        raise RuntimeError("formal SR-GAS execution requires one declared physical GPU 5, 6, or 7")
    print(json.dumps(ContinualRunner(config).run(resume_checkpoint=args.resume), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
