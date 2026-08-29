#!/usr/bin/env python3
"""Run one registered SR-GAS V0.2 parent, pilot, or full continuation."""
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


PARENT_SEED0_SHA256 = "8f188ba27074ecb09a689377982774e6cf59e8c1c652d3927be54fd7c377bf55"
CONFIGS = {
    "L0": "srgas_v0_2_l0_cosine.yaml",
    "L1": "srgas_v0_2_l1_isotropic_warm.yaml",
    "L2": "srgas_v0_2_l2_lag_totalgas_warm.yaml",
    "L3": "srgas_v0_2_l3_lag_supgas_warm.yaml",
    "L4": "srgas_v0_2_l4_lag_srgas_warm.yaml",
    "D1": "srgas_v0_2_d1_same_srgas_warm.yaml",
    "D2": "srgas_v0_2_d2_lag_srgas_nowarm.yaml",
}
FULL_STEMS = {
    "L0": "l0_cosine",
    "L1": "l1_isotropic_warm",
    "L2": "l2_lag_totalgas_warm",
    "L3": "l3_lag_supgas_warm",
    "L4": "l4_lag_srgas_warm",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def build_config(
    *,
    variant: str,
    stage: str,
    seed: int,
    run_root: Path,
    parent_checkpoint: Path | None,
) -> dict:
    base = _load(PROJECT_ROOT / "configs/experiments/srgas_a1_cosine.yaml")
    fragment = _load(PROJECT_ROOT / "configs/experiments" / CONFIGS[variant])
    base["method"].update(fragment["method"])
    base["method"].update({"split_seed": seed, "noise_seed": seed})
    base["experiment"].update(
        {"seed": seed, "optimization_seed": seed, "run_root": str(run_root.resolve())}
    )
    if stage == "parent":
        if variant != "L0" or seed not in {1, 2} or parent_checkpoint is not None:
            raise ValueError("V0.2 parents are registered only as seed-1/2 L0 REFUGE runs")
        base["experiment"]["run_name"] = f"fundus_seed{seed}_srgas_v02_l0_cosine_site1"
        base["data"]["site_order"] = ["REFUGE"]
        return base
    if parent_checkpoint is None or not parent_checkpoint.is_file():
        raise FileNotFoundError("pilot/full continuation requires a completed common parent checkpoint")
    payload = load_checkpoint(parent_checkpoint, map_location="cpu")
    if payload["site_id"] != "REFUGE" or int(payload["site_index"]) != 0:
        raise ValueError("parent checkpoint is not a registered REFUGE site-0 artifact")
    if payload["method_name"] not in {"srgas_v0_1", "srgas_v0_2"}:
        raise ValueError("parent checkpoint method is not compatible with the architecture-identical SR-GAS line")
    if int(payload["site_step"]) < 1:
        raise ValueError("parent checkpoint has no completed optimizer steps")
    if seed == 0 and sha256_path(parent_checkpoint) != PARENT_SEED0_SHA256:
        raise ValueError("seed-0 parent SHA256 differs from the frozen V0.1a common parent")
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
        registered = {0: set(CONFIGS), 1: {"L0", "L3", "L4"}}
        if variant not in registered.get(seed, set()):
            raise ValueError("variant/seed pair is not in the preregistered pilot matrix")
        base["experiment"]["run_name"] = f"fundus_seed{seed}_srgas_v02_{variant.lower()}_pilot1000"
        base["data"]["site_order"] = ["RIM_ONE_r3"]
        base["training"]["steps_per_site"] = 1000
        base["training"]["checkpoint_interval_steps"] = 250
        base["training"]["preserve_interval_checkpoints"] = True
        base["training"]["pilot_trajectory_interval"] = 50
        return base
    if stage != "full" or variant not in FULL_STEMS:
        raise ValueError("full stage accepts only L0-L4")
    registered_full = {0: set(FULL_STEMS), 1: {"L0", "L2", "L3", "L4"}, 2: {"L0", "L2", "L3", "L4"}}
    if variant not in registered_full[seed]:
        raise ValueError("variant/seed pair is not in the preregistered full matrix")
    base["experiment"]["run_name"] = f"fundus_seed{seed}_srgas_v02_{FULL_STEMS[variant]}_full200e"
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
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"5", "6", "7"}:
        raise RuntimeError("formal SR-GAS V0.2 execution requires declared physical GPU 5, 6, or 7")
    print(json.dumps(ContinualRunner(config).run(resume_checkpoint=args.resume), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
