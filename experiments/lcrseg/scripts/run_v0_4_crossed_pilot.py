#!/usr/bin/env python3
"""Run one preregistered V0.4 diagnostic-only crossed 1,000-step pilot."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.v0_4 import checkpoint_variant
from lcrseg.engine.checkpoint import load_checkpoint
from lcrseg.engine.continual_runner import ContinualRunner


def pilot_run_name(family: str, split_seed: int, optimization_seed: int, variant: str) -> str:
    return (
        f"fundus_v0_4_diagnostic_only_pilot{family}_split{split_seed}"
        f"_opt{optimization_seed}_{variant}_rimone1000"
    )


def _validate_cross(family: str, split_seed: int, optimization_seed: int) -> None:
    if family == "O" and not (split_seed == 0 and optimization_seed in (10, 11, 12)):
        raise ValueError("Pilot O fixes split seed 0 and varies optimization RNG 10/11/12")
    if family == "S" and not (split_seed in (0, 1, 2) and optimization_seed == 20):
        raise ValueError("Pilot S varies split seed 0/1/2 and fixes optimization RNG 20")


def build_config(
    *,
    family: str,
    split_seed: int,
    optimization_seed: int,
    variant: str,
    parent: Path,
    run_root: Path,
) -> dict:
    _validate_cross(family, split_seed, optimization_seed)
    parent = parent.resolve()
    payload = load_checkpoint(parent, map_location="cpu")
    if str(payload["site_id"]) != "REFUGE" or int(payload["site_index"]) != 0:
        raise ValueError("crossed pilot parent must be a frozen REFUGE site-end checkpoint")
    if int(payload["global_step"]) != 8000 or int(payload["site_step"]) != 8000:
        raise ValueError("crossed pilot parent must be the complete 8,000-step REFUGE checkpoint")
    parent_seed = int(dict(payload["config_resolved"])["experiment"]["seed"])
    if parent_seed != split_seed:
        raise ValueError("parent checkpoint seed must match the frozen split seed")
    if checkpoint_variant(payload) != variant:
        raise ValueError("parent checkpoint R0/R1 variant does not match requested pilot")
    base = PROJECT_ROOT / "configs" / "experiments" / f"lcrseg_v0_3_{variant.lower()}.yaml"
    config = json.loads(base.read_text(encoding="utf-8"))
    config["experiment"].update(
        {
            "run_name": pilot_run_name(family, split_seed, optimization_seed, variant),
            "run_root": str(run_root.resolve()),
            "seed": split_seed,
            "optimization_seed": optimization_seed,
            "initial_previous_checkpoint": str(parent),
            "initial_previous_run": str(parent.parent),
            "initial_global_step": 8000,
            "diagnostic_only": True,
            "diagnostic_family": family,
            "formal_method_result": False,
        }
    )
    config["data"].update({"site_order": ["RIM_ONE_r3"], "site_index_offset": 1})
    config["training"].update(
        {
            "steps_per_site": 1000,
            "checkpoint_interval_steps": 250,
            "preserve_interval_checkpoints": True,
            "gradient_cosine_interval": 50,
            "max_steps_this_invocation": None,
        }
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("O", "S"), required=True)
    parser.add_argument("--split-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--optimization-seed", type=int, required=True)
    parser.add_argument("--variant", choices=("R0", "R1"), required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = build_config(
        family=args.family,
        split_seed=args.split_seed,
        optimization_seed=args.optimization_seed,
        variant=args.variant,
        parent=args.parent,
        run_root=args.run_root,
    )
    runner = ContinualRunner(config)
    if args.validate_only:
        print(json.dumps(runner.config, ensure_ascii=False, sort_keys=True))
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {"5", "6", "7"}:
        raise RuntimeError("V0.4 crossed pilots require physical CUDA_VISIBLE_DEVICES in {5,6,7}")
    print(json.dumps(runner.run(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
