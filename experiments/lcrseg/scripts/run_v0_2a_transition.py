#!/usr/bin/env python3
"""Run an immutable V0.2a bridge or engineering pilot transition."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.engine.checkpoint import load_checkpoint
from lcrseg.engine.continual_runner import ContinualRunner


KINDS = {
    "bridge_legacy": ("bridge_legacy_v01_rimone_500steps", 500, "legacy"),
    "bridge_amended": ("bridge_v02a_r0_rimone_500steps", 500, "R0"),
    "pilot_r1": ("pilot_v02a_r1_rimone_1000steps", 1000, "R1"),
    "pilot_r2": ("pilot_v02a_r2_rimone_1000steps", 1000, "R2"),
    "pilot_r3": ("pilot_v02a_r3_rimone_1000steps", 1000, "R3"),
}


def _assert_gpu4() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise RuntimeError("bridge and pilots require CUDA_VISIBLE_DEVICES=4")


def _config(kind: str, *, parent: Path, run_root: Path) -> dict:
    run_name, steps, variant = KINDS[kind]
    parent_payload = load_checkpoint(parent, map_location="cpu")
    if parent_payload["site_id"] != "REFUGE" or int(parent_payload["site_index"]) != 0:
        raise ValueError("transition parent must be the corresponding REFUGE site-end checkpoint")
    if int(parent_payload["global_step"]) != 8000:
        raise ValueError("the REFUGE parent must end at global step 8000")
    base_path = PROJECT_ROOT / "configs/experiments" / (
        "lcrseg_v0_2a_r0.yaml" if variant == "legacy" else f"lcrseg_v0_2a_{variant.lower()}.yaml"
    )
    config = json.loads(base_path.read_text())
    config["experiment"].update(
        {
            "run_name": run_name,
            "run_root": str(run_root),
            "initial_previous_checkpoint": str(parent),
            "initial_global_step": 8000,
        }
    )
    config["data"].update({"site_order": ["RIM_ONE_r3"], "site_index_offset": 1})
    config["training"].update(
        {
            "steps_per_site": steps,
            "checkpoint_interval_steps": 50 if kind.startswith("bridge_") else 500,
            "preserve_interval_checkpoints": kind.startswith("bridge_"),
            "gradient_cosine_interval": 50,
        }
    )
    if variant == "legacy":
        config["method"] = {
            "name": "lcrseg_v0_1",
            "version": "0.1",
            "use_learnability": True,
            "use_compatibility": False,
        }
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(KINDS), required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--run-name")
    parser.add_argument("--interrupt-after", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = _config(args.kind, parent=args.parent.resolve(), run_root=args.run_root.resolve())
    if args.run_name:
        config["experiment"]["run_name"] = args.run_name
    if args.interrupt_after is not None:
        if args.interrupt_after < 1:
            raise ValueError("--interrupt-after must be positive")
        if args.resume is not None:
            raise ValueError("--interrupt-after cannot be combined with --resume")
        config["training"]["max_steps_this_invocation"] = args.interrupt_after
    elif args.resume is not None:
        config["training"]["max_steps_this_invocation"] = None
    runner = ContinualRunner(config)
    if args.validate_only:
        print(json.dumps(runner.config, ensure_ascii=False, sort_keys=True))
        return
    _assert_gpu4()
    resume_checkpoint = args.resume.resolve() if args.resume is not None else None
    print(json.dumps(runner.run(resume_checkpoint=resume_checkpoint), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
