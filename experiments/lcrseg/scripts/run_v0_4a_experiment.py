#!/usr/bin/env python3
"""Run one preregistered V0.4a engineering pilot or formal Fundus run."""
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
from lcrseg.engine.checkpoint import load_checkpoint
from lcrseg.engine.continual_runner import ContinualRunner
from lcrseg.methods.lcrseg_v0_3 import FROZEN_FUNDUS_MANIFEST_HASHES, FROZEN_FUNDUS_SPLIT_HASHES


PILOT_RUN_NAMES = {seed: f"fundus_seed{seed}_lcrseg_v0_4a_sra_pilot1000" for seed in (0, 1, 2)}
FULL_RUN_NAMES = {seed: f"fundus_seed{seed}_lcrseg_v0_4a_sra_uniform_full200e" for seed in (0, 1, 2)}
R1_PARENT_RUNS = {
    0: "fundus_seed0_lcrseg_v0_2a_r1_progressive_uniform_full200e",
    1: "fundus_seed1_lcrseg_v0_3_r1_progressive_uniform_full200e",
    2: "fundus_seed2_lcrseg_v0_3_r1_progressive_uniform_full200e",
}
R1_PARENT_SHA256 = {
    0: "9bdadf34a5a32d936b14cfff3f4c9ffa2ee62c5f24142ca12b4a3b9815c46b32",
    1: "c92150508253ac1a468d61d1361bdac91b2722c86810fc2d8a51952571ec1a4f",
    2: "ba5e14b72e23699ed1c573ec2f347fae6065d3c9289a91fcf44947e38100ae8d",
}


def parent_checkpoint(root: Path, seed: int) -> Path:
    return root / "runs" / R1_PARENT_RUNS[seed] / "checkpoint_final_site0_REFUGE.pt"


def validate_parent(path: Path, seed: int) -> dict:
    payload = load_checkpoint(path, map_location="cpu")
    identity = (str(payload["site_id"]), int(payload["site_index"]), int(payload["site_step"]), int(payload["global_step"]))
    if identity != ("REFUGE", 0, 8000, 8000):
        raise ValueError("V0.4a pilot parent must be a complete frozen REFUGE checkpoint")
    config = dict(payload["config_resolved"])
    if int(config["experiment"]["seed"]) != seed:
        raise ValueError("V0.4a pilot parent seed differs from the registered split seed")
    method = dict(config["method"])
    if (method.get("assimilation_mode"), method.get("consolidation_mode")) != (
        "progressive_admission",
        "uniform_relation",
    ):
        raise ValueError("V0.4a pilot parent must have frozen R1 semantics")
    digest = sha256_path(path)
    if digest != R1_PARENT_SHA256[seed]:
        raise ValueError("V0.4a pilot parent SHA-256 differs from preregistration")
    return {"path": str(path), "sha256": digest, "identity": identity}


def build_config(*, mode: str, seed: int, root: Path, run_root: Path, interrupt_after: int | None = None) -> dict:
    config = json.loads((PROJECT_ROOT / "configs/experiments/lcrseg_v0_4a_sra.yaml").read_text(encoding="utf-8"))
    config["experiment"].update(
        {
            "seed": seed,
            "optimization_seed": seed,
            "run_name": PILOT_RUN_NAMES[seed] if mode == "pilot" else FULL_RUN_NAMES[seed],
            "run_root": str(run_root.resolve()),
        }
    )
    if mode == "pilot":
        parent = parent_checkpoint(root, seed)
        validate_parent(parent, seed)
        config["experiment"].update(
            {
                "initial_previous_checkpoint": str(parent),
                "initial_previous_run": str(parent.parent),
                "initial_global_step": 8000,
                "engineering_pilot": True,
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
                "max_steps_this_invocation": interrupt_after,
            }
        )
    return config


def preflight(config: dict, *, mode: str) -> dict:
    runner = ContinualRunner(config)
    total = 0
    sites = []
    for site_id in runner.site_order:
        labeled, unlabeled = runner._datasets((site_id,))
        labeled_batcher = DeterministicBatcher(
            labeled,
            batch_size=int(runner.config["training"]["labeled_batch_size"]),
            seed=runner.optimization_seed,
            namespace=f"v0_4a_preflight:{site_id}:labeled",
            collate=collate_labeled,
        )
        unlabeled_batcher = DeterministicBatcher(
            unlabeled,
            batch_size=int(runner.config["training"]["unlabeled_batch_size"]),
            seed=runner.optimization_seed,
            namespace=f"v0_4a_preflight:{site_id}:unlabeled",
            collate=collate_unlabeled,
        )
        site_total, per_epoch = runner._total_steps(labeled_batcher, unlabeled_batcher, scope=(site_id,))
        total += site_total
        sites.append({"site": site_id, "steps_per_epoch": per_epoch, "total_steps": site_total})
    expected = 1000 if mode == "pilot" else 13_400
    if total != expected:
        raise AssertionError(f"V0.4a {mode} budget resolved to {total}, not {expected}")
    manifest = sha256_path(runner.manifest_path)
    split = sha256_path(runner.split_path)
    if manifest != FROZEN_FUNDUS_MANIFEST_HASHES[runner.seed]:
        raise AssertionError("V0.4a manifest hash differs from the frozen seed manifest")
    if split != FROZEN_FUNDUS_SPLIT_HASHES[runner.seed]:
        raise AssertionError("V0.4a split hash differs from the frozen seed split")
    return {
        "protocol_id": "lcrseg_v0_4a",
        "mode": mode,
        "seed": runner.seed,
        "manifest_sha256": manifest,
        "split_sha256": split,
        "total_steps": total,
        "sites": sites,
        "parent": validate_parent(parent_checkpoint(Path(config["data"]["data_root"]), runner.seed), runner.seed) if mode == "pilot" else None,
    }


def run_pilot(seed: int, root: Path, run_root: Path) -> dict:
    run_dir = run_root / PILOT_RUN_NAMES[seed]
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            raise FileExistsError(f"refusing to rerun completed V0.4a pilot: {run_dir}")
    if not run_dir.exists():
        first = build_config(mode="pilot", seed=seed, root=root, run_root=run_root, interrupt_after=500)
        interrupted = ContinualRunner(first).run()
        if interrupted.get("status") != "interrupted" or int(interrupted.get("completed_global_steps", -1)) != 8500:
            raise RuntimeError("V0.4a pilot did not create the registered 500-step resume boundary")
    checkpoint = run_dir / "checkpoint_last.pt"
    payload = load_checkpoint(checkpoint, map_location="cpu")
    if (int(payload["site_step"]), int(payload["global_step"])) != (500, 8500):
        raise ValueError("V0.4a pilot resume checkpoint is not the registered 500-step boundary")
    resumed = build_config(mode="pilot", seed=seed, root=root, run_root=run_root, interrupt_after=None)
    result = ContinualRunner(resumed).run(resume_checkpoint=checkpoint)
    if result.get("status") != "complete" or int(result.get("completed_global_steps", -1)) != 9000:
        raise RuntimeError("V0.4a pilot resume did not complete exactly 1,000 new steps")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL"))
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = build_config(mode=args.mode, seed=args.seed, root=args.root.resolve(), run_root=args.run_root.resolve())
    plan = preflight(config, mode=args.mode)
    if args.validate_only:
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"5", "6", "7"}:
        raise RuntimeError("V0.4a runs require one declared physical GPU in {5,6,7}")
    if args.mode == "pilot":
        result = run_pilot(args.seed, args.root.resolve(), args.run_root.resolve())
    else:
        run_dir = args.run_root.resolve() / FULL_RUN_NAMES[args.seed]
        if (run_dir / "run_summary.json").is_file():
            existing = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                raise FileExistsError(f"refusing to rerun completed V0.4a full run: {run_dir}")
        result = ContinualRunner(config).run(resume_checkpoint=args.resume)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
