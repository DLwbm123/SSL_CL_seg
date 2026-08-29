#!/usr/bin/env python3
"""Run one GPU worker of the fixed 12-run V0.4 crossed diagnostic pilot queue."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.v0_4 import RUN_NAMES
from lcrseg.common import sha256_path, write_json
from scripts.run_v0_4_crossed_pilot import pilot_run_name


def registered_jobs(root: Path) -> list[dict[str, object]]:
    crosses = [("O", 0, rng) for rng in (10, 11, 12)] + [("S", seed, 20) for seed in (0, 1, 2)]
    jobs: list[dict[str, object]] = []
    for family, split_seed, optimization_seed in crosses:
        for variant in ("R0", "R1"):
            run_name = RUN_NAMES[(split_seed, variant)]
            parent = root / "runs" / run_name / "checkpoint_final_site0_REFUGE.pt"
            if not parent.is_file():
                raise FileNotFoundError(parent)
            jobs.append(
                {
                    "family": family,
                    "split_seed": split_seed,
                    "optimization_seed": optimization_seed,
                    "variant": variant,
                    "parent": parent,
                    "run_name": pilot_run_name(family, split_seed, optimization_seed, variant),
                }
            )
    return jobs


def _complete(run_dir: Path) -> bool:
    summary_path = run_dir / "run_summary.json"
    final_path = run_dir / "checkpoint_final_site1_RIM_ONE_r3.pt"
    if not summary_path.is_file() or not final_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return bool(
        summary.get("status") == "complete"
        and int(summary.get("new_optimizer_steps", -1)) == 1000
        and int(summary.get("completed_global_steps", -1)) == 9000
        and int(summary.get("optimization_seed", -1)) in {10, 11, 12, 20}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL"))
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers != 3 or not 0 <= args.worker_index < args.workers:
        raise ValueError("the preregistered queue uses exactly three GPU workers")
    root = args.root.resolve()
    jobs = registered_jobs(root)
    assigned = [job for index, job in enumerate(jobs) if index % args.workers == args.worker_index]
    completed: list[dict[str, object]] = []
    for ordinal, job in enumerate(assigned, start=1):
        run_dir = root / "runs" / str(job["run_name"])
        if _complete(run_dir):
            print(f"SKIP complete {ordinal}/{len(assigned)} {job['run_name']}", flush=True)
        else:
            if run_dir.exists():
                raise RuntimeError(f"incomplete pilot exists; diagnose before resume: {run_dir}")
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_v0_4_crossed_pilot.py"),
                "--family",
                str(job["family"]),
                "--split-seed",
                str(job["split_seed"]),
                "--optimization-seed",
                str(job["optimization_seed"]),
                "--variant",
                str(job["variant"]),
                "--parent",
                str(job["parent"]),
                "--run-root",
                str(root / "runs"),
            ]
            print(f"RUN {ordinal}/{len(assigned)} {job['run_name']}", flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        summary_path = run_dir / "run_summary.json"
        final_path = run_dir / "checkpoint_final_site1_RIM_ONE_r3.pt"
        if not _complete(run_dir):
            raise RuntimeError(f"pilot did not satisfy completion contract: {run_dir}")
        completed.append(
            {
                **job,
                "parent": str(job["parent"]),
                "run_dir": str(run_dir),
                "run_summary_sha256": sha256_path(summary_path),
                "final_checkpoint_sha256": sha256_path(final_path),
            }
        )
    report = {
        "protocol_id": "lcrseg_v0_4_failure_audit",
        "status": "complete",
        "diagnostic_only": True,
        "worker_index": args.worker_index,
        "workers": args.workers,
        "completed": completed,
    }
    output = root / "runs" / "v0_4_failure_audit" / "crossed_pilots" / f"worker_{args.worker_index}_complete.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
