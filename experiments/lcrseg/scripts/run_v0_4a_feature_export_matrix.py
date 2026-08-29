#!/usr/bin/env python3
"""Run one deterministic worker of the frozen V0.4a SRA feature-export matrix."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.v0_4 import SITE_ORDER
from lcrseg.common import sha256_path, write_json


def _jobs(root: Path, output_dir: Path) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for seed in (0, 1, 2):
        run_name = f"fundus_seed{seed}_lcrseg_v0_4a_sra_uniform_full200e"
        for site_index, trained_site in enumerate(SITE_ORDER):
            checkpoint = root / "runs" / run_name / f"checkpoint_final_site{site_index}_{trained_site}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            for evaluation_site in SITE_ORDER:
                stem = (
                    f"seed{seed}_SRA_through{site_index}-{trained_site}"
                    f"_eval-{evaluation_site}_max200000"
                )
                jobs.append(
                    {
                        "seed": seed,
                        "trained_site": trained_site,
                        "trained_site_index": site_index,
                        "evaluation_site": evaluation_site,
                        "checkpoint": checkpoint,
                        "output": output_dir / f"{stem}.npz",
                    }
                )
    return jobs


def _is_complete(job: dict[str, object]) -> bool:
    output = Path(job["output"])
    companion = output.with_suffix(".json")
    if not output.is_file() or not companion.is_file():
        return False
    metadata = json.loads(companion.read_text(encoding="utf-8"))
    return bool(
        metadata.get("status") == "complete"
        and metadata.get("hidden_gt_usage") == "post_hoc_only"
        and metadata.get("variant") == "SRA"
        and metadata.get("output_sha256") == sha256_path(output)
        and metadata.get("checkpoint_sha256") == sha256_path(Path(job["checkpoint"]))
        and int(metadata.get("max_pixels_per_class", -1)) == 200_000
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.workers != 3 or not 0 <= args.worker_index < args.workers:
        raise ValueError("V0.4a feature export requires exactly three workers")
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = _jobs(root, output_dir)
    assigned = [job for index, job in enumerate(jobs) if index % args.workers == args.worker_index]
    completed: list[dict[str, object]] = []
    for ordinal, job in enumerate(assigned, start=1):
        output = Path(job["output"])
        if _is_complete(job):
            print(f"SKIP complete {ordinal}/{len(assigned)} {output.name}", flush=True)
        else:
            if output.exists() or output.with_suffix(".json").exists():
                raise RuntimeError(f"incomplete existing shard requires audit, refusing overwrite: {output}")
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "export_v0_4_diagnostic_features.py"),
                "--root", str(root),
                "--checkpoint", str(job["checkpoint"]),
                "--site", str(job["evaluation_site"]),
                "--seed", str(job["seed"]),
                "--max-pixels-per-class", "200000",
                "--output", str(output),
                "--device", args.device,
            ]
            print(f"RUN {ordinal}/{len(assigned)} {output.name}", flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        completed.append(
            {
                "output": str(output),
                "output_sha256": metadata["output_sha256"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
            }
        )
    report = {
        "protocol_id": "lcrseg_v0_4a",
        "status": "complete",
        "worker_index": args.worker_index,
        "workers": args.workers,
        "assigned_shards": len(assigned),
        "completed_shards": completed,
        "hidden_gt_usage": "post_hoc_only",
    }
    write_json(output_dir / f"worker_{args.worker_index}_complete.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
