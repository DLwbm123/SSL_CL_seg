#!/usr/bin/env python3
"""Summarize measured V0.2 pilot training cost."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median


VARIANTS = ("L0", "L1", "L2", "L3", "L4", "D1", "D2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "reports/analysis/srgas_v0_2/cost_analysis.csv")
    args = parser.parse_args()
    output = []
    for variant in VARIANTS:
        run = args.run_root / f"fundus_seed0_srgas_v02_{variant.lower()}_pilot1000"
        with (run / "train_log.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        seconds = [float(row["training_step_seconds"]) for row in rows]
        output.append({
            "seed": 0, "variant": variant, "steps": len(rows),
            "median_step_seconds": median(seconds), "summed_step_seconds": sum(seconds),
            "peak_vram_bytes": max(float(row["peak_memory_bytes"]) for row in rows),
            "checkpoint_bytes": sum(path.stat().st_size for path in run.glob("*.pt")),
            "inference_overhead": 0,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)


if __name__ == "__main__":
    main()
