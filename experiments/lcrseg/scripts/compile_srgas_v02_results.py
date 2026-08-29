#!/usr/bin/env python3
"""Compile seed-0 endpoint, paired-difference, and classwise V0.2 tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VARIANTS = ("L0", "L1", "L2", "L3", "L4", "D1", "D2")
METRICS = {"Final": "final_average_dice", "BWT": "bwt", "Previous": "previous_site_dice", "Incoming": "incoming_dice"}


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "reports/analysis/srgas_v0_2")
    args = parser.parse_args()
    endpoints, classwise = {}, []
    for variant in VARIANTS:
        run = args.run_root / f"fundus_seed0_srgas_v02_{variant.lower()}_pilot1000"
        summary = json.loads((run / "run_summary.json").read_text())["summary"]
        endpoints[variant] = {name: float(summary[key]) for name, key in METRICS.items()}
        with (run / "site_matrix_long.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                for class_id in (1, 2):
                    classwise.append({"seed": 0, "variant": variant, "trained_site": row["trained_site"], "evaluation_site": row["evaluation_site"], "class_id": class_id, "dice": row[f"dice_class_{class_id}"]})
    _write(args.output_dir / "seedwise_metrics.csv", [{"seed": 0, "variant": v, **endpoints[v]} for v in VARIANTS])
    paired = []
    for comparator in ("L0", "L1", "L2", "L3", "D1", "D2"):
        paired.append({"seed": 0, "left": "L4", "right": comparator, **{f"delta_{metric}": endpoints["L4"][metric] - endpoints[comparator][metric] for metric in METRICS}})
    _write(args.output_dir / "paired_differences.csv", paired)
    _write(args.output_dir / "classwise_metrics.csv", classwise)


if __name__ == "__main__":
    main()
