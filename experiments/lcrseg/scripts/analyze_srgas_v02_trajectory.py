#!/usr/bin/env python3
"""Compile the frozen seed-0 V0.2 pilot trajectories without retraining."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


VARIANTS = ("L0", "L1", "L2", "L3", "L4", "D1", "D2")
SITES = {"REFUGE": "refuge_mean_foreground_dice", "RIM_ONE_r3": "rim_one_mean_foreground_dice"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "reports/analysis/srgas_v0_2/pilot_seed0_trajectory.csv")
    args = parser.parse_args()
    source = {}
    for variant in VARIANTS:
        path = args.run_root / f"fundus_seed0_srgas_v02_{variant.lower()}_pilot1000" / "pilot_trajectory.csv"
        with path.open(newline="") as handle:
            source[variant] = list(csv.DictReader(handle))
    baseline = {(int(row["site_step"]), site): float(row[column]) for row in source["L0"] for site, column in SITES.items()}
    output = []
    for variant, rows in source.items():
        for row in rows:
            step = int(row["site_step"])
            phase = "early" if step <= 300 else "middle" if step <= 650 else "late"
            for site, column in SITES.items():
                dice = float(row[column])
                output.append({"seed": 0, "variant": variant, "site": site, "eval_step": step, "phase": phase, "dice": dice, "delta_vs_l0": dice - baseline[(step, site)]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)


if __name__ == "__main__":
    main()
