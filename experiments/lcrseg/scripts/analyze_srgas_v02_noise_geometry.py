#!/usr/bin/env python3
"""Export V0.2 shared-noise and effective perturbation geometry."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


VARIANTS = ("L1", "L2", "L3", "L4", "D1", "D2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "reports/analysis/srgas_v0_2/noise_geometry.csv")
    args = parser.parse_args()
    output = []
    for variant in VARIANTS:
        path = args.run_root / f"fundus_seed0_srgas_v02_{variant.lower()}_pilot1000" / "train_log.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            step = int(row["site_step"])
            if step % 50 and step not in {1, 2, 200, 201}:
                continue
            output.append({key: value for key, value in {
                "seed": 0, "variant": variant, "site_step": step,
                "successful_site_step_before_step": row.get("successful_site_step_before_step"),
                "noise_warmup_multiplier": row.get("noise_warmup_multiplier"),
                "effective_sigma": row.get("effective_noise_sigma"),
                "noise_scale_p10": row.get("noise_scale_p10"), "noise_scale_p50": row.get("noise_scale_p50"), "noise_scale_p90": row.get("noise_scale_p90"),
                "perturbation_weight_ratio": row.get("perturbation_l2_ratio"),
                "classifier_angular_drift": row.get("classifier_angular_drift"),
                "raw_noise_checksum": row.get("raw_noise_checksum"),
            }.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)


if __name__ == "__main__":
    main()
