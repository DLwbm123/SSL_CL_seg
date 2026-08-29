#!/usr/bin/env python3
"""Export available V0.2 sensitivity diagnostics; mark unlogged metrics unavailable."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


VARIANTS = ("L1", "L2", "L3", "L4", "D1", "D2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "reports/analysis/srgas_v0_2/sensitivity_temporal_stability.csv")
    args = parser.parse_args()
    output = []
    for variant in VARIANTS:
        path = args.run_root / f"fundus_seed0_srgas_v02_{variant.lower()}_pilot1000" / "train_log.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            step = int(row["site_step"])
            if step % 50:
                continue
            sup_mean = float(row.get("s_sup_mean") or 0.0)
            sup_std = float(row.get("s_sup_std") or 0.0)
            r2c_mean = float(row.get("s_r2c_mean") or 0.0)
            r2c_std = float(row.get("s_r2c_std") or 0.0)
            output.append({
                "seed": 0,
                "variant": variant,
                "site_step": step,
                "sensitivity_timing": "same_step" if variant == "D1" else "lagged" if variant in {"L2", "L3", "L4", "D2"} else "isotropic",
                "cos_s_t_s_t_minus_1": "unavailable",
                "temporal_cosine_reason": "full sensitivity tensors were not persisted at the 50-step evaluation grid",
                "lagged_l1_to_current": row.get("lagged_sensitivity_l1_to_current") or "unavailable",
                "s_sup_cv_within_step": sup_std / sup_mean if sup_mean > 0 else "unavailable",
                "s_r2c_cv_within_step": r2c_std / r2c_mean if r2c_mean > 0 else "unavailable",
                "s_sup_s_r2c_cosine": row.get("sensitivity_cosine") or "unavailable",
                "sensitivity_p10": row.get("sensitivity_p10") or "unavailable",
                "sensitivity_p50": row.get("sensitivity_p50") or "unavailable",
                "sensitivity_p90": row.get("sensitivity_p90") or "unavailable",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)


if __name__ == "__main__":
    main()
