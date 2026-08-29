"""Frozen-checkpoint aggregation and patient-level statistics for V0.3."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


METRICS = ("final_average_dice", "bwt", "incoming_dice", "previous_site_dice")


def aggregate_paired_seed_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_seed[int(row["seed"])][str(row["variant"]).upper()] = row
    paired = []
    for seed in sorted(by_seed):
        if set(by_seed[seed]) != {"R0", "R1"}:
            raise ValueError(f"seed {seed} does not contain exactly one R0/R1 pair")
        record: dict[str, Any] = {"seed": seed}
        for metric in METRICS:
            record[f"delta_{metric}"] = float(by_seed[seed]["R1"][metric]) - float(by_seed[seed]["R0"][metric])
        paired.append(record)
    summary: dict[str, Any] = {"seeds": [row["seed"] for row in paired], "metrics": {}}
    for metric in METRICS:
        values = np.asarray([row[f"delta_{metric}"] for row in paired], dtype=np.float64)
        summary["metrics"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "positive_direction_count": int((values > 0).sum()),
            "nonnegative_direction_count": int((values >= 0).sum()),
        }
    return {"paired": paired, "summary": summary}


def paired_patient_bootstrap(
    first: Mapping[str, float],
    second: Mapping[str, float],
    *,
    samples: int = 10_000,
    seed: int = 20260827,
) -> dict[str, Any]:
    """Bootstrap a paired mean difference with patients as sampling units."""

    patients = sorted(set(first).intersection(second))
    if not patients:
        raise ValueError("paired patient bootstrap requires common patients")
    if set(first) != set(second):
        raise ValueError("paired patient bootstrap requires identical patient sets")
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    differences = np.asarray([float(first[patient]) - float(second[patient]) for patient in patients], dtype=np.float64)
    if not np.isfinite(differences).all():
        raise ValueError("patient metric contains NaN/Inf")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(patients), size=(samples, len(patients)))
    boot = differences[indices].mean(axis=1)
    return {
        "sampling_unit": "patient",
        "paired": True,
        "patients": len(patients),
        "bootstrap_samples": samples,
        "mean_difference": float(differences.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "seed": seed,
    }

