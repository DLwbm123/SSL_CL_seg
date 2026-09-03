"""Pure accounting helpers for proposal-level review tests."""
from __future__ import annotations


def proposal_precision(accepted, beneficial):
    denominator = int(sum(bool(value) for value in accepted))
    numerator = int(sum(bool(take) and bool(good) for take, good in zip(accepted, beneficial)))
    return {
        "proposal_precision": None if denominator == 0 else numerator / denominator,
        "proposal_precision_numerator": numerator,
        "proposal_precision_denominator": denominator,
    }


def aligned_global(rows, count):
    ordered = sorted(rows, key=lambda row: row["row_index"])
    if [row["row_index"] for row in ordered] != list(range(count)):
        raise ValueError("global row_index alignment failed")
    return ordered
