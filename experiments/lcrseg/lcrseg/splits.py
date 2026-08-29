from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import PROJECT_ROOT, read_csv, utc_now, write_json


MANIFEST_ROOT = PROJECT_ROOT / "manifests"
SPLIT_ROOT = PROJECT_ROOT / "splits"

PROSTATE_COUNTS = {
    "RUNMC": (18, 4, 8),
    "BMC": (18, 4, 8),
    "I2CVB": (11, 3, 5),
    "UCL": (8, 2, 3),
    "BIDMC": (7, 2, 3),
    "HK": (7, 2, 3),
}
PROSTATE_LABELED = {"RUNMC": (2, 4, 7), "BMC": (2, 4, 7), "I2CVB": (1, 2, 4), "UCL": (1, 2, 3), "BIDMC": (1, 2, 3), "HK": (1, 2, 3)}
FUNDUS_COUNTS = {"REFUGE": (200, 100, 100), "RIM_ONE_r3": (79, 40, 40), "Drishti_GS": (51, 25, 25)}
FUNDUS_LABELED = {"REFUGE": (20, 40, 80), "RIM_ONE_r3": (8, 16, 32), "Drishti_GS": (5, 10, 20)}


def _shuffle(rows: list[dict[str, str]], seed: int, group: str) -> list[dict[str, str]]:
    output = list(sorted(rows, key=lambda row: row["case_id"]))
    random.Random(f"lcrseg-v1:{seed}:{group}").shuffle(output)
    return output


def _make_records(
    grouped: dict[str, list[dict[str, str]]],
    counts: dict[str, tuple[int, int, int]],
    labeled: dict[str, tuple[int, int, int]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group, members in sorted(grouped.items()):
        if group not in counts:
            raise ValueError(f"Unexpected group {group!r}")
        train_count, val_count, test_count = counts[group]
        if len(members) != train_count + val_count + test_count:
            raise ValueError(f"{group}: expected {train_count + val_count + test_count}, got {len(members)}")
        ordered = _shuffle(members, seed, group)
        l10, l20, l40 = labeled[group]
        if not (l10 <= l20 <= l40 <= train_count):
            raise ValueError(f"invalid nested label counts for {group}")
        for index, base in enumerate(ordered):
            record = {
                "case_id": base["case_id"],
                "patient_id": base["patient_id"],
                "dataset": base["dataset"],
                "site_or_vendor": group,
                "split": "test" if index >= train_count + val_count else ("val" if index >= train_count else "train"),
                "labelled_at_10pct": index < l10,
                "labelled_at_20pct": index < l20,
                "labelled_at_40pct": index < l40,
                "source_partition": base.get("source_partition", ""),
            }
            record["primary_20pct_split"] = (
                "train_labeled" if record["split"] == "train" and record["labelled_at_20pct"] else
                "train_unlabeled" if record["split"] == "train" else record["split"]
            )
            out.append(record)
    return out


def _payload(protocol: str, seed: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in records:
        counts[row["site_or_vendor"]][row["primary_20pct_split"]] += 1
    ids = [row["patient_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("patient overlap or duplicate patient ID")
    return {
        "schema_version": 1,
        "protocol": protocol,
        "seed": seed,
        "generated_at": utc_now(),
        "records": sorted(records, key=lambda row: row["case_id"]),
        "primary_20pct_counts": {group: dict(value) for group, value in sorted(counts.items())},
        "patient_overlap_check": "passed",
    }


def generate_prostate(seed: int) -> dict[str, Any]:
    rows = read_csv(MANIFEST_ROOT / "prostate_cases.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["site"]].append(row)
    records = _make_records(grouped, PROSTATE_COUNTS, PROSTATE_LABELED, seed=seed)
    payload = _payload("prostate_six_site_60_15_25", seed, records)
    write_json(SPLIT_ROOT / f"prostate_seed{seed}.json", payload)
    return payload


def generate_fundus(seed: int) -> dict[str, Any]:
    rows = read_csv(MANIFEST_ROOT / "fundus_cases.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["site"]].append(row)
    records = _make_records(grouped, FUNDUS_COUNTS, FUNDUS_LABELED, seed=seed)
    payload = _payload("fundus_three_site_50_25_25", seed, records)
    write_json(SPLIT_ROOT / f"fundus_seed{seed}.json", payload)
    return payload


MNMS_CANONICAL_COUNTS = {
    "Siemens": (48, 23, 24),
    "Philips": (63, 31, 31),
    "GE": (25, 12, 13),
    "Canon": (25, 12, 13),
}


def _mnms_labeled_counts(train_count: int) -> tuple[int, int, int]:
    values = (
        max(5, round(train_count * 0.1)),
        max(5, round(train_count * 0.2)),
        max(5, round(train_count * 0.4)),
    )
    return tuple(min(train_count, value) for value in values)  # type: ignore[return-value]


def generate_mnms(seed: int) -> dict[str, Any]:
    """Freeze canonical320 evaluation splits and append auxiliary25 as GE unlabeled.

    The 25 official ``Training/Unlabeled`` patients are intentionally excluded
    from evaluation and labelled training.  Their images are retained in the
    ``train_unlabeled`` partition, while their diagnostic labels remain outside
    training manifests.
    """
    rows = read_csv(MANIFEST_ROOT / "mnms_cases.csv")
    unknown = [row["case_id"] for row in rows if row.get("vendor") == "UNKNOWN"]
    if unknown:
        raise RuntimeError(f"BLOCKER: {len(unknown)} M&Ms cases have no vendor mapping")

    canonical = [row for row in rows if row.get("source_partition") != "Training/Unlabeled"]
    auxiliary = [row for row in rows if row.get("source_partition") == "Training/Unlabeled"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in canonical:
        grouped[row["vendor"]].append(row)
    observed_counts = {vendor: len(members) for vendor, members in grouped.items()}
    expected_counts = {vendor: sum(parts) for vendor, parts in MNMS_CANONICAL_COUNTS.items()}
    if observed_counts != expected_counts:
        raise RuntimeError(f"unexpected canonical320 vendor counts: {observed_counts}")
    if len(auxiliary) != 25 or any(row.get("vendor") != "GE" for row in auxiliary):
        raise RuntimeError("unexpected auxiliary25 M&Ms cohort; expected 25 GE Training/Unlabeled patients")

    labels = {vendor: _mnms_labeled_counts(parts[0]) for vendor, parts in MNMS_CANONICAL_COUNTS.items()}
    records = _make_records(grouped, MNMS_CANONICAL_COUNTS, labels, seed=seed)
    for record in records:
        record.update(cohort="canonical320", training_role="standard", evaluation_eligible=True)
    for base in _shuffle(auxiliary, seed, "GE_auxiliary25"):
        records.append(
            {
                "case_id": base["case_id"],
                "patient_id": base["patient_id"],
                "dataset": base["dataset"],
                "site_or_vendor": "GE",
                "split": "train",
                "labelled_at_10pct": False,
                "labelled_at_20pct": False,
                "labelled_at_40pct": False,
                "primary_20pct_split": "train_unlabeled",
                "source_partition": base["source_partition"],
                "cohort": "auxiliary25",
                "training_role": "train_unlabeled_only",
                "evaluation_eligible": False,
            }
        )
    payload = _payload("mnms_canonical320_vendor_stratified_plus_auxiliary25", seed, records)
    payload["cohort_counts"] = {"canonical320": len(canonical), "auxiliary25": len(auxiliary)}
    payload["auxiliary25_policy"] = "GE Training/Unlabeled; images only in train_unlabeled; diagnostics labels only"
    write_json(SPLIT_ROOT / f"mnms_seed{seed}.json", payload)
    return payload
