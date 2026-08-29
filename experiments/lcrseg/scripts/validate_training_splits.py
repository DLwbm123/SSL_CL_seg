#!/usr/bin/env python3
"""Read-only all-seed training-manifest split and hidden-label gate."""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _role_group(role: str) -> str:
    if role in {"train_labeled", "train_unlabeled"}:
        return "train"
    if role in {"val", "test"}:
        return role
    raise ValueError(f"unexpected training-manifest role: {role}")


def validate(root: Path, seeds: list[int]) -> dict[str, Any]:
    errors: list[str] = []
    reports: list[dict[str, Any]] = []
    for seed in seeds:
        manifest = root / "manifests" / "training" / f"lcrseg_v1_seed{seed}.csv"
        rows = list(csv.DictReader(manifest.open()))
        patient_roles: dict[tuple[str, str], set[str]] = {}
        auxiliary_rows = 0
        duplicate_case_ids: set[str] = set()
        seen_case_ids: set[str] = set()
        for row in rows:
            case_id = row.get("case_id", "")
            if case_id in seen_case_ids:
                duplicate_case_ids.add(case_id)
            seen_case_ids.add(case_id)
            dataset = row.get("dataset", "")
            patient_id = row.get("patient_id", "") or case_id
            role = row.get("primary_20pct_split", "")
            try:
                grouped = _role_group(role)
            except ValueError as exc:
                errors.append(f"seed {seed} {case_id}: {exc}")
                continue
            patient_roles.setdefault((dataset, patient_id), set()).add(grouped)
            if role == "train_unlabeled" and row.get("label_h5_relpath", ""):
                errors.append(f"seed {seed} {case_id}: hidden label path exposed")
            if dataset == "mnms" and row.get("cohort") == "auxiliary25":
                auxiliary_rows += 1
                if role != "train_unlabeled" or row.get("evaluation_eligible", "").lower() != "false":
                    errors.append(f"seed {seed} {case_id}: auxiliary25 violates image-only train-unlabeled contract")
        for (dataset, patient_id), roles in patient_roles.items():
            if len(roles) != 1:
                errors.append(f"seed {seed} {dataset} patient {patient_id}: cross-split roles {sorted(roles)}")
        if duplicate_case_ids:
            errors.append(f"seed {seed}: duplicate case IDs {sorted(duplicate_case_ids)[:5]}")
        reports.append({"seed": seed, "rows": len(rows), "patients": len(patient_roles), "auxiliary25_rows": auxiliary_rows, "duplicate_case_ids": len(duplicate_case_ids)})
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "root": str(root), "seeds": reports, "errors": errors, "valid": not errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("LCRSEG_DATA_ROOT", "/home/jiangsuiyang/SSL_CL")))
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("LCRSEG_RUN_ROOT", "/home/jiangsuiyang/SSL_CL/runs")))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = validate(args.root.resolve(), args.seeds)
    output = args.output or args.run_root.resolve() / "m0" / "training_split_validation_all_seeds.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite split validation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
