#!/usr/bin/env python3
"""Validate the audit-created case manifests without reading source pixels again."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.common import PROJECT_ROOT, read_csv, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", choices=("prostate", "mnms", "fundus", "all"), default="all")
    args = parser.parse_args()
    names = ("prostate", "mnms", "fundus") if args.manifest == "all" else (args.manifest,)
    summary = {}
    for name in names:
        rows = read_csv(PROJECT_ROOT / "manifests" / f"{name}_cases.csv")
        ids = [row["case_id"] for row in rows]
        missing = [row["case_id"] for row in rows if not row.get("image_path_raw") or not row.get("label_path_raw")]
        summary[name] = {"records": len(rows), "duplicate_case_ids": len(ids) - len(set(ids)), "missing_pair_paths": missing}
    write_json(PROJECT_ROOT / "reports" / "data_audit" / "manifest_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
