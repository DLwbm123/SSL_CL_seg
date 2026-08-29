#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.common import PROJECT_ROOT, read_csv  # noqa: E402


def main() -> None:
    failures = []
    expected = {"prostate": 116, "mnms": 345, "fundus": 660}
    for name, count in expected.items():
        rows = read_csv(PROJECT_ROOT / "manifests" / f"{name}_cases.csv")
        if len(rows) != count:
            failures.append(f"{name}: expected {count}, got {len(rows)}")
        if len({row['case_id'] for row in rows}) != len(rows):
            failures.append(f"{name}: duplicate case_id")
    if failures:
        raise SystemExit("; ".join(failures))
    print("manifest validation passed")


if __name__ == "__main__":
    main()
