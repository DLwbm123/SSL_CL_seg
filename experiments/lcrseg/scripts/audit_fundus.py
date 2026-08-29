#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.audit import audit_fundus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only three-site fundus audit.")
    parser.add_argument("--qc-per-dataset", type=int, default=20)
    args = parser.parse_args()
    print(audit_fundus(qc_per_dataset=args.qc_per_dataset))


if __name__ == "__main__":
    main()
