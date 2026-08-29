#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.splits import generate_fundus, generate_mnms, generate_prostate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("prostate", "fundus", "mnms", "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    try:
        for seed in args.seeds:
            if args.dataset in {"prostate", "all"}:
                print("prostate", seed, generate_prostate(seed)["primary_20pct_counts"])
            if args.dataset in {"fundus", "all"}:
                print("fundus", seed, generate_fundus(seed)["primary_20pct_counts"])
            if args.dataset in {"mnms", "all"}:
                print("mnms", seed, generate_mnms(seed)["primary_20pct_counts"])
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
