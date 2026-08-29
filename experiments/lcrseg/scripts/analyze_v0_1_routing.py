#!/usr/bin/env python3
"""Run the preregistered, post-hoc LCR-Seg V0.1 routing diagnostic."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.v0_1_routing import DEFAULT_RUNS, analyze_v0_1_routing


def _run_names(value: str | None) -> list[str]:
    if value is None:
        return list(DEFAULT_RUNS)
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("--runs must contain at least one run directory")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("LCRSEG_DATA_ROOT", "/home/jiangsuiyang/SSL_CL")))
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("LCRSEG_RUN_ROOT", "/home/jiangsuiyang/SSL_CL/runs")))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("fundus",), default="fundus")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", help="comma-separated run directory names; defaults to all four preregistered runs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = analyze_v0_1_routing(
        root=args.root,
        run_root=args.run_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        seed=args.seed,
        run_names=_run_names(args.runs),
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
