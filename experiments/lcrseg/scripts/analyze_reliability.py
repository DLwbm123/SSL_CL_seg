#!/usr/bin/env python3
"""Run the explicitly separate hidden-GT reliability analysis process."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.diagnostics import analyze_reliability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("LCRSEG_DATA_ROOT", "/home/jiangsuiyang/SSL_CL")))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=("fundus", "prostate", "mnms"), required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output_dir or args.checkpoint.resolve().parent / "analysis"
    summary = analyze_reliability(
        root=args.root,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        site=args.site,
        seed=args.seed,
        output_dir=output,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
