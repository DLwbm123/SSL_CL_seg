#!/usr/bin/env python3
"""Create or verify a fixed LCR-Seg V0.1 golden batch outside frozen data."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.golden import golden_payload, write_or_verify_golden


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("LCRSEG_DATA_ROOT", "/home/jiangsuiyang/SSL_CL")))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=("fundus", "prostate", "mnms"), required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--atol", type=float, default=1.0e-4)
    args = parser.parse_args()
    output = args.output_dir or args.checkpoint.resolve().parent / "golden"
    losses, arrays, metadata = golden_payload(
        root=args.root,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        site=args.site,
        seed=args.seed,
        device=args.device,
    )
    result = write_or_verify_golden(
        output_dir=output,
        losses=losses,
        arrays=arrays,
        metadata=metadata,
        verify=args.verify,
        atol=args.atol,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
