#!/usr/bin/env python3
"""Execute the final local HDF5 acceptance and freezing gates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.common import DATA_ROOT  # noqa: E402
from lcrseg.finalize import finalize_local_preprocessing  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    result = finalize_local_preprocessing(data_root=args.root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
