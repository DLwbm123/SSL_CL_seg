#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.preprocess import preprocess_fundus  # noqa: E402
from preprocess_common import add_common_options, options_from_args  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate-protected three-site fundus HDF5 preprocessing.")
    add_common_options(parser)
    args = parser.parse_args()
    try:
        result = preprocess_fundus(options_from_args(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
