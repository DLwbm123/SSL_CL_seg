#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.acceptance import build_transfer_manifest  # noqa: E402
from lcrseg.common import DATA_ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the data-only transfer manifest and SHA-256 checksum list.")
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_transfer_manifest(args.root), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
