#!/usr/bin/env python3
"""Verify the frozen, transfer-scoped SHA-256 checksum list."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.acceptance import verify_checksums  # noqa: E402
from lcrseg.common import DATA_ROOT, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    result = verify_checksums(args.root)
    write_json(args.root / "reports" / "validation" / "checksum_validation.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
