#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.acceptance import validate_h5_tree  # noqa: E402
from lcrseg.common import DATA_ROOT, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate immutable LCR-Seg HDF5 files and image/label pairs.")
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    result = validate_h5_tree(args.root)
    write_json(args.root / "reports" / "validation" / "h5_validation.json", result)
    print(json.dumps({key: result[key] for key in ("valid", "h5_files", "complete_pairs", "errors")}, ensure_ascii=False))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
