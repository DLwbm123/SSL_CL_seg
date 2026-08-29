#!/usr/bin/env python3
"""Remove only macOS AppleDouble sidecars from the derived HDF5 root."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.acceptance import appledouble_files  # noqa: E402
from lcrseg.common import DATA_ROOT, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="List or remove only ._*.h5 AppleDouble sidecars under derived HDF5.")
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    targets = appledouble_files(args.root)
    payload = {"root": str(args.root), "targets": [path.relative_to(args.root).as_posix() for path in targets], "execute": args.execute}
    if args.execute:
        for path in targets:
            path.unlink()
        payload["removed"] = len(targets)
    else:
        payload["would_remove"] = len(targets)
    write_json(args.root / "reports" / "validation" / "appledouble_cleanup.json", payload)
    print(json.dumps({key: payload[key] for key in payload if key != "targets"}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
