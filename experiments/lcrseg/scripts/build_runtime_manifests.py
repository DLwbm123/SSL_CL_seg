#!/usr/bin/env python3
"""Build portable training/diagnostics manifests from accepted HDF5 records."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.common import DATA_ROOT, write_json  # noqa: E402
from lcrseg.runtime_manifests import build_runtime_manifests, validate_runtime_manifests  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    result = build_runtime_manifests(seeds=args.seeds)
    validation = validate_runtime_manifests(seeds=args.seeds)
    write_json(DATA_ROOT / "reports" / "validation" / "runtime_manifest_validation.json", validation)
    print(json.dumps({"build": result, "validation": validation}, ensure_ascii=False, sort_keys=True))
    if not validation["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
