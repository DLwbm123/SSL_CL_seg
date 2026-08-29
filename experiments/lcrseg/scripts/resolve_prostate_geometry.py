#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.geometry import resolve_prostate_geometry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen automatic prostate geometry rule.")
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args()
    print(json.dumps(resolve_prostate_geometry(render_overlays=not args.no_overlays), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
