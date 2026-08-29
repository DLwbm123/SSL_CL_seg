#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.audit import audit_prostate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only six-site prostate audit.")
    parser.add_argument("--no-overlays", action="store_true", help="Do not write the required candidate overlays.")
    args = parser.parse_args()
    print(audit_prostate(render_overlays=not args.no_overlays))


if __name__ == "__main__":
    main()
