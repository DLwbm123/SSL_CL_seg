#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.audit import audit_mnms  # noqa: E402


if __name__ == "__main__":
    print(audit_mnms())
