#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.preprocess import select_mnms_fov  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(select_mnms_fov(), ensure_ascii=False, sort_keys=True))
