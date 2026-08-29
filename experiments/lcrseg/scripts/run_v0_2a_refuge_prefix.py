#!/usr/bin/env python3
"""Build the preregistered progressive REFUGE parent for R1/R3 pilots."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.engine.continual_runner import ContinualRunner


RUN_NAME = "pilot_parent_v02a_progressive_refuge_8000steps"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((PROJECT_ROOT / "configs/experiments/lcrseg_v0_2a_r1.yaml").read_text())
    config["experiment"].update({"run_name": RUN_NAME, "run_root": str(args.run_root.resolve())})
    config["data"].update({"site_order": ["REFUGE"], "site_index_offset": 0})
    config["training"].update(
        {
            "steps_per_site": 8000,
            "checkpoint_interval_steps": 500,
            "preserve_interval_checkpoints": False,
            "gradient_cosine_interval": 500,
        }
    )
    runner = ContinualRunner(config)
    if args.validate_only:
        print(json.dumps(runner.config, ensure_ascii=False, sort_keys=True))
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise RuntimeError("the progressive REFUGE pilot parent requires CUDA_VISIBLE_DEVICES=4")
    print(json.dumps(runner.run(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
