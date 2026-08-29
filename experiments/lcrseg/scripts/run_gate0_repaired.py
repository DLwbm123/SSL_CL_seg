#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from di_dmpa_jascl.config import load_yaml  # noqa: E402
from di_dmpa_jascl.runner import Gate0RepairedRunner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the protocol-bounded gate0_repaired baseline")
    parser.add_argument("--config", default="configs/gate0_repaired/fundus.yaml")
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-global-step", type=int)
    args = parser.parse_args()

    config_path = (REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_yaml(config_path)
    protocol_path = Path(config["data"]["protocol"])
    if not protocol_path.is_absolute():
        protocol_path = REPO_ROOT / protocol_path
    protocol = load_yaml(protocol_path)
    runner = Gate0RepairedRunner(
        repo_root=REPO_ROOT,
        config=config,
        protocol=protocol,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
    )
    result = runner.run(resume_path=args.resume, stop_after_global_step=args.stop_after_global_step)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
