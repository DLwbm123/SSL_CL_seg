#!/usr/bin/env python3
"""Run one reproducible LCR-Seg/baseline experiment on frozen HDF5 data."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.engine.continual_runner import ContinualRunner


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("comma-separated site list is empty")
    return values


def _value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("LCRSEG_DATA_ROOT", "/home/jiangsuiyang/SSL_CL")))
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("LCRSEG_RUN_ROOT", "/home/jiangsuiyang/SSL_CL/runs")))
    parser.add_argument("--dataset", choices=("fundus", "prostate", "mnms"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--sites", help="ordered comma-separated training sites")
    parser.add_argument("--evaluation-sites", help="comma-separated evaluation sites; defaults to the dataset protocol")
    parser.add_argument("--evaluation-role", choices=("val", "test"), default="test")
    parser.add_argument("--allow-writable-inputs", action="store_true", help="development-only override; remote frozen inputs should remain read-only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs-per-site", type=int, default=5)
    parser.add_argument("--steps-per-site", type=int, default=None)
    parser.add_argument("--labeled-batch-size", type=int, default=2)
    parser.add_argument("--unlabeled-batch-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-init-scale", type=float, default=1024.0)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=0)
    parser.add_argument("--gradient-cosine-interval", type=int, default=100)
    parser.add_argument("--max-steps-this-invocation", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--method-override", action="append", default=[], metavar="KEY=JSON_VALUE")
    parser.add_argument("--transform-override", action="append", default=[], metavar="KEY=JSON_VALUE")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{args.dataset}_{args.method}_seed{args.seed}_{timestamp}"
    config = ContinualRunner.default_config(
        data_root=args.root,
        run_root=args.run_root,
        dataset=args.dataset,
        method_name=args.method,
        seed=args.seed,
        site_order=_csv(args.sites),
        run_name=run_name,
        device=args.device,
    )
    if args.evaluation_sites:
        config["data"]["evaluation_sites"] = _csv(args.evaluation_sites)
    config["data"]["evaluation_role"] = args.evaluation_role
    config["data"]["require_readonly"] = not args.allow_writable_inputs
    config["training"].update(
        {
            "epochs_per_site": args.epochs_per_site,
            "steps_per_site": args.steps_per_site,
            "labeled_batch_size": args.labeled_batch_size,
            "unlabeled_batch_size": args.unlabeled_batch_size,
            "evaluation_batch_size": args.evaluation_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "amp": not args.no_amp,
            "amp_init_scale": args.amp_init_scale,
            "checkpoint_interval_steps": args.checkpoint_interval_steps,
            "gradient_cosine_interval": args.gradient_cosine_interval,
            "max_steps_this_invocation": args.max_steps_this_invocation,
        }
    )
    for item in args.method_override:
        if "=" not in item:
            raise ValueError(f"invalid --method-override: {item!r}")
        key, value = item.split("=", 1)
        config["method"][key] = _value(value)
    if args.transform_override:
        config["transforms"] = {}
        for item in args.transform_override:
            if "=" not in item:
                raise ValueError(f"invalid --transform-override: {item!r}")
            key, value = item.split("=", 1)
            config["transforms"][key] = _value(value)
    summary = ContinualRunner(config).run(resume_checkpoint=args.resume)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
