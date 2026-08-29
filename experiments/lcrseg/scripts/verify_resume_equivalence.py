#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


GROUPS = (
    "student",
    "ema_teacher",
    "optimizer",
    "scheduler",
    "gas_state",
    "stage_state",
    "sampler_state",
    "rng_state",
)


def compare(left: Any, right: Any, *, atol: float, rtol: float) -> tuple[bool, float]:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            return False, float("inf")
        if left.numel() == 0:
            return True, 0.0
        if left.is_floating_point() or left.is_complex():
            maximum = float((left - right).abs().max())
            return bool(torch.allclose(left, right, atol=atol, rtol=rtol)), maximum
        return bool(torch.equal(left, right)), 0.0 if torch.equal(left, right) else float("inf")
    if isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or left.shape != right.shape or left.dtype != right.dtype:
            return False, float("inf")
        if left.size == 0:
            return True, 0.0
        if np.issubdtype(left.dtype, np.floating):
            maximum = float(np.max(np.abs(left - right)))
            return bool(np.allclose(left, right, atol=atol, rtol=rtol)), maximum
        return bool(np.array_equal(left, right)), 0.0 if np.array_equal(left, right) else float("inf")
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            return False, float("inf")
        results = [compare(left[key], right[key], atol=atol, rtol=rtol) for key in left]
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            return False, float("inf")
        results = [compare(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right)]
    else:
        return left == right, 0.0 if left == right else float("inf")
    return all(item[0] for item in results), max((item[1] for item in results), default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    groups: dict[str, Any] = {}
    for group in GROUPS:
        matched, maximum = compare(reference[group], candidate[group], atol=args.atol, rtol=args.rtol)
        groups[group] = {"within_tolerance": matched, "max_abs_difference": maximum}
    report = {
        "status": "PASS" if all(item["within_tolerance"] for item in groups.values()) else "FAIL",
        "atol": args.atol,
        "rtol": args.rtol,
        "reference": str(Path(args.reference).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "groups": groups,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
