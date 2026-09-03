#!/usr/bin/env python3
"""CARe-HR V0.7 review-only CLI."""
import sys


_DANGER = ("--train", "--fit", "--evaluate", "--formal", "--data-root", "--nas-root",
           "--checkpoint", "--v0-6b-root")
if any(argument == flag or argument.startswith(flag + "=") for argument in sys.argv[1:] for flag in _DANGER):
    raise SystemExit("BLOCKED_AWAITING_EXTERNAL_CODE_REVIEW")

import argparse
import json
from pathlib import Path

from care_hr_v0_7.executor import print_contract, run_static_audit, synthetic_tests


def main(arguments=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("synthetic-tests", "print-contract", "static-audit"))
    args = parser.parse_args(arguments)
    if args.mode == "synthetic-tests":
        result = synthetic_tests()
    elif args.mode == "print-contract":
        result = print_contract()
    else:
        result = run_static_audit(Path(__file__).resolve().parents[2])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
