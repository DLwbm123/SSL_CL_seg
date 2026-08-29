#!/usr/bin/env python3
"""Freeze the exact TARC metric callables reused by BPRC feasibility."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.common import sha256_path, write_json, write_text  # noqa: E402


CALLABLES = (
    ("scripts.audit_tarc_relation_fidelity", "_margin", "top1-minus-top2 probability margin"),
    ("scripts.audit_tarc_relation_fidelity", "_previous_fidelity", "relation KL, top1 agreement, and classwise margin agreement"),
    ("scripts.audit_tarc_relation_fidelity", "_current_safety", "current-site relation accuracy, margin, entropy, and finite checks"),
    ("scripts.audit_tarc_virtual_step", "_supervised_r0_loss", "exact frozen R0 supervised validation loss"),
    ("scripts.audit_tarc_virtual_step", "_baseline_loss", "fixed-batch baseline validation loss"),
    ("scripts.audit_tarc_virtual_step", "_functional_val_loss", "stateless functional-view validation loss"),
)


def audit() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    for module_name, function_name, semantics in CALLABLES:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        source = inspect.getsource(function)
        path = Path(inspect.getsourcefile(function) or "").resolve()
        signature = str(inspect.signature(function))
        row = {
            "module_path": module_name,
            "file_path": str(path),
            "function_name": function_name,
            "signature": signature,
            "function_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "module_sha256": sha256_path(path),
            "input_output_semantics": semantics,
        }
        rows.append(row)
        checks[f"{module_name}.{function_name}_callable"] = callable(function)
        checks[f"{module_name}.{function_name}_source_present"] = bool(source.strip())
    relation_module = ROOT / "scripts/audit_tarc_relation_fidelity.py"
    virtual_module = ROOT / "scripts/audit_tarc_virtual_step.py"
    checks["frozen_relation_module_sha_exact"] = sha256_path(relation_module) == "6616b91df4acdca9cab063a720752a9f3d3231ffddf876fdcb56df750e997559"
    checks["frozen_virtual_module_sha_exact"] = sha256_path(virtual_module) == "7539772dc8f7fb0faeada907ebca8712e58fc4d4d4b64e3ba56c26d2890eff54"
    status = "BPRC_METRIC_REUSE_AUDIT_PASSED" if all(checks.values()) else "HARD_STOP_BPRC_AUDIT_ENGINEERING"
    return {
        "protocol_id": "bprcseg_v0_1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "optimizer_steps": 0,
        "hidden_gt_usage": "none",
        "callables": rows,
        "checks": checks,
        "reuse_rule": "BPRC feasibility imports these exact callables; it must not rewrite an approximately equivalent metric.",
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BPRC-Seg V0.1 exact TARC metric reuse audit",
        "",
        f"**Status:** `{report['status']}`  ",
        "**Optimizer steps:** `0`",
        "",
        "| Module | Function | Function SHA-256 | Semantics |",
        "|---|---|---|---|",
    ]
    for row in report["callables"]:
        lines.append(
            f"| `{row['module_path']}` | `{row['function_name']}` | `{row['function_source_sha256']}` | {row['input_output_semantics']} |"
        )
    lines.extend(
        [
            "",
            "The TARC source modules are frozen at their recorded SHA-256 values. BPRC feasibility must import these exact functions and may only add pairwise/class-balanced diagnostics that TARC did not define.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiment_status")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    json_path = args.output_dir.resolve() / "BPRC_METRIC_REUSE_AUDIT.json"
    md_path = args.output_dir.resolve() / "BPRC_METRIC_REUSE_AUDIT.md"
    for path in (json_path, md_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite metric reuse audit: {path}")
    report = audit()
    write_json(json_path, report)
    write_text(md_path, markdown(report))
    print(json.dumps({"status": report["status"], "json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0 if report["status"] == "BPRC_METRIC_REUSE_AUDIT_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
