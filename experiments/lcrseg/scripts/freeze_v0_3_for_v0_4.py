#!/usr/bin/env python3
"""Freeze all V0.3 inputs and evidence before the V0.4 failure audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.common import canonical_json, sha256_path, utc_now, write_json, write_text


RUN_NAMES = {
    "R0_seed0": "fundus_seed0_lcrseg_uniform_relation_kd_full200e",
    "R1_seed0": "fundus_seed0_lcrseg_v0_2a_r1_progressive_uniform_full200e",
    "P0_seed0": "fundus_seed0_lcrseg_v0_3_p0_progressive_norelation_full200e",
    "R0_seed1": "fundus_seed1_lcrseg_v0_3_r0_legacy_uniform_full200e",
    "R1_seed1": "fundus_seed1_lcrseg_v0_3_r1_progressive_uniform_full200e",
    "R0_seed2": "fundus_seed2_lcrseg_v0_3_r0_legacy_uniform_full200e",
    "R1_seed2": "fundus_seed2_lcrseg_v0_3_r1_progressive_uniform_full200e",
}


def _tree_manifest(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    digest = hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()
    return rows, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports/experiment_status/V0_3_FREEZE_FOR_V0_4.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROJECT_ROOT / "reports/experiment_status/V0_3_FREEZE_FOR_V0_4.md",
    )
    args = parser.parse_args()
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("refusing to overwrite the V0.3-for-V0.4 freeze")
    gate_path = PROJECT_ROOT / "reports/experiment_status/V0_3_FUNDUS_INTERNAL_GATE.json"
    completion_path = PROJECT_ROOT / "reports/experiment_status/V0_3_FUNDUS_MULTISEED_COMPLETION.json"
    gate = json.loads(gate_path.read_text())
    completion = json.loads(completion_path.read_text())
    if gate.get("status") != "FUNDUS_V0_3_INTERNAL_GATE_FAILED" or gate.get("internal_gate_passed") is not False:
        raise RuntimeError("V0.4 requires the frozen V0.3 internal-gate failure")

    runs: dict[str, Any] = {}
    for identity, name in RUN_NAMES.items():
        run_dir = (args.run_root / name).resolve()
        summary_path = run_dir / "run_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or int(summary.get("completed_global_steps", -1)) != 13_400:
            raise RuntimeError(f"cannot freeze incomplete formal run: {run_dir}")
        artifacts, tree_sha256 = _tree_manifest(run_dir)
        checkpoints = [row for row in artifacts if row["path"].startswith("checkpoint") and row["path"].endswith(".pt")]
        if not checkpoints:
            raise RuntimeError(f"formal run has no checkpoint: {run_dir}")
        runs[identity] = {
            "path": str(run_dir),
            "summary": summary,
            "tree_sha256": tree_sha256,
            "artifact_count": len(artifacts),
            "artifact_hashes": artifacts,
            "checkpoint_hashes": checkpoints,
        }
    payload = {
        "protocol_id": "lcrseg_v0_4_failure_audit",
        "generated_at": utc_now(),
        "v0_3_status": gate["status"],
        "v0_3_failure_level": "research_not_engineering",
        "r2_r3_status": "frozen_negative_results",
        "v0_4_mode": "diagnostic_first",
        "internal_gate": gate,
        "v0_3_completion": completion,
        "runs": runs,
        "conditional_stages_not_executed_in_v0_3": completion.get("unexecuted_due_to_internal_gate", []),
        "immutability_contract": {
            "existing_v0_3_artifacts_read_only": True,
            "v0_4_will_not_overwrite_v0_3_runs": True,
            "hidden_gt_training_usage_allowed": False,
        },
        "source_report_hashes": {
            "V0_3_FINAL_REPORT.md": sha256_path(PROJECT_ROOT / "reports/experiment_status/V0_3_FINAL_REPORT.md"),
            "V0_3_FUNDUS_INTERNAL_GATE.json": sha256_path(gate_path),
            "V0_3_FUNDUS_MULTISEED_COMPLETION.json": sha256_path(completion_path),
        },
    }
    write_json(args.output_json, payload)
    write_text(
        args.output_md,
        "\n".join(
            [
                "# V0.3 freeze for LCR-Seg V0.4",
                "",
                "**V0.3 status:** `FUNDUS_V0_3_INTERNAL_GATE_FAILED`",
                "",
                "- Failure level: `research_not_engineering`",
                "- R2/R3: `frozen_negative_results`",
                "- V0.4 execution mode: `diagnostic_first`",
                "- Existing V0.3 artifacts are immutable and will not be overwritten.",
                f"- Frozen formal runs: `{len(runs)}`",
                f"- Frozen checkpoints: `{sum(len(run['checkpoint_hashes']) for run in runs.values())}`",
                "",
                "The JSON companion contains every run artifact and checkpoint SHA-256.",
                "",
            ]
        ),
    )
    print(json.dumps({"status": "V0_3_FROZEN_FOR_V0_4", "runs": len(runs)}, sort_keys=True))


if __name__ == "__main__":
    main()
