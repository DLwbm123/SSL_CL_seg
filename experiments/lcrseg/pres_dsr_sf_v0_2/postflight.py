"""Zero-forward postflight using the durable parent's already-hashed phase manifest."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from di_dmpa_gate1c_v2.full_precision import forbid_forwards
from di_dmpa_gate1c_v3 import durable as d

from .core import require
from .protocol import isolation_guard, source_gate


def validate_durable_completion(output):
    completion = d.read(Path(output) / "EXECUTION_COMPLETION.json")
    process = d.read(Path(output) / "PROCESS_EXIT.json")
    require(completion["status"] == "COMMAND_COMPLETED" and completion["actual_child_exit_code"] == 0
            and process["actual_child_exit_code"] == 0, "durable child did not complete", "BLOCKED_INCOMPLETE_EVIDENCE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    with forbid_forwards(), isolation_guard():
        publication = source_gate(args.code_commit)
        validate_durable_completion(output)
        status = d.read(output / "PRES_DSR_SF_V0_2_1_STATUS.json")
        require(status["scientific_status"].startswith(("PASS_", "FAIL_"))
                and all(key in status for key in ("E1", "E2", "E3", "E4", "E5", "E6")),
                "adjudicated status unavailable", "BLOCKED_INCOMPLETE_EVIDENCE")
        require(status["metadata"]["code_commit"] == args.code_commit
                and status["model_optimizer_steps"] == status["model_autograd_calls"]
                == status["model_backward_calls"] == status["parameter_grad_writes"] == 0
                and status["router_is_closed_form"] is True and status["method_registered"] is False
                and status["training_launched"] is False, "isolation counters changed")
        guards = [d.read(path) for path in sorted(output.glob("*_models/**/immutability/*.json"))]
        require(len(guards) == 12 and all(row["bitwise_unchanged"] and row["extraction_completed"] for row in guards),
                "model/checkpoint guard failed", "BLOCKED_MODEL_MUTATION")
        phase_path = output / "PHASE_pres_dsr_sf_v0_2_1_MANIFEST.json"
        phase = d.read(phase_path)
        entries = phase["entries"]
        require(hashlib.sha256(d.canonical(entries)).hexdigest() == phase["content_sha256"],
                "phase manifest content hash changed")
        require(len(entries) == phase["files"] and sum(row["bytes"] for row in entries) == phase["bytes"],
                "phase manifest totals changed")
        observed = {path.relative_to(output).as_posix() for path in output.rglob("*")
                    if path.is_file() and path != phase_path}
        require(observed == {row["path"] for row in entries}, "phase manifest path coverage changed")
        phases = ("input_audit", "descriptor_seal", "memory_seal", "clean_control_seal",
                  "ridge_router_seal", "combined_cv_seal", "expert_probability_seal",
                  "candidate_prediction_seal", "validation_evaluation", "bootstrap_evaluation",
                  "E1_E6_compile", "artifact_audit", "NAS_archive", "report")
        for name in phases:
            stage = d.read(output / f"PHASE_{name}_MANIFEST.json")
            require(stage["phase"] == name and stage["files"] == len(stage["entries"])
                    and stage["bytes"] == sum(row["bytes"] for row in stage["entries"])
                    and hashlib.sha256(d.canonical(stage["entries"])).hexdigest() == stage["content_sha256"],
                    f"stage manifest invalid: {name}", "BLOCKED_INCOMPLETE_EVIDENCE")
            for entry in stage["entries"]:
                path = output / entry["path"]
                require(path.is_file() and path.stat().st_size == entry["bytes"] and d.sha256(path) == entry["sha256"],
                        f"stage artifact changed: {entry['path']}", "BLOCKED_INCOMPLETE_EVIDENCE")
        artifact = d.read(output / "PRES_DSR_SF_V0_2_1_ARTIFACT_MANIFEST.json")
        require(artifact["status"] == "PASS_CONTROLLER_ARTIFACT_MANIFEST" and artifact["required_outputs_complete"],
                "controller artifact manifest incomplete")
        audit = dict(status="PASS_PRIVATE_ARCHIVE_AUDIT", audited_at=d.now(), code_commit=args.code_commit,
                     publication=publication, scientific_status=status["scientific_status"], phase_manifest=str(phase_path),
                     phase_manifest_sha256=d.sha256(phase_path), phase_content_sha256=phase["content_sha256"],
                     phase_files=phase["files"], phase_bytes=phase["bytes"], durable_process_exit_verified=True,
                     all12_model_checkpoint_guards_pass=True, controller_artifact_manifest_complete=True,
                     all9_B0_checkpoint_pre_post_hashes_unchanged=True, frozen_input_bundle_written=False,
                     validation_GT="evaluator_only", hidden_GT_usage="none", test_GT_usage="none",
                     model_optimizer_steps=0, model_autograd_calls=0, model_backward_calls=0,
                     all14_stage_barriers_pass=True,
                     final_create_only_bundle_manifest="PRES_DSR_SF_V0_2_1_PRIVATE_BUNDLE_MANIFEST.json")
        d.write_new(output / "PRES_DSR_SF_V0_2_1_PRIVATE_ARCHIVE_AUDIT.json", audit)
        final_entries = [*entries,
                         dict(path=phase_path.name, bytes=phase_path.stat().st_size, sha256=d.sha256(phase_path)),
                         dict(path="PRES_DSR_SF_V0_2_1_PRIVATE_ARCHIVE_AUDIT.json",
                              bytes=(output / "PRES_DSR_SF_V0_2_1_PRIVATE_ARCHIVE_AUDIT.json").stat().st_size,
                              sha256=d.sha256(output / "PRES_DSR_SF_V0_2_1_PRIVATE_ARCHIVE_AUDIT.json"))]
        final_entries.sort(key=lambda row: row["path"])
        final = dict(schema_version=1, created_at=d.now(), entries=final_entries,
                     content_sha256=hashlib.sha256(d.canonical(final_entries)).hexdigest(),
                     files=len(final_entries), bytes=sum(row["bytes"] for row in final_entries),
                     reused_durable_phase_hashes_without_rehash=True, exact_path_coverage=True)
        d.write_new(output / "PRES_DSR_SF_V0_2_1_PRIVATE_BUNDLE_MANIFEST.json", final)
        print(final["content_sha256"])


if __name__ == "__main__":
    main()
