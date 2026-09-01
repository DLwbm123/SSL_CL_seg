"""Zero-forward SHOR postflight over durable and stage manifests."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from di_dmpa_gate1c_v3 import durable as d
from pres_jascl_v0_1.core import require

from .protocol import PHASES, isolation_guard, source_gate, verify_private_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--private-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    with isolation_guard():
        publication = source_gate(args.code_commit)
        completion = d.read(output / "EXECUTION_COMPLETION.json")
        process = d.read(output / "PROCESS_EXIT.json")
        require(completion["status"] == "COMMAND_COMPLETED" and completion["actual_child_exit_code"] == 0
                and process["actual_child_exit_code"] == 0, "durable child did not complete",
                "BLOCKED_INCOMPLETE_EVIDENCE")
        status = d.read(output / "SHOR_STATUS.json")
        require(status["scientific_status"].startswith(("PASS_", "FAIL_"))
                and all(key in status for key in ("H1", "H2", "H3", "H4", "H5", "H6")),
                "SHOR adjudication unavailable", "BLOCKED_INCOMPLETE_EVIDENCE")
        require(status["metadata"]["code_commit"] == args.code_commit
                and status["new_model_forwards"] == status["model_constructions"]
                == status["checkpoint_tensor_reads"] == status["model_autograd_calls"]
                == status["model_backward_calls"] == status["model_optimizer_steps"]
                == status["router_optimizer_steps"] == status["parameter_grad_writes"] == 0
                and status["training_launched"] is False and status["test_GT_reads"] == 0,
                "zero-forward isolation changed", "BLOCKED_PROTOCOL_OR_LEAKAGE")
        old_private = verify_private_bundle(args.private_root)
        phase_path = output / "PHASE_shor_jascl_v0_3_MANIFEST.json"
        phase = d.read(phase_path)
        entries = phase["entries"]
        require(hashlib.sha256(d.canonical(entries)).hexdigest() == phase["content_sha256"]
                and len(entries) == phase["files"]
                and sum(row["bytes"] for row in entries) == phase["bytes"],
                "durable phase manifest changed", "BLOCKED_INCOMPLETE_EVIDENCE")
        observed = {path.relative_to(output).as_posix() for path in output.rglob("*")
                    if path.is_file() and path != phase_path}
        require(observed == {row["path"] for row in entries}, "durable phase path coverage changed",
                "BLOCKED_INCOMPLETE_EVIDENCE")
        for name in PHASES:
            stage = d.read(output / f"PHASE_{name}_MANIFEST.json")
            require(stage["phase"] == name and stage["files"] == len(stage["entries"])
                    and stage["bytes"] == sum(row["bytes"] for row in stage["entries"])
                    and hashlib.sha256(d.canonical(stage["entries"])).hexdigest() == stage["content_sha256"],
                    f"stage manifest invalid: {name}", "BLOCKED_INCOMPLETE_EVIDENCE")
            for entry in stage["entries"]:
                path = output / entry["path"]
                require(path.is_file() and path.stat().st_size == entry["bytes"] and d.sha256(path) == entry["sha256"],
                        f"stage artifact changed: {entry['path']}", "BLOCKED_INCOMPLETE_EVIDENCE")
        artifact = d.read(output / "SHOR_ARTIFACT_MANIFEST.json")
        require(artifact["status"] == "PASS_SHOR_ARTIFACT_MANIFEST" and artifact["required_outputs_complete"],
                "SHOR artifact manifest incomplete", "BLOCKED_INCOMPLETE_EVIDENCE")
        audit = dict(status="PASS_SHOR_PRIVATE_ARCHIVE_AUDIT", audited_at=d.now(), code_commit=args.code_commit,
                     publication=publication, scientific_status=status["scientific_status"],
                     durable_process_exit_verified=True, all9_stage_barriers_pass=True,
                     controller_artifact_manifest_complete=True, phase_manifest=phase_path.name,
                     phase_manifest_sha256=d.sha256(phase_path), phase_content_sha256=phase["content_sha256"],
                     phase_files=phase["files"], phase_bytes=phase["bytes"], private_input=old_private,
                     private_input_unchanged=True, new_model_forwards=0, model_constructions=0,
                     checkpoint_tensor_reads=0, model_autograd_calls=0, model_backward_calls=0,
                     model_optimizer_steps=0, router_optimizer_steps=0, parameter_grad_writes=0,
                     threshold_builder_segmentation_GT_usage="none", validation_GT="evaluator_only_after_candidate_seal",
                     test_GT_reads=0, final_create_only_bundle_manifest="SHOR_PRIVATE_BUNDLE_MANIFEST.json")
        d.write_new(output / "SHOR_PRIVATE_ARCHIVE_AUDIT.json", audit)
        final_entries = [*entries,
                         dict(path=phase_path.name, bytes=phase_path.stat().st_size, sha256=d.sha256(phase_path)),
                         dict(path="SHOR_PRIVATE_ARCHIVE_AUDIT.json",
                              bytes=(output / "SHOR_PRIVATE_ARCHIVE_AUDIT.json").stat().st_size,
                              sha256=d.sha256(output / "SHOR_PRIVATE_ARCHIVE_AUDIT.json"))]
        final_entries.sort(key=lambda row: row["path"])
        final = dict(schema_version=1, created_at=d.now(), entries=final_entries,
                     content_sha256=hashlib.sha256(d.canonical(final_entries)).hexdigest(), files=len(final_entries),
                     bytes=sum(row["bytes"] for row in final_entries), exact_path_coverage=True,
                     reused_durable_phase_hashes_without_rehash=True)
        d.write_new(output / "SHOR_PRIVATE_BUNDLE_MANIFEST.json", final)
        print(final["content_sha256"])


if __name__ == "__main__":
    main()
