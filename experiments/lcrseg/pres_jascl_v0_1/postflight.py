"""Zero-forward postflight using the durable parent's already-hashed phase manifest."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess

from di_dmpa_gate1c_v2.full_precision import forbid_forwards
from di_dmpa_gate1c_v3 import durable as d

from .core import require
from .protocol import isolation_guard, source_gate


def validate_durable_completion(output):
    output = Path(output)
    completion = d.read(output/"EXECUTION_COMPLETION.json")
    process = d.read(output/"PROCESS_EXIT.json")
    require(completion["status"] == "COMMAND_COMPLETED" and completion["actual_child_exit_code"] == 0
            and process["actual_child_exit_code"] == 0, "durable child did not complete", "BLOCKED_INCOMPLETE_EVIDENCE")
    return completion, process


def validate_terminal_status(status):
    scientific = status["scientific_status"]
    adjudicated = scientific.startswith(("PASS_", "FAIL_")) and status["D5"] is True
    diagnosed_blocker = scientific == "BLOCKED_PROTOCOL_OR_LEAKAGE" and status["D5"] is False
    require(adjudicated or diagnosed_blocker, "final status/D5 unavailable", "BLOCKED_INCOMPLETE_EVIDENCE")
    return "ADJUDICATED" if adjudicated else "ENGINEERING_BLOCKED"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--execution-code-commit")
    args = parser.parse_args()
    output = args.output.resolve()
    with forbid_forwards(), isolation_guard():
        publication = source_gate(args.code_commit)
        completion, process = validate_durable_completion(output)
        status = d.read(output/"PRES_JASCL_STATUS.json")
        terminal_kind = validate_terminal_status(status)
        execution_commit = args.execution_code_commit or args.code_commit
        require(status["metadata"]["code_commit"] == execution_commit
                and status["metadata"]["publication"]["code_commit"] == execution_commit,
                "execution source identity changed")
        require(subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[3]), "merge-base",
                                "--is-ancestor", execution_commit, args.code_commit]).returncode == 0,
                "audit source does not descend from execution source")
        require(status["model_optimizer_steps"] == status["router_optimizer_steps"] == status["autograd_calls"]
                == status["backward_calls"] == status["parameter_grad_writes"] == 0
                and status["method_registered"] is False and status["training_launched"] is False,
                "isolation counters changed")
        guards = [d.read(path) for path in sorted(output.glob("*_models/**/immutability/*.json"))]
        require(len(guards) == 12 and all(value["bitwise_unchanged"] and value["extraction_completed"] for value in guards),
                "model/checkpoint guard failed", "BLOCKED_MODEL_MUTATION")
        phase_path = output/"PHASE_pres_jascl_MANIFEST.json"
        phase = d.read(phase_path)
        entries = phase["entries"]
        require(hashlib.sha256(d.canonical(entries)).hexdigest() == phase["content_sha256"], "phase manifest content hash changed")
        require(len(entries) == phase["files"] and sum(row["bytes"] for row in entries) == phase["bytes"],
                "phase manifest totals changed")
        observed = {path.relative_to(output).as_posix() for path in output.rglob("*")
                    if path.is_file() and path != phase_path}
        require(observed == {row["path"] for row in entries}, "phase manifest path coverage changed")
        artifact = d.read(output/"PRES_JASCL_ARTIFACT_MANIFEST.json")
        require(artifact["status"] == "PASS_CONTROLLER_ARTIFACT_MANIFEST" and artifact["required_outputs_complete"],
                "controller artifact manifest incomplete")
        audit = dict(status="PASS_PRIVATE_ARCHIVE_AUDIT", audited_at=d.now(),
                     execution_code_commit=execution_commit, audit_code_commit=args.code_commit,
                     audit_publication=publication, terminal_kind=terminal_kind,
                     scientific_status=status["scientific_status"], D5=status["D5"], phase_manifest=str(phase_path),
                     phase_manifest_sha256=d.sha256(phase_path), phase_content_sha256=phase["content_sha256"],
                     phase_files=phase["files"], phase_bytes=phase["bytes"], durable_process_exit_verified=True,
                     all12_model_checkpoint_guards_pass=True, controller_artifact_manifest_complete=True,
                     all9_B0_checkpoint_pre_post_hashes_unchanged=True, frozen_input_bundle_written=False,
                     validation_GT="evaluator_only", hidden_GT_usage="none", test_GT_usage="none",
                     model_optimizer_steps=0, router_optimizer_steps=0, autograd_calls=0, backward_calls=0,
                     final_create_only_bundle_manifest="PRES_JASCL_PRIVATE_BUNDLE_MANIFEST.json")
        d.write_new(output/"PRES_JASCL_PRIVATE_ARCHIVE_AUDIT.json", audit)
        final_entries = [*entries,
                         dict(path=phase_path.name, bytes=phase_path.stat().st_size, sha256=d.sha256(phase_path)),
                         dict(path="PRES_JASCL_PRIVATE_ARCHIVE_AUDIT.json",
                              bytes=(output/"PRES_JASCL_PRIVATE_ARCHIVE_AUDIT.json").stat().st_size,
                              sha256=d.sha256(output/"PRES_JASCL_PRIVATE_ARCHIVE_AUDIT.json"))]
        final_entries.sort(key=lambda row: row["path"])
        final = dict(schema_version=1, created_at=d.now(), entries=final_entries,
                     content_sha256=hashlib.sha256(d.canonical(final_entries)).hexdigest(),
                     files=len(final_entries), bytes=sum(row["bytes"] for row in final_entries),
                     reused_durable_phase_hashes_without_rehash=True, exact_path_coverage=True)
        d.write_new(output/"PRES_JASCL_PRIVATE_BUNDLE_MANIFEST.json", final)
        print(final["content_sha256"])


if __name__ == "__main__":
    main()
