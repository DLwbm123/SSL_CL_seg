"""Authority, zero-forward admission, private-bundle audit, call graph, and barriers."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
import socket
import subprocess
from unittest.mock import patch
from urllib.request import Request, urlopen

import torch

from di_dmpa_gate1c_v2 import binding as b
from di_dmpa_gate1c_v2.full_precision import forbid_forwards
from di_dmpa_gate1c_v3 import durable as d
from pres_jascl_v0_1.core import Blocked, require
from pres_jascl_v0_1.protocol import gate1c_contract

from . import REGISTRATION

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DOCS = ROOT / "docs/shor_jascl_v0_3"
BRANCH = "codex/shor-jascl-v0-3-feasibility"
REMOTE = "https://github.com/DLwbm123/SSL_CL_seg.git"
BASE_HEAD = "c854bd28b1a69ce001646201a824b8bb75141c67"
NAS_ROOT = Path("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg")
PRIVATE_ROOT = NAS_ROOT / "protocols/pres_dsr_sf_v0_2_1_752e1ac/formal_01"
PRIVATE_BUNDLE = PRIVATE_ROOT / "PRES_DSR_SF_V0_2_1_PRIVATE_BUNDLE_MANIFEST.json"
PRIVATE_FILES = 183
PRIVATE_BYTES = 4386018614
PRIVATE_CONTENT_SHA = "05c9008ad4496ccbdc51df6103638024d49fae4b3b4cdc2a9f829c5f3ab165bb"

AUTHORITY = (
    ("9feee43c5e34c427356ceaaafa6f691dd14186a3", "PRES_DSR_SF_V0_2_1_FINAL_CLOSURE.json",
     "61375093d58c73727e1ae6500f9a6e3371e8bc2e566807c92e8d00c6972f841d"),
    ("6eaf8b8a299a47dec7a296ef2d784a105a53ab55", "SHOR_JASCL_V0_3_PREREGISTRATION.json",
     "c9c4bfcc1d4ade83c98a5b7171f29b4b927166dfa4adbf7ef9235cbc7e9534b7"),
    ("9a229531cfb553aa4f44d7780b3b5110b6344f0f", "SHOR_JASCL_V0_3_EXECUTION_AUTHORIZATION.json",
     "2b526d4edf95a5f6d34d91455af0ccfa277a3c8ca24d4fbefac33a8e57d05aa3"),
)


def authority(code_commit=None):
    values = {}
    for commit, name, digest in AUTHORITY:
        path = DOCS / name
        b.check_hash(path, digest)
        blob = subprocess.check_output(["git", "-C", str(REPO), "show", f"{commit}:{path.relative_to(REPO)}"])
        require(hashlib.sha256(blob).hexdigest() == digest, f"published authority changed: {name}")
        if code_commit:
            require(subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, code_commit],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0,
                    f"authority is not an ancestor: {commit}")
        values[name] = d.read(path)
    closure, prereg, auth = (values[row[1]] for row in AUTHORITY)
    require(closure["soft_expert_fusion_status"] == "FAIL_SOFT_EXPERT_FUSION_VALUE"
            and closure["additional_soft_fusion_attempts_authorized"] is False
            and closure["next_protocol"] == "SHOR_JASCL_V0_3", "V0.2.1 closure changed")
    require(prereg["registration_id"] == auth["registration_id"] == REGISTRATION
            and prereg["base"]["branch_head"] == BASE_HEAD, "SHOR identity changed")
    require(auth["preregistration_commit"] == AUTHORITY[1][0]
            and auth["formal_attempts_authorized"] == 1 and auth["zero_forward_required"] is True,
            "SHOR authorization changed")
    require(prereg["frozen_input"]["bundle"]["files"] == PRIVATE_FILES
            and prereg["frozen_input"]["bundle"]["bytes"] == PRIVATE_BYTES
            and prereg["frozen_input"]["bundle"]["content_sha256"] == PRIVATE_CONTENT_SHA,
            "private bundle binding changed")
    return prereg


def source_gate(code_commit):
    observed = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    require(observed == code_commit, "execution HEAD differs from requested code commit",
            "BLOCKED_BASE_COMMIT_AMBIGUOUS")
    require(not subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip(),
            "execution checkout is dirty", "BLOCKED_PROTOCOL_OR_LEAKAGE")
    changes = subprocess.check_output(["git", "-C", str(REPO), "diff", "--name-status", BASE_HEAD, code_commit],
                                      text=True).splitlines()
    prefixes = ("experiments/lcrseg/docs/shor_jascl_v0_3/", "experiments/lcrseg/shor_jascl_v0_3/",
                "experiments/lcrseg/tests/shor_jascl_v0_3/")
    require(changes and all(row.startswith("A\t") and row.split("\t", 1)[1].startswith(prefixes) for row in changes),
            "unregistered SHOR source delta", "BLOCKED_PROTOCOL_OR_LEAKAGE")
    require(subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "58ee45b12aae662c8fe61595dc4068094c783f7c",
                            "--", "experiments/lcrseg/pres_dsr_sf_v0_2"], stdout=subprocess.DEVNULL).returncode == 0,
            "frozen PRES-DSR-SF source changed", "BLOCKED_PROTOCOL_OR_LEAKAGE")
    remote = subprocess.check_output(["git", "ls-remote", REMOTE, "refs/heads/" + BRANCH], text=True).split()
    require(remote and remote[0] == code_commit, "git ls-remote code barrier failed",
            "BLOCKED_BASE_COMMIT_AMBIGUOUS")
    url = "https://api.github.com/repos/DLwbm123/SSL_CL_seg/git/ref/heads/" + BRANCH
    with urlopen(Request(url + "?expected=" + code_commit,
                         headers={"User-Agent": "SHOR-JASCL", "Cache-Control": "no-cache"}), timeout=30) as response:
        github = json.load(response)
    require(github["object"]["sha"] == code_commit, "GitHub API code barrier failed",
            "BLOCKED_BASE_COMMIT_AMBIGUOUS")
    prereg = authority(code_commit)
    files = sorted((ROOT / "shor_jascl_v0_3").glob("*.py"))
    source = {path.name: d.sha256(path) for path in files}
    require(set(source) == {"__init__.py", "core.py", "postflight.py", "protocol.py", "run.py", "testing.py"},
            "SHOR source set changed", "BLOCKED_PROTOCOL_OR_LEAKAGE")
    return dict(status="PASS_EXACT_PUBLISHED_SOURCE", branch=BRANCH, code_commit=code_commit,
                git_ls_remote_sha=remote[0], github_api_sha=github["object"]["sha"], source_sha256=source,
                registration_id=prereg["registration_id"], checked_at=d.now())


def execution_gate(output, code_commit, test_report, private_root):
    require(socket.gethostname() == "zmic44" and os.getuid() == os.geteuid() == 1006, "wrong execution host/uid")
    output = Path(output).resolve()
    require(output.is_relative_to(NAS_ROOT) and not output.is_symlink(), "NAS-only non-symlink output required")
    require(os.environ.get("SSLCL_STORAGE_ROOT") == str(NAS_ROOT)
            and Path(os.environ.get("TMPDIR", "/")).resolve().is_relative_to(NAS_ROOT), "NAS wrapper/cache policy absent")
    require(Path(private_root).resolve() == PRIVATE_ROOT and not Path(private_root).is_symlink(),
            "private input root changed", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
    publication = source_gate(code_commit)
    report = d.read(test_report)
    require(report["status"] == "PASS" and report["code_commit"] == code_commit
            and report["failures"] == report["errors"] == report["skips"] == 0, "test gate failed")
    b.check_hash(report["junit_path"], report["junit_sha256"])
    b.check_hash(report["pytest_output_path"], report["pytest_output_sha256"])
    return dict(registration_id=REGISTRATION, code_commit=code_commit, publication=publication,
                test_report=str(Path(test_report).resolve()), test_report_sha256=d.sha256(test_report),
                exact_test_command=report["exact_test_command"], hostname=socket.gethostname(), uid=os.getuid(),
                started_at=d.now(), private_root=str(Path(private_root).resolve()), new_model_forwards=0,
                model_autograd_calls=0, model_optimizer_steps=0, router_optimizer_steps=0,
                model_constructions=0, training_launched=False, method_registered=False)


def isolation_guard():
    stack = ExitStack()
    stack.enter_context(forbid_forwards())
    stack.enter_context(b.no_updates())
    stack.enter_context(patch.object(torch, "load", side_effect=Blocked("checkpoint tensor load forbidden")))
    stack.enter_context(patch.object(torch.autograd, "grad", side_effect=Blocked("autograd forbidden")))
    stack.enter_context(patch.object(torch.Tensor, "backward", side_effect=Blocked("backward forbidden")))
    stack.enter_context(patch.object(torch.nn.Module, "__init__", side_effect=Blocked("model construction forbidden")))
    return stack


def verify_private_bundle(private_root=PRIVATE_ROOT):
    root = Path(private_root).resolve()
    require(root == PRIVATE_ROOT and root.is_dir() and not root.is_symlink(), "private root mismatch",
            "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
    try:
        bundle = d.read(root / PRIVATE_BUNDLE.name)
        require(bundle["files"] == PRIVATE_FILES and bundle["bytes"] == PRIVATE_BYTES
                and bundle["content_sha256"] == PRIVATE_CONTENT_SHA and bundle["exact_path_coverage"] is True,
                "private bundle identity mismatch", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
        require(hashlib.sha256(d.canonical(bundle["entries"])).hexdigest() == PRIVATE_CONTENT_SHA,
                "private bundle entry digest mismatch", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
        for entry in bundle["entries"]:
            path = root / entry["path"]
            require(path.is_file() and not path.is_symlink() and path.stat().st_size == entry["bytes"]
                    and d.sha256(path) == entry["sha256"], f"private artifact changed: {entry['path']}",
                    "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
        observed = {path.relative_to(root).as_posix() for path in root.rglob("*")
                    if path.is_file() and path != root / PRIVATE_BUNDLE.name}
        require(observed == {entry["path"] for entry in bundle["entries"]}, "private path coverage changed",
                "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
    except Blocked:
        raise
    except Exception as error:
        raise Blocked(str(error), "BLOCKED_PRIVATE_BUNDLE_MISMATCH") from error
    return dict(status="PASS_PRIVATE_BUNDLE_FULL_VERIFICATION", root=str(root), files=bundle["files"],
                bytes=bundle["bytes"], content_sha256=bundle["content_sha256"],
                manifest_sha256=d.sha256(root / PRIVATE_BUNDLE.name), every_file_sha256_verified=True,
                exact_path_coverage=True)


def input_audit(output, metadata, private_root=PRIVATE_ROOT):
    with isolation_guard():
        private = verify_private_bundle(private_root)
        required = ("descriptor_cache", "memory_cache", "PRES_DSR_SF_ROUTER_MANIFEST.json",
                    "PRES_DSR_SF_ROUTING_METADATA.json", "expert_probability_cache",
                    "PRES_DSR_SF_EXPERT_PROBABILITY_MANIFEST.json", "PRES_DSR_SF_CANDIDATE_MANIFEST.json")
        require(all((Path(private_root) / name).exists() for name in required), "sealed SHOR input missing",
                "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
        old = d.read(Path(private_root) / "PRES_DSR_SF_V0_2_1_STATUS.json")
        require(old["scientific_status"] == "FAIL_SOFT_EXPERT_FUSION_VALUE"
                and [old[f"E{i}"] for i in range(1, 7)] == [True, True, True, False, True, True],
                "V0.2.1 scientific status changed", "BLOCKED_BASE_COMMIT_AMBIGUOUS")
        result = dict(status="PASS_SHOR_INPUT_AUDIT", metadata=metadata, private_bundle=private,
                      allowed_cache_roots=list(required), prior_scientific_status=old["scientific_status"],
                      prior_E=[old[f"E{i}"] for i in range(1, 7)], model_constructions=0,
                      checkpoint_tensor_reads=0, new_model_forwards=0, validation_GT_reads=0,
                      test_objects_constructed=0, threshold_builder_segmentation_GT_fields=0,
                      audited_at=d.now())
        d.write_new(Path(output) / "SHOR_INPUT_AUDIT.json", result)
    return result


def compile_call_graph(output, stage_case_counts, code_commit):
    stage1, stage2 = stage_case_counts[1], stage_case_counts[2]
    graph = dict(status="PASS_ZERO_FORWARD_CALL_GRAPH_SEALED", code_commit=code_commit,
                 new_model_forwards=0, model_constructions=0, checkpoint_tensor_reads=0,
                 formal_threshold_units=9, bootstrap_threshold_units=45, formal_ridge_units=6,
                 bootstrap_ridge_units=30, ridge_closed_form_fits=1116, bootstrap_operations=30,
                 formal_route_rows=3 * (stage1 + stage2), formal_candidate_case_predictions=4 * 3 * (stage1 + stage2),
                 bootstrap_candidate_case_predictions=5 * 3 * (stage1 + stage2),
                 validation_GT_case_reads=495, segmentation_rows=75,
                 failure_attribution_rows=3 * (stage1 + stage2), override_utility_rows=15,
                 bootstrap_metric_rows=75, compiled_at=d.now())
    require(stage1 == 140 and stage2 == 165 and graph["formal_route_rows"] == 915
            and graph["formal_candidate_case_predictions"] == 3660
            and graph["bootstrap_candidate_case_predictions"] == 4575,
            "SHOR call graph cardinality changed", "BLOCKED_OUTPUT_KEYSET_MISMATCH")
    d.write_new(Path(output) / "SHOR_CALL_GRAPH.json", graph)
    return graph


def verify_call_graph(expected, observed):
    keys = ("new_model_forwards", "model_constructions", "checkpoint_tensor_reads", "formal_threshold_units",
            "bootstrap_threshold_units", "ridge_closed_form_fits", "bootstrap_operations", "formal_route_rows",
            "formal_candidate_case_predictions", "bootstrap_candidate_case_predictions",
            "validation_GT_case_reads", "segmentation_rows", "failure_attribution_rows",
            "override_utility_rows", "bootstrap_metric_rows")
    require(all(observed[key] == expected[key] for key in keys), "executed SHOR call graph changed",
            "BLOCKED_OUTPUT_KEYSET_MISMATCH")


PHASES = ("input_audit", "oof_threshold_seal", "candidate_prediction_seal", "validation_evaluation",
          "bootstrap_evaluation", "H1_H6_compile", "artifact_audit", "NAS_archive", "report")


def phase_barrier(output, name, artifacts):
    output = Path(output).resolve()
    require(name in PHASES, "unregistered SHOR phase", "BLOCKED_PROTOCOL_OR_LEAKAGE")
    paths = [output / path for path in artifacts]
    require(all(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(output) for path in paths),
            f"phase {name} artifact missing", "BLOCKED_INCOMPLETE_EVIDENCE")
    phase_path = output / f"PHASE_{name}.json"
    d.write_new(phase_path, dict(status="PASS", phase=name, artifacts=list(artifacts), completed_at=d.now()))
    entries = [dict(path=path.relative_to(output).as_posix(), bytes=path.stat().st_size, sha256=d.sha256(path))
               for path in [*paths, phase_path]]
    entries.sort(key=lambda row: row["path"])
    manifest = dict(schema_version=1, phase=name, created_at=d.now(), entries=entries, files=len(entries),
                    bytes=sum(row["bytes"] for row in entries),
                    content_sha256=hashlib.sha256(d.canonical(entries)).hexdigest())
    d.write_new(output / f"PHASE_{name}_MANIFEST.json", manifest)
    return manifest
