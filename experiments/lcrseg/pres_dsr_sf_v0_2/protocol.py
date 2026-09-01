"""Authority, backend barrier, NAS admission, private-input audit, and call graph."""
from __future__ import annotations

import hashlib
import json
import math
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
from di_dmpa_gate1c_v3.baseline import verify_payload
from di_dmpa_jascl.modeling import _official_probabilistic_classifier
from lcrseg.acceptance import verify_checksums
from pres_jascl_v0_1 import protocol as v1
from pres_jascl_v0_1.core import Blocked, DOMAINS, require
from pres_jascl_v0_1.run import deterministic_backend_state, enforce_deterministic_backend

from . import REGISTRATION

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DOCS = ROOT / "docs/pres_dsr_sf_v0_2"
BRANCH = "codex/pres-dsr-sf-v0-2-feasibility"
REMOTE = "https://github.com/DLwbm123/SSL_CL_seg.git"
BASE_HEAD = "ab71694ad6b3134fe1b45bd479658349e619fdc5"
NAS_ROOT = Path("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg")
BATCH_SIZE = 8

AUTHORITY = (
    ("607a067319a6e8f0bfc1b8d6a305f014cd6ab676", "PRES_JASCL_V0_1_FINAL_CLOSURE.json",
     "cf80645489b5157e2cb664013746084e9745ecaa9b06173049b59e9560820135"),
    ("c4767688e01ee9106d172a88a95f7e6c8a5de0eb", "PRES_DSR_SF_V0_2_PREREGISTRATION.json",
     "f40bc7bb8b6cc26e72946527d959e0bfda863acacb04947b72912a3be0f1d955"),
    ("78427b35ae5101c0576863386df0c434f77d2734", "PRES_DSR_SF_V0_2_EXECUTION_AUTHORIZATION.json",
     "cde55e4c15900c66fe823593f91733327073ac272362280b1940e3bff48095cd"),
)


def authority(code_commit=None):
    values = {}
    for commit, name, digest in AUTHORITY:
        path = DOCS / name
        b.check_hash(path, digest)
        blob = subprocess.check_output(["git", "-C", str(REPO), "show", f"{commit}:{path.relative_to(REPO)}"])
        require(hashlib.sha256(blob).hexdigest() == digest, f"published authority changed: {name}")
        values[name] = d.read(path)
        if code_commit:
            require(subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, code_commit]).returncode == 0,
                    f"authority is not an ancestor: {commit}")
    closure = values[AUTHORITY[0][1]]
    prereg = values[AUTHORITY[1][1]]
    auth = values[AUTHORITY[2][1]]
    require(closure["formal_status"] == "BLOCKED_PROTOCOL_OR_LEAKAGE"
            and closure["V0_1_additional_attempts_authorized"] is False
            and closure["next_protocol"] == "PRES_DSR_SF_V0_2", "V0.1 closure changed")
    require(prereg["registration_id"] == auth["frozen_lineage"]["registration_id"] == REGISTRATION,
            "registration identity changed")
    require(auth["frozen_lineage"]["preregistration_commit"] == AUTHORITY[1][0]
            and auth["formal_attempt"]["count"] == 1, "authorization binding changed")
    return prereg


def source_gate(code_commit):
    observed = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    require(observed == code_commit, "execution HEAD differs from requested code commit")
    require(not subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip(),
            "execution checkout is dirty")
    changes = subprocess.check_output(["git", "-C", str(REPO), "diff", "--name-status", BASE_HEAD, code_commit],
                                      text=True).splitlines()
    require(changes and all(row.startswith("A\t") for row in changes), "historical tracked file changed")
    remote = subprocess.check_output(["git", "ls-remote", REMOTE, "refs/heads/" + BRANCH], text=True).split()
    require(remote and remote[0] == code_commit, "git ls-remote code barrier failed")
    url = "https://api.github.com/repos/DLwbm123/SSL_CL_seg/git/ref/heads/" + BRANCH
    with urlopen(Request(url + "?expected=" + code_commit,
                         headers={"User-Agent": "PRES-DSR-SF", "Cache-Control": "no-cache"}), timeout=30) as response:
        github = json.load(response)
    require(github["object"]["sha"] == code_commit, "GitHub API code barrier failed")
    authority(code_commit)
    return dict(branch=BRANCH, code_commit=code_commit, git_ls_remote_sha=remote[0],
                github_api_sha=github["object"]["sha"], checked_at=d.now())


def execution_gate(output, code_commit, test_report):
    require(socket.gethostname() == "zmic44" and os.getuid() == os.geteuid() == 1006, "wrong execution host/uid")
    output = Path(output).resolve()
    require(output.is_relative_to(NAS_ROOT) and not output.is_symlink(), "NAS-only non-symlink output required")
    require(os.environ.get("SSLCL_STORAGE_ROOT") == str(NAS_ROOT)
            and Path(os.environ.get("TMPDIR", "/")).resolve().is_relative_to(NAS_ROOT), "NAS wrapper/cache policy absent")
    publication = source_gate(code_commit)
    report = d.read(test_report)
    require(report["status"] == "PASS" and report["code_commit"] == code_commit
            and report["failures"] == report["errors"] == report["skips"] == 0, "test gate failed")
    b.check_hash(report["junit_path"], report["junit_sha256"])
    b.check_hash(report["pytest_output_path"], report["pytest_output_sha256"])
    return dict(registration_id=REGISTRATION, code_commit=code_commit, publication=publication,
                test_report=str(Path(test_report).resolve()), test_report_sha256=d.sha256(test_report),
                exact_test_command=report["exact_test_command"], hostname=socket.gethostname(), uid=os.getuid(),
                started_at=d.now(), model_optimizer_steps=0, model_autograd_calls=0,
                router_is_closed_form=True, method_registered=False, training_launched=False)


def freeze_backend(importer):
    initial = deterministic_backend_state()
    cuda_before = torch.cuda.is_initialized()
    with forbid_forwards():
        importer()
    require(torch.cuda.is_initialized() == cuda_before, "pinned import initialized CUDA", "BLOCKED_BACKEND_STATE_MUTATION")
    after_import = deterministic_backend_state()
    registered = enforce_deterministic_backend()
    require(all(registered.values()), "registered backend state incomplete", "BLOCKED_BACKEND_STATE_MUTATION")
    return dict(initial=initial, after_pinned_import=after_import, registered=registered,
                CUDA_initialized_before=cuda_before, CUDA_initialized_after_import=torch.cuda.is_initialized(),
                model_constructions_before_freeze=0, checkpoint_tensor_reads_before_freeze=0,
                private_HDF5_reads_before_freeze=0, model_forwards_before_freeze=0)


def backend_import_gate(output):
    reference = ROOT / "third_party/JASCL_REFERENCE"
    upstream = "Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT"
    audit = freeze_backend(lambda: _official_probabilistic_classifier(reference, upstream_path=upstream))
    audit.update(status="PASS_BACKEND_IMPORTED_THEN_REGISTERED", registered_at=d.now())
    d.write_new(Path(output) / "PRES_DSR_SF_BACKEND_AUDIT.json", audit)
    return audit


def require_backend(phase):
    state = deterministic_backend_state()
    require(all(state.values()), f"backend changed after {phase}", "BLOCKED_BACKEND_STATE_MUTATION")
    return dict(phase=phase, state=state, checked_at=d.now())


def isolation_guard():
    stack = ExitStack()
    stack.enter_context(b.no_updates())
    stack.enter_context(patch.object(torch.autograd, "grad", side_effect=Blocked("model autograd forbidden")))
    return stack


def gate1c_contract():
    return v1.gate1c_contract()


def input_audit(output, contract, metadata):
    output = Path(output)
    with forbid_forwards(), isolation_guard():
        try:
            private = v1._verify_private_bundle()
            data_root = Path(contract["destination"]["data_root"])
            b.check_hash(data_root / "checksums/checksums.sha256", contract["checksums_sha256"])
            data = verify_checksums(data_root)
            require(data["valid"] and data["entries"] == 2962, "2962 frozen checksums changed",
                    "BLOCKED_PRIVATE_INPUT_MISMATCH")
            b.check_hash(contract["baseline_manifest"]["path"], v1.BASELINE_MANIFEST_SHA)
            baseline = d.read(contract["baseline_manifest"]["path"])
            require(baseline["checkpoints"] == contract["immutable_baseline"]["checkpoint_inputs"],
                    "baseline/checkpoint contract mismatch", "BLOCKED_PRIVATE_INPUT_MISMATCH")
            checkpoints = []
            for checkpoint in contract["immutable_baseline"]["checkpoint_inputs"]:
                b.check_hash(checkpoint["path"], checkpoint["sha256"])
                payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
                verify_payload(payload)
                require(payload["stage_state"]["stage_index"] == checkpoint["stage_index"],
                        "checkpoint stage changed", "BLOCKED_PRIVATE_INPUT_MISMATCH")
                checkpoints.append(dict(checkpoint_id=checkpoint["checkpoint_id"], path=checkpoint["path"],
                                        sha256=checkpoint["sha256"], bytes=Path(checkpoint["path"]).stat().st_size,
                                        student_complete="student" in payload, ema_teacher_complete="ema_teacher" in payload))
            records, units = {}, []
            for seed in range(3):
                records[seed] = {}
                for stage in range(3):
                    records[seed][stage] = {}
                    for role in ("train_labeled", "train_unlabeled", "val"):
                        records[seed][stage][role] = b.records(data_root, contract, seed, stage, role)
                    units.append(dict(seed=seed, stage_index=stage, domain=DOMAINS[stage],
                                      **{role: len(records[seed][stage][role])
                                         for role in ("train_labeled", "train_unlabeled", "val")}))
            require(len(checkpoints) == 9 and sum(row["val"] for row in units) == 495
                    and sum(row["train_unlabeled"] for row in units) == 792, "input coverage changed",
                    "BLOCKED_INCOMPLETE_EVIDENCE")
        except Blocked as error:
            if error.status == "BLOCKED_PRIVATE_BUNDLE_MISMATCH":
                raise Blocked(str(error), "BLOCKED_PRIVATE_INPUT_MISMATCH") from error
            raise
        except Exception as error:
            raise Blocked(str(error), "BLOCKED_PRIVATE_INPUT_MISMATCH") from error
        result = dict(status="PASS_INPUT_AUDIT", metadata=metadata, private_bundle=private,
                      data_checksums=dict(entries=data["entries"], errors=data["errors"], valid=data["valid"],
                                          checksums_sha256=contract["checksums_sha256"]),
                      checkpoints=checkpoints, checkpoint_count=9, units=units, data_checksum_count=2962,
                      model_forwards=0, private_HDF5_decodes=0, segmentation_label_reads=0,
                      hidden_GT_usage="none", test_GT_usage="none")
        d.write_new(output / "PRES_DSR_SF_INPUT_AUDIT.json", result)
    return records


def compile_call_graph(output, records, code_commit):
    descriptor_cases = sum(len(records[s][stage][role]) for s in range(3) for stage in range(3)
                           for role in ("train_labeled", "train_unlabeled", "val"))
    per_seed = [sum(len(records[s][stage][role]) for stage in range(3)
                    for role in ("train_labeled", "train_unlabeled", "val")) for s in range(3)]
    validation_cases = sum(len(records[s][stage]["val"]) for s in range(3) for stage in range(3))
    seen_validation = sum(len(records[s][domain]["val"]) for s in range(3) for stage in (1, 2)
                          for domain in range(stage + 1))
    rows = dict(cv=78, router_scores=seen_validation, routing_confusion=117, cross_expert=27,
                soft_fusion=120, bootstrap=90, memory_cost=9)
    graph = dict(status="PASS_FROZEN_BEFORE_REAL_FORWARD", code_commit=code_commit, batch_size=BATCH_SIZE,
                 descriptor_forwards=sum(math.ceil(value / BATCH_SIZE) for value in per_seed),
                 descriptor_case_passes=descriptor_cases,
                 expert_probability_forwards=sum(math.ceil(sum(len(records[s][stage]["val"]) for stage in range(3))
                                                           / BATCH_SIZE) for s in range(3) for _ in range(3)),
                 expert_probability_case_passes=validation_cases * 3,
                 ridge_closed_form_fits=936, m1_cv_prototype_fits=75, clean_control_prototype_fits=168,
                 bootstrap_operations=30, clean_control_bootstrap_operations=60, model_guards=12,
                 formal_candidate_case_predictions=seen_validation * 8,
                 bootstrap_soft_case_predictions=seen_validation * 5,
                 validation_GT_case_reads=validation_cases, output_rows=rows, total_output_rows=sum(rows.values()),
                 model_optimizer_steps=0, model_autograd_calls=0, model_backward_calls=0, compiled_at=d.now())
    require(graph["expert_probability_forwards"] == 189 and graph["expert_probability_case_passes"] == 1485
            and graph["validation_GT_case_reads"] == 495 and graph["total_output_rows"] == 1356,
            "manifest call graph changed", "BLOCKED_INCOMPLETE_EVIDENCE")
    d.write_new(Path(output) / "PRES_DSR_SF_CALL_GRAPH.json", graph)
    return graph


def verify_call_graph(expected, observed):
    keys = ("descriptor_forwards", "descriptor_case_passes", "expert_probability_forwards",
            "expert_probability_case_passes", "ridge_closed_form_fits", "m1_cv_prototype_fits",
            "clean_control_prototype_fits", "bootstrap_operations", "clean_control_bootstrap_operations", "model_guards", "formal_candidate_case_predictions",
            "bootstrap_soft_case_predictions", "validation_GT_case_reads", "output_rows", "total_output_rows")
    require(all(observed[key] == expected[key] for key in keys), "executed call graph mismatch",
            "BLOCKED_INCOMPLETE_EVIDENCE")
