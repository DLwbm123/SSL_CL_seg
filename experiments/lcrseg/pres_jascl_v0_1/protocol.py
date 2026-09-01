"""Published authority, NAS admission, frozen-input audit and call graph."""
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
from di_dmpa_gate1c_v3 import durable as d
from di_dmpa_gate1c_v3.baseline import verify_payload
from di_dmpa_gate1c_v2.full_precision import forbid_forwards
from lcrseg.acceptance import verify_checksums

from . import REGISTRATION
from .core import Blocked, DOMAINS, require

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DOCS = ROOT / "docs/pres_jascl_v0_1"
GATE1C_DOCS = ROOT / "docs/di_dmpa_jascl"
BRANCH = "codex/pres-jascl-v0-1-routing-feasibility"
REMOTE = "https://github.com/DLwbm123/SSL_CL_seg.git"
NAS_ROOT = Path("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg")
PRIVATE_BUNDLE = NAS_ROOT / "gate1c_v3_clean_regeneration_20260831/runs/complete_v3_evidence_bundle"
GATE1C_REG = GATE1C_DOCS / "DI_DMPA_GATE1C_V3_PREREGISTRATION_SERIALIZATION_ERRATUM.json"
GATE1C_REG_SHA = "ad27b069121cfe41880a2da5429d8f9b688515d19942d14e2960212e4b192c06"
BASELINE_MANIFEST_SHA = "e6d3c833865a3c6773d28851c11e7da48b6496661eac34e34478ff01714a6f1d"
BATCH_SIZE = 8

AUTHORITY = (
    ("c003e13cb14ee1b9c14c6b445ff66c364e6c68b7", "PMGC_V0_1_FINAL_CLOSURE.json", "b180f728e9ea4fc9684f47d1e1f76e4fe8909955916e1be934fbdb94765372b1"),
    ("cd797d55362fd997beb6a9b7d5878aa790392831", "PRES_JASCL_V0_1_PREREGISTRATION.json", "d974613c6978586c7f410b24867b56370a9d1dbc0856ac9a1144342bf6082cd8"),
    ("8608353f0753a8a194799c9bc573f9f6962983c9", "PRES_JASCL_V0_1_EXECUTION_AUTHORIZATION.json", "dc5e00435d3caed176840efca513e19d4026738c0545cbdb6e836c3de5f58cd0"),
    ("238fae3f5db08bdba17bf81380227e01e17ef0ba", "PRES_JASCL_V0_1_PREREGISTRATION_AGGREGATION_CLARIFICATION.json", "d3ac4308de07401aede4b84e6de484b8c26566e50d41bac4af043113e488defa"),
    ("7e43519eb1945bbf433812cd831968c9602bec44", "PRES_JASCL_V0_1_EXECUTION_AUTHORIZATION_AGGREGATION_CLARIFICATION.json", "9f8e407adba171a9fb2e2ef44ebbaa79736962ec2bf873ee14064c276976adae"),
)


def authority(code_commit=None):
    values = {}
    for commit, name, digest in AUTHORITY:
        path = DOCS / name
        b.check_hash(path, digest)
        blob = subprocess.check_output(["git", "-C", str(REPO), "show", f"{commit}:{path.relative_to(REPO)}"])
        require(hashlib.sha256(blob).hexdigest() == digest, f"published authority changed: {name}")
        values[name] = d.read(path)
        if code_commit is not None:
            require(subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, code_commit]).returncode == 0,
                    f"authority is not an ancestor: {commit}")
    closure = values["PMGC_V0_1_FINAL_CLOSURE.json"]
    reg = values["PRES_JASCL_V0_1_PREREGISTRATION.json"]
    auth = values["PRES_JASCL_V0_1_EXECUTION_AUTHORIZATION.json"]
    clarify = values["PRES_JASCL_V0_1_PREREGISTRATION_AGGREGATION_CLARIFICATION.json"]
    clarify_auth = values["PRES_JASCL_V0_1_EXECUTION_AUTHORIZATION_AGGREGATION_CLARIFICATION.json"]
    require(closure["pmgc_status"] == "FAIL_PMGC_FEASIBILITY" and closure["shared_parameter_prototype_control_line"] == "ENDED",
            "PMGC was not closed")
    require(reg["registration_id"] == auth["frozen_lineage"]["registration_id"] == clarify["binding"]["registration_id"]
            == clarify_auth["registration_id"] == REGISTRATION, "registration identity mismatch")
    require(auth["frozen_lineage"]["preregistration_commit"] == AUTHORITY[1][0]
            and clarify_auth["clarification_commit"] == AUTHORITY[3][0], "authorization binding mismatch")
    require(all(value == 0 or value is False for value in reg["training_flags"].values()), "training flag enabled")
    return reg


def gate1c_contract():
    b.check_hash(GATE1C_REG, GATE1C_REG_SHA)
    p = d.read(GATE1C_REG)
    public_baseline = GATE1C_DOCS / "BASELINE_V3_CHECKPOINT_MANIFEST.json"
    b.check_hash(public_baseline, BASELINE_MANIFEST_SHA)
    require(p["baseline_manifest"]["sha256"] == BASELINE_MANIFEST_SHA, "Gate1C baseline manifest binding changed")
    require(len(p["immutable_baseline"]["checkpoint_inputs"]) == 9, "nine Gate1C B0 checkpoints required")
    return p


def source_gate(code_commit):
    observed = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    require(observed == code_commit, "execution HEAD differs from requested code commit")
    require(not subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip(),
            "execution checkout is dirty")
    changes = subprocess.check_output(["git", "-C", str(REPO), "diff", "--name-status",
                                       "06945fa738cf87a6cef22db949b627490e7847f1", code_commit], text=True).splitlines()
    require(changes and all(line.startswith("A\t") for line in changes), "historical tracked file changed")
    remote = subprocess.check_output(["git", "ls-remote", REMOTE, "refs/heads/" + BRANCH], text=True).split()
    require(remote and remote[0] == code_commit, "git ls-remote code barrier failed")
    url = "https://api.github.com/repos/DLwbm123/SSL_CL_seg/git/ref/heads/" + BRANCH
    request = Request(url + "?expected=" + code_commit, headers={"User-Agent": "PRES-JASCL", "Cache-Control": "no-cache"})
    with urlopen(request, timeout=30) as response:
        github = json.load(response)
    require(github["object"]["sha"] == code_commit, "GitHub API code barrier failed")
    authority(code_commit)
    return dict(code_commit=code_commit, git_ls_remote_sha=remote[0], github_api_sha=github["object"]["sha"],
                branch=BRANCH, checked_at=d.now())


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
                exact_test_command=report["exact_test_command"],
                hostname=socket.gethostname(), uid=os.getuid(), started_at=d.now(),
                model_optimizer_steps=0, router_optimizer_steps=0, autograd_calls=0,
                method_registered=False, training_launched=False)


def isolation_guard():
    stack = ExitStack()
    stack.enter_context(b.no_updates())
    stack.enter_context(patch.object(torch.autograd, "grad", side_effect=Blocked("autograd.grad forbidden")))
    return stack


def verify_bundle_at(path, *, manifest_sha256, content_sha256, files, bytes_):
    path = Path(path)
    try:
        manifest_path = path / "PRIVATE_BUNDLE_MANIFEST.json"
        require(d.sha256(manifest_path) == manifest_sha256,
                "private manifest SHA mismatch", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
        manifest = d.verify(path, exact=False)
        root_manifest = path / "PRIVATE_BUNDLE_MANIFEST.json"
        observed = {item.relative_to(path).as_posix() for item in path.rglob("*")
                    if item.is_file() and item != root_manifest}
        require(observed == {entry["path"] for entry in manifest["entries"]}, "private bundle path coverage mismatch",
                "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
        require(manifest["content_sha256"] == content_sha256 and manifest["files"] == files and manifest["bytes"] == bytes_,
                "private bundle content mismatch", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
    except Blocked:
        raise
    except Exception as error:
        raise Blocked(str(error), "BLOCKED_PRIVATE_BUNDLE_MISMATCH") from error
    return dict(root=str(path), content_sha256=manifest["content_sha256"],
                manifest_sha256=d.sha256(path / "PRIVATE_BUNDLE_MANIFEST.json"),
                files=manifest["files"], bytes=manifest["bytes"], every_file_sha256_verified=True)


def _verify_private_bundle():
    return verify_bundle_at(PRIVATE_BUNDLE,
                            manifest_sha256="480b627e0f63839ff5430d980020ca026c45838cf5eeb345f2b4cf7c4d578bb2",
                            content_sha256="8a82c7b8f0c72eb4faf619f51d7c1eae67a5f81059bc7f283b6b8df22d563526",
                            files=14470, bytes_=17712127650)


def input_audit(output, p, metadata):
    output = Path(output)
    with forbid_forwards(), isolation_guard():
        private = _verify_private_bundle()
        data_root = Path(p["destination"]["data_root"])
        b.check_hash(data_root / "checksums/checksums.sha256", p["checksums_sha256"])
        data = verify_checksums(data_root)
        require(data["valid"] and data["entries"] == 2962, "2962 frozen checksums changed",
                "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
        b.check_hash(p["baseline_manifest"]["path"], BASELINE_MANIFEST_SHA)
        baseline = d.read(p["baseline_manifest"]["path"])
        require(baseline["checkpoints"] == p["immutable_baseline"]["checkpoint_inputs"], "baseline/checkpoint contract mismatch",
                "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
        checkpoints = []
        for checkpoint in p["immutable_baseline"]["checkpoint_inputs"]:
            b.check_hash(checkpoint["path"], checkpoint["sha256"])
            payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
            verify_payload(payload)
            require(payload["stage_state"]["stage_index"] == checkpoint["stage_index"], "checkpoint stage mismatch",
                    "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
            require(set(("student", "ema_teacher", "prototypes")).issubset(payload), "checkpoint tensor schema incomplete",
                    "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
            checkpoints.append(dict(checkpoint_id=checkpoint["checkpoint_id"], path=checkpoint["path"],
                                    sha256=checkpoint["sha256"], bytes=Path(checkpoint["path"]).stat().st_size,
                                    student_complete=True, ema_teacher_complete=True, direct_PAS_complete=True))
        records = {}
        units = []
        for seed in range(3):
            records[seed] = {}
            for stage in range(3):
                records[seed][stage] = {}
                for role in ("train_unlabeled", "val"):
                    rows = b.records(data_root, p, seed, stage, role)
                    records[seed][stage][role] = rows
                units.append(dict(seed=seed, stage_index=stage, domain=DOMAINS[stage],
                                  train_unlabeled=len(records[seed][stage]["train_unlabeled"]),
                                  val=len(records[seed][stage]["val"])))
        require(len(checkpoints) == 9 and sum(u["val"] for u in units) == 495
                and sum(u["train_unlabeled"] for u in units) == 792, "input coverage mismatch",
                "BLOCKED_INCOMPLETE_EVIDENCE")
        result = dict(metadata=metadata, status="PASS_INPUT_AUDIT", private_bundle=private,
                      data_checksums=dict(entries=data["entries"], errors=data["errors"], valid=data["valid"],
                                          checksums_sha256=p["checksums_sha256"]),
                      checkpoints=checkpoints, units=units, checkpoint_count=9, data_checksum_count=2962,
                      decoded_private_HDF5=0, model_forwards=0, test_role_constructions=0,
                      hidden_GT_usage="none", test_GT_usage="none")
        d.write_new(output / "PRES_JASCL_INPUT_AUDIT.json", result)
    return records


def compile_call_graph(output, records, code_commit):
    descriptor_cases = sum(len(records[seed][stage][role]) for seed in range(3)
                           for stage in range(3) for role in ("train_unlabeled", "val"))
    validation_cases = sum(len(records[seed][stage]["val"]) for seed in range(3) for stage in range(3))
    router_forwards = sum(math.ceil(sum(len(records[seed][stage][role]) for stage in range(3)
                                        for role in ("train_unlabeled", "val")) / BATCH_SIZE) for seed in range(3))
    segmentation_forwards = sum(math.ceil(sum(len(records[seed][stage]["val"]) for stage in range(3)) / BATCH_SIZE)
                                for seed in range(3) for _expert in range(3))
    rows = dict(router_scores=sum(2 * (len(records[seed][0]["val"]) + len(records[seed][1]["val"])
                                      + sum(len(records[seed][stage]["val"]) for stage in range(3))) for seed in range(3)),
                router_confusion=3 * 2 * (2**2 + 3**2), router_bootstrap=3 * 2 * 2 * 5,
                cross_expert=3 * 3 * 3, oracle_vs_routed=3 * 3 * 4)
    graph = dict(status="PASS_FROZEN_BEFORE_REAL_FORWARD", code_commit=code_commit, batch_size=BATCH_SIZE,
                 router_extraction_forwards=router_forwards, router_extraction_case_passes=descriptor_cases,
                 cross_expert_segmentation_forwards=segmentation_forwards,
                 cross_expert_segmentation_case_passes=validation_cases * 3,
                 bootstrap_operations=3 * 2 * 2 * 5, model_guards=12, output_rows=rows,
                 total_output_rows=sum(rows.values()), model_optimizer_steps=0, router_optimizer_steps=0,
                 autograd_calls=0, backward_calls=0, compiled_at=d.now())
    require(graph["router_extraction_forwards"] == 162 and graph["router_extraction_case_passes"] == 1287
            and graph["cross_expert_segmentation_forwards"] == 189
            and graph["cross_expert_segmentation_case_passes"] == 1485
            and graph["total_output_rows"] == 2031, "manifest call graph differs", "BLOCKED_CALL_GRAPH_MISMATCH")
    d.write_new(Path(output) / "PRES_JASCL_CALL_GRAPH.json", graph)
    return graph


def verify_call_graph(expected, observed):
    keys = ("router_extraction_forwards", "router_extraction_case_passes", "cross_expert_segmentation_forwards",
            "cross_expert_segmentation_case_passes", "bootstrap_operations", "model_guards", "output_rows", "total_output_rows")
    require(all(observed[key] == expected[key] for key in keys), "executed call graph mismatch", "BLOCKED_CALL_GRAPH_MISMATCH")
