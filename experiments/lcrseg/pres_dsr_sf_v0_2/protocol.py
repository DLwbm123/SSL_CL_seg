"""Authority, backend barrier, NAS admission, private-input audit, and call graph."""
from __future__ import annotations

import ast
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
from .core import LAMBDAS, TEMPERATURES

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DOCS = ROOT / "docs/pres_dsr_sf_v0_2_1"
BRANCH = "codex/pres-dsr-sf-v0-2-1-callgraph-recovery"
REMOTE = "https://github.com/DLwbm123/SSL_CL_seg.git"
BASE_HEAD = "ff42db2ec2381aad176139ab788a9925eef9d147"
NAS_ROOT = Path("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg")
BATCH_SIZE = 8

AUTHORITY = (
    ("de82bf94f27f91e071f9bab4e9432f1c0ee263d3", "PRES_DSR_SF_V0_2_FORMAL01_CLOSURE.json",
     "1620cde0a273b00cd3333097ae5110446a3f37fc062886aed9161c261667dd65"),
    ("44a8870765d1ebb5efa38843f3c20b79aeb721ec", "PRES_DSR_SF_V0_2_CALLGRAPH_ERRATUM.json",
     "ee36a2a66aeb1beb4d3394a3874d86607a9cb82819f6a0a8f4f9dc433604da55"),
    ("752e1ac7a016d619ffaa624c347fbeefa7883137", "PRES_DSR_SF_V0_2_1_PREREGISTRATION.json",
     "28680d9f4b2d989d54ef4df969b6c5c1a84319782c384d6bb1c6bd9a52973efb"),
    ("1eaf16c876a180fc9eaff6fc893e134d10518d02", "PRES_DSR_SF_V0_2_1_EXECUTION_AUTHORIZATION.json",
     "3846b81763e3e974ecb850bd257a1fcd9414a16a3081d375aa1e1da7332d232d"),
)

SCIENCE_FUNCTIONS = ("raw_style_block", "raw_style_descriptors", "fit_standardizer", "apply_standardizer",
                     "ridge_fit", "fit_router", "router_probabilities", "hard_routes", "probability_fusion",
                     "bootstrap_multiplicity", "adjudicate")
ALLOWED_MODIFIED = {
    "experiments/lcrseg/pres_dsr_sf_v0_2/__init__.py",
    "experiments/lcrseg/pres_dsr_sf_v0_2/protocol.py",
    "experiments/lcrseg/pres_dsr_sf_v0_2/run.py",
    "experiments/lcrseg/pres_dsr_sf_v0_2/testing.py",
    "experiments/lcrseg/pres_dsr_sf_v0_2/postflight.py",
    "experiments/lcrseg/tests/pres_dsr_sf_v0_2/test_protocol.py",
}


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
    closure, erratum, prereg, auth = (values[row[1]] for row in AUTHORITY)
    require(closure["formal_status"] == "BLOCKED_INCOMPLETE_EVIDENCE"
            and closure["scientific_status"] == "NOT_ADJUDICATED"
            and closure["old_attempt_additional_run_authorized"] is False
            and closure["next_protocol"] == "PRES_DSR_SF_V0_2_1_CALLGRAPH_RECOVERY", "V0.2 closure changed")
    require(erratum["ridge_local"]["total_rows"] == 54
            and erratum["combined_cv"]["total_rows"] == 78
            and erratum["total_output_rows"] == 1356, "call-graph erratum changed")
    require(prereg["registration_id"] == auth["authorized_protocol"] == REGISTRATION,
            "registration identity changed")
    require(auth["lineage"]["preregistration_commit"] == AUTHORITY[2][0]
            and auth["formal_execution"]["count"] == 1
            and auth["formal_execution"]["additional_attempts_authorized"] is False,
            "authorization binding changed")
    return prereg


def science_source_audit(preregistration):
    path = ROOT / "pres_dsr_sf_v0_2/core.py"
    text = path.read_text()
    nodes = {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    observed = {}
    for name in SCIENCE_FUNCTIONS:
        require(name in nodes, f"missing frozen science function: {name}", "BLOCKED_SCIENCE_SOURCE_CHANGED")
        node = nodes[name]
        source = "".join(text.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]).encode()
        observed[name] = {
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "ast_sha256": hashlib.sha256(ast.dump(node, annotate_fields=True,
                                                    include_attributes=False).encode()).hexdigest(),
        }
    expected = preregistration["frozen_science"]["functions"]
    require(all(observed[name]["source_sha256"] == expected[name]["source_sha256"] for name in SCIENCE_FUNCTIONS),
            "frozen scientific function changed", "BLOCKED_SCIENCE_SOURCE_CHANGED")
    source_digest = hashlib.sha256("\n".join(name + ":" + observed[name]["source_sha256"]
                                              for name in SCIENCE_FUNCTIONS).encode()).hexdigest()
    ast_digest = hashlib.sha256("\n".join(name + ":" + observed[name]["ast_sha256"]
                                           for name in SCIENCE_FUNCTIONS).encode()).hexdigest()
    return dict(status="PASS_FROZEN_SCIENCE_SOURCE", reference_code_commit="09f4600348f8708ca9e865f7d5c925b6472cd013",
                comparison="byte-equivalent", functions=observed, combined_source_sha256=source_digest,
                runtime_ast_sha256=ast_digest)


def source_gate(code_commit):
    observed = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    require(observed == code_commit, "execution HEAD differs from requested code commit")
    require(not subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip(),
            "execution checkout is dirty")
    changes = subprocess.check_output(["git", "-C", str(REPO), "diff", "--name-status", BASE_HEAD, code_commit],
                                      text=True).splitlines()
    require(changes, "V0.2.1 source delta is empty")
    for row in changes:
        status, path = row.split("\t", 1)
        allowed = (status == "A" and path.startswith("experiments/lcrseg/docs/pres_dsr_sf_v0_2_1/")) or (
            status == "M" and path in ALLOWED_MODIFIED)
        require(allowed, f"unregistered V0.2.1 source change: {row}", "BLOCKED_PROTOCOL_OR_LEAKAGE")
    remote = subprocess.check_output(["git", "ls-remote", REMOTE, "refs/heads/" + BRANCH], text=True).split()
    require(remote and remote[0] == code_commit, "git ls-remote code barrier failed")
    url = "https://api.github.com/repos/DLwbm123/SSL_CL_seg/git/ref/heads/" + BRANCH
    with urlopen(Request(url + "?expected=" + code_commit,
                         headers={"User-Agent": "PRES-DSR-SF", "Cache-Control": "no-cache"}), timeout=30) as response:
        github = json.load(response)
    require(github["object"]["sha"] == code_commit, "GitHub API code barrier failed")
    preregistration = authority(code_commit)
    science = science_source_audit(preregistration)
    return dict(branch=BRANCH, code_commit=code_commit, git_ls_remote_sha=remote[0],
                github_api_sha=github["object"]["sha"], science_source=science, checked_at=d.now())


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
    d.write_new(Path(output) / "PRES_DSR_SF_V0_2_1_BACKEND_AUDIT.json", audit)
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
        d.write_new(output / "PRES_DSR_SF_V0_2_1_INPUT_AUDIT.json", result)
    return records


def output_key_plan(output, records, code_commit):
    seeds, stages = tuple(range(3)), (1, 2)
    m1 = [("M1_temperature", "M1", seed, stage, "temperature", temperature)
          for seed in seeds for stage in stages for temperature in TEMPERATURES]
    ridge_lambda = [("ridge_lambda", "ridge", seed, stage, "lambda", value)
                    for seed in seeds for stage in stages for value in LAMBDAS]
    ridge_temperature = [("ridge_temperature", "ridge", seed, stage, "temperature", temperature)
                         for seed in seeds for stage in stages for temperature in TEMPERATURES]
    router_scores = [(seed, stage, row["case_id"]) for seed in seeds for stage in stages
                     for domain in range(stage + 1) for row in records[seed][domain]["val"]]
    confusion = [(seed, stage, router, true, routed) for seed in seeds for stage in stages
                 for router in ("M1_HARD", "M2_HARD", "RIDGE_HARD")
                 for true in range(stage + 1) for routed in range(stage + 1)]
    cross_expert = [(seed, domain, expert) for seed in seeds for domain in range(3) for expert in range(3)]
    soft_fusion = [(seed, stage, domain, policy) for seed in seeds for stage in stages
                   for domain in range(stage + 1)
                   for policy in ("C0_SHARED", "C1_ORACLE", "C2_M1_HARD", "C3_M2_HARD",
                                  "C4_M1_SOFT", "C5_RIDGE_HARD", "C6_RIDGE_SOFT", "C7_UNIFORM")]
    bootstrap = [("clean_control", seed, stage, M, replicate) for seed in seeds for M in (1, 2)
                 for stage in stages for replicate in range(5)]
    bootstrap += [("ridge", seed, stage, replicate) for seed in seeds for stage in stages for replicate in range(5)]
    memory = [(seed, domain) for seed in seeds for domain in range(3)]
    key_sets = dict(M1_temperature=m1, ridge_lambda=ridge_lambda, ridge_temperature=ridge_temperature,
                    router_scores=router_scores, routing_confusion=confusion, cross_expert=cross_expert,
                    soft_fusion=soft_fusion, bootstrap=bootstrap, memory=memory)
    counts = {name: len(keys) for name, keys in key_sets.items()}
    require(counts == dict(M1_temperature=24, ridge_lambda=30, ridge_temperature=24, router_scores=915,
                           routing_confusion=117, cross_expert=27, soft_fusion=120, bootstrap=90, memory=9),
            "declarative key cardinality changed", "BLOCKED_CALLGRAPH_CARDINALITY_MISMATCH")
    result = dict(status="PASS_DECLARATIVE_OUTPUT_KEYS_BEFORE_FORWARD", code_commit=code_commit,
                  generated_from="frozen grids, stage structure, and input manifest records only",
                  key_sets=key_sets, counts=counts,
                  content_sha256=hashlib.sha256(d.canonical(key_sets)).hexdigest(), created_at=d.now())
    d.write_new(Path(output) / "PRES_DSR_SF_V0_2_1_OUTPUT_KEY_PLAN.json", result)
    return result


def compile_call_graph(output, records, code_commit, plan=None):
    plan = plan or output_key_plan(output, records, code_commit)
    seeds, stages = tuple(range(3)), (1, 2)
    descriptor_cases = sum(len(records[s][stage][role]) for s in range(3) for stage in range(3)
                           for role in ("train_labeled", "train_unlabeled", "val"))
    per_seed = [sum(len(records[s][stage][role]) for stage in range(3)
                    for role in ("train_labeled", "train_unlabeled", "val")) for s in range(3)]
    validation_cases = sum(len(records[s][stage]["val"]) for s in range(3) for stage in range(3))
    seen_validation = sum(len(records[s][domain]["val"]) for s in range(3) for stage in (1, 2)
                          for domain in range(stage + 1))
    router_units = len(seeds) * len(stages)
    bootstrap_operations = router_units * 5
    rows = dict(cv=sum(plan["counts"][name] for name in ("M1_temperature", "ridge_lambda", "ridge_temperature")),
                router_scores=plan["counts"]["router_scores"], routing_confusion=plan["counts"]["routing_confusion"],
                cross_expert=plan["counts"]["cross_expert"], soft_fusion=plan["counts"]["soft_fusion"],
                bootstrap=plan["counts"]["bootstrap"], memory_cost=plan["counts"]["memory"])
    graph = dict(status="PASS_FROZEN_BEFORE_REAL_FORWARD", code_commit=code_commit, batch_size=BATCH_SIZE,
                 descriptor_forwards=sum(math.ceil(value / BATCH_SIZE) for value in per_seed),
                 descriptor_case_passes=descriptor_cases,
                 expert_probability_forwards=sum(math.ceil(sum(len(records[s][stage]["val"]) for stage in range(3))
                                                           / BATCH_SIZE) for s in range(3) for _ in range(3)),
                 expert_probability_case_passes=validation_cases * 3,
                 total_model_forwards=sum(math.ceil(value / BATCH_SIZE) for value in per_seed)
                 + sum(math.ceil(sum(len(records[s][stage]["val"]) for stage in range(3)) / BATCH_SIZE)
                       for s in seeds for _ in range(3)),
                 ridge_closed_form_fits=(router_units + bootstrap_operations) * (len(LAMBDAS) * 5 + 1),
                 m1_cv_prototype_fits=len(seeds) * sum(stage + 1 for stage in stages) * 5,
                 clean_control_prototype_fits=(len(seeds) * 2 * 3
                                               + len(seeds) * 2 * sum(stage + 1 for stage in stages) * 5),
                 bootstrap_operations=bootstrap_operations,
                 clean_control_bootstrap_operations=len(seeds) * 2 * len(stages) * 5, model_guards=12,
                 formal_candidate_case_predictions=seen_validation * 8,
                 bootstrap_soft_case_predictions=seen_validation * 5,
                 validation_GT_case_reads=validation_cases, output_rows=rows, total_output_rows=sum(rows.values()),
                 model_optimizer_steps=0, model_autograd_calls=0, model_backward_calls=0, compiled_at=d.now())
    frozen = dict(descriptor_case_passes=1485, descriptor_forwards=186, expert_probability_case_passes=1485,
                  expert_probability_forwards=189, total_model_forwards=375, model_guards=12,
                  validation_GT_case_reads=495, ridge_closed_form_fits=936, m1_cv_prototype_fits=75,
                  clean_control_prototype_fits=168, bootstrap_operations=30,
                  clean_control_bootstrap_operations=60, total_output_rows=1356)
    require(all(graph[key] == value for key, value in frozen.items()),
            "manifest call graph changed", "BLOCKED_CALLGRAPH_CARDINALITY_MISMATCH")
    d.write_new(Path(output) / "PRES_DSR_SF_V0_2_1_CALL_GRAPH.json", graph)
    return graph


def verify_call_graph(expected, observed):
    keys = ("descriptor_forwards", "descriptor_case_passes", "expert_probability_forwards",
            "expert_probability_case_passes", "ridge_closed_form_fits", "m1_cv_prototype_fits",
            "clean_control_prototype_fits", "bootstrap_operations", "clean_control_bootstrap_operations", "model_guards", "formal_candidate_case_predictions",
            "bootstrap_soft_case_predictions", "validation_GT_case_reads", "output_rows", "total_output_rows")
    require(all(observed[key] == expected[key] for key in keys), "executed call graph mismatch",
            "BLOCKED_CALLGRAPH_CARDINALITY_MISMATCH")


def phase_barrier(output, name, artifacts):
    output = Path(output).resolve()
    require(name in ("input_audit", "descriptor_seal", "memory_seal", "clean_control_seal",
                     "ridge_router_seal", "combined_cv_seal", "expert_probability_seal",
                     "candidate_prediction_seal", "validation_evaluation", "bootstrap_evaluation",
                     "E1_E6_compile", "artifact_audit", "NAS_archive", "report"),
            "unregistered phase", "BLOCKED_PROTOCOL_OR_LEAKAGE")
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
