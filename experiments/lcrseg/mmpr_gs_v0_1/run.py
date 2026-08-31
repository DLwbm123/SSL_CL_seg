"""Create-only NAS diagnostics; published authority, durable exits, no retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from di_dmpa_gate1c_v3 import durable as d
from di_dmpa_gate1c_v3.inputs import load_models
from di_dmpa_gate1c_v3.baseline import verify_payload
from di_dmpa_gate1c_v2 import binding as b, execution as e, reliability as r
from di_dmpa_gate1c_v2.precision import attach_gradient_student
from di_dmpa_gate1c_v2.full_precision import forbid_forwards
from di_dmpa_gate1_v2.features import ImmutableModels
from lcrseg.acceptance import verify_checksums
from . import REGISTRATION
from .core import require, Blocked, mass_match
from .diagnostic import compute_pair, LIMITS, ROW_COUNTS
from .evaluator import evaluate_case, aggregate
from .report import csv_new, adjudicate

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DOCS = ROOT / "docs/di_dmpa_jascl"
REG_COMMIT = "13494e175a2f5cd9a262c03c22d3ca45bfda7619"
AUTH_COMMIT = "639bf974383b8d11d490902ea4a7d73e4a89ba25"
REG_SHA = "8dcd6091613150abbdf6b3621de921f2988475bee95f3248449353c04aa11018"
AUTH_SHA = "cc564755ec0a8947109a80470f242f54badba8285fc30342fa1741338738c096"


def authority():
    paths = [(REG_COMMIT, "MMPR_GS_V0_1_FEASIBILITY_PREREGISTRATION.json", REG_SHA),
             (AUTH_COMMIT, "MMPR_GS_V0_1_EXECUTION_AUTHORIZATION.json", AUTH_SHA)]
    for commit, name, digest in paths:
        b.check_hash(DOCS / name, digest)
        blob = subprocess.check_output(["git", "-C", str(REPO), "show", commit + ":" + str((DOCS/name).relative_to(REPO))])
        require(hashlib.sha256(blob).hexdigest() == digest, "published authority blob changed")
    reg, auth = (d.read(DOCS / name) for _, name, _ in paths)
    require(reg["registration_id"] == auth["registration_id"] == REGISTRATION and
            auth["preregistration_commit"] == auth["preregistration_remote_verified_commit"] == REG_COMMIT,
            "independent registration binding")
    b.check_hash(REPO / reg["closure"]["json_path"], reg["closure"]["json_sha256"])
    closure = d.read(REPO / reg["closure"]["json_path"])
    require(not closure["DI_DMPA_additional_attempts_authorized"] and closure["reduced_candidate"] == "NONE" and
            not closure["current_only_reliability_admission"], "old line reopened")
    for name, digest in reg["frozen_source_sha256"].items():
        b.check_hash(REPO / name, digest)
    inherited = REPO / reg["inherited_numeric_inputs"]["path"]
    b.check_hash(inherited, reg["inherited_numeric_inputs"]["sha256"])
    p = d.read(inherited)
    require(b.H(p["gradient_diagnostic"]["batch_pairs"]) == reg["inputs"]["fixed_batch_pairs_sha256"], "fixed pairs changed")
    require(reg["call_graph"]["per_pair"] == dict(LIMITS, total_forwards=5, **ROW_COUNTS), "compiled budget differs from registration", "BLOCKED_CALL_GRAPH_MISMATCH")
    return reg, p


def execution_gate(args, reg):
    dest = reg["destination"]
    require(socket.gethostname() == dest["hostname"] and os.getuid() == os.geteuid() == dest["uid"], "wrong server identity")
    require(sys.executable == reg["runtime"]["python_executable"] and str(torch.__version__) == reg["runtime"]["torch"]
            and torch.version.cuda == reg["runtime"]["cuda"], "runtime changed")
    code = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    require(code == args.code_commit, "wrong exact execution commit")
    require(not subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], text=True).strip(), "dirty tracked source")
    changes = subprocess.check_output(["git", "-C", str(REPO), "diff", "--name-status", reg["base_commit"], code], text=True).splitlines()
    require(changes and all(line.startswith("A\t") for line in changes), "historical tracked file changed")
    root = Path(dest["root"])
    require(root.resolve().is_relative_to(Path("/data_nas")) and not root.is_symlink(), "NAS-only root required")
    require(os.environ.get("SSLCL_STORAGE_ROOT") == "/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg" and
            Path(os.environ.get("TMPDIR", "/")).resolve().is_relative_to(Path("/data_nas")), "NAS wrapper/cache policy absent")
    require(args.output.resolve() == Path(dest["phase_roots"][args.phase]) and not args.output.is_symlink(), "unregistered phase output")
    gate = d.read(args.gate)
    require(gate["authorization_remote_verified_commit"] == AUTH_COMMIT and gate["preregistration_remote_verified_commit"] == REG_COMMIT and
            gate["code_commit"] == gate["remote_verified_code_commit"] == code, "remote publication gate incomplete")
    for path, digest in gate["exact_source_sha256"].items():
        b.check_hash(REPO / path, digest)
    tests = d.read(gate["test_report"])
    b.check_hash(gate["test_report"], gate["test_report_sha256"])
    b.check_hash(tests["junit_path"], tests["junit_sha256"])
    require(tests["code_commit"] == code and tests["failures"] == tests["errors"] == tests["skips"] == 0 and
            tests["status"] == "PASS", "test prerequisite failed")
    compiler = d.read(gate["call_graph"])
    b.check_hash(gate["call_graph"], gate["call_graph_sha256"])
    require(compiler["code_commit"] == code and compiler["per_pair"] == reg["call_graph"]["per_pair"] and
            compiler["status"] == "PASS", "exact call graph not frozen", "BLOCKED_CALL_GRAPH_MISMATCH")
    return dict(registration_id=REGISTRATION, preregistration_commit=REG_COMMIT, authorization_commit=AUTH_COMMIT,
                code_commit=code, publication_gate_sha256=d.sha256(args.gate), phase=args.phase,
                exact_command=sys.argv, started_at=d.now(), hostname=socket.gethostname(), uid=os.getuid(),
                model_optimizer_steps=0, transport_optimizer_steps=0, method_registered=False, training_launched=False)


def phase_completed(path, report):
    exit_receipt = d.read(Path(path) / "PROCESS_EXIT.json")
    require(exit_receipt["actual_child_exit_code"] == 0, "previous phase real process failed", "BLOCKED_INCOMPLETE_EVIDENCE")
    value = d.read(Path(path) / report)
    require(value["status"].startswith("PASS"), "previous phase did not pass", "BLOCKED_INCOMPLETE_EVIDENCE")
    return value


def verify_bundle(reg):
    spec = reg["private_bundle"]
    path = Path(spec["root"])
    try:
        require(d.sha256(path/spec["manifest_filename"]) == spec["manifest_sha256"], "private manifest digest")
        manifest = d.verify(path, filename=spec["manifest_filename"])
        require(manifest["content_sha256"] == spec["content_sha256"] and manifest["files"] == spec["logical_files"] and
                manifest["bytes"] == spec["logical_bytes"], "private content/count/bytes")
    except Exception as error:
        raise Blocked(str(error), "BLOCKED_PRIVATE_BUNDLE_MISMATCH") from error
    return dict(status="PASS_PRIVATE_BUNDLE", root=str(path), logical_files=manifest["files"], logical_bytes=manifest["bytes"],
                content_sha256=manifest["content_sha256"], manifest_sha256=d.sha256(path/spec["manifest_filename"]),
                every_file_SHA_verified=True, exact_path_coverage=True, checked_at=d.now())


def input_audit(args, reg, p):
    with forbid_forwards(), b.no_updates():
        audit = verify_bundle(reg)
        d.write_new(args.output / "MMPR_GS_PRIVATE_BUNDLE_AUDIT.json", audit)
        checks = verify_checksums(Path(reg["inputs"]["data_root"]))
        require(checks["valid"] and checks["entries"] == 2962, "frozen data checksum mismatch")
        b.check_hash(Path(reg["inputs"]["data_root"])/"checksums/checksums.sha256", reg["inputs"]["checksums_sha256"])
        for spec in (reg["inputs"]["baseline_manifest"], reg["inputs"]["k2_freeze"]):
            b.check_hash(spec["path"], spec["sha256"])
        rows = []
        for cp in p["immutable_baseline"]["checkpoint_inputs"]:
            b.check_hash(cp["path"], cp["sha256"])
            payload = torch.load(cp["path"], map_location="cpu", weights_only=False)
            verify_payload(payload)
            legacy = b.legacy_input(payload, cp, p, "cpu")
            require(b.tensor_hash(legacy) == cp["legacy_pas_tensor_sha256"], "direct checkpoint PAS identity")
            rows.append(dict(checkpoint_id=cp["checkpoint_id"], path=cp["path"], sha256=cp["sha256"],
                             legacy_pas_sha256=b.tensor_hash(legacy), direct_PAS=True, checkpoint_payload_verified=True))
        role_rows = []
        for seed in range(3):
            for stage in range(3):
                role_rows.append(dict(seed=seed, stage_index=stage, counts={role: len(b.records(reg["inputs"]["data_root"], p, seed, stage, role))
                                                                          for role in ("train_labeled", "train_unlabeled", "val")}))
        require(len(rows) == 9 and sum(u["counts"]["val"] for u in role_rows) == 495, "9 checkpoints/495 cases required")
        result = dict(status="PASS_INPUT_AUDIT", checkpoints=rows, role_units=role_rows, data_checksums=checks,
                      fixed_batch_pairs_sha256=reg["inputs"]["fixed_batch_pairs_sha256"], private_bundle=audit,
                      hidden_gt_training_usage="none", test_gt_usage="none", model_forwards=0, new_cache_generation=0)
        d.write_new(args.output / "MMPR_GS_INPUT_AUDIT.json", result)


def check_cache(case, scores, raw, current, history):
    e.validate_scores(scores, case["stage_index"], 384 * 384)
    rebuilt = r.score_arrays(raw["teacher_probability"].transpose(0, 2, 3, 1).reshape(-1, 3),
                             raw["teacher_features"].transpose(0, 2, 3, 1).reshape(-1, 16),
                             scores["R1"], current, history)
    for key in e.CACHE_FIELDS:
        require(np.array_equal(scores[key], rebuilt[key], equal_nan=True), "original cache score changed: " + key)
    for source in ("student", "teacher"):
        validity = (raw[source+"_pas_confidence"] > .7) & (raw[source+"_pas_similarity"] > .7)
        require(np.array_equal(validity, raw[source+"_pas_valid_mask"]), "strict >0.7 PAS changed")
    require(np.array_equal(scores["R1"], (raw["student_pas_valid_mask"] & raw["teacher_pas_valid_mask"]).reshape(-1)), "cached joint PAS changed")
    require(np.array_equal(raw["null_mask"], ~scores["active_mask"]), "cached null mask changed")
    return dict(original_R3_unrounded_exact=True, original_R2_unrounded_exact=True, direct_R1_parity=True, GT_in_builder=False)


def validation(args, reg, p):
    phase_completed(reg["destination"]["phase_roots"]["input_audit"], "MMPR_GS_INPUT_AUDIT.json")
    source = Path(reg["inputs"]["validation_root"])
    cache_manifest = d.read(source/"VALIDATION_CACHE_V3_MANIFEST.json")
    freeze = d.read(reg["inputs"]["k2_freeze"]["path"])
    masks_root = args.output / "masks"
    masks_root.mkdir()
    indices, mass, selection = [], [], []
    with forbid_forwards(), b.no_updates():
        for unit in cache_manifest["units"]:
            seed, stage = unit["seed"], unit["stage_index"]
            expected = next(u for u in p["validation"]["plans"] if (u["seed"], u["stage_index"]) == (seed, stage))
            require([c["case_id"] for c in unit["cases"]] == [c["case_id"] for c in expected["cases"]], "cache case plan/order changed")
            current, history = r.banks(freeze, seed, stage)
            for case, planned in zip(unit["cases"], expected["cases"]):
                require(all(case[k] == planned[k] for k in ("case_id", "student_seed", "teacher_draw0_seed")), "cache draw seeds changed")
                scores, raw = b.read_arrays(case["arrays"]), b.read_arrays(case["arrays"]["raw_values"])
                checked = check_cache(dict(case, stage_index=stage), scores, raw, current, history)
                pred = scores["teacher_probability"].argmax(1)
                result = {}
                for candidate, rank in (("Q1", "R3"), ("Q2", "R2")):
                    result[candidate], rows = mass_match(scores[rank], pred, scores["active_mask"], scores["R1"],
                                                         seed=seed, stage=stage, cases=[case["case_id"]])
                    if candidate == "Q1":
                        mass.extend(rows)
                    selection.extend(dict(row, candidate=candidate) for row in rows)
                name = f"seed{seed}_stage{stage}_" + case["case_id"]
                desc = b.save_arrays(masks_root / (name + ".npz"), result)
                indices.append(dict(seed=seed, stage_index=stage, case_id=case["case_id"], source=case["arrays"], masks=desc,
                                    source_parity=checked, R1_sha256=b.array_hash(scores["R1"]), R3_sha256=b.array_hash(scores["R3"])))
            print(json.dumps(dict(event="selection_unit_complete", seed=seed, stage=stage, cases=len(unit["cases"]))), flush=True)
        require(len(indices) == 495 and len(mass) == 1485, "incomplete selection", "BLOCKED_INCOMPLETE_EVIDENCE")
        d.write_new(masks_root/"MASK_INDEX.json", dict(cases=indices, mask_count=495, GT_reads=0, model_forwards=0))
        seal = d.seal(masks_root)
        d.write_new(args.output/"MMPR_GS_MASK_SEAL.json", dict(content_sha256=seal["content_sha256"],
                                                              manifest_sha256=d.sha256(masks_root/"PRIVATE_BUNDLE_MANIFEST.json"),
                                                              GT_reads_before_seal=0, model_forwards=0))
        # This is the first validation GT access. No selection function runs below.
        case_rows, changes, regions = [], [], []
        for seed in range(3):
            for stage in range(3):
                records = b.records(reg["inputs"]["data_root"], p, seed, stage, "val")
                cases = [i for i in indices if (i["seed"], i["stage_index"]) == (seed, stage)]
                require([x["case_id"] for x in cases] == [x["case_id"] for x in records], "evaluator case order")
                for info, record in zip(cases, records):
                    scores, masks = b.read_arrays(info["source"]), b.read_arrays(info["masks"])
                    labels = e.visible_labels([record], reg["inputs"]["data_root"], role="val")[0]
                    rows, cr, br = evaluate_case(scores, dict(masks, Q0=scores["R1"], Q3=scores["R3"]), labels,
                                                seed=seed, stage=stage, case=info["case_id"])
                    case_rows.extend(rows)
                    changes.extend(cr)
                    regions.extend(br)
                print(json.dumps(dict(event="evaluation_unit_complete", seed=seed, stage=stage)), flush=True)
        units = aggregate(case_rows)
        result = dict(status="PASS_VALIDATION_EXECUTION", case_count=495, unit_count=9, foreground_units=18,
                      mass_rows=mass, units=units, validation_GT="evaluator_only", model_forwards=0,
                      hidden_gt_training_usage="none", test_gt_usage="none", mask_seal_content_sha256=seal["content_sha256"],
                      full_case_rows=len(case_rows), no_threshold_or_mass_search=True, controls_cannot_rescue=True)
        d.write_new(args.output/"MMPR_GS_VALIDATION_DIAGNOSTIC.json", result)
        csv_new(args.output/"mmpr_matched_mass_classwise.csv", case_rows+units)
        csv_new(args.output/"mmpr_selection_changes.csv", changes)
        csv_new(args.output/"mmpr_boundary_interior.csv", regions)
        csv_new(args.output/"mmpr_selection_mass_and_ties.csv", selection)
        with (args.output/"MMPR_GS_VALIDATION_DIAGNOSTIC.md").open("x") as f:
            f.write("# MMPR-GS validation diagnostic\n\nAll495 old caches were read-only, original unrounded scores were reproduced, and Q1/Q2 masks were sealed before any validation GT read. Full-image class/case mass is exact. GT255 is ignored only in the independent evaluator. See the CSVs for every case, class, selection change and boundary/interior stratum. No model forward or optimizer operation was performed. Scientific F2 is adjudicated with all18 units in the final report.\n")


def pair_worker(args, reg, p, meta):
    gpu = args.shard
    require(gpu in (4, 5, 6, 7) and os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu) and torch.cuda.device_count() == 1, "GPU assignment")
    pairs = p["gradient_diagnostic"]["batch_pairs"]
    if args.phase == "integration":
        pairs = [q for q in pairs if q["batch_id"] in reg["inputs"]["integration_pair_ids"]]
    ngpu = 3 if args.phase == "integration" else 4
    assigned = pairs[gpu-4::ngpu]
    d.write_new(args.output/f"WORKER_{gpu}_START.json", dict(meta, gpu=gpu, pid=os.getpid(), pairs=[q["batch_id"] for q in assigned]))
    freeze = d.read(reg["inputs"]["k2_freeze"]["path"])
    completed = []
    torch.set_num_threads(1)
    with b.no_updates():
        for pair in assigned:
            require(not list(args.output.glob("FAILURE_*.json")), "another worker failed")
            cp = b.checkpoint(p, pair["seed"], pair["stage_index"])
            models, payload = load_models(ROOT, cp, device="cuda:0")
            legacy = b.legacy_input(payload, cp, p, "cuda:0")
            require(b.tensor_hash(legacy) == cp["legacy_pas_tensor_sha256"], "direct PAS differs from registration")
            current, history = r.banks(freeze, pair["seed"], pair["stage_index"])
            models["student"].requires_grad_(True)
            attach_gradient_student(models, dict(diagnostic_precision="float64_shadow", _precision_contract_verified=True))
            inputs = tuple(x.to("cuda:0") for x in e.pair_inputs(reg["inputs"]["data_root"], p, pair))
            out = args.output / "pairs" / e.pair_name(pair)
            out.mkdir(parents=True, exist_ok=False)
            guard = out / "model_guard"
            with ImmutableModels(models, cp, guard, meta):
                result, arrays = compute_pair(models, legacy, current, history, pair, inputs)
            checked = d.read(guard/"immutability"/(cp["checkpoint_id"].replace("/", "_")+".json"))
            require(checked["bitwise_unchanged"] and checked["extraction_completed"], "checkpoint guard failed", "BLOCKED_MODEL_MUTATION")
            result.update(metadata=meta, checkpoint_guard_pass=True, checkpoint_sha256=cp["sha256"])
            reference = d.read(Path(reg["inputs"]["formal_reference_root"])/"probes/draw0"/e.pair_name(pair)/"result.json")
            parity_keys = ("student_logits_sha256", "student_features_sha256", "labeled_logits_sha256", "teacher_features_sha256",
                           "native_student_probability_sha256", "teacher_probability_sha256", "R1_validity_sha256")
            require(all(result[k] == reference[k] for k in parity_keys), "fixed draw0 native values differ from Gate1C")
            old_r1 = next(row for row in reference["alignment"] if row["candidate"] == "R1" and row["normalization"] == "pixel_normalized" and row["block"] == "global")
            new_r1 = next(row for row in result["alignment"] if row["candidate"] == "Q0")
            require(old_r1["cosine"] is not None and new_r1["cosine"] is not None and abs(old_r1["cosine"]-new_r1["cosine"]) <= 1e-10,
                    "R1 reference VJP differs from frozen diagnostic", "BLOCKED_NUMERICAL_FAILURE")
            result["frozen_native_parity"] = dict(all_bitwise_equal=True, fields=list(parity_keys), reference_path=str(Path(reg["inputs"]["formal_reference_root"])/"probes/draw0"/e.pair_name(pair)/"result.json"),
                                                 R1_reference_cosine=old_r1["cosine"], R1_new_cosine=new_r1["cosine"])
            result["arrays"] = b.save_arrays(out/"diagnostic_arrays.npz", arrays)
            d.write_new(out/"result.json", result)
            completed.append(dict(batch_id=pair["batch_id"], result=str(out/"result.json"), sha256=d.sha256(out/"result.json"), counts=result["counts"]))
            print(json.dumps(dict(event="pair_complete", gpu=gpu, pair=pair["batch_id"], counts=result["counts"])), flush=True)
            del models, payload, legacy, inputs, arrays, result
            torch.cuda.empty_cache()
    d.write_new(args.output/f"WORKER_{gpu}_RESULT.json", dict(status="PASS", completed=completed, gpu=gpu, completed_at=d.now()))


def pair_phase(args, reg, p, meta):
    phase_completed(reg["destination"]["phase_roots"]["input_audit"], "MMPR_GS_INPUT_AUDIT.json")
    phase_completed(reg["destination"]["phase_roots"]["validation"], "MMPR_GS_VALIDATION_DIAGNOSTIC.json")
    if args.phase == "formal":
        phase_completed(reg["destination"]["phase_roots"]["integration"], "MMPR_GS_INTEGRATION_REPORT.json")
    if args.worker:
        return pair_worker(args, reg, p, meta)
    gpus = (4, 5, 6) if args.phase == "integration" else (4, 5, 6, 7)
    processes, handles, exits = [], [], []
    try:
        for gpu in gpus:
            command = [sys.executable, "-B", "-m", "mmpr_gs_v0_1.run", "--phase", args.phase, "--output", str(args.output),
                       "--gate", str(args.gate), "--code-commit", args.code_commit, "--worker", "--shard", str(gpu)]
            log = (args.output/f"worker_{gpu}.log").open("x")
            handles.append(log)
            proc = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT)
            processes.append((gpu, proc, command))
        while any(proc.poll() is None for _, proc, _ in processes):
            if any(proc.poll() not in (None, 0) for _, proc, _ in processes):
                for _, proc, _ in processes:
                    if proc.poll() is None:
                        proc.terminate()
                break
            time.sleep(.5)
    finally:
        for gpu, proc, command in processes:
            try:
                code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait(timeout=10)
            exits.append(dict(gpu=gpu, pid=proc.pid, actual_exit_code=code, command=command))
        for handle in handles:
            handle.close()
        d.write_new(args.output/"WORKER_ACTUAL_EXITS.json", dict(workers=exits, recorded_at=d.now()))
    if not all(r["actual_exit_code"] == 0 for r in exits):
        failures = [d.read(path) for path in sorted(args.output.glob("FAILURE_*.json"))]
        raise Blocked("worker failed; no retry", failures[0]["status"] if failures else "BLOCKED_INCOMPLETE_EVIDENCE")
    expected = p["gradient_diagnostic"]["batch_pairs"]
    if args.phase == "integration":
        expected = [q for q in expected if q["batch_id"] in reg["inputs"]["integration_pair_ids"]]
    paths = list((args.output/"pairs").glob("*/result.json"))
    require(len(paths) == len(expected), "pair result coverage", "BLOCKED_INCOMPLETE_EVIDENCE")
    results = [d.read(args.output/"pairs"/e.pair_name(q)/"result.json") for q in expected]
    totals = {k: sum(result["counts"][k] for result in results) for k in LIMITS}
    require(totals == {k: n * len(expected) for k, n in LIMITS.items()}, "phase compute totals", "BLOCKED_CALL_GRAPH_MISMATCH")
    table_specs = [("alignment", "mmpr_gradient_alignment.csv"), ("blockwise", "mmpr_gradient_blockwise.csv"),
                   ("components", "mmpr_gradient_class_components.csv"), ("retention", "mmpr_projection_retention.csv"),
                   ("precision", "mmpr_native_fp64_precision.csv"), ("mass_rows", "mmpr_gradient_matched_mass.csv")]
    for key, filename in table_specs:
        csv_new(args.output/filename, [row for result in results for row in result[key]])
    status = "PASS_MMPR_INTEGRATION" if args.phase == "integration" else "PASS_GRADIENT_EXECUTION"
    report = dict(status=status, metadata=meta, pair_count=len(results), pairs=[q["batch_id"] for q in expected], counts=totals,
                  new_forwards=5 * len(results), model_guards=len(results), actual_worker_exits=exits,
                  every_native_draw0_bitwise_equal=True, every_R1_reference_VJP_equal_within_1e_10=True,
                  all_model_bank_rng_guards_pass=True, results=[dict(path=str(path), sha256=d.sha256(path)) for path in sorted(paths)],
                  scientific_admission=None, model_optimizer_steps=0, transport_optimizer_steps=0)
    name = "MMPR_GS_INTEGRATION_REPORT" if args.phase == "integration" else "MMPR_GS_GRADIENT_DIAGNOSTIC"
    d.write_new(args.output/(name+".json"), report)
    with (args.output/(name+".md")).open("x") as f:
        f.write(f"# {name}\n\n{status}. Completed{len(results)} original fixed draw0 pairs and{5*len(results)} new forwards. Native values/PAS match frozen v3; FP64 uses the same Gaussian draws. Full parameter inventories, None-zero placeholders, every raw/projected gradient, all six blocks, class components and native comparisons are retained privately. All model/checkpoint/bank/RNG guards and actual worker exits passed; no optimizer/backward/update occurred. Scientific gates are evaluated separately with all72 formal pairs.\n")
    if args.phase == "formal":
        val = d.read(Path(reg["destination"]["phase_roots"]["validation"])/"MMPR_GS_VALIDATION_DIAGNOSTIC.json")
        status = adjudicate(val, results, expected)
        d.write_new(args.output/"MMPR_GS_STATUS.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("input_audit", "validation", "integration", "formal", "final_audit"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shard", type=int, default=-1)
    args = parser.parse_args()
    try:
        reg, p = authority()
        meta = execution_gate(args, reg)
        if not args.worker:
            d.write_new(args.output/"RUN_METADATA.json", meta)
        torch.set_num_threads(1)
        if args.phase == "input_audit":
            input_audit(args, reg, p)
        elif args.phase == "validation":
            validation(args, reg, p)
        elif args.phase in ("integration", "formal"):
            pair_phase(args, reg, p, meta)
        else:
            with forbid_forwards(), b.no_updates():
                audit = verify_bundle(reg)
                d.write_new(args.output/"MMPR_GS_FINAL_INPUT_IMMUTABILITY.json", audit)
    except BaseException as error:
        status = getattr(error, "status", "BLOCKED_INCOMPLETE_EVIDENCE")
        if status in ("BLOCKED_NONFINITE_EVIDENCE", "BLOCKED_NONFINITE_FEATURE"):
            status = "BLOCKED_NUMERICAL_FAILURE"
        if status not in ("BLOCKED_PROTOCOL_OR_LEAKAGE", "BLOCKED_PRIVATE_BUNDLE_MISMATCH", "BLOCKED_MODEL_MUTATION",
                          "BLOCKED_CALL_GRAPH_MISMATCH", "BLOCKED_NUMERICAL_FAILURE", "BLOCKED_INCOMPLETE_EVIDENCE"):
            status = "BLOCKED_PROTOCOL_OR_LEAKAGE"
        d.write_new(args.output/f"FAILURE_{args.shard if args.worker else 'controller'}.json",
                    dict(status=status, error=str(error), traceback=traceback.format_exc(), command=sys.argv,
                         recorded_at=d.now(), new_attempt_authorized=False, model_optimizer_steps=0, transport_optimizer_steps=0))
        raise


if __name__ == "__main__":
    main()
