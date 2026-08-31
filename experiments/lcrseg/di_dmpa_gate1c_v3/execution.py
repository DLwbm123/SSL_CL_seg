"""Finite v3 phase orchestration; all scoring, VJPs and C1-C8 use frozen code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import torch

from di_dmpa_gate1c_v2 import binding as b, execution as e, reporting, precision_pilot as pilot
from di_dmpa_gate1c_v2.full_precision import validate_isolation, forbid_forwards
from di_dmpa_gate1c_v2.runner import disk_hashes
from . import PROTOCOL
from .binding import ROOT, load_authority, audit_inputs
from .durable import canonical, now, read, sha256, verify, write_new
from .inputs import load_models
from . import validation


def pairs(p, scope):
    rows = p["gradient_diagnostic"]["batch_pairs"]
    return [next(q for q in rows if q["batch_id"] == key) for key in p["integration_pair_ids"]] if scope == "integration" else rows


def shard_count(scope, phase):
    return 2 if phase.endswith("metrics") else 3 if scope == "integration" else 4


def phase_receipt(output, phase, meta, details, paths):
    paths = sorted(set(Path(x) for x in paths))
    evidence = {str(path.relative_to(output)): sha256(path) for path in paths}
    path = output / f"PHASE_{phase}.json"
    write_new(path, dict(metadata=meta, phase=phase, status="PASS", evidence_sha256=evidence, **details))
    entries = [dict(path=str(q.relative_to(output)), bytes=q.stat().st_size, sha256=sha256(q)) for q in [*paths, path]]
    write_new(output / f"PHASE_{phase}_MANIFEST.json", dict(entries=entries, files=len(entries),
        bytes=sum(r["bytes"] for r in entries), content_sha256=hashlib.sha256(canonical(entries)).hexdigest()))


def worker(args, p, freeze, base_meta):
    out = args.output
    meta = read(out / "GATE1C_V3_RUN_METADATA.json")
    b.require(all(meta[k] == v for k, v in base_meta.items()) and os.getppid() == meta["controller_pid"], "unowned or mixed worker")
    n = shard_count(args.scope, args.phase)
    b.require(args.shard in range(n), "unknown worker shard")
    cpu = args.phase.endswith("metrics")
    gpu = None if cpu else args.shard + 4
    b.require(os.environ.get("CUDA_VISIBLE_DEVICES") == ("" if cpu else str(gpu)), "physical GPU assignment changed")
    if not cpu:
        b.require(torch.cuda.device_count() == 1, "one explicitly mapped GPU required")
    label = f"{args.phase}_{args.shard}"
    start = dict(metadata=meta, phase=args.phase, shard=args.shard, pid=os.getpid(), parent_pid=os.getppid(), physical_gpu=gpu, started_at=now())
    write_new(out / f"WORKER_{label}_START.json", start)
    counts = dict.fromkeys(pilot.COUNT_KEYS, 0)
    completed, parity = [], []
    active_counts = None
    def timeout(*unused):
        raise TimeoutError("registered worker phase time budget exhausted")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(p["execution"]["maximum_worker_phase_seconds"])
    try:
        with b.no_updates(), patch.object(b, "load_models", load_models):
            if args.phase == "validation":
                for i, (seed, stage) in enumerate((s, t) for s in range(3) for t in range(3)):
                    if i % n != args.shard:
                        continue
                    b.require(not list(out.glob("FAILURE_*.json")), "another worker failed; stop before next unit")
                    plan = next(u for u in p["validation"]["plans"] if (u["seed"], u["stage_index"]) == (seed, stage))
                    with validation.capture(out, 2 * len(plan["cases"])) as seen:
                        active_counts = seen
                        e.validation_unit(ROOT, p["destination"]["data_root"], p, freeze, meta, seed, stage, out, "cuda:0")
                    b.require(seen == dict(native_forwards=2 * len(plan["cases"]), cases=len(plan["cases"]), original_PAS_calls=2 * len(plan["cases"])),
                              "incomplete fresh validation forwards/PAS")
                    counts["native_forwards"] += seen["native_forwards"]
                    completed.append(dict(seed=seed, stage_index=stage, cases=seen["cases"]))
                    active_counts = None
            elif cpu:
                with forbid_forwards():
                    for i, (seed, stage) in enumerate((s, t) for s in range(3) for t in range(3)):
                        if i % n != args.shard:
                            continue
                        b.require(not list(out.glob("FAILURE_*.json")), "another worker failed; stop before next evaluator")
                        completed.append(e.evaluate_unit((p["destination"]["data_root"], p, str(out), seed, stage, args.phase == "poe_metrics")))
            else:
                for pair in pairs(p, args.scope)[args.shard::n]:
                    b.require(not list(out.glob("FAILURE_*.json")), "another worker failed; stop before next pair")
                    active_counts = dict.fromkeys(pilot.COUNT_KEYS, 0)
                    limits = dict(zip(pilot.COUNT_KEYS, pilot.COUNTS[args.phase]))
                    with pilot.observe_pair(pair, active_counts, parity, limits=limits):
                        e.probe_unit(ROOT, p["destination"]["data_root"], p, freeze, meta, pair["seed"], pair["stage_index"],
                                     out, "cuda:0", args.phase, pair_indices=[pair["pair_index"]])
                    b.require(active_counts == limits, "registered per-pair compute count mismatch")
                    for key in counts:
                        counts[key] += active_counts[key]
                    completed.append(pair["batch_id"])
                    active_counts = None
        disk_hashes(p)
        if not cpu:
            b.require(torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.deterministic
                      and not torch.backends.cudnn.benchmark and not torch.backends.cudnn.allow_tf32
                      and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled(), "native backend flags changed")
        write_new(out / f"WORKER_{label}.json", dict(status="PASS", **start, completed_at=now(), counts=counts,
            completed_units=completed, PAS_parity=parity, all_checkpoints_unchanged=True))
    except BaseException as error:
        write_new(out / f"FAILURE_{label}.json", dict(**start, error=str(error), traceback=traceback.format_exc(),
            completed_counts=counts, current_incomplete_unit_counts=active_counts, completed_units=completed, PAS_parity=parity,
            no_automatic_retry=True, failed_at=now()))
        raise
    finally:
        signal.alarm(0)


def dispatch(args, p, meta, phase):
    processes, logs = [], []
    try:
        for shard in range(shard_count(args.scope, phase)):
            command = [sys.executable, "-B", "-m", "di_dmpa_gate1c_v3.execution", "--worker", "--scope", args.scope,
                "--phase", phase, "--shard", str(shard), "--output", str(args.output), "--authorization", str(args.authorization),
                "--authorization-sha256", args.authorization_sha256, "--authorization-commit", args.authorization_commit]
            gpu = "" if phase.endswith("metrics") else str(shard + 4)
            log = (args.output / f"WORKER_{phase}_{shard}.log").open("xb")
            logs.append(log)
            processes.append(subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
        while any(proc.poll() is None for proc in processes):
            if any(proc.returncode not in (None, 0) for proc in processes):
                marker = args.output / f"FAILURE_child_exit_{phase}.json"
                if not marker.exists():
                    write_new(marker, dict(metadata=meta, failed_at=now(), error="worker exited unsuccessfully",
                        worker_pids=[proc.pid for proc in processes], exit_snapshot=[proc.returncode for proc in processes]))
            time.sleep(0.25)
        codes = [proc.wait() for proc in processes]
    except BaseException as error:
        write_new(args.output / f"FAILURE_dispatch_{phase}.json", dict(metadata=meta, error=str(error), failed_at=now()))
        codes = [proc.wait() for proc in processes]
        raise
    finally:
        for log in logs:
            log.close()
        write_new(args.output / f"PROCESS_EXIT_{phase}.json", dict(metadata=meta, phase=phase,
            worker_pids=[proc.pid for proc in processes], actual_child_exit_codes=[proc.returncode for proc in processes],
            parent_pid=os.getpid(), written_by_server_local_parent=True, observed_at=now()))
    b.require(codes == [0] * shard_count(args.scope, phase) and not list(args.output.glob("FAILURE_*.json")), "worker failure; no phase admission")


def worker_evidence(args, p, meta, phase):
    out, totals, paths = args.output, dict.fromkeys(pilot.COUNT_KEYS, 0), []
    exits = read(out / f"PROCESS_EXIT_{phase}.json")
    n = shard_count(args.scope, phase)
    b.require(exits["actual_child_exit_codes"] == [0] * n and exits["parent_pid"] == meta["controller_pid"], "worker exit provenance mismatch")
    paths.append(out / f"PROCESS_EXIT_{phase}.json")
    for shard in range(n):
        start_path = out / f"WORKER_{phase}_{shard}_START.json"
        done_path = out / f"WORKER_{phase}_{shard}.json"
        start, done = read(start_path), read(done_path)
        b.require(start["pid"] == exits["worker_pids"][shard] and start["parent_pid"] == meta["controller_pid"]
                  and start["metadata"] == done["metadata"] == meta and done["status"] == "PASS"
                  and done["all_checkpoints_unchanged"] and start["phase"] == done["phase"] == phase
                  and start["shard"] == done["shard"] == shard and start["pid"] == done["pid"]
                  and start["physical_gpu"] == done["physical_gpu"] == (None if phase.endswith("metrics") else shard + 4),
                  "incomplete worker proof")
        if phase in pilot.PHASES:
            selected = pairs(p, args.scope)[shard::n]
            expected = dict(zip(pilot.COUNT_KEYS, [x * len(selected) for x in pilot.COUNTS[phase]]))
            b.require(done["counts"] == expected and done["completed_units"] == [q["batch_id"] for q in selected], "wrong worker coverage/counts")
            multiplier = {"draw0": 1, "noise": 8, "posterior": 1, "poe": 0}[phase]
            b.require(len(done["PAS_parity"]) == multiplier * len(selected)
                      and all(r["exact_R1_parity"] and r["pixels"] == 294912 for r in done["PAS_parity"]), "incomplete native PAS parity")
        else:
            expected = [(s, t) for i, (s, t) in enumerate((s, t) for s in range(3) for t in range(3)) if i % n == shard]
            b.require([(r["seed"], r["stage_index"]) for r in done["completed_units"]] == expected, "incomplete evaluator/cache worker coverage")
        for key in totals:
            totals[key] += done["counts"][key]
        paths.extend([start_path, done_path, out / f"WORKER_{phase}_{shard}.log"])
    return totals, paths


def golden(result):
    fields = ("student_logits_sha256", "student_features_sha256", "labeled_logits_sha256", "teacher_features_sha256",
              "teacher_probability_sha256", "R1_validity_sha256", "native_supervised_gradient_sha256",
              "supervised_gradient_sha256", "gradient_hashes", "student_draw_replay")
    return dict(batch_id=result["pair"]["batch_id"], **{key: result[key] for key in fields})


def probe_barrier(args, p, meta, phase):
    out = args.output
    counts, paths = worker_evidence(args, p, meta, phase)
    selected, results, guards = pairs(p, args.scope), [], []
    coverage = dict(alignment_rows=0, global_comparisons=0, class_components=0, supervised_global_comparisons=0)
    for pair in selected:
        path = out / "probes" / phase / e.pair_name(pair) / "result.json"
        result = read(path)
        values = pilot.validate_result(result, pair, phase, meta)
        for key, value in values.items():
            coverage[key] += value
        iso_path, guard_path, guard = validate_isolation(out, pair, phase, meta)
        guards.append(guard)
        paths.extend([path, iso_path, guard_path])
        for field in ("primary_cache", "teacher_cache"):
            if field in result:
                desc = result[field]
                target = Path(desc["path"])
                b.require(target.resolve().is_relative_to(out.resolve()) and target.stat().st_size == desc["bytes"], "probe array path/size changed")
                b.read_arrays(desc)
                paths.append(target)
        results.append(result)
    b.require({q.name for q in (out / "probes" / phase).iterdir()} == {e.pair_name(q) for q in selected}, "extra/missing probe directory")
    b.require(counts == dict(zip(pilot.COUNT_KEYS, [x * len(selected) for x in pilot.COUNTS[phase]])), "phase compute budget mismatch")
    if args.scope == "formal":
        reporting.validate_probe_results(p, results, phase)
    if phase == "draw0":
        if args.scope == "integration":
            path = out / "GATE1C_V3_GOLDENS.json"
            write_new(path, dict(metadata=meta, source="new v3 integration only", old_private_hashes_used=False,
                                 records=[golden(r) for r in results]))
            paths.append(path)
        else:
            prior = read(Path(p["output_roots"]["integration"]) / "GATE1C_V3_GOLDENS.json")
            by_id = {r["pair"]["batch_id"]: r for r in results}
            b.require(len(prior["records"]) == 3 and all(golden(by_id[g["batch_id"]]) == g for g in prior["records"]),
                      "new v3 integration golden changed in formal execution")
    phase_receipt(out, phase, meta, dict(counts=counts, coverage=coverage, guards=guards, numerical_checks_complete=True), paths)


def prior_gate(args, p, meta):
    scope = "validation" if args.scope == "integration" else "integration"
    status_name = "VALIDATION_CACHE_V3_AUDIT.json" if scope == "validation" else "GATE1C_V3_INTEGRATION_REPORT.json"
    expected = "PASS_FRESH_VALIDATION_CACHE" if scope == "validation" else "PASS_NEW_V3_INTEGRATION"
    b.require(args.prior_gate is not None and args.prior_gate_sha256 is not None, "verified local archive gate required")
    b.check_hash(args.prior_gate, args.prior_gate_sha256)
    gate = read(args.prior_gate)
    source = Path(p["output_roots"][scope])
    b.require(gate["protocol"] == PROTOCOL and gate["source_scope"] == scope and gate["actual_child_exit_code"] == 0
              and gate["archive_audit"]["status"] == "PASS_PRIVATE_ARCHIVE", "prior archive gate did not pass")
    manifest = verify(source)
    b.require(manifest["content_sha256"] == gate["archive_audit"]["content_sha256"]
              and sha256(source / "PRIVATE_BUNDLE_MANIFEST.json") == gate["archive_audit"]["manifest_sha256"], "prior remote/local archive mismatch")
    b.require(read(source / "PROCESS_EXIT.json")["actual_child_exit_code"] == 0, "prior parent exit failed")
    b.check_hash(source / status_name, gate["source_status_sha256"])
    result = read(source / status_name)
    b.require(result["status"] == expected and result["new_model_forwards"] == p["budget"][scope], "prior numerical/cache gate incomplete")
    original = read(source / "GATE1C_V3_RUN_METADATA.json")
    for key in ("registration_id", "diagnostic_code_commit", "preregistration_sha256", "authorization_sha256", "baseline_manifest_sha256", "k2_freeze_sha256"):
        b.require(original[key] == meta[key], "prior gate from another input/code/authority")
    return dict(source_scope=scope, source_status_sha256=gate["source_status_sha256"], archive_audit=gate["archive_audit"])


def fresh_cache_references(args, p, meta):
    source = Path(p["output_roots"]["validation"])
    verify(source)
    audit = read(source / "VALIDATION_CACHE_V3_AUDIT.json")
    b.require(audit["status"] == "PASS_FRESH_VALIDATION_CACHE" and audit["cases"] == 495
              and audit["model_guards"] == 9 and audit["new_model_forwards"] == 990, "fresh cache barrier incomplete")
    manifest = source / "VALIDATION_CACHE_V3_MANIFEST.json"
    b.check_hash(manifest, audit["manifest_sha256"])
    index = read(manifest)
    for key in ("diagnostic_code_commit", "preregistration_sha256", "authorization_sha256", "baseline_manifest_sha256", "k2_freeze_sha256"):
        b.require(index["metadata"][key] == meta[key], "cache from another v3 generation")
    b.require(len(index["units"]) == 9 and sum(len(u["cases"]) for u in index["units"]) == 495, "fresh cache index incomplete")
    if args.scope == "formal":
        directory = args.output / "validation_units"
        directory.mkdir(exist_ok=False)
        for item in index["units"]:
            b.check_hash(item["path"], item["sha256"])
            unit = read(item["path"])
            b.require(unit["cases"] == item["cases"], "fresh cache index/cases changed")
            extra = {k: unit["metadata"][k] for k in ("seed", "stage_index", "role", "bank", "legacy_prototypes_sha256")}
            copied = dict(unit, metadata=dict(meta, **extra), fresh_v3_source_unit=dict(path=item["path"], sha256=item["sha256"],
                original_metadata=unit["metadata"], generation_new_forwards=2 * unit["case_count"], old_private_cache_reused=False))
            write_new(directory / Path(item["path"]).name, copied)
    return dict(status="PASS", source=str(source), manifest_sha256=sha256(manifest), fresh_generation=True,
                generation_forwards=990, model_guards=9, old_private_cache_reused=False)


def numerical_summary(args, p, meta):
    out, selected = args.output, pairs(p, args.scope)
    totals = dict.fromkeys(pilot.COUNT_KEYS, 0)
    coverage = dict(alignment_rows=0, global_comparisons=0, class_components=0, supervised_global_comparisons=0)
    comparisons, supervised, components, guards = [], [], [], []
    for phase in pilot.PHASES:
        receipt = read(out / f"PHASE_{phase}.json")
        b.require(receipt["metadata"] == meta and receipt["status"] == "PASS", "missing/mixed phase barrier")
        for path, digest in receipt["evidence_sha256"].items():
            b.check_hash(out / path, digest)
        for key in totals:
            totals[key] += receipt["counts"][key]
        for key in coverage:
            coverage[key] += receipt["coverage"][key]
        guards.extend(receipt["guards"])
        for pair in selected:
            result = read(out / "probes" / phase / e.pair_name(pair) / "result.json")
            pilot.validate_result(result, pair, phase, meta)
            comparisons.extend(r for r in result["native_precision_comparisons"] if r["block"] == "global")
            supervised.append(result["supervised_precision_comparisons"]["global"])
            components.extend(result["class_contribution"])
    n = len(selected)
    b.require(totals == dict(zip(pilot.COUNT_KEYS, (17 * n, 8 * n, 92 * n, 122 * n)))
              and totals["native_forwards"] + totals["shadow_forwards"] == p["budget"][args.scope], "total diagnostic budget mismatch")
    b.require(coverage == dict(zip(coverage, (672 * n, 96 * n, 210 * n, 4 * n)))
              and len(guards) == 4 * n == len(list((out / "probe_models").rglob("immutability/*.json")))
              and len(list((out / "probes").glob("*/*/result.json"))) == 4 * n, "incomplete numeric/guard coverage")
    result = dict(metadata=meta, status="PASS", counts=totals, new_model_forwards=25 * n, coverage=coverage,
        model_guards=len(guards), maximum_objective_relative_l2=max((r["relative_l2"] for r in comparisons if r["relative_l2"] is not None), default=None),
        minimum_objective_cosine=min((r["cosine"] for r in comparisons if r["cosine"] is not None), default=None),
        maximum_supervised_relative_l2=max((r["relative_l2"] for r in supervised if r["relative_l2"] is not None), default=None),
        minimum_supervised_cosine=min((r["cosine"] for r in supervised if r["cosine"] is not None), default=None),
        maximum_component_sum_abs_residual=max(r["component_sum_max_abs_error"] for r in components),
        old_private_golden_hashes_used=False, optimizer_steps=0)
    write_new(out / "NUMERICAL_COMPARISON_AUDIT.json", result)
    return result


def reduced_candidate(status):
    if status["reliability_status"] == "PASS_IDENTITY_HISTORY_WEIGHT_ONLY":
        return "HVR_MPA_JASCL"
    if status["reliability_status"] == "PASS_IDENTITY_HISTORY_CLASS_BALANCED_ONLY":
        return "CB_HVR_MPA_JASCL"
    if status["R0_R1_R2_R3_results"]["R2"]["pixel_normalized"]["all_pass"]:
        return "CURRENT_ONLY_MPA_JASCL"
    return "NONE"


def compile_v3(args, p, meta, numbers):
    audit = dict(metadata=meta, status="PASS", guard_count=297, new_probe_guard_count=288,
                 validation_guards_generated_in_v3=9, all_model_states_bitwise_unchanged=True, all_grad_fields_None=True)
    write_new(args.output / "GATE1C_V3_MODEL_IMMUTABILITY_AUDIT.json", audit)
    original_json, original_text, original_csv = reporting.write_json, reporting.write_text, reporting.write_csv
    def destination(path):
        path = Path(path)
        return path.with_name(path.name.replace("_V2", "_V3").replace("_v2", "_v3"))
    def write_json(path, value):
        if Path(path).name == "GATE1C_V2_STATUS.json":
            value.update(status=value["reliability_status"], protocol=PROTOCOL, reduced_method_candidate=reduced_candidate(value),
                historical_bank_claim_allowed=value["reliability_status"].startswith("PASS_IDENTITY_HISTORY"),
                k2_replication_status="K2_REPLICATION_PASS", new_validation_forwards=990, new_integration_forwards=75,
                new_formal_forwards=1800, total_new_gate1c_forwards=2865, new_formal_probe_guards=288,
                validation_guards_generated_in_v3=9, old_private_cache_reused=False, legacy_pas_reconstruction=False,
                numerical_comparison_audit_sha256=sha256(args.output / "NUMERICAL_COMPARISON_AUDIT.json"),
                next_action="REPORT_AND_HARD_STOP_NO_METHOD_IMPLEMENTATION")
            value["R3_reduced_candidate_status"] = value["reduced_candidate_status"]
            value["reduced_candidate_status"] = ("ELIGIBLE_FOR_NEW_NON_TRANSPORT_METHOD_PREREGISTRATION"
                if value["reduced_method_candidate"] != "NONE" else "NOT_ELIGIBLE")
        return original_json(destination(path), value)
    def write_text(path, value):
        value += ("\nFresh v3 execution: 990 validation forwards, 75 separate integration forwards and 1800 formal forwards; "
                  "total 2865. All 495 validation caches, raw native tensors/PAS intermediates, 9 validation guards, "
                  "12 integration guards and 288 formal guards were generated in this protocol. R1 reads the direct "
                  "historically hashed PAS bank; no reconstructed bank or old private cache/golden is used. "
                  "Reduced-method candidates are decisions for a separate preregistration only; no method or C0 is implemented here.\n")
        status_path = args.output / "GATE1C_V3_STATUS.json"
        if status_path.exists():
            status = read(status_path)
            value += (f"\nReduced method candidate: {status['reduced_method_candidate']}. "
                      f"Historical-bank claim allowed: {status['historical_bank_claim_allowed']}. "
                      "A passing pixel-normalized R2 can nominate a current-only future candidate but cannot change the R3 Gate1C status.\n")
        return original_text(destination(path), value)
    with forbid_forwards(), patch.object(reporting, "write_json", write_json), patch.object(reporting, "write_text", write_text), \
            patch.object(reporting, "write_csv", lambda path, rows: original_csv(destination(path), rows)):
        return reporting.compile_report(args.output, p, meta, audit)


def run(args, p, freeze, meta):
    out = args.output
    b.require(os.statvfs(out).f_bavail * os.statvfs(out).f_frsize >= p["execution"]["minimum_free_bytes"],
              "BLOCKED_STORAGE_OR_ARCHIVE_FAILURE: insufficient reserved headroom")
    meta.update(controller_pid=os.getpid(), started_at=now(), exact_command=sys.argv, old_private_inputs_read=False)
    write_new(out / "GATE1C_V3_RUN_METADATA.json", meta)
    with forbid_forwards(), b.no_updates():
        inputs = audit_inputs(p, meta)
    write_new(out / "GATE1C_V3_INPUT_AUDIT.json", inputs)
    # The unchanged compiler reads this fixed filename; its content is new v3.
    write_new(out / "GATE1C_V2_INPUT_AUDIT.json", inputs)
    if args.scope != "validation":
        write_new(out / "PRIOR_VERIFIED_ARCHIVE_GATE.json", prior_gate(args, p, meta))
        write_new(out / "FRESH_V3_CACHE_REFERENCES.json", fresh_cache_references(args, p, meta))
    if args.scope == "validation":
        dispatch(args, p, meta, "validation")
        counts, paths = worker_evidence(args, p, meta, "validation")
        b.require(counts == dict(native_forwards=990, shadow_forwards=0, native_autograd=0, shadow_autograd=0), "fresh cache budget mismatch")
        with forbid_forwards(), b.no_updates():
            result = validation.audit(out, p["destination"]["data_root"], p, freeze, meta)
        result.update(counts=counts, new_model_forwards=990)
        write_new(out / "VALIDATION_CACHE_V3_AUDIT.json", result)
        paths.extend(q for folder in ("validation_units", "validation_models", "validation_cache", "validation_raw")
                     for q in (out / folder).rglob("*") if q.is_file())
        paths.extend([out / "VALIDATION_CACHE_V3_MANIFEST.json", out / "VALIDATION_CACHE_V3_AUDIT.json"])
        phase_receipt(out, "validation", meta, dict(counts=counts, cases=495, model_guards=9), paths)
    else:
        phases = ["validation_metrics", *pilot.PHASES, "poe_metrics"] if args.scope == "formal" else pilot.PHASES
        for phase in phases:
            b.require(not list(out.glob("FAILURE_*.json")), "prior worker failure; no continuation")
            dispatch(args, p, meta, phase)
            if phase.endswith("metrics"):
                counts, paths = worker_evidence(args, p, meta, phase)
                folder = out / ("poe_validation" if phase == "poe_metrics" else "reliability_units")
                values = [read(q) for q in sorted(folder.glob("*.json"))]
                b.require(counts == dict.fromkeys(pilot.COUNT_KEYS, 0) and len(values) == 9
                          and {(u["seed"], u["stage_index"]) for u in values} == {(s, t) for s in range(3) for t in range(3)}, "nine zero-forward evaluators required")
                phase_receipt(out, phase, meta, dict(counts=counts, units=9), paths + list(folder.glob("*.json")))
            else:
                probe_barrier(args, p, meta, phase)
            print(json.dumps(dict(scope=args.scope, phase=phase, status="PASS")), flush=True)
        numbers = numerical_summary(args, p, meta)
        if args.scope == "formal":
            result = compile_v3(args, p, meta, numbers)
        else:
            result = dict(numbers, status="PASS_NEW_V3_INTEGRATION", pairs=3, phases=4, scientific_admission=None,
                          retry_allowed=False, new_v3_goldens_sha256=sha256(out / "GATE1C_V3_GOLDENS.json"))
            write_new(out / "GATE1C_V3_INTEGRATION_REPORT.json", result)
            (out / "GATE1C_V3_INTEGRATION_REPORT.md").open("x").write(
                "# Gate1C v3 integration\n\nPASS_NEW_V3_INTEGRATION. Exactly three fixed pairs, four phases, 75 new native/shadow forwards and 12 complete guards. "
                "The frozen PAS parity, relative-L2/cosine, decomposition, RNG and zero-update checks all passed. "
                "New v3 goldens are sealed; old private goldens were not required. Scientific C1-C8 admission is not made here. "
                "Formal execution still requires the server-local parent exit and verified local private archive. No second integration attempt is allowed.\n")
    with forbid_forwards(), b.no_updates():
        after = audit_inputs(p, meta)
    comparable_after = dict(after, data_recheck={k: v for k, v in after["data_recheck"].items() if k != "generated_at"})
    comparable_before = dict(inputs, data_recheck={k: v for k, v in inputs["data_recheck"].items() if k != "generated_at"})
    b.require(comparable_after == comparable_before, "input/model/data audit changed during phase")
    write_new(out / "GATE1C_V3_INPUT_AFTER.json", after)
    write_new(out / "V3_SCOPE_COMPLETION.json", dict(metadata=meta, status="COMPUTATION_AND_AUDITS_COMPLETE",
        scope=args.scope, completed_at=now(), result_status=result["status"], new_model_forwards=p["budget"][args.scope],
        parent_exit="PENDING_SERVER_LOCAL_PARENT_RECEIPT", private_archive="PENDING_LOCAL_VERIFICATION"))
    print(json.dumps(dict(scope=args.scope, status=result["status"], new_model_forwards=p["budget"][args.scope])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--authorization-commit", required=True)
    parser.add_argument("--scope", choices=("validation", "integration", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-gate", type=Path)
    parser.add_argument("--prior-gate-sha256")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--phase", choices=("validation", "validation_metrics", "poe_metrics", *pilot.PHASES))
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    p, freeze, meta = load_authority(args)
    torch.set_num_threads(1)
    if args.worker:
        worker(args, p, freeze, meta)
        return
    try:
        run(args, p, freeze, meta)
    except BaseException as error:
        write_new(args.output / "FAILURE_controller.json", dict(metadata=meta, failed_at=now(), error=str(error),
            traceback=traceback.format_exc(), scope=args.scope, no_automatic_retry=True,
            status="BLOCKED_STORAGE_OR_ARCHIVE_FAILURE" if "STORAGE" in str(error) else getattr(error, "status", "BLOCKED_INCOMPLETE_EVIDENCE")))
        raise


if __name__ == "__main__":
    main()
