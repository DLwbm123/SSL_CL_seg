"""Only regenerated B0-EMA and K1/K2; reuse frozen null-aware geometry math."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import socket
import subprocess

import numpy as np
import torch

from di_dmpa_gate1.binding import H
from di_dmpa_gate1.sampling import sample_layout
from di_dmpa_gate1.gate1a_reporting import primary_conditions, write_csv
from di_dmpa_gate1_v2.features import extract_unit, ImmutableModels, weight_hash
from di_dmpa_gate1_v2.geometry import verify_null_identity
from di_dmpa_gate1_v2.runner import geometry_job
from di_dmpa_gate1_v2.reporting import statistics
from di_dmpa_gate1c_v2.binding import no_updates
from di_dmpa_jascl.provenance import git_revision
from lcrseg.acceptance import verify_checksums
from .durable import read, sha256, write_new, now, verify
from .inputs import load_models

ROOT = Path(__file__).resolve().parents[1]


def adjudicate(rows):
    keys = [(r["panel_id"], r["seed"], r["stage_index"], r["class_id"], r["K"]) for r in rows]
    expected = {("B0-EMA", s, t, c, k) for s in range(3) for t in range(3) for c in range(3) for k in (1, 2)}
    if len(keys) != 54 or set(keys) != expected:
        raise RuntimeError("replication requires exactly 54 unique B0-EMA K1/K2 geometry units")
    lookup = {key: row for key, row in zip(keys, rows)}
    identities = []
    for row in rows:
        if (row["admission_radius_field"] != "R95_null_worst_case"
                or set(row["metrics"]) != {"train_labeled", "val"} or len(row["bootstrap"]) != 5):
            raise RuntimeError("incomplete roles or registered bootstrap draws")
        for role, value in row["metrics"].items():
            if (value["metric_schema"] != "NULL_AWARE_SPHERE_V2" or value["admission_radius_field"] != "R95_null_worst_case"
                    or not all(np.isfinite(value[k]) for k in ("R95_null_worst_case", "Q_null_worst_case", "null_mass"))):
                raise RuntimeError("wrong/nonfinite replication radius")
            if (not value["registered_count"] == value["full_uid_count_used"] == row["expected_registered_counts"][role]
                    or not value["null_count"] == value["null_rows_retained"] == row["expected_null_counts"][role]
                    or value["active_count"] + value["null_count"] != value["registered_count"]):
                raise RuntimeError("replication lost UID/null observations")
        if any(len(b["matched_cosines"]) != row["K"] or not np.isfinite(b["matched_cosines"]).all() for b in row["bootstrap"]):
            raise RuntimeError("missing matched bootstrap slot")
        if row["K"] == 2:
            ref = lookup[("B0-EMA", row["seed"], row["stage_index"], row["class_id"], 1)]
            for role in ("train_labeled", "val"):
                identities.append(verify_null_identity(ref["metrics"][role], row["metrics"][role]))
    stats = statistics(rows, "B0-EMA", 2)
    conditions = primary_conditions(stats)
    passed = all(conditions.values())
    return dict(status="K2_REPLICATION_PASS" if passed else "K2_REPLICATION_FAIL", selected_K=2 if passed else None,
                primary_panel="B0-EMA", K_reference=1, K_candidate=2, statistics=stats, conditions=conditions,
                background_excluded=True, null_identity_checks=len(identities), new_K_selection=False,
                other_panels_executed=False, gate1_overall_status="FAIL_TRANSPORT_NOT_SUPPORTED")


def run(args):
    if sha256(args.preregistration) != args.preregistration_sha256:
        raise RuntimeError("K2 registration hash mismatch")
    p = read(args.preregistration)
    if p["protocol"] != "K2_REPLICATION_B0_EMA_ONLY" or git_revision(ROOT) != p["code_commit"]:
        raise RuntimeError("wrong replication protocol/code")
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise RuntimeError("tracked exact execution source is dirty")
    for relative, digest in p["frozen_source_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError("frozen numerical source changed: " + relative)
    dest = p["destination"]
    if socket.gethostname() != dest["hostname"] or os.getuid() != dest["uid"]:
        raise RuntimeError("destination identity changed")
    if args.output.is_symlink() or args.output.resolve() != Path(dest["output_root"]) or args.output.stat().st_uid != os.getuid():
        raise RuntimeError("unsafe or unregistered output path")
    if os.statvfs(args.output).f_bavail * os.statvfs(args.output).f_frsize < 10 * 1024**3:
        raise RuntimeError("BLOCKED_STORAGE_OR_ARCHIVE_FAILURE: less than 10 GiB free")
    if args.data_root.resolve() != Path(dest["data_root"]):
        raise RuntimeError("unregistered frozen data root")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("4", "5", "6", "7"):
        raise RuntimeError("GPU outside user-authorized shared scope")
    if sha256(args.baseline_manifest) != p["baseline_manifest_sha256"]:
        raise RuntimeError("baseline manifest mismatch")
    baseline = read(args.baseline_manifest)
    cp_keys = [(c["seed"], c["stage_index"]) for c in baseline["checkpoints"]]
    admissions = baseline["seed_admissions"]
    if (baseline["status"] != "PASS_ALL_THREE_REGENERATED_B0_SEEDS" or len(cp_keys) != 9
            or set(cp_keys) != {(s, t) for s in range(3) for t in range(3)}
            or len(admissions) != 3 or {s["seed"] for s in admissions} != {0, 1, 2}):
        raise RuntimeError("three admitted B0 seeds required before any K2 forward")
    for seed in admissions:
        if seed["archive_audit"]["status"] != "PASS_PRIVATE_ARCHIVE" or seed["actual_child_exit_code"] != 0:
            raise RuntimeError("baseline private archive is not verified")
        if verify(seed["remote_root"])["content_sha256"] != seed["archive_audit"]["content_sha256"]:
            raise RuntimeError("baseline remote bundle changed")
        root = Path(seed["remote_root"])
        if root.resolve() != Path(dest["baseline_run_root"]) / f"B0_seed{seed['seed']}":
            raise RuntimeError("only new v3 baseline bundles are eligible")
        if (read(root / "PROCESS_EXIT.json")["actual_child_exit_code"] != 0
                or read(root / "BASELINE_V3_SEED_ENGINEERING_AUDIT.json")["status"] != "PASS_SEED_ENGINEERING"):
            raise RuntimeError("baseline engineering/exit gate failed")
        for checkpoint in [c for c in baseline["checkpoints"] if c["seed"] == seed["seed"]]:
            if not Path(checkpoint["path"]).resolve().is_relative_to(root.resolve()):
                raise RuntimeError("checkpoint outside its admitted v3 bundle")
    if sha256(args.data_root / "checksums/checksums.sha256") != p["checksums_sha256"]:
        raise RuntimeError("frozen data checksum list changed")
    data_check = verify_checksums(args.data_root)
    write_new(args.output / "K2_FROZEN_DATA_RECHECK.json", data_check)
    if not data_check["valid"] or data_check["entries"] != 2962:
        raise RuntimeError("frozen data verification failed")
    plan_path = ROOT / p["sampling_plan_path"]
    old_path = ROOT / p["geometry_registration_path"]
    if sha256(plan_path) != p["sampling_plan_sha256"] or sha256(old_path) != p["geometry_registration_sha256"]:
        raise RuntimeError("frozen coordinate/bootstrap registration changed")
    plan, old = read(plan_path), read(old_path)
    if (len(plan["units"]) != 18 or {(u["seed"], u["stage_index"], u["role"]) for u in plan["units"]}
            != {(s, t, r) for s in range(3) for t in range(3) for r in ("train_labeled", "val")}):
        raise RuntimeError("wrong shared coordinate plan")
    metadata = dict(registration_id=p["protocol"], preregistration_sha256=args.preregistration_sha256,
                    diagnostic_code_git_commit=p["code_commit"], baseline_manifest_sha256=p["baseline_manifest_sha256"],
                    sampling_plan_sha256=p["sampling_plan_sha256"], panel_id="B0-EMA", K_values=[1, 2],
                    physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]), checksums_sha256=p["checksums_sha256"],
                    hidden_gt_training_usage="none", test_gt_usage="none", model_optimizer_steps=0,
                    old_raw_cache_reused=False, formal_gate1c_budget_consumed=0)
    write_new(args.output / "K2_REPLICATION_RUN_METADATA.json", dict(metadata, started_at=now()))
    entries = []
    with no_updates():
        for checkpoint in baseline["checkpoints"]:
            models, payload = load_models(ROOT, checkpoint, device="cuda:0", sources=("ema_teacher",))
            forward_counts = []
            hook = models["ema_teacher"].register_forward_hook(lambda *unused: forward_counts.append(1))
            with ImmutableModels(models, checkpoint, args.output, metadata):
                for role in ("train_labeled", "val"):
                    unit = next(u for u in plan["units"] if (u["seed"], u["stage_index"], u["role"]) ==
                                (checkpoint["seed"], checkpoint["stage_index"], role))
                    context = dict(panel_id="B0-EMA", seed=checkpoint["seed"], stage_index=checkpoint["stage_index"],
                        domain=checkpoint["domain"], role=role, source="ema_teacher", checkpoint_id=checkpoint["checkpoint_id"],
                        checkpoint_sha256=checkpoint["sha256"], sampling_plan_sha256=p["sampling_plan_sha256"], sampling_unit_sha256=H(unit))
                    if unit["domain"] != checkpoint["domain"]:
                        raise RuntimeError("current-domain mismatch")
                    before = len(forward_counts)
                    arrays, cases = extract_unit(models["ema_teacher"], unit, args.data_root, context, device="cuda:0", batch_size=8)
                    forwards = len(forward_counts) - before
                    if forwards != math.ceil(len(unit["cases"]) / 8):
                        raise RuntimeError("unexpected replication forward count")
                    caches = []
                    for c, group in arrays.items():
                        layout = sample_layout(unit, c)
                        cache = dict(class_id=c, registered_count=len(layout["uids"]), uid_order_sha256=H(layout["uids"]),
                            original_weight_order_sha256=weight_hash(layout["weights"]), active_count=int(group["active_mask"].sum()),
                            null_count=int((~group["active_mask"]).sum()), arrays={})
                        for name, array in group.items():
                            rel = Path("features") / f"seed{checkpoint['seed']}_stage{checkpoint['stage_index']}_{role}_class{c}_{name}.npy"
                            path = args.output / rel; path.parent.mkdir(exist_ok=True)
                            with path.open("xb") as handle:
                                np.save(handle, array, allow_pickle=False)
                            cache["arrays"][name] = dict(path=str(rel), shape=list(array.shape), dtype=str(array.dtype), sha256=sha256(path))
                        caches.append(cache)
                    entry = dict(context, metadata=metadata, class_caches=caches, case_support=cases, model_forwards=forwards,
                                 all_finite=True, null_rows_preserved=True, old_raw_cache_reused=False)
                    folder = args.output / "feature_units"; folder.mkdir(exist_ok=True)
                    write_new(folder / f"seed{checkpoint['seed']}_stage{checkpoint['stage_index']}_{role}.json", entry)
                    entries.append(entry)
                    print(f"K2 features complete seed={checkpoint['seed']} stage={checkpoint['stage_index']} {role}", flush=True)
            hook.remove()
            del models, payload
            torch.cuda.empty_cache()
    if len(entries) != 18 or len(list((args.output / "immutability").glob("*.json"))) != 9:
        raise RuntimeError("incomplete feature/model guard barrier")
    for guard in (args.output / "immutability").glob("*.json"):
        g = read(guard)
        if g["status"] != "PASS" or not g["bitwise_unchanged"] or not g["extraction_completed"] or g["metadata"] != metadata:
            raise RuntimeError("failed feature/model guard")
    write_new(args.output / "K2_FEATURE_START_BARRIER.json", dict(status="PASS", feature_units=18, model_guards=9,
                                                               geometry_jobs_started=0))
    tasks = []
    for seed in range(3):
        for stage in range(3):
            es = {r: next(e for e in entries if (e["seed"], e["stage_index"], e["role"]) == (seed, stage, r))
                  for r in ("train_labeled", "val")}
            us = {r: next(u for u in plan["units"] if (u["seed"], u["stage_index"], u["role"]) == (seed, stage, r))
                  for r in ("train_labeled", "val")}
            for c in range(3):
                for k in (1, 2):
                    tasks.append((str(args.output), {"shared_sampling": old["shared_sampling"]}, metadata, es, us, c, k))
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [read(path) for path in pool.map(geometry_job, tasks)]
    verdict = adjudicate(rows)
    report = dict(**verdict, metadata=metadata, units=rows, completed_at=now(),
                  model_forwards=sum(e["model_forwards"] for e in entries), model_forwards_role="new K2 replication features only")
    write_new(args.output / "K2_REPLICATION_REPORT.json", report)
    (args.output / "K2_REPLICATION_REPORT.md").write_text(
        "# B0-EMA K2 replication\n\n" + verdict["status"] + "\n\n" + json.dumps(verdict["statistics"], indent=2) +
        "\n\nOnly K1/K2 and regenerated B0-EMA were evaluated. Foreground-only admission; null rows retained. No new K selection.\n")
    write_csv(args.output / "K2_REPLICATION_GEOMETRY.csv", [dict(seed=r["seed"], stage=r["stage_index"], domain=r["domain"],
        class_id=r["class_id"], K=r["K"], role=role, R95_null_worst_case=m["R95_null_worst_case"],
        null_mass=m["null_mass"], registered_count=m["registered_count"]) for r in rows for role, m in r["metrics"].items()])
    if verdict["status"] == "K2_REPLICATION_PASS":
        records = [dict(seed=r["seed"], stage_index=r["stage_index"], domain=r["domain"], class_id=r["class_id"],
            panel="B0-EMA", baseline="B0", feature_source="ema_teacher", K=2, training_source="train_labeled",
            converged=r["fit"]["converged"], centers=r["fit"]["centers"], active_mask=r["fit"]["active"],
            operational_refit_allowed=False) for r in rows if r["K"] == 2]
        write_new(args.output / "K2_REPLICATION_FREEZE.json", dict(selected_K=2, primary_panel="B0-EMA",
            baseline_manifest_sha256=p["baseline_manifest_sha256"], prototype_records=records,
            replication_report_sha256=sha256(args.output / "K2_REPLICATION_REPORT.json")))
    print(json.dumps(verdict, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    run(parser.parse_args())
