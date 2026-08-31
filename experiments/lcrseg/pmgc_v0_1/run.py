"""One create-only preparation, six-pair integration and 48-pair formal audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from di_dmpa_gate1.feature_extraction import _images, seed_after_load
from di_dmpa_gate1c_v2 import binding as b, execution as e
from di_dmpa_gate1c_v2.precision import attach_gradient_student, student_forward
from di_dmpa_gate1c_v3 import durable as d
from di_dmpa_gate1c_v3.inputs import load_models
from di_dmpa_gate1c_v3.baseline import verify_payload
from .core import (require, parameters, full_inventory, immutable, measured, COUNT_KEYS,
                   native_gaussian, raw_pair, constraint_sets, projections, CANDIDATES)
from .modes import fit_bank, assignments, old_correct, support_rows, guard_vjps
from . import evaluator as ev
from .protocol import authority, execution_gate, completed, ROOT, REPO


def preparation_counts(unit):
    n = len(unit["guard_batches"])
    return dict(native_forwards=2*n+16, fp64_forwards=n+16, native_autograd=0, fp64_autograd=12*n)


def model_bundle(p, unit, device):
    cp = b.checkpoint(p, unit["seed"], unit["stage_index"])
    previous = b.checkpoint(p, unit["seed"], unit["stage_index"]-1)
    models, payload = load_models(ROOT, cp, device=device)
    past, _ = load_models(ROOT, previous, device=device, sources=("ema_teacher",))
    models["old"] = past["ema_teacher"]
    models["student"].requires_grad_(True)
    full_inventory(parameters(models["student"]))
    attach_gradient_student(models, dict(diagnostic_precision="float64_shadow", _precision_contract_verified=True))
    full_inventory(parameters(models["gradient_student"]))
    bank = b.legacy_input(payload, cp, p, device)
    return models, bank, [cp, previous]


def prepare(models, bank, unit, geom, p, data_root, output, device):
    """Same kernel is used by the synthetic graph compiler and real preparation."""
    output = Path(output)
    rows = {r["case_id"]: r for r in b.records(data_root, p, unit["seed"], unit["stage_index"], "train_labeled")}
    require(set(rows) == set(unit["guard_case_ids"]), "whole labeled panel mismatch")
    caches, descriptors = {}, []
    with immutable(models, bank) as isolation, measured(models, preparation_counts(unit)) as (counts, trace):
        for batch in unit["guard_batches"]:
            selected = [rows[case] for case in batch["case_ids"]]
            images = _images([b.image_only(r) for r in selected], data_root).to(device)
            labels = np.stack(e.visible_labels(selected, data_root, role="train_labeled"))
            with torch.no_grad():
                seed_after_load(batch["old_classifier_seed"])
                old_logits, features = models["old"](images, stochastic_classifier=False)
                probability = old_logits.float().softmax(1)
            arrays = dict(features=features.cpu().numpy(), labels=labels.astype(np.uint8), old_probability=probability.cpu().numpy())
            desc = b.save_arrays(output/"old_labeled"/f'batch{batch["batch_index"]:02d}.npz', arrays)
            descriptors.append(dict(batch=batch, arrays=desc, rows=[b.image_only(r) for r in selected], role="train_labeled"))
            for i, case in enumerate(batch["case_ids"]):
                caches[case] = {name: values[i] for name, values in arrays.items()}
        centers, active, k1, fits = fit_bank(geom, caches)
        mode_maps = {}
        for case in unit["guard_case_ids"]:
            entry = caches[case]
            modes, directional = assignments(entry["features"][None], entry["labels"][None], centers, active)
            correct = old_correct(entry["old_probability"][None], entry["labels"][None])
            mode_maps[case] = dict(modes=modes[0], active=directional[0], old_correct=correct[0])
        support = support_rows(unit["guard_case_ids"], caches, mode_maps, active, fits)
        named = parameters(models["gradient_student"])
        count = sum(p.numel() for _, p in named)
        sup, old = np.zeros((6, count), np.float64), np.zeros((6, count), np.float64)
        guard_batches, before_batches, none = [], [], []
        for desc in descriptors:
            batch = desc["batch"]
            images = _images(desc["rows"], data_root).to(device)
            values = {name: np.stack([caches[c][name] for c in batch["case_ids"]]) for name in ("labels", "old_probability")}
            for name in ("modes", "active", "old_correct"):
                values[name] = np.stack([mode_maps[c][name] for c in batch["case_ids"]])
            labels = torch.as_tensor(values["labels"].astype(np.int64), device=device)
            modes = torch.as_tensor(values["modes"], device=device)
            correct = torch.as_tensor(values["old_correct"], device=device)
            probability = torch.as_tensor(values["old_probability"], device=device)
            sl, sf, dl, draw = student_forward(models, images, batch["student_classifier_seed"])
            stats = ev.statistics(dl, labels, modes, correct, probability)
            gs, go, masks = guard_vjps(dl, labels, modes, correct, probability, named, support)
            sup += gs; old += go
            gaussian = native_gaussian(models["student"], batch["student_classifier_seed"], draw["native_gaussian_sha256"])
            values["gaussian"] = gaussian.cpu().numpy()
            saved = b.save_arrays(output/"guard_panel"/f'batch{batch["batch_index"]:02d}.npz', values)
            guard_batches.append(dict(batch=batch, arrays=saved, rows=desc["rows"], role="train_labeled", native_draw=draw))
            before_batches.append(dict(batch_id=batch["batch_id"], case_ids=batch["case_ids"], **stats))
            none.append(dict(batch_id=batch["batch_id"], none_masks=masks))
            del sl, sf, dl
        guard_arrays = b.save_arrays(output/"guards.npz", dict(centers=centers, center_active=active, K1_centers=k1, supervised=sup, old=old))
        d.write_new(output/"MODE_GUARD_FREEZE.json", dict(unit_id=unit["unit_id"], support=support, fits=fits,
                    arrays=guard_arrays, role="train_labeled", validation_GT_received=False, frozen_before_validation=True))
        validation = ev.prepare_validation(models, centers, active, unit, p, data_root, output/"evaluator", device)
    return dict(status="PASS_PREPARATION_ENGINEERING", unit=unit, source_checkpoint_ids=[unit["previous_checkpoint_id"], unit["current_checkpoint_id"]],
                mode_support=support, fit_records=fits, guard_arrays=guard_arrays, old_labeled_caches=descriptors,
                guard_none_masks=none, counts=counts, call_trace=trace, isolation=isolation,
                validation=validation, train_labeled=dict(role="train_labeled", descriptors=guard_batches,
                    before_batches=before_batches, before=ev.aggregate(before_batches), case_ids=unit["guard_case_ids"]),
                validation_GT_in_guards=False, online_updates=False, K2_reduction=False)


def pair_kernel(models, bank, pair, inputs, unit, prepared, panels, expected):
    arrays = b.read_arrays(prepared["guard_arrays"])
    with immutable(models, bank) as isolation, measured(models, expected) as (counts, trace):
        raw, raw_arrays = raw_pair(models, bank, pair, inputs)
        sets = constraint_sets(raw_arrays["fp64_supervised"], arrays["supervised"], arrays["old"], prepared["mode_support"])
        projected, directions = projections(raw_arrays["fp64_g0"], sets)
        # Projection is fully determined and sealed in memory before any evaluator call.
        direction_hashes = {candidate: b.array_hash(value) for candidate, value in directions.items()}
        evaluated = {candidate: ev.candidate(models["gradient_student"], directions[candidate], projected["P0"]["norm"], panels)
                     for candidate in CANDIDATES}
        require(direction_hashes == {candidate: b.array_hash(value) for candidate, value in directions.items()}, "evaluator mutated directions")
    return dict(raw=raw, pair=pair, unit_id=unit["unit_id"], projections=projected, evaluated=evaluated,
                counts=counts, call_trace=trace, isolation=isolation,
                validation_GT_received_by_projection=False, direction_hashes_before_evaluator=direction_hashes), dict(raw_arrays, **directions)


def allowed_data_inventory(reg, p):
    """Hash only used image/train-label/val-label files; never open hidden/test GT."""
    root = reg["destination"]["data_root"]
    selected = {}

    def add(row, labels):
        for kind in (("image", "label") if labels else ("image",)):
            relative = row[kind+"_h5_relpath"]
            key = (relative, row[kind+"_sha256"])
            selected[key] = dict(path=str(b.safe_asset(root, relative)), sha256=key[1], kind=kind,
                                 role=row["primary_20pct_split"])

    for unit in reg["fixed_units"]:
        train = b.records(root, p, unit["seed"], unit["stage_index"], "train_labeled")
        for row in train: add(row, True)
        unlabeled = {r["case_id"]: r for r in b.records(root, p, unit["seed"], unit["stage_index"], "train_unlabeled")}
        for pair in unit["formal_pairs"]:
            for case in pair["unlabeled_case_ids"]: add(unlabeled[case], False)
        for panel in unit["validation"].values():
            rows = {r["case_id"]: r for r in b.records(root, p, unit["seed"], panel["domain_stage"], "val")}
            for case in panel["case_ids"]: add(rows[case], True)
    result = sorted(selected.values(), key=lambda x: x["path"])
    for row in result:
        b.check_hash(row["path"], row["sha256"])
    return result


def input_audit(reg, p, output):
    roots = []
    gate1c = reg["inputs"]["gate1c_private_bundle"]
    mmpr = reg["inputs"]["mmpr_private_archive"]
    for spec in (dict(root=gate1c["root"], filename=gate1c["manifest_filename"], manifest_sha256=gate1c["manifest_sha256"], content_sha256=gate1c["content_sha256"], files=gate1c["logical_files"], bytes=gate1c["logical_bytes"]),
                 dict(root=mmpr["archive"], filename="PRIVATE_BUNDLE_MANIFEST.json", manifest_sha256=mmpr["manifest_sha256"], content_sha256=mmpr["content_sha256"], files=mmpr["files"], bytes=mmpr["bytes"])):
        b.check_hash(Path(spec["root"])/spec["filename"], spec["manifest_sha256"])
        found = d.verify(spec["root"], spec["filename"])
        require(all(found[key] == spec[key] for key in ("content_sha256", "files", "bytes")), "historical bundle identity", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
        roots.append(dict(spec, status="PASS", checked_at=d.now(), every_file_verified=True))
        print("verified immutable bundle", spec["root"], flush=True)
    checksum = Path(reg["destination"]["data_root"])/"checksums/checksums.sha256"
    b.check_hash(checksum, reg["inputs"]["checksums_sha256"])
    require(len(checksum.read_text().splitlines()) == 2962, "checksum inventory size changed")
    assets = allowed_data_inventory(reg, p)
    cp_rows = []
    for cp in reg["inputs"]["checkpoints"]:
        b.check_hash(cp["path"], cp["sha256"])
        payload = torch.load(cp["path"], map_location="cpu", weights_only=False)
        verify_payload(payload)
        legacy = b.legacy_input(payload, cp, p, "cpu")
        require(b.tensor_hash(legacy) == cp["legacy_pas_tensor_sha256"], "direct PAS bank changed")
        cp_rows.append(dict(checkpoint_id=cp["checkpoint_id"], path=cp["path"], sha256=cp["sha256"],
                            direct_PAS_sha256=b.tensor_hash(legacy), reconstructed=False))
    for key in ("baseline_manifest", "k2_freeze"):
        spec = reg["inputs"][key]; b.check_hash(spec["path"], spec["sha256"])
    for key in ("sampling_plan", "geometry_registration"):
        spec = reg["inputs"][key]; b.check_hash(REPO/spec["path"], spec["sha256"])
    result = dict(status="PASS_INPUT_AUDIT", historical_bundles=roots, checkpoints=cp_rows,
                  used_HDF5_assets=assets, used_HDF5_count=len(assets), checksum_inventory_entries=2962,
                  complete_historical_data_audit_preserved=True, unused_hidden_test_GT_files_opened=0,
                  validation_GT_tensor_reads=0, model_forwards=0, autograd_calls=0, checked_at=d.now())
    d.write_new(output/"PMGC_INPUT_AUDIT.json", result)
    return result


def unit_run(args, reg, p, metadata):
    unit = next(u for u in reg["fixed_units"] if u["unit_id"] == args.unit)
    phase = args.phase.removesuffix("_unit")
    require(args.output == Path(reg["destination"]["root"])/phase/"units"/unit["unit_id"], "unit output identity")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(unit["physical_gpu"]), "wrong registered physical GPU")
    device = torch.device("cuda:0")
    models, bank, cps = model_bundle(p, unit, device)
    if phase == "preparation":
        spec = reg["inputs"]["sampling_plan"]
        b.check_hash(REPO/spec["path"], spec["sha256"])
        plan = d.read(REPO/spec["path"])
        geom = next(g for g in plan["units"] if (g["seed"], g["stage_index"], g["role"]) == (unit["seed"], unit["stage_index"], "train_labeled"))
        require(b.H(geom) == unit["geometry_sampling_unit_sha256"], "geometry unit changed")
        result = prepare(models, bank, unit, geom, p, reg["destination"]["data_root"], args.output, device)
        name = "PREPARATION_UNIT.json"
    else:
        source = Path(reg["destination"]["root"])/"preparation/units"/unit["unit_id"]/"PREPARATION_UNIT.json"
        prepared = d.read(source)
        require(prepared["unit"] == unit and prepared["metadata"]["code_commit"] == args.code_commit, "preparation binding differs")
        panels = {side: ev.load_panel(prepared["validation"][side], reg["destination"]["data_root"], device, role="val") for side in ("previous", "current")}
        panels["train_labeled"] = ev.load_panel(prepared["train_labeled"], reg["destination"]["data_root"], device, role="train_labeled")
        pairs = unit["formal_pairs"] if phase == "formal" else [next(p for p in unit["formal_pairs"] if p["batch_id"] == unit["integration_pair_id"])]
        results, totals = [], dict.fromkeys(COUNT_KEYS, 0)
        for pair in pairs:
            inputs = tuple(x.to(device) for x in e.pair_inputs(reg["destination"]["data_root"], p, pair))
            row, arrays = pair_kernel(models, bank, pair, inputs, unit, prepared, panels, reg["call_graph"]["per_pair_by_stage"][str(unit["stage_index"])])
            row["before"] = {side: prepared["validation"][side]["before"] for side in ("previous", "current")}
            row["before"]["train_labeled"] = prepared["train_labeled"]["before"]
            row["before_batches"] = {side: prepared["validation"][side]["before_batches"] for side in ("previous", "current")}
            row["before_batches"]["train_labeled"] = prepared["train_labeled"]["before_batches"]
            directory = args.output/f'pair{pair["pair_index"]:02d}'; directory.mkdir()
            row["arrays"] = b.save_arrays(directory/"vectors.npz", arrays)
            row["metadata"] = metadata
            row["preparation_sha256"] = d.sha256(source)
            d.write_new(directory/"PAIR_RESULT.json", row)
            results.append(dict(pair=pair, path=str(directory/"PAIR_RESULT.json"), sha256=d.sha256(directory/"PAIR_RESULT.json")))
            for key in COUNT_KEYS: totals[key] += row["counts"][key]
            print("completed", phase, unit["unit_id"], pair["pair_index"], flush=True)
        result = dict(status="PASS_PAIR_EXECUTION_ENGINEERING", unit=unit, pairs=results, counts=totals)
        name = "PAIR_UNIT.json"
    for cp in cps: b.check_hash(cp["path"], cp["sha256"])
    result.update(metadata=metadata, checkpoint_hashes_unchanged=True)
    d.write_new(args.output/name, result)


def controller(args, reg, p, metadata):
    root = Path(reg["destination"]["root"])
    require(args.output == root/args.phase, "phase output differs")
    if args.phase == "preparation":
        completed(root/"input_audit", "input_audit", "PMGC_INPUT_AUDIT.json")
    else:
        completed(root/"preparation", "preparation", "PREPARATION_REPORT.json")
    if args.phase == "formal":
        completed(root/"integration", "integration", "INTEGRATION_EXECUTION.json")
        audit = d.read(root/"control/PMGC_INTEGRATION_REPORT.json")
        require(audit["status"] == "PASS_PMGC_INTEGRATION" and audit["code_commit"] == args.code_commit, "integration artifact audit failed")
        archive = d.read(root/"control/PMGC_INTEGRATION_ARCHIVE.json")
        require(archive["status"] == "PASS_PRIVATE_ARCHIVE", "integration archive absent")
        b.check_hash(Path(archive["archive"])/"PRIVATE_BUNDLE_MANIFEST.json", archive["manifest_sha256"])
        d.verify(archive["archive"])
    (args.output/"units").mkdir()

    def queue(gpu):
        done = []
        for unit in [u for u in reg["fixed_units"] if u["physical_gpu"] == gpu]:
            output = args.output/"units"/unit["unit_id"]
            phase = args.phase+"_unit"
            command = [sys.executable, "-B", "-m", "pmgc_v0_1.run", "--phase", phase, "--unit", unit["unit_id"],
                       "--output", str(output), "--code-commit", args.code_commit, "--gate", str(args.gate)]
            d.launch(output, phase, command, cwd=ROOT, env={"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            while not (output/f"PHASE_{phase}_MANIFEST.json").exists():
                if (output/"PROCESS_EXIT.json").exists():
                    require(d.read(output/"PROCESS_EXIT.json")["actual_child_exit_code"] == 0, "unit child failed: "+unit["unit_id"], "BLOCKED_INCOMPLETE_EVIDENCE")
                time.sleep(1)
            name = "PREPARATION_UNIT.json" if args.phase == "preparation" else "PAIR_UNIT.json"
            value = completed(output, phase, name)
            done.append(dict(unit_id=unit["unit_id"], path=str(output/name), sha256=d.sha256(output/name), counts=value["counts"]))
        return done

    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(queue, (4, 5, 6, 7)))
    units = sorted([u for group in groups for u in group], key=lambda u: u["unit_id"])
    require([u["unit_id"] for u in units] == [u["unit_id"] for u in reg["fixed_units"]], "phase unit coverage")
    counts = {key: sum(u["counts"][key] for u in units) for key in COUNT_KEYS}
    expected = reg["call_graph"][args.phase]
    require(counts == {key: expected[key] for key in COUNT_KEYS}, "phase total graph differs", "BLOCKED_CALL_GRAPH_MISMATCH")
    name = "PREPARATION_REPORT.json" if args.phase == "preparation" else args.phase.upper()+"_EXECUTION.json"
    d.write_new(args.output/name, dict(status="PASS_PHASE_ENGINEERING", metadata=metadata, units=units, counts=counts,
                                      real_forwards=counts["native_forwards"]+counts["fp64_forwards"],
                                      scientific_admission=None, one_create_only_attempt=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("input_audit", "preparation", "integration", "formal", "preparation_unit", "integration_unit", "formal_unit"))
    parser.add_argument("--unit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    try:
        reg, p, metadata = execution_gate(args)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with b.no_updates():
            if args.phase == "input_audit":
                require(args.output == Path(reg["destination"]["root"])/"input_audit", "input audit root")
                from di_dmpa_gate1c_v2.full_precision import forbid_forwards
                with forbid_forwards(): input_audit(reg, p, args.output)
            elif args.phase.endswith("_unit"):
                unit_run(args, reg, p, metadata)
            else:
                controller(args, reg, p, metadata)
    except BaseException as error:
        d.write_new(args.output/"PMGC_BLOCKED.json", dict(status=getattr(error, "status", "BLOCKED_EXECUTION_FAILURE"),
                     error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
                     model_optimizer_steps=0, transport_optimizer_steps=0, training_launched=False))
        raise


if __name__ == "__main__":
    main()
