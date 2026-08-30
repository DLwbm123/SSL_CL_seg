"""One locked, model-independent coordinate plan, shared by every panel."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from .binding import H, ROLES, check_hash, gate1a_records, require, safe_asset, write_json, write_text
from .geometry_metrics import boundary_band


def select_coordinates(labels, class_id, quota, seed, role, case_id):
    coordinates = np.argwhere(labels == class_id)
    ranked = sorted((H(["geometry-pixel-v1", seed, role, case_id, class_id, int(y), int(x)]), int(y), int(x))
                    for y, x in coordinates)
    return [[y, x] for _, y, x in ranked[:quota]]


def plan_from_labels(records, labels, *, seed, stage, domain, role, pixel_seed):
    require(role in ROLES, f"Gate1A forbidden sampling role: {role}")
    counts = [[int((array == c).sum()) for c in range(3)] for array in labels]
    positives = [n for row in counts for n in row if n > 0]
    require(bool(positives), "no valid sampling pixels")
    quota = min(2048, min(positives))
    for c in (1, 2):
        require(any(row[c] > 0 for row in counts), f"required foreground unit missing: {seed}/{domain}/{role}/{c}")
    cases = []
    for record, array, count in zip(records, labels, counts):
        require(array.ndim == 2 and set(np.unique(array)).issubset({0, 1, 2, 255}), "invalid stored label mapping")
        classes = []
        for c in range(3):
            coords = select_coordinates(array, c, quota, pixel_seed, role, record["case_id"])
            band = boundary_band(array, c)
            classes.append(dict(class_id=c, available_pixels=count[c], sampled_pixels=len(coords),
                                coordinates=coords, boundary=[bool(band[y,x]) for y,x in coords],
                                multiplicity=1, zero_count=count[c] == 0,
                                coordinate_sha256=H([[record["case_id"],c,y,x] for y,x in coords])))
        cases.append(dict(case_id=record["case_id"], image_h5_relpath=record.get("image_h5_relpath"),
                          label_h5_relpath=record.get("label_h5_relpath"), image_sha256=record.get("image_sha256"),
                          label_sha256=record.get("label_sha256"), classes=classes))
    return dict(seed=seed, stage_index=stage, domain=domain, role=role, pixel_sampling_seed=pixel_seed,
                common_quota=quota, case_count=len(cases), cases=cases,
                uid_schema="[case_id,class_id,y,x]; coordinates and class/case prefixes are stored losslessly",
                gt_consumer="diagnostic_evaluator_only" if role == "val" else "labeled_prototype_fit_only")


def _materialize_unit(task):
    data_root, prereg, case_plan, role = task
    seed, domain, stage = case_plan["seed"], case_plan["domain"], case_plan["stage_index"]
    rows = gate1a_records(data_root, prereg, seed, domain, role)
    labels = []
    for row in rows:
        path = safe_asset(data_root, row["label_h5_relpath"])
        check_hash(path, row["label_sha256"])
        with h5py.File(path, "r") as handle:
            labels.append(handle["label"][...])
        require(labels[-1].shape == (384,384), "stored label shape changed")
    registered = next(p for p in prereg["shared_sampling"]["plans"] if p["seed"]==seed and p["stage_index"]==stage)
    result = plan_from_labels(rows, labels, seed=seed, stage=stage, domain=domain, role=role,
                              pixel_seed=registered["pixel_sampling_seed"])
    print(f"sampling complete seed={seed} domain={domain} role={role} quota={result['common_quota']}", flush=True)
    return result


def materialize_plan(data_root, prereg, output, *, workers=8):
    tasks = [(str(data_root), prereg, p, role) for p in prereg["benchmark"]["case_plans"] for role in ROLES]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        units = list(pool.map(_materialize_unit, tasks))
    plan = dict(schema_version=1, scope="GATE1A_ONLY", panel_independent=True, units=units,
                shared_panels=[p["panel_id"] for p in prereg["panels"]["definitions"]],
                all_K=[1,2,3,5], bootstrap_rule="immutable coordinates; registered case multiplicities only")
    path = Path(output)/"SHARED_GEOMETRY_SAMPLING_PLAN.json"
    digest = write_json(path, plan)
    write_text(path.with_suffix(".sha256"), digest+"  "+path.name+"\n")
    path.chmod(0o444)
    path.with_suffix(".sha256").chmod(0o444)
    audit = dict(status="PASS", sampling_plan_sha256=digest, units=len(units),
                 roles=list(ROLES), shared_panel_hashes={panel:digest for panel in plan["shared_panels"]},
                 plan_locked_before_feature_extraction=True, panel_specific_sampling=False,
                 test_role_constructions=0, train_unlabeled_constructions=0,
                 hidden_gt_training_usage="none", test_gt_usage="none",
                 quotas=[{k:u[k] for k in ("seed","domain","role","common_quota","case_count")} for u in units])
    write_json(Path(output)/"SAMPLING_PLAN_AUDIT.json",audit)
    return plan, digest


def sample_layout(unit, class_id):
    uids, boundary, case_ids, weights = [], [], [], []
    nonempty = [c for c in unit["cases"] if c["classes"][class_id]["sampled_pixels"] > 0]
    require(bool(nonempty), "missing class in sampling layout")
    for case in nonempty:
        cls = case["classes"][class_id]
        for coord, is_boundary in zip(cls["coordinates"], cls["boundary"]):
            uids.append((case["case_id"],class_id,*coord))
            boundary.append(is_boundary)
            case_ids.append(case["case_id"])
            weights.append(1/len(nonempty)/len(cls["coordinates"]))
    ranks = np.empty(len(uids), dtype=np.int64)
    for rank,index in enumerate(sorted(range(len(uids)), key=lambda i:uids[i])):
        ranks[index]=rank
    return dict(uids=uids, uid_rank=ranks, case_ids=np.asarray(case_ids),
                weights=np.asarray(weights,dtype=np.float64), boundary=np.asarray(boundary,dtype=bool))
