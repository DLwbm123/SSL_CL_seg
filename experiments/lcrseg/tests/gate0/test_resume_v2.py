"""Full production state-machine tests on explicitly synthetic, hashed HDF5 fixtures."""
import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from di_dmpa_jascl.config import load_yaml, sha256_file
from di_dmpa_jascl.metrics import write_json
from di_dmpa_jascl.runner import Gate0RepairedRunner
from scripts.verify_resume_equivalence import compare, GROUPS
from .test_model_checkpoint import TinySegNet

ROOT = Path(__file__).resolve().parents[2]


def synthetic_bundle(root):
    config = load_yaml(ROOT / "configs/gate0_repaired_v2/fundus_pas_probmse.yaml")
    protocol = load_yaml(ROOT / "docs/di_dmpa_jascl/DOMAIN_PROTOCOL.yaml")
    root.mkdir()
    (root / "h5/v1").mkdir(parents=True)
    (root / "manifests/training").mkdir(parents=True)
    (root / "splits").mkdir()
    rng = np.random.default_rng(15)
    rows = []
    for domain in config["data"]["domain_order"]:
        for role, count in (("train_labeled", 3), ("train_unlabeled", 3), ("val", 1), ("test", 1)):
            for index in range(count):
                stem = f"{domain}_{role}_{index}"
                with h5py.File(root / f"h5/v1/{stem}_image.h5", "w") as handle:
                    handle["image"] = rng.integers(0, 256, (3, 16, 16), dtype=np.uint8)
                label_rel = ""
                if role != "train_unlabeled":
                    label_rel = f"{stem}_label.h5"
                    with h5py.File(root / "h5/v1" / label_rel, "w") as handle:
                        handle["label"] = np.tile(np.arange(16) % 3, (16, 1)).astype(np.uint8)
                rows.append(dict(case_id=stem, patient_id=stem, dataset="fundus", site_or_vendor=domain,
                                 primary_20pct_split=role, image_h5_relpath=f"{stem}_image.h5",
                                 label_h5_relpath=label_rel, label_sha256=""))
    manifest = root / "manifests/training/lcrseg_v1_seed0.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    split = root / "splits/fundus_seed0.json"
    split.write_text("{\"synthetic_fixture\":true}\n")
    protocol["frozen_seed_assets"][0]["training_manifest_sha256"] = sha256_file(manifest)
    protocol["frozen_seed_assets"][0]["fundus_split_sha256"] = sha256_file(split)
    protocol["benchmarks"]["fundus"]["spatial_preprocessing"]["stored_resize_hw"] = [16, 16]
    config["data"]["root"] = str(root)
    return config, protocol


def check_trajectory(tmp_path, case, cut, target):
    config, protocol = synthetic_bundle(tmp_path / "data")
    def runner(output):
        return Gate0RepairedRunner(repo_root=ROOT, config=config, protocol=protocol, seed=0,
                                   output_dir=output, device="cpu", model_factory=TinySegNet)
    reference = runner(tmp_path / "reference")
    reference.run(stop_after_global_step=target)
    first = runner(tmp_path / "candidate")
    result = first.run(**({"stop_at_event": cut} if isinstance(cut, str) else {"stop_after_global_step": cut}))
    checkpoint = torch.load(result["checkpoint"], weights_only=False)
    if case == "mid_unlabeled":
        assert checkpoint["sampler_state"]["phase"] == "unlabeled"
        assert checkpoint["prototypes"] is not None
    resumed = runner(tmp_path / "candidate")
    resumed.run(resume_path=result["checkpoint"], stop_after_global_step=target)
    left_path, right_path = tmp_path / "reference/last.pt", tmp_path / "candidate/last.pt"
    left, right = (torch.load(path, weights_only=False) for path in (left_path, right_path))
    groups = {}
    for group in GROUPS:
        matched, maximum = compare(left[group], right[group], atol=1e-6, rtol=1e-6)
        groups[group] = {"within_tolerance": matched, "max_abs_difference": maximum}
        assert matched, group
    x = torch.arange(768).float().reshape(1, 3, 16, 16) / 768
    with torch.no_grad():
        a, _ = reference.wrapper.student(x, stochastic_classifier=False)
        b, _ = resumed.wrapper.student(x, stochastic_classifier=False)
    matched, maximum = compare(a, b, atol=1e-6, rtol=1e-6)
    assert matched
    groups["deterministic_evaluation_output"] = {"within_tolerance": matched, "max_abs_difference": maximum}
    report = {"status": "PASS", "groups": groups, "reference": str(left_path), "candidate": str(right_path),
              "reference_sha256": sha256_file(left_path), "candidate_sha256": sha256_file(right_path),
              "interruption": cut, "target_global_step": target, "config_hash": reference.config_hash,
              "data_kind": "synthetic_hashed_hdf5", "model": "TinySegNet_with_explicit_stochastic_features"}
    if os.environ.get("GATE0_RESUME_REPORT_DIR"):
        write_json(Path(os.environ["GATE0_RESUME_REPORT_DIR"]) / f"{case}.json", report)


def test_resume_mid_supervised(tmp_path):
    check_trajectory(tmp_path, "mid_supervised", 1, 6)


def test_resume_inside_unlabeled_phase(tmp_path):
    check_trajectory(tmp_path, "mid_unlabeled", 53, 55)


@pytest.mark.parametrize("event", ["before_stage_transition", "after_stage_transition"])
def test_resume_across_stage_boundary(tmp_path, event):
    check_trajectory(tmp_path, event, event, 235)
