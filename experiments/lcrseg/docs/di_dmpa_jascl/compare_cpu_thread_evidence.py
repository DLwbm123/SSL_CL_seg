#!/usr/bin/env python3
"""Read-only cross-thread evidence comparison; no data or optimizer access."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from di_dmpa_jascl.config import sha256_file
from scripts.verify_resume_equivalence import GROUPS, compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite resource evidence")
    names = ["UNIT_INTEGRATION_TEST_REPORT.json", "RESUME_EQUIVALENCE_REPORT.json", "PAS_GRADIENT_AUDIT.json"]
    sources = {}
    for label, root in (("reference", args.reference_dir), ("candidate", args.candidate_dir)):
        sources[label] = {name: json.loads((root / name).read_text()) for name in names}
        assert all(report["status"] == "PASS" for report in sources[label].values())
    a, b = sources["reference"], sources["candidate"]
    assert a[names[0]]["git_commit"] == b[names[0]]["git_commit"]
    assert a[names[0]]["config_hashes"] == b[names[0]]["config_hashes"]
    trajectories = {}
    for name, original in a[names[1]]["trajectories"].items():
        adjusted = b[names[1]]["trajectories"][name]
        left_path, right_path = Path(original["reference"]), Path(adjusted["reference"])
        left, right = [torch.load(path, map_location="cpu", weights_only=False) for path in (left_path, right_path)]
        groups = {}
        for group in GROUPS:
            matched, maximum = compare(left[group], right[group], atol=1e-6, rtol=1e-6)
            groups[group] = {"within_tolerance": matched, "max_abs_difference": maximum}
        trajectories[name] = {"groups": groups, "reference": str(left_path), "candidate": str(right_path),
                              "reference_sha256": sha256_file(left_path), "candidate_sha256": sha256_file(right_path)}
    gradients = {}
    for domain, original in a[names[2]]["domains"].items():
        adjusted = b[names[2]]["domains"][domain]
        fields = ("valid_student_pixels", "valid_teacher_pixels", "joint_valid_pixels", "joint_coverage",
                  "consistency_loss", "student_unsupervised_gradient_norm", "student_total_gradient_norm",
                  "total_minus_supervised_gradient_norm", "teacher_nonnull_gradient_count")
        differences = {field: abs(original[field]-adjusted[field]) for field in fields}
        gradients[domain] = {"max_absolute_differences": differences,
                             "within_tolerance": all(diff <= 1e-6 for diff in differences.values())}
    passed = all(group["within_tolerance"] for case in trajectories.values() for group in case["groups"].values())
    passed = passed and all(row["within_tolerance"] for row in gradients.values())
    report = {"status": "PASS" if passed else "FAIL", "git_commit": a[names[0]]["git_commit"],
              "atol": 1e-6, "rtol": 1e-6, "scope": "four synthetic actual-model trajectories plus three real-batch gradient audits",
              "cpu_quota_cores": 16, "baseline_observed_compute_threads_per_process": 56,
              "candidate_environment": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
              "candidate_test_count": b[names[0]]["passed"], "training_code_or_config_changed": False,
              "trajectories": trajectories, "real_batch_gradients": gradients,
              "report_sha256": {label: {name: sha256_file(root / name) for name in names}
                                for label, root in (("reference", args.reference_dir), ("candidate", args.candidate_dir))},
              "comparison_script_sha256": sha256_file(__file__)}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status": report["status"], "trajectories": len(trajectories), "real_batch_gradients": gradients}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
