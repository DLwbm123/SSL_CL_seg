from __future__ import annotations

import csv

from scripts.evaluate_v0_2_gate import _admission_gate, _calibration_gate, _ess_gate, _rejection_cap_gate


def _write_csv(path, fields, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_v0_2_mechanism_gate_helpers_accept_registered_valid_evidence(tmp_path) -> None:
    run_dir = tmp_path / "r3"
    run_dir.mkdir()
    _write_csv(
        run_dir / "train_log.csv",
        ["site_id", "site_progress", "assim_candidate_counts_by_class", "assim_selected_fraction_by_class", "relation_valid_counts_by_class", "compat_rejected_fraction_by_class"],
        [
            {"site_id": "REFUGE", "site_progress": 0.0, "assim_candidate_counts_by_class": "[100, 100, 100]", "assim_selected_fraction_by_class": "[0.4, 0.4, 0.4]", "relation_valid_counts_by_class": "[100, 100, 100]", "compat_rejected_fraction_by_class": "[0.0, 0.2, 0.1]"},
            {"site_id": "REFUGE", "site_progress": 1.0, "assim_candidate_counts_by_class": "[100, 100, 100]", "assim_selected_fraction_by_class": "[0.8, 0.8, 0.8]", "relation_valid_counts_by_class": "[100, 100, 100]", "compat_rejected_fraction_by_class": "[0.1, 0.2, 0.0]"},
        ],
    )
    assert _admission_gate(run_dir)["passed"]
    assert _rejection_cap_gate(run_dir)["passed"]

    analysis_r3 = tmp_path / "analysis_r3"
    analysis_r3.mkdir()
    analysis_r0 = tmp_path / "analysis_r0"
    analysis_r0.mkdir()
    calibration_rows = []
    for site in ("RIM_ONE_r3", "Drishti_GS"):
        for scope, class_id in (("global", ""), ("class", "0"), ("class", "1"), ("class", "2")):
            for bin_index, probability in enumerate((0.4, 0.6, 0.8)):
                calibration_rows.append({"site_id": site, "epoch": "9", "scope": scope, "class_id": class_id, "bin": str(bin_index), "pava_probability": str(probability)})
    _write_csv(analysis_r3 / "calibration_tables.csv", ["site_id", "epoch", "scope", "class_id", "bin", "pava_probability"], calibration_rows)
    assert _calibration_gate(analysis_r3, ["REFUGE", "RIM_ONE_r3", "Drishti_GS"], 3)["passed"]

    fields = ["site_id", "route", "scope", "effective_pixel_count"]
    _write_csv(analysis_r0 / "effective_sample_size.csv", fields, [{"site_id": "RIM_ONE_r3", "route": "consolidation", "scope": "global", "effective_pixel_count": "100"}])
    _write_csv(analysis_r3 / "effective_sample_size.csv", fields, [{"site_id": "RIM_ONE_r3", "route": "consolidation", "scope": "global", "effective_pixel_count": "70"}])
    assert _ess_gate(analysis_r0, analysis_r3)["passed"]
