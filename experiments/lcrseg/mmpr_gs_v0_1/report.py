"""Fail-closed fixed-denominator F1-F5 adjudication and transparent tables."""
import csv
import json
from pathlib import Path

import numpy as np

from di_dmpa_gate1c_v2.gradients import BLOCKS
from .core import require
from .evaluator import relative


def csv_new(path, rows):
    require(bool(rows), "empty required table", "BLOCKED_INCOMPLETE_EVIDENCE")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def adjudicate(validation, results, expected_pairs):
    def finite_tree(value):
        if isinstance(value, dict):
            for child in value.values(): finite_tree(child)
        elif isinstance(value, (list, tuple)):
            for child in value: finite_tree(child)
        elif isinstance(value, (float, np.floating)):
            require(np.isfinite(value), "nonfinite report evidence", "BLOCKED_NUMERICAL_FAILURE")
    finite_tree(validation)
    finite_tree(results)
    require(len(expected_pairs) == 72 and len({p["batch_id"] for p in expected_pairs}) == 72, "registered72 required")
    require(len(results) == 72 and [r["pair"] for r in results] == expected_pairs,
            "formal pair coverage/order", "BLOCKED_INCOMPLETE_EVIDENCE")
    require(validation["case_count"] == 495 and validation["unit_count"] == 9,
            "validation495/9 coverage", "BLOCKED_INCOMPLETE_EVIDENCE")
    vrows = validation["mass_rows"]
    require(len(vrows) == 495 * 3 and len({(r["seed"], r["stage_index"], r["case_id"], r["class_id"]) for r in vrows}) == 495 * 3,
            "validation mass coverage", "BLOCKED_INCOMPLETE_EVIDENCE")
    prows = [m for r in results for m in r["mass_rows"]]
    require(len(prows) == 72 * 6, "pair mass row coverage", "BLOCKED_INCOMPLETE_EVIDENCE")
    F1 = all(m["mass_difference"] == 0 and m["R1_null_mass"] == m["MMPR_null_mass"]
             and not m["GT_received_by_builder"] for m in vrows + prows)
    units = [u for u in validation["units"] if u["candidate"] == "Q1" and u["class_id"] in (1, 2)]
    require(len(units) == 18 and {(u["seed"], u["stage_index"], u["class_id"]) for u in units} ==
            {(s, t, c) for s in range(3) for t in range(3) for c in (1, 2)}, "exact18 foreground denominator", "BLOCKED_INCOMPLETE_EVIDENCE")
    defined = all(u["weighted_error"] is not None and u["reference_error"] is not None and u["precision_drop"] is not None for u in units)
    ref = float(np.mean([u["reference_error"] for u in units])) if defined else None
    cand = float(np.mean([u["weighted_error"] for u in units])) if defined else None
    reduction = relative(ref, cand)
    improved = sum(u["weighted_error"] is not None and u["reference_error"] is not None and
                   u["weighted_error"] < u["reference_error"] for u in units)
    drops = [u["precision_drop"] for u in units]
    F2 = bool(defined and reduction is not None and reduction >= .10 and improved >= 12 and max(drops) <= .02)
    alignment, retention = [], []
    for result in results:
        require({(m["case_id"], m["class_id"]) for m in result["mass_rows"]} ==
                {(case, cls) for case in result["pair"]["unlabeled_case_ids"] for cls in range(3)},
                "pair case/class mass denominator", "BLOCKED_INCOMPLETE_EVIDENCE")
        require(len(result["alignment"]) == 2 and len(result["blockwise"]) == 12 and len(result["components"]) == 21
                and len(result["retention"]) == 7 and len(result["precision"]) == 21,
                "pair output coverage", "BLOCKED_INCOMPLETE_EVIDENCE")
        require({r["candidate"] for r in result["alignment"]} == {"Q0", "Q1"}, "gradient reference/primary missing")
        alignment.extend(result["alignment"])
        retention.extend(r for r in result["retention"] if r["block"] == "global")
    a0 = [r for r in alignment if r["candidate"] == "Q0"]
    a1 = [r for r in alignment if r["candidate"] == "Q1"]
    undefined = sum(r["cosine"] is None for r in a0 + a1)
    negative0 = sum(r["cosine"] is not None and r["cosine"] < 0 for r in a0)
    negative1 = sum(r["cosine"] is not None and r["cosine"] < 0 for r in a1)
    median0 = float(np.median([r["cosine"] for r in a0])) if not undefined else None
    median1 = float(np.median([r["cosine"] for r in a1])) if not undefined else None
    increase = median1-median0 if not undefined else None
    stage_worsening, stage_ratios = [], []
    for stage in range(3):
        x0, x1 = [[r["cosine"] for r in a if r["stage_index"] == stage] for a in (a0, a1)]
        require(len(x0) == len(x1) == 24, "exact24 pairs/stage", "BLOCKED_INCOMPLETE_EVIDENCE")
        stage_worsening.append(float(np.median(x0)-np.median(x1)) if all(v is not None for v in x0+x1) else None)
        values = [r["norm_ratio"] for r in retention if r["stage_index"] == stage]
        stage_ratios.append(float(np.median(values)) if len(values) == 24 and all(v is not None for v in values) else None)
    F3 = bool(not undefined and negative1 <= 43 and increase >= .05 and all(v is not None and v <= .05 for v in stage_worsening))
    ratios = [r["norm_ratio"] for r in retention]
    ratio_defined = len(ratios) == 72 and all(r is not None and np.isfinite(r) for r in ratios)
    med = float(np.median(ratios)) if ratio_defined else None
    p10 = float(np.quantile(ratios, .1, method="linear")) if ratio_defined else None
    dots_pass = sum(r["projected_dot"] >= -1e-10 for r in retention)
    zero = sum(r["projected_zero"] for r in retention)
    finite_values = all(np.isfinite(row[k]) for r in results for row in r["retention"]
                        for k in ("raw_dot", "projected_dot", "raw_norm", "projected_norm"))
    F4 = bool(ratio_defined and dots_pass == 72 and zero == 0 and med >= .50 and p10 >= .20 and
              all(v is not None and v >= .40 for v in stage_ratios) and finite_values)
    F5 = bool(validation["validation_GT"] == "evaluator_only" and validation["model_forwards"] == 0 and
              all(r["isolation"]["model_bitwise_unchanged"] and r["isolation"]["banks_unchanged"] and r["isolation"]["rng_restored"] and
                  r["isolation"]["teacher_gradients"] == r["isolation"]["bank_gradients"] == r["isolation"]["parameter_grad_fields"] == "None" and
                  r["isolation"]["model_optimizer_steps"] == r["isolation"]["transport_optimizer_steps"] == 0 and
                  not r["isolation"]["backward_called"] and not r["isolation"]["optimizer_constructed"] and
                  r["isolation"]["hidden_gt_training_usage"] == r["isolation"]["test_gt_usage"] == "none" and
                  r["checkpoint_guard_pass"] for r in results))
    gates = dict(F1=F1, F2=F2, F3=F3, F4=F4, F5=F5)
    failures = [status for ok, status in ((F2, "FAIL_MATCHED_MASS_RANKING_NOT_SUPPORTED"),
                (F3, "FAIL_RAW_GRADIENT_COMPATIBILITY"), (F4, "FAIL_PROJECTED_SIGNAL_RETENTION")) if not ok]
    status = "BLOCKED_PROTOCOL_OR_LEAKAGE" if not F1 or not F5 else (failures[0] if failures else "PASS_MMPR_GS_FEASIBILITY")
    return dict(status=status, gates=gates, all_scientific_failures=failures,
                F1=dict(validation_case_class_rows=len(vrows), formal_case_class_rows=len(prows), mass_differences=[r["mass_difference"] for r in vrows+prows]),
                F2=dict(reference_macro_error=ref, candidate_macro_error=cand, relative_error_reduction=reduction,
                        improving_units=improved, units=18, precision_drops=drops, required_metrics_defined=defined),
                F3=dict(reference_negative_count=negative0, candidate_negative_count=negative1, pairs=72, undefined=undefined,
                        reference_median=median0, candidate_median=median1, median_increase=increase, stage_worsening=stage_worsening),
                F4=dict(projected_dots_pass=dots_pass, projected_zero_count=zero, median_norm_ratio=med, p10_norm_ratio=p10,
                        stage_median_norm_ratios=stage_ratios, finite_values=finite_values),
                model_optimizer_steps=0, transport_optimizer_steps=0, method_registered=False, training_launched=False,
                historical_bank_claim_allowed=False, control_rescue_allowed=False,
                prototype_pseudo_label_selection_line="STOPPED_PENDING_INDEPENDENT_REVIEW",
                next_action="STOP_FOR_INDEPENDENT_REVIEW")
