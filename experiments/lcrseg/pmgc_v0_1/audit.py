"""Separate read-only artifact verification, fixed gates and public summaries."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from di_dmpa_gate1c_v2 import binding as b
from di_dmpa_gate1c_v2.full_precision import forbid_forwards
from di_dmpa_gate1c_v3 import durable as d
from .core import require, COUNT_KEYS
from .modes import assignments, old_correct, support_rows
from .protocol import execution_gate, completed
from .report import adjudicate, audit_pair, tables


def audit_modes(prepared):
    unit = prepared["unit"]
    arrays = b.read_arrays(prepared["guard_arrays"])
    require(arrays["supervised"].shape == arrays["old"].shape == (6, 484016), "mode guard full inventory")
    require(arrays["centers"].shape == (3, 2, 16) and arrays["K1_centers"].shape == (3, 1, 16), "bank K was reduced")
    b.finite(arrays["supervised"], arrays["old"], arrays["centers"], arrays["K1_centers"])
    caches = {}
    for desc in prepared["old_labeled_caches"]:
        require(desc["role"] == desc["batch"]["role"] == "train_labeled", "hidden/val mode source")
        batch = b.read_arrays(desc["arrays"])
        for i, case in enumerate(desc["batch"]["case_ids"]):
            require(case not in caches, "duplicate mode source case")
            caches[case] = {name: array[i] for name, array in batch.items()}
    require(list(caches) == unit["guard_case_ids"], "mode source UID coverage/order")
    maps = {}
    for case, entry in caches.items():
        mode, active = assignments(entry["features"][None], entry["labels"][None], arrays["centers"], arrays["center_active"])
        correct = old_correct(entry["old_probability"][None], entry["labels"][None])
        maps[case] = dict(modes=mode[0], active=active[0], old_correct=correct[0])
    support = support_rows(unit["guard_case_ids"], caches, maps, arrays["center_active"], prepared["fit_records"])
    require(support == prepared["mode_support"], "independent mode support/UID audit differs")
    for desc in prepared["train_labeled"]["descriptors"]:
        values = b.read_arrays(desc["arrays"])
        for i, case in enumerate(desc["batch"]["case_ids"]):
            require(np.array_equal(values["labels"][i], caches[case]["labels"]), "guard GT differs")
            for name in ("modes", "active", "old_correct"):
                require(np.array_equal(values[name][i], maps[case][name]), "guard support map differs")
    return dict(status="PASS_MODE_ARTIFACT_AUDIT", unit_id=unit["unit_id"], cases=len(caches), modes=len(support),
                guard_array_sha256=prepared["guard_arrays"]["sha256"], full_parameters=484016,
                old_mean_recomputed=False, assignment_support_and_UID_recomputed=True, new_mode_fit_performed=False,
                hidden_GT_received=False, val_GT_received=False, model_forwards=0, autograd_calls=0)


def collect(reg, phase, code_commit):
    root = Path(reg["destination"]["root"])
    prep_report = completed(root/"preparation", "preparation", "PREPARATION_REPORT.json")
    phase_report = completed(root/phase, phase, phase.upper()+"_EXECUTION.json")
    require(len(prep_report["units"]) == len(phase_report["units"]) == 6, "audit requires exactly six unit records")
    prepared, rows, mode_audits, pair_audits = [], [], [], []
    for expected_unit, prep_ref, phase_ref in zip(reg["fixed_units"], prep_report["units"], phase_report["units"]):
        require(expected_unit["unit_id"] == prep_ref["unit_id"] == phase_ref["unit_id"], "audit unit order mismatch")
        b.check_hash(prep_ref["path"], prep_ref["sha256"])
        b.check_hash(phase_ref["path"], phase_ref["sha256"])
        prep, group = d.read(prep_ref["path"]), d.read(phase_ref["path"])
        require(prep["unit"] == group["unit"] == expected_unit, "audit unit metadata mismatch")
        require(prep["metadata"]["code_commit"] == group["metadata"]["code_commit"] == code_commit, "mixed execution source")
        require(prep["checkpoint_hashes_unchanged"] and group["checkpoint_hashes_unchanged"], "checkpoint guard failed")
        prepared.append(prep)
        mode_audits.append(audit_modes(prep))
        expected_pairs = expected_unit["formal_pairs"] if phase == "formal" else [p for p in expected_unit["formal_pairs"] if p["batch_id"] == expected_unit["integration_pair_id"]]
        require([r["pair"] for r in group["pairs"]] == expected_pairs, "integration/formal exact pair denominator")
        for ref in group["pairs"]:
            b.check_hash(ref["path"], ref["sha256"])
            row = d.read(ref["path"])
            require(row["pair"] == ref["pair"] and row["metadata"]["code_commit"] == code_commit and row["preparation_sha256"] == prep_ref["sha256"], "pair provenance differs")
            rows.append(row)
            pair_audits.append(audit_pair(row, prep, reg["call_graph"]["per_pair_by_stage"][str(expected_unit["stage_index"])]))
    require(len(prepared) == 6 and len(rows) == (48 if phase == "formal" else 6), "audit incomplete denominator")
    for cp in reg["inputs"]["checkpoints"]: b.check_hash(cp["path"], cp["sha256"])
    for key in ("baseline_manifest", "k2_freeze"):
        spec = reg["inputs"][key]; b.check_hash(spec["path"], spec["sha256"])
    counts = {key: sum(row["counts"][key] for row in rows) for key in COUNT_KEYS}
    require(counts == phase_report["counts"] == {key: reg["call_graph"][phase][key] for key in COUNT_KEYS}, "audit/phase/registered counters differ")
    evidence = dict(status="PASS_INDEPENDENT_ARTIFACT_AUDIT", phase=phase, code_commit=code_commit,
                    pairs=len(rows), units=len(prepared), counts=counts, mode_audits=mode_audits, pair_audits=pair_audits,
                    checkpoints_banks_unchanged=True, historical_bundle_files_modified=False,
                    actual_child_exit_verified=True, full_phase_manifest_verified=True,
                    audit_model_forwards=0, audit_autograd_calls=0, independent_scientific_peer_review=False,
                    evaluator_check="Reaggregate saved per-batch loss sums and confusion matrices; do not rerun model predictions")
    return prepared, rows, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("integration", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    reg, _, metadata = execution_gate(args)
    with b.no_updates(), forbid_forwards(), patch.object(torch.autograd, "grad", side_effect=RuntimeError("artifact audit cannot compute autograd")):
        prepared, rows, audit = collect(reg, args.phase, args.code_commit)
        d.write_new(args.output/"PMGC_INDEPENDENT_ARTIFACT_AUDIT.json", audit)
        if args.phase == "integration":
            report = dict(status="PASS_PMGC_INTEGRATION", code_commit=args.code_commit, pairs=6, units=6,
                          registered_forwards=705, actual_forwards=sum(audit["counts"][key] for key in ("native_forwards", "fp64_forwards")),
                          counts=audit["counts"], independent_artifact_audit_sha256=d.sha256(args.output/"PMGC_INDEPENDENT_ARTIFACT_AUDIT.json"),
                          mode_bank_support_verified=True, solver_KKT_verified=True, parameter_inventory=dict(tensors=51,elements=484016),
                          model_optimizer_steps=0, transport_optimizer_steps=0, method_registered=False, training_launched=False,
                          scientific_verdict_not_adjudicated=True, required_next_step="ARCHIVE_INTEGRATION_THEN_ONE_FORMAL_ATTEMPT")
            d.write_new(args.output/"PMGC_INTEGRATION_REPORT.json", report)
            with (args.output/"PMGC_INTEGRATION_REPORT.md").open("x") as f:
                f.write("# PMGC integration\n\nPASS_PMGC_INTEGRATION: six new pairs, one per seed-transition; 705/705 real forwards. Exact51/484016 inventory, all solver certificates, fixed evaluator support, model/bank/checkpoint guards and durable actual child exit receipts passed independent artifact arithmetic checks. No scientific gate has been adjudicated from this small integration. Archive it before the one registered48-pair formal attempt. No optimizer, backward or training.\n")
            return
        expected = [p for u in reg["fixed_units"] for p in u["formal_pairs"]]
        verdict = adjudicate(prepared, rows, expected)
        verdict.update(code_commit=args.code_commit, registration_id=reg["registration_id"],
                       exact_real_forwards=reg["call_graph"]["total_real_forwards"], exact_real_autograd_calls=reg["call_graph"]["total_real_autograd_calls"],
                       independent_artifact_audit_status=audit["status"])
        d.write_new(args.output/"PMGC_STATUS.json", verdict)
        table_counts = tables(args.output, prepared, rows, verdict)
        d.write_new(args.output/"PMGC_MODE_BANK_MANIFEST.json", dict(status="PASS_MODE_ARTIFACT_AUDIT", code_commit=args.code_commit,
                    units=[dict(unit_id=p["unit"]["unit_id"], source_checkpoint_ids=p["source_checkpoint_ids"], support=p["mode_support"], fits=p["fit_records"],
                                private_arrays=p["guard_arrays"], role="train_labeled", validation_GT_received=False) for p in prepared]))
        d.write_new(args.output/"PMGC_GRADIENT_GUARD_AUDIT.json", dict(status="PASS_GRADIENT_GUARD_AUDIT", code_commit=args.code_commit,
                    formal_pairs=48, parameter_tensors=51, parameter_elements=484016, constraint_limit=13,
                    all_None_gradients_zero_filled=True, no_dropped_parameters=True, private_gradients_hash_verified=True,
                    whole_labeled_panel_guard_units=6, inactive_KD_rule="fewer than32 old-correct pixels; omitted from P4, never merged",
                    G2=verdict["G1_G7"]["G2"], G6=verdict["G1_G7"]["G6"], G7=verdict["G1_G7"]["G7"],
                    audit_sha256=d.sha256(args.output/"PMGC_INDEPENDENT_ARTIFACT_AUDIT.json")))
        d.write_new(args.output/"PMGC_VIRTUAL_STEP_DIAGNOSTIC.json", dict(status=verdict["status"], code_commit=args.code_commit, primary="P4",
                    formal_pairs=48, candidates=list(("P0","P1","P2","P3","P4")), formal_candidate_steps=240,
                    displacement_norm=.001, stateless=True, validation_classifier="posterior_mean", precision="float64_shadow",
                    metrics="fixed-panel valid-pixel CE and exact pooled confusion-matrix Dice; per-class/per-mode/KL diagnostics",
                    G1_G7=verdict["G1_G7"], failed_gates=verdict["failed_gates"], table_rows=table_counts,
                    validation_GT="evaluator_only", control_rescue=False, model_optimizer_steps=0))
        with (args.output/"PMGC_VIRTUAL_STEP_DIAGNOSTIC.md").open("x") as f:
            f.write("# PMGC virtual-step diagnostic\n\n"+verdict["status"]+". All48 fixed pairs and five candidates were evaluated with stateless FP64 displacements of norm0.001 (any invalid zero direction is explicitly marked). Previous/current validation uses posterior-mean classifiers, exact pooled Dice and fixed valid-pixel CE. Frozen mode guards use current labeled data only. See the paired CSV tables and full JSON gate values. Failed gates: "+", ".join(verdict["failed_gates"])+". Controls cannot rescue P4.\n")
        with (args.output/"PMGC_FAILURES_AND_WARNINGS.md").open("x") as f:
            f.write("# PMGC failures and limitations\n\nScientific status: "+verdict["status"]+". Failed gates: "+", ".join(verdict["failed_gates"])+". All failures remain in the final JSON; no threshold, seed, sample, denominator or method was changed after real execution.\n\nThis is a local normalized virtual-step feasibility diagnostic, not performance training or a clinical validation. It uses the immediately previous domain and fixed validation panels. Old-correct masks can deactivate modes; all such support is disclosed. Native repaired-B0 PAS/teacher and same-Gaussian FP64 parity are retained. Artifact audit recomputes counts/KKT/aggregations; it is not independent scientific peer review or a new model prediction run. Archives are on the same NAS, not an independent-device disaster backup. Existing training/legacy-real tests were prospectively excluded, not skipped; no historical test was changed.\n\nIf scientific feasibility failed, this project's prototype-derived new-method line is ended. The later long-running user authorization may pursue separately registered strong baselines and external-method comparisons; it cannot reopen selection, relation, transport, memory or gradient variants or reinterpret this result as success.\n")
        d.write_new(args.output/"PMGC_REPORT_COMPILATION.json", dict(status="PASS_REPORT_COMPILATION", metadata=metadata,
                    scientific_status=verdict["status"], table_rows=table_counts, complete48_denominator=True,
                    phase_actual_exits_verified=True, required_archive_before_publication=True))


if __name__ == "__main__":
    main()
