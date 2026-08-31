"""Fixed-denominator G1-G7 adjudication; controls never rescue P4."""
from collections import Counter
from pathlib import Path

import numpy as np

from di_dmpa_gate1c_v2 import binding as b
from di_dmpa_gate1c_v3 import durable as d
from mmpr_gs_v0_1.report import csv_new
from .core import require, CANDIDATES, COUNT_KEYS, constraint_sets
from .evaluator import aggregate


def finite_tree(value):
    if isinstance(value, dict):
        for child in value.values(): finite_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value: finite_tree(child)
    elif isinstance(value, float):
        require(np.isfinite(value), "nonfinite report field", "BLOCKED_NUMERICAL_FAILURE")


def median(values):
    return float(np.median(values)) if values and all(v is not None for v in values) else None


def maximum(values):
    return float(max(values)) if values and all(v is not None for v in values) else None


def at_least(value, limit): return value is not None and value >= limit
def at_most(value, limit): return value is not None and value <= limit


def delta(row, candidate, panel, field, index=None):
    before = row["before"][panel][field]
    after = row["evaluated"][candidate]["panels"][panel]["after"][field]
    if index is not None: before, after = before[index], after[index]
    return after-before if before is not None and after is not None else None


def difference(a, b): return a-b if a is not None and b is not None else None


def comparative(rows, control, panel, field, index=None):
    return [difference(delta(r, control, panel, field, index), delta(r, "P4", panel, field, index)) for r in rows]


def improvement_summary(values):
    defined = bool(values) and all(v is not None for v in values)
    return dict(denominator=len(values), defined=defined, undefined=sum(v is None for v in values),
                better_count=sum(v is not None and v > 0 for v in values),
                better_fraction=sum(v is not None and v > 0 for v in values)/len(values) if values else None,
                median_improvement=median(values))


def adjudicate(prepared, rows, expected_pairs):
    require(len(expected_pairs) == 48 and len({p["batch_id"] for p in expected_pairs}) == 48, "registered48 required")
    require(len(rows) == 48 and [r["pair"] for r in rows] == expected_pairs, "exact48 formal pair order/coverage", "BLOCKED_INCOMPLETE_EVIDENCE")
    require(len(prepared) == 6 and len({p["unit"]["unit_id"] for p in prepared}) == 6, "six mode units required")
    finite_tree(prepared); finite_tree(rows)
    support = [dict(seed=p["unit"]["seed"], stage_index=p["unit"]["stage_index"], unit_id=p["unit"]["unit_id"], **m)
               for p in prepared for m in p["mode_support"] if m["class_id"] in (1, 2)]
    require(len(support) == 24 and {(r["seed"], r["stage_index"], r["class_id"], r["mode"]) for r in support} ==
            {(s, t, c, k) for s in range(3) for t in (1, 2) for c in (1, 2) for k in range(2)}, "foreground24 denominator")
    checks = [m["center_active"] and m["active_pixels"] >= 32 and m["case_count"] >= 2 and m["occupancy"] >= .05
              and m["finite"] and m["null_rule_pass"] and m["fit_converged"] for m in support]
    kd = sum(m["KD_active"] for m in support)
    G1 = dict(pass_=bool(all(checks) and kd >= 18), denominator=24, fully_supported_modes=sum(checks), active_KD_modes=kd,
              active_KD_fraction=kd/24, units=support)
    dots, undefined = [], 0
    for row in rows:
        require(set(row["projections"]) == set(row["evaluated"]) == set(CANDIDATES), "five candidate coverage")
        r = row["projections"]["P4"]
        require(r["constraint_count"] <= 13 and len(r["guard_names"]) == len(r["raw_guard_dots"]), "constraint output coverage")
        dots.extend(r["raw_guard_dots"])
        undefined += sum(v is None for v in r["raw_guard_dots"])
        for candidate in CANDIDATES:
            step = row["evaluated"][candidate]["step"]
            require(step["stateless"] and not step["optimizer_constructed"] and not step["checkpoint_written"], "virtual step mutated model")
            require(not step["step_valid"] or abs(step["step_norm"]-.001) <= 1e-15, "virtual norm evidence invalid")
    G2 = dict(pass_=bool(undefined == 0 and dots and all(v is not None and v >= -1e-10 for v in dots)), pairs=48,
              guard_comparisons=len(dots), undefined_required_comparisons=undefined, minimum_guard_dot=min(dots) if dots and not undefined else None)
    previous0 = improvement_summary(comparative(rows, "P0", "previous", "CE"))
    previous1 = improvement_summary(comparative(rows, "P1", "previous", "CE"))
    class_worse = {str(c): median([None if v is None else -v for v in comparative(rows, "P0", "previous", "class_CE", c)]) for c in (1, 2)}
    prev_dice_drop = median(comparative(rows, "P0", "previous", "foreground_Dice"))
    G3 = dict(pass_=bool(previous0["defined"] and previous1["defined"] and at_least(previous0["better_fraction"], .60)
                        and at_least(previous0["median_improvement"], 1e-4) and all(at_most(v, 5e-5) for v in class_worse.values())
                        and at_most(prev_dice_drop, .002) and at_least(previous1["better_fraction"], .55)
                        and at_least(previous1["median_improvement"], 5e-5)),
              P4_vs_P0=previous0, P4_vs_P1=previous1, FG_class_median_CE_worsening=class_worse, previous_Dice_median_drop=prev_dice_drop)
    current_ce = median([None if v is None else -v for v in comparative(rows, "P0", "current", "CE")])
    current_dice = median(comparative(rows, "P0", "current", "foreground_Dice"))
    current_class_drop = maximum([v for c in (1, 2) for v in comparative(rows, "P0", "current", "Dice", c)])
    G4 = dict(pass_=bool(at_most(current_ce, 1e-4) and at_most(current_dice, .002) and at_most(current_class_drop, .003)),
              current_CE_median_worsening=current_ce, current_Dice_median_drop=current_dice, any_pair_FG_class_Dice_drop=current_class_drop)
    worst, mode_worse, by_unit, k2_rows = [], [], {}, []
    for row in rows:
        support_indices = [i for i in range(2, 6) if row["before"]["previous"]["totals"]["mode_pixels"][i] > 0]
        maxima = {candidate: maximum([delta(row, candidate, "previous", "mode_CE", i) for i in support_indices]) for candidate in ("P2", "P4")}
        improved = difference(maxima["P2"], maxima["P4"])
        worst.append(improved)
        by_unit.setdefault(row["unit_id"], []).append(improved)
        new_worse = [difference(delta(row, "P4", "previous", "mode_CE", i), delta(row, "P2", "previous", "mode_CE", i)) for i in support_indices]
        mode_worse.extend(new_worse)
        k2_rows.append(dict(batch_id=row["pair"]["batch_id"], unit_id=row["unit_id"], supported_FG_mode_indices=support_indices,
                           P2_worst_previous_mode_delta=maxima["P2"], P4_worst_previous_mode_delta=maxima["P4"], improvement=improved,
                           max_new_mode_worsening=maximum(new_worse)))
    worst_summary = improvement_summary(worst)
    unit_medians = {key: median(values) for key, values in by_unit.items()}
    require(len(unit_medians) == 6 and all(len(v) == 8 for v in by_unit.values()), "G5 6x8 denominator")
    positives = sum(v is not None and v > 0 for v in unit_medians.values())
    mode_max = maximum(mode_worse)
    G5 = dict(pass_=bool(worst_summary["defined"] and at_least(worst_summary["better_fraction"], .60)
                        and at_least(worst_summary["median_improvement"], 5e-5) and positives >= 4 and at_most(mode_max, 1e-4)),
              **worst_summary, positive_seed_transition_units=positives, seed_transition_medians=unit_medians,
              maximum_new_FG_mode_worsening=mode_max, pair_comparisons=k2_rows)
    ratios = [r["projections"]["P4"]["norm_ratio"] for r in rows]
    med = median(ratios)
    p10 = float(np.quantile(ratios, .10, method="linear")) if all(x is not None for x in ratios) else None
    transitions = {str(stage): median([r["projections"]["P4"]["norm_ratio"] for r in rows if r["pair"]["stage_index"] == stage]) for stage in (1, 2)}
    zero = sum(r["projections"]["P4"]["zero_direction"] for r in rows)
    G6 = dict(pass_=bool(at_least(med, .5) and at_least(p10, .2) and all(at_least(v, .4) for v in transitions.values()) and zero == 0),
              median_norm_ratio=med, p10_norm_ratio=p10, transition_medians=transitions, zero_direction_count=zero,
              undefined_norm_ratios=sum(x is None for x in ratios), all_finite=all(x is not None and np.isfinite(x) for x in ratios))
    isolation = [p["isolation"] for p in prepared] + [r["isolation"] for r in rows]
    iso = all(v["models_bitwise_unchanged"] and v["model_before"] == v["model_after"] and v["banks_unchanged"]
              and v["teacher_bank_gradients"] == v["parameter_grad_fields"] == "None"
              and not v["optimizer_constructed"] and not v["backward_called"] and v["model_optimizer_steps"] == v["transport_optimizer_steps"] == 0
              and v["hidden_gt_training_usage"] == v["test_gt_usage"] == "none" for v in isolation)
    G7 = dict(pass_=bool(iso and all(p["checkpoint_hashes_unchanged"] and not p["validation_GT_in_guards"] for p in prepared)
                        and all(not r["validation_GT_received_by_projection"] for r in rows)),
              model_guard_records=len(isolation), model_checkpoint_bank_unchanged=iso, parameter_grad_fields="None",
              teacher_bank_gradients="None", optimizer_constructed=False, backward_called=False,
              hidden_gt_training_usage="none", test_gt_usage="none", validation_GT="evaluator_only")
    gates = dict(G1=G1, G2=G2, G3=G3, G4=G4, G5=G5, G6=G6, G7=G7)
    failed = [name for name, value in gates.items() if not value["pass_"]]
    return dict(status="FAIL_PMGC_FEASIBILITY" if failed else "PASS_PMGC_FEASIBILITY", G1_G7=gates, failed_gates=failed,
                primary="P4", controls_cannot_rescue=True, formal_pairs=48, seed_transition_units=6,
                model_optimizer_steps=0, transport_optimizer_steps=0, method_registered=False, training_launched=False,
                prototype_derived_new_method_line="ENDED" if failed else "AWAIT_INDEPENDENT_REVIEW",
                additional_variants_authorized=False, next_action="STOP_FOR_INDEPENDENT_REVIEW")


def audit_pair(row, prepared, expected_counts, *, require_full_inventory=True):
    """Independent KKT arithmetic on saved arrays; never call a model or autograd."""
    arrays = b.read_arrays(row["arrays"])
    guard_arrays = b.read_arrays(prepared["guard_arrays"])
    sets = constraint_sets(arrays["fp64_supervised"], guard_arrays["supervised"], guard_arrays["old"], prepared["mode_support"])
    if require_full_inventory:
        inv = row["raw"]["parameter_inventory"]
        require(len(inv) == 51 and sum(x["elements"] for x in inv) == 484016 and arrays["fp64_g0"].shape == (484016,), "full inventory artifact mismatch")
        require(len({x["name"] for x in inv}) == 51 and all(x["trainable"] and x["None_gradient_zero_placeholder"] and x["parameter_grad_is_None"] for x in inv), "parameter inventory silently filtered")
        offset = 0
        for item in inv:
            end = offset + item["elements"]
            if all(item["gradient_is_None"].values()):
                require(all(not arrays[k][offset:end].any() for k in ("native_supervised", "native_g0", "fp64_supervised", "fp64_g0")), "None gradient has nonzero placeholder")
            offset = end
    require(np.array_equal(arrays["P0"], arrays["fp64_g0"]), "P0 is not raw B0")
    require(row["counts"] == {k: expected_counts[k] for k in COUNT_KEYS}, "pair counter mismatch")
    require(Counter(e["kind"] for e in row["call_trace"]) == Counter(row["counts"]), "trace/counter mismatch")
    errors = []
    for candidate in CANDIDATES:
        direction, certificate = arrays[candidate], row["projections"][candidate]
        b.finite(direction)
        require(b.array_hash(direction) == certificate["direction_sha256"] == row["direction_hashes_before_evaluator"][candidate], "direction hash mismatch")
        if candidate != "P0":
            H = np.stack([h for _, h in sets[candidate]])
            norms = np.linalg.norm(H, axis=1)
            N = np.zeros_like(H); nonzero = norms > 0; N[nonzero] = H[nonzero]/norms[nonzero, None]
            dual = np.asarray(certificate["dual"], np.float64)
            require(dual.shape == (len(H),) and (dual >= 0).all(), "dual infeasible artifact")
            raw_dots = H @ direction
            disp = direction-arrays["fp64_g0"]
            objective = .5*float(disp @ disp)
            comp = float(np.max(np.abs(dual*(N @ direction))))
            stationarity = float(np.linalg.norm(disp-N.T @ dual))
            dual_displacement = N.T @ dual
            dual_objective = -.5*float(dual_displacement @ dual_displacement)-float(dual @ (N @ arrays["fp64_g0"]))
            gap = abs(objective-dual_objective)
            require(float(raw_dots.min()) >= -1e-10 and comp <= 1e-10*(1+objective) and gap <= 1e-10*(1+objective)
                    and stationarity <= 1e-10*(1+np.linalg.norm(arrays["fp64_g0"])), "independent KKT certificate failed")
            require(np.allclose(raw_dots, certificate["raw_guard_dots"], atol=1e-12, rtol=1e-12) and abs(objective-certificate["objective"]) <= 1e-10*(1+objective), "saved solver evidence differs")
            errors.append(dict(candidate=candidate, minimum_raw_dot=float(raw_dots.min()), complementarity=comp, stationarity=stationarity, duality_gap=gap))
        for panel in ("previous", "current", "train_labeled"):
            require(aggregate(row["before_batches"][panel]) == row["before"][panel], "before aggregation differs")
            found = row["evaluated"][candidate]["panels"][panel]
            require(aggregate(found["after_batches"]) == found["after"], "after aggregation differs")
            before = row["before"][panel]["totals"]
            after = found["after"]["totals"]
            require(all(before[key] == after[key] for key in ("pixels", "class_pixels", "mode_pixels", "old_correct_pixels")), "candidate-dependent support")
    return dict(batch_id=row["pair"]["batch_id"], status="PASS_ARTIFACT_PAIR_AUDIT", KKT=errors,
                model_forwards=0, autograd_calls=0, evaluator_arithmetic_recomputed=True, raw_predictions_replayed=False)


def tables(output, prepared, rows, verdict):
    output = Path(output)
    modes = [dict(unit_id=p["unit"]["unit_id"], seed=p["unit"]["seed"], stage_index=p["unit"]["stage_index"], **m)
             for p in prepared for m in p["mode_support"]]
    csv_new(output/"pmgc_mode_support.csv", modes)
    constraints, solver, retention, previous, current, mode_utility = [], [], [], [], [], []
    for row in rows:
        context = {k: row["pair"][k] for k in ("batch_id", "seed", "stage_index", "domain", "pair_index")}
        for candidate in CANDIDATES:
            projection = row["projections"][candidate]
            ctx = dict(context, candidate=candidate, primary=candidate == "P4")
            solver.append(dict(ctx, **projection))
            retention.append({**ctx, **{k: projection[k] for k in ("norm", "raw_norm", "norm_ratio", "zero_direction", "direction_sha256")}, **row["evaluated"][candidate]["step"]})
            for i, (name, dot) in enumerate(zip(projection["guard_names"], projection["raw_guard_dots"])):
                constraints.append(dict(ctx, guard=name, guard_norm=projection["guard_norms"][i], dot=dot, dual=projection["dual"][i], halfspace_satisfied=dot >= -1e-10))
            for panel, table in (("previous", previous), ("current", current)):
                table.append(dict(ctx, CE_before=row["before"][panel]["CE"], CE_after=row["evaluated"][candidate]["panels"][panel]["after"]["CE"],
                                  CE_delta=delta(row, candidate, panel, "CE"), foreground_Dice_delta=delta(row, candidate, panel, "foreground_Dice"),
                                  FG_class_CE_deltas=[delta(row, candidate, panel, "class_CE", c) for c in (1, 2)],
                                  FG_class_Dice_deltas=[delta(row, candidate, panel, "Dice", c) for c in (1, 2)]))
            for panel in ("previous", "current", "train_labeled"):
                for i in range(6):
                    mode_utility.append(dict(ctx, panel=panel, class_id=i//2, mode=i%2,
                        pixels=row["before"][panel]["totals"]["mode_pixels"][i], old_correct_pixels=row["before"][panel]["totals"]["old_correct_pixels"][i],
                        CE_delta=delta(row, candidate, panel, "mode_CE", i), Dice_delta=delta(row, candidate, panel, "mode_Dice", i), old_KL_delta=delta(row, candidate, panel, "old_KL", i)))
    for name, entries in (("pmgc_constraint_gradients", constraints), ("pmgc_projection_solver", solver), ("pmgc_signal_retention", retention),
                          ("pmgc_previous_utility", previous), ("pmgc_current_safety", current), ("pmgc_mode_utility", mode_utility),
                          ("pmgc_k2_vs_k1", verdict["G1_G7"]["G5"]["pair_comparisons"])):
        csv_new(output/(name+".csv"), entries)
    return {name: len(value) for name, value in dict(mode_support=modes, constraints=constraints, projection_solver=solver,
            signal_retention=retention, previous_utility=previous, current_safety=current, mode_utility=mode_utility).items()}
