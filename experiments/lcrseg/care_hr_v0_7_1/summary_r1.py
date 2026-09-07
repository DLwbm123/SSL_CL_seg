"""Frozen case-first grouping, sensitivity and screening; evaluator-only."""
import numpy as np
from .semantic_audit import frozen_functions

FIELDS = ("foreground_dice", "rim_dice", "cup_dice", "mean_iou")
DOMAINS = ("REFUGE", "RIM_ONE_r3", "Drishti_GS")


def legacy_aggregate(rows):
    ns, _ = frozen_functions("ppc_shor_v0_6a.py", ("aggregate_case_metrics",))
    ns["POLICIES"] = tuple(dict.fromkeys(r["policy"] for r in rows))
    return ns["aggregate_case_metrics"](rows)


def groups_for(rows):
    yield "overall", "all", rows
    yield "historical", "history", [r for r in rows if r["domain_index"] < 2]
    for field, level in (("seed", "seed"), ("domain", "domain")):
        for value in sorted({r[field] for r in rows}):
            yield level, str(value), [r for r in rows if r[field] == value]
    for seed, domain in sorted({(r["seed"], r["domain"]) for r in rows}):
        yield "seed_domain", f"{seed}:{domain}", [r for r in rows if r["seed"] == seed and r["domain"] == domain]


def balanced(rows, field, evaluable_only=False, patient_equal=False):
    groups = {}
    for row in rows:
        key = (row["seed"], row["domain_index"])
        group = groups.setdefault(key, [])
        if not evaluable_only or row["has_evaluable_gt"]:
            group.append(row)
    if not groups or any(not g for g in groups.values()):
        return None
    values = []
    for group in groups.values():
        if patient_equal:
            patients = {}
            for row in group: patients.setdefault(row["patient_id"], []).append(row[field])
            values.append(np.mean([np.mean(v) for v in patients.values()]))
        else:
            values.append(np.mean([r[field] for r in group]))
    return float(np.mean(values))


def summarize(rows, evaluable_only=False, patient_equal=False):
    output = []
    policies = tuple(dict.fromkeys(r["policy"] for r in rows))
    for level, key, subset in groups_for(rows):
        for policy in policies:
            selected = [r for r in subset if r["policy"] == policy]
            kept = [r for r in selected if not evaluable_only or r["has_evaluable_gt"]]
            summary = dict(level=level, key=key, policy=policy, rows=len(selected), retained_rows=len(kept),
                           excluded_rows=len(selected)-len(kept), patients=len({r["patient_id"] for r in selected}),
                           retained_patients=len({r["patient_id"] for r in kept}),
                           groups=len({(r["seed"], r["domain_index"]) for r in selected}),
                           weighting="original_seed_domain_equal_then_" + ("patient_equal" if patient_equal else "case_equal"))
            for field in (*FIELDS, "gain_macro_fg", "gain_rim", "gain_cup", "harm"):
                summary[field] = balanced(selected, field, evaluable_only, patient_equal)
            summary["status"] = "UNDEFINED" if summary["foreground_dice"] is None else "DEFINED"
            output.append(summary)
    # Differences are between independently grouped policy gains, not a reweighted cohort.
    ref = {(r["level"], r["key"]): r for r in output if r["policy"] == "frozen_PPC_C6"}
    for row in output:
        base = ref.get((row["level"], row["key"]))
        row["gain_minus_frozen_PPC"] = None if base is None or row["gain_macro_fg"] is None or base["gain_macro_fg"] is None else row["gain_macro_fg"]-base["gain_macro_fg"]
        row["rim_drop"] = None if row["gain_rim"] is None else max(0.0, -row["gain_rim"])
        row["cup_drop"] = None if row["gain_cup"] is None else max(0.0, -row["gain_cup"])
    return output


def terminal(rows):
    support = [r for r in rows if r["policy"] == "current"]
    if balanced(support, "foreground_dice", evaluable_only=True) is None:
        return "BLOCKED_NO_EVALUABLE_SUPPORT"
    table = {(r["policy"], r["level"], r["key"]): r["gain_macro_fg"] for r in summarize(rows)}
    cap = "O_CAP_envelope"; safe = "O_SAFE0_envelope"
    if table[cap, "overall", "all"] < .17 or table[cap, "historical", "history"] < .27:
        return "FAIL_FROZEN_ACTION_SPACE_CAPACITY"
    if (table[safe, "overall", "all"] >= .17 and table[safe, "historical", "history"] >= .27
            and all(table[safe, "domain", d] > 0 for d in DOMAINS[:2])
            and all(table[safe, "seed", str(s)] > 0 for s in range(3))):
        return "PASS_ACTION_SPACE_SCREEN_ONLY"
    return "CAPACITY_PRESENT_SAFETY_UNRESOLVED"
