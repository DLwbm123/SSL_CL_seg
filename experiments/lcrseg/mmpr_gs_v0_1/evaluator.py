"""Independent validation GT consumer; cannot construct or change MMPR masks."""
import numpy as np

from .core import require


def boundary_mask(labels):
    y = np.asarray(labels)
    require(y.ndim == 2 and np.isin(y, [0, 1, 2, 255]).all(), "evaluator GT geometry/mapping")
    valid = y != 255
    boundary = np.zeros(y.shape, bool)
    vertical = valid[1:] & valid[:-1] & (y[1:] != y[:-1])
    horizontal = valid[:, 1:] & valid[:, :-1] & (y[:, 1:] != y[:, :-1])
    boundary[1:] |= vertical
    boundary[:-1] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    return boundary


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def relative(reference, candidate):
    if reference is None or candidate is None:
        return None
    return (reference-candidate)/reference if reference > 0 else (0.0 if candidate == 0 else None)


def evaluate_case(scores, masks, labels, *, seed, stage, case):
    """Masks have already been sealed. Ignore255 only affects this evaluator."""
    y = np.asarray(labels).reshape(-1)
    pred = scores["teacher_probability"].argmax(1)
    require(y.shape == pred.shape and np.isin(y, [0, 1, 2, 255]).all(), "evaluator labels")
    valid, correct = y != 255, y == pred
    boundary = boundary_mask(labels).reshape(-1)
    require(set(masks) == {"Q0", "Q1", "Q2", "Q3"}, "evaluator candidates")
    active = scores["active_mask"]
    rows, changes, regions = [], [], []
    for c in range(3):
        full_stratum = pred == c
        stratum = full_stratum & valid
        n = int(stratum.sum())
        reference = masks["Q0"].astype(bool)
        for candidate, weight in masks.items():
            weight = np.asarray(weight, np.float64)
            require(weight.shape == y.shape and np.isfinite(weight).all() and (weight >= 0).all(), "evaluator weights")
            amount = float(weight[stratum].sum())
            error = float(weight[stratum & ~correct].sum())
            context = dict(seed=seed, stage_index=stage, case_id=case, class_id=c, candidate=candidate)
            rows.append(dict(context, level="case", valid_pixels=n, full_pixels=int(full_stratum.sum()),
                             full_mass=float(weight[full_stratum].sum()), R1_full_mass=int(reference[full_stratum].sum()),
                             full_mass_difference=float(weight[full_stratum].sum()-reference[full_stratum].sum()),
                             selected_nonignore_mass=amount, error_mass=error,
                             case_fraction_selected=amount/n if n else 0.0,
                             case_fraction_error=error/n if n else 0.0,
                             precision=ratio(amount-error, amount), weighted_error=ratio(error, amount),
                             ignored_pixels=int((full_stratum & ~valid).sum()), selected_ignore_mass=float(weight[full_stratum & ~valid].sum()),
                             null_pixels=int((full_stratum & ~active).sum()), selected_null_mass=float(weight[full_stratum & ~active].sum())))
            for region, membership in (("boundary", boundary), ("interior", valid & ~boundary)):
                selected = stratum & membership
                mass = float(weight[selected].sum())
                good = float(weight[selected & correct].sum())
                regions.append(dict(context, region=region, valid_pixels=int(selected.sum()), selected_mass=mass,
                                    correct_mass=good, precision=ratio(good, mass), weighted_error=ratio(mass-good, mass)))
            if candidate in ("Q1", "Q2"):
                chosen = weight.astype(bool)
                added = stratum & chosen & ~reference
                removed = stratum & reference & ~chosen
                changes.append(dict(context, newly_selected=int((full_stratum & chosen & ~reference).sum()),
                                    deselected=int((full_stratum & reference & ~chosen).sum()),
                                    new_nonignore_count=int(added.sum()), removed_nonignore_count=int(removed.sum()),
                                    new_correct=int((added & correct).sum()), removed_correct=int((removed & correct).sum()),
                                    newly_selected_precision=ratio(int((added & correct).sum()), int(added.sum())),
                                    removed_precision=ratio(int((removed & correct).sum()), int(removed.sum()))))
    return rows, changes, regions


def aggregate(case_rows):
    units = []
    for seed in range(3):
        for stage in range(3):
            for c in range(3):
                for candidate in ("Q0", "Q1", "Q2", "Q3"):
                    rows = [r for r in case_rows if (r["seed"], r["stage_index"], r["class_id"], r["candidate"]) ==
                            (seed, stage, c, candidate)]
                    require(bool(rows), "missing validation unit", "BLOCKED_INCOMPLETE_EVIDENCE")
                    require(len({r["case_id"] for r in rows}) == len(rows), "duplicate validation case")
                    mass = sum(r["case_fraction_selected"] for r in rows)
                    error = sum(r["case_fraction_error"] for r in rows)
                    case_errors = [r["weighted_error"] for r in rows if r["weighted_error"] is not None]
                    units.append(dict(seed=seed, stage_index=stage, class_id=c, candidate=candidate, level="unit",
                                      cases=len(rows), valid_stratum_cases=sum(r["valid_pixels"] > 0 for r in rows),
                                      weighted_selected_mass=mass, weighted_error_mass=error,
                                      weighted_error=ratio(error, mass), precision=ratio(mass-error, mass),
                                      case_error_defined=len(case_errors), case_error_undefined=len(rows)-len(case_errors),
                                      case_error_quantiles=np.quantile(case_errors, [.1, .5, .9], method="linear").tolist() if case_errors else None,
                                      full_mass=sum(r["full_mass"] for r in rows),
                                      selected_null_mass=sum(r["selected_null_mass"] for r in rows)))
    references = {(u["seed"], u["stage_index"], u["class_id"]): u for u in units if u["candidate"] == "Q0"}
    for u in units:
        ref = references[u["seed"], u["stage_index"], u["class_id"]]
        u["reference_error"] = ref["weighted_error"]
        u["relative_error_reduction"] = relative(ref["weighted_error"], u["weighted_error"])
        u["precision_drop"] = ref["precision"]-u["precision"] if ref["precision"] is not None and u["precision"] is not None else None
    return units
