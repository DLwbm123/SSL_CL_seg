"""Pure controls and the budgeted CARe-HR primary policy."""
from __future__ import annotations

import numpy as np


FAILURE = "FAIL_CARE_HR_CALIBRATION"


def _probability(value):
    value = np.asarray(value)
    if value.ndim != 3 or not np.issubdtype(value.dtype, np.floating):
        raise ValueError("probability arrays must be CxHxW floats")
    if not np.all(np.isfinite(value)) or np.any(value < 0) or np.any(value > 1) or not np.allclose(value.sum(axis=0), 1, atol=1e-7):
        raise ValueError("invalid probability array")
    return value


def _apply_anchored_revision(current_probability, historical_probability, accepted_regions,
                             blend_lambda, allow_oracle=False):
    current = _probability(current_probability)
    historical = _probability(historical_probability)
    if current.shape != historical.shape:
        raise ValueError("probability shapes differ")
    allowed = (0.5, 0.75, 1.0) if allow_oracle else (0.5, 0.75)
    if blend_lambda not in allowed:
        raise ValueError("blend lambda is outside the review contract")
    if not accepted_regions:
        return current.copy()
    region = np.zeros(current.shape[1:], dtype=bool)
    for item in accepted_regions:
        mask = np.asarray(getattr(item, "mask", item), dtype=bool)
        if mask.shape != region.shape:
            raise ValueError("region shape differs")
        region |= mask
    output = current.copy()
    output[:, region] = (1.0 - blend_lambda) * current[:, region] + blend_lambda * historical[:, region]
    _probability(output)
    if not np.array_equal(output[:, ~region], current[:, ~region]):
        raise AssertionError("outside-region probability changed")
    return output


def apply_anchored_revision(current_probability, historical_probability, accepted_regions,
                            blend_lambda):
    return _apply_anchored_revision(current_probability, historical_probability, accepted_regions,
                                    blend_lambda)


def c0_current(current_probability):
    return np.asarray(current_probability).copy()


def c3_shor_whole_case(current_probability, historical_probability, route_historical):
    return np.asarray(historical_probability if route_historical else current_probability).copy()


def c4_ppc_whole_case(current_probability, historical_probability, route_historical):
    return c3_shor_whole_case(current_probability, historical_probability, route_historical)


def c5_disagreement_veto(current_probability, proposed_probability, disagreement, threshold):
    return np.asarray(current_probability if disagreement > threshold else proposed_probability).copy()


def c6_classwise_anchored(current_probability, historical_probability, regions, blend_lambda=0.5):
    return apply_anchored_revision(current_probability, historical_probability, regions, blend_lambda)


def c7_confidence_regions(current_probability, historical_probability, regions, confidence, threshold,
                          blend_lambda=0.5):
    accepted = [region for region, score in zip(regions, confidence) if np.isfinite(score) and score >= threshold]
    return apply_anchored_revision(current_probability, historical_probability, accepted, blend_lambda)


def accept_regions(proposals, gain_lcb, harm_ucb, accepted_votes, finite_predictions,
                   out_of_distribution, epsilon_gain, delta_harm, rho, current_foreground_pixels,
                   image_pixels):
    accepted = []
    changed = 0
    class_counts = {1: 0, 2: 0}
    for proposal, gain, harm, votes, finite, ood in zip(
            proposals, gain_lcb, harm_ucb, accepted_votes, finite_predictions, out_of_distribution):
        gain = float(np.median(np.asarray(gain, dtype=np.float64)))
        harm = float(np.median(np.asarray(harm, dtype=np.float64)))
        consensus = 0.0 if finite <= 0 else votes / finite
        eligible = (np.isfinite(gain) and np.isfinite(harm) and gain > epsilon_gain
                    and harm <= delta_harm and consensus >= rho and not ood)
        new_changed = changed + proposal.area
        if (not eligible or len(accepted) >= 4 or class_counts[proposal.target_class] >= 3
                or new_changed > 0.15 * current_foreground_pixels
                or new_changed > 0.02 * image_pixels):
            continue
        accepted.append(proposal)
        changed = new_changed
        class_counts[proposal.target_class] += 1
    return tuple(accepted)


def c8_care_hr(current_probability, historical_probability, proposals, gain_lcb, harm_ucb,
               accepted_votes, finite_predictions, out_of_distribution, candidate,
               current_foreground_pixels, image_pixels):
    if any(getattr(item, "whole_case", False) for item in proposals):
        raise ValueError("primary policy forbids whole-case replacement")
    accepted = accept_regions(
        proposals, gain_lcb, harm_ucb, accepted_votes, finite_predictions, out_of_distribution,
        candidate["epsilon_gain"], candidate["delta_harm"], candidate["rho"],
        current_foreground_pixels, image_pixels,
    )
    return apply_anchored_revision(current_probability, historical_probability, accepted,
                                   candidate["blend_lambda"]), accepted


def candidate_sort_key(row):
    return (
        -int(bool(row["all_inner_safety_gates"])),
        int(bool(row["catastrophic_current_event"])),
        -row["shared_gain_p10"],
        -row["historical_gain_p10"],
        row["current_drop_p90"],
        row["maximum_seed_domain_drop_p90"],
        row["blend_lambda"],
        row["delta_harm"],
        -row["epsilon_gain"],
        row["candidate_id"],
    )


def select_candidate(rows):
    eligible = [row for row in rows if row["all_inner_safety_gates"] and not row["catastrophic_current_event"]]
    if not eligible:
        raise RuntimeError(FAILURE)
    return min(eligible, key=candidate_sort_key)
