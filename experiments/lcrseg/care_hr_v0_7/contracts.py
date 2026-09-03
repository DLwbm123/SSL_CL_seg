"""Pure review contract definitions. No execution authority is granted here."""
from __future__ import annotations

from dataclasses import dataclass


REVIEW_STATUS = "BLOCKED_AWAITING_EXTERNAL_CODE_REVIEW"
FEATURE_NAMES = (
    "class_is_cup",
    "direction_is_add",
    "log_region_area_over_image",
    "log_region_area_over_current_foreground",
    "compactness",
    "normalized_centroid_radius",
    "mean_current_target_probability",
    "mean_historical_target_probability",
    "mean_target_probability_delta",
    "p10_target_probability_delta",
    "mean_current_entropy",
    "mean_historical_entropy",
    "mean_JS_divergence",
    "mean_current_winning_margin",
    "mean_historical_winning_margin",
    "whole_case_hard_disagreement_fraction",
    "PPC_calibrated_probability",
    "PPC_consensus",
    "raw_SHOR_log_alpha_contrast",
    "ridge_top1_top2_margin",
)
RIDGE_LAMBDAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
BOOTSTRAP_REPLICATES = 200
BOOTSTRAP_SEED = 2026090701


class ReviewBlocked(RuntimeError):
    """Raised for every real training, data, or evaluation entry point."""


def require_external_review_authorization(*_args, **_kwargs):
    raise ReviewBlocked(REVIEW_STATUS)


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    target_class: int
    direction: str
    area: int
    centroid_row: float
    centroid_col: float
    mask: object


def candidate_grid():
    return tuple(
        {
            "candidate_id": f"l{blend:.2f}_e{epsilon:.4f}_d{delta:.4f}",
            "blend_lambda": blend,
            "epsilon_gain": epsilon,
            "delta_harm": delta,
            "rho": 0.80,
        }
        for blend in (0.50, 0.75)
        for epsilon in (0.0, 0.0025)
        for delta in (0.005, 0.010)
    )


def review_contract():
    return {
        "method": "CARe-HR V0.7",
        "status": REVIEW_STATUS,
        "draft_state": "DRAFT_NOT_REGISTERED",
        "training_authority": "NO_TRAINING_AUTHORITY",
        "evaluation_authority": "NO_EVALUATION_AUTHORITY",
        "feature_names": list(FEATURE_NAMES),
        "candidate_grid": list(candidate_grid()),
        "primary_whole_case_replacement": False,
    }
