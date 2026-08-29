"""LCR-Seg V0.4a: preregistered Soft Reliability Allocation (SRA)."""
from __future__ import annotations

from typing import Any, Mapping

import torch

from .base import merged_method_config
from .components.compatibility import CompatibilityOutput
from .components.learnability import LearnabilityOutput
from .components.progressive_admission import ProgressiveAdmissionOutput, strict_relation_valid_mask
from .components.pseudo_label import PseudoLabelOutput
from .components.relation_field import RelationOutput
from .components.soft_reliability_allocation import (
    SoftAssimilationLossOutput,
    SoftReliabilityAllocationOutput,
    anchor_update_weights,
    soft_reliability_allocation,
    soft_reliability_assimilation_loss,
)
from .lcrseg_v0_2a import LCRSegV02AMethod, resolve_v02a_method_config


FROZEN_RANK_SCHEDULE = {
    "start_hard_fraction": 0.40,
    "end_hard_fraction": 0.80,
    "type": "linear",
    "scope": "per_site",
    "classwise": True,
    "minimum_pixels_for_class_cdf": 32,
}

FROZEN_SOFT_ALLOCATION = {
    "tau": 0.10,
    "current_relation_temperature": 1.0,
}


def resolve_v04a_method_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve and validate the exact registered V0.4a SRA semantics."""

    provided = dict(config or {})
    if provided.get("protocol_id", "lcrseg_v0_4a") != "lcrseg_v0_4a":
        raise ValueError("V0.4a requires protocol_id=lcrseg_v0_4a")
    expected = {
        "assimilation_mode": "soft_reliability_allocation",
        "consolidation_mode": "uniform_relation",
        "single_anchor": True,
        "multi_agent": False,
        "ric": False,
        "teacher_rejection": False,
    }
    for key, value in expected.items():
        if key in provided and provided[key] != value:
            raise ValueError(f"V0.4a frozen field differs: {key}")
    if "rank_schedule" in provided and dict(provided["rank_schedule"]) != FROZEN_RANK_SCHEDULE:
        raise ValueError("V0.4a rank schedule differs from the frozen protocol")
    if "soft_allocation" in provided and dict(provided["soft_allocation"]) != FROZEN_SOFT_ALLOCATION:
        raise ValueError("V0.4a soft allocation differs from the frozen protocol")
    if float(provided.get("lambda_relation", 1.0)) != 1.0:
        raise ValueError("V0.4a historical relation weight must remain 1.0")

    mapped = dict(provided)
    for key in (
        "rank_schedule",
        "soft_allocation",
        "single_anchor",
        "multi_agent",
        "ric",
        "teacher_rejection",
    ):
        mapped.pop(key, None)
    mapped.update(
        {
            "protocol_id": "lcrseg_v0_2a",
            "variant_id": "R1",
            "assimilation_mode": "progressive_admission",
            "consolidation_mode": "uniform_relation",
            "lambda_relation": 1.0,
        }
    )
    base = resolve_v02a_method_config(mapped)
    base.update(
        {
            "protocol_id": "lcrseg_v0_4a",
            "variant_id": "SRA",
            **expected,
            "lambda_relation": 1.0,
            "rank_schedule": dict(FROZEN_RANK_SCHEDULE),
            "soft_allocation": dict(FROZEN_SOFT_ALLOCATION),
            "historical_relation_path": "frozen_uniform_relation_v0_2a",
        }
    )
    return merged_method_config(base)


class LCRSegV04AMethod(LCRSegV02AMethod):
    method_name = "lcrseg_v0_4a"
    method_version = "0.4a"

    def __init__(self, model, *, config: Mapping[str, Any] | None = None) -> None:
        resolved = resolve_v04a_method_config(config)
        mapped = dict(resolved)
        mapped.update(
            {
                "protocol_id": "lcrseg_v0_2a",
                "variant_id": "R1",
                "assimilation_mode": "progressive_admission",
                "consolidation_mode": "uniform_relation",
                "lambda_relation": 1.0,
            }
        )
        super().__init__(model, config=mapped)
        self.config = resolved
        self.requires_labeled_calibration = False
        self._sra_allocation: SoftReliabilityAllocationOutput | None = None
        self._sra_loss: SoftAssimilationLossOutput | None = None
        self._sra_anchor_weights: torch.Tensor | None = None
        self.v04a_statistics = self._new_v04a_statistics()

    @staticmethod
    def _new_v04a_statistics() -> dict[str, Any]:
        return {
            "steps": 0,
            "valid_candidate_count": 0,
            "alpha_mass": 0.0,
            "soft_mass": 0.0,
            "anchor_update_mass": 0.0,
            "hidden_gt_training_usage": 0,
            "old_model_gradient_detected": 0,
            "historical_anchor_mutation": 0,
            "historical_relation_path": "frozen_uniform_relation_v0_2a",
        }

    def protocol_semantics(self) -> dict[str, Any]:
        return {
            key: self.config[key]
            for key in (
                "protocol_id",
                "variant_id",
                "assimilation_mode",
                "consolidation_mode",
                "single_anchor",
                "multi_agent",
                "ric",
                "teacher_rejection",
                "lambda_assim",
                "lambda_relation",
                "learnability_formula_version",
                "rank_schedule",
                "soft_allocation",
                "historical_relation_path",
            )
        }

    def begin_site(self, site_id: str, previous_checkpoint, total_steps: int) -> None:  # type: ignore[override]
        super().begin_site(site_id, previous_checkpoint, total_steps)
        self._sra_allocation = None
        self._sra_loss = None
        self._sra_anchor_weights = None
        self.v04a_statistics = self._new_v04a_statistics()

    def _compute_admission(
        self,
        pseudo: PseudoLabelOutput,
        learnability: LearnabilityOutput,
        strong_valid_mask: torch.Tensor,
        *,
        site_step: int,
    ) -> ProgressiveAdmissionOutput:
        relation_valid = strict_relation_valid_mask(strong_valid_mask, learnability.score.shape[-2:])
        schedule = self.config["rank_schedule"]
        soft = self.config["soft_allocation"]
        allocation = soft_reliability_allocation(
            pseudo,
            learnability.score,
            relation_valid,
            num_classes=self.num_classes,
            site_step=site_step,
            total_site_steps=self.total_steps,
            start_hard_fraction=float(schedule["start_hard_fraction"]),
            end_hard_fraction=float(schedule["end_hard_fraction"]),
            tau=float(soft["tau"]),
            minimum_pixels_for_class_cdf=int(schedule["minimum_pixels_for_class_cdf"]),
        )
        self._sra_allocation = allocation
        hard_dominant = (allocation.alpha.ge(0.5) & allocation.candidate_mask).detach()
        selected_counts = tuple(
            int((hard_dominant[:, 0] & pseudo.labels.eq(class_id)).sum()) for class_id in range(self.num_classes)
        )
        return ProgressiveAdmissionOutput(
            mask=hard_dominant,
            candidate_mask=allocation.candidate_mask,
            site_progress=allocation.site_progress,
            target_fraction=allocation.target_hard_fraction,
            candidate_counts=allocation.candidate_counts,
            selected_counts=selected_counts,
            learnability_thresholds=tuple(float("nan") for _ in range(self.num_classes)),
        )

    def _assimilation_objective(
        self,
        *,
        strong_logits: torch.Tensor,
        pseudo: PseudoLabelOutput,
        learnability: LearnabilityOutput,
        admission: ProgressiveAdmissionOutput,
        weak_relation: RelationOutput,
        strong_relation: RelationOutput,
        strong_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        del learnability, admission
        if self._sra_allocation is None:
            raise AssertionError("SRA allocation was not computed before the objective")
        result = soft_reliability_assimilation_loss(
            strong_logits,
            pseudo,
            weak_relation,
            strong_relation,
            self._sra_allocation,
            strong_valid_mask,
            current_relation_temperature=float(self.config["soft_allocation"]["current_relation_temperature"]),
        )
        self._sra_loss = result
        return result.loss

    def _cache_protocol_unlabeled_anchor_update(
        self,
        *,
        features: torch.Tensor,
        pseudo: PseudoLabelOutput,
        learnability: LearnabilityOutput,
        weak_relation: RelationOutput,
        uniform_compatibility: CompatibilityOutput,
        site_step: int,
    ) -> None:
        del learnability, uniform_compatibility
        if self._sra_allocation is None:
            raise AssertionError("SRA allocation was not computed before the anchor update")
        weights = anchor_update_weights(pseudo, weak_relation, self._sra_allocation)
        self._sra_anchor_weights = weights
        self._pending_anchor_updates.append(
            (features.detach(), pseudo.labels.detach(), weights.detach(), "unlabeled", int(site_step))
        )

    def training_step(self, labeled_batch, unlabeled_batch, global_step: int, site_step: int):  # type: ignore[override]
        self._sra_allocation = None
        self._sra_loss = None
        self._sra_anchor_weights = None
        result = super().training_step(labeled_batch, unlabeled_batch, global_step, site_step)
        allocation = self._sra_allocation
        loss = self._sra_loss
        anchor_weights = self._sra_anchor_weights
        if allocation is None or loss is None or anchor_weights is None:
            alpha_values = torch.empty(0, device=result.total_loss.device)
            percentiles = torch.empty(0, device=result.total_loss.device)
            deciles = [0.0] * 11
            anchor_mass = 0.0
            hard_mean = soft_mean = weighted_hard = weighted_soft = 0.0
            valid_count = 0
            quantile_boundary = 0.0
            target_hard_fraction = 0.0
        else:
            valid = allocation.candidate_mask
            alpha_values = allocation.alpha[valid]
            percentiles = allocation.percentile[valid]
            q = torch.linspace(0.0, 1.0, 11, device=allocation.alpha.device)
            deciles = [float(value) for value in torch.quantile(alpha_values.float(), q)] if alpha_values.numel() else [0.0] * 11
            anchor_mass = float(anchor_weights.sum())
            hard_mean = float(loss.hard_mean.detach())
            soft_mean = float(loss.soft_mean.detach())
            weighted_hard = float(loss.weighted_hard_mean.detach())
            weighted_soft = float(loss.weighted_soft_mean.detach())
            valid_count = int(loss.valid_count)
            quantile_boundary = float(allocation.quantile_boundary)
            target_hard_fraction = float(allocation.target_hard_fraction)
            self.v04a_statistics["steps"] += 1
            self.v04a_statistics["valid_candidate_count"] += valid_count
            self.v04a_statistics["alpha_mass"] += float(alpha_values.sum())
            self.v04a_statistics["soft_mass"] += float((1.0 - alpha_values).sum())
            self.v04a_statistics["anchor_update_mass"] += anchor_mass
        result.scalars.update(
            {
                "sra_quantile_boundary": quantile_boundary,
                "sra_target_hard_fraction": target_hard_fraction,
                "sra_alpha_mean": float(alpha_values.mean()) if alpha_values.numel() else 0.0,
                "sra_alpha_deciles": deciles,
                "sra_percentile_mean": float(percentiles.mean()) if percentiles.numel() else 0.0,
                "sra_hard_loss_mean": hard_mean,
                "sra_soft_loss_mean": soft_mean,
                "sra_weighted_hard_loss": weighted_hard,
                "sra_weighted_soft_loss": weighted_soft,
                "sra_hard_soft_loss_ratio": weighted_hard / max(weighted_soft, 1.0e-12),
                "sra_valid_candidate_count": valid_count,
                "sra_anchor_update_mass": anchor_mass,
                "sra_anchor_update_count": int(anchor_weights.gt(0).sum()) if anchor_weights is not None else 0,
                "sra_gradient_finite": float(bool(torch.isfinite(result.total_loss).all())),
                "sra_historical_relation_exact_path": "frozen_uniform_relation_v0_2a",
                "sra_current_relation_target_stopgrad": 1,
            }
        )
        if result.maps is None:
            result.maps = {}
        if allocation is not None:
            result.maps.update(
                {
                    "sra_alpha": allocation.alpha.detach(),
                    "sra_percentile": allocation.percentile.detach(),
                    "sra_candidate_mask": allocation.candidate_mask.detach(),
                }
            )
        return result

    def end_site(self, site_id: str) -> dict[str, Any]:
        result = super().end_site(site_id)
        result["v04a_statistics"] = dict(self.v04a_statistics)
        return result

    def method_state_dict(self) -> dict[str, Any]:
        state = super().method_state_dict()
        state["method_statistics"] = dict(state["method_statistics"])
        state["method_statistics"]["v04a_statistics"] = dict(self.v04a_statistics)
        return state

    def load_method_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_method_state_dict(state)
        statistics = dict(state.get("method_statistics") or {})
        self.v04a_statistics = dict(statistics.get("v04a_statistics") or self._new_v04a_statistics())
