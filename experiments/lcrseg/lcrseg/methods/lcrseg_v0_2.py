"""LCR-Seg V0.2: asymmetric reliability routing.

V0.2 is intentionally a separate method class.  It preserves the frozen V0.1
raw L/C definitions, model, relation field, anchor lifecycle, and shared
training engine while changing only how L admits current pseudo-label pixels
and how labeled-calibrated C can conservatively reject relation supervision.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from ..common import write_csv, write_json
from ..contracts import LabeledBatch, MethodStepOutput, UnlabeledBatch, differentiable_zero
from ..data.transforms import downsample_valid_mask
from .components.compatibility import CompatibilityOutput, compute_compatibility, zero_compatibility
from .components.compatibility_calibrator import LabeledOnlyCompatibilityCalibrator
from .components.learnability import LearnabilityOutput, compute_learnability
from .components.progressive_admission import (
    ProgressiveAdmissionOutput,
    admission_assimilation_loss,
    classwise_progressive_admission,
    strict_relation_valid_mask,
)
from .components.pseudo_label import PseudoLabelOutput, build_pseudo_labels
from .components.rejection_only_routing import RejectionOnlyOutput, rejection_only_relation_loss, rejection_only_weights
from .components.relation_field import RelationOutput
from .lcrseg_v0_1 import LCR_DEFAULTS, LCRSegV01Method


LCR_V02_DEFAULTS: dict[str, Any] = {
    **LCR_DEFAULTS,
    "progressive_admission": True,
    "compatibility_calibration": True,
    "compatibility_rejection": True,
    "pi_start": 0.4,
    "pi_end": 0.8,
    "calibration_bins": 10,
    "calibration_min_pixels": 500,
    "calibration_update_epochs": 10,
    "compatibility_probability_threshold": 0.7,
    "max_reject_fraction_per_class": 0.2,
    "rejected_weight_floor": 0.5,
}


def _uniform_admission(pseudo: PseudoLabelOutput, relation_valid_mask: torch.Tensor) -> ProgressiveAdmissionOutput:
    mask = (pseudo.valid.detach().bool() & relation_valid_mask.detach().bool()).detach()
    counts: list[int] = []
    for class_id in range(int(pseudo.labels.max().detach().item()) + 1 if bool(pseudo.valid.any()) else 0):
        counts.append(int((mask[:, 0] & pseudo.labels.eq(class_id)).sum()))
    return ProgressiveAdmissionOutput(
        mask=mask,
        candidate_mask=mask,
        site_progress=0.0,
        target_fraction=1.0,
        candidate_counts=tuple(counts),
        selected_counts=tuple(counts),
        learnability_thresholds=tuple(float("nan") for _ in counts),
    )


def _uniform_rejection(
    raw_compatibility: CompatibilityOutput,
    relation_valid_mask: torch.Tensor,
    *,
    num_classes: int,
    old_predicted_class: torch.Tensor | None,
) -> RejectionOnlyOutput:
    zeros = torch.zeros_like(relation_valid_mask, dtype=torch.bool)
    counts = tuple(
        int((relation_valid_mask[:, 0] & old_predicted_class.eq(class_id)).sum()) if old_predicted_class is not None else 0
        for class_id in range(num_classes)
    )
    return RejectionOnlyOutput(
        calibrated_compatibility=raw_compatibility.score.detach(),
        rejection_mask=zeros,
        weights=torch.ones_like(raw_compatibility.score).detach(),
        relation_valid_mask=relation_valid_mask.detach().bool(),
        candidate_counts=counts,
        rejected_counts=tuple(0 for _ in range(num_classes)),
        calibrator_available=False,
    )


def _effective_sample_size(weights: torch.Tensor, valid: torch.Tensor) -> float:
    effective = weights.detach().float() * valid.detach().float()
    denominator = effective.square().sum()
    if not bool(denominator.gt(0)):
        return 0.0
    return float((effective.sum().square() / denominator.clamp_min(1.0e-8)).cpu())


class LCRSegV02Method(LCRSegV01Method):
    """Single-anchor V0.2 with admission and rejection-only routing."""

    method_name = "lcrseg_v0_2"
    method_version = "0.2"
    requires_labeled_calibration = True

    def __init__(self, model, *, config: Mapping[str, Any] | None = None) -> None:
        merged = dict(LCR_V02_DEFAULTS)
        if config:
            merged.update(dict(config))
        super().__init__(model, config=merged)
        self.compatibility_calibrator = self._new_calibrator()
        self.calibration_history: list[dict[str, Any]] = []
        self.v02_statistics: dict[str, Any] = self._new_v02_statistics()

    def _new_calibrator(self) -> LabeledOnlyCompatibilityCalibrator:
        return LabeledOnlyCompatibilityCalibrator(
            num_classes=self.num_classes,
            bins=int(self.config["calibration_bins"]),
            min_pixels=int(self.config["calibration_min_pixels"]),
        )

    @staticmethod
    def _new_v02_statistics() -> dict[str, Any]:
        return {
            "steps": 0,
            "pseudo_valid_count": 0,
            "assim_selected_count": 0,
            "relation_valid_count": 0,
            "compat_rejected_count": 0,
            "candidate_counts_by_class": [],
            "selected_counts_by_class": [],
            "relation_counts_by_class": [],
            "rejected_counts_by_class": [],
        }

    def begin_site(self, site_id: str, previous_checkpoint, total_steps: int) -> None:  # type: ignore[override]
        super().begin_site(site_id, previous_checkpoint, total_steps)
        self.compatibility_calibrator = self._new_calibrator()
        self.calibration_history = []
        self.v02_statistics = self._new_v02_statistics()

    def _admission(
        self,
        pseudo: PseudoLabelOutput,
        raw_learnability: LearnabilityOutput,
        strong_valid_mask: torch.Tensor,
        *,
        site_step: int,
    ) -> ProgressiveAdmissionOutput:
        relation_valid = strict_relation_valid_mask(strong_valid_mask, raw_learnability.score.shape[-2:])
        if bool(self.config["progressive_admission"]):
            return classwise_progressive_admission(
                pseudo,
                raw_learnability.score,
                relation_valid,
                num_classes=self.num_classes,
                site_step=site_step,
                total_site_steps=self.total_steps,
                pi_start=float(self.config["pi_start"]),
                pi_end=float(self.config["pi_end"]),
            )
        output = _uniform_admission(pseudo, relation_valid)
        counts = tuple(int((output.mask[:, 0] & pseudo.labels.eq(class_id)).sum()) for class_id in range(self.num_classes))
        return replace(output, candidate_counts=counts, selected_counts=counts)

    def _consolidation(
        self,
        raw_compatibility: CompatibilityOutput,
        old_relation: RelationOutput | None,
        strong_valid_mask: torch.Tensor,
    ) -> RejectionOnlyOutput:
        relation_valid = strict_relation_valid_mask(strong_valid_mask, raw_compatibility.score.shape[-2:])
        if old_relation is None:
            return _uniform_rejection(raw_compatibility, relation_valid, num_classes=self.num_classes, old_predicted_class=None)
        if not bool(self.config["compatibility_calibration"]) or not bool(self.config["compatibility_rejection"]):
            return _uniform_rejection(
                raw_compatibility,
                relation_valid,
                num_classes=self.num_classes,
                old_predicted_class=old_relation.predicted_class,
            )
        calibrated, available = self.compatibility_calibrator.calibrate(raw_compatibility.score, old_relation.predicted_class)
        return rejection_only_weights(
            calibrated,
            old_relation.predicted_class,
            relation_valid,
            num_classes=self.num_classes,
            calibrator_available=available,
            probability_threshold=float(self.config["compatibility_probability_threshold"]),
            max_reject_fraction_per_class=float(self.config["max_reject_fraction_per_class"]),
            rejected_weight_floor=float(self.config["rejected_weight_floor"]),
        )

    def _calibrator_status(self, *, old_relation: RelationOutput | None) -> str:
        if old_relation is None:
            return "not_applicable_first_site"
        if not bool(self.config["compatibility_calibration"]) or not bool(self.config["compatibility_rejection"]):
            return "disabled_uniform"
        return self.compatibility_calibrator.status

    def _record_branch_statistics(self, admission: ProgressiveAdmissionOutput, consolidation: RejectionOnlyOutput) -> None:
        stats = self.v02_statistics
        stats["steps"] += 1
        stats["pseudo_valid_count"] += int(sum(admission.candidate_counts))
        stats["assim_selected_count"] += int(sum(admission.selected_counts))
        stats["relation_valid_count"] += int(sum(consolidation.candidate_counts))
        stats["compat_rejected_count"] += int(sum(consolidation.rejected_counts))
        for key, values in (
            ("candidate_counts_by_class", admission.candidate_counts),
            ("selected_counts_by_class", admission.selected_counts),
            ("relation_counts_by_class", consolidation.candidate_counts),
            ("rejected_counts_by_class", consolidation.rejected_counts),
        ):
            if not stats[key]:
                stats[key] = [0 for _ in values]
            stats[key] = [int(left) + int(right) for left, right in zip(stats[key], values, strict=True)]

    def training_step(
        self,
        labeled_batch: LabeledBatch,
        unlabeled_batch: UnlabeledBatch | None,
        global_step: int,
        site_step: int,
    ) -> MethodStepOutput:
        if unlabeled_batch is None:
            raise ValueError("LCR-Seg V0.2 requires an unlabeled batch")
        labeled_output = self.model(labeled_batch.image)
        with torch.no_grad():
            weak_output = self.model(unlabeled_batch.weak_image)
        strong_output = self.model(unlabeled_batch.strong_image)
        anchor_ready = self.current_anchor_bank.all_classes_valid
        weak_relation: RelationOutput | None = None
        strong_relation: RelationOutput | None = None
        anchor_loss = differentiable_zero(labeled_output.logits)
        if anchor_ready:
            labeled_relation = self._relation(labeled_output.relation_features, self.current_anchor_bank)
            weak_relation = self._relation(weak_output.relation_features, self.current_anchor_bank)
            strong_relation = self._relation(strong_output.relation_features, self.current_anchor_bank)
            from .base import relation_supervision_loss

            anchor_loss = relation_supervision_loss(labeled_relation.logits, labeled_batch.label, labeled_batch.valid_mask)
        losses = self._all_loss_keys(
            labeled_output.logits,
            self._supervised_losses(labeled_output, labeled_batch, relation_anchor_loss=anchor_loss),
        )
        pseudo: PseudoLabelOutput | None = None
        raw_learnability: LearnabilityOutput | None = None
        raw_compatibility = zero_compatibility(
            weak_relation.probabilities if weak_relation is not None else weak_output.relation_features[:, :1]
        )
        old_relation: RelationOutput | None = None
        admission: ProgressiveAdmissionOutput | None = None
        consolidation: RejectionOnlyOutput | None = None
        assimilation = differentiable_zero(strong_output.logits)
        relation_loss = differentiable_zero(strong_output.logits)
        if anchor_ready and weak_relation is not None and strong_relation is not None and not self._bootstrap_active():
            pseudo = build_pseudo_labels(
                weak_output.logits.detach().softmax(dim=1),
                weak_relation,
                tau_cls=float(self.config["tau_cls"]),
                tau_anchor=float(self.config["tau_anchor"]),
                delta_anchor=float(self.config["delta_anchor"]),
                tau_spatial=float(self.config["tau_spatial"]),
                temperature_cls=float(self.config["temperature_cls"]),
                temperature_anchor=float(self.config["temperature_anchor"]),
                spatial_floor=float(self.config["spatial_floor"]),
            )
            raw_learnability = compute_learnability(
                weak_output.logits,
                weak_relation,
                pseudo,
                site_step=site_step,
                total_steps=self.total_steps,
                rank_start=float(self.config["rank_start"]),
                rank_end=float(self.config["rank_end"]),
                rank_temperature=float(self.config["rank_temperature"]),
                relation_margin_center=float(self.config["relation_margin_center"]),
                relation_margin_temperature=float(self.config["relation_margin_temperature"]),
                min_rank_pixels=int(self.config["min_rank_pixels"]),
            )
            admission = self._admission(pseudo, raw_learnability, unlabeled_batch.strong_valid_mask, site_step=site_step)
            assimilation = admission_assimilation_loss(strong_output.logits, pseudo, admission, unlabeled_batch.strong_valid_mask)
            if self.old_model is not None and self.old_anchor_bank is not None:
                self.old_model.eval()
                with torch.no_grad():
                    old_output = self.old_model(unlabeled_batch.weak_image)
                    old_relation = self._relation(old_output.relation_features, self.old_anchor_bank)
                raw_compatibility = compute_compatibility(
                    weak_relation,
                    old_relation,
                    old_margin_center=float(self.config["old_margin_center"]),
                    old_margin_temperature=float(self.config["old_margin_temperature"]),
                    js_temperature=float(self.config["js_temperature"]),
                    spatial_floor=float(self.config["spatial_floor"]),
                )
            consolidation = self._consolidation(raw_compatibility, old_relation, unlabeled_batch.strong_valid_mask)
            if old_relation is not None:
                relation_loss = rejection_only_relation_loss(
                    strong_relation,
                    old_relation,
                    consolidation,
                    unlabeled_batch.strong_valid_mask,
                    distill_temperature=float(self.config["distill_temperature"]),
                )
            # Anchor-memory behavior remains the frozen V0.1 raw L/C rule; the
            # new admission/rejection masks change only the two permitted losses.
            self._cache_unlabeled_anchor_update(
                weak_output.relation_features,
                pseudo,
                raw_learnability,
                raw_compatibility,
                site_step,
            )
        if not self._bootstrap_active() or site_step >= int(self.config["anchor_bootstrap_steps"]):
            self._cache_labeled_anchor_update(labeled_output.relation_features, labeled_batch, site_step)
        losses["loss_assim"] = assimilation
        losses["loss_relation"] = relation_loss
        lambda_assim = 0.0 if self._bootstrap_active() else float(self.config["lambda_assim"]) * self._assimilation_ramp(
            site_step,
            bootstrap_complete_at=int(self.bootstrap_state.get("completed_at_site_step", -1)),
        )
        lambda_relation = 0.0 if self.old_model is None else float(self.config["lambda_relation"]) * self._relation_ramp(site_step)
        total = losses["loss_sup"] + lambda_assim * assimilation + lambda_relation * relation_loss
        relation_valid_mask = consolidation.relation_valid_mask if consolidation is not None else torch.zeros_like(raw_compatibility.score, dtype=torch.bool)
        selected_counts = admission.selected_counts if admission is not None else tuple(0 for _ in range(self.num_classes))
        candidate_counts = admission.candidate_counts if admission is not None else tuple(0 for _ in range(self.num_classes))
        relation_counts = consolidation.candidate_counts if consolidation is not None else tuple(0 for _ in range(self.num_classes))
        rejected_counts = consolidation.rejected_counts if consolidation is not None else tuple(0 for _ in range(self.num_classes))
        selected_total = int(sum(selected_counts))
        candidate_total = int(sum(candidate_counts))
        rejected_total = int(sum(rejected_counts))
        relation_total = int(sum(relation_counts))
        calibrated_c = consolidation.calibrated_compatibility if consolidation is not None else raw_compatibility.score
        consolidation_weights = consolidation.weights if consolidation is not None else torch.ones_like(raw_compatibility.score)
        scalars: dict[str, Any] = {
            "lambda_assim_effective": lambda_assim,
            "lambda_relation_effective": lambda_relation,
            "loss_rel": float(relation_loss.detach()),
            "bootstrap_complete": float(not self._bootstrap_active()),
            "relation_anchor_ready": float(anchor_ready),
            "site_progress": admission.site_progress if admission is not None else 0.0,
            "pseudo_valid_count": candidate_total,
            "assim_selected_count": selected_total,
            "assim_selected_fraction": selected_total / candidate_total if candidate_total else 0.0,
            "assim_candidate_counts_by_class": list(candidate_counts),
            "assim_selected_counts_by_class": list(selected_counts),
            "assim_selected_fraction_by_class": list(admission.selected_fraction_by_class) if admission is not None else [0.0] * self.num_classes,
            "relation_valid_count": relation_total,
            "compat_rejected_count": rejected_total,
            "compat_rejected_fraction": rejected_total / relation_total if relation_total else 0.0,
            "relation_valid_counts_by_class": list(relation_counts),
            "compat_rejected_counts_by_class": list(rejected_counts),
            "compat_rejected_fraction_by_class": list(consolidation.rejected_fraction_by_class) if consolidation is not None else [0.0] * self.num_classes,
            "consolidation_weight_mean": float(consolidation_weights[relation_valid_mask].float().mean()) if bool(relation_valid_mask.any()) else 0.0,
            "consolidation_effective_sample_size": _effective_sample_size(consolidation_weights, relation_valid_mask),
            "raw_C_mean": float(raw_compatibility.score.detach().mean()),
            "calibrated_C_mean": float(calibrated_c.detach().mean()),
            "calibrator_status": self._calibrator_status(old_relation=old_relation),
            "calibrator_last_update_epoch": int(self.compatibility_calibrator.last_update_epoch),
            "pseudo_valid_ratio": float(candidate_total / max(1, raw_compatibility.score.numel())),
            "compatibility_mean": float(raw_compatibility.score.detach().mean()),
            "relation_js_mean": float(raw_compatibility.js_divergence.detach().mean()),
        }
        maps: dict[str, torch.Tensor] = {
            "raw_learnability": raw_learnability.score.detach() if raw_learnability is not None else torch.zeros_like(raw_compatibility.score),
            "admission_mask": admission.mask.detach() if admission is not None else torch.zeros_like(raw_compatibility.score, dtype=torch.bool),
            "raw_compatibility": raw_compatibility.score.detach(),
            "calibrated_compatibility": calibrated_c.detach(),
            "rejection_mask": consolidation.rejection_mask.detach() if consolidation is not None else torch.zeros_like(raw_compatibility.score, dtype=torch.bool),
            "consolidation_weights": consolidation_weights.detach(),
            "relation_valid_mask": relation_valid_mask.detach(),
            "compatibility": raw_compatibility.score.detach(),
        }
        if pseudo is not None:
            maps.update(
                {
                    "pseudo_labels": pseudo.labels.detach(),
                    "pseudo_valid": pseudo.valid.detach(),
                    "pseudo_source": pseudo.source.detach(),
                    "learnability": raw_learnability.score.detach() if raw_learnability is not None else torch.zeros_like(raw_compatibility.score),
                }
            )
        if weak_relation is not None:
            maps["current_relation_probability"] = weak_relation.probabilities.detach()
        if old_relation is not None:
            maps["old_relation_probability"] = old_relation.probabilities.detach()
        if admission is not None and consolidation is not None:
            self._record_branch_statistics(admission, consolidation)
        return MethodStepOutput(total_loss=total, losses=losses, scalars=scalars, maps=maps)

    @torch.no_grad()
    def on_epoch_end(self, *, epoch: int, calibration_batcher: Any, device: torch.device | str) -> dict[str, Any]:
        """Fit only on deterministic current-site visible labels at fixed epochs."""

        if self.old_model is None or self.old_anchor_bank is None:
            self.compatibility_calibrator.status = "not_applicable_first_site"
            return {"status": self.compatibility_calibrator.status, "updated": False, "rows": []}
        if not bool(self.config["compatibility_calibration"]) or not bool(self.config["compatibility_rejection"]):
            self.compatibility_calibrator.status = "disabled_uniform"
            return {"status": self.compatibility_calibrator.status, "updated": False, "rows": []}
        interval = int(self.config["calibration_update_epochs"])
        if epoch < interval - 1:
            self.compatibility_calibrator.status = "warmup_uniform"
            return {"status": self.compatibility_calibrator.status, "updated": False, "rows": []}
        if (epoch + 1) % interval:
            return {"status": self.compatibility_calibrator.status, "updated": False, "rows": []}
        device_obj = torch.device(device)
        was_training = self.model.training
        self.model.eval()
        self.old_model.eval()
        raw_scores: list[torch.Tensor] = []
        old_classes: list[torch.Tensor] = []
        correctness: list[torch.Tensor] = []
        valid_masks: list[torch.Tensor] = []
        for step in range(int(calibration_batcher.steps_per_epoch)):
            batch = calibration_batcher.batch_at(step)
            if not isinstance(batch, LabeledBatch):
                raise TypeError("compatibility calibration accepts only LabeledBatch from current train_labeled")
            visible = batch.to(device_obj)
            current_output = self.model(visible.image)
            current_relation = self._relation(current_output.relation_features, self.current_anchor_bank)
            old_output = self.old_model(visible.image)
            old_relation = self._relation(old_output.relation_features, self.old_anchor_bank)
            raw = compute_compatibility(
                current_relation,
                old_relation,
                old_margin_center=float(self.config["old_margin_center"]),
                old_margin_temperature=float(self.config["old_margin_temperature"]),
                js_temperature=float(self.config["js_temperature"]),
                spatial_floor=float(self.config["spatial_floor"]),
            )
            grid_label = F.interpolate(visible.label.unsqueeze(1).float(), size=raw.score.shape[-2:], mode="nearest")[:, 0].long()
            valid = downsample_valid_mask(visible.valid_mask, raw.score.shape[-2:])
            raw_scores.append(raw.score.detach())
            old_classes.append(old_relation.predicted_class.detach())
            correctness.append(old_relation.predicted_class.eq(grid_label).detach())
            valid_masks.append(valid.detach())
        if was_training:
            self.model.train()
        rows = self.compatibility_calibrator.fit(
            torch.cat(raw_scores, dim=0),
            torch.cat(old_classes, dim=0),
            torch.cat(correctness, dim=0),
            torch.cat(valid_masks, dim=0),
            epoch=int(epoch),
        )
        annotated = [{**row, "site_id": self.site_id or "", "site_index": self.site_index, "epoch": int(epoch)} for row in rows]
        self.calibration_history.extend(annotated)
        return {"status": self.compatibility_calibrator.status, "updated": True, "rows": annotated}

    def end_site(self, site_id: str) -> dict[str, Any]:
        result = super().end_site(site_id)
        result.update(
            {
                "compatibility_calibrator_status": self.compatibility_calibrator.status,
                "compatibility_calibrator_last_update_epoch": self.compatibility_calibrator.last_update_epoch,
                "calibration_rows": len(self.calibration_history),
                "v02_branch_statistics": dict(self.v02_statistics),
            }
        )
        return result

    def write_site_artifacts(self, *, run_dir: Path, site_id: str, site_index: int) -> None:
        prefix = f"site{site_index}_{site_id}"
        write_csv(
            run_dir / f"calibration_{prefix}.csv",
            self.calibration_history,
            fieldnames=["site_id", "site_index", "epoch", "scope", "class_id", "bin", "upper_edge", "pixel_count", "correct_count", "laplace_accuracy", "pava_probability", "pava_weight"],
        )
        write_json(
            run_dir / f"branch_statistics_{prefix}.json",
            {
                "site_id": site_id,
                "site_index": site_index,
                "calibrator_status": self.compatibility_calibrator.status,
                "calibrator_last_update_epoch": self.compatibility_calibrator.last_update_epoch,
                "statistics": self.v02_statistics,
            },
        )

    def method_state_dict(self) -> dict[str, Any]:
        state = super().method_state_dict()
        statistics = dict(state["method_statistics"])
        statistics.update(
            {
                "compatibility_calibrator_state": self.compatibility_calibrator.state_dict(),
                "calibration_history": [dict(row) for row in self.calibration_history],
                "v02_statistics": dict(self.v02_statistics),
            }
        )
        state["method_statistics"] = statistics
        return state

    def load_method_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_method_state_dict(state)
        statistics = dict(state.get("method_statistics") or {})
        self.compatibility_calibrator = self._new_calibrator()
        self.compatibility_calibrator.load_state_dict(dict(statistics.get("compatibility_calibrator_state") or {}))
        self.calibration_history = [dict(row) for row in statistics.get("calibration_history", [])]
        self.v02_statistics = dict(statistics.get("v02_statistics") or self._new_v02_statistics())
