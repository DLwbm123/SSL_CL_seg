"""LCR-Seg V0.2a: explicit, preregistered 2x2 routing semantics."""
from __future__ import annotations

import csv
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from ..common import write_csv, write_json
from ..contracts import LabeledBatch, MethodStepOutput, UnlabeledBatch, differentiable_zero
from ..data.transforms import downsample_valid_mask
from .base import merged_method_config, relation_supervision_loss
from .components.compatibility import CompatibilityOutput, compute_compatibility, zero_compatibility
from .components.learnability import LearnabilityOutput, compute_learnability
from .components.progressive_admission import (
    ProgressiveAdmissionOutput,
    admission_assimilation_loss,
    classwise_progressive_admission,
    strict_relation_valid_mask,
)
from .components.pseudo_label import PseudoLabelOutput, build_pseudo_labels
from .components.rejection_only_routing import (
    RejectionOnlyOutput,
    rejection_only_relation_loss,
    rejection_only_weights,
)
from .components.relation_field import RelationOutput
from .components.routing import assimilation_loss, relation_consolidation_loss
from .components.teacher_validity import TeacherValidityCalibrator, TeacherValidityOutput, compute_teacher_validity
from .lcrseg_v0_1 import LCR_DEFAULTS, LCRSegV01Method, _uniform_compatibility


class AssimilationMode(str, Enum):
    LEGACY_CONTINUOUS_V01 = "legacy_continuous_v01"
    PROGRESSIVE_ADMISSION = "progressive_admission"
    UNIT_ALL = "unit_all"


class ConsolidationMode(str, Enum):
    UNIFORM_RELATION = "uniform_relation"
    CALIBRATED_TEACHER_REJECTION = "calibrated_teacher_rejection"


_REMOVED_BOOLEAN_KEYS = {
    "use_learnability",
    "use_compatibility",
    "progressive_admission",
    "compatibility_calibration",
    "compatibility_rejection",
}

LCR_V02A_DEFAULTS: dict[str, Any] = {
    **{key: value for key, value in LCR_DEFAULTS.items() if key not in _REMOVED_BOOLEAN_KEYS},
    "protocol_id": "lcrseg_v0_2a",
    "assimilation_mode": AssimilationMode.LEGACY_CONTINUOUS_V01.value,
    "consolidation_mode": ConsolidationMode.UNIFORM_RELATION.value,
    "variant_id": "R0",
    "learnability_formula_version": "lcrseg_v0_1_frozen",
    "teacher_validity_formula_version": "old_only_geometric_mean_v0_2a",
    "calibrator_version": TeacherValidityCalibrator.version,
    "progressive_schedule": {
        "start_fraction": 0.40,
        "end_fraction": 0.80,
        "schedule": "linear",
        "schedule_scope": "per_site",
        "classwise": True,
        "minimum_pixels_for_class_quantile": 32,
        "minimum_admitted_per_present_class": 1,
        "weight_after_admission": 1.0,
    },
    "teacher_validity_margin_temperature": 0.05,
    "teacher_validity_spatial_floor": 0.25,
    "calibration_bins": 20,
    "calibration_minimum_pixels_per_class": 2048,
    "calibration_maximum_pixels_per_class": 100000,
    "rejection_threshold": 0.70,
    "rejection_floor": 0.50,
    "rejection_cap": 0.20,
    "formal_r0_run": "/home/jiangsuiyang/SSL_CL/runs/fundus_seed0_lcrseg_uniform_relation_kd_full200e",
    "auxiliary_u0_run": "/home/jiangsuiyang/SSL_CL/runs/fundus_seed0_lcrseg_v0_2_r0_uniform_full200e",
    "u0_registered_as_formal_r0": False,
}


_REGISTERED_VARIANTS = {
    "R0": (AssimilationMode.LEGACY_CONTINUOUS_V01.value, ConsolidationMode.UNIFORM_RELATION.value),
    "R1": (AssimilationMode.PROGRESSIVE_ADMISSION.value, ConsolidationMode.UNIFORM_RELATION.value),
    "R2": (
        AssimilationMode.LEGACY_CONTINUOUS_V01.value,
        ConsolidationMode.CALIBRATED_TEACHER_REJECTION.value,
    ),
    "R3": (
        AssimilationMode.PROGRESSIVE_ADMISSION.value,
        ConsolidationMode.CALIBRATED_TEACHER_REJECTION.value,
    ),
    "U0": (AssimilationMode.UNIT_ALL.value, ConsolidationMode.UNIFORM_RELATION.value),
}


def _boolean_assimilation_mode(config: Mapping[str, Any]) -> str | None:
    if "progressive_admission" in config:
        return (
            AssimilationMode.PROGRESSIVE_ADMISSION.value
            if bool(config["progressive_admission"])
            else AssimilationMode.UNIT_ALL.value
        )
    if "use_learnability" in config:
        return (
            AssimilationMode.LEGACY_CONTINUOUS_V01.value
            if bool(config["use_learnability"])
            else AssimilationMode.UNIT_ALL.value
        )
    return None


def _boolean_consolidation_mode(config: Mapping[str, Any]) -> str | None:
    calibration_present = "compatibility_calibration" in config
    rejection_present = "compatibility_rejection" in config
    if calibration_present or rejection_present:
        if not calibration_present or not rejection_present:
            raise ValueError("legacy calibration/rejection booleans must be provided together")
        calibration = bool(config["compatibility_calibration"])
        rejection = bool(config["compatibility_rejection"])
        if calibration != rejection:
            raise ValueError("conflicting legacy calibration/rejection booleans")
        return (
            ConsolidationMode.CALIBRATED_TEACHER_REJECTION.value
            if calibration
            else ConsolidationMode.UNIFORM_RELATION.value
        )
    if "use_compatibility" in config:
        if bool(config["use_compatibility"]):
            raise ValueError("legacy current-old compatibility has no registered V0.2a consolidation enum")
        return ConsolidationMode.UNIFORM_RELATION.value
    return None


def resolve_v02a_method_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve booleans once, remove them, and reject semantic conflicts."""

    provided = dict(config or {})
    if provided.get("protocol_id", "lcrseg_v0_2a") != "lcrseg_v0_2a":
        raise ValueError("V0.2a requires protocol_id=lcrseg_v0_2a")
    boolean_assimilation = _boolean_assimilation_mode(provided)
    boolean_consolidation = _boolean_consolidation_mode(provided)
    explicit_assimilation = provided.get("assimilation_mode")
    explicit_consolidation = provided.get("consolidation_mode")
    if explicit_assimilation is not None and boolean_assimilation is not None and explicit_assimilation != boolean_assimilation:
        raise ValueError("assimilation enum conflicts with legacy booleans")
    if explicit_consolidation is not None and boolean_consolidation is not None and explicit_consolidation != boolean_consolidation:
        raise ValueError("consolidation enum conflicts with legacy booleans")
    resolved = dict(LCR_V02A_DEFAULTS)
    resolved.update({key: value for key, value in provided.items() if key not in _REMOVED_BOOLEAN_KEYS})
    if explicit_assimilation is None and boolean_assimilation is not None:
        resolved["assimilation_mode"] = boolean_assimilation
    if explicit_consolidation is None and boolean_consolidation is not None:
        resolved["consolidation_mode"] = boolean_consolidation
    try:
        resolved["assimilation_mode"] = AssimilationMode(str(resolved["assimilation_mode"])).value
        resolved["consolidation_mode"] = ConsolidationMode(str(resolved["consolidation_mode"])).value
    except ValueError as exc:
        raise ValueError("unregistered V0.2a method enum") from exc
    if any(key in resolved for key in _REMOVED_BOOLEAN_KEYS):
        raise AssertionError("resolved V0.2a config retained an ambiguous boolean")
    variant = str(resolved.get("variant_id", "")).upper()
    if variant in _REGISTERED_VARIANTS:
        actual = (resolved["assimilation_mode"], resolved["consolidation_mode"])
        if actual != _REGISTERED_VARIANTS[variant]:
            raise ValueError(f"{variant} enum combination conflicts with the registered protocol")
    if variant == "R0" and resolved["assimilation_mode"] == AssimilationMode.UNIT_ALL.value:
        raise ValueError("U0 unit-all semantics may not be registered as formal R0")
    schedule = dict(resolved["progressive_schedule"])
    expected_schedule = {
        "start_fraction": 0.40,
        "end_fraction": 0.80,
        "schedule": "linear",
        "schedule_scope": "per_site",
        "classwise": True,
        "minimum_pixels_for_class_quantile": 32,
        "minimum_admitted_per_present_class": 1,
        "weight_after_admission": 1.0,
    }
    if schedule != expected_schedule:
        raise ValueError("V0.2a progressive schedule differs from the frozen protocol")
    for key, expected in (("rejection_threshold", 0.70), ("rejection_floor", 0.50), ("rejection_cap", 0.20)):
        if float(resolved[key]) != expected:
            raise ValueError(f"{key} differs from the frozen V0.2a protocol")
    resolved["progressive_schedule"] = schedule
    return merged_method_config(resolved)


def _uniform_admission(
    pseudo: PseudoLabelOutput,
    relation_valid_mask: torch.Tensor,
    *,
    num_classes: int,
) -> ProgressiveAdmissionOutput:
    mask = (pseudo.valid.detach().bool() & relation_valid_mask.detach().bool()).detach()
    counts = tuple(int((mask[:, 0] & pseudo.labels.eq(class_id)).sum()) for class_id in range(num_classes))
    return ProgressiveAdmissionOutput(
        mask=mask,
        candidate_mask=mask,
        site_progress=0.0,
        target_fraction=1.0,
        candidate_counts=counts,
        selected_counts=counts,
        learnability_thresholds=tuple(float("nan") for _ in counts),
    )


def _uniform_consolidation(
    reference: CompatibilityOutput,
    relation_valid_mask: torch.Tensor,
    *,
    num_classes: int,
    old_predicted_class: torch.Tensor | None,
) -> RejectionOnlyOutput:
    valid = relation_valid_mask.detach().bool()
    counts = tuple(
        int((valid[:, 0] & old_predicted_class.eq(class_id)).sum()) if old_predicted_class is not None else 0
        for class_id in range(num_classes)
    )
    return RejectionOnlyOutput(
        calibrated_compatibility=torch.ones_like(reference.score).detach(),
        rejection_mask=torch.zeros_like(valid),
        weights=torch.ones_like(reference.score).detach(),
        relation_valid_mask=valid,
        candidate_counts=counts,
        rejected_counts=tuple(0 for _ in counts),
        calibrator_available=False,
    )


def _effective_sample_size(weights: torch.Tensor, valid: torch.Tensor) -> float:
    effective = weights.detach().float() * valid.detach().float()
    denominator = effective.square().sum()
    if not bool(denominator.gt(0)):
        return 0.0
    return float((effective.sum().square() / denominator.clamp_min(1.0e-8)).cpu())


class LCRSegV02AMethod(LCRSegV01Method):
    method_name = "lcrseg_v0_2a"
    method_version = "0.2a"

    def __init__(self, model, *, config: Mapping[str, Any] | None = None) -> None:
        resolved = resolve_v02a_method_config(config)
        super().__init__(model, config=resolved)
        self.config = resolved
        self.requires_labeled_calibration = (
            self.config["consolidation_mode"] == ConsolidationMode.CALIBRATED_TEACHER_REJECTION.value
        )
        self.teacher_validity_calibrator = self._new_teacher_validity_calibrator()
        self.calibration_rows: list[dict[str, Any]] = []
        self.branch_rows: list[dict[str, Any]] = []
        self.v02a_statistics = self._new_statistics()
        self._epoch_context = 0

    def _new_teacher_validity_calibrator(self) -> TeacherValidityCalibrator:
        return TeacherValidityCalibrator(
            num_classes=self.num_classes,
            bins=int(self.config["calibration_bins"]),
            minimum_pixels_per_class=int(self.config["calibration_minimum_pixels_per_class"]),
            maximum_pixels_per_class=int(self.config["calibration_maximum_pixels_per_class"]),
        )

    @staticmethod
    def _new_statistics() -> dict[str, Any]:
        return {
            "steps": 0,
            "hidden_gt_training_usage": 0,
            "old_model_only_teacher_validity": True,
            "current_old_js_used_for_gate": False,
            "pseudo_valid_count": 0,
            "assim_candidate_count": 0,
            "admitted_count": 0,
            "relation_valid_count": 0,
            "rejected_count": 0,
        }

    def protocol_semantics(self) -> dict[str, Any]:
        return {
            key: self.config[key]
            for key in (
                "protocol_id",
                "variant_id",
                "assimilation_mode",
                "consolidation_mode",
                "learnability_formula_version",
                "teacher_validity_formula_version",
                "calibrator_version",
                "progressive_schedule",
                "rejection_threshold",
                "rejection_floor",
                "rejection_cap",
                "formal_r0_run",
                "auxiliary_u0_run",
                "u0_registered_as_formal_r0",
            )
        }

    def begin_site(self, site_id: str, previous_checkpoint, total_steps: int) -> None:  # type: ignore[override]
        super().begin_site(site_id, previous_checkpoint, total_steps)
        self.teacher_validity_calibrator = self._new_teacher_validity_calibrator()
        self.calibration_rows = []
        self.branch_rows = []
        self.v02a_statistics = self._new_statistics()
        self._epoch_context = 0

    def set_training_context(self, *, epoch: int, steps_per_epoch: int) -> None:
        if epoch < 0 or steps_per_epoch < 1:
            raise ValueError("invalid V0.2a training context")
        self._epoch_context = int(epoch)

    @torch.no_grad()
    def calibrate_at_site_start(
        self,
        *,
        calibration_batcher: Any,
        device: torch.device | str,
        run_dir: Path,
    ) -> dict[str, Any]:
        if self.config["consolidation_mode"] != ConsolidationMode.CALIBRATED_TEACHER_REJECTION.value:
            self.teacher_validity_calibrator.status = "disabled_uniform_relation"
            return {"status": self.teacher_validity_calibrator.status, "updated": False}
        if self.old_model is None or self.old_anchor_bank is None:
            self.teacher_validity_calibrator.status = "not_applicable_first_site"
            return {"status": self.teacher_validity_calibrator.status, "updated": False}
        if self.teacher_validity_calibrator.available:
            return {"status": self.teacher_validity_calibrator.status, "updated": False, "resume_reused": True}
        device_obj = torch.device(device)
        self.old_model.eval()
        raw_scores: list[torch.Tensor] = []
        old_classes: list[torch.Tensor] = []
        correctness: list[torch.Tensor] = []
        valid_masks: list[torch.Tensor] = []
        for step in range(int(calibration_batcher.steps_per_epoch)):
            batch = calibration_batcher.batch_at(step)
            if not isinstance(batch, LabeledBatch):
                raise TypeError("V0.2a calibration accepts current-site train_labeled batches only")
            visible = batch.to(device_obj)
            old_output = self.old_model(visible.image)
            old_relation = self._relation(old_output.relation_features, self.old_anchor_bank)
            validity = compute_teacher_validity(
                old_output.logits,
                old_relation,
                margin_temperature=float(self.config["teacher_validity_margin_temperature"]),
                spatial_floor=float(self.config["teacher_validity_spatial_floor"]),
            )
            grid_label = F.interpolate(
                visible.label.unsqueeze(1).float(), size=validity.raw_score.shape[-2:], mode="nearest"
            )[:, 0].long()
            valid = downsample_valid_mask(visible.valid_mask, validity.raw_score.shape[-2:])
            raw_scores.append(validity.raw_score)
            old_classes.append(validity.old_predicted_class)
            correctness.append(validity.old_predicted_class.eq(grid_label).detach())
            valid_masks.append(valid.detach())
        rows = self.teacher_validity_calibrator.fit(
            torch.cat(raw_scores, dim=0),
            torch.cat(old_classes, dim=0),
            torch.cat(correctness, dim=0),
            torch.cat(valid_masks, dim=0),
            site_id=self.site_id or "",
        )
        self.calibration_rows = [
            {"site": self.site_id or "", "site_index": self.site_index, **row}
            for row in rows
        ]
        self._write_calibrator_artifacts(Path(run_dir))
        return {
            "status": self.teacher_validity_calibrator.status,
            "updated": True,
            "rows": len(self.calibration_rows),
            "sample_counts_by_class": self.teacher_validity_calibrator.sample_counts_by_class,
        }

    def _write_calibrator_artifacts(self, run_dir: Path) -> None:
        if not self.calibration_rows:
            return
        site_dir = run_dir / f"site_{self.site_id}"
        site_dir.mkdir(parents=True, exist_ok=True)
        fields = [
            "site",
            "site_index",
            "class_id",
            "fallback_scope",
            "bin",
            "bin_left",
            "bin_right",
            "bin_count",
            "raw_accuracy",
            "smoothed_accuracy",
            "pava_accuracy",
            "brier_raw",
            "brier_calibrated",
            "ece_raw",
            "ece_calibrated",
        ]
        write_csv(site_dir / "teacher_validity_calibrator.csv", self.calibration_rows, fieldnames=fields)
        write_json(site_dir / "teacher_validity_calibrator.json", self.teacher_validity_calibrator.state_dict())

    def _compute_admission(
        self,
        pseudo: PseudoLabelOutput,
        learnability: LearnabilityOutput,
        strong_valid_mask: torch.Tensor,
        *,
        site_step: int,
    ) -> ProgressiveAdmissionOutput:
        relation_valid = strict_relation_valid_mask(strong_valid_mask, learnability.score.shape[-2:])
        if self.config["assimilation_mode"] == AssimilationMode.PROGRESSIVE_ADMISSION.value:
            schedule = self.config["progressive_schedule"]
            return classwise_progressive_admission(
                pseudo,
                learnability.score,
                relation_valid,
                num_classes=self.num_classes,
                site_step=site_step,
                total_site_steps=self.total_steps,
                pi_start=float(schedule["start_fraction"]),
                pi_end=float(schedule["end_fraction"]),
                minimum_pixels_for_class_quantile=int(schedule["minimum_pixels_for_class_quantile"]),
                minimum_admitted_per_present_class=int(schedule["minimum_admitted_per_present_class"]),
            )
        return _uniform_admission(pseudo, relation_valid, num_classes=self.num_classes)

    def _record_branch_rows(
        self,
        *,
        site_step: int,
        admission: ProgressiveAdmissionOutput,
        consolidation: RejectionOnlyOutput,
    ) -> None:
        for class_id in range(self.num_classes):
            valid_count = int(admission.candidate_counts[class_id])
            admitted_count = int(admission.selected_counts[class_id])
            relation_count = int(consolidation.candidate_counts[class_id])
            rejected_count = int(consolidation.rejected_counts[class_id])
            self.branch_rows.append(
                {
                    "site": self.site_id or "",
                    "site_index": self.site_index,
                    "epoch": self._epoch_context,
                    "step": site_step,
                    "predicted_class": class_id,
                    "assimilation_mode": self.config["assimilation_mode"],
                    "consolidation_mode": self.config["consolidation_mode"],
                    "target_fraction": admission.target_fraction,
                    "realized_fraction": admitted_count / valid_count if valid_count else 0.0,
                    "valid_count": valid_count,
                    "admitted_count": admitted_count,
                    "deferred_count": valid_count - admitted_count,
                    "learnability_threshold": admission.learnability_thresholds[class_id],
                    "relation_valid_count": relation_count,
                    "rejected_count": rejected_count,
                    "rejected_fraction": rejected_count / relation_count if relation_count else 0.0,
                }
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
        """Protocol hook retaining the exact frozen V0.1/R1 objectives."""

        if self.config["assimilation_mode"] == AssimilationMode.LEGACY_CONTINUOUS_V01.value:
            return assimilation_loss(strong_logits, pseudo, learnability, strong_valid_mask)
        return admission_assimilation_loss(strong_logits, pseudo, admission, strong_valid_mask)

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
        """Protocol hook retaining the exact frozen V0.1/R1 anchor update."""

        self._cache_unlabeled_anchor_update(features, pseudo, learnability, uniform_compatibility, site_step)

    def training_step(
        self,
        labeled_batch: LabeledBatch,
        unlabeled_batch: UnlabeledBatch | None,
        global_step: int,
        site_step: int,
    ) -> MethodStepOutput:
        if unlabeled_batch is None:
            raise ValueError("LCR-Seg V0.2a requires an unlabeled batch")
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
            anchor_loss = relation_supervision_loss(labeled_relation.logits, labeled_batch.label, labeled_batch.valid_mask)
        losses = self._all_loss_keys(
            labeled_output.logits,
            self._supervised_losses(labeled_output, labeled_batch, relation_anchor_loss=anchor_loss),
        )
        pseudo: PseudoLabelOutput | None = None
        learnability: LearnabilityOutput | None = None
        diagnostic_compatibility = zero_compatibility(
            weak_relation.probabilities if weak_relation is not None else weak_output.relation_features[:, :1]
        )
        old_relation: RelationOutput | None = None
        teacher_validity: TeacherValidityOutput | None = None
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
            learnability = compute_learnability(
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
            admission = self._compute_admission(pseudo, learnability, unlabeled_batch.strong_valid_mask, site_step=site_step)
            assimilation = self._assimilation_objective(
                strong_logits=strong_output.logits,
                pseudo=pseudo,
                learnability=learnability,
                admission=admission,
                weak_relation=weak_relation,
                strong_relation=strong_relation,
                strong_valid_mask=unlabeled_batch.strong_valid_mask,
            )
            uniform_compatibility = _uniform_compatibility(diagnostic_compatibility)
            if self.old_model is not None and self.old_anchor_bank is not None:
                self.old_model.eval()
                with torch.no_grad():
                    old_output = self.old_model(unlabeled_batch.weak_image)
                    old_relation = self._relation(old_output.relation_features, self.old_anchor_bank)
                diagnostic_compatibility = compute_compatibility(
                    weak_relation,
                    old_relation,
                    old_margin_center=float(self.config["old_margin_center"]),
                    old_margin_temperature=float(self.config["old_margin_temperature"]),
                    js_temperature=float(self.config["js_temperature"]),
                    spatial_floor=float(self.config["spatial_floor"]),
                )
                uniform_compatibility = _uniform_compatibility(diagnostic_compatibility)
                relation_valid = strict_relation_valid_mask(
                    unlabeled_batch.strong_valid_mask, diagnostic_compatibility.score.shape[-2:]
                )
                if self.config["consolidation_mode"] == ConsolidationMode.UNIFORM_RELATION.value:
                    relation_loss = relation_consolidation_loss(
                        strong_relation,
                        old_relation,
                        uniform_compatibility,
                        unlabeled_batch.strong_valid_mask,
                        distill_temperature=float(self.config["distill_temperature"]),
                    )
                    consolidation = _uniform_consolidation(
                        diagnostic_compatibility,
                        relation_valid,
                        num_classes=self.num_classes,
                        old_predicted_class=old_relation.predicted_class,
                    )
                elif self.config["consolidation_mode"] == "none":
                    # V0.3 P0 keeps the R1 representation/anchor path but has
                    # no historical relation objective.  The differentiable
                    # zero was initialized above and therefore contributes an
                    # exact zero gradient.
                    consolidation = _uniform_consolidation(
                        diagnostic_compatibility,
                        relation_valid,
                        num_classes=self.num_classes,
                        old_predicted_class=old_relation.predicted_class,
                    )
                else:
                    teacher_validity = compute_teacher_validity(
                        old_output.logits,
                        old_relation,
                        margin_temperature=float(self.config["teacher_validity_margin_temperature"]),
                        spatial_floor=float(self.config["teacher_validity_spatial_floor"]),
                    )
                    calibrated, available = self.teacher_validity_calibrator.calibrate(
                        teacher_validity.raw_score, teacher_validity.old_predicted_class
                    )
                    if not available:
                        raise RuntimeError("incremental C1 training started without a frozen site-start calibrator")
                    consolidation = rejection_only_weights(
                        calibrated,
                        teacher_validity.old_predicted_class,
                        relation_valid,
                        num_classes=self.num_classes,
                        calibrator_available=True,
                        probability_threshold=float(self.config["rejection_threshold"]),
                        max_reject_fraction_per_class=float(self.config["rejection_cap"]),
                        rejected_weight_floor=float(self.config["rejection_floor"]),
                    )
                    relation_loss = rejection_only_relation_loss(
                        strong_relation,
                        old_relation,
                        consolidation,
                        unlabeled_batch.strong_valid_mask,
                        distill_temperature=float(self.config["distill_temperature"]),
                    )
            if consolidation is None:
                relation_valid = strict_relation_valid_mask(
                    unlabeled_batch.strong_valid_mask, diagnostic_compatibility.score.shape[-2:]
                )
                consolidation = _uniform_consolidation(
                    diagnostic_compatibility,
                    relation_valid,
                    num_classes=self.num_classes,
                    old_predicted_class=None,
                )
            # Keep anchor memory identical across the 2x2 factors. Formal R0
            # therefore preserves the legacy uniform-relation artifact path.
            self._cache_protocol_unlabeled_anchor_update(
                features=weak_output.relation_features,
                pseudo=pseudo,
                learnability=learnability,
                weak_relation=weak_relation,
                uniform_compatibility=uniform_compatibility,
                site_step=site_step,
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
        if admission is None:
            empty = tuple(0 for _ in range(self.num_classes))
            admission = ProgressiveAdmissionOutput(
                mask=torch.zeros_like(diagnostic_compatibility.score, dtype=torch.bool),
                candidate_mask=torch.zeros_like(diagnostic_compatibility.score, dtype=torch.bool),
                site_progress=0.0,
                target_fraction=0.0,
                candidate_counts=empty,
                selected_counts=empty,
                learnability_thresholds=tuple(float("nan") for _ in empty),
            )
        if consolidation is None:
            consolidation = _uniform_consolidation(
                diagnostic_compatibility,
                torch.zeros_like(diagnostic_compatibility.score, dtype=torch.bool),
                num_classes=self.num_classes,
                old_predicted_class=None,
            )
        self._record_branch_rows(site_step=site_step, admission=admission, consolidation=consolidation)
        selected_total = int(sum(admission.selected_counts))
        candidate_total = int(sum(admission.candidate_counts))
        pseudo_valid_total = int(pseudo.valid.sum().detach()) if pseudo is not None else 0
        relation_total = int(sum(consolidation.candidate_counts))
        rejected_total = int(sum(consolidation.rejected_counts))
        self.v02a_statistics["steps"] += 1
        self.v02a_statistics["pseudo_valid_count"] += pseudo_valid_total
        self.v02a_statistics["assim_candidate_count"] += candidate_total
        self.v02a_statistics["admitted_count"] += selected_total
        self.v02a_statistics["relation_valid_count"] += relation_total
        self.v02a_statistics["rejected_count"] += rejected_total
        weights = consolidation.weights
        valid = consolidation.relation_valid_mask
        calibrated = consolidation.calibrated_compatibility
        if pseudo is not None and learnability is not None:
            if self.config["assimilation_mode"] == AssimilationMode.LEGACY_CONTINUOUS_V01.value:
                pseudo_valid_full = F.interpolate(
                    pseudo.valid.float(), size=strong_output.logits.shape[-2:], mode="nearest"
                ).bool()
                learnability_full = F.interpolate(
                    learnability.score.detach(),
                    size=strong_output.logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                assimilation_denominator = float(
                    (
                        learnability_full
                        * (pseudo_valid_full & unlabeled_batch.strong_valid_mask.bool()).float()
                    ).sum().detach()
                )
            else:
                selected_full = F.interpolate(
                    admission.mask.detach().float(), size=strong_output.logits.shape[-2:], mode="nearest"
                ).bool()
                assimilation_denominator = float(
                    (selected_full & unlabeled_batch.strong_valid_mask.bool()).sum().detach()
                )
        else:
            assimilation_denominator = 0.0
        if old_relation is None:
            relation_denominator = 0.0
        elif self.config["consolidation_mode"] == ConsolidationMode.UNIFORM_RELATION.value:
            relation_denominator = float(
                downsample_valid_mask(
                    unlabeled_batch.strong_valid_mask, old_relation.probabilities.shape[-2:]
                ).sum().detach()
            )
        elif self.config["consolidation_mode"] == "none":
            relation_denominator = 0.0
        else:
            relation_denominator = float((weights.detach() * valid.float()).sum().detach())
        scalars: dict[str, Any] = {
            "lambda_assim_effective": lambda_assim,
            "lambda_relation_effective": lambda_relation,
            "bootstrap_complete": float(not self._bootstrap_active()),
            "relation_anchor_ready": float(anchor_ready),
            "site_progress": admission.site_progress,
            "target_admission_fraction": admission.target_fraction,
            "pseudo_valid_count": pseudo_valid_total,
            "assim_candidate_count": candidate_total,
            "assim_selected_count": selected_total,
            "assim_selected_fraction": selected_total / candidate_total if candidate_total else 0.0,
            "assim_candidate_counts_by_class": list(admission.candidate_counts),
            "assim_selected_counts_by_class": list(admission.selected_counts),
            "assim_selected_fraction_by_class": list(admission.selected_fraction_by_class),
            "relation_valid_count": relation_total,
            "teacher_rejected_count": rejected_total,
            "teacher_rejected_fraction": rejected_total / relation_total if relation_total else 0.0,
            "relation_valid_counts_by_class": list(consolidation.candidate_counts),
            "teacher_rejected_counts_by_class": list(consolidation.rejected_counts),
            "teacher_rejected_fraction_by_class": list(consolidation.rejected_fraction_by_class),
            "relation_weight_mean": float(weights[valid].float().mean()) if bool(valid.any()) else 0.0,
            "relation_effective_sample_size": _effective_sample_size(weights, valid),
            "raw_teacher_validity_mean": float(teacher_validity.raw_score.mean()) if teacher_validity is not None else 0.0,
            "calibrated_teacher_validity_mean": float(calibrated.mean()) if teacher_validity is not None else 0.0,
            "teacher_validity_calibrator_status": self.teacher_validity_calibrator.status,
            "pseudo_valid_ratio": float(pseudo_valid_total / max(1, diagnostic_compatibility.score.numel())),
            "compatibility_mean": float(diagnostic_compatibility.score.detach().mean()),
            "relation_js_mean": float(diagnostic_compatibility.js_divergence.detach().mean()),
            "current_old_agreement_mean": float(diagnostic_compatibility.agreement.detach().mean()),
            "hidden_gt_training_usage": 0,
            "current_old_js_gate_usage": 0,
            "old_model_gradient_detected": 0,
            "historical_anchor_changed": 0,
            "assimilation_denominator": assimilation_denominator,
            "relation_denominator": relation_denominator,
            "relation_loss_numerator": 0.0 if self.config["consolidation_mode"] == "none" else float(relation_loss.detach()),
            "relation_backward_norm_declared": 0.0 if self.config["consolidation_mode"] == "none" else None,
        }
        maps: dict[str, torch.Tensor] = {
            "raw_learnability": learnability.score.detach() if learnability is not None else torch.zeros_like(diagnostic_compatibility.score),
            "admission_mask": admission.mask.detach(),
            "compatibility": diagnostic_compatibility.score.detach(),
            "current_old_js": diagnostic_compatibility.js_divergence.detach(),
            "raw_teacher_validity": teacher_validity.raw_score.detach() if teacher_validity is not None else torch.zeros_like(diagnostic_compatibility.score),
            "calibrated_teacher_validity": calibrated.detach(),
            "rejection_mask": consolidation.rejection_mask.detach(),
            "consolidation_weights": weights.detach(),
            "relation_valid_mask": valid.detach(),
        }
        if pseudo is not None:
            maps.update(
                {
                    "pseudo_labels": pseudo.labels.detach(),
                    "pseudo_valid": pseudo.valid.detach(),
                    "pseudo_source": pseudo.source.detach(),
                    "learnability": learnability.score.detach() if learnability is not None else torch.zeros_like(diagnostic_compatibility.score),
                }
            )
        if weak_relation is not None:
            maps["current_relation_probability"] = weak_relation.probabilities.detach()
        if old_relation is not None:
            maps["old_relation_probability"] = old_relation.probabilities.detach()
        return MethodStepOutput(total_loss=total, losses=losses, scalars=scalars, maps=maps)

    def end_site(self, site_id: str) -> dict[str, Any]:
        result = super().end_site(site_id)
        result.update(
            {
                "protocol_semantics": self.protocol_semantics(),
                "teacher_validity_calibrator_status": self.teacher_validity_calibrator.status,
                "v02a_statistics": dict(self.v02a_statistics),
            }
        )
        return result

    def write_site_artifacts(self, *, run_dir: Path, site_id: str, site_index: int) -> None:
        path = Path(run_dir) / "branch_coverage.csv"
        existing = list(csv.DictReader(path.open())) if path.is_file() else []
        fields = [
            "site",
            "site_index",
            "epoch",
            "step",
            "predicted_class",
            "assimilation_mode",
            "consolidation_mode",
            "target_fraction",
            "realized_fraction",
            "valid_count",
            "admitted_count",
            "deferred_count",
            "learnability_threshold",
            "relation_valid_count",
            "rejected_count",
            "rejected_fraction",
        ]
        combined = [*existing, *self.branch_rows]
        write_csv(path, combined, fieldnames=fields)
        if self.config.get("protocol_id") in {"lcrseg_v0_3", "lcrseg_v0_4a"}:
            write_csv(Path(run_dir) / "admission_coverage.csv", combined, fieldnames=fields)
        self._write_calibrator_artifacts(Path(run_dir))
        write_json(
            Path(run_dir) / f"branch_statistics_site{site_index}_{site_id}.json",
            {
                "site": site_id,
                "site_index": site_index,
                "protocol_semantics": self.protocol_semantics(),
                "statistics": self.v02a_statistics,
            },
        )

    def method_state_dict(self) -> dict[str, Any]:
        state = super().method_state_dict()
        statistics = dict(state["method_statistics"])
        statistics.update(
            {
                "protocol_semantics": self.protocol_semantics(),
                "teacher_validity_calibrator_state": self.teacher_validity_calibrator.state_dict(),
                "calibration_rows": [dict(row) for row in self.calibration_rows],
                "branch_rows": [dict(row) for row in self.branch_rows],
                "v02a_statistics": dict(self.v02a_statistics),
                "epoch_context": self._epoch_context,
            }
        )
        state["method_statistics"] = statistics
        return state

    def load_method_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_method_state_dict(state)
        statistics = dict(state.get("method_statistics") or {})
        semantics = dict(statistics.get("protocol_semantics") or {})
        if semantics and semantics != self.protocol_semantics():
            raise ValueError("checkpoint V0.2a protocol semantics differ from resolved config")
        self.teacher_validity_calibrator = self._new_teacher_validity_calibrator()
        self.teacher_validity_calibrator.load_state_dict(
            dict(statistics.get("teacher_validity_calibrator_state") or {})
        )
        self.calibration_rows = [dict(row) for row in statistics.get("calibration_rows", [])]
        self.branch_rows = [dict(row) for row in statistics.get("branch_rows", [])]
        self.v02a_statistics = dict(statistics.get("v02a_statistics") or self._new_statistics())
        self._epoch_context = int(statistics.get("epoch_context", 0))
