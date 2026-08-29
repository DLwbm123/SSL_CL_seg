"""LCR-Seg V0.1: V0 relation field through V3 continuous routing.

This module is intentionally the only method-specific training path.  The
optimizer, batches, checkpointing, evaluation, and site lifecycle remain in
the shared engine.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from ..contracts import LabeledBatch, MethodStepOutput, UnlabeledBatch, differentiable_zero
from ..data.transforms import downsample_valid_mask
from .base import ContinualSegMethod, clone_state_dict, relation_supervision_loss
from .components.anchor_bank import AnchorBank
from .components.compatibility import CompatibilityOutput, compute_compatibility, zero_compatibility
from .components.learnability import LearnabilityOutput, compute_learnability
from .components.pseudo_label import PseudoLabelOutput, build_pseudo_labels
from .components.relation_field import RelationOutput, relation_field
from .components.routing import assimilation_loss, relation_consolidation_loss


LCR_DEFAULTS: dict[str, Any] = {
    "relation_temperature": 0.1,
    "distill_temperature": 0.5,
    "anchor_k": 1,
    "anchor_momentum": 0.99,
    "anchor_max_pixels_per_class": 2048,
    "anchor_bootstrap_steps": 500,
    "anchor_min_support_pixels": 64,
    "background_boundary_exclusion": 3,
    "memory_eta": 0.25,
    "tau_cls": 0.95,
    "tau_anchor": 0.80,
    "delta_anchor": 0.15,
    "tau_spatial": 0.60,
    "temperature_cls": 0.05,
    "temperature_anchor": 0.05,
    "rank_start": 0.80,
    "rank_end": 0.20,
    "rank_temperature": 0.10,
    "relation_margin_center": 0.10,
    "relation_margin_temperature": 0.05,
    "spatial_floor": 0.25,
    "min_rank_pixels": 128,
    "old_margin_center": 0.10,
    "old_margin_temperature": 0.05,
    "js_temperature": 0.20,
    # Experimental ablation switches.  The frozen V0.1 default remains the
    # full continuous L_i/C_i routing defined by the method specification.
    "use_learnability": True,
    "use_compatibility": True,
    "require_all_anchors_at_end": True,
}


def _anchor_checksum(bank: AnchorBank | None) -> str:
    if bank is None:
        return ""
    digest = hashlib.sha256()
    for name, value in sorted(bank.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _quantiles(value: torch.Tensor, prefix: str) -> dict[str, float]:
    flat = value.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_p10": 0.0, f"{prefix}_p50": 0.0, f"{prefix}_p90": 0.0}
    q = torch.quantile(flat, torch.tensor((0.1, 0.5, 0.9), device=flat.device))
    return {
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_p10": float(q[0]),
        f"{prefix}_p50": float(q[1]),
        f"{prefix}_p90": float(q[2]),
    }


@torch.no_grad()
def _uniform_learnability(pseudo: PseudoLabelOutput) -> LearnabilityOutput:
    """Return detached unit weights for the documented no-L_i ablation.

    The pseudo-label validity mask remains authoritative: this ablation only
    removes continuous learnability routing, not the deferred-label safety
    rule or strong-view cutout masking.
    """

    score = pseudo.valid.float().detach()
    return LearnabilityOutput(
        score=score,
        robust_progress_index=score,
        percentile_rank=score,
        progress_weight=torch.ones_like(score),
        relation_weight=torch.ones_like(score),
        spatial_weight=pseudo.spatial_weight.detach(),
        source_weight=pseudo.source_weight.detach(),
    )


@torch.no_grad()
def _uniform_compatibility(reference: CompatibilityOutput) -> CompatibilityOutput:
    """Return detached unit weights for the documented uniform-KD ablation."""

    score = torch.ones_like(reference.score).detach()
    return CompatibilityOutput(
        score=score,
        js_divergence=reference.js_divergence.detach(),
        old_margin_weight=torch.ones_like(score),
        agreement=torch.ones_like(score),
        spatial_weight=torch.ones_like(score),
    )


class LCRSegV01Method(ContinualSegMethod):
    """Single-anchor LCR-Seg implementation matching the frozen V0.1 spec."""

    method_name = "lcrseg_v0_1"
    method_version = "0.1"

    def __init__(self, model, *, config: Mapping[str, Any] | None = None) -> None:
        merged = dict(LCR_DEFAULTS)
        if config:
            merged.update(dict(config))
        super().__init__(model, config=merged)
        self.current_anchor_bank = self._new_anchor_bank()
        self.old_anchor_bank: AnchorBank | None = None
        self._old_anchor_checksum = ""
        self._site_start_anchors: torch.Tensor | None = None
        self.bootstrap_state: dict[str, Any] = {
            "complete": False,
            "completed_at_site_step": -1,
            "configured_steps": int(self.config["anchor_bootstrap_steps"]),
        }
        self.method_statistics: dict[str, Any] = {"anchor_update_events": [], "warnings": []}
        self._pending_anchor_updates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int]] = []

    def _new_anchor_bank(self) -> AnchorBank:
        return AnchorBank(
            self.model.num_classes,
            self.model.relation_dim,
            k=int(self.config["anchor_k"]),
            momentum=float(self.config["anchor_momentum"]),
            max_pixels_per_class=int(self.config["anchor_max_pixels_per_class"]),
            min_support_pixels=int(self.config["anchor_min_support_pixels"]),
            background_boundary_exclusion=int(self.config["background_boundary_exclusion"]),
        )

    def _reset_first_site_anchors(self) -> None:
        device = next(self.model.parameters()).device
        self.current_anchor_bank = self._new_anchor_bank().to(device)
        self.old_anchor_bank = None
        self._old_anchor_checksum = ""
        self._site_start_anchors = self.current_anchor_bank.anchors.detach().clone()
        self.bootstrap_state = {
            "complete": False,
            "completed_at_site_step": -1,
            "configured_steps": int(self.config["anchor_bootstrap_steps"]),
        }

    def begin_site(self, site_id: str, previous_checkpoint, total_steps: int) -> None:  # type: ignore[override]
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        self.site_id = str(site_id)
        self.total_steps = int(total_steps)
        self._pending_anchor_updates = []
        self.method_statistics = {"anchor_update_events": [], "warnings": []}
        self.old_model = None
        self._old_model_checksum = None
        if previous_checkpoint is None:
            self._reset_first_site_anchors()
            return

        payload = self._load_previous_model(previous_checkpoint)
        previous_anchor_state = payload.get("current_anchor_state") or {}
        if not previous_anchor_state:
            raise RuntimeError("incremental LCR-Seg site requires historical current_anchor_state")
        device = next(self.model.parameters()).device
        current = self._new_anchor_bank().to(device)
        current.load_state_dict(previous_anchor_state, strict=True)
        if not current.all_classes_valid:
            raise RuntimeError("previous checkpoint has invalid semantic anchors; cannot begin incremental site")
        self.current_anchor_bank = current.clone().to(device)
        self.old_anchor_bank = current.clone().to(device)
        if self.current_anchor_bank.anchors.data_ptr() == self.old_anchor_bank.anchors.data_ptr():
            raise AssertionError("current and historical anchors share storage")
        self._make_old_model()
        self._old_anchor_checksum = _anchor_checksum(self.old_anchor_bank)
        self._site_start_anchors = self.current_anchor_bank.anchors.detach().clone()
        self.bootstrap_state = {
            "complete": True,
            "completed_at_site_step": -1,
            "configured_steps": int(self.config["anchor_bootstrap_steps"]),
        }

    def _relation(self, features: torch.Tensor, bank: AnchorBank) -> RelationOutput:
        return relation_field(features, bank, temperature=float(self.config["relation_temperature"]))

    def _cache_labeled_anchor_update(self, features: torch.Tensor, batch: LabeledBatch, site_step: int) -> None:
        labels = F.interpolate(batch.label.unsqueeze(1).float(), size=features.shape[-2:], mode="nearest")[:, 0].long()
        weights = downsample_valid_mask(batch.valid_mask, features.shape[-2:]).float()
        self._pending_anchor_updates.append((features.detach(), labels.detach(), weights.detach(), "labeled", int(site_step)))

    def _cache_unlabeled_anchor_update(
        self,
        features: torch.Tensor,
        pseudo: PseudoLabelOutput,
        learnability: LearnabilityOutput,
        compatibility: CompatibilityOutput,
        site_step: int,
    ) -> None:
        memory_weight = learnability.score.detach() * (
            float(self.config["memory_eta"]) + (1.0 - float(self.config["memory_eta"])) * compatibility.score.detach()
        )
        self._pending_anchor_updates.append(
            (features.detach(), pseudo.labels.detach(), memory_weight.detach(), "unlabeled", int(site_step))
        )

    def _bootstrap_active(self) -> bool:
        return not bool(self.bootstrap_state.get("complete", False))

    def training_step(
        self,
        labeled_batch: LabeledBatch,
        unlabeled_batch: UnlabeledBatch | None,
        global_step: int,
        site_step: int,
    ) -> MethodStepOutput:
        if unlabeled_batch is None:
            raise ValueError("LCR-Seg requires an unlabeled batch")
        # Labeled and strong paths retain gradients; the weak current path is a
        # stop-gradient teacher by definition.
        labeled_output = self.model(labeled_batch.image)
        with torch.no_grad():
            weak_output = self.model(unlabeled_batch.weak_image)
        strong_output = self.model(unlabeled_batch.strong_image)
        anchor_ready = self.current_anchor_bank.all_classes_valid
        labeled_relation: RelationOutput | None = None
        weak_relation: RelationOutput | None = None
        strong_relation: RelationOutput | None = None
        anchor_loss = differentiable_zero(labeled_output.logits)
        if anchor_ready:
            labeled_relation = self._relation(labeled_output.relation_features, self.current_anchor_bank)
            weak_relation = self._relation(weak_output.relation_features, self.current_anchor_bank)
            strong_relation = self._relation(strong_output.relation_features, self.current_anchor_bank)
            anchor_loss = relation_supervision_loss(labeled_relation.logits, labeled_batch.label, labeled_batch.valid_mask)
        losses = self._all_loss_keys(labeled_output.logits, self._supervised_losses(labeled_output, labeled_batch, relation_anchor_loss=anchor_loss))
        pseudo: PseudoLabelOutput | None = None
        learnability: LearnabilityOutput | None = None
        compatibility: CompatibilityOutput = zero_compatibility(
            weak_relation.probabilities if weak_relation is not None else weak_output.relation_features[:, :1]
        )
        old_relation: RelationOutput | None = None
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
            if not bool(self.config["use_learnability"]):
                learnability = _uniform_learnability(pseudo)
            assimilation = assimilation_loss(strong_output.logits, pseudo, learnability, unlabeled_batch.strong_valid_mask)
            if self.old_model is not None and self.old_anchor_bank is not None:
                self.old_model.eval()
                with torch.no_grad():
                    old_output = self.old_model(unlabeled_batch.weak_image)
                    old_relation = self._relation(old_output.relation_features, self.old_anchor_bank)
                compatibility = compute_compatibility(
                    weak_relation,
                    old_relation,
                    old_margin_center=float(self.config["old_margin_center"]),
                    old_margin_temperature=float(self.config["old_margin_temperature"]),
                    js_temperature=float(self.config["js_temperature"]),
                    spatial_floor=float(self.config["spatial_floor"]),
                )
                if not bool(self.config["use_compatibility"]):
                    compatibility = _uniform_compatibility(compatibility)
                relation_loss = relation_consolidation_loss(
                    strong_relation,
                    old_relation,
                    compatibility,
                    unlabeled_batch.strong_valid_mask,
                    distill_temperature=float(self.config["distill_temperature"]),
                )
            self._cache_unlabeled_anchor_update(weak_output.relation_features, pseudo, learnability, compatibility, site_step)
        # V0.1 bootstrap is a genuine supervised/SSL warm-up: no semantic
        # direction is written before ``anchor_bootstrap_steps`` has elapsed.
        # Once warm-up is complete, the first current-model labeled batch
        # initializes anchors without retaining any patient feature maps.
        if not self._bootstrap_active() or site_step >= int(self.config["anchor_bootstrap_steps"]):
            self._cache_labeled_anchor_update(labeled_output.relation_features, labeled_batch, site_step)
        losses["loss_assim"] = assimilation
        losses["loss_relation"] = relation_loss
        lambda_assim = 0.0 if self._bootstrap_active() else float(self.config["lambda_assim"]) * self._assimilation_ramp(
            site_step, bootstrap_complete_at=int(self.bootstrap_state.get("completed_at_site_step", -1))
        )
        lambda_relation = 0.0 if self.old_model is None else float(self.config["lambda_relation"]) * self._relation_ramp(site_step)
        total = losses["loss_sup"] + lambda_assim * assimilation + lambda_relation * relation_loss
        assimilation_denominator = 0.0
        relation_denominator = 0.0
        pseudo_valid_count = 0
        if pseudo is not None and learnability is not None:
            target_size = strong_output.logits.shape[-2:]
            pseudo_valid_full = F.interpolate(pseudo.valid.float(), size=target_size, mode="nearest").bool()
            learnability_full = F.interpolate(
                learnability.score.detach(), size=target_size, mode="bilinear", align_corners=False
            )
            assimilation_weights = learnability_full * (pseudo_valid_full & unlabeled_batch.strong_valid_mask.bool()).float()
            assimilation_denominator = float(assimilation_weights.sum().detach())
            pseudo_valid_count = int(pseudo.valid.sum().detach())
        if old_relation is not None:
            relation_valid = downsample_valid_mask(
                unlabeled_batch.strong_valid_mask, old_relation.probabilities.shape[-2:]
            )
            relation_denominator = float((compatibility.score.detach() * relation_valid.float()).sum().detach())
        scalars: dict[str, float] = {
            "lambda_assim_effective": lambda_assim,
            "lambda_relation_effective": lambda_relation,
            "bootstrap_complete": float(not self._bootstrap_active()),
            "relation_anchor_ready": float(anchor_ready),
            "pseudo_valid_ratio": 0.0,
            "pseudo_classifier_ratio": 0.0,
            "pseudo_anchor_ratio": 0.0,
            "pseudo_deferred_ratio": 1.0,
            "learnability_mean": 0.0,
            "compatibility_mean": float(compatibility.score.mean().detach()),
            "relation_js_mean": float(compatibility.js_divergence.mean().detach()),
            "pseudo_valid_count": float(pseudo_valid_count),
            "assimilation_denominator": assimilation_denominator,
            "relation_denominator": relation_denominator,
        }
        maps: dict[str, torch.Tensor] = {}
        if pseudo is not None and learnability is not None:
            source = pseudo.source
            scalars.update(
                {
                    "pseudo_valid_ratio": float(pseudo.valid.float().mean().detach()),
                    "pseudo_classifier_ratio": float(source.eq(1).float().mean().detach()),
                    "pseudo_anchor_ratio": float(source.eq(2).float().mean().detach()),
                    "pseudo_deferred_ratio": float(source.eq(0).float().mean().detach()),
                }
            )
            scalars.update(_quantiles(learnability.score[pseudo.valid], "learnability") if bool(pseudo.valid.any()) else _quantiles(torch.empty(0, device=strong_output.logits.device), "learnability"))
            maps.update(
                {
                    "pseudo_labels": pseudo.labels.detach(),
                    "pseudo_valid": pseudo.valid.detach(),
                    "pseudo_source": pseudo.source.detach(),
                    "learnability": learnability.score.detach(),
                    "compatibility": compatibility.score.detach(),
                }
            )
        else:
            scalars.update(_quantiles(torch.empty(0, device=strong_output.logits.device), "learnability"))
        scalars.update(_quantiles(compatibility.score, "compatibility"))
        if weak_relation is not None:
            maps["current_relation_probability"] = weak_relation.probabilities.detach()
        if old_relation is not None:
            maps["old_relation_probability"] = old_relation.probabilities.detach()
        return MethodStepOutput(total_loss=total, losses=losses, scalars=scalars, maps=maps or None)

    @torch.no_grad()
    def after_optimizer_step(self) -> None:
        events: list[dict[str, Any]] = []
        for features, labels, weights, source, step in self._pending_anchor_updates:
            update = self.current_anchor_bank.update(features, labels, weights, source=source, step=step)
            events.append(
                {"source": source, "step": step, "updated_pixels": update.updated_pixels, "skipped_classes": update.skipped_classes}
            )
        self._pending_anchor_updates = []
        self.method_statistics["anchor_update_events"].extend(events)
        if self._bootstrap_active() and events:
            latest_step = max(event["step"] for event in events)
            if latest_step + 1 >= int(self.config["anchor_bootstrap_steps"]) and self.current_anchor_bank.all_classes_valid:
                self.bootstrap_state["complete"] = True
                self.bootstrap_state["completed_at_site_step"] = int(latest_step)
            elif latest_step + 1 >= int(self.config["anchor_bootstrap_steps"]):
                warning = "bootstrap elapsed but one or more class anchors are still invalid"
                if warning not in self.method_statistics["warnings"]:
                    self.method_statistics["warnings"].append(warning)
        self.assert_old_state_unchanged()
        if self.old_anchor_bank is not None and self._old_anchor_checksum != _anchor_checksum(self.old_anchor_bank):
            raise AssertionError("historical anchor bank changed during current-site training")

    def end_site(self, site_id: str) -> dict[str, Any]:
        result = super().end_site(site_id)
        if bool(self.config["require_all_anchors_at_end"]) and not self.current_anchor_bank.all_classes_valid:
            missing = torch.nonzero(~self.current_anchor_bank.valid_class_mask, as_tuple=False).flatten().tolist()
            raise RuntimeError(f"LCR-Seg anchor bootstrap incomplete; invalid classes: {missing}")
        result.update(
            {
                "bootstrap_state": dict(self.bootstrap_state),
                "anchor_diagnostics": self.current_anchor_bank.diagnostics(start_anchors=self._site_start_anchors),
                "warnings": list(self.method_statistics.get("warnings", [])),
            }
        )
        return result

    def method_state_dict(self) -> dict[str, Any]:
        base = super().method_state_dict()
        statistics = dict(base["method_statistics"])
        statistics.update(
            {
                "anchor_diagnostics": self.current_anchor_bank.diagnostics(start_anchors=self._site_start_anchors),
                "anchor_update_events": list(self.method_statistics.get("anchor_update_events", [])),
                "warnings": list(self.method_statistics.get("warnings", [])),
                "old_anchor_checksum": self._old_anchor_checksum,
            }
        )
        return {
            "current_anchor_state": self.current_anchor_bank.exported_state(),
            "historical_anchor_state": self.old_anchor_bank.exported_state() if self.old_anchor_bank is not None else {},
            "bootstrap_state": dict(self.bootstrap_state),
            "method_statistics": statistics,
        }

    def load_method_state_dict(self, state: Mapping[str, Any]) -> None:
        current_state = state.get("current_anchor_state") or {}
        if current_state:
            current = self._new_anchor_bank().to(next(self.model.parameters()).device)
            current.load_state_dict(current_state, strict=True)
            self.current_anchor_bank = current
        old_state = state.get("historical_anchor_state") or {}
        if old_state:
            old = self._new_anchor_bank().to(next(self.model.parameters()).device)
            old.load_state_dict(old_state, strict=True)
            self.old_anchor_bank = old
            self._old_anchor_checksum = _anchor_checksum(old)
        else:
            self.old_anchor_bank = None
            self._old_anchor_checksum = ""
        self.bootstrap_state = dict(state.get("bootstrap_state") or self.bootstrap_state)
        statistics = dict(state.get("method_statistics") or {})
        self.method_statistics = {
            "anchor_update_events": list(statistics.get("anchor_update_events", [])),
            "warnings": list(statistics.get("warnings", [])),
        }
        self._restore_old_model(statistics.get("old_model_state") or None)
        self._site_start_anchors = self.current_anchor_bank.anchors.detach().clone()
