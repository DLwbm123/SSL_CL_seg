"""SR-GAS V0.1a on the frozen V0.2a-R0 continual segmentation path."""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn

from ..common import write_csv, write_json
from ..contracts import MethodStepOutput, differentiable_zero
from ..models import CosineSegmentationHead
from ..regularization import (
    SpatialRelationShuffler,
    jascl_inverse_minmax_scale,
    relation_to_classifier_loss,
    sample_perturbed_weight,
    unit_mean_source_normalize,
)
from .base import clone_state_dict, merged_method_config
from .components.learnability import LearnabilityOutput
from .components.pseudo_label import PseudoLabelOutput
from .components.routing import assimilation_loss
from .lcrseg_v0_2a import LCR_V02A_DEFAULTS, LCRSegV02AMethod


SRGAS_DEFAULTS: dict[str, Any] = {
    **LCR_V02A_DEFAULTS,
    "protocol_id": "srgas_v0_1",
    "variant_id": "A1",
    "srgas_variant": "A1",
    "relation_conditioning": "none",
    "cosine_temperature": 10.0,
    "cosine_eps": 1.0e-8,
    "gas_epsilon": 1.0e-8,
    "noise_variance": 0.1,
    "noise_distribution": "standard_normal",
    "noise_scope": "classifier_weight_only",
    "scale_normalization": "jascl_inverse_minmax",
    "same_step_sensitivity": True,
    "noise_seed": 0,
    "r2c_temperature": 1.0,
    "r2c_resize_mode": "bilinear",
    "r2c_align_corners": False,
    "r2c_reduction": "valid_pixel_mean",
    "r2c_source_weight": 0.5,
    "supervised_source_weight": 0.5,
    "r2c_added_to_training_objective": False,
    "channel_mapping": "none",
    "architecture_change": False,
    "shuffle_r2c_target": False,
}

_VARIANTS = {"A1", "A2", "A3", "A4", "A5", "A5_SHUFFLE", "A6"}


def resolve_srgas_method_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved = dict(SRGAS_DEFAULTS)
    resolved.update(dict(config or {}))
    variant = str(resolved.get("srgas_variant", resolved.get("variant_id", "A1"))).upper().replace("-", "_")
    if variant not in _VARIANTS:
        raise ValueError(f"unregistered SR-GAS variant: {variant}")
    resolved["srgas_variant"] = variant
    resolved["variant_id"] = variant
    expected_protocol = "srgas_v0_1a" if variant in {"A5", "A5_SHUFFLE"} else "srgas_v0_1"
    if str(resolved["protocol_id"]) != expected_protocol:
        raise ValueError(f"{variant} requires protocol_id={expected_protocol}")
    expected_relation = "relation_to_classifier_proxy" if variant in {"A5", "A5_SHUFFLE"} else "none"
    if str(resolved["relation_conditioning"]) != expected_relation:
        raise ValueError(f"{variant} relation conditioning differs from the registered protocol")
    exact = {
        "cosine_temperature": 10.0,
        "cosine_eps": 1.0e-8,
        "gas_epsilon": 1.0e-8,
        "noise_variance": 0.1,
        "r2c_temperature": 1.0,
        "r2c_resize_mode": "bilinear",
        "r2c_align_corners": False,
        "r2c_reduction": "valid_pixel_mean",
        "r2c_source_weight": 0.5,
        "supervised_source_weight": 0.5,
        "r2c_added_to_training_objective": False,
        "channel_mapping": "none",
        "architecture_change": False,
    }
    for key, expected in exact.items():
        if resolved[key] != expected:
            raise ValueError(f"{key} differs from the frozen SR-GAS V0.1a protocol")
    if str(resolved["assimilation_mode"]) != "legacy_continuous_v01" or str(resolved["consolidation_mode"]) != "uniform_relation":
        raise ValueError("SR-GAS must retain the exact V0.2a R0 learner")
    if bool(resolved["shuffle_r2c_target"]) != (variant == "A5_SHUFFLE"):
        raise ValueError("R2C shuffle flag conflicts with the registered variant")
    return merged_method_config(resolved)


def _dummy_pseudo_and_learnability(maps: Mapping[str, torch.Tensor]) -> tuple[PseudoLabelOutput, LearnabilityOutput]:
    labels = maps["pseudo_labels"].detach()
    valid = maps["pseudo_valid"].detach().bool()
    source = maps["pseudo_source"].detach().long()
    score = maps["learnability"].detach()
    zeros = torch.zeros_like(score)
    pseudo = PseudoLabelOutput(
        labels=labels,
        valid=valid,
        source=source,
        source_weight=zeros,
        spatial_weight=zeros,
        spatial_agreement=zeros,
    )
    learnability = LearnabilityOutput(
        score=score,
        robust_progress_index=zeros,
        percentile_rank=zeros,
        progress_weight=zeros,
        relation_weight=zeros,
        spatial_weight=zeros,
        source_weight=zeros,
    )
    return pseudo, learnability


def _quantiles(value: torch.Tensor, prefix: str) -> dict[str, float]:
    flat = value.detach().float().reshape(-1)
    if not flat.numel():
        return {f"{prefix}_p10": 0.0, f"{prefix}_p50": 0.0, f"{prefix}_p90": 0.0}
    quantile = torch.quantile(flat, torch.tensor([0.1, 0.5, 0.9], device=flat.device))
    return {f"{prefix}_p10": float(quantile[0]), f"{prefix}_p50": float(quantile[1]), f"{prefix}_p90": float(quantile[2])}


class SRGASV01Method(LCRSegV02AMethod):
    method_name = "srgas_v0_1"
    method_version = "0.1a"
    fail_on_optimizer_skip = True

    def __init__(
        self,
        model,
        *,
        config: Mapping[str, Any] | None = None,
        _resolved_config: Mapping[str, Any] | None = None,
    ) -> None:
        resolved = dict(_resolved_config) if _resolved_config is not None else resolve_srgas_method_config(config)
        existing = model.segmentation_head
        if not isinstance(existing, nn.Conv2d):
            raise TypeError("SR-GAS expects the frozen U-Net Conv2d classifier before replacement")
        model.segmentation_head = CosineSegmentationHead.from_conv2d(
            existing,
            temperature=float(resolved["cosine_temperature"]),
            eps=float(resolved["cosine_eps"]),
        )
        base_config = dict(resolved)
        base_config["protocol_id"] = "lcrseg_v0_2a"
        base_config["variant_id"] = "R0"
        super().__init__(model, config=base_config)
        self.config = resolved
        self.variant = str(resolved["srgas_variant"])
        self.behavior_variant = str(resolved.get("srgas_behavior_variant", self.variant))
        if self.behavior_variant not in _VARIANTS:
            raise ValueError(f"unregistered SR-GAS behavior variant: {self.behavior_variant}")
        self._head_hook: Any = None
        self._strong_features: torch.Tensor | None = None
        self._strong_clean_logits: torch.Tensor | None = None
        self._noise_generator: torch.Generator | None = None
        self.last_sensitivity = torch.empty(0)
        self.last_noise_scale = torch.empty(0)
        self._site_start_classifier = torch.empty(0)
        self._site_start_relation_head = torch.empty(0)
        self.shuffler = SpatialRelationShuffler(protocol_seed=int(resolved["noise_seed"]))
        self.srgas_rows: list[dict[str, Any]] = []
        self.r2c_rows: list[dict[str, Any]] = []
        if self.behavior_variant == "A6":
            self.model.requires_grad_(False)
            self.classifier.weight.requires_grad_(True)

    @property
    def classifier(self) -> CosineSegmentationHead:
        head = self.model.segmentation_head
        if not isinstance(head, CosineSegmentationHead):
            raise TypeError("SR-GAS classifier was replaced unexpectedly")
        return head

    def _register_head_capture(self) -> None:
        if self._head_hook is not None:
            self._head_hook.remove()

        def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self._strong_features = inputs[0]
            self._strong_clean_logits = output

        self._head_hook = self.classifier.register_forward_hook(capture)

    def _ensure_generator(self) -> torch.Generator:
        device = self.classifier.weight.device
        if self._noise_generator is None:
            self._noise_generator = torch.Generator(device=device)
            self._noise_generator.manual_seed(int(self.config["noise_seed"]))
        return self._noise_generator

    def begin_site(self, site_id: str, previous_checkpoint, total_steps: int) -> None:  # type: ignore[override]
        # Do not copy a closure-bearing forward hook into the frozen old model.
        if self._head_hook is not None:
            self._head_hook.remove()
            self._head_hook = None
        super().begin_site(site_id, previous_checkpoint, total_steps)
        self._register_head_capture()
        self._ensure_generator()
        self._site_start_classifier = self.classifier.weight.detach().float().cpu().clone()
        self._site_start_relation_head = torch.cat(
            [parameter.detach().float().cpu().flatten() for parameter in self.model.projection_head.parameters()]
        )
        self._strong_features = None
        self._strong_clean_logits = None

    def protocol_semantics(self) -> dict[str, Any]:
        keys = (
            "protocol_id",
            "variant_id",
            "srgas_variant",
            "relation_conditioning",
            "cosine_temperature",
            "cosine_eps",
            "gas_epsilon",
            "noise_variance",
            "noise_distribution",
            "noise_scope",
            "scale_normalization",
            "same_step_sensitivity",
            "r2c_temperature",
            "r2c_resize_mode",
            "r2c_align_corners",
            "r2c_reduction",
            "r2c_source_weight",
            "supervised_source_weight",
            "r2c_added_to_training_objective",
            "channel_mapping",
            "architecture_change",
            "shuffle_r2c_target",
        )
        return {key: self.config[key] for key in keys}

    def _squared_gradient(self, loss: torch.Tensor) -> torch.Tensor:
        gradient = torch.autograd.grad(
            loss,
            self.classifier.weight,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        return gradient.detach().float().square()

    def _r2c(self, result: MethodStepOutput, strong_valid_mask: torch.Tensor):
        maps = result.maps or {}
        old_probability = maps.get("old_relation_probability")
        if old_probability is None or self._strong_clean_logits is None:
            reference = self._strong_clean_logits if self._strong_clean_logits is not None else result.total_loss
            empty_probability = torch.empty(0, device=reference.device)
            return None, differentiable_zero(reference), empty_probability
        target = old_probability
        if bool(self.config.get("shuffle_r2c_target", False)):
            target = self.shuffler.shuffle(target, site_id=str(self.site_id))
        output = relation_to_classifier_loss(
            self._strong_clean_logits,
            target,
            strong_valid_mask,
            historical_anchors_available=self.old_anchor_bank is not None and self.old_anchor_bank.all_classes_valid,
            temperature=float(self.config["r2c_temperature"]),
        )
        return output, output.loss, output.kl_map[output.valid_mask]

    def _sensitivity(self, result: MethodStepOutput, strong_valid_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "r2c_loss": 0.0,
            "r2c_valid_count": 0,
            "s_sup_mean": 0.0,
            "s_sup_std": 0.0,
            "s_r2c_mean": 0.0,
            "s_r2c_std": 0.0,
            "r2c_fraction_of_combined_sensitivity": 0.0,
            "classwise_r2c_sensitivity_mass": [0.0 for _ in range(self.num_classes)],
            "supervised_sensitivity_mass": 0.0,
            "relation_sensitivity_mass": 0.0,
            "sensitivity_cosine": 0.0,
            "a5_a4_noise_scale_l1": 0.0,
            "old_relation_entropy": 0.0,
            "current_seg_down_entropy": 0.0,
            "projection_head_proxy_grad_norm": 0.0,
            "r2c_total_objective_coefficient": 0.0,
        }
        if self.behavior_variant in {"A2"}:
            return torch.ones_like(self.classifier.weight, dtype=torch.float32), diagnostics
        if self.behavior_variant in {"A3", "A6"}:
            return self._squared_gradient(result.total_loss), diagnostics
        supervised = self._squared_gradient(result.losses["loss_sup"])
        diagnostics.update({"s_sup_mean": float(supervised.mean()), "s_sup_std": float(supervised.std(unbiased=False))})
        diagnostics["supervised_sensitivity_mass"] = float(supervised.sum())
        if self.behavior_variant == "A4":
            return supervised, diagnostics
        if self.behavior_variant not in {"A5", "A5_SHUFFLE"}:
            raise AssertionError(f"sensitivity requested for {self.variant}")
        r2c_output, r2c_loss, valid_kl = self._r2c(result, strong_valid_mask)
        diagnostics["r2c_loss"] = float(r2c_loss.detach())
        if r2c_output is not None:
            diagnostics["r2c_valid_count"] = r2c_output.valid_count
            diagnostics.update(_quantiles(valid_kl, "r2c_kl"))
            diagnostics["old_relation_entropy"] = float(
                (-(r2c_output.target_probability * r2c_output.target_probability.clamp_min(1.0e-8).log()).sum(dim=1))[r2c_output.valid_mask[:, 0]].mean()
            ) if r2c_output.valid_count else 0.0
            diagnostics["current_seg_down_entropy"] = float(
                (-(r2c_output.current_probability * r2c_output.current_probability.clamp_min(1.0e-8).log()).sum(dim=1))[r2c_output.valid_mask[:, 0]].mean()
            ) if r2c_output.valid_count else 0.0
        if r2c_output is None or r2c_output.valid_count == 0 or self.old_model is None:
            return unit_mean_source_normalize(supervised, float(self.config["gas_epsilon"])), diagnostics
        relation = self._squared_gradient(r2c_loss)
        diagnostics.update({
            "s_r2c_mean": float(relation.mean()),
            "s_r2c_std": float(relation.std(unbiased=False)),
            "classwise_r2c_sensitivity_mass": [float(value) for value in relation.flatten(1).sum(dim=1)],
            "relation_sensitivity_mass": float(relation.sum()),
        })
        normalized_sup = unit_mean_source_normalize(supervised, float(self.config["gas_epsilon"]))
        normalized_r2c = unit_mean_source_normalize(relation, float(self.config["gas_epsilon"]))
        combined = 0.5 * normalized_sup + 0.5 * normalized_r2c
        diagnostics["sensitivity_cosine"] = float(
            torch.nn.functional.cosine_similarity(normalized_sup.flatten(), normalized_r2c.flatten(), dim=0)
        )
        a4_scale = jascl_inverse_minmax_scale(normalized_sup, float(self.config["gas_epsilon"]))
        combined_scale = jascl_inverse_minmax_scale(combined, float(self.config["gas_epsilon"]))
        diagnostics["a5_a4_noise_scale_l1"] = float((combined_scale - a4_scale).abs().mean())
        diagnostics["r2c_fraction_of_combined_sensitivity"] = float(
            (0.5 * normalized_r2c.sum() / combined.sum().clamp_min(1.0e-8)).detach()
        )
        return combined.detach(), diagnostics

    def _sensitivity_for_stochastic(
        self,
        result: MethodStepOutput,
        strong_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Protocol hook: V0.1a consumes current clean sensitivity immediately."""

        return self._sensitivity(result, strong_valid_mask)

    def _noise_sigma(self, *, site_step: int) -> float:
        del site_step
        return math.sqrt(float(self.config["noise_variance"]))

    def _sample_classifier_weight(
        self,
        scale: torch.Tensor,
        *,
        site_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        sigma = self._noise_sigma(site_step=site_step)
        noise = torch.randn(
            self.classifier.weight.shape,
            device=self.classifier.weight.device,
            dtype=torch.float32,
            generator=self._ensure_generator(),
        ).detach()
        perturbation = (sigma * scale.detach().float() * noise).to(self.classifier.weight.dtype).detach()
        return self.classifier.weight + perturbation, noise, ""

    def _relation_head_drift(self) -> float:
        current = torch.cat([parameter.detach().float().flatten() for parameter in self.model.projection_head.parameters()])
        start = self._site_start_relation_head.to(current.device)
        return float((current - start).norm() / start.norm().clamp_min(1.0e-8))

    def _branch_gradient_diagnostics(
        self,
        *,
        stable_loss: torch.Tensor,
        ssl_loss: torch.Tensor,
        site_step: int,
    ) -> dict[str, float]:
        if (site_step + 1) % 50:
            return {}
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        stable_grad = torch.autograd.grad(stable_loss, parameters, retain_graph=True, create_graph=False, allow_unused=True)
        ssl_grad = torch.autograd.grad(ssl_loss, parameters, retain_graph=True, create_graph=False, allow_unused=True)
        dot = torch.zeros((), device=self.classifier.weight.device)
        stable_norm = torch.zeros_like(dot)
        ssl_norm = torch.zeros_like(dot)
        for parameter, first, second in zip(parameters, stable_grad, ssl_grad, strict=True):
            first_value = first.detach().float() if first is not None else torch.zeros_like(parameter, dtype=torch.float32)
            second_value = second.detach().float() if second is not None else torch.zeros_like(parameter, dtype=torch.float32)
            dot += (first_value * second_value).sum()
            stable_norm += first_value.square().sum()
            ssl_norm += second_value.square().sum()
        denominator = stable_norm.sqrt() * ssl_norm.sqrt()
        return {
            "g_stable_norm": float(stable_norm.sqrt()),
            "g_ssl_norm": float(ssl_norm.sqrt()),
            "gradient_cosine_ssl_stable": float(dot / denominator.clamp_min(1.0e-12)) if bool(denominator.gt(1.0e-12)) else 0.0,
        }

    def training_step(self, labeled_batch, unlabeled_batch, global_step: int, site_step: int) -> MethodStepOutput:  # type: ignore[override]
        self._strong_features = None
        self._strong_clean_logits = None
        result = super().training_step(labeled_batch, unlabeled_batch, global_step, site_step)
        result.scalars.update(
            {
                "srgas_variant": self.variant,
                "hidden_gt_training_usage": 0,
                "classifier_weight_norm": float(self.classifier.weight.detach().float().flatten(1).norm(dim=1).mean()),
                "gas_active": 0.0,
                "stochastic_classifier_forward_count": 0.0,
                "sensitivity_autograd_count": 0.0,
                "r2c_added_to_training_objective": 0.0,
                "stochastic_eval_enabled": 0.0,
                "sampled_weight_norm": float(self.classifier.weight.detach().float().norm()),
                "perturbation_l2_ratio": 0.0,
                "top_sensitivity_quartile_noise_median": 1.0,
                "bottom_sensitivity_quartile_noise_median": 1.0,
                "top_bottom_noise_ratio": 1.0,
                "relation_head_drift": self._relation_head_drift(),
                "clean_assimilation_loss": float(result.losses["loss_assim"].detach()),
                "stochastic_assimilation_loss": float(result.losses["loss_assim"].detach()),
                "sensitivity_p10": 0.0,
                "sensitivity_p50": 0.0,
                "sensitivity_p90": 0.0,
                "noise_scale_p10": 1.0,
                "noise_scale_p50": 1.0,
                "noise_scale_p90": 1.0,
                "supervised_sensitivity_mass": 0.0,
                "relation_sensitivity_mass": 0.0,
                "r2c_loss": 0.0,
                "r2c_valid_count": 0,
            }
        )
        # The first site is the common deterministic A1 parent for A1--A5.
        if self.behavior_variant == "A1" or self.old_model is None:
            stable = result.losses["loss_sup"] + float(result.scalars["lambda_relation_effective"]) * result.losses["loss_relation"]
            plastic = float(result.scalars["lambda_assim_effective"]) * result.losses["loss_assim"]
            result.scalars.update(self._branch_gradient_diagnostics(stable_loss=stable, ssl_loss=plastic, site_step=site_step))
            return result
        if self._strong_features is None or self._strong_clean_logits is None:
            raise RuntimeError("failed to capture the current strong decoder feature")
        maps = result.maps or {}
        required = {"pseudo_labels", "pseudo_valid", "pseudo_source", "learnability"}
        clean_assimilation = result.losses["loss_assim"]
        diagnostics: dict[str, Any] = {}
        if self.behavior_variant == "A2":
            sensitivity = torch.ones_like(self.classifier.weight, dtype=torch.float32)
        else:
            sensitivity, diagnostics = self._sensitivity_for_stochastic(result, unlabeled_batch.strong_valid_mask)
            result.scalars["sensitivity_autograd_count"] = float(
                1 if self.behavior_variant in {"A3", "A4", "A6"} else 2
            )
        scale = (
            torch.ones_like(sensitivity)
            if self.behavior_variant == "A2"
            else jascl_inverse_minmax_scale(sensitivity, float(self.config["gas_epsilon"]))
        )
        before = self.classifier.weight.detach().clone()
        sampled_weight, noise, noise_checksum = self._sample_classifier_weight(scale, site_step=site_step)
        perturbation = (sampled_weight.detach() - self.classifier.weight.detach()).float()
        if not torch.equal(before, self.classifier.weight.detach()):
            raise AssertionError("GAS modified the master classifier weight in-place")
        stochastic_logits = self.classifier(self._strong_features, weight_override=sampled_weight)
        if required.issubset(maps):
            pseudo, learnability = _dummy_pseudo_and_learnability(maps)
            stochastic_assimilation = assimilation_loss(
                stochastic_logits,
                pseudo,
                learnability,
                unlabeled_batch.strong_valid_mask,
            )
        else:
            stochastic_assimilation = differentiable_zero(stochastic_logits)
        lambda_assim = float(result.scalars["lambda_assim_effective"])
        total = result.total_loss - lambda_assim * clean_assimilation + lambda_assim * stochastic_assimilation
        losses = dict(result.losses)
        losses["loss_assim"] = stochastic_assimilation
        flat_sensitivity = sensitivity.detach().float().flatten()
        flat_scale = scale.detach().float().flatten()
        order = torch.argsort(flat_sensitivity)
        quartile = max(1, flat_sensitivity.numel() // 4)
        bottom = flat_scale.index_select(0, order[:quartile]).median()
        top = flat_scale.index_select(0, order[-quartile:]).median()
        site_start = self._site_start_classifier.to(self.classifier.weight.device)
        current = self.classifier.weight.detach().float().flatten(1)
        initial = site_start.float().flatten(1)
        angular = torch.acos(torch.nn.functional.cosine_similarity(current, initial, dim=1).clamp(-1.0, 1.0)).mean()
        result.scalars.update(
            {
                **diagnostics,
                **_quantiles(sensitivity, "sensitivity"),
                **_quantiles(scale, "noise_scale"),
                "gas_active": 1.0,
                "stochastic_classifier_forward_count": 1.0,
                "clean_assimilation_loss": float(clean_assimilation.detach()),
                "stochastic_assimilation_loss": float(stochastic_assimilation.detach()),
                "noise_mean": float(noise.mean()),
                "noise_std": float(noise.std(unbiased=False)),
                "raw_noise_checksum": noise_checksum,
                "effective_noise_sigma": self._noise_sigma(site_step=site_step),
                "sampled_weight_norm": float(sampled_weight.detach().float().norm()),
                "perturbation_l2_ratio": float(perturbation.norm() / self.classifier.weight.detach().float().norm().clamp_min(1.0e-8)),
                "top_sensitivity_quartile_noise_median": float(top),
                "bottom_sensitivity_quartile_noise_median": float(bottom),
                "top_bottom_noise_ratio": float(top / bottom.clamp_min(1.0e-8)),
                "classifier_angular_drift": float(angular),
                "relation_head_drift": self._relation_head_drift(),
            }
        )
        stable = result.losses["loss_sup"] + float(result.scalars["lambda_relation_effective"]) * result.losses["loss_relation"]
        plastic = lambda_assim * stochastic_assimilation
        result.scalars.update(self._branch_gradient_diagnostics(stable_loss=stable, ssl_loss=plastic, site_step=site_step))
        self.last_sensitivity = sensitivity.detach().cpu().clone()
        self.last_noise_scale = scale.detach().cpu().clone()
        row = {"site_id": str(self.site_id), "site_step": int(site_step + 1), "global_step": int(global_step + 1), **result.scalars}
        self.srgas_rows.append(row)
        if self.behavior_variant in {"A5", "A5_SHUFFLE"}:
            self.r2c_rows.append(row)
        return MethodStepOutput(total_loss=total, losses=losses, scalars=result.scalars, maps=result.maps)

    def snapshot_stochastic_state(self) -> dict[str, Any]:
        return {
            "noise_rng_state": self._ensure_generator().get_state().clone(),
            "last_sensitivity": self.last_sensitivity.clone(),
            "last_noise_scale": self.last_noise_scale.clone(),
            "srgas_rows": len(self.srgas_rows),
            "r2c_rows": len(self.r2c_rows),
        }

    def restore_stochastic_state(self, state: Mapping[str, Any]) -> None:
        self._ensure_generator().set_state(state["noise_rng_state"])
        self.last_sensitivity = state["last_sensitivity"].clone()
        self.last_noise_scale = state["last_noise_scale"].clone()
        del self.srgas_rows[int(state["srgas_rows"]):]
        del self.r2c_rows[int(state["r2c_rows"]):]

    def method_state_dict(self) -> dict[str, Any]:
        state = super().method_state_dict()
        statistics = dict(state["method_statistics"])
        statistics.update(
            {
                "srgas_protocol_semantics": self.protocol_semantics(),
                "noise_rng_state": self._ensure_generator().get_state().clone(),
                "last_sensitivity": self.last_sensitivity.clone(),
                "last_noise_scale": self.last_noise_scale.clone(),
                "shuffle_state": self.shuffler.state_dict(),
                "srgas_rows": [dict(row) for row in self.srgas_rows],
                "r2c_rows": [dict(row) for row in self.r2c_rows],
            }
        )
        state["method_statistics"] = statistics
        return state

    def load_method_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_method_state_dict(state)
        statistics = dict(state.get("method_statistics") or {})
        semantics = dict(statistics.get("srgas_protocol_semantics") or {})
        if semantics and semantics != self.protocol_semantics():
            raise ValueError("checkpoint SR-GAS protocol semantics differ from resolved config")
        if statistics.get("noise_rng_state") is not None:
            self._ensure_generator().set_state(statistics["noise_rng_state"])
        self.last_sensitivity = statistics.get("last_sensitivity", torch.empty(0)).clone()
        self.last_noise_scale = statistics.get("last_noise_scale", torch.empty(0)).clone()
        self.shuffler.load_state_dict(dict(statistics.get("shuffle_state") or {}))
        self.srgas_rows = [dict(row) for row in statistics.get("srgas_rows", [])]
        self.r2c_rows = [dict(row) for row in statistics.get("r2c_rows", [])]

    def write_site_artifacts(self, *, run_dir, site_id: str, site_index: int) -> None:
        super().write_site_artifacts(run_dir=run_dir, site_id=site_id, site_index=site_index)
        if self.srgas_rows:
            write_csv(run_dir / "srgas_statistics.csv", self.srgas_rows)
        if self.r2c_rows:
            write_csv(run_dir / "r2c_proxy_statistics.csv", self.r2c_rows)
            write_csv(run_dir / "r2c_sensitivity_statistics.csv", self.r2c_rows)
        write_json(
            run_dir / f"srgas_protocol_site_{site_index}_{site_id}.json",
            {
                "protocol_semantics": self.protocol_semantics(),
                "shuffle_hashes": self.shuffler.state_dict()["hashes"],
            },
        )

    def end_site(self, site_id: str) -> dict[str, Any]:
        summary = super().end_site(site_id)
        summary["srgas_protocol_semantics"] = self.protocol_semantics()
        summary["srgas_steps"] = len(self.srgas_rows)
        summary["r2c_steps"] = len(self.r2c_rows)
        return summary


__all__ = ["SRGAS_DEFAULTS", "SRGASV01Method", "resolve_srgas_method_config"]
