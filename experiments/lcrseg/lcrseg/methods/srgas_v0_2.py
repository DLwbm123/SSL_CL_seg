"""SR-GAS V0.2 lagged warm-start timing amendment."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch

from ..common import write_json
from ..contracts import MethodStepOutput
from ..regularization import LaggedSensitivityState, SharedNoiseStream, linear_noise_warmup
from .base import merged_method_config
from .srgas_v0_1 import SRGAS_DEFAULTS, SRGASV01Method


SRGAS_V02_DEFAULTS: dict[str, Any] = {
    **SRGAS_DEFAULTS,
    "protocol_id": "srgas_v0_2",
    "variant_id": "L0",
    "srgas_variant": "L0",
    "srgas_behavior_variant": "A1",
    "relation_conditioning": "none",
    "noise_variance": 0.1,
    "noise_variance_max": 0.1,
    "noise_warmup_fraction": 0.20,
    "noise_warm_start": False,
    "sensitivity_timing": "none",
    "shared_noise_stream": True,
    "protocol_seed": 20260828,
    "split_seed": 0,
    "r2c_formula_version": "v0_1a",
    "r2c_source_weight": 0.5,
    "supervised_source_weight": 0.5,
    "same_step_sensitivity": False,
}

_VARIANT_CONTRACTS = {
    "L0": {"behavior": "A1", "relation": "none", "timing": "none", "warm": False},
    "L1": {"behavior": "A2", "relation": "none", "timing": "none", "warm": True},
    "L2": {"behavior": "A3", "relation": "none", "timing": "lagged_previous_successful_step", "warm": True},
    "L3": {"behavior": "A4", "relation": "none", "timing": "lagged_previous_successful_step", "warm": True},
    "L4": {
        "behavior": "A5",
        "relation": "relation_to_classifier_proxy",
        "timing": "lagged_previous_successful_step",
        "warm": True,
    },
    "D1": {
        "behavior": "A5",
        "relation": "relation_to_classifier_proxy",
        "timing": "same_step_current_clean",
        "warm": True,
    },
    "D2": {
        "behavior": "A5",
        "relation": "relation_to_classifier_proxy",
        "timing": "lagged_previous_successful_step",
        "warm": False,
    },
}


def resolve_srgas_v02_method_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved = dict(SRGAS_V02_DEFAULTS)
    resolved.update(dict(config or {}))
    variant = str(resolved.get("srgas_variant", resolved.get("variant_id", "L0"))).upper().replace("-", "_")
    if variant not in _VARIANT_CONTRACTS:
        raise ValueError(f"unregistered SR-GAS V0.2 variant: {variant}")
    contract = _VARIANT_CONTRACTS[variant]
    resolved["variant_id"] = variant
    resolved["srgas_variant"] = variant
    expected = {
        "protocol_id": "srgas_v0_2",
        "srgas_behavior_variant": contract["behavior"],
        "relation_conditioning": contract["relation"],
        "sensitivity_timing": contract["timing"],
        "noise_warm_start": contract["warm"],
        "noise_variance": 0.1,
        "noise_variance_max": 0.1,
        "noise_warmup_fraction": 0.20,
        "shared_noise_stream": True,
        "r2c_formula_version": "v0_1a",
        "r2c_source_weight": 0.5,
        "supervised_source_weight": 0.5,
        "r2c_added_to_training_objective": False,
        "channel_mapping": "none",
        "architecture_change": False,
        "shuffle_r2c_target": False,
        "same_step_sensitivity": contract["timing"] == "same_step_current_clean",
    }
    for key, value in expected.items():
        if resolved.get(key) != value:
            raise ValueError(f"{key} differs from the frozen SR-GAS V0.2 {variant} contract")
    if str(resolved["assimilation_mode"]) != "legacy_continuous_v01" or str(resolved["consolidation_mode"]) != "uniform_relation":
        raise ValueError("SR-GAS V0.2 must retain the exact frozen V0.2a R0 learner")
    if int(resolved["protocol_seed"]) != 20260828:
        raise ValueError("SR-GAS V0.2 protocol seed is frozen")
    if int(resolved["split_seed"]) not in {0, 1, 2}:
        raise ValueError("split seed must be a registered Fundus seed")
    return merged_method_config(resolved)


class SRGASV02Method(SRGASV01Method):
    method_name = "srgas_v0_2"
    method_version = "0.2"

    def __init__(self, model, *, config: Mapping[str, Any] | None = None) -> None:
        resolved = resolve_srgas_v02_method_config(config)
        super().__init__(model, _resolved_config=resolved)
        self.lagged_state = LaggedSensitivityState.empty()
        self._pending_sensitivity = torch.empty(0)
        self.noise_stream = SharedNoiseStream(
            protocol_seed=int(resolved["protocol_seed"]),
            split_seed=int(resolved["split_seed"]),
        )

    @property
    def uses_lag(self) -> bool:
        return self.config["sensitivity_timing"] == "lagged_previous_successful_step"

    def protocol_semantics(self) -> dict[str, Any]:
        keys = (
            "protocol_id",
            "variant_id",
            "srgas_variant",
            "srgas_behavior_variant",
            "relation_conditioning",
            "cosine_temperature",
            "cosine_eps",
            "gas_epsilon",
            "noise_variance_max",
            "noise_warmup_fraction",
            "noise_warm_start",
            "sensitivity_timing",
            "shared_noise_stream",
            "protocol_seed",
            "split_seed",
            "r2c_formula_version",
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

    def begin_site(self, site_id: str, previous_checkpoint, total_steps: int) -> None:  # type: ignore[override]
        super().begin_site(site_id, previous_checkpoint, total_steps)
        self.lagged_state.reset_for_site(site_id=site_id, reference=self.classifier.weight)
        self._pending_sensitivity = torch.empty(0)

    def _sensitivity_for_stochastic(
        self,
        result: MethodStepOutput,
        strong_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        current, diagnostics = self._sensitivity(result, strong_valid_mask)
        diagnostics["lagged_buffer_valid_before_step"] = float(self.lagged_state.valid)
        diagnostics["successful_site_step_before_step"] = int(self.lagged_state.successful_site_step)
        if not self.uses_lag:
            self._pending_sensitivity = torch.empty(0)
            return current, diagnostics
        self._pending_sensitivity = current.detach().float().cpu().clone()
        used = self.lagged_state.current_or_ones(self.classifier.weight)
        diagnostics["lagged_sensitivity_l1_to_current"] = float(
            (used.detach().float() - current.detach().float()).abs().mean()
        )
        return used, diagnostics

    def _warmup_multiplier(self) -> float:
        if not bool(self.config["noise_warm_start"]):
            return 1.0
        return linear_noise_warmup(
            successful_site_step=self.lagged_state.successful_site_step,
            total_site_steps=self.total_steps,
            warmup_fraction=float(self.config["noise_warmup_fraction"]),
        )

    def _noise_sigma(self, *, site_step: int) -> float:
        del site_step
        return math.sqrt(float(self.config["noise_variance_max"])) * self._warmup_multiplier()

    def _sample_classifier_weight(
        self,
        scale: torch.Tensor,
        *,
        site_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        del site_step
        if self.site_id is None:
            raise RuntimeError("cannot sample shared noise before site initialization")
        noise, checksum = self.noise_stream.sample(
            site_id=self.site_id,
            successful_site_step=self.lagged_state.successful_site_step,
            weight_shape=self.classifier.weight.shape,
            device=self.classifier.weight.device,
        )
        perturbation = (
            self._noise_sigma(site_step=0) * scale.detach().float() * noise
        ).to(self.classifier.weight.dtype).detach()
        return self.classifier.weight + perturbation, noise.detach(), checksum

    def training_step(self, labeled_batch, unlabeled_batch, global_step: int, site_step: int) -> MethodStepOutput:  # type: ignore[override]
        result = super().training_step(labeled_batch, unlabeled_batch, global_step, site_step)
        result.scalars.update(
            {
                "successful_site_step_before_step": int(self.lagged_state.successful_site_step),
                "noise_warmup_multiplier": self._warmup_multiplier(),
                "lagged_buffer_valid_before_step": float(self.lagged_state.valid),
                "lagged_buffer_is_parameter": 0.0,
                "noise_stream_split_seed": int(self.config["split_seed"]),
            }
        )
        if self.srgas_rows and int(self.srgas_rows[-1].get("global_step", -1)) == int(global_step + 1):
            self.srgas_rows[-1].update(result.scalars)
        return result

    def after_optimizer_step(self) -> None:
        super().after_optimizer_step()
        if self.uses_lag and self.old_model is not None and self.behavior_variant != "A1":
            if self._pending_sensitivity.numel() == 0:
                raise RuntimeError("successful lagged step has no pending clean sensitivity")
            self.lagged_state.commit_after_success(self._pending_sensitivity)
        else:
            self.lagged_state.advance_without_commit()
        self._pending_sensitivity = torch.empty(0)

    def snapshot_stochastic_state(self) -> dict[str, Any]:
        state = super().snapshot_stochastic_state()
        state.update(
            {
                "v02_lagged_state": self.lagged_state.state_dict(),
                "v02_pending_sensitivity": self._pending_sensitivity.clone(),
                "v02_noise_stream": self.noise_stream.state_dict(),
            }
        )
        return state

    def restore_stochastic_state(self, state: Mapping[str, Any]) -> None:
        super().restore_stochastic_state(state)
        self.lagged_state.load_state_dict(dict(state["v02_lagged_state"]))
        self._pending_sensitivity = state["v02_pending_sensitivity"].clone()
        self.noise_stream.load_state_dict(dict(state["v02_noise_stream"]))

    def method_state_dict(self) -> dict[str, Any]:
        state = super().method_state_dict()
        statistics = dict(state["method_statistics"])
        statistics.update(
            {
                "v02_lagged_state": self.lagged_state.state_dict(),
                "v02_pending_sensitivity": self._pending_sensitivity.clone(),
                "v02_noise_stream": self.noise_stream.state_dict(),
            }
        )
        state["method_statistics"] = statistics
        return state

    def load_method_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_method_state_dict(state)
        statistics = dict(state.get("method_statistics") or {})
        self.lagged_state.load_state_dict(dict(statistics.get("v02_lagged_state") or {}))
        self._pending_sensitivity = statistics.get("v02_pending_sensitivity", torch.empty(0)).clone()
        self.noise_stream.load_state_dict(dict(statistics.get("v02_noise_stream") or {}))

    def write_site_artifacts(self, *, run_dir: Path, site_id: str, site_index: int) -> None:
        super().write_site_artifacts(run_dir=run_dir, site_id=site_id, site_index=site_index)
        write_json(
            Path(run_dir) / f"srgas_v02_timing_site_{site_index}_{site_id}.json",
            {
                "protocol_semantics": self.protocol_semantics(),
                "lagged_state": {
                    "valid": self.lagged_state.valid,
                    "site_id": self.lagged_state.site_id,
                    "successful_site_step": self.lagged_state.successful_site_step,
                    "buffer_shape": list(self.lagged_state.buffer.shape),
                    "buffer_finite": bool(torch.isfinite(self.lagged_state.buffer).all()),
                },
                "noise_stream": self.noise_stream.state_dict(),
                "noise_device": str(self.classifier.weight.device),
            },
        )

    def end_site(self, site_id: str) -> dict[str, Any]:
        summary = super().end_site(site_id)
        summary.update(
            {
                "successful_site_steps": self.lagged_state.successful_site_step,
                "lagged_buffer_valid": self.lagged_state.valid,
                "shared_noise_keys": len(self.noise_stream.hashes),
            }
        )
        return summary


__all__ = ["SRGAS_V02_DEFAULTS", "SRGASV02Method", "resolve_srgas_v02_method_config"]
