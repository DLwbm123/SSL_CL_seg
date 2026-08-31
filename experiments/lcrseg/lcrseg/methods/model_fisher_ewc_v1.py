"""Three-class pixel model-Fisher EWC adaptation; legacy EWC stays unchanged."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import torch

from ..data.continual_sampler import DeterministicBatcher, _namespace_seed
from ..engine.checkpoint import capture_rng_state, load_checkpoint, restore_rng_state
from .base import clone_state_dict, merged_method_config
from .sequential_ssl import SequentialSSLMethod


_STATE_SCHEMA = "MODEL_FISHER_EWC_STATE_V1"
_REQUIRED_CONFIG = {
    "ewc_lambda",
    "ewc_gamma",
    "fisher_max_images",
    "fisher_points_per_image",
    "fisher_seed",
}


def _exact_integer(value: Any, name: str, *, positive: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if (positive and value < 1) or (not positive and not 0 <= value < 2**63):
        bound = "positive" if positive else "in [0, 2^63)"
        raise ValueError(f"{name} must be {bound}")
    return value


def resolve_model_fisher_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    provided = dict(config or {})
    missing = _REQUIRED_CONFIG.difference(provided)
    if missing:
        raise ValueError(f"model-Fisher EWC config misses required keys: {sorted(missing)}")
    if "ewc_fisher_batches" in provided:
        raise ValueError("legacy ewc_fisher_batches is invalid for model_fisher_ewc_v1")
    resolved = dict(provided)
    for name in ("ewc_lambda", "ewc_gamma"):
        value = float(resolved[name])
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        resolved[name] = value
    if resolved["ewc_lambda"] < 0 or not 0 <= resolved["ewc_gamma"] <= 1:
        raise ValueError("ewc_lambda must be nonnegative and ewc_gamma must be in [0,1]")
    resolved["fisher_max_images"] = _exact_integer(resolved["fisher_max_images"], "fisher_max_images", positive=True)
    resolved["fisher_points_per_image"] = _exact_integer(
        resolved["fisher_points_per_image"], "fisher_points_per_image", positive=True
    )
    resolved["fisher_seed"] = _exact_integer(resolved["fisher_seed"], "fisher_seed", positive=False)
    return merged_method_config(resolved)


class ModelFisherEWCSegMethod(SequentialSSLMethod):
    """Sequential SSL with a separately identified categorical pixel Fisher."""

    method_name = "model_fisher_ewc_v1"
    method_version = "1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        config = resolve_model_fisher_config(kwargs.pop("config", None))
        super().__init__(*args, config=config, **kwargs)
        if self.num_classes != 3:
            raise ValueError("model_fisher_ewc_v1 requires exactly three output classes")
        parameters = self._trainable_parameters()
        if not parameters:
            raise ValueError("model-Fisher EWC requires trainable parameters")
        if any(parameter.dtype not in (torch.float32, torch.float64) for parameter in parameters.values()):
            raise ValueError("model-Fisher EWC V1 supports only float32/float64 parameters")
        self.ewc_lambda = float(self.config["ewc_lambda"])
        self.ewc_gamma = float(self.config["ewc_gamma"])
        self.fisher_max_images = int(self.config["fisher_max_images"])
        self.fisher_points_per_image = int(self.config["fisher_points_per_image"])
        self.fisher_seed = int(self.config["fisher_seed"])
        self.reference_parameters: dict[str, torch.Tensor] = {}
        self.fisher_diagonal: dict[str, torch.Tensor] = {}
        self.completed_consolidations = 0

    def _trainable_parameters(self) -> dict[str, torch.nn.Parameter]:
        return {name: parameter for name, parameter in self.model.named_parameters() if parameter.requires_grad}

    def _configuration(self) -> dict[str, Any]:
        return clone_state_dict(self.config)

    def _validated_state(
        self, state: Mapping[str, Any] | None
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
        if not isinstance(state, Mapping) or set(state) != {
            "schema",
            "resolved_method_config",
            "completed_consolidations",
            "reference_parameters",
            "fisher_diagonal",
        }:
            raise ValueError("model-Fisher EWC state has incomplete or extra keys")
        if state["schema"] != _STATE_SCHEMA or state["resolved_method_config"] != self._configuration():
            raise ValueError("model-Fisher EWC state schema or resolved settings differ")
        count = state["completed_consolidations"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("completed_consolidations must be a nonnegative integer")
        expected = self._trainable_parameters()
        reference = state["reference_parameters"]
        fisher = state["fisher_diagonal"]
        if not isinstance(reference, Mapping) or not isinstance(fisher, Mapping):
            raise ValueError("model-Fisher EWC tensor dictionaries are missing")
        if count == 0:
            if reference or fisher:
                raise ValueError("fresh model-Fisher EWC state must be empty")
            return {}, {}, 0
        expected_keys = set(expected)
        if set(reference) != expected_keys or set(fisher) != expected_keys:
            raise ValueError("model-Fisher EWC parameter keys are incomplete or extra")
        checked_reference: dict[str, torch.Tensor] = {}
        checked_fisher: dict[str, torch.Tensor] = {}
        for name, parameter in expected.items():
            first, second = reference[name], fisher[name]
            if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
                raise ValueError(f"non-tensor EWC state for {name}")
            if first.shape != parameter.shape or second.shape != parameter.shape:
                raise ValueError(f"wrong EWC state shape for {name}")
            if first.dtype != parameter.dtype or second.dtype != parameter.dtype:
                raise ValueError(f"wrong EWC state dtype for {name}")
            if first.requires_grad or second.requires_grad:
                raise ValueError(f"EWC state must be detached for {name}")
            if not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all()):
                raise ValueError(f"non-finite EWC state for {name}")
            if bool(second.lt(0).any()):
                raise ValueError(f"negative Fisher state for {name}")
            checked_reference[name] = first.detach().clone().to(parameter.device)
            checked_fisher[name] = second.detach().clone().to(parameter.device)
        return checked_reference, checked_fisher, count

    def _load_model_fisher_state(self, statistics: Mapping[str, Any]) -> None:
        if statistics.get("old_model_state") or statistics.get("old_model_checksum"):
            raise ValueError("model_fisher_ewc_v1 cannot restore an old teacher")
        reference, fisher, count = self._validated_state(statistics.get("model_fisher_ewc_state"))
        self.reference_parameters = reference
        self.fisher_diagonal = fisher
        self.completed_consolidations = count

    def _load_previous_model(self, previous_checkpoint):  # type: ignore[override]
        payload = load_checkpoint(previous_checkpoint, map_location="cpu")
        if payload["method_name"] != self.method_name or payload["method_version"] != self.method_version:
            raise ValueError("previous checkpoint has a different method identity")
        statistics = payload["method_statistics"]
        if statistics.get("old_model_state") or statistics.get("old_model_checksum"):
            raise ValueError("model_fisher_ewc_v1 cannot restore an old teacher")
        reference, fisher, count = self._validated_state(statistics.get("model_fisher_ewc_state"))
        if count != self.site_index:
            raise ValueError("previous checkpoint consolidation count differs from the next site index")
        expected_model = self.model.state_dict()
        incoming_model = payload["current_model_state"]
        if set(incoming_model) != set(expected_model) or any(
            incoming_model[name].shape != value.shape or incoming_model[name].dtype != value.dtype
            for name, value in expected_model.items()
        ):
            raise ValueError("previous checkpoint model keys, shapes or dtypes differ")
        if not all(bool(torch.isfinite(value).all()) for value in incoming_model.values()):
            raise ValueError("previous checkpoint model state is non-finite")
        self.model.load_state_dict(payload["current_model_state"], strict=True)
        self.reference_parameters = reference
        self.fisher_diagonal = fisher
        self.completed_consolidations = count
        return payload

    def begin_site(self, site_id: str, previous_checkpoint, total_steps: int) -> None:  # type: ignore[override]
        super().begin_site(site_id, previous_checkpoint, total_steps)
        if previous_checkpoint is None:
            if self.site_index != 0:
                raise ValueError("only site index zero may start without a previous checkpoint")
            self.reference_parameters = {}
            self.fisher_diagonal = {}
            self.completed_consolidations = 0

    def _ewc_loss(self, reference: torch.Tensor) -> torch.Tensor:
        penalty = reference.sum() * 0.0
        if self.completed_consolidations == 0:
            return penalty
        parameters = self._trainable_parameters()
        if set(parameters) != set(self.reference_parameters) or set(parameters) != set(self.fisher_diagonal):
            raise RuntimeError("live model and model-Fisher EWC state differ")
        for name, parameter in parameters.items():
            penalty = penalty + 0.5 * (
                self.fisher_diagonal[name].to(parameter) * (parameter - self.reference_parameters[name].to(parameter)).square()
            ).sum()
        return penalty

    def training_step(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        result = super().training_step(*args, **kwargs)
        ewc = self._ewc_loss(result.total_loss)
        losses = dict(result.losses)
        losses["loss_relation"] = ewc
        scalars = dict(result.scalars)
        scalars["loss_model_fisher_ewc"] = float(ewc.detach())
        scalars["lambda_model_fisher_ewc"] = self.ewc_lambda
        return type(result)(
            total_loss=result.total_loss + self.ewc_lambda * ewc,
            losses=losses,
            scalars=scalars,
            maps=result.maps,
        )

    def estimate_fisher(
        self, labeled_batcher: DeterministicBatcher, *, device: torch.device | str
    ) -> dict[str, Any]:
        if torch.is_inference_mode_enabled():
            raise RuntimeError("model-Fisher estimation cannot run in inference mode")
        if self.site_id is None or self.site_index < 0 or self.completed_consolidations != self.site_index:
            raise RuntimeError("model-Fisher consolidation is duplicate or out of stage order")
        dataset = labeled_batcher.dataset
        if not hasattr(dataset, "image_at"):
            raise TypeError("model-Fisher input dataset must provide image_at(index)")
        parameters = self._trainable_parameters()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in parameters.values()):
            raise ValueError("current model parameters are non-finite")
        target_device = torch.device(device)
        if any(parameter.device != target_device for parameter in parameters.values()):
            raise ValueError("model-Fisher device differs from current model")
        first_parameter = next(iter(parameters.values()))
        modes = [(module, module.training) for module in self.model.modules()]
        parameter_values = {name: value.detach().clone() for name, value in parameters.items()}
        parameter_gradients = {
            name: (value.grad, value.grad.detach().clone() if value.grad is not None else None)
            for name, value in parameters.items()
        }
        buffer_values = {name: value.detach().clone() for name, value in self.model.named_buffers()}
        rng_state = capture_rng_state()
        estimates = {name: torch.zeros_like(value) for name, value in parameters.items()}
        selected_images: list[int] = []
        selected_points: list[list[int]] = []
        actual_points = 0
        mutation_detected = False
        try:
            self.model.eval()
            total_images = len(dataset)
            if total_images < 1:
                raise ValueError("model-Fisher input dataset is empty")
            generator = torch.Generator()
            generator.manual_seed(_namespace_seed(self.fisher_seed, f"model_fisher_ewc_v1:{self.site_id}", 0))
            selected_images = torch.randperm(total_images, generator=generator)[: self.fisher_max_images].tolist()
            with torch.enable_grad():
                for dataset_index in selected_images:
                    image = dataset.image_at(dataset_index)
                    if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] < 1:
                        raise ValueError("model-Fisher image must be a [C,H,W] tensor")
                    if not image.is_floating_point() or not bool(torch.isfinite(image).all()):
                        raise ValueError("model-Fisher image must be finite floating point")
                    image = image.unsqueeze(0).to(target_device, dtype=first_parameter.dtype)
                    output = self.model(image)
                    logits = output.logits
                    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != self.num_classes:
                        raise ValueError("model-Fisher logits must be [1,3,H,W]")
                    if not bool(torch.isfinite(logits).all()):
                        raise ValueError("model-Fisher logits are non-finite")
                    flat = logits[0].reshape(self.num_classes, -1).log_softmax(dim=0)
                    point_indices = torch.randperm(flat.shape[1], generator=generator)[: self.fisher_points_per_image].tolist()
                    if not point_indices:
                        raise ValueError("model-Fisher output contains no pixels")
                    selected_points.append(point_indices)
                    actual_points += len(point_indices)
                    for point_position, point_index in enumerate(point_indices):
                        point_log_probability = flat[:, point_index]
                        probabilities = point_log_probability.detach().exp()
                        for class_index in range(self.num_classes):
                            retain = point_position + 1 < len(point_indices) or class_index + 1 < self.num_classes
                            gradients = torch.autograd.grad(
                                point_log_probability[class_index],
                                tuple(parameters.values()),
                                retain_graph=retain,
                                create_graph=False,
                                allow_unused=True,
                            )
                            for (name, _), gradient in zip(parameters.items(), gradients, strict=True):
                                if gradient is not None:
                                    estimates[name].add_(gradient.detach().square() * probabilities[class_index])
            if actual_points < 1:
                raise ValueError("model-Fisher selected zero pixels")
            for name, value in estimates.items():
                value.div_(actual_points)
                if not bool(torch.isfinite(value).all()) or bool(value.lt(0).any()):
                    raise FloatingPointError(f"invalid Fisher diagonal for {name}")
            mutation_detected = any(
                not torch.equal(parameters[name].detach(), value) for name, value in parameter_values.items()
            ) or any(
                not torch.equal(dict(self.model.named_buffers())[name].detach(), value)
                for name, value in buffer_values.items()
            )
        finally:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(parameter_values[name])
                    original, copied = parameter_gradients[name]
                    if original is None:
                        parameter.grad = None
                    else:
                        original.copy_(copied)
                        parameter.grad = original
                current_buffers = dict(self.model.named_buffers())
                for name, value in buffer_values.items():
                    current_buffers[name].copy_(value)
            for module, mode in modes:
                module.training = mode
            restore_rng_state(rng_state)
        if mutation_detected:
            raise RuntimeError("model or buffer changed during model-Fisher estimation")
        updated = {
            name: value if name not in self.fisher_diagonal else self.ewc_gamma * self.fisher_diagonal[name].to(value) + value
            for name, value in estimates.items()
        }
        new_reference = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        if any(new_reference[name].data_ptr() == updated[name].data_ptr() for name in updated):
            raise AssertionError("model-Fisher reference and Fisher state alias")
        self.fisher_diagonal = updated
        self.reference_parameters = new_reference
        self.completed_consolidations += 1
        selector = json.dumps([selected_images, selected_points], separators=(",", ":")).encode()
        total_values = sum(value.numel() for value in updated.values())
        fisher_mean = sum(float(value.detach().double().sum()) for value in updated.values()) / total_values
        return {
            "estimator": _STATE_SCHEMA,
            "selected_image_indices": selected_images,
            "selected_point_indices": selected_points,
            "selection_sha256": hashlib.sha256(selector).hexdigest(),
            "actual_images": len(selected_images),
            "actual_points": actual_points,
            "classes": self.num_classes,
            "model_forward_calls": len(selected_images),
            "autograd_grad_calls": actual_points * self.num_classes,
            "completed_consolidations": self.completed_consolidations,
            "fisher_mean": fisher_mean,
        }

    def method_state_dict(self) -> dict[str, Any]:
        state = super().method_state_dict()
        statistics = dict(state["method_statistics"])
        statistics["model_fisher_ewc_state"] = {
            "schema": _STATE_SCHEMA,
            "resolved_method_config": self._configuration(),
            "completed_consolidations": self.completed_consolidations,
            "reference_parameters": clone_state_dict(self.reference_parameters),
            "fisher_diagonal": clone_state_dict(self.fisher_diagonal),
        }
        state["method_statistics"] = statistics
        return state

    def load_method_state_dict(self, state: Mapping[str, Any]) -> None:
        statistics = state.get("method_statistics")
        if not isinstance(statistics, Mapping):
            raise ValueError("checkpoint lacks method_statistics")
        self._load_model_fisher_state(statistics)
        if self.site_index >= 0 and self.completed_consolidations not in {self.site_index, self.site_index + 1}:
            raise ValueError("checkpoint consolidation count differs from active site")
        super().load_method_state_dict(state)
