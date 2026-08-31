"""V2 device resolution around the unchanged model-Fisher EWC estimator."""
from __future__ import annotations

from typing import Any

import torch

from ..data.continual_sampler import DeterministicBatcher
from .model_fisher_ewc_v1 import ModelFisherEWCSegMethod


class ModelFisherEWCV2SegMethod(ModelFisherEWCSegMethod):
    method_name = "model_fisher_ewc_v2"
    method_version = "2.0"

    def estimate_fisher(
        self, labeled_batcher: DeterministicBatcher, *, device: torch.device | str
    ) -> dict[str, Any]:
        parameters = self._trainable_parameters()
        if not parameters:
            raise ValueError("model-Fisher EWC requires trainable parameters")
        live_device = next(iter(parameters.values())).device
        requested = torch.device(device)
        if requested.type != live_device.type or (
            requested.index is not None and requested.index != live_device.index
        ):
            raise ValueError("model-Fisher device differs from current model")
        return super().estimate_fisher(labeled_batcher, device=live_device)
