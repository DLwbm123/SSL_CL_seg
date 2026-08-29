"""Static and sequential supervised baselines on the shared training engine."""
from __future__ import annotations

from typing import Any, Mapping

from ..contracts import LabeledBatch, MethodStepOutput, UnlabeledBatch
from .base import ContinualSegMethod


class SupervisedMethod(ContinualSegMethod):
    """A segmentation-only baseline; relation features are intentionally unused."""

    method_name = "static_sup"

    def __init__(self, *args: Any, continual: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.continual = bool(continual)
        self.method_name = "finetune_sup" if self.continual else "static_sup"

    def training_step(
        self,
        labeled_batch: LabeledBatch,
        unlabeled_batch: UnlabeledBatch | None,
        global_step: int,
        site_step: int,
    ) -> MethodStepOutput:
        output = self.model(labeled_batch.image)
        losses = self._all_loss_keys(output.logits, self._supervised_losses(output, labeled_batch))
        return MethodStepOutput(
            total_loss=losses["loss_sup"],
            losses=losses,
            scalars={"pseudo_valid_ratio": 0.0, "compatibility_mean": 0.0, "learnability_mean": 0.0},
        )
