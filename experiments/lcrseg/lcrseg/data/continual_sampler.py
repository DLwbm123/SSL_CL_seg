"""Deterministic, resume-friendly batch scheduling for continual sites.

The HDF5 datasets themselves remain standard ``Dataset`` objects.  This
small scheduler deliberately materializes samples in the parent process so a
checkpoint's captured PyTorch RNG state fully determines the next transform
and sample sequence.  Multi-worker HDF5 safety is tested independently at M0;
formal deterministic runs default to this scheduler's worker-free path.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

import torch
from torch.utils.data import Dataset


BatchT = TypeVar("BatchT")


def _namespace_seed(seed: int, namespace: str, epoch: int) -> int:
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    namespace_value = int.from_bytes(digest[:8], "little")
    return (int(seed) * 1_000_003 + namespace_value + int(epoch) * 9_999_991) % (2**63 - 1)


@dataclass(frozen=True)
class BatchScheduleState:
    namespace: str
    seed: int
    dataset_length: int
    batch_size: int
    steps_per_epoch: int


class DeterministicBatcher(Generic[BatchT]):
    """Yield deterministic batches by absolute site step without hidden state."""

    def __init__(
        self,
        dataset: Dataset[Any],
        *,
        batch_size: int,
        seed: int,
        namespace: str,
        collate: Callable[[list[dict[str, Any]]], BatchT],
        shuffle: bool = True,
    ) -> None:
        if len(dataset) < 1:
            raise ValueError("cannot schedule an empty dataset")
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.namespace = str(namespace)
        self.collate = collate
        self.shuffle = bool(shuffle)
        self.steps_per_epoch = int(math.ceil(len(dataset) / self.batch_size))

    @property
    def state(self) -> BatchScheduleState:
        return BatchScheduleState(
            namespace=self.namespace,
            seed=self.seed,
            dataset_length=len(self.dataset),
            batch_size=self.batch_size,
            steps_per_epoch=self.steps_per_epoch,
        )

    def indices_for_step(self, site_step: int) -> list[int]:
        if site_step < 0:
            raise ValueError("site_step must be non-negative")
        epoch, batch_index = divmod(int(site_step), self.steps_per_epoch)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(_namespace_seed(self.seed, self.namespace, epoch))
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        start = batch_index * self.batch_size
        selected = indices[start : start + self.batch_size]
        if not selected:
            raise AssertionError("batch scheduler produced an empty batch")
        return selected

    def batch_at(self, site_step: int) -> BatchT:
        return self.collate([self.dataset[index] for index in self.indices_for_step(site_step)])
