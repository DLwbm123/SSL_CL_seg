from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from lcrseg.data.continual_sampler import DeterministicBatcher
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from lcrseg.engine.continual_runner import ContinualRunner


class _IndexDataset(Dataset[dict[str, int]]):
    def __len__(self) -> int:
        return 11

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"index": index}


def test_rng_restore_and_deterministic_batch_schedule() -> None:
    random.seed(20260818)
    np.random.seed(20260818)
    torch.manual_seed(20260818)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(5))
    random.random()
    np.random.rand()
    torch.rand(5)
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), torch.rand(5))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])

    batcher = DeterministicBatcher(
        _IndexDataset(),
        batch_size=3,
        seed=20260818,
        namespace="golden",
        collate=lambda rows: [row["index"] for row in rows],
    )
    assert batcher.indices_for_step(0) == batcher.indices_for_step(0)
    assert batcher.batch_at(5) == batcher.batch_at(5)
    assert batcher.state.steps_per_epoch == 4


def test_joint_budget_is_the_cumulative_site_budget_not_a_second_multiplier() -> None:
    """Joint budget preserves the exact sequential per-site rounding budget."""

    class Batcher:
        def __init__(self, steps_per_epoch: int) -> None:
            self.steps_per_epoch = steps_per_epoch

    runner = ContinualRunner.__new__(ContinualRunner)
    runner.config = {
        "training": {
            "epochs_per_site": 200,
            "steps_per_site": None,
            "joint_scope": "all",
            "labeled_batch_size": 2,
            "unlabeled_batch_size": 4,
        }
    }
    runner.site_order = ("A", "B", "C")
    runner.method_name = "joint_ssl"
    class SizedDataset:
        def __init__(self, length: int) -> None:
            self.length = length

        def __len__(self) -> int:
            return self.length

    lengths = {"A": (80, 160), "B": (32, 64), "C": (22, 44)}
    runner._datasets = lambda scope: tuple(SizedDataset(length) for length in lengths[scope[0]])  # type: ignore[method-assign]
    # The concatenated dataset needs 66 batches/epoch, whereas the three
    # independent schedules need 40 + 16 + 11 = 67 steps/epoch.
    total, per_epoch = runner._total_steps(Batcher(66), Batcher(66), scope=("A", "B", "C"))
    assert per_epoch == 66
    assert total == 200 * 67


def test_supervised_controls_match_the_reference_ssl_step_budget() -> None:
    class Batcher:
        steps_per_epoch = 20

    runner = ContinualRunner.__new__(ContinualRunner)
    runner.config = {
        "training": {
            "epochs_per_site": 200,
            "steps_per_site": None,
            "unlabeled_batch_size": 4,
        }
    }
    runner.method_name = "finetune_sup"
    runner._reference_unlabeled_steps = lambda scope: 40  # type: ignore[method-assign]
    total, per_epoch = runner._total_steps(Batcher(), None, scope=("A",))
    assert per_epoch == 40
    assert total == 8_000
