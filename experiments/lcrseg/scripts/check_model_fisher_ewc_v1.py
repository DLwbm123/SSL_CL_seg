"""Bounded CPU checks for the registered model-Fisher EWC engineering phase."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import resource
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lcrseg.contracts import LabeledBatch, SegModelOutput, UnlabeledBatch
from lcrseg.data.continual_sampler import DeterministicBatcher
from lcrseg.engine.checkpoint import capture_rng_state, checkpoint_payload, save_checkpoint
from lcrseg.engine.continual_runner import ContinualRunner
from lcrseg.engine.trainer import Trainer, TrainerState, build_optimizer, build_scheduler
from lcrseg.methods.model_fisher_ewc_v1 import ModelFisherEWCSegMethod
from lcrseg.methods.sequential_ssl import SequentialSSLMethod
from scripts.verify_resume_equivalence import compare
from tests.conftest import make_synthetic_root


LIMITS = {
    "model_forward_calls": 2048,
    "model_forward_images": 4096,
    "autograd_grad_calls": 2048,
    "backward_calls": 512,
    "optimizer_steps": 256,
}
ACTIVE_BUDGET: "Budget | None" = None


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self) -> None:
        self.counts = {name: 0 for name in LIMITS}

    def bump(self, name: str, amount: int = 1) -> None:
        if self.counts[name] + amount > LIMITS[name]:
            raise BudgetExceeded(f"{name} would exceed {LIMITS[name]}")
        self.counts[name] += amount

    @contextmanager
    def instrument(self):
        global ACTIVE_BUDGET
        if ACTIVE_BUDGET is not None:
            raise RuntimeError("budget instrumentation is already active")
        original_grad = torch.autograd.grad
        original_backward = torch.autograd.backward
        original_adam_step = torch.optim.Adam.step

        def counted_grad(*args, **kwargs):
            self.bump("autograd_grad_calls")
            return original_grad(*args, **kwargs)

        def counted_backward(*args, **kwargs):
            self.bump("backward_calls")
            return original_backward(*args, **kwargs)

        def counted_adam_step(optimizer, *args, **kwargs):
            self.bump("optimizer_steps")
            return original_adam_step(optimizer, *args, **kwargs)

        ACTIVE_BUDGET = self
        try:
            with mock.patch.object(torch.autograd, "grad", counted_grad), mock.patch.object(
                torch.autograd, "backward", counted_backward
            ), mock.patch.object(torch.optim.Adam, "step", counted_adam_step):
                yield
        finally:
            ACTIVE_BUDGET = None


class TinyPixelModel(nn.Module):
    """Counted 1x1 three-class model used by every executable check."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        *unused,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        **unused_keywords,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.unused = nn.Parameter(torch.zeros(()))
        self.batch_norm = nn.BatchNorm2d(in_channels) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, image: torch.Tensor) -> SegModelOutput:
        if ACTIVE_BUDGET is None:
            raise RuntimeError("untracked toy model forward")
        ACTIVE_BUDGET.bump("model_forward_calls")
        ACTIVE_BUDGET.bump("model_forward_images", int(image.shape[0]))
        logits = self.classifier(self.dropout(self.batch_norm(image)))
        return SegModelOutput(logits=logits, relation_features=logits)


class ImageOnlyDataset:
    def __init__(self, images: list[torch.Tensor], *, consume_rng: bool = False, fail_at: int | None = None) -> None:
        self.images = images
        self.consume_rng = consume_rng
        self.fail_at = fail_at
        self.visits: list[int] = []

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        raise AssertionError("model-Fisher touched the label-bearing dataset accessor")

    def image_at(self, index: int) -> torch.Tensor:
        self.visits.append(index)
        if self.consume_rng:
            random.random()
            np.random.rand()
            torch.rand(())
        if self.fail_at is not None and len(self.visits) == self.fail_at:
            raise OSError("injected image read failure")
        return self.images[index].clone()


def method_config(*, gamma: float = 0.6, maximum: int = 3, points: int = 4, seed: int = 271828) -> dict[str, object]:
    return {
        "ewc_lambda": 1.7,
        "ewc_gamma": gamma,
        "fisher_max_images": maximum,
        "fisher_points_per_image": points,
        "fisher_seed": seed,
        "lambda_assim": 0.0,
        "tau_cls": 0.0,
    }


def new_method(
    *,
    dtype: torch.dtype = torch.float64,
    config: dict[str, object] | None = None,
    model: TinyPixelModel | None = None,
) -> ModelFisherEWCSegMethod:
    model = model or TinyPixelModel()
    model = model.to(dtype=dtype)
    method = ModelFisherEWCSegMethod(model, config=config or method_config(), static=False)
    method.set_site_index(0)
    method.begin_site("A", None, 1)
    return method


def batcher(dataset: ImageOnlyDataset, size: int = 1) -> DeterministicBatcher:
    return DeterministicBatcher(
        dataset,
        batch_size=size,
        seed=7,
        namespace="partition-is-not-the-fisher-unit",
        collate=lambda rows: rows,
        shuffle=True,
    )


def tensor_dict_equal(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor], *, atol: float = 0.0) -> bool:
    return set(first) == set(second) and all(
        torch.allclose(first[name], second[name], atol=atol, rtol=0.0) for name in first
    )


def assert_same(first, second, *, atol: float = 0.0) -> None:
    ok, message = compare(first, second, atol=atol, rtol=0.0)
    assert ok, message


def configure_known_probabilities(method: ModelFisherEWCSegMethod) -> torch.Tensor:
    probability = torch.tensor([0.2, 0.3, 0.5], dtype=next(method.model.parameters()).dtype)
    with torch.no_grad():
        method.model.classifier.weight.zero_()
        method.model.classifier.bias.copy_(probability.log())
        method.model.unused.zero_()
    return probability


def save_stage_checkpoint(method: ModelFisherEWCSegMethod, path: Path) -> None:
    optimizer = build_optimizer(method, lr=0.01, weight_decay=0.0)
    state = method.method_state_dict()
    payload = checkpoint_payload(
        method_name=method.method_name,
        method_version=method.method_version,
        git_commit="SYNTHETIC_CHECK",
        config_resolved={"method": method.config},
        site_id=method.site_id or "",
        site_index=method.site_index,
        epoch=0,
        site_step=1,
        global_step=1,
        current_model_state=method.model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=build_scheduler(optimizer, total_steps=1).state_dict(),
        scaler_state={},
        current_anchor_state=state["current_anchor_state"],
        historical_anchor_state=state["historical_anchor_state"],
        bootstrap_state=state["bootstrap_state"],
        method_statistics=state["method_statistics"],
        data_split_hash="synthetic",
        manifest_hash="synthetic",
    )
    save_checkpoint(path, payload)


def sample_batches(dtype: torch.dtype = torch.float64) -> tuple[LabeledBatch, UnlabeledBatch]:
    label = torch.arange(64).reshape(8, 8).remainder(3).long()
    image = F.one_hot(label, num_classes=3).permute(2, 0, 1).to(dtype)
    images = torch.stack((image, image.roll(1, dims=-1)))
    labels = torch.stack((label, label.roll(1, dims=-1)))
    valid = torch.ones((2, 1, 8, 8), dtype=torch.bool)
    labeled = LabeledBatch(
        image=images,
        label=labels,
        valid_mask=valid,
        case_id=["synthetic-a", "synthetic-b"],
        patient_id=["synthetic-a", "synthetic-b"],
        site=["S", "S"],
        slice_index=[None, None],
    )
    unlabeled = UnlabeledBatch(
        weak_image=images.clone(),
        strong_image=images.clone(),
        strong_valid_mask=valid.clone(),
        case_id=["synthetic-u", "synthetic-v"],
        patient_id=["synthetic-u", "synthetic-v"],
        site=["S", "S"],
        slice_index=[None, None],
        geometry_record=[{}, {}],
    )
    return labeled, unlabeled


def check_closed_form(_: Path) -> dict[str, object]:
    images = [
        torch.tensor([[[1.0, 2.0], [-1.0, 0.0]], [[0.5, -0.5], [1.5, 2.5]], [[-2.0, 1.0], [0.5, -1.5]]], dtype=torch.float64),
        torch.tensor([[[0.25, -0.75], [1.25, 2.25]], [[-1.0, 0.0], [1.0, 2.0]], [[0.5, 1.5], [-0.5, -1.5]]], dtype=torch.float64),
    ]
    method = new_method(config=method_config(maximum=5, points=99))
    probability = configure_known_probabilities(method)
    provider = ImageOnlyDataset(images)
    summary = method.estimate_fisher(batcher(provider, size=2), device="cpu")
    all_pixels = torch.cat([image.reshape(3, -1).t() for image in images])
    expected_bias = probability * (1 - probability)
    expected_weight = expected_bias[:, None] * all_pixels.square().mean(dim=0)[None, :]
    assert torch.allclose(method.fisher_diagonal["classifier.bias"], expected_bias, atol=1e-10, rtol=0)
    assert torch.allclose(method.fisher_diagonal["classifier.weight"][:, :, 0, 0], expected_weight, atol=1e-10, rtol=0)
    assert torch.equal(method.fisher_diagonal["unused"], torch.zeros_like(method.fisher_diagonal["unused"]))
    assert summary["actual_images"] == 2 and summary["actual_points"] == 8
    assert summary["autograd_grad_calls"] == 24 and sorted(provider.visits) == [0, 1]
    assert all(not value.requires_grad for value in method.fisher_diagonal.values())
    return {"bias_fisher": expected_bias.tolist(), "images": 2, "points": 8, "classes": 3}


def check_counting_and_partition(_: Path) -> dict[str, object]:
    source = [torch.full((3, 2, 2), float(index + 1), dtype=torch.float64) for index in range(3)]
    cases = []
    for length, maximum, expected in [(1, 2, 1), (2, 9, 2), (3, 1, 1), (3, 2, 2)]:
        method = new_method(config=method_config(maximum=maximum, points=9))
        configure_known_probabilities(method)
        provider = ImageOnlyDataset(source[:length])
        result = method.estimate_fisher(batcher(provider, size=2), device="cpu")
        assert result["actual_images"] == expected and result["actual_points"] == expected * 4
        assert result["autograd_grad_calls"] == expected * 12
        assert len(provider.visits) == expected and len(set(provider.visits)) == expected
        cases.append((length, maximum, result["actual_images"], result["actual_points"]))

    empty = new_method(config=method_config())
    before = empty.method_state_dict()
    try:
        empty.estimate_fisher(type("B", (), {"dataset": ImageOnlyDataset([])})(), device="cpu")
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty input was accepted")
    assert_same(before, empty.method_state_dict())

    first = new_method(config=method_config(maximum=3, points=3))
    second = new_method(config=method_config(maximum=3, points=3))
    second.model.load_state_dict(first.model.state_dict())
    first_result = first.estimate_fisher(batcher(ImageOnlyDataset(source), size=1), device="cpu")
    second_result = second.estimate_fisher(batcher(ImageOnlyDataset(source), size=2), device="cpu")
    assert first_result["selected_image_indices"] == second_result["selected_image_indices"]
    assert first_result["selected_point_indices"] == second_result["selected_point_indices"]
    assert tensor_dict_equal(first.fisher_diagonal, second.fisher_diagonal)

    bad = new_method(config=method_config())
    bad_before = bad.method_state_dict()
    provider = ImageOnlyDataset([torch.full((3, 1, 1), float("nan"), dtype=torch.float64)])
    try:
        bad.estimate_fisher(batcher(provider), device="cpu")
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite input was accepted")
    assert_same(bad_before, bad.method_state_dict())
    return {"count_cases": cases, "partition_invariant": True, "empty_and_nonfinite_rejected": True}


def check_state_and_rng(path: Path) -> dict[str, object]:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model = TinyPixelModel(dropout=0.4, use_batch_norm=True).double()
    method = new_method(model=model)
    configure_known_probabilities(method)
    method.model.train()
    method.model.dropout.eval()
    parameters = dict(method.model.named_parameters())
    for index, parameter in enumerate(parameters.values()):
        parameter.grad = torch.full_like(parameter, index + 0.25)
    modes = [module.training for module in method.model.modules()]
    model_state = {name: value.detach().clone() for name, value in method.model.state_dict().items()}
    gradients = {name: value.grad.detach().clone() for name, value in parameters.items()}
    optimizer = torch.optim.Adam(parameters.values(), lr=0.01)
    first_parameter = next(iter(parameters.values()))
    optimizer.state[first_parameter] = {
        "step": torch.tensor(4.0),
        "exp_avg": torch.ones_like(first_parameter),
        "exp_avg_sq": torch.full_like(first_parameter, 2.0),
    }
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    rng_before = capture_rng_state()
    provider = ImageOnlyDataset([torch.randn(3, 2, 2, dtype=torch.float64) for _ in range(2)], consume_rng=True)
    method.estimate_fisher(batcher(provider), device="cpu")
    assert_same(rng_before, capture_rng_state())
    assert modes == [module.training for module in method.model.modules()]
    assert tensor_dict_equal(model_state, {name: value.detach() for name, value in method.model.state_dict().items()})
    assert tensor_dict_equal(gradients, {name: value.grad for name, value in parameters.items()})
    assert_same(optimizer_before, optimizer.state_dict())
    assert method.old_model is None
    assert all(value.grad is None and not value.requires_grad for value in method.fisher_diagonal.values())
    assert all(value.grad is None and not value.requires_grad for value in method.reference_parameters.values())

    state = method.method_state_dict()
    exported = state["method_statistics"]["model_fisher_ewc_state"]
    live = {name: value.detach().clone() for name, value in method.reference_parameters.items()}
    first_name = next(iter(live))
    exported["reference_parameters"][first_name].add_(10)
    assert tensor_dict_equal(live, method.reference_parameters)
    state = method.method_state_dict()
    target = new_method(model=copy.deepcopy(method.model), config=method_config())
    target.set_site_index(1)
    target.load_method_state_dict(state)
    state["method_statistics"]["model_fisher_ewc_state"]["reference_parameters"][first_name].sub_(8)
    assert not torch.equal(target.reference_parameters[first_name], state["method_statistics"]["model_fisher_ewc_state"]["reference_parameters"][first_name])

    pristine = method.method_state_dict()
    own = pristine["method_statistics"]["model_fisher_ewc_state"]
    invalid = []
    value = copy.deepcopy(pristine); del value["method_statistics"]["model_fisher_ewc_state"]["fisher_diagonal"][first_name]; invalid.append(value)
    value = copy.deepcopy(pristine); value["method_statistics"]["model_fisher_ewc_state"]["extra"] = 1; invalid.append(value)
    value = copy.deepcopy(pristine); value["method_statistics"]["model_fisher_ewc_state"]["reference_parameters"][first_name] = torch.zeros(2, dtype=torch.float64); invalid.append(value)
    value = copy.deepcopy(pristine); value["method_statistics"]["model_fisher_ewc_state"]["fisher_diagonal"][first_name] = own["fisher_diagonal"][first_name].float(); invalid.append(value)
    value = copy.deepcopy(pristine); value["method_statistics"]["model_fisher_ewc_state"]["reference_parameters"][first_name].fill_(float("nan")); invalid.append(value)
    value = copy.deepcopy(pristine); value["method_statistics"]["model_fisher_ewc_state"]["fisher_diagonal"][first_name].fill_(-1); invalid.append(value)
    value = copy.deepcopy(pristine); value["method_statistics"]["model_fisher_ewc_state"]["resolved_method_config"]["ewc_lambda"] = 9.0; invalid.append(value)
    value = copy.deepcopy(pristine); value["method_statistics"]["model_fisher_ewc_state"]["completed_consolidations"] = 4; invalid.append(value)
    for candidate in invalid:
        rejected = new_method(model=copy.deepcopy(method.model), config=method_config())
        rejected.set_site_index(1)
        before = rejected.method_state_dict()
        try:
            rejected.load_method_state_dict(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid EWC state was accepted")
        assert_same(before, rejected.method_state_dict())

    checkpoint = path / "state_checkpoint.pt"
    save_stage_checkpoint(method, checkpoint)
    checkpoint_loaded = new_method(model=copy.deepcopy(method.model), config=method_config())
    checkpoint_loaded.set_site_index(1)
    checkpoint_loaded.begin_site("B", checkpoint, 1)
    assert tensor_dict_equal(method.reference_parameters, checkpoint_loaded.reference_parameters)
    assert tensor_dict_equal(method.fisher_diagonal, checkpoint_loaded.fisher_diagonal)

    method.set_site_index(1)
    method.site_id = "B"
    old_reference = {name: value.detach().clone() for name, value in method.reference_parameters.items()}
    old_fisher = {name: value.detach().clone() for name, value in method.fisher_diagonal.items()}
    rng_before_failure = capture_rng_state()
    failure_provider = ImageOnlyDataset([torch.randn(3, 2, 2, dtype=torch.float64)], consume_rng=True, fail_at=1)
    try:
        method.estimate_fisher(batcher(failure_provider), device="cpu")
    except OSError as error:
        assert "injected" in str(error)
    else:
        raise AssertionError("injected read failure was not preserved")
    assert_same(rng_before_failure, capture_rng_state())
    assert tensor_dict_equal(old_reference, method.reference_parameters)
    assert tensor_dict_equal(old_fisher, method.fisher_diagonal)

    return {"invalid_states_rejected": len(invalid), "rng_and_modes_restored": True, "failure_atomic": True}


def check_penalty_backward_and_golden(path: Path) -> dict[str, object]:
    images = [torch.randn(3, 2, 2, dtype=torch.float64) for _ in range(2)]
    method = new_method(config=method_config(gamma=0.6, maximum=2, points=4))
    configure_known_probabilities(method)
    method.estimate_fisher(batcher(ImageOnlyDataset(images)), device="cpu")
    prior_fisher = {name: value.detach().clone() for name, value in method.fisher_diagonal.items()}
    prior_reference = {name: value.detach().clone() for name, value in method.reference_parameters.items()}
    with torch.no_grad():
        method.model.classifier.weight.add_(0.1)
    parameter_map = method._trainable_parameters()
    expected_penalty = sum(
        0.5 * (prior_fisher[name] * (parameter - prior_reference[name]).square()).sum()
        for name, parameter in parameter_map.items()
    )
    penalty = method._ewc_loss(torch.zeros((), dtype=torch.float64, requires_grad=True))
    assert torch.allclose(penalty, expected_penalty, atol=1e-12, rtol=0)
    gradients = torch.autograd.grad(penalty, tuple(parameter_map.values()))
    for (name, parameter), gradient in zip(parameter_map.items(), gradients, strict=True):
        assert torch.allclose(gradient, prior_fisher[name] * (parameter - prior_reference[name]), atol=1e-12, rtol=0)

    checkpoint = path / "stage_a.pt"
    save_stage_checkpoint(method, checkpoint)
    second = new_method(config=method_config(gamma=0.6, maximum=2, points=4))
    second.set_site_index(1)
    second.begin_site("B", checkpoint, 1)
    second_summary = second.estimate_fisher(batcher(ImageOnlyDataset(images)), device="cpu")
    fresh = new_method(model=copy.deepcopy(second.model), config=method_config(gamma=0.0, maximum=2, points=4))
    fresh.site_id = "B"
    fresh.estimate_fisher(batcher(ImageOnlyDataset(images)), device="cpu")
    expected_running = {name: 0.6 * prior_fisher[name] + fresh.fisher_diagonal[name] for name in prior_fisher}
    assert tensor_dict_equal(expected_running, second.fisher_diagonal, atol=1e-12)
    assert all(torch.equal(second.reference_parameters[name], parameter) for name, parameter in second._trainable_parameters().items())
    assert tensor_dict_equal(prior_reference, method.reference_parameters)
    assert tensor_dict_equal(prior_fisher, method.fisher_diagonal)
    assert second_summary["completed_consolidations"] == 2

    with torch.no_grad():
        second.model.classifier.bias.add_(0.05)
    labeled, unlabeled = sample_batches()
    control = SequentialSSLMethod(copy.deepcopy(second.model), config=second.config, static=False)
    result = second.training_step(labeled, unlabeled, 1, 0)
    control_result = control.training_step(labeled, unlabeled, 1, 0)
    ewc = second._ewc_loss(control_result.total_loss)
    for name in ("loss_sup", "loss_seg_ce", "loss_seg_dice", "loss_anchor_sup", "loss_assim"):
        assert torch.equal(result.losses[name], control_result.losses[name])
    assert torch.allclose(result.total_loss, control_result.total_loss + 1.7 * ewc, atol=1e-12, rtol=0)
    first_logits = second.model(labeled.image).logits.detach().clone()
    repeat = second.training_step(labeled, unlabeled, 1, 0)
    assert torch.equal(first_logits, second.model(labeled.image).logits)
    assert torch.equal(result.total_loss, repeat.total_loss)

    reference_before = {name: value.detach().clone() for name, value in second.reference_parameters.items()}
    fisher_before = {name: value.detach().clone() for name, value in second.fisher_diagonal.items()}
    model_before = {name: value.detach().clone() for name, value in second.model.state_dict().items()}
    optimizer = build_optimizer(second, lr=0.01, weight_decay=0.0)
    trainer = Trainer(second, optimizer=optimizer, scheduler=build_scheduler(optimizer, total_steps=1), device="cpu", amp=False)
    trainer.train_step(labeled, unlabeled, state=TrainerState(global_step=1, site_step=0, epoch=0))
    assert any(not torch.equal(model_before[name], value) for name, value in second.model.state_dict().items())
    assert tensor_dict_equal(reference_before, second.reference_parameters)
    assert tensor_dict_equal(fisher_before, second.fisher_diagonal)
    assert all(value.grad is None for value in second.reference_parameters.values())
    assert all(value.grad is None for value in second.fisher_diagonal.values())
    return {"lambda": 1.7, "gamma": 0.6, "penalty": float(penalty), "shared_trainer_update": True}


def sha_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def check_shared_runner_resume(path: Path) -> dict[str, object]:
    data_root = make_synthetic_root(path / "fixture", records=8)
    manifest = data_root / "manifests/training/lcrseg_v1_seed0.csv"
    rows = list(csv.DictReader(manifest.open()))
    for index, row in enumerate(rows):
        row["site_or_vendor"] = "REFUGE" if index < 4 else "RIM_ONE_r3"
        row["primary_20pct_split"] = ("train_labeled", "train_labeled", "train_unlabeled", "val")[index % 4]
        row["label_h5_relpath"] = "" if index % 4 == 2 else f"labels/fundus/SITE/case{index}.h5"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    source_hashes = sha_tree(data_root)
    config = ContinualRunner.default_config(
        data_root=data_root,
        run_root=path / "runs",
        dataset="fundus",
        method_name="model_fisher_ewc_v1",
        seed=0,
        site_order=("REFUGE", "RIM_ONE_r3"),
        run_name="reference",
        device="cpu",
    )
    config["data"].update(require_readonly=False, evaluation_sites=["REFUGE", "RIM_ONE_r3"], evaluation_role="val")
    config["training"].update(steps_per_site=4, amp=False, gradient_cosine_interval=0, checkpoint_interval_steps=2)
    config["method"].update(version="1.0", **method_config(gamma=0.6, maximum=2, points=2, seed=271828))

    def toy_model(*args, **kwargs):
        return TinyPixelModel(*args, dropout=0.2, use_batch_norm=True, **kwargs)

    import lcrseg.engine.continual_runner as runner_module

    with mock.patch.object(runner_module, "UNet2D", toy_model):
        assert ContinualRunner(config).run()["completed_global_steps"] == 8
        resumed_config = json.loads(json.dumps(config))
        resumed_config["experiment"]["run_name"] = "resumed"
        resumed_config["training"]["max_steps_this_invocation"] = 6
        assert ContinualRunner(resumed_config).run()["status"] == "interrupted"
        resumed_config["training"]["max_steps_this_invocation"] = None
        assert ContinualRunner(resumed_config).run(
            resume_checkpoint=path / "runs/resumed/checkpoint_last.pt"
        )["status"] == "complete"
    reference = torch.load(path / "runs/reference/checkpoint_final.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(path / "runs/resumed/checkpoint_final.pt", map_location="cpu", weights_only=False)
    assert reference["method_name"] == resumed["method_name"] == "model_fisher_ewc_v1"
    assert reference["method_version"] == resumed["method_version"] == "1.0"
    for key in reference:
        if key == "config_resolved":
            continue
        assert_same(reference[key], resumed[key])
    ewc_state = reference["method_statistics"]["model_fisher_ewc_state"]
    assert ewc_state["completed_consolidations"] == 2
    assert not reference["method_statistics"]["old_model_state"]
    assert source_hashes == sha_tree(data_root)
    return {"full_updates": 8, "interrupted_after": 6, "resumed_updates": 2, "exact_checkpoint_match": True}


def check_two_case_overfit(_: Path) -> dict[str, object]:
    torch.manual_seed(314159)
    method = new_method(dtype=torch.float32, config=method_config())
    labeled, unlabeled = sample_batches(torch.float32)
    first_output = method.model(labeled.image)
    first = float(method._supervised_losses(first_output, labeled)["loss_sup"].detach())
    optimizer = build_optimizer(method, lr=0.1, weight_decay=0.0)
    trainer = Trainer(method, optimizer=optimizer, scheduler=build_scheduler(optimizer, total_steps=100), device="cpu", amp=False)
    for step in range(100):
        trainer.train_step(labeled, unlabeled, state=TrainerState(global_step=step, site_step=step, epoch=0))
    final_output = method.model(labeled.image)
    final = float(method._supervised_losses(final_output, labeled)["loss_sup"].detach())
    accuracy = float(final_output.logits.argmax(1).eq(labeled.label).float().mean())
    assert final <= 0.1
    assert final <= first * 0.1
    assert accuracy >= 0.98
    return {"updates": 100, "initial_supervised_loss": first, "final_supervised_loss": final, "pixel_accuracy": accuracy}


GROUPS = [
    ("closed_form_fisher", check_closed_form),
    ("counting_and_partition", check_counting_and_partition),
    ("state_and_rng_isolation", check_state_and_rng),
    ("penalty_backward_and_golden", check_penalty_backward_and_golden),
    ("shared_runner_resume", check_shared_runner_resume),
    ("two_case_overfit", check_two_case_overfit),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=False)
    if args.result.exists():
        raise FileExistsError(args.result)
    budget = Budget()
    groups = []
    started = time.perf_counter()
    budget_failure = False
    with budget.instrument():
        for name, function in GROUPS:
            before = dict(budget.counts)
            output = args.scratch / name
            output.mkdir(exist_ok=False)
            try:
                details = function(output)
                status, failure = "PASS", None
            except BaseException as error:
                status = "FAIL"
                failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
                details = None
                budget_failure = budget_failure or isinstance(error, BudgetExceeded)
            groups.append(
                {
                    "name": name,
                    "status": status,
                    "details": details,
                    "failure": failure,
                    "counts": {key: budget.counts[key] - before[key] for key in budget.counts},
                }
            )
            if budget_failure:
                break
    elapsed = time.perf_counter() - started
    for name, _ in GROUPS[len(groups) :]:
        groups.append({"name": name, "status": "NOT_RUN_BUDGET_EXHAUSTED", "details": None, "failure": None, "counts": {key: 0 for key in LIMITS}})
    passed = len(groups) == len(GROUPS) and all(group["status"] == "PASS" for group in groups) and elapsed <= 300
    result = {
        "schema_version": 1,
        "status": "PASS_SYNTHETIC_ENGINEERING" if passed else "FAIL_SYNTHETIC_ENGINEERING",
        "groups": groups,
        "groups_passed": sum(group["status"] == "PASS" for group in groups),
        "groups_total": len(GROUPS),
        "groups_skipped": sum(group["status"].startswith("NOT_RUN") for group in groups),
        "budget_limits": LIMITS,
        "actual_counts": budget.counts,
        "elapsed_seconds": elapsed,
        "wall_time_limit_seconds": 300,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "real_data_access": False,
        "real_model_forwards": 0,
        "gpu_used": False,
        "external_source_executed": False,
        "optimizer_updates_are_synthetic_only": True,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with args.result.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
