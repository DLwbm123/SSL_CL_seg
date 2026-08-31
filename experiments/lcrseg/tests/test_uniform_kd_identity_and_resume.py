"""Baseline identities and complete incremental checkpoint continuation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from lcrseg.data import collate_labeled, collate_unlabeled
from lcrseg.engine.checkpoint import load_checkpoint
from lcrseg.engine.continual_runner import ContinualRunner
from lcrseg.methods import build_method
from lcrseg.methods.base import model_checksum
from lcrseg.models import UNet2D
from scripts.verify_resume_equivalence import compare
from .conftest import make_synthetic_root


@pytest.mark.parametrize(("name", "expected"), [
    ("sequential_ssl", "sequential_ssl"), ("static_ssl", "static_ssl"),
    ("uniform_kd", "uniform_kd"), ("lwf", "uniform_kd"),
    ("ss_ewc", "ss_ewc"), ("joint_ssl", "joint_ssl"),
])
def test_factory_preserves_method_identity(name, expected):
    assert build_method(name, UNet2D(3, 3)).method_name == expected


def test_uniform_kd_temperature_mask_and_teacher_gradient():
    method = build_method("uniform_kd", UNet2D(3, 3), config={"uniform_kd_temperature": 2.0})
    current = torch.tensor([[[[2., 90.]], [[0., -90.]], [[-2., 0.]]]], requires_grad=True)
    old = torch.zeros_like(current, requires_grad=True)
    mask = torch.tensor([[[[True, False]]]])
    loss = method._uniform_kd_loss(current, old, mask)
    # Uniform teacher: KL = -log(3) - mean(log(student probability)).
    expected = 4 * (-torch.tensor(3.).log() - torch.log_softmax(current[0, :, 0, 0] / 2, 0).mean())
    assert torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    assert old.grad is None
    assert torch.count_nonzero(current.grad[..., 1]) == 0
    assert torch.isfinite(current.grad).all() and current.grad[..., 0].abs().sum() > 0


@pytest.mark.parametrize("arm", ["sequential_ssl", "uniform_kd"])
def test_two_domain_runner_resume_and_golden(tmp_path, arm):
    root = make_synthetic_root(tmp_path, records=8)
    manifest = root / "manifests/training/lcrseg_v1_seed0.csv"
    rows = list(csv.DictReader(manifest.open()))
    for index, row in enumerate(rows):
        row["site_or_vendor"] = "REFUGE" if index < 4 else "RIM_ONE_r3"
        row["primary_20pct_split"] = ("train_labeled", "train_labeled", "train_unlabeled", "val")[index % 4]
        row["label_h5_relpath"] = "" if index % 4 == 2 else f"labels/fundus/SITE/case{index}.h5"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    config = ContinualRunner.default_config(
        data_root=root, run_root=tmp_path / "runs", dataset="fundus", method_name=arm,
        seed=0, site_order=("REFUGE", "RIM_ONE_r3"), run_name="reference", device="cpu")
    config["data"].update(require_readonly=False, evaluation_sites=["REFUGE", "RIM_ONE_r3"], evaluation_role="val")
    config["training"].update(steps_per_site=4, amp=False, gradient_cosine_interval=0, checkpoint_interval_steps=2)
    config["method"].update(uniform_kd_temperature=2.0, tau_cls=0.0)
    assert ContinualRunner(config).run()["completed_global_steps"] == 8
    interrupted = json.loads(json.dumps(config))
    interrupted["experiment"]["run_name"] = "resumed"
    interrupted["training"]["max_steps_this_invocation"] = 6
    assert ContinualRunner(interrupted).run()["status"] == "interrupted"
    interrupted["training"]["max_steps_this_invocation"] = None
    assert ContinualRunner(interrupted).run(resume_checkpoint=tmp_path / "runs/resumed/checkpoint_last.pt")["status"] == "complete"
    reference = load_checkpoint(tmp_path / "runs/reference/checkpoint_final.pt")
    resumed = load_checkpoint(tmp_path / "runs/resumed/checkpoint_final.pt")
    assert reference["method_name"] == resumed["method_name"] == arm
    for key in reference:
        if key == "config_resolved":  # Only the output run name differs.
            continue
        assert compare(reference[key], resumed[key], atol=1e-6, rtol=1e-6)[0], key
    runner = ContinualRunner(config)
    method = runner._build_method()
    method.model.load_state_dict(resumed["current_model_state"])
    method.load_method_state_dict(resumed)
    labeled, unlabeled = runner._datasets(("RIM_ONE_r3",))
    batch_l = collate_labeled([labeled[0], labeled[1]])
    batch_u = collate_unlabeled([unlabeled[0]])
    checksum = model_checksum(method.model)
    old_checksum = model_checksum(method.old_model) if method.old_model is not None else None
    with torch.no_grad():
        first = method.training_step(batch_l, batch_u, 8, 4)
        logits = method.model(batch_l.image).logits.clone()
        second = method.training_step(batch_l, batch_u, 8, 4)
        assert torch.equal(logits, method.model(batch_l.image).logits)
    for key in first.losses:
        assert torch.allclose(first.losses[key], second.losses[key], atol=1e-6, rtol=1e-6)
    assert checksum == model_checksum(method.model)
    if old_checksum is not None:
        assert old_checksum == model_checksum(method.old_model)
    method.assert_old_state_unchanged()
