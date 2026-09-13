"""Regression expectations for b909a913 external review (CPU synthetic only).

Run in the repository root:
    PYTHONPATH=. python -m pytest -q /path/to/test_external_review_R1.py

On the reviewed commit these expectations intentionally expose defects. Do not
turn them into xfail/skip or weaken assertions instead of repairing the code.
The environment prefix supports the reviewer's SHA-verified local source copy.
ToyStructure is explicitly an exception-contract fixture, NOT a CWMI substitute.
"""
import copy
import importlib
import os

import numpy as np
import pytest
import torch

PREFIX = os.environ.get("SSLCL_REVIEW_IMPORT_PREFIX", "experiments.lcrseg.five_frameworks_v1")
Model = importlib.import_module(PREFIX + ".model").Model
SyntheticParentBridge = importlib.import_module(PREFIX + ".parent_bridge").SyntheticParentBridge
SyntheticCurrentDomain = importlib.import_module(PREFIX + ".recipes").SyntheticCurrentDomain
StageTrainer = importlib.import_module(PREFIX + ".train_stage").StageTrainer
checkpoint = importlib.import_module(PREFIX + ".checkpoint")
segmentation_metrics = importlib.import_module(PREFIX + ".evaluate").segmentation_metrics

OPTIONS = {"total_steps": 5, "PAS_confidence": 0., "PAS_cosine": -1.,
           "lambda_U": .5, "lambda_JML": .25}
IDENTITY = {"family": "F1", "candidate_id": "F1_SYNTHETIC", "seed": 161,
            "order": 1, "stage": 1}

@pytest.fixture(autouse=True)
def deterministic_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def make_trainer(initialize=True, *, options=None, stage=1, parent=None, previous=None):
    torch.manual_seed(4)
    if parent is None:
        parent = SyntheticParentBridge()
    return StageTrainer(Model(parent, "F1", previous=previous),
                        SyntheticCurrentDomain(seed=161, order=1, stage=stage),
                        {**OPTIONS, **(options or {})}, initialize=initialize)


def test_R00_existing_stage1_resume_control(tmp_path):
    a = make_trainer()
    a.update(); a.update()
    checkpoint.save(a, tmp_path / "control.pt", IDENTITY)
    b = make_trainer(False)
    checkpoint.restore(b, tmp_path / "control.pt", IDENTITY)
    a.update(); b.update()
    for key, value in a.model.state_dict().items():
        assert torch.equal(value, b.model.state_dict()[key]), key


def test_R01_stage2_resume_restores_trainability_and_trajectory(tmp_path):
    source = make_trainer()
    source.update(); source.update()
    deployed = source.model.deploy()
    entry_parent = copy.deepcopy(deployed.parent)
    entry_transform = deployed.transform.detach().clone()
    a = make_trainer(stage=2, parent=copy.deepcopy(entry_parent), previous=entry_transform)
    a.update(); a.update()
    identity = {**IDENTITY, "stage": 2}
    checkpoint.save(a, tmp_path / "stage2.pt", identity)
    b = make_trainer(False, stage=2, parent=copy.deepcopy(entry_parent), previous=entry_transform)
    checkpoint.restore(b, tmp_path / "stage2.pt", identity)
    expected = {name: p.requires_grad for name, p in a.model.named_parameters()}
    actual = {name: p.requires_grad for name, p in b.model.named_parameters()}
    assert actual == expected, "resume left the sealed parent's A/B frozen"
    a.update(); b.update()
    for key, value in a.model.state_dict().items():
        assert torch.equal(value, b.model.state_dict()[key]), key


def test_R02a_resume_rejects_changed_semantic_options(tmp_path):
    a = make_trainer(); a.update(); a.update()
    checkpoint.save(a, tmp_path / "config.pt", IDENTITY)
    b = make_trainer(False, options={"lambda_U": 0.})
    with pytest.raises((ValueError, RuntimeError)):
        checkpoint.restore(b, tmp_path / "config.pt", IDENTITY)


def test_R02b_resume_rejects_provider_identity_mismatch(tmp_path):
    a = make_trainer(); a.update()
    checkpoint.save(a, tmp_path / "provider.pt", IDENTITY)
    b = make_trainer(False, stage=2)
    with pytest.raises((ValueError, RuntimeError)):
        checkpoint.restore(b, tmp_path / "provider.pt", IDENTITY)


def test_R03a_nonfinite_loss_with_finite_gradient_cannot_commit():
    class BadLoss(SyntheticParentBridge):
        def supervised(self, logp, labels):
            return super().supervised(logp, labels) + logp.new_tensor(float("inf"))
    t = make_trainer(parent=BadLoss())
    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        t.update()
    assert t.step == 0 and t.cursor == 0
    assert t.telemetry["synthetic_optimizer_updates"] == 0


def test_R03b_nonfinite_post_constraint_state_cannot_commit():
    class BadConstraint(SyntheticParentBridge):
        @torch.no_grad()
        def apply_constraints(self):
            self.adapters[-1].b[0, 0] = float("inf")
    t = make_trainer(parent=BadConstraint())
    teacher_before = copy.deepcopy(t.ema.state_dict())
    physical = []
    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        t.update(physical=physical.append)
    assert physical == [1], "an already executed optimizer call is physical cost"
    assert t.step == 0 and t.cursor == 0
    for key, value in teacher_before.items():
        assert torch.equal(value, t.ema.state_dict()[key]), "invalid student propagated into EMA"


def test_R04_invalid_projector_is_not_rank_fallback():
    class ToyStructure:
        def __call__(self, p, y):
            return ((p[:, 1] - (y == 1).to(p)).square() * (y != 255)).mean()
    torch.manual_seed(6)
    parent = SyntheticParentBridge()
    for adapter in list(parent.adapters)[-2:]:
        adapter.free_projector = 2 * torch.eye(parent.d)
    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        StageTrainer(Model(parent, "F2"), SyntheticCurrentDomain(size=16),
                     {"lambda_structure": 1.}, cwmi=ToyStructure())


def test_R05_teacher_features_use_teacher_bridge_mode():
    class ModeParent(SyntheticParentBridge):
        def __init__(self):
            super().__init__(); self.seen = []
        def features(self, x, mode="student", overrides=None):
            self.seen.append(mode)
            return super().features(x, mode, overrides)
    t = make_trainer(parent=ModeParent())
    assert t.ema.parent.seen and set(t.ema.parent.seen) == {"teacher"}


def test_R06_ignored_pixels_cannot_change_segmentation_dice():
    target = np.array([[1, 255], [0, 0]])
    a = np.array([[1, 0], [0, 0]])
    b = np.array([[1, 1], [0, 0]])
    ma, mb = segmentation_metrics(a, target), segmentation_metrics(b, target)
    for name in ("rim", "cup", "disc_union"):
        assert ma[name]["Dice"] == mb[name]["Dice"], name
