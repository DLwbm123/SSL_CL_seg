from __future__ import annotations

import torch

from lcrseg.contracts import LabeledBatch, UnlabeledBatch
from lcrseg.engine.checkpoint import checkpoint_payload, load_checkpoint, save_checkpoint
from lcrseg.engine.trainer import Trainer, TrainerState, build_optimizer, build_scheduler
from lcrseg.methods.components.compatibility import zero_compatibility
from lcrseg.methods.components.learnability import LearnabilityOutput
from lcrseg.methods.components.pseudo_label import PseudoLabelOutput
from lcrseg.methods.components.relation_field import RelationOutput
from lcrseg.methods.components.routing import assimilation_loss, relation_consolidation_loss
from lcrseg.methods.lcrseg_v0_1 import LCRSegV01Method
from lcrseg.methods.ewc import EWCSegMethod
from lcrseg.methods import resolve_method_config
from lcrseg.models import UNet2D


def _batches() -> tuple[LabeledBatch, UnlabeledBatch]:
    torch.manual_seed(123)
    image = torch.randn(2, 3, 32, 32)
    label = torch.zeros((2, 32, 32), dtype=torch.long)
    label[:, 3:16, 3:16] = 1
    label[:, 17:29, 17:29] = 2
    labeled = LabeledBatch(
        image=image,
        label=label,
        valid_mask=torch.ones((2, 1, 32, 32), dtype=torch.bool),
        case_id=["a", "b"], patient_id=["a", "b"], site=["A", "A"], slice_index=[None, None],
    )
    unlabeled = UnlabeledBatch(
        weak_image=image.clone(),
        strong_image=image * 1.15 + 0.05,
        strong_valid_mask=torch.ones((2, 1, 32, 32), dtype=torch.bool),
        case_id=["u", "v"], patient_id=["u", "v"], site=["A", "A"], slice_index=[None, None], geometry_record=[{}, {}],
    )
    return labeled, unlabeled


def _method() -> LCRSegV01Method:
    return LCRSegV01Method(
        UNet2D(3, 3),
        config={
            "anchor_min_support_pixels": 1,
            "anchor_max_pixels_per_class": 128,
            "anchor_bootstrap_steps": 0,
            "background_boundary_exclusion": 0,
            "tau_cls": 0.0,
            "tau_anchor": 0.0,
            "delta_anchor": 0.0,
            "tau_spatial": 0.0,
            "assim_ramp_steps": 1,
            "relation_ramp_steps": 1,
            "min_rank_pixels": 1,
        },
    )


def _trainer(method: LCRSegV01Method, steps: int) -> Trainer:
    optimizer = build_optimizer(method, lr=1e-3, weight_decay=0.0)
    return Trainer(method, optimizer=optimizer, scheduler=build_scheduler(optimizer, total_steps=steps), device="cpu", amp=False)


def test_empty_assimilation_relation_and_cutout_are_safe_and_differentiable() -> None:
    logits = torch.randn(1, 3, 8, 8, requires_grad=True)
    labels = torch.full((1, 2, 2), -100, dtype=torch.long)
    pseudo = PseudoLabelOutput(
        labels=labels,
        valid=torch.zeros((1, 1, 2, 2), dtype=torch.bool),
        source=torch.zeros((1, 2, 2), dtype=torch.long),
        source_weight=torch.zeros((1, 1, 2, 2)),
        spatial_weight=torch.zeros((1, 1, 2, 2)),
        spatial_agreement=torch.zeros((1, 1, 2, 2)),
    )
    learnability = LearnabilityOutput(
        score=torch.zeros((1, 1, 2, 2)), robust_progress_index=torch.zeros((1, 1, 2, 2)), percentile_rank=torch.zeros((1, 1, 2, 2)),
        progress_weight=torch.zeros((1, 1, 2, 2)), relation_weight=torch.zeros((1, 1, 2, 2)), spatial_weight=torch.zeros((1, 1, 2, 2)), source_weight=torch.zeros((1, 1, 2, 2)),
    )
    loss = assimilation_loss(logits, pseudo, learnability, torch.ones((1, 1, 8, 8), dtype=torch.bool))
    assert float(loss) == 0.0
    loss.backward()
    assert logits.grad is not None

    labels = torch.ones((1, 2, 2), dtype=torch.long)
    pseudo = PseudoLabelOutput(labels, torch.ones((1, 1, 2, 2), dtype=torch.bool), torch.ones((1, 2, 2), dtype=torch.long), torch.ones((1, 1, 2, 2)), torch.ones((1, 1, 2, 2)), torch.ones((1, 1, 2, 2)))
    learnability = LearnabilityOutput(*(torch.ones((1, 1, 2, 2)) for _ in range(7)))
    mask = torch.ones((1, 1, 8, 8), dtype=torch.bool)
    mask[:, :, 2:6, 2:6] = False
    first = assimilation_loss(logits.detach(), pseudo, learnability, mask)
    changed = logits.detach().clone()
    changed[:, :, 2:6, 2:6] = 1000.0
    second = assimilation_loss(changed, pseudo, learnability, mask)
    assert torch.allclose(first, second, atol=1e-6)

    probability = torch.tensor([[[[0.2, 0.2]], [[0.8, 0.8]]]])
    relation = RelationOutput(
        logits=probability.log(), probabilities=probability, predicted_class=probability.argmax(1),
        top1=probability.max(1, keepdim=True).values, top2=probability.min(1, keepdim=True).values,
        margin=probability.max(1, keepdim=True).values - probability.min(1, keepdim=True).values,
        valid_class_mask=torch.ones(2, dtype=torch.bool),
    )
    compatibility = zero_compatibility(probability)
    relation_zero = relation_consolidation_loss(relation, relation, compatibility, torch.ones((1, 1, 8, 8), dtype=torch.bool), distill_temperature=0.5)
    assert float(relation_zero) == 0.0
    identity_compatibility = type(compatibility)(
        score=torch.ones_like(compatibility.score), js_divergence=compatibility.js_divergence,
        old_margin_weight=torch.ones_like(compatibility.score), agreement=torch.ones_like(compatibility.score), spatial_weight=torch.ones_like(compatibility.score),
    )
    identity = relation_consolidation_loss(relation, relation, identity_compatibility, torch.ones((1, 1, 8, 8), dtype=torch.bool), distill_temperature=0.5)
    assert float(identity) < 1e-6


def test_lcr_incremental_old_state_is_frozen_current_anchor_updates_and_checkpoint_recovers(tmp_path) -> None:
    labeled, unlabeled = _batches()
    first = _method()
    first.begin_site("A", None, 1)
    first_trainer = _trainer(first, 1)
    first_trainer.train_step(labeled, unlabeled, state=TrainerState(global_step=0, site_step=0, epoch=0))
    assert first.current_anchor_bank.all_classes_valid
    assert first.bootstrap_state["complete"]
    first_state = first.method_state_dict()
    checkpoint = checkpoint_payload(
        method_name=first.method_name,
        method_version=first.method_version,
        git_commit="NO_GIT_WORKTREE",
        config_resolved={"seed": 123}, site_id="A", site_index=0, epoch=0, site_step=1, global_step=1,
        current_model_state=first.model.state_dict(), optimizer_state=first_trainer.optimizer.state_dict(),
        scheduler_state=first_trainer.scheduler.state_dict(), scaler_state=first_trainer.scaler.state_dict(),
        current_anchor_state=first_state["current_anchor_state"], historical_anchor_state=first_state["historical_anchor_state"],
        bootstrap_state=first_state["bootstrap_state"], method_statistics=first_state["method_statistics"],
        data_split_hash="split", manifest_hash="manifest",
    )
    path = tmp_path / "first.pt"
    save_checkpoint(path, checkpoint)
    restored = load_checkpoint(path)
    assert restored["current_anchor_state"]

    second = _method()
    second.begin_site("B", path, 1)
    assert second.old_model is not None and second.old_anchor_bank is not None
    assert second.model.enc1.block[0].weight.data_ptr() != second.old_model.enc1.block[0].weight.data_ptr()
    assert second.current_anchor_bank.anchors.data_ptr() != second.old_anchor_bank.anchors.data_ptr()
    old_anchor = second.old_anchor_bank.anchors.detach().clone()
    current_before = second.current_anchor_bank.anchors.detach().clone()
    second_trainer = _trainer(second, 1)
    result = second_trainer.train_step(labeled, unlabeled, state=TrainerState(global_step=1, site_step=0, epoch=0))
    assert all(parameter.grad is None for parameter in second.old_model.parameters())
    assert torch.equal(second.old_anchor_bank.anchors, old_anchor)
    assert not torch.equal(second.current_anchor_bank.anchors, current_before)
    assert result.losses["loss_relation"].requires_grad
    assert any(parameter.grad is not None for parameter in second.model.projection_head.parameters())
    assert any(parameter.grad is not None for parameter in second.model.segmentation_head.parameters())
    assert any(parameter.grad is not None for parameter in second.model.enc1.parameters())
    second.end_site("B")


def test_ss_ewc_is_a_separate_baseline_with_diagonal_fisher() -> None:
    labeled, unlabeled = _batches()
    method = EWCSegMethod(UNet2D(3, 3), config={"ewc_lambda": 0.5, "ewc_fisher_batches": 1}, static=False)
    method.begin_site("A", None, 1)

    class Batcher:
        steps_per_epoch = 1

        @staticmethod
        def batch_at(index: int) -> LabeledBatch:
            return labeled

    summary = method.estimate_fisher(Batcher(), device="cpu")
    assert summary["fisher_batches"] == 1.0
    assert method.fisher_diagonal and method.reference_parameters
    zero = method._ewc_loss(torch.zeros((), requires_grad=True))
    assert float(zero) == 0.0
    with torch.no_grad():
        next(method.model.parameters()).add_(0.01)
    assert float(method._ewc_loss(torch.zeros((), requires_grad=True))) > 0.0


def test_anchor_bootstrap_warms_up_before_any_anchor_write() -> None:
    labeled, unlabeled = _batches()
    method = LCRSegV01Method(
        UNet2D(3, 3),
        config={
            "anchor_bootstrap_steps": 2,
            "anchor_min_support_pixels": 1,
            "background_boundary_exclusion": 0,
            "anchor_max_pixels_per_class": 128,
        },
    )
    method.begin_site("A", None, 3)
    trainer = _trainer(method, 3)
    for step in range(2):
        trainer.train_step(labeled, unlabeled, state=TrainerState(global_step=step, site_step=step, epoch=0))
        assert not method.current_anchor_bank.valid.any()
    trainer.train_step(labeled, unlabeled, state=TrainerState(global_step=2, site_step=2, epoch=0))
    assert method.current_anchor_bank.all_classes_valid
    assert method.bootstrap_state["complete"]


def test_lcr_ablation_switches_keep_weights_detached_and_uniform(tmp_path) -> None:
    """The requested no-L_i/uniform-C ablations must not change default V0.1."""

    labeled, unlabeled = _batches()
    first = _method()
    first.begin_site("A", None, 1)
    first_trainer = _trainer(first, 1)
    first_trainer.train_step(labeled, unlabeled, state=TrainerState(global_step=0, site_step=0, epoch=0))
    first_state = first.method_state_dict()
    checkpoint = checkpoint_payload(
        method_name=first.method_name,
        method_version=first.method_version,
        git_commit="NO_GIT_WORKTREE",
        config_resolved={"seed": 123}, site_id="A", site_index=0, epoch=0, site_step=1, global_step=1,
        current_model_state=first.model.state_dict(), optimizer_state=first_trainer.optimizer.state_dict(),
        scheduler_state=first_trainer.scheduler.state_dict(), scaler_state=first_trainer.scaler.state_dict(),
        current_anchor_state=first_state["current_anchor_state"], historical_anchor_state=first_state["historical_anchor_state"],
        bootstrap_state=first_state["bootstrap_state"], method_statistics=first_state["method_statistics"],
        data_split_hash="split", manifest_hash="manifest",
    )
    checkpoint_path = tmp_path / "previous.pt"
    save_checkpoint(checkpoint_path, checkpoint)
    ablated = LCRSegV01Method(
        UNet2D(3, 3),
        config={
            "anchor_bootstrap_steps": 0,
            "anchor_min_support_pixels": 1,
            "background_boundary_exclusion": 0,
            "anchor_max_pixels_per_class": 128,
            "tau_cls": 0.0,
            "tau_anchor": 0.0,
            "delta_anchor": 0.0,
            "tau_spatial": 0.0,
            "assim_ramp_steps": 1,
            "relation_ramp_steps": 1,
            "min_rank_pixels": 1,
            "use_learnability": False,
            "use_compatibility": False,
        },
    )
    ablated.begin_site("B", checkpoint_path, 1)
    result = _trainer(ablated, 1).train_step(labeled, unlabeled, state=TrainerState(global_step=1, site_step=0, epoch=0))
    assert result.maps is not None
    learnability = result.maps["learnability"]
    compatibility = result.maps["compatibility"]
    assert not learnability.requires_grad and not compatibility.requires_grad
    assert torch.all(learnability.eq(1.0))
    assert torch.all(compatibility.eq(1.0))


def test_lcr_resolved_config_persists_default_method_contract() -> None:
    resolved = resolve_method_config("lcrseg_v0_1", {"anchor_bootstrap_steps": 20})
    assert resolved["anchor_bootstrap_steps"] == 20
    assert resolved["anchor_k"] == 1
    assert resolved["use_learnability"] is True
    assert resolved["use_compatibility"] is True
    assert resolved["lambda_assim"] == 1.0
    assert resolved["lambda_relation"] == 1.0
    ewc = resolve_method_config("ss_ewc")
    assert ewc["ewc_lambda"] == 0.1
    assert ewc["ewc_fisher_batches"] == 8
