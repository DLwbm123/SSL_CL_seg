from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from .checkpoint import build_checkpoint, load_checkpoint, save_checkpoint
from .config import resolved_config_hash, validate_gate0_config
from .data import LCRSegH5Dataset, batch_indices, collate
from .manifest import LCRSegManifestAdapter
from .metrics import ConfusionMetrics, write_json, write_lower_triangular_csv
from .modeling import (
    assert_complete_classifier_load,
    build_mean_teacher,
    compute_single_prototypes,
    restore_gas_state,
    update_gas_from_supervised_gradient,
    upstream_pas_labels,
)
from .provenance import assert_upstream_unchanged


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # PyTorch 2.2.1 has no declared deterministic CUDA implementation for
    # nll_loss2d. Keep deterministic cuDNN/CUBLAS settings and surface a
    # warning for that kernel; resume equivalence is judged with the frozen
    # numerical tolerance in the Gate 0 config.
    torch.use_deterministic_algorithms(True, warn_only=True)


def _finite_scalar(name: str, value: torch.Tensor) -> None:
    if value.numel() != 1 or not torch.isfinite(value).item():
        raise FloatingPointError(f"non-finite {name}: {value.detach().cpu()}")


class Gate0RepairedRunner:
    """One config-driven fixed-class stage machine, separate from upstream scripts."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        config: dict[str, Any],
        protocol: dict[str, Any],
        seed: int,
        output_dir: str | Path,
        device: str | torch.device = "cuda",
        model_factory: Callable[[], nn.Module] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.protocol = protocol
        self.config = validate_gate0_config(config, protocol)
        self.config_hash = resolved_config_hash(self.config)
        self.seed = int(seed)
        if self.seed not in tuple(int(value) for value in self.config["seeds"]):
            raise ValueError(f"seed {seed} is not preregistered")
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.model_factory = model_factory
        self.training = self.config["training"]
        self.data_config = self.config["data"]
        self.model_config = self.config["model"]
        configured_reference_root = Path(self.model_config["reference_root"])
        self.reference_root = (
            configured_reference_root.resolve()
            if configured_reference_root.is_absolute()
            else (self.repo_root / configured_reference_root).resolve()
        )
        self.domain_order = list(self.data_config["domain_order"])
        self.num_classes = int(self.model_config["num_classes"])
        self.ignore_label = int(self.training["ignore_label"])
        stored_hw = self.protocol["benchmarks"][self.config["benchmark"]]["spatial_preprocessing"]["stored_resize_hw"]
        self.output_hw = (int(stored_hw[0]), int(stored_hw[1]))
        self.adapter = LCRSegManifestAdapter(
            self.data_config["root"], protocol, seed=self.seed, benchmark=self.config["benchmark"]
        )
        self.adapter_audit = self.adapter.leakage_audit()
        assert_upstream_unchanged(self.reference_root, self.model_config["upstream_path"])

        seed_everything(self.seed)
        self.wrapper = build_mean_teacher(
            self.reference_root,
            upstream_path=self.model_config["upstream_path"],
            input_channels=int(self.model_config["input_channels"]),
            num_classes=self.num_classes,
            device=self.device,
            factory=self.model_factory,
        )
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.optimizer, self.scheduler = self._new_optimizer_scheduler()
        self.stage_state = {
            "stage_index": 0,
            "epoch": 0,
            "global_step": 0,
            "epoch_lr_initialized": False,
        }
        self.sampler_state = {"stage_index": 0, "epoch": 0, "phase": "supervised", "next_batch": 0}
        self.prototypes: torch.Tensor | None = None
        self.best_metric = -math.inf
        self.matrices: dict[str, dict[str, dict[str, float]]] = {
            "mean_iou": {},
            "mean_dice": {},
            "mean_foreground_dice": {},
        }
        self._write_static_metadata()

    def _new_optimizer_scheduler(self):
        optimizer = torch.optim.Adam(
            [parameter for parameter in self.wrapper.student.parameters() if parameter.requires_grad],
            lr=float(self.training["lr"]),
            weight_decay=float(self.training["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1.0 - min(float(epoch), float(self.training["epochs_per_stage"])) / float(self.training["epochs_per_stage"]))
            ** float(self.training["scheduler_power"]),
        )
        self.wrapper.assert_optimizer_excludes_teacher(optimizer)
        return optimizer, scheduler

    def _write_static_metadata(self) -> None:
        (self.output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8"
        )
        write_json(self.output_dir / "leakage_preflight.json", self.adapter_audit)
        write_json(
            self.output_dir / "run_metadata.json",
            {
                "seed": self.seed,
                "benchmark": self.config["benchmark"],
                "domain_order": self.domain_order,
                "config_hash": self.config_hash,
                "method_registered": False,
                "constant_patch_classifier_regularization": False,
                "hidden_gt_training_usage": "none",
                "upstream_commit": self.config["upstream_commit"],
                "jascl_reference_root": str(self.reference_root),
                "device": str(self.device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
        )

    def _datasets(self, domain: str):
        labeled_records = self.adapter.records(domain=domain, role="train_labeled", purpose="train")
        unlabeled_records = self.adapter.records(domain=domain, role="train_unlabeled", purpose="train")
        self.adapter.assert_current_domain_only([*labeled_records, *unlabeled_records], domain)
        labeled = LCRSegH5Dataset(
            self.data_config["root"], labeled_records, require_label=True, output_hw=self.output_hw, augment=True
        )
        unlabeled = LCRSegH5Dataset(
            self.data_config["root"], unlabeled_records, require_label=False, output_hw=self.output_hw, augment=False
        )
        return labeled, unlabeled

    def _evaluation_dataset(self, domain: str, role: str):
        records = self.adapter.records(domain=domain, role=role, purpose="evaluate")
        return LCRSegH5Dataset(
            self.data_config["root"], records, require_label=True, output_hw=self.output_hw, augment=False
        )

    def _append_log(self, payload: dict[str, Any]) -> None:
        with (self.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _checkpoint_payload(self) -> dict[str, Any]:
        return build_checkpoint(
            wrapper=self.wrapper,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            stage_state=self.stage_state,
            sampler_state=self.sampler_state,
            prototypes=self.prototypes,
            config_hash=self.config_hash,
            evaluation_matrices=self.matrices,
            best_metric=self.best_metric,
        )

    def _save_last(self) -> Path:
        path = self.output_dir / "last.pt"
        save_checkpoint(path, self._checkpoint_payload())
        return path

    def resume(self, checkpoint_path: str | Path) -> None:
        payload = load_checkpoint(
            checkpoint_path,
            wrapper=self.wrapper,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            expected_config_hash=self.config_hash,
            restore_rng=True,
        )
        self.stage_state = dict(payload["stage_state"])
        self.sampler_state = dict(payload["sampler_state"])
        self.prototypes = None if payload["prototypes"] is None else payload["prototypes"].to(self.device)
        self.matrices = payload["evaluation_matrices"]
        self.best_metric = float(payload["best_metric"])
        self.wrapper.assert_optimizer_excludes_teacher(self.optimizer)

    def _set_epoch_lr(self, epoch: int) -> None:
        self.scheduler.step(epoch)
        self.stage_state["epoch_lr_initialized"] = True

    def _next_state(self, *, phase: str, next_batch: int) -> None:
        self.sampler_state = {
            "stage_index": int(self.stage_state["stage_index"]),
            "epoch": int(self.stage_state["epoch"]),
            "phase": phase,
            "next_batch": int(next_batch),
        }

    def _after_step(self, *, phase: str, next_batch: int, stop_after_global_step: int | None) -> bool:
        self.stage_state["global_step"] += 1
        self._next_state(phase=phase, next_batch=next_batch)
        global_step = int(self.stage_state["global_step"])
        checkpoint_interval = int(self.training["checkpoint_interval_steps"])
        if global_step % checkpoint_interval == 0 or (
            stop_after_global_step is not None and global_step >= stop_after_global_step
        ):
            self._save_last()
        return stop_after_global_step is not None and global_step >= stop_after_global_step

    def _supervised_phase(self, labeled, domain: str, epoch: int, start_batch: int, stop_after_global_step: int | None) -> bool:
        self.wrapper.student.train()
        self.wrapper.teacher.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        for batch_index, indices in batch_indices(
            len(labeled),
            int(self.training["labeled_batch_size"]),
            shuffle=True,
            seed_parts=("gate0", self.seed, domain, epoch, "supervised"),
            start_batch=start_batch,
        ):
            batch = collate(labeled, indices, require_label=True)
            if set(batch["domain"]) != {domain} or set(batch["role"]) != {"train_labeled"}:
                raise RuntimeError("current-domain-only assertion failed in supervised batch")
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits, _ = self.wrapper.student(images)
            loss = criterion(logits, labels)
            _finite_scalar("supervised loss", loss)
            loss.backward()
            update_gas_from_supervised_gradient(self.wrapper.student)
            self.optimizer.step()
            self._append_log(
                {
                    "stage_index": self.stage_state["stage_index"],
                    "domain": domain,
                    "epoch": epoch,
                    "phase": "supervised",
                    "batch_index": batch_index,
                    "global_step": self.stage_state["global_step"] + 1,
                    "loss_total": float(loss.detach()),
                    "loss_supervised": float(loss.detach()),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "hidden_gt_training_usage": "none",
                }
            )
            if self._after_step(phase="supervised", next_batch=batch_index + 1, stop_after_global_step=stop_after_global_step):
                return True
        return False

    def _prototype_batches(self, labeled, domain: str, epoch: int) -> list[dict]:
        batches: list[dict] = []
        for _, indices in batch_indices(
            len(labeled),
            int(self.training["labeled_batch_size"]),
            shuffle=False,
            seed_parts=("gate0", self.seed, domain, epoch, "prototype"),
        ):
            batches.append(collate(labeled, indices, require_label=True))
        return batches

    def _unsupervised_phase(
        self,
        labeled,
        unlabeled,
        domain: str,
        epoch: int,
        start_batch: int,
        stop_after_global_step: int | None,
    ) -> bool:
        if self.prototypes is None:
            raise RuntimeError("PAS phase requires current-domain prototypes")
        self.wrapper.student.train()
        self.wrapper.teacher.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        labeled_batches = list(
            batch_indices(
                len(labeled),
                int(self.training["labeled_batch_size"]),
                shuffle=True,
                seed_parts=("gate0", self.seed, domain, epoch, "unlabeled_labeled_cycle"),
            )
        )
        invalid_token = self.num_classes
        for batch_index, indices in batch_indices(
            len(unlabeled),
            int(self.training["unlabeled_batch_size"]),
            shuffle=False,
            seed_parts=("gate0", self.seed, domain, epoch, "unlabeled"),
            start_batch=start_batch,
        ):
            unlabeled_batch = collate(unlabeled, indices, require_label=False)
            if set(unlabeled_batch["domain"]) != {domain} or set(unlabeled_batch["role"]) != {"train_unlabeled"}:
                raise RuntimeError("current-domain-only assertion failed in unlabeled batch")
            if "label" in unlabeled_batch:
                raise RuntimeError("hidden GT tensor entered unlabeled training")
            unlabeled_images = unlabeled_batch["image"].to(self.device)
            with torch.no_grad():
                student_logits, student_features = self.wrapper.student(unlabeled_images)
                student_pseudo = upstream_pas_labels(
                    student_logits,
                    student_features,
                    self.prototypes,
                    confidence_threshold=float(self.training["confidence_threshold"]),
                    similarity_threshold=float(self.training["similarity_threshold"]),
                    invalid_token=invalid_token,
                )
                teacher_logits, teacher_features = self.wrapper.teacher(unlabeled_images)
                teacher_pseudo = upstream_pas_labels(
                    teacher_logits,
                    teacher_features,
                    self.prototypes,
                    confidence_threshold=float(self.training["confidence_threshold"]),
                    similarity_threshold=float(self.training["similarity_threshold"]),
                    invalid_token=invalid_token,
                )
                pseudo_consistency = F.mse_loss(student_pseudo.float(), teacher_pseudo.float())

            _, labeled_indices = labeled_batches[batch_index % len(labeled_batches)]
            labeled_batch = collate(labeled, labeled_indices, require_label=True)
            images = labeled_batch["image"].to(self.device)
            labels = labeled_batch["label"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits, _ = self.wrapper.student(images)
            supervised_loss = criterion(logits, labels)
            total_loss = supervised_loss + float(self.training["unsupervised_consistency_weight"]) * pseudo_consistency.detach()
            _finite_scalar("unlabeled-phase total loss", total_loss)
            total_loss.backward()
            self.optimizer.step()  # repaired upstream defect: the released code omitted this step.
            self._append_log(
                {
                    "stage_index": self.stage_state["stage_index"],
                    "domain": domain,
                    "epoch": epoch,
                    "phase": "unlabeled",
                    "batch_index": batch_index,
                    "global_step": self.stage_state["global_step"] + 1,
                    "loss_total": float(total_loss.detach()),
                    "loss_supervised": float(supervised_loss.detach()),
                    "pseudo_consistency": float(pseudo_consistency.detach()),
                    "optimizer_step_executed": True,
                    "teacher_forward_no_grad": True,
                    "hidden_gt_training_usage": "none",
                }
            )
            if self._after_step(phase="unlabeled", next_batch=batch_index + 1, stop_after_global_step=stop_after_global_step):
                return True
        self.wrapper.update_teacher(float(self.training["teacher_ema_alpha"]))
        return False

    @torch.no_grad()
    def evaluate_domain(self, domain: str, role: str) -> dict[str, Any]:
        if role not in {"val", "test"}:
            raise ValueError("evaluator role must be val or test")
        dataset = self._evaluation_dataset(domain, role)
        metrics = ConfusionMetrics(self.num_classes, self.ignore_label)
        was_training = self.wrapper.student.training
        self.wrapper.student.eval()
        for _, indices in batch_indices(
            len(dataset),
            max(1, int(self.training["labeled_batch_size"])),
            shuffle=False,
            seed_parts=("evaluate", self.seed, domain, role),
        ):
            batch = collate(dataset, indices, require_label=True)
            logits, _ = self.wrapper.student(batch["image"].to(self.device))
            prediction = logits.argmax(dim=1)
            metrics.update(prediction, batch["label"])
        if was_training:
            self.wrapper.student.train()
        summary = metrics.summary()
        summary.update({"domain": domain, "role": role, "gt_consumer": "evaluator_only"})
        return summary

    def _load_best_models(self, best_path: Path) -> None:
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        assert_complete_classifier_load(payload["student"], self.wrapper.student)
        assert_complete_classifier_load(payload["ema_teacher"], self.wrapper.teacher)
        self.wrapper.student.load_state_dict(payload["student"], strict=True)
        self.wrapper.teacher.load_state_dict(payload["ema_teacher"], strict=True)
        restore_gas_state(self.wrapper.student, payload["gas_state"])
        self.wrapper.freeze_teacher()
        self.wrapper.teacher.eval()

    def _evaluate_stage_matrix(self, stage_index: int, trained_domain: str) -> None:
        details: dict[str, Any] = {}
        for evaluation_domain in self.domain_order[: stage_index + 1]:
            result = self.evaluate_domain(evaluation_domain, "test")
            details[evaluation_domain] = result
            for metric_name in self.matrices:
                self.matrices[metric_name].setdefault(trained_domain, {})[evaluation_domain] = result[metric_name]
        write_json(self.output_dir / f"stage_{stage_index}_{trained_domain}" / "test_metrics.json", details)
        write_json(self.output_dir / "stage_by_domain_matrices.json", self.matrices)
        for metric_name, matrix in self.matrices.items():
            write_lower_triangular_csv(
                self.output_dir / f"stage_by_domain_{metric_name}.csv", self.domain_order, matrix
            )

    def _stage_end(self, stage_index: int, domain: str) -> None:
        stage_dir = self.output_dir / f"stage_{stage_index}_{domain}"
        best_path = stage_dir / "best.pt"
        if not best_path.is_file():
            raise RuntimeError(f"stage has no best checkpoint: {best_path}")
        self._load_best_models(best_path)
        self._evaluate_stage_matrix(stage_index, domain)
        write_json(
            stage_dir / "stage_completion.json",
            {
                "stage_index": stage_index,
                "domain": domain,
                "best_validation_mean_iou": self.best_metric,
                "global_step": self.stage_state["global_step"],
                "classifier_load": "strict_complete",
                "nan_detected": False,
                "hidden_gt_training_usage": "none",
            },
        )

    def run(self, *, resume_path: str | Path | None = None, stop_after_global_step: int | None = None) -> dict[str, Any]:
        start_time = time.time()
        if resume_path is not None:
            self.resume(resume_path)
        while int(self.stage_state["stage_index"]) < len(self.domain_order):
            stage_index = int(self.stage_state["stage_index"])
            domain = self.domain_order[stage_index]
            labeled, unlabeled = self._datasets(domain)
            stage_dir = self.output_dir / f"stage_{stage_index}_{domain}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            epoch = int(self.stage_state["epoch"])
            while epoch < int(self.training["epochs_per_stage"]):
                self.stage_state["epoch"] = epoch
                phase = str(self.sampler_state["phase"])
                start_batch = int(self.sampler_state["next_batch"])
                if not self.stage_state.get("epoch_lr_initialized", False):
                    self._set_epoch_lr(epoch)
                if phase == "supervised":
                    if self._supervised_phase(labeled, domain, epoch, start_batch, stop_after_global_step):
                        return {"status": "INTERRUPTED", "checkpoint": str(self.output_dir / "last.pt")}
                    phase, start_batch = "unlabeled", 0
                    self._next_state(phase=phase, next_batch=0)

                pseudo_epoch = (
                    epoch >= int(self.training["prototype_start_epoch"])
                    and epoch % int(self.training["pseudo_label_interval_epochs"]) == 0
                )
                if epoch == int(self.training["prototype_start_epoch"]) and self.prototypes is None:
                    self.prototypes = compute_single_prototypes(
                        self.wrapper.student,
                        self._prototype_batches(labeled, domain, epoch),
                        num_classes=self.num_classes,
                        device=self.device,
                        ignore_label=self.ignore_label,
                    )
                if phase == "unlabeled" and pseudo_epoch:
                    if self._unsupervised_phase(labeled, unlabeled, domain, epoch, start_batch, stop_after_global_step):
                        return {"status": "INTERRUPTED", "checkpoint": str(self.output_dir / "last.pt")}

                validation = self.evaluate_domain(domain, "val")
                current_metric = float(validation["mean_iou"])
                is_best = current_metric > self.best_metric
                self.best_metric = max(self.best_metric, current_metric)
                self.stage_state["epoch"] = epoch + 1
                self.stage_state["epoch_lr_initialized"] = False
                self._next_state(phase="supervised", next_batch=0)
                payload = self._checkpoint_payload()
                save_checkpoint(self.output_dir / "last.pt", payload)
                if is_best:
                    save_checkpoint(stage_dir / "best.pt", payload)
                    write_json(stage_dir / "best_validation.json", validation)
                epoch += 1

            self._stage_end(stage_index, domain)
            self.stage_state["stage_index"] = stage_index + 1
            self.stage_state["epoch"] = 0
            self.stage_state["epoch_lr_initialized"] = False
            self.sampler_state = {
                "stage_index": stage_index + 1,
                "epoch": 0,
                "phase": "supervised",
                "next_batch": 0,
            }
            self.prototypes = None
            self.best_metric = -math.inf
            if stage_index + 1 < len(self.domain_order):
                self.optimizer, self.scheduler = self._new_optimizer_scheduler()
                self._save_last()

        elapsed = time.time() - start_time
        final = {
            "status": "COMPLETE",
            "seed": self.seed,
            "benchmark": self.config["benchmark"],
            "domain_order": self.domain_order,
            "global_step": int(self.stage_state["global_step"]),
            "elapsed_seconds": elapsed,
            "method_registered": False,
            "hidden_gt_training_usage": "none",
            "nan_detected": False,
            "stage_by_domain_matrices": self.matrices,
        }
        write_json(self.output_dir / "run_completion.json", final)
        (self.output_dir / ".complete").write_text("complete\n", encoding="utf-8")
        return final
