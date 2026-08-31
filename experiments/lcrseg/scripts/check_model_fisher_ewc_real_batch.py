#!/usr/bin/env python3
"""One registered real-training engineering check for Fundus Model-Fisher EWC."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.data import (
    DeterministicBatcher,
    H5LabeledDataset,
    H5UnlabeledDataset,
    LabeledTransform,
    WeakStrongTransform,
    collate_labeled,
    collate_unlabeled,
)
from lcrseg.engine.checkpoint import capture_rng_state
from lcrseg.engine.trainer import Trainer, TrainerState, build_optimizer, build_scheduler, seed_everything
from lcrseg.methods import build_method
from lcrseg.methods.base import model_checksum
from lcrseg.models import UNet2D
from scripts.verify_resume_equivalence import compare


REGISTRATION_SHA256 = "316bc9fefef0ce8ce433c28d156d35ecbc6f26991cfe459ffd37f42b4020183e"
REGISTRATION_PATH = "docs/fundus_model_fisher_ewc_v1/registration.json"
REGISTRATION_ID = "FUNDUS_MODEL_FISHER_EWC_V1"
METHOD_ARM = "model_fisher_ewc_v1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_same(first, second, path="root"):
    matched, maximum = compare(first, second, atol=0.0, rtol=0.0)
    require(matched, f"{path}: max_abs_difference={maximum}")


def model_state(method):
    return {name: value.detach().cpu().clone() for name, value in method.model.state_dict().items()}


def referenced_files(data_root: Path, *datasets) -> list[Path]:
    files = set()
    for dataset in datasets:
        for sample in dataset.samples:
            row = sample.row
            files.add(data_root / "h5/v1" / row["image_h5_relpath"])
            if dataset.require_label:
                files.add(data_root / "h5/v1" / row["label_h5_relpath"])
    return sorted(files)


def main(
    *,
    registration_path=REGISTRATION_PATH,
    registration_sha256=REGISTRATION_SHA256,
    registration_id=REGISTRATION_ID,
    method_arm=METHOD_ARM,
):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    registration = PROJECT_ROOT / registration_path
    require(sha256(registration) == registration_sha256, "registration bytes differ")
    reg = json.loads(registration.read_text())
    require(reg["registration_id"] == registration_id, "registration identity differs")
    output = args.output_dir.resolve()
    require(output == Path(reg["nas_root"]) / "real_batch", "unexpected output directory")
    require(
        subprocess.check_output(["findmnt", "-rn", "-T", str(output.parent), "-o", "FSTYPE"], text=True).strip()
        in {"nfs", "nfs4"},
        "NAS unavailable",
    )
    require(os.environ.get("CUDA_VISIBLE_DEVICES") in {str(x) for x in reg["allowed_gpus"]}, "one allowed GPU required")
    require(torch.cuda.is_available(), "CUDA unavailable; no CPU or HOME fallback")
    data_root = Path(reg["data_root"])
    for entry in reg["frozen_inputs"]:
        for path, expected in (
            (data_root / f'manifests/training/lcrseg_v1_seed{entry["seed"]}.csv', entry["manifest_sha256"]),
            (data_root / f'splits/fundus_seed{entry["seed"]}.json', entry["split_sha256"]),
        ):
            require(sha256(path) == expected, "frozen metadata hash differs")
    output.mkdir(exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "RUNNING",
        "registration_id": reg["registration_id"],
        "registration_sha256": registration_sha256,
        "arms": [],
        "model_forward_calls": 0,
        "autograd_grad_calls": 0,
        "optimizer_steps": 0,
        "hidden_training_gt_used": False,
        "test_role_used": False,
    }
    started = time.perf_counter()
    hooks = []
    original_grad = torch.autograd.grad
    try:
        seed_everything(0)
        labeled = H5LabeledDataset(
            data_root, seed=0, dataset="fundus", sites=("REFUGE",), transform=LabeledTransform()
        )
        unlabeled = H5UnlabeledDataset(
            data_root, seed=0, dataset="fundus", sites=("REFUGE",), transform=WeakStrongTransform()
        )
        batch_l = collate_labeled([labeled[index] for index in range(2)]).to("cuda")
        batch_u = collate_unlabeled([unlabeled[index] for index in range(4)]).to("cuda")
        report["private_batch_ids_sha256"] = hashlib.sha256(
            json.dumps([batch_l.case_id, batch_u.case_id], separators=(",", ":")).encode()
        ).hexdigest()
        protected = {str(path): sha256(path) for path in referenced_files(data_root, labeled, unlabeled)}
        report["protected_input_files"] = len(protected)
        report["protected_input_bytes"] = sum(Path(path).stat().st_size for path in protected)
        control_after_update = None
        common = {
            key: value
            for key, value in reg["method"].items()
            if not key.startswith("fisher_")
            and key not in {"ewc_lambda", "ewc_gamma", "terminal_stage_consolidation"}
        }
        for arm in reg["arms"]:
            seed_everything(0)
            config = dict(common)
            if arm == method_arm:
                config.update(
                    ewc_lambda=reg["method"]["ewc_lambda"],
                    ewc_gamma=reg["method"]["ewc_gamma"],
                    fisher_max_images=reg["method"]["fisher_max_images"],
                    fisher_points_per_image=reg["method"]["fisher_points_per_image"],
                    fisher_seed=0,
                )
            method = build_method(arm, UNet2D(3, 3).cuda(), config=config)
            method.set_site_index(0)
            method.begin_site("REFUGE", None, 1)

            def count_forward(module, inputs):
                require(
                    report["model_forward_calls"] < reg["engineering"]["real_batch_model_forwards_max"],
                    "registered model-forward budget exceeded",
                )
                report["model_forward_calls"] += 1

            hooks.append(method.model.register_forward_pre_hook(count_forward))
            method.model.eval()
            before_golden = model_checksum(method.model)
            with torch.no_grad():
                first = method.training_step(batch_l, batch_u, 0, 0)
                second = method.training_step(batch_l, batch_u, 0, 0)
            for key in first.losses:
                require(
                    torch.allclose(
                        first.losses[key], second.losses[key],
                        atol=reg["engineering"]["golden_atol"], rtol=reg["engineering"]["golden_rtol"],
                    ),
                    f"nonrepeatable golden loss: {key}",
                )
            require(before_golden == model_checksum(method.model), "golden calls changed the model")
            optimizer = build_optimizer(method, lr=reg["training"]["lr"], weight_decay=reg["training"]["weight_decay"])
            trainer = Trainer(
                method, optimizer=optimizer, scheduler=build_scheduler(optimizer, total_steps=1), device="cuda", amp=False
            )
            result = trainer.train_step(batch_l, batch_u, state=TrainerState())
            report["optimizer_steps"] += 1
            require(all(torch.isfinite(value).all() for value in method.model.state_dict().values()), "nonfinite model")
            require(math.isfinite(float(result.total_loss.detach())), "nonfinite training loss")
            state_after_update = model_state(method)
            if arm == "sequential_ssl":
                control_after_update = state_after_update
                report["arms"].append({"arm": arm, "golden_equal": True, "first_stage_update": True})
                continue
            assert_same(control_after_update, state_after_update, "paired_first_stage_model")
            parameters_before = {name: value.detach().clone() for name, value in method.model.named_parameters()}
            buffers_before = {name: value.detach().clone() for name, value in method.model.named_buffers()}
            gradients_before = {
                name: None if value.grad is None else value.grad.detach().clone()
                for name, value in method.model.named_parameters()
            }
            optimizer_before = copy.deepcopy(optimizer.state_dict())
            rng_before = capture_rng_state()
            modes_before = [module.training for module in method.model.modules()]
            label_reads = 0

            def forbidden_label_read(*unused_args, **unused_kwargs):
                nonlocal label_reads
                label_reads += 1
                raise RuntimeError("Fisher attempted to read a label")

            labeled._read_label = forbidden_label_read
            batcher = DeterministicBatcher(
                labeled,
                batch_size=reg["training"]["labeled_batch_size"],
                seed=0,
                namespace="fundus:REFUGE:labeled:REFUGE",
                collate=collate_labeled,
            )

            def counted_grad(*grad_args, **grad_kwargs):
                require(
                    report["autograd_grad_calls"] < reg["engineering"]["real_batch_autograd_grad_calls_max"],
                    "registered autograd.grad budget exceeded",
                )
                report["autograd_grad_calls"] += 1
                return original_grad(*grad_args, **grad_kwargs)

            torch.autograd.grad = counted_grad
            fisher = method.estimate_fisher(batcher, device="cuda")
            torch.autograd.grad = original_grad
            require(label_reads == 0, "Fisher label reader was called")
            require(fisher["actual_images"] == 16 and fisher["actual_points"] == 256, "Fisher count differs")
            require(fisher["model_forward_calls"] == 16 and fisher["autograd_grad_calls"] == 768, "Fisher work differs")
            require(fisher["completed_consolidations"] == 1 and math.isfinite(fisher["fisher_mean"]), "Fisher summary differs")
            assert_same(parameters_before, {name: value.detach() for name, value in method.model.named_parameters()}, "parameters")
            assert_same(buffers_before, {name: value.detach() for name, value in method.model.named_buffers()}, "buffers")
            assert_same(gradients_before, {name: value.grad for name, value in method.model.named_parameters()}, "gradients")
            assert_same(optimizer_before, optimizer.state_dict(), "optimizer")
            assert_same(rng_before, capture_rng_state(), "rng")
            require(modes_before == [module.training for module in method.model.modules()], "module modes changed")
            state = method.method_state_dict()["method_statistics"]["model_fisher_ewc_state"]
            require(state["completed_consolidations"] == 1, "consolidation state differs")
            require(set(state["reference_parameters"]) == set(state["fisher_diagonal"]), "Fisher keys differ")
            positive = []
            for name, value in state["fisher_diagonal"].items():
                require(torch.isfinite(value).all() and not value.lt(0).any(), f"invalid Fisher: {name}")
                if value.gt(0).any():
                    positive.append((name, int(value.reshape(-1).argmax())))
            require(positive and fisher["fisher_mean"] > 0, "Fisher is identically zero")
            name, flat_index = positive[0]
            live = dict(method.model.named_parameters())[name]
            with torch.no_grad():
                live.reshape(-1)[flat_index].add_(0.001)
            penalty = method._ewc_loss(live)
            require(torch.isfinite(penalty) and penalty.item() > 0, "perturbed EWC penalty is not positive")
            next_optimizer = build_optimizer(method, lr=reg["training"]["lr"], weight_decay=reg["training"]["weight_decay"])
            next_trainer = Trainer(
                method,
                optimizer=next_optimizer,
                scheduler=build_scheduler(next_optimizer, total_steps=1),
                device="cuda",
                amp=False,
            )
            active = next_trainer.train_step(batch_l, batch_u, state=TrainerState(global_step=1))
            report["optimizer_steps"] += 1
            require(active.scalars["loss_model_fisher_ewc"] > 0 and torch.isfinite(active.total_loss), "active EWC update failed")
            report["arms"].append(
                {
                    "arm": arm,
                    "golden_equal": True,
                    "paired_first_stage_exact": True,
                    "first_stage_update": True,
                    "active_ewc_update": True,
                    "actual_fisher_images": fisher["actual_images"],
                    "actual_fisher_points": fisher["actual_points"],
                    "actual_fisher_autograd_grad_calls": fisher["autograd_grad_calls"],
                    "selection_sha256": fisher["selection_sha256"],
                    "fisher_mean": fisher["fisher_mean"],
                    "positive_penalty": float(penalty.detach()),
                }
            )
        require(report["model_forward_calls"] == 37, "actual model-forward count differs")
        require(report["autograd_grad_calls"] == 768, "actual autograd.grad count differs")
        require(report["optimizer_steps"] == 3, "actual optimizer-step count differs")
        for path, expected in protected.items():
            require(sha256(Path(path)) == expected, "protected input changed")
        report.update(
            status="PASS_REAL_BATCH_ENGINEERING",
            protected_input_bytes_unchanged=True,
            elapsed_seconds=time.perf_counter() - started,
            peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated()),
        )
    except BaseException as exc:
        report.update(status="FAIL_ENGINEERING", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        torch.autograd.grad = original_grad
        for hook in hooks:
            hook.remove()
        with (output / "report.json").open("x") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
