#!/usr/bin/env python3
"""Zero-real-data CUDA device-resolution preflight for Model-Fisher EWC V2."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.engine.checkpoint import capture_rng_state
from lcrseg.engine.trainer import seed_everything
from lcrseg.methods import build_method
from lcrseg.models import UNet2D
from scripts.verify_resume_equivalence import compare


REGISTRATION_SHA256 = "cc86b8518de7ad622a41dc20db896310ebdcf59176d4b62ad58b7e6b6db4b670"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(method):
    return {
        "model": {name: value.detach().clone() for name, value in method.model.state_dict().items()},
        "gradients": {
            name: None if value.grad is None else value.grad.detach().clone()
            for name, value in method.model.named_parameters()
        },
        "method": copy.deepcopy(method.method_state_dict()),
        "rng": capture_rng_state(),
        "modes": [module.training for module in method.model.modules()],
    }


def require_same(first, second, label):
    matched, maximum = compare(first, second, atol=0.0, rtol=0.0)
    require(matched, f"{label}: max_abs_difference={maximum}")


class EmptyImageDataset:
    def __len__(self):
        return 0

    def image_at(self, index):
        raise AssertionError(f"empty dataset image accessed: {index}")


class EmptyBatcher:
    dataset = EmptyImageDataset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    registration = PROJECT_ROOT / "docs/fundus_model_fisher_ewc_v2/registration.json"
    require(sha256(registration) == REGISTRATION_SHA256, "registration bytes differ")
    reg = json.loads(registration.read_text())
    require(reg["registration_id"] == "FUNDUS_MODEL_FISHER_EWC_V2", "registration identity differs")
    output = args.output_dir.resolve()
    require(output == Path(reg["nas_root"]) / "device_preflight", "unexpected output directory")
    require(
        subprocess.check_output(["findmnt", "-rn", "-T", str(output.parent), "-o", "FSTYPE"], text=True).strip()
        in {"nfs", "nfs4"},
        "NAS unavailable",
    )
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "7", "registered GPU7 mapping required")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one mapped CUDA device required")
    expected_source = os.environ.get("EXPECTED_SOURCE_COMMIT", "")
    require(len(expected_source) == 40, "expected source commit is missing")
    require(
        subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()
        == expected_source,
        "execution checkout differs",
    )
    output.mkdir(exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "RUNNING",
        "registration_id": reg["registration_id"],
        "registration_sha256": REGISTRATION_SHA256,
        "source_commit": expected_source,
        "real_data_access": False,
        "model_forward_calls": 0,
        "autograd_grad_calls": 0,
        "optimizer_steps": 0,
        "requests": [],
    }
    started = time.perf_counter()
    original_grad = torch.autograd.grad
    hook = None
    try:
        seed_everything(0)
        common = {
            key: value
            for key, value in reg["method"].items()
            if not key.startswith("fisher_")
            and key not in {"ewc_lambda", "ewc_gamma", "terminal_stage_consolidation"}
        }
        common.update(
            ewc_lambda=reg["method"]["ewc_lambda"],
            ewc_gamma=reg["method"]["ewc_gamma"],
            fisher_max_images=reg["method"]["fisher_max_images"],
            fisher_points_per_image=reg["method"]["fisher_points_per_image"],
            fisher_seed=0,
        )
        method = build_method("model_fisher_ewc_v2", UNet2D(3, 3).cuda(), config=common)
        require(method.method_name == "model_fisher_ewc_v2" and method.method_version == "2.0", "method identity differs")
        method.set_site_index(0)
        method.begin_site("REFUGE", None, 1)

        def forbidden_forward(module, inputs):
            report["model_forward_calls"] += 1
            raise RuntimeError("zero-real preflight called the model")

        def forbidden_grad(*grad_args, **grad_kwargs):
            report["autograd_grad_calls"] += 1
            raise RuntimeError("zero-real preflight called autograd.grad")

        hook = method.model.register_forward_pre_hook(forbidden_forward)
        torch.autograd.grad = forbidden_grad
        for requested in reg["zero_real_device_preflight"]["accepted_requests"]:
            before = snapshot(method)
            try:
                method.estimate_fisher(EmptyBatcher(), device=requested)
            except ValueError as error:
                require(str(error) == "model-Fisher input dataset is empty", f"accepted request failed early: {requested}")
            else:
                raise RuntimeError(f"accepted request did not reach empty dataset: {requested}")
            require_same(before, snapshot(method), f"accepted request mutated state: {requested}")
            report["requests"].append({"requested": requested, "result": "ACCEPTED_TO_EMPTY_DATASET_BOUNDARY"})
        for requested in reg["zero_real_device_preflight"]["rejected_requests"]:
            before = snapshot(method)
            try:
                method.estimate_fisher(EmptyBatcher(), device=requested)
            except ValueError as error:
                require(str(error) == "model-Fisher device differs from current model", f"wrong rejection: {requested}")
            else:
                raise RuntimeError(f"mismatched request was accepted: {requested}")
            require_same(before, snapshot(method), f"rejected request mutated state: {requested}")
            report["requests"].append({"requested": requested, "result": "REJECTED_DEVICE_MISMATCH"})
        require(report["model_forward_calls"] == 0, "model-forward budget differs")
        require(report["autograd_grad_calls"] == 0, "autograd.grad budget differs")
        require(report["optimizer_steps"] == 0, "optimizer-step budget differs")
        report.update(
            status="PASS_ZERO_REAL_DEVICE_PREFLIGHT",
            elapsed_seconds=time.perf_counter() - started,
            live_parameter_device=str(next(method.model.parameters()).device),
            peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated()),
        )
    except BaseException as error:
        report.update(
            status="FAIL_ZERO_REAL_DEVICE_PREFLIGHT",
            error=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.perf_counter() - started,
        )
        raise
    finally:
        torch.autograd.grad = original_grad
        if hook is not None:
            hook.remove()
        with (output / "report.json").open("x") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
