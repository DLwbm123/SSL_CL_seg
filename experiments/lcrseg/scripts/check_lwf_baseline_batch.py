#!/usr/bin/env python3
"""One registered real-training-batch engineering check, never a PMGC replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.data import H5LabeledDataset, H5UnlabeledDataset, LabeledTransform, WeakStrongTransform, collate_labeled, collate_unlabeled
from lcrseg.engine.trainer import Trainer, TrainerState, build_optimizer, build_scheduler, seed_everything
from lcrseg.methods import build_method
from lcrseg.methods.base import model_checksum
from lcrseg.models import UNet2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    registration = PROJECT_ROOT / "docs/fundus_lwf_v1/registration.json"
    reg = json.loads(registration.read_text())
    output = args.output_dir.resolve()
    if output != Path(reg["nas_root"]) / "real_batch":
        raise ValueError("output must be the registered create-only NAS real_batch directory")
    if subprocess.check_output(["findmnt", "-rn", "-T", str(output.parent), "-o", "FSTYPE"], text=True).strip() not in {"nfs", "nfs4"}:
        raise RuntimeError("NAS is unavailable")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {str(x) for x in reg["allowed_gpus"]}:
        raise RuntimeError("exactly one authorized physical GPU must be visible")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU or HOME fallback")
    root = Path(reg["data_root"])
    for entry in reg["frozen_inputs"]:
        for path, expected in [(root / f'manifests/training/lcrseg_v1_seed{entry["seed"]}.csv', entry["manifest_sha256"]),
                               (root / f'splits/fundus_seed{entry["seed"]}.json', entry["split_sha256"])]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError("frozen metadata hash differs")
    output.mkdir(exist_ok=False)
    report = {"status": "RUNNING", "registration_id": reg["registration_id"], "registration_sha256": hashlib.sha256(registration.read_bytes()).hexdigest(),
              "arms": [], "forwards": 0, "optimizer_steps": 0, "hidden_gt_usage": "none", "test_gt_usage": "none"}
    try:
        seed_everything(0)
        labeled = H5LabeledDataset(root, seed=0, dataset="fundus", sites=("REFUGE",), transform=LabeledTransform())
        unlabeled = H5UnlabeledDataset(root, seed=0, dataset="fundus", sites=("REFUGE",), transform=WeakStrongTransform())
        batch_l = collate_labeled([labeled[i] for i in range(2)]).to("cuda")
        batch_u = collate_unlabeled([unlabeled[i] for i in range(4)]).to("cuda")
        report["private_batch_ids_sha256"] = hashlib.sha256(json.dumps([batch_l.case_id, batch_u.case_id]).encode()).hexdigest()
        for arm in reg["arms"]:
            seed_everything(0)
            method = build_method(arm, UNet2D(3, 3).cuda(), config=reg["method"])
            method.begin_site("REFUGE", None, 1)
            if arm == "uniform_kd":
                method._make_old_model()  # Fresh independent engineering fixture, not a historical checkpoint.
            assert method.method_name == arm
            captured = []
            def count_forward(module, inputs):
                if report["forwards"] >= reg["engineering"]["real_batch_forwards_max"]:
                    raise RuntimeError("registered forward budget exceeded")
                report["forwards"] += 1
            def capture_logits(module, inputs, result):
                captured.append(result.logits.detach().cpu().clone())
            hooks = [method.model.register_forward_pre_hook(count_forward), method.model.register_forward_hook(capture_logits)]
            if method.old_model is not None:
                hooks.append(method.old_model.register_forward_pre_hook(count_forward))
            before = model_checksum(method.model)
            with torch.no_grad():
                first = method.training_step(batch_l, batch_u, 0, 0)
                first_logits = captured[:]; captured.clear()
                second = method.training_step(batch_l, batch_u, 0, 0)
            assert len(first_logits) == len(captured)
            for a, b in zip(first_logits, captured):
                assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)
            for key in first.losses:
                assert torch.allclose(first.losses[key], second.losses[key], atol=1e-6, rtol=1e-6)
            assert before == model_checksum(method.model)
            np.save(output / f"{arm}_golden_labeled_logits.npy", first_logits[0].numpy())
            captured.clear()
            optimizer = build_optimizer(method, lr=reg["training"]["lr"], weight_decay=reg["training"]["weight_decay"])
            trainer = Trainer(method, optimizer=optimizer, scheduler=build_scheduler(optimizer, total_steps=1), device="cuda", amp=False)
            result = trainer.train_step(batch_l, batch_u, state=TrainerState())
            report["optimizer_steps"] += 1
            gradients = [p.grad for p in method.model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(x).all() for x in gradients)
            assert before != model_checksum(method.model)
            method.assert_old_state_unchanged()
            report["arms"].append({"arm": arm, "golden_equal": True, "current_updated": True, "old_unchanged_no_grad": True,
                                   "loss": float(result.total_loss.detach()), "training_labeled": 2, "training_unlabeled": 4})
            for hook in hooks:
                hook.remove()
        assert report["forwards"] == 24 and report["optimizer_steps"] == 2
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL_ENGINEERING", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
