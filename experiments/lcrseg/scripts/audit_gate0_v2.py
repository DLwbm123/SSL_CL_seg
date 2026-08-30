#!/usr/bin/env python3
"""Read-only v1 checkpoint diagnostics; never optimizer-step or access test GT."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from di_dmpa_jascl.config import load_yaml, sha256_file
from di_dmpa_jascl.checkpoint import capture_rng_state
from di_dmpa_jascl.data import LCRSegH5Dataset, batch_indices, collate
from di_dmpa_jascl.manifest import LCRSegManifestAdapter
from di_dmpa_jascl.metrics import ConfusionMetrics, write_json
from di_dmpa_jascl.modeling import (
    build_mean_teacher, compute_single_prototypes, pas_probability_objective, gradient_norm,
)
from di_dmpa_jascl.provenance import assert_upstream_unchanged, git_revision
from di_dmpa_jascl.runner import seed_everything
from scripts.compile_gate0_reports import config_contract
from scripts.verify_resume_equivalence import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gradient", "evaluation", "leakage", "all"), default="all")
    parser.add_argument("--v1-runs-root", type=Path, default=Path("/root/LCRSeg/runs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    configs, hashes = config_contract()
    config = configs["B0"]
    protocol = load_yaml(ROOT / config["data"]["protocol"])
    commit, device = git_revision(ROOT), torch.device(args.device)
    adapter = LCRSegManifestAdapter(config["data"]["root"], protocol, seed=0, benchmark="fundus")
    reference = ROOT / config["model"]["reference_root"]
    assert_upstream_unchanged(reference, config["model"]["upstream_path"])
    domains = config["data"]["domain_order"]
    common = {"git_commit": commit, "config_hashes": hashes, "method_registered": False,
              "di_dmpa_training_launched": False, "frozen_v1_commit": "46e892960240543c946c570a9378d409b226384b"}

    def emit(name, payload):
        path = args.output_dir / name
        if path.exists():
            raise RuntimeError(f"refusing to overwrite audit evidence: {path}")
        write_json(path, {**common, **payload})
        print(json.dumps({"report": str(path), "status": payload["status"]}), flush=True)

    def dataset(domain, role, purpose):
        records = adapter.records(domain=domain, role=role, purpose=purpose)
        return LCRSegH5Dataset(config["data"]["root"], records, require_label=role != "train_unlabeled",
                              output_hw=(384,384), augment=role == "train_labeled")

    def model(stage, domain):
        path = args.v1_runs_root / f"gate0_repaired_unet_fundus_seed0/stage_{stage}_{domain}/best.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        wrapper = build_mean_teacher(reference, upstream_path=config["model"]["upstream_path"],
                                     input_channels=3, num_classes=3, device=device)
        wrapper.student.load_state_dict(payload["student"], strict=True)
        wrapper.teacher.load_state_dict(payload["ema_teacher"], strict=True)
        wrapper.freeze_teacher()
        wrapper.teacher.eval()
        return wrapper, payload, path

    if args.mode in ("leakage", "all"):
        audits = {str(seed): LCRSegManifestAdapter(config["data"]["root"], protocol, seed=seed,
                  benchmark="fundus").leakage_audit() for seed in (0,1,2)}
        emit("LEAKAGE_AUDIT_REPORT.json", {"status": "PASS", "hidden_gt_training_usage": "none",
              "seeds": audits, "data_kind": "real_frozen_manifest", "optimizer_steps": 0})

    if args.mode in ("gradient", "all"):
        rows = {}
        for stage, domain in enumerate(domains):
            wrapper, payload, checkpoint = model(stage, domain)
            seed_everything(20260830 + stage)
            wrapper.student.train()
            labeled, unlabeled = dataset(domain, "train_labeled", "train"), dataset(domain, "train_unlabeled", "train")
            prototypes = payload.get("prototypes")
            prototype_source = "v1_checkpoint"
            if prototypes is None:
                batches = [collate(labeled, ids, require_label=True) for _, ids in batch_indices(
                    len(labeled), 2, shuffle=False, seed_parts=("audit_prototype", domain))]
                prototypes = compute_single_prototypes(wrapper.student, batches, num_classes=3,
                                                       device=device, ignore_label=255)
                prototype_source = "recomputed_current_domain_labeled_only"
            prototypes = prototypes.detach().to(device)
            u_batch = collate(unlabeled, list(range(min(2, len(unlabeled)))), require_label=False)
            assert "label" not in u_batch
            epoch = int(payload["stage_state"]["epoch"])
            _, ids = next(batch_indices(len(labeled), 2, shuffle=True,
                          seed_parts=("gate0", 0, domain, epoch, "unlabeled_labeled_cycle")))
            l_batch = collate(labeled, ids, require_label=True)
            wrapper.zero_grad(set_to_none=True)
            loss, vs, vt, joint = pas_probability_objective(wrapper.student, wrapper.teacher,
                                                          u_batch["image"].to(device), prototypes)
            params = [p for p in wrapper.student.parameters() if p.requires_grad]
            gu = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
            logits, _ = wrapper.student(l_batch["image"].to(device), stochastic_classifier=True)
            supervised = F.cross_entropy(logits, l_batch["label"].to(device), ignore_index=255)
            gs = torch.autograd.grad(supervised, params, allow_unused=True, retain_graph=True)
            total = supervised + 0.5 * loss
            gt = torch.autograd.grad(total, params, allow_unused=True, retain_graph=True)
            total.backward()
            norm_u = gradient_norm(gu)
            norm_delta = gradient_norm([b-a for a,b in zip(gs,gt) if a is not None and b is not None])
            row = {"stage": stage, "domain": domain, "valid_student_pixels": int(vs.valid_mask.sum()),
                   "valid_teacher_pixels": int(vt.valid_mask.sum()), "joint_valid_pixels": int(joint.sum()),
                   "joint_coverage": float(joint.float().mean()), "consistency_loss": float(loss.detach()),
                   "consistency_requires_grad": loss.requires_grad, "student_unsupervised_gradient_norm": norm_u,
                   "student_total_gradient_norm": gradient_norm(gt), "total_minus_supervised_gradient_norm": norm_delta,
                   "teacher_nonnull_gradient_count": sum(p.grad is not None for p in wrapper.teacher.parameters()),
                   "prototype_requires_grad": prototypes.requires_grad, "hidden_gt_training_usage": "none",
                   "stochastic_classifier_train_mode": True, "exact_checkpoint": str(checkpoint),
                   "checkpoint_sha256": sha256_file(checkpoint), "checkpoint_stage_state": payload["stage_state"],
                   "prototype_source": prototype_source, "unlabeled_case_ids": u_batch["case_id"],
                   "labeled_case_ids": l_batch["case_id"], "rng_seed": 20260830+stage,
                   "config_hash": hashes["B0"], "git_commit": commit, "optimizer_steps": 0}
            row["status"] = ("BLOCKED_ZERO_PAS_COVERAGE" if row["joint_valid_pixels"] == 0 else
                             "PASS" if norm_u > 1e-8 and norm_delta > 1e-8 and
                             row["teacher_nonnull_gradient_count"] == 0 and not prototypes.requires_grad else "FAIL")
            rows[domain] = row
            print(json.dumps(row), flush=True)
            del wrapper, payload, loss, total, gu, gt, gs
        status = "BLOCKED_ZERO_PAS_COVERAGE" if any(r["joint_valid_pixels"] == 0 for r in rows.values()) else (
            "PASS" if all(r["status"] == "PASS" for r in rows.values()) else "FAIL")
        emit("PAS_GRADIENT_AUDIT.json", {"status": status, "data_kind": "real_fundus_unlabeled",
             "domains": rows, "thresholds": {"confidence": 0.7, "similarity": 0.7},
             "batch_selection": "first manifest-ordered unlabeled batch per domain; no search or retries",
             "preflight_only_not_v2_training": True})

    if args.mode in ("evaluation", "all"):
        domain = domains[0]
        wrapper, payload, checkpoint = model(0, domain)
        wrapper.student.eval()
        validation = dataset(domain, "val", "evaluate")
        def evaluate(stochastic, seed):
            seed_everything(seed)
            before = capture_rng_state()
            metrics, outputs = ConfusionMetrics(3,255), []
            with torch.no_grad():
                for _, ids in batch_indices(len(validation), 2, shuffle=False, seed_parts=("eval_audit",)):
                    batch = collate(validation, ids, require_label=True)
                    logits, _ = wrapper.student(batch["image"].to(device), stochastic_classifier=stochastic)
                    metrics.update(logits.argmax(1), batch["label"])
                    if not stochastic:
                        outputs.append(logits.cpu())
            unchanged, _ = compare(before, capture_rng_state(), atol=0, rtol=0)
            return metrics.summary()["mean_dice"], outputs, unchanged
        seeds = list(range(100,120))
        values = []
        for seed in seeds:
            value, _, _ = evaluate(True, seed)
            values.append(value)
        d1, first, rng1 = evaluate(False, 200)
        d2, second, rng2 = evaluate(False, 201)
        maximum = max(float((a-b).abs().max()) for a,b in zip(first,second))
        emit("EVAL_STOCHASTICITY_AUDIT.json", {
            "status": "PASS" if maximum == 0 and rng1 and rng2 else "FAIL",
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "domain": domain, "role": "val", "metric": "mean_dice", "gt_consumer": "evaluator_only",
            "stochastic_single_draw": {"values": values, "mean": statistics.mean(values), "std": statistics.stdev(values),
                                       "min": min(values), "max": max(values)},
            "deterministic_values": [d1,d2], "deterministic_repeat_count": 2,
            "deterministic_repeat_max_absolute_difference": maximum, "deterministic_rng_unchanged": rng1 and rng2,
            "MC_16_mean": None, "MC_16_status": "NOT_RUN_OPTIONAL", "rng_seeds": {"stochastic": seeds, "deterministic": [200,201]},
            "formal_evaluation_policy": "posterior_mean_preselected", "optimizer_steps": 0,
        })


if __name__ == "__main__":
    main()
