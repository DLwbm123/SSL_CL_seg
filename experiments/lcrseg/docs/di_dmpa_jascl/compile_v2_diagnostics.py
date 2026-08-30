#!/usr/bin/env python3
"""Post-hoc report aggregation only: reads logs/matrices, never datasets or GT."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def stats(values):
    return {"n": len(values), "mean": statistics.mean(values), "min": min(values), "max": max(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "zero_count": sum(value == 0 for value in values)}


def audit_checkpoint(path, metadata):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 2
    assert payload["config_hash"] == metadata["config_hash"]
    assert payload["git_commit"] == metadata["git_commit"]
    count = 0
    def visit(value):
        nonlocal count
        if isinstance(value, torch.Tensor):
            count += 1
            assert torch.isfinite(value).all(), f"non-finite checkpoint tensor: {path}"
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
    visit(payload)
    assert any("decoder.conv_logit.mu.weight" == key for key in payload["student"])
    assert payload["student"].keys() == payload["ema_teacher"].keys()
    return {"status": "PASS", "finite_tensor_count": count, "schema_version": 2,
            "config_and_commit_match": True, "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("do not overwrite a previous diagnostic report")
    domains = None
    metrics = ["mean_dice", "mean_foreground_dice", "mean_iou"]
    results, commits, hashes, initial_paths = {}, set(), {}, {}
    for variant, prefix in (("C0", "lambda0"), ("B0", "pas_probmse")):
        results[variant] = {}
        for seed in (0,1,2):
            root = args.runs_root / f"gate0_v2_{prefix}_fundus_seed{seed}"
            assert (root / ".complete").exists() and (root / ".exit").read_text().strip() == "0"
            metadata = json.loads((root / "run_metadata.json").read_text())
            completion = json.loads((root / "run_completion.json").read_text())
            if domains is None:
                domains = metadata["domain_order"]
            assert metadata["domain_order"] == domains
            matrices = json.loads((root / "stage_by_domain_matrices.json").read_text())
            rows = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines() if line.strip()]
            first_u = next(index for index, row in enumerate(rows) if row["phase"] == "unlabeled")
            initial_paths[(variant,seed)] = (rows[:first_u], rows[first_u])
            commits.add(metadata["git_commit"])
            hashes[variant] = metadata["config_hash"]
            stages = {}
            for index, domain in enumerate(domains):
                subset = [row for row in rows if row["domain"] == domain and row["phase"] == "unlabeled"]
                assert subset
                stage_dir = root / f"stage_{index}_{domain}"
                stages[domain] = {
                    "pas_joint_coverage": stats([row["pas_joint_coverage"] for row in subset]),
                    "loss_consistency": stats([row["loss_consistency"] for row in subset]),
                    "student_unsupervised_gradient_norm": stats([row["student_unsupervised_gradient_norm"] for row in subset]),
                    "student_total_gradient_norm": stats([row["student_total_gradient_norm"] for row in subset]),
                    "validation_only_pseudo_label_precision": json.loads((stage_dir / "validation_pas_precision.json").read_text()),
                    "metrics": {metric: {
                        "current_domain_performance": matrices[metric][domain][domain],
                        "historical_domain_performance": {d: matrices[metric][domain][d] for d in domains[:index]},
                    } for metric in metrics},
                    "best_checkpoint": str(stage_dir / "best.pt"),
                    "best_checkpoint_sha256": sha256(stage_dir / "best.pt"),
                    "best_checkpoint_audit": audit_checkpoint(stage_dir / "best.pt", metadata),
                }
            final = {metric: {
                "domain_average": statistics.mean(matrices[metric][domains[-1]].values()),
                "forgetting": {domain: max(matrices[metric][stage][domain] for stage in domains[index:]) -
                                      matrices[metric][domains[-1]][domain]
                               for index,domain in enumerate(domains[:-1])},
                "backward_transfer": statistics.mean(
                    matrices[metric][domains[-1]][domain]-matrices[metric][domain][domain] for domain in domains[:-1]),
            } for metric in metrics}
            results[variant][str(seed)] = {"run_dir": str(root), "global_steps": len(rows), "stages": stages,
                "elapsed_seconds": completion["elapsed_seconds"],
                "run_metadata": metadata, "run_metadata_sha256": sha256(root / "run_metadata.json"),
                "run_completion_sha256": sha256(root / "run_completion.json"),
                "final": final, "train_log_sha256": sha256(root / "train.jsonl"),
                "final_checkpoint_sha256": sha256(root / "last.pt"),
                "final_checkpoint_audit": audit_checkpoint(root / "last.pt", metadata),
                "stderr": (root / "stderr.log").read_text()}
    assert len(commits) == 1
    prefix_checks = {}
    for seed in (0,1,2):
        a, u_a = initial_paths[("C0",seed)]
        b, u_b = initial_paths[("B0",seed)]
        assert len(a) == len(b)
        prefix_diff = max(abs(x["loss_supervised"]-y["loss_supervised"]) for x,y in zip(a,b))
        first_u_diff = {key: abs(u_a[key]-u_b[key]) for key in
                        ("loss_consistency", "student_unsupervised_gradient_norm", "pas_joint_coverage")}
        prefix_checks[str(seed)] = {"supervised_prefix_steps": len(a), "max_loss_difference": prefix_diff,
                                    "first_unlabeled_path_differences": first_u_diff,
                                    "within_1e_6": prefix_diff <= 1e-6 and max(first_u_diff.values()) <= 1e-6}
    paired = {}
    for metric in metrics:
        a = [results["C0"][str(seed)]["final"][metric]["domain_average"] for seed in (0,1,2)]
        b = [results["B0"][str(seed)]["final"][metric]["domain_average"] for seed in (0,1,2)]
        differences = [y-x for x,y in zip(a,b)]
        paired[metric] = {"C0_per_seed": a, "B0_per_seed": b,
                          "B0_minus_C0_per_seed": differences,
                          "C0": stats(a), "B0": stats(b), "paired_difference": stats(differences)}
    report = {"status": "COMPLETE_DESCRIPTIVE_REPORT", "source_commit": next(iter(commits)),
              "analysis_script_sha256": sha256(Path(__file__)), "config_hashes": hashes,
              "domain_order": domains, "domain_order_source": "validated run_metadata.json, frozen protocol",
              "runs": results, "paired_comparison": paired,
              "initial_compute_rng_path_audit": prefix_checks,
              "performance_superiority_required": False,
              "GT_read_by_this_script": "none; validation diagnostic JSON only"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
