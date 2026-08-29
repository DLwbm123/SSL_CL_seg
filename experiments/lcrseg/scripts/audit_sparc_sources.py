#!/usr/bin/env python3
"""Audit the frozen source semantics used by the SPARC-Seg V0.1 protocol.

This script is deliberately read-only with respect to the reference checkouts.  It
does not import either project and does not turn source implementation details
into SPARC hyper-parameters; it only verifies and records the preregistered
provenance boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_JASCL_COMMIT = "3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def require(text: str, needle: str, label: str, checks: dict[str, bool]) -> None:
    checks[label] = needle in text


def write_reports(report_dir: Path, payload: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "SPARC_SOURCE_AUDIT.json"
    md_path = report_dir / "SPARC_SOURCE_AUDIT.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    jascl = payload["jascl"]
    star = payload["star"]
    lines = [
        "# SPARC-Seg V0.1 source audit",
        "",
        f"**Status:** `{payload['status']}`  ",
        f"**Generated:** {payload['generated_at']}  ",
        "**Scope:** source semantics and adaptation boundary only; no method implementation or optimizer step",
        "",
        "## JASCL: paper/source-derived evidence",
        "",
        f"- Official repository: `{jascl['repository']}`.",
        f"- Audited commit: `{jascl['commit']}` (required exact commit matched: `{jascl['required_commit_match']}`).",
        "- Prototype source: `methods/utils.py::get_prototype` and `norm_mean`.",
        "- Source normalizes each pixel feature, averages pixels within each case, then averages case prototypes. It does not explicitly normalize the final cross-case mean and does not enforce SPARC's 32-cell minimum.",
        "- PAS source uses softmax confidence and class-wise cosine similarity with strict `> 0.7` / `> 0.7` comparisons.",
        "- Student and EMA-teacher label maps are filtered independently. The source computes MSE directly between the two filtered integer-valued maps; no explicit Boolean student/teacher-mask intersection is constructed.",
        "- The teacher update is EMA with `alpha=0.99`, and PAS is invoked periodically after prototype refresh.",
        "",
        "### Paper/source distinction and SPARC adaptation",
        "",
        "The paper-level contribution describes prototype-assisted pseudo-label validation and a mean-teacher consistency path. The exact per-case reduction, final-normalization omission, filtered-label-map MSE, and absence of an explicit intersection are source-level details. SPARC does not silently copy those implementation details: it freezes thresholds at 0.7/0.7, uses current plus frozen-previous validators, has no EMA teacher, and retains the R0 hard-CE pseudo target.",
        "",
        "## STAR: paper/source-derived evidence",
        "",
        f"- Official repository: `{star['repository']}`.",
        f"- Audited repository HEAD: `{star['commit']}`.",
        "- The old-class region in the verifiable VOC implementation is `(current label == background) AND (old-model prediction > background)`. The old prediction uses thresholded old logits: pixels with no old logit above 0.5 are reset to background.",
        "- `PKDLoss` compares `features[5]`, which is the ASPP/classifier-input tensor `x_pl` appended last by `DeepLabV3.forward(..., ret_intermediate=True)`.",
        "- It uses elementwise MSE, bilinear-resizes the region mask, and divides the masked sum by `mask_sum * channels`.",
        "- No explicit source flag implementing an all-feature PKD ablation was found in the audited checkout; SPARC's registered S5 all-valid-spatial control is therefore a protocol control, not a claimed STAR-source reproduction.",
        "- SPARC adopts only the targeted-maintaining idea. Its frozen contract is dual-stable foreground, channel-normalized cosine distance at same-name layers; it does not copy STAR's region rule, bilinear soft mask, layer, or MSE formula.",
        "",
        "## LAG: conceptual provenance only",
        "",
        "LAG motivates stable/sample-specific semantic information as a concept. SPARC does not implement LAG's channel-wise split, spatial decoupling, asymmetric contrastive module, NSC, or LRP.",
        "",
        "## Frozen SPARC source boundary",
        "",
        "- `thresholds = 0.7 / 0.7`",
        "- current plus frozen-previous semantic validators",
        "- no EMA teacher in proposed SPARC",
        "- R0 hard-CE pseudo target remains unchanged",
        "- no cross-site prototype memory or replay",
        "- no STAR formula is transplanted",
        "- no LAG module is implemented",
        "",
        "## Integrity checks",
        "",
    ]
    for name, passed in sorted(payload["checks"].items()):
        lines.append(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Reference file SHA-256",
            "",
            "| File | SHA-256 |",
            "|---|---|",
        ]
    )
    for path, digest in payload["reference_sha256"].items():
        lines.append(f"| `{path}` | `{digest}` |")
    lines.append("")
    md_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--generated-at", default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    jascl_repo = root / "third_party" / "JASCL_REFERENCE"
    star_repo = root / "third_party" / "STAR_REFERENCE"
    jascl_base = jascl_repo / "Med_Semi-Supervised-FoSSIL" / "inc" / "Medformer_inc"
    files = {
        "JASCL methods/utils.py": jascl_base / "methods" / "utils.py",
        "JASCL methods/trainer.py": jascl_base / "methods" / "trainer.py",
        "JASCL run.py": jascl_base / "run.py",
        "STAR models/loss.py": star_repo / "models" / "loss.py",
        "STAR models/model.py": star_repo / "models" / "model.py",
        "STAR trainer/trainer_voc.py": star_repo / "trainer" / "trainer_voc.py",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise SystemExit("Missing audited source files: " + ", ".join(missing))

    jascl_utils = files["JASCL methods/utils.py"].read_text(errors="replace")
    jascl_trainer = files["JASCL methods/trainer.py"].read_text(errors="replace")
    jascl_run = files["JASCL run.py"].read_text(errors="replace")
    star_loss = files["STAR models/loss.py"].read_text(errors="replace")
    star_model = files["STAR models/model.py"].read_text(errors="replace")
    star_trainer = files["STAR trainer/trainer_voc.py"].read_text(errors="replace")

    jascl_commit = git_value(jascl_repo, "rev-parse", "HEAD")
    star_commit = git_value(star_repo, "rev-parse", "HEAD")
    checks: dict[str, bool] = {
        "jascl_exact_required_commit": jascl_commit == REQUIRED_JASCL_COMMIT,
        "jascl_origin_official": git_value(jascl_repo, "remote", "get-url", "origin") == "https://github.com/prinshul/JASCL.git",
        "star_origin_official": git_value(star_repo, "remote", "get-url", "origin") == "https://github.com/jinpeng0528/STAR-TPAMI.git",
    }
    require(jascl_utils, "def get_prototype", "jascl_prototype_function", checks)
    require(jascl_utils, "F.normalize(x, dim=1).mean", "jascl_per_pixel_normalize_then_mean", checks)
    require(jascl_utils, "return protos.mean(dim=0)", "jascl_cross_case_mean", checks)
    require(jascl_trainer, "confidence_thresh=0.7", "jascl_confidence_default_0_7", checks)
    require(jascl_trainer, "similarity_thresh=0.7", "jascl_similarity_default_0_7", checks)
    require(jascl_trainer, "(max_probs_flat > confidence_thresh) & (similarity > similarity_thresh)", "jascl_strict_joint_filter", checks)
    require(jascl_trainer, "pseudo_out_tea = self.filter_pseudo_labels", "jascl_teacher_filtered_independently", checks)
    require(jascl_trainer, "mse_loss = F.mse_loss(pseudo_out_tea,pseudo_out)", "jascl_filtered_map_consistency_mse", checks)
    require(jascl_trainer, "def update_teacher(self, alpha=0.99)", "jascl_ema_alpha_0_99", checks)
    require(jascl_run, "if cur_epoch % 25 == 0 and cur_epoch>0", "jascl_periodic_pas_schedule", checks)
    require(star_trainer, "data['label'] == 0, pred > 0", "star_old_class_region_rule", checks)
    require(star_trainer, "idx = (logit_old > 0.5).float()", "star_old_logit_threshold", checks)
    require(star_loss, "self.criterion = nn.MSELoss(reduction='none')", "star_elementwise_mse", checks)
    require(star_loss, "size=features[2].shape[2:], mode=\"bilinear\"", "star_bilinear_region_resize", checks)
    require(star_loss, "self.criterion(features[5], features_old[5])", "star_feature_index_5", checks)
    require(star_loss, "pseudo_label_region_5.sum() * features[5].shape[1]", "star_mask_channel_reduction", checks)
    require(star_model, "features.append(x_pl)", "star_feature_5_is_classifier_input", checks)

    generated_at = args.generated_at or datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "protocol_id": "sparcseg_v0_1",
        "status": "SPARC_SOURCE_AUDIT_PASSED" if all(checks.values()) else "HARD_STOP_SPARC_SOURCE_AUDIT",
        "generated_at": generated_at,
        "jascl": {
            "repository": "https://github.com/prinshul/JASCL",
            "paper": "https://arxiv.org/abs/2605.20538",
            "commit": jascl_commit,
            "required_commit": REQUIRED_JASCL_COMMIT,
            "required_commit_match": jascl_commit == REQUIRED_JASCL_COMMIT,
            "paper_explicit": ["prototype-assisted pseudo-label validation", "joint confidence/prototype validation", "mean-teacher consistency"],
            "source_explicit": ["per-pixel normalize then per-case mean", "cross-case mean without explicit final normalize", "strict confidence/similarity filters", "independent student and teacher filtering", "filtered-label-map MSE", "EMA alpha=0.99"],
            "source_paper_mismatch_or_underspecification": ["no explicit student/teacher Boolean intersection in source", "implementation-specific filtered integer-map MSE", "no SPARC-style minimum-cell or final-normalization rule"],
        },
        "star": {
            "repository": "https://github.com/jinpeng0528/STAR-TPAMI",
            "paper": "https://ieeexplore.ieee.org/document/10904177",
            "commit": star_commit,
            "source_explicit": ["current-background and old-foreground region", "old-logit threshold 0.5", "classifier-input feature index 5", "bilinear mask resize", "masked channel-mean MSE"],
            "all_feature_ablation_source_flag_found": False,
        },
        "lag": {
            "paper": "https://arxiv.org/abs/2407.15429",
            "role": "conceptual inspiration only",
            "implemented_modules": [],
        },
        "sparc_adaptation": {
            "thresholds": {"confidence": 0.7, "similarity": 0.7, "strict_comparison": True},
            "validators": ["current", "frozen_previous"],
            "ema_teacher": False,
            "pseudo_target": "unchanged R0 hard CE",
            "feature_maintaining": "dual-stable foreground channel-normalized cosine at dec3 and dec1",
            "cross_site_prototype_memory": False,
            "copies_star_formula": False,
            "implements_lag_module": False,
        },
        "checks": checks,
        "reference_sha256": {name: sha256(path) for name, path in files.items()},
    }
    write_reports(root / "reports" / "experiment_status", payload)
    print(json.dumps({"status": payload["status"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
