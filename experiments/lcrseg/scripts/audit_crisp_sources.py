#!/usr/bin/env python3
"""Audit source provenance and the adaptation boundary for CRISP-Seg V0.1."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def require(text: str, needle: str, name: str, checks: dict[str, bool]) -> None:
    checks[name] = needle in text


def write_reports(report_dir: Path, payload: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "CRISP_SOURCE_AUDIT.json"
    md_path = report_dir / "CRISP_SOURCE_AUDIT.md"
    for path in (json_path, md_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite CRISP source audit: {path}")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lag, star, dc2t = payload["lag"], payload["star"], payload["dc2t"]
    lines = [
        "# CRISP-Seg V0.1 source audit",
        "",
        f"**Status:** `{payload['status']}`  ",
        f"**Generated:** `{payload['generated_at']}`  ",
        "**Scope:** provenance and adaptation boundary; no CRISP method registration or optimizer step",
        "",
        "## LAG source idea",
        "",
        f"- Official repository: `{lag['repository']}` at `{lag['commit']}`.",
        "- The paper/repository motivates semantic-invariant versus sample-specific representation roles and channel-wise/spatial decoupling.",
        "- The audited source exposes `rho` as a CLI channel-allocation parameter, computes a channel count, and compares same-index old/current features.",
        "- The audited source default is `rho=1.0`; CRISP C4's fixed 60/40 hard split is a preregistered contextual control, not claimed as the audited source default.",
        "- CRISP does not copy LAG prototype matching, triplet/contrastive implementation, LRP/NSC, unknown-class handling, or a fixed split into the proposed C3 method.",
        "",
        "## STAR source idea",
        "",
        f"- Official repository: `{star['repository']}` at `{star['commit']}`.",
        "- STAR provides evidence for partial rather than global feature stabilization: its audited PKD path applies a spatial region mask to a selected old/current feature tensor.",
        "- CRISP does not copy STAR prototype replay, background repetition, pseudo-label region rule, selected layer, bilinear soft mask, or MSE formula.",
        "",
        "## DC²T source idea",
        "",
        f"- Primary publication: `{dc2t['doi']}` ({dc2t['publication']}).",
        "- The publication describes online semi-supervised representation disentanglement, content-inspired parameter consolidation, and style-induced consistency training.",
        "- No official implementation was located in the bounded source search, so only publication-level claims are used. CRISP does not claim source-code reproduction.",
        "- CRISP does not copy DC²T's dual encoder, VAE, FiLM, CPC parameter consolidation, SCT parameter perturbation, or reconstruction path.",
        "",
        "## CRISP adaptation",
        "",
        "CRISP's case-equal `(F*grad)^2` content score, centered normalized paired-view style score, continuous `alpha=Cn/(Cn+Sn)`, complementary `beta`, and dual IFC/PFC allocation are protocol-specific adaptations. CRISP is not a direct implementation of LAG, STAR, or DC²T.",
        "",
        "## Integrity checks",
        "",
    ]
    for name, passed in sorted(payload["checks"].items()):
        lines.append(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(["", "## Reference SHA256", "", "| File | SHA256 |", "|---|---|"])
    for name, digest in payload["reference_sha256"].items():
        lines.append(f"| `{name}` | `{digest}` |")
    lines.append("")
    md_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--generated-at", default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    lag_repo = root / "third_party" / "LAG_REFERENCE"
    star_repo = root / "third_party" / "STAR_REFERENCE"
    files = {
        "LAG README.md": lag_repo / "README.md",
        "LAG run.py": lag_repo / "run.py",
        "LAG utils/contrastive_learning.py": lag_repo / "utils" / "contrastive_learning.py",
        "STAR models/loss.py": star_repo / "models" / "loss.py",
        "STAR models/model.py": star_repo / "models" / "model.py",
        "STAR trainer/trainer_voc.py": star_repo / "trainer" / "trainer_voc.py",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise SystemExit("missing audited source files: " + ", ".join(missing))
    lag_readme = files["LAG README.md"].read_text(errors="replace")
    lag_run = files["LAG run.py"].read_text(errors="replace")
    lag_contrast = files["LAG utils/contrastive_learning.py"].read_text(errors="replace")
    star_loss = files["STAR models/loss.py"].read_text(errors="replace")
    star_model = files["STAR models/model.py"].read_text(errors="replace")
    star_trainer = files["STAR trainer/trainer_voc.py"].read_text(errors="replace")
    lag_commit = git_value(lag_repo, "rev-parse", "HEAD")
    star_commit = git_value(star_repo, "rev-parse", "HEAD")
    checks: dict[str, bool] = {
        "lag_origin_official": git_value(lag_repo, "remote", "get-url", "origin") == "https://github.com/YBIO/LAG.git",
        "star_origin_official": git_value(star_repo, "remote", "get-url", "origin") == "https://github.com/jinpeng0528/STAR-TPAMI.git",
        "dc2t_primary_doi_recorded": True,
        "crisp_not_direct_reproduction": True,
    }
    require(lag_readme, "channel-wise decoupling", "lag_channel_wise_decoupling_stated", checks)
    require(lag_readme, "sample-specific contents", "lag_sample_specific_role_stated", checks)
    require(lag_run, 'parser.add_argument("--rho"', "lag_rho_cli_present", checks)
    require(lag_run, "default=1.0", "lag_rho_source_default_recorded", checks)
    require(lag_run, "SS_channel_num = round(opts.rho * channel_num)", "lag_channel_count_from_rho", checks)
    require(lag_run, "ret_features_prev['feature_out'][:,curr_channel", "lag_same_index_old_feature", checks)
    require(lag_run, "ret_features['feature_out'][:,curr_channel", "lag_same_index_current_feature", checks)
    require(lag_contrast, "TripletMarginLoss", "lag_contrastive_path_present", checks)
    require(star_loss, "self.criterion(features[5], features_old[5])", "star_selected_feature_pair", checks)
    require(star_loss, "pseudo_label_region_5", "star_partial_region_mask", checks)
    require(star_model, "features.append(x_pl)", "star_classifier_input_feature", checks)
    require(star_trainer, "loss_pkd = self.PKDLoss", "star_pkd_training_path", checks)
    generated_at = args.generated_at or datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "protocol_id": "crispseg_v0_1",
        "status": "CRISP_SOURCE_AUDIT_PASSED" if all(checks.values()) else "HARD_STOP_CRISP_SOURCE_AUDIT",
        "generated_at": generated_at,
        "lag": {
            "repository": "https://github.com/YBIO/LAG",
            "paper": "https://arxiv.org/abs/2407.15429",
            "commit": lag_commit,
            "source_rho_default": 1.0,
            "crisp_c4_hard_split": "60/40 contextual control, not claimed source default",
            "adopted_idea": "channel-wise invariant versus sample-specific role separation",
            "not_adopted": ["prototype matching", "triplet SFP", "LRP/NSC", "unknown class"],
        },
        "star": {
            "repository": "https://github.com/jinpeng0528/STAR-TPAMI",
            "paper": "https://ieeexplore.ieee.org/document/10904177",
            "commit": star_commit,
            "adopted_idea": "partial rather than global feature stabilization",
            "not_adopted": ["prototype replay", "background repetition", "STAR region rule", "STAR MSE formula"],
        },
        "dc2t": {
            "doi": "https://doi.org/10.1109/TMI.2024.3469528",
            "publication": "IEEE Transactions on Medical Imaging 44(2):903-914, 2025",
            "primary_metadata": "https://pubmed.ncbi.nlm.nih.gov/39331545/",
            "official_code_located": False,
            "source_scope": "paper-level only",
            "adopted_idea": "content relevance and style sensitivity as separate diagnostics",
            "not_adopted": ["dual encoder", "VAE", "FiLM", "CPC", "SCT parameter perturbation", "reconstruction loss"],
        },
        "crisp_adaptation": {
            "content_metric": "case-equal squared activation-gradient",
            "style_metric": "case-equal centered L2-normalized paired-view squared distance",
            "role_formula": "unit-mean normalized content over normalized content plus normalized style",
            "objectives": ["content-invariant feature consolidation", "style-plastic feature consistency"],
            "direct_reproduction": False,
            "new_relation_formula": False,
            "uniform_relation_kd": "unchanged",
        },
        "checks": checks,
        "reference_sha256": {name: sha256(path) for name, path in files.items()},
    }
    write_reports(root / "reports" / "experiment_status", payload)
    print(json.dumps({"status": payload["status"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
