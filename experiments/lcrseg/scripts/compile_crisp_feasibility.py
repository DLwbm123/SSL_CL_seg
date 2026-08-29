#!/usr/bin/env python3
"""Compile the preregistered CRISP A-D feasibility gates across three seeds."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.common import sha256_path, write_csv, write_json, write_text  # noqa: E402
from scripts.audit_aspr_relation_space import _workspace_hash  # noqa: E402


SEEDS = (0, 1, 2)
LAYERS = ("dec3", "dec1")
EXPECTED_PAIRS_PER_LAYER = 6
EXPECTED_GRADIENT_ROWS = 3 * 2 * 32
EXPECTED_VIRTUAL_ROWS = EXPECTED_GRADIENT_ROWS * 6


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _bool(row: dict[str, str], key: str) -> bool:
    return row[key].strip().lower() in {"true", "1", "yes"}


def _median(values: Iterable[float]) -> float:
    sequence = list(values)
    if not sequence:
        raise ValueError("median of empty gate input")
    return float(statistics.median(sequence))


def _quantile(values: Iterable[float], q: float) -> float:
    sequence = np.asarray(list(values), dtype=np.float64)
    if not sequence.size:
        raise ValueError("quantile of empty gate input")
    return float(np.quantile(sequence, q))


def _layer_rows(rows: list[dict[str, str]], layer: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["layer"] == layer]
    if len(selected) != EXPECTED_PAIRS_PER_LAYER:
        raise ValueError(f"{layer} expected {EXPECTED_PAIRS_PER_LAYER} pairs, found {len(selected)}")
    return selected


def _role_gates(
    nondegenerate: list[dict[str, str]],
    reproducibility: list[dict[str, str]],
) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for layer in LAYERS:
        nondeg = _layer_rows(nondegenerate, layer)
        repro = _layer_rows(reproducibility, layer)
        pair_passes = [
            0.20 <= _float(row, "mean_alpha") <= 0.80
            and _float(row, "alpha_iqr") >= 0.10
            and _float(row, "ess_alpha_over_d") >= 0.25
            and _float(row, "ess_beta_over_d") >= 0.25
            for row in nondeg
        ]
        spearman = [_float(row, "spearman_alpha_a_b") for row in repro]
        top_jaccard = [_float(row, "top_quartile_jaccard") for row in repro]
        bottom_jaccard = [_float(row, "bottom_quartile_jaccard") for row in repro]
        layers[layer] = {
            "nondegenerate_pair_pass_count": sum(pair_passes),
            "nondegenerate_pass": sum(pair_passes) >= 5,
            "median_spearman": _median(spearman),
            "spearman_ge_0_30_count": sum(value >= 0.30 for value in spearman),
            "median_top_quartile_jaccard": _median(top_jaccard),
            "median_bottom_quartile_jaccard": _median(bottom_jaccard),
            "reproducibility_pass": _median(spearman) >= 0.50
            and sum(value >= 0.30 for value in spearman) >= 5
            and _median(top_jaccard) >= 0.40
            and _median(bottom_jaccard) >= 0.40,
        }
    return {
        "layers": layers,
        "nondegeneracy_pass": all(value["nondegenerate_pass"] for value in layers.values()),
        "reproducibility_pass": all(value["reproducibility_pass"] for value in layers.values()),
    }


def _semantic_style_gates(rows: list[dict[str, str]]) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for layer in LAYERS:
        selected = _layer_rows(rows, layer)
        fisher = [_float(row, "fisher_top_bottom_ratio") for row in selected]
        style = [_float(row, "style_plastic_stable_ratio") for row in selected]
        content_complete = all(
            _bool(row, "foreground_class_1_nonzero") and _bool(row, "foreground_class_2_nonzero")
            for row in selected
        )
        style_complete = all(_bool(row, "plastic_alive_pass") for row in selected)
        layers[layer] = {
            "median_fisher_top_bottom_ratio": _median(fisher),
            "fisher_positive_pair_count": sum(value > 1.0 for value in fisher),
            "content_complete": content_complete,
            "content_pass": _median(fisher) >= 1.10 and sum(value > 1.0 for value in fisher) >= 4 and content_complete,
            "median_style_plastic_stable_ratio": _median(style),
            "style_positive_pair_count": sum(value > 1.0 for value in style),
            "minimum_plastic_alive_fraction": min(_float(row, "plastic_alive_fraction") for row in selected),
            "style_complete": style_complete,
            "style_pass": _median(style) >= 1.10 and sum(value > 1.0 for value in style) >= 4 and style_complete,
        }
    return {
        "layers": layers,
        "content_pass": all(value["content_pass"] for value in layers.values()),
        "style_pass": all(value["style_pass"] for value in layers.values()),
    }


def _gradient_gates(rows: list[dict[str, str]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_GRADIENT_ROWS:
        raise ValueError(f"expected {EXPECTED_GRADIENT_ROWS} gradient rows, found {len(rows)}")
    ifc = [_float(row, "ifc_to_relation_ratio") for row in rows]
    pfc = [_float(row, "pfc_to_assimilation_ratio") for row in rows]
    total = [_float(row, "c3_to_c0_total_ratio") for row in rows]
    finite_count = sum(
        _bool(row, "finite")
        and all(math.isfinite(value) for value in (_float(row, "ifc_to_relation_ratio"), _float(row, "pfc_to_assimilation_ratio"), _float(row, "c3_to_c0_total_ratio")))
        for row in rows
    )
    old_gradient_nonnull = sum(int(float(row["old_model_gradient_nonnull"])) for row in rows)
    ifc_summary = {"median": _median(ifc), "p10": _quantile(ifc, 0.10), "p90": _quantile(ifc, 0.90)}
    pfc_summary = {"median": _median(pfc), "p10": _quantile(pfc, 0.10), "p90": _quantile(pfc, 0.90)}
    total_summary = {"median": _median(total), "p90": _quantile(total, 0.90)}
    ifc_pass = 0.10 <= ifc_summary["median"] <= 1.50 and ifc_summary["p10"] >= 0.02 and ifc_summary["p90"] <= 3.0
    pfc_pass = 0.10 <= pfc_summary["median"] <= 1.50 and pfc_summary["p10"] >= 0.02 and pfc_summary["p90"] <= 3.0
    total_pass = 0.75 <= total_summary["median"] <= 2.00 and total_summary["p90"] <= 3.0
    engineering_pass = finite_count == len(rows) and old_gradient_nonnull == 0
    return {
        "ifc": {**ifc_summary, "pass": ifc_pass},
        "pfc": {**pfc_summary, "pass": pfc_pass},
        "total": {**total_summary, "pass": total_pass},
        "finite_rows": finite_count,
        "nonfinite_rows": len(rows) - finite_count,
        "old_model_gradient_nonnull": old_gradient_nonnull,
        "engineering_pass": engineering_pass,
        "pass": ifc_pass and pfc_pass and total_pass and engineering_pass,
    }


def _virtual_gates(rows: list[dict[str, str]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_VIRTUAL_ROWS:
        raise ValueError(f"expected {EXPECTED_VIRTUAL_ROWS} virtual rows, found {len(rows)}")
    indexed: dict[tuple[int, str, int, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["seed"]), row["transition"], int(row["update_batch"]), row["variant"])
        if key in indexed:
            raise ValueError(f"duplicate virtual row: {key}")
        indexed[key] = row
    comparison_keys = sorted({key[:3] for key in indexed})
    if len(comparison_keys) != EXPECTED_GRADIENT_ROWS:
        raise ValueError("virtual comparison coverage is incomplete")

    def values(variant: str, field: str) -> list[float]:
        return [_float(indexed[(*key, variant)], field) for key in comparison_keys]

    def better_fraction(candidate: str, baseline: str, field: str) -> float:
        candidate_values, baseline_values = values(candidate, field), values(baseline, field)
        return sum(left < right for left, right in zip(candidate_values, baseline_values, strict=True)) / len(comparison_keys)

    c3_prev = _median(values("C3", "previous_val_loss_delta"))
    c3_curr = _median(values("C3", "current_val_loss_delta"))
    c3_prev_dice = _median(values("C3", "previous_val_dice_delta"))
    c3_curr_dice = _median(values("C3", "current_val_dice_delta"))
    medians: dict[str, dict[str, float]] = {}
    for variant in ("C0", "C1", "C2", "C3", "C4", "C5"):
        medians[variant] = {
            "previous_loss_delta": _median(values(variant, "previous_val_loss_delta")),
            "current_loss_delta": _median(values(variant, "current_val_loss_delta")),
            "previous_dice_delta": _median(values(variant, "previous_val_dice_delta")),
            "current_dice_delta": _median(values(variant, "current_val_dice_delta")),
        }
    d1 = {
        "previous_loss_better_fraction": better_fraction("C3", "C0", "previous_val_loss_delta"),
        "previous_loss_margin": medians["C0"]["previous_loss_delta"] - c3_prev,
        "current_loss_excess": c3_curr - medians["C0"]["current_loss_delta"],
        "current_dice_difference": c3_curr_dice - medians["C0"]["current_dice_delta"],
    }
    d1["pass"] = (
        d1["previous_loss_better_fraction"] >= 0.60
        and c3_prev <= medians["C0"]["previous_loss_delta"] - 1.0e-4
        and c3_curr <= medians["C0"]["current_loss_delta"] + 0.02 * abs(medians["C0"]["current_loss_delta"])
        and c3_curr_dice >= medians["C0"]["current_dice_delta"] - 0.002
    )
    d2 = {
        "previous_loss_better_fraction": better_fraction("C3", "C2", "previous_val_loss_delta"),
        "previous_loss_margin": medians["C2"]["previous_loss_delta"] - c3_prev,
        "current_dice_difference": c3_curr_dice - medians["C2"]["current_dice_delta"],
    }
    d2["pass"] = (
        d2["previous_loss_better_fraction"] >= 0.60
        and c3_prev <= medians["C2"]["previous_loss_delta"] - 1.0e-4
        and c3_curr_dice >= medians["C2"]["current_dice_delta"] - 0.002
    )
    d3 = {
        "current_loss_better_fraction": better_fraction("C3", "C1", "current_val_loss_delta"),
        "current_loss_margin": medians["C1"]["current_loss_delta"] - c3_curr,
        "previous_dice_difference": c3_prev_dice - medians["C1"]["previous_dice_delta"],
    }
    d3["pass"] = (
        d3["current_loss_better_fraction"] >= 0.55
        and c3_curr <= medians["C1"]["current_loss_delta"] - 5.0e-5
        and c3_prev_dice >= medians["C1"]["previous_dice_delta"] - 0.002
    )

    def continuous_control(control: str) -> dict[str, Any]:
        previous_margin = medians[control]["previous_loss_delta"] - c3_prev
        current_margin = medians[control]["current_loss_delta"] - c3_curr
        return {
            "control": control,
            "previous_loss_margin": previous_margin,
            "current_loss_margin": current_margin,
            "pass": c3_prev <= medians[control]["previous_loss_delta"]
            and c3_curr <= medians[control]["current_loss_delta"]
            and max(previous_margin, current_margin) >= 5.0e-5,
        }

    d4_c4, d4_c5 = continuous_control("C4"), continuous_control("C5")
    finite = all(
        math.isfinite(float(row[field]))
        for row in rows
        for field in (
            "raw_gradient_norm",
            "previous_val_loss_delta",
            "previous_val_dice_delta",
            "current_val_loss_delta",
            "current_val_dice_delta",
        )
    )
    no_mutation = all(not _bool(row, "checkpoint_or_model_mutated") for row in rows)
    return {
        "medians": medians,
        "D1_full_vs_C0": d1,
        "D2_ifc_contribution_vs_C2": d2,
        "D3_pfc_contribution_vs_C1": d3,
        "D4_continuous_vs_C4": d4_c4,
        "D4_continuous_vs_C5": d4_c5,
        "finite": finite,
        "no_mutation": no_mutation,
        "pass": bool(d1["pass"] and d2["pass"] and d3["pass"] and d4_c4["pass"] and d4_c5["pass"] and finite and no_mutation),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "reports" / "analysis" / "crispseg_v0_1")
    parser.add_argument("--status-dir", type=Path, default=ROOT / "reports" / "experiment_status")
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    status_dir = args.status_dir.resolve()
    output_csvs = (
        "role_non_degeneracy.csv",
        "role_reproducibility.csv",
        "semantic_style_validation.csv",
        "gradient_scale.csv",
        "virtual_steps.csv",
    )
    report_json = status_dir / "CRISP_FEASIBILITY_AUDIT.json"
    report_md = status_dir / "CRISP_FEASIBILITY_AUDIT.md"
    for path in [report_json, report_md, *(analysis_dir / name for name in output_csvs)]:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite CRISP feasibility output: {path}")
    role_rows: list[dict[str, str]] = []
    repro_rows: list[dict[str, str]] = []
    semantic_rows: list[dict[str, str]] = []
    gradient_rows: list[dict[str, str]] = []
    virtual_rows: list[dict[str, str]] = []
    seed_summaries: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_dir = args.input_dir.resolve() / f"seed{seed}"
        roles_summary_path = seed_dir / "roles_summary.json"
        functional_summary_path = seed_dir / "functional_summary.json"
        if not roles_summary_path.is_file() or not functional_summary_path.is_file():
            raise FileNotFoundError(f"seed{seed} feasibility phases are incomplete")
        roles_summary = json.loads(roles_summary_path.read_text())
        functional_summary = json.loads(functional_summary_path.read_text())
        if roles_summary["status"] != "CRISP_ROLE_PHASE_COMPLETE" or functional_summary["status"] != "CRISP_FUNCTIONAL_PHASE_COMPLETE":
            raise RuntimeError(f"seed{seed} has a non-final phase status")
        if roles_summary["optimizer_steps"] != 0 or functional_summary["optimizer_steps"] != 0:
            raise RuntimeError("feasibility audit unexpectedly reports optimizer steps")
        role_rows.extend(_rows(seed_dir / "role_non_degeneracy.csv"))
        repro_rows.extend(_rows(seed_dir / "role_reproducibility.csv"))
        semantic_rows.extend(_rows(seed_dir / "semantic_style_validation.csv"))
        gradient_rows.extend(_rows(seed_dir / "gradient_scale.csv"))
        virtual_rows.extend(_rows(seed_dir / "virtual_steps.csv"))
        seed_summaries.append(
            {
                "seed": seed,
                "roles_summary": str(roles_summary_path),
                "roles_summary_sha256": sha256_path(roles_summary_path),
                "functional_summary": str(functional_summary_path),
                "functional_summary_sha256": sha256_path(functional_summary_path),
                "role_status": roles_summary["status"],
                "functional_status": functional_summary["status"],
                "physical_gpu_roles": roles_summary["environment"]["physical_gpu"],
                "physical_gpu_functional": functional_summary["environment"]["physical_gpu"],
            }
        )
    gate_a = _role_gates(role_rows, repro_rows)
    gate_b = _semantic_style_gates(semantic_rows)
    gate_c = _gradient_gates(gradient_rows)
    gate_d = _virtual_gates(virtual_rows)
    freeze = json.loads((status_dir / "SPARC_V0_1_FREEZE_FOR_CRISP.json").read_text())
    declarations = freeze["declarations"]
    prerequisites = {
        "sparc_freeze": freeze["status"] == "SPARC_V0_1_FROZEN_FOR_CRISP"
        and declarations["sparc_status"] == "SPARC_PAS_NOT_SUPPORTED"
        and declarations["feasibility_method_registration_allowed"] is False
        and declarations["training_authorization_status"] == "CRISP_FEASIBILITY_SUPPORTED",
        "source_audit": json.loads((status_dir / "CRISP_SOURCE_AUDIT.json").read_text())["status"] == "CRISP_SOURCE_AUDIT_PASSED",
        "model_path": json.loads((status_dir / "CRISP_MODEL_PATH_REVALIDATION.json").read_text())["status"] == "CRISP_MODEL_PATH_REVALIDATION_PASSED",
        "style_probe": json.loads((status_dir / "CRISP_STYLE_PROBE_AUDIT.json").read_text())["status"] == "CRISP_STYLE_PROBE_AUDIT_PASSED",
        "method_not_registered_before_gate": not (ROOT / "lcrseg" / "methods" / "crispseg_v0_1.py").exists(),
        "configs_not_registered_before_gate": not any((ROOT / "configs" / "experiments").glob("crisp_c*.yaml")),
    }
    if not all(prerequisites.values()):
        status = "HARD_STOP_CRISP_PREREQUISITE"
    elif not gate_a["nondegeneracy_pass"] or not gate_a["reproducibility_pass"]:
        status = "CRISP_ROLE_NOT_REPRODUCIBLE"
    elif not gate_b["content_pass"]:
        status = "CRISP_CONTENT_ROLE_NOT_SUPPORTED"
    elif not gate_b["style_pass"]:
        status = "CRISP_STYLE_ROLE_NOT_SUPPORTED"
    elif not gate_c["pass"]:
        status = "CRISP_GRADIENT_SCALE_NOT_SUPPORTED"
    elif not gate_d["pass"]:
        status = "CRISP_FEASIBILITY_NOT_SUPPORTED"
    else:
        status = "CRISP_FEASIBILITY_SUPPORTED"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_dir / "role_non_degeneracy.csv", role_rows)
    write_csv(analysis_dir / "role_reproducibility.csv", repro_rows)
    write_csv(analysis_dir / "semantic_style_validation.csv", semantic_rows)
    write_csv(analysis_dir / "gradient_scale.csv", gradient_rows)
    write_csv(analysis_dir / "virtual_steps.csv", virtual_rows)
    output_hashes = {name: sha256_path(analysis_dir / name) for name in output_csvs}
    payload = {
        "protocol_id": "crispseg_v0_1",
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "optimizer_steps": 0,
        "hidden_gt_usage": "none",
        "method_registered": False,
        "config_registered": False,
        "prerequisites": prerequisites,
        "gate_A": gate_a,
        "gate_B": gate_b,
        "gate_C": gate_c,
        "gate_D": gate_d,
        "seed_summaries": seed_summaries,
        "row_counts": {
            "role_non_degeneracy": len(role_rows),
            "role_reproducibility": len(repro_rows),
            "semantic_style_validation": len(semantic_rows),
            "gradient_scale": len(gradient_rows),
            "virtual_steps": len(virtual_rows),
        },
        "analysis_dir": str(analysis_dir),
        "analysis_sha256": output_hashes,
        "workspace_hash": _workspace_hash(),
    }
    write_json(report_json, payload)
    lines = [
        "# CRISP-Seg V0.1 feasibility audit",
        "",
        f"**Status:** `{status}`  ",
        "**Optimizer steps:** `0`  ",
        "**Hidden-GT usage:** `none`  ",
        "**Method/config registration:** `none`",
        "",
        "## Gate A — channel roles",
        "",
    ]
    for layer in LAYERS:
        value = gate_a["layers"][layer]
        lines.append(
            f"- `{layer}`: nondegenerate `{value['nondegenerate_pair_pass_count']}/6`; median Spearman `{value['median_spearman']:.6f}`; top/bottom Jaccard `{value['median_top_quartile_jaccard']:.6f}` / `{value['median_bottom_quartile_jaccard']:.6f}`; pass `{value['nondegenerate_pass'] and value['reproducibility_pass']}`."
        )
    lines.extend(["", "## Gate B — independent validation", ""])
    for layer in LAYERS:
        value = gate_b["layers"][layer]
        lines.append(
            f"- `{layer}`: Fisher ratio `{value['median_fisher_top_bottom_ratio']:.6f}` ({value['fisher_positive_pair_count']}/6 positive); style ratio `{value['median_style_plastic_stable_ratio']:.6f}` ({value['style_positive_pair_count']}/6 positive); content/style pass `{value['content_pass']}` / `{value['style_pass']}`."
        )
    lines.extend(
        [
            "",
            "## Gate C — gradient scale",
            "",
            f"- IFC ratio median/p10/p90: `{gate_c['ifc']['median']:.6f}` / `{gate_c['ifc']['p10']:.6f}` / `{gate_c['ifc']['p90']:.6f}`; pass `{gate_c['ifc']['pass']}`.",
            f"- PFC ratio median/p10/p90: `{gate_c['pfc']['median']:.6f}` / `{gate_c['pfc']['p10']:.6f}` / `{gate_c['pfc']['p90']:.6f}`; pass `{gate_c['pfc']['pass']}`.",
            f"- C3/C0 total median/p90: `{gate_c['total']['median']:.6f}` / `{gate_c['total']['p90']:.6f}`; nonfinite `{gate_c['nonfinite_rows']}`; pass `{gate_c['total']['pass']}`.",
            "",
            "## Gate D — stateless virtual step",
            "",
            f"- D1 C3 vs C0: `{gate_d['D1_full_vs_C0']['pass']}`",
            f"- D2 IFC contribution: `{gate_d['D2_ifc_contribution_vs_C2']['pass']}`",
            f"- D3 PFC contribution: `{gate_d['D3_pfc_contribution_vs_C1']['pass']}`",
            f"- D4 continuous vs C4/C5: `{gate_d['D4_continuous_vs_C4']['pass']}` / `{gate_d['D4_continuous_vs_C5']['pass']}`",
            "",
            "No CRISP method or training configuration was registered unless and until the exact supported status is emitted.",
            "",
        ]
    )
    write_text(report_md, "\n".join(lines))
    print(json.dumps({"status": status, "optimizer_steps": 0, "report": str(report_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
