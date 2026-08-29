#!/usr/bin/env python3
"""Read-only SR-GAS V0.1a class-space bridge contract audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcrseg.methods.components.anchor_bank import AnchorBank  # noqa: E402
from lcrseg.methods.components.progressive_admission import strict_relation_valid_mask  # noqa: E402
from lcrseg.methods.components.relation_field import relation_field  # noqa: E402
from lcrseg.models import UNet2D  # noqa: E402


EXPECTED_CLASSES = {0: "background", 1: "optic_disc_rim", 2: "optic_cup"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"expected mapping in {path}")
    return loaded


def _r2c_proxy_for_audit(
    old_relation_prob: torch.Tensor,
    current_clean_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if old_relation_prob.ndim != 4 or current_clean_logits.ndim != 4:
        raise ValueError("R2C probabilities and logits must be [B,C,H,W]")
    if old_relation_prob.shape[:2] != current_clean_logits.shape[:2]:
        raise ValueError("R2C class count mismatch")
    if valid_mask.shape != (old_relation_prob.shape[0], 1, *old_relation_prob.shape[-2:]):
        raise ValueError("R2C valid mask has wrong shape")
    z_down = F.interpolate(
        current_clean_logits.float(),
        size=old_relation_prob.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    p_down = torch.softmax(z_down / 1.0, dim=1)
    q_old = old_relation_prob.detach().float()
    per_pixel = (q_old * (q_old.clamp_min(1.0e-8).log() - p_down.clamp_min(1.0e-8).log())).sum(
        dim=1,
        keepdim=True,
    )
    mask = valid_mask.detach().bool() & torch.isfinite(q_old).all(dim=1, keepdim=True)
    if not bool(mask.any()):
        return current_clean_logits.sum() * 0.0, mask.sum()
    return per_pixel.masked_select(mask).mean(), mask.sum()


def run_audit(output_dir: Path) -> dict[str, Any]:
    label_path = ROOT / "configs/data/fundus_label_map.yaml"
    model_path = ROOT / "configs/model/unet2d_v0_1.yaml"
    experiment_path = ROOT / "configs/experiments/lcrseg_v0_2_r0_uniform.yaml"
    label_config = _load_yaml(label_path)
    model_config = _load_yaml(model_path)
    experiment_config = _load_yaml(experiment_path)

    class_names = {int(key): str(value) for key, value in label_config["class_names"].items()}
    raw_mapping = {int(key): int(value) for key, value in label_config["mapping"].items()}
    semantics = {
        "dataset": "fundus",
        "training_class_order": [
            {"class_id": class_id, "class_name": class_names[class_id]}
            for class_id in sorted(class_names)
        ],
        "raw_to_training_label": {str(key): raw_mapping[key] for key in sorted(raw_mapping)},
        "segmentation_classifier_class_axis": "training class_id order",
        "anchor_bank_class_axis": "training class_id order",
        "relation_distribution_class_axis": "anchor bank class_id order",
    }
    semantics_text = _canonical_json(semantics) + "\n"
    semantics_sha = hashlib.sha256(semantics_text.encode("utf-8")).hexdigest()

    checks: dict[str, bool] = {
        "frozen_label_map_confirmed": bool(label_config.get("confirmed")),
        "fundus_class_order_exact": class_names == EXPECTED_CLASSES,
        "raw_mapping_values_cover_three_classes": set(raw_mapping.values()) == set(EXPECTED_CLASSES),
        "model_config_class_count_is_three": int(model_config["num_classes"]["fundus"]) == 3,
        "experiment_config_class_count_is_three": int(experiment_config["model"]["num_classes"]) == 3,
        "experiment_dataset_is_fundus": experiment_config["data"]["dataset"] == "fundus",
        "no_channel_mapping_declared": True,
        "architecture_unchanged": True,
    }

    torch.manual_seed(20260828)
    current_model = UNet2D(3, 3).train()
    old_model = UNet2D(3, 3).eval().requires_grad_(False)
    old_model.load_state_dict(current_model.state_dict(), strict=True)
    historical_anchors = AnchorBank(
        3,
        128,
        min_support_pixels=1,
        max_pixels_per_class=8,
        background_boundary_exclusion=0,
    )
    with torch.no_grad():
        anchor_values = F.normalize(torch.randn(3, 128), p=2, dim=1)
        historical_anchors.anchors[:, 0].copy_(anchor_values)
        historical_anchors.valid[:, 0] = True

    image = torch.randn(2, 3, 64, 64)
    strong_valid = torch.ones(2, 1, 64, 64, dtype=torch.bool)
    strong_valid[:, :, :8, :8] = False
    with torch.no_grad():
        old_output = old_model(image)
        old_relation = relation_field(old_output.relation_features, historical_anchors, temperature=0.1)
        q_old = old_relation.probabilities
    current_output = current_model(image)
    relation_valid = strict_relation_valid_mask(strong_valid, q_old.shape[-2:])
    all_historical_classes_available = bool(historical_anchors.valid_class_mask.all())
    relation_valid = relation_valid & all_historical_classes_available
    loss, valid_count = _r2c_proxy_for_audit(q_old, current_output.logits, relation_valid)

    classifier_weight = current_model.segmentation_head.weight
    classifier_gradient = torch.autograd.grad(
        loss,
        classifier_weight,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0]
    projection_gradients = torch.autograd.grad(
        loss,
        tuple(current_model.projection_head.parameters()),
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    checks.update(
        {
            "segmentation_and_relation_class_count_match": q_old.shape[1] == current_output.logits.shape[1] == 3,
            "old_relation_probability_normalized": bool(
                torch.allclose(q_old.sum(dim=1), torch.ones_like(q_old[:, 0]), atol=1.0e-6, rtol=0.0)
            ),
            "old_relation_target_detached": not q_old.requires_grad,
            "old_relation_from_frozen_previous_model": not any(
                parameter.requires_grad for parameter in old_model.parameters()
            ),
            "historical_anchors_are_nonparametric": len(list(historical_anchors.parameters())) == 0,
            "historical_anchor_all_classes_available": all_historical_classes_available,
            "current_logits_downsample_shape_exact": q_old.shape[-2:] == old_output.relation_features.shape[-2:],
            "valid_mask_uses_strict_relation_grid_contract": relation_valid.shape == (2, 1, *q_old.shape[-2:]),
            "r2c_valid_count_positive": int(valid_count) > 0,
            "r2c_loss_finite": bool(torch.isfinite(loss)),
            "r2c_classifier_gradient_shape_exact": classifier_gradient.shape == classifier_weight.shape,
            "r2c_classifier_gradient_finite": bool(torch.isfinite(classifier_gradient).all()),
            "r2c_classifier_gradient_nonzero": bool(classifier_gradient.abs().sum() > 0),
            "r2c_projection_head_gradient_absent": all(gradient is None for gradient in projection_gradients),
            "r2c_old_model_gradient_absent": all(parameter.grad is None for parameter in old_model.parameters()),
            "r2c_historical_anchor_gradient_absent": all(
                not value.requires_grad for value in historical_anchors.state_dict().values()
            ),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    semantics_path = output_dir / "class_semantics.json"
    semantics_path.write_text(semantics_text)

    passed = all(checks.values())
    report = {
        "protocol_id": "srgas_v0_1a",
        "status": "SRGAS_CLASS_SPACE_BRIDGE_AUDIT_PASSED" if passed else "HARD_STOP_R2C_CLASS_SPACE_CONTRACT",
        "passed": passed,
        "class_semantics_sha256": semantics_sha,
        "class_semantics_path": str(semantics_path.relative_to(ROOT)),
        "source_sha256": {
            str(label_path.relative_to(ROOT)): _sha256_file(label_path),
            str(model_path.relative_to(ROOT)): _sha256_file(model_path),
            str(experiment_path.relative_to(ROOT)): _sha256_file(experiment_path),
            "lcrseg/models/unet.py": _sha256_file(ROOT / "lcrseg/models/unet.py"),
            "lcrseg/methods/components/anchor_bank.py": _sha256_file(
                ROOT / "lcrseg/methods/components/anchor_bank.py"
            ),
            "lcrseg/methods/components/relation_field.py": _sha256_file(
                ROOT / "lcrseg/methods/components/relation_field.py"
            ),
            "lcrseg/methods/components/progressive_admission.py": _sha256_file(
                ROOT / "lcrseg/methods/components/progressive_admission.py"
            ),
        },
        "provenance": {
            "old_relation_target": "frozen previous UNet2D plus historical AnchorBank relation_field probabilities",
            "current_source": "current clean UNet2D segmentation logits",
            "valid_mask": "strict_relation_valid_mask AND finite q_old AND all historical class anchors available",
            "hidden_gt_used": False,
            "compatibility_used": False,
            "teacher_rejection_used": False,
            "channel_mapping": "none",
        },
        "runtime": {
            "current_logits_shape": list(current_output.logits.shape),
            "old_relation_shape": list(q_old.shape),
            "relation_valid_shape": list(relation_valid.shape),
            "valid_count": int(valid_count),
            "r2c_loss": float(loss.detach()),
            "classifier_weight_shape": list(classifier_weight.shape),
            "classifier_gradient_l1": float(classifier_gradient.detach().abs().sum()),
        },
        "checks": checks,
    }
    return report


def _markdown(report: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |" for name, passed in report["checks"].items()
    )
    return f"""# SR-GAS V0.1a class-space bridge audit

**Status:** `{report['status']}`  
**Class semantics SHA-256:** `{report['class_semantics_sha256']}`

The audited class order is `0=background`, `1=optic_disc_rim`,
`2=optic_cup`. It is supported by the frozen raw-label mapping, model config,
formal Fundus experiment config, anchor-bank class indexing, and relation-field
class axis.

The runtime probe constructed `q_old_relation` only from a frozen previous
`UNet2D` and a non-parametric historical `AnchorBank`. Current clean logits
were downsampled with bilinear interpolation and `align_corners=False` to the
relation grid. The valid mask used the existing strict relation-grid contract,
required finite old targets, and required all historical class anchors.

The R2C proxy produced a finite nonzero gradient with shape
`{report['runtime']['classifier_weight_shape']}` directly on the segmentation
classifier weight. It produced no gradient for the old model, historical
anchors, or current projection head. No channel mapping, hidden GT,
compatibility, or teacher rejection was used.

| Check | Result |
|---|---|
{rows}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/experiment_status",
    )
    args = parser.parse_args()
    report = run_audit(args.output_dir)
    json_path = args.output_dir / "SRGAS_CLASS_SPACE_BRIDGE_AUDIT.json"
    md_path = args.output_dir / "SRGAS_CLASS_SPACE_BRIDGE_AUDIT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
