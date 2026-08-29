from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage

from .common import PROJECT_ROOT, read_csv, utc_now, write_csv, write_json, write_text


MANIFEST_ROOT = PROJECT_ROOT / "manifests"
AUDIT_ROOT = PROJECT_ROOT / "reports" / "data_audit"
QC_ROOT = PROJECT_ROOT / "reports" / "qc_overlays" / "prostate_geometry"


def _direction(affine: np.ndarray) -> np.ndarray:
    basis = np.asarray(affine, dtype=float)[:3, :3]
    norms = np.linalg.norm(basis, axis=0)
    if np.any(norms == 0):
        raise ValueError("degenerate affine direction")
    return basis / norms


def _index_geometry_compatible(image: nib.spatialimages.SpatialImage, label: nib.spatialimages.SpatialImage) -> bool:
    """Whether a header-only geometry repair is preferred by the frozen rule."""
    return bool(
        tuple(image.shape) == tuple(label.shape)
        and nib.aff2axcodes(image.affine) == nib.aff2axcodes(label.affine)
        and np.allclose(image.header.get_zooms()[:3], label.header.get_zooms()[:3], rtol=0, atol=1e-4)
        and np.allclose(_direction(image.affine), _direction(label.affine), rtol=0, atol=1e-4)
    )


def _boundary_fraction(foreground: np.ndarray) -> float:
    if not foreground.any():
        return 1.0
    boundary = np.zeros(foreground.shape, dtype=bool)
    for axis in range(foreground.ndim):
        lower = [slice(None)] * foreground.ndim
        upper = [slice(None)] * foreground.ndim
        lower[axis] = 0
        upper[axis] = -1
        boundary[tuple(lower)] = True
        boundary[tuple(upper)] = True
    return float((foreground & boundary).sum() / foreground.sum())


def candidate_metrics(
    image_data: np.ndarray,
    candidate_label: np.ndarray | None,
    *,
    original_label: np.ndarray,
    original_voxel_volume: float,
    candidate_voxel_volume: float,
) -> dict[str, Any]:
    """Deterministic, intentionally conservative QC for geometry candidates."""
    if candidate_label is None or candidate_label.shape != image_data.shape:
        return {
            "candidate_available": False,
            "foreground_nonempty": False,
            "foreground_in_fov": False,
            "body_support_overlap": 0.0,
            "volume_retention": 0.0,
            "boundary_contact": 1.0,
            "axial_continuity": 0.0,
            "component_count": 0,
            "normalized_centroid_distance": float("inf"),
            "qc_pass": False,
            "score": -999.0,
        }
    foreground = np.asarray(candidate_label) > 0
    original_foreground = np.asarray(original_label) > 0
    foreground_count = int(foreground.sum())
    original_count = int(original_foreground.sum())
    support = np.isfinite(image_data) & (np.abs(image_data) > 1e-8)
    body_overlap = float((foreground & support).sum() / foreground_count) if foreground_count else 0.0
    volume_retention = (
        float(foreground_count * candidate_voxel_volume / (original_count * original_voxel_volume))
        if original_count and original_voxel_volume > 0
        else 0.0
    )
    z_present = np.flatnonzero(foreground.any(axis=(0, 1)))
    if z_present.size <= 1:
        continuity = 1.0 if z_present.size else 0.0
    else:
        continuity = float(z_present.size / (z_present[-1] - z_present[0] + 1))
    components = int(ndimage.label(foreground)[1]) if foreground_count else 0
    support_points = np.argwhere(support)
    if foreground_count:
        centroid = np.asarray(ndimage.center_of_mass(foreground), dtype=float)
        if support_points.size:
            support_center = (support_points.min(axis=0) + support_points.max(axis=0)) / 2.0
        else:
            support_center = (np.asarray(foreground.shape, dtype=float) - 1.0) / 2.0
        centroid_distance = float(np.linalg.norm(centroid - support_center) / np.linalg.norm(np.asarray(foreground.shape, dtype=float)))
    else:
        centroid_distance = float("inf")
    boundary = _boundary_fraction(foreground)
    qc_pass = bool(
        foreground_count > 0
        and (not support.any() or body_overlap >= 0.01)
        and 0.05 <= volume_retention <= 20.0
        and boundary < 0.98
        and continuity >= 0.05
        and components <= 1000
        and centroid_distance <= 1.25
    )
    score = (
        4.0 * float(foreground_count > 0)
        + min(body_overlap, 1.0)
        + min(volume_retention, 1.0 / max(volume_retention, 1e-8))
        + (1.0 - min(boundary, 1.0))
        + continuity
        + (1.0 - min(centroid_distance, 1.0))
        - min(components, 100) / 100.0
    )
    return {
        "candidate_available": True,
        "foreground_nonempty": bool(foreground_count > 0),
        "foreground_in_fov": bool(foreground.shape == image_data.shape),
        "body_support_overlap": round(body_overlap, 8),
        "volume_retention": round(volume_retention, 8),
        "boundary_contact": round(boundary, 8),
        "axial_continuity": round(continuity, 8),
        "component_count": components,
        "normalized_centroid_distance": round(centroid_distance, 8),
        "qc_pass": qc_pass,
        "score": round(float(score), 8),
    }


def _save_overlay(case_id: str, image_data: np.ndarray, index_data: np.ndarray | None, physical_data: np.ndarray | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = (index_data, physical_data)
    foreground = next((candidate > 0 for candidate in candidates if candidate is not None and (candidate > 0).any()), None)
    z = int(np.argmax(foreground.sum(axis=(0, 1)))) if foreground is not None else image_data.shape[2] // 2
    low, high = np.percentile(image_data[..., z], [1, 99])
    if high <= low:
        low, high = float(image_data[..., z].min()), float(image_data[..., z].max()) + 1.0
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = (("image", None), ("index-geometry repair", index_data), ("physical resample", physical_data))
    for axis, (title, mask) in zip(axes, panels, strict=True):
        axis.imshow(image_data[..., z].T, cmap="gray", origin="lower", vmin=low, vmax=high)
        if mask is not None:
            axis.contour(mask[..., z].T > 0, levels=[0.5], colors="lime", linewidths=0.7, origin="lower")
        axis.set_title(title)
        axis.axis("off")
    QC_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(QC_ROOT / f"{case_id}.png", dpi=150)
    plt.close(fig)


def resolve_prostate_geometry(*, render_overlays: bool = True) -> dict[str, Any]:
    """Apply the user-authorized deterministic rule to all audited mismatches."""
    rows = read_csv(MANIFEST_ROOT / "prostate_cases.csv")
    decisions: list[dict[str, Any]] = []
    for row in rows:
        if row.get("geometry_status") != "manual_review_required":
            continue
        case_id = row["case_id"]
        try:
            image = nib.load(row["image_path_raw"])
            label = nib.load(row["label_path_raw"])
            image_data = np.asanyarray(image.dataobj)
            label_data = np.asanyarray(label.dataobj)
            if image_data.ndim != 3 or label_data.ndim != 3:
                raise ValueError(f"expected 3-D pair, got {image_data.shape}/{label_data.shape}")
            index_data = label_data if tuple(image.shape) == tuple(label.shape) else None
            physical_data = np.asanyarray(resample_from_to(label, image, order=0).dataobj)
            original_volume = float(abs(np.linalg.det(label.affine[:3, :3])))
            candidate_volume = float(abs(np.linalg.det(image.affine[:3, :3])))
            index = candidate_metrics(
                image_data, index_data, original_label=label_data, original_voxel_volume=original_volume, candidate_voxel_volume=candidate_volume
            )
            physical = candidate_metrics(
                image_data, physical_data, original_label=label_data, original_voxel_volume=original_volume, candidate_voxel_volume=candidate_volume
            )
            preferred = "index_geometry_repair" if _index_geometry_compatible(image, label) else "physical_resample"
            alternate = "physical_resample" if preferred == "index_geometry_repair" else "index_geometry_repair"
            metrics = {"index_geometry_repair": index, "physical_resample": physical}
            if metrics[preferred]["qc_pass"]:
                selected, trigger, qc_status = preferred, f"preferred_{preferred}_passed", "pass"
            elif metrics[alternate]["qc_pass"]:
                selected, trigger, qc_status = alternate, f"fallback_{alternate}_passed", "pass"
            else:
                selected, trigger, qc_status = "manual_review_required", "both_candidates_failed_qc", "manual_review_required"
            if render_overlays:
                _save_overlay(case_id, image_data, index_data, physical_data)
            decisions.append(
                {
                    "case_id": case_id,
                    "selected_decision": selected,
                    "rule_triggered": trigger,
                    "preferred_candidate": preferred,
                    "index_score": index["score"],
                    "physical_score": physical["score"],
                    "index_qc_pass": index["qc_pass"],
                    "physical_qc_pass": physical["qc_pass"],
                    "volume_retention": metrics.get(selected, {}).get("volume_retention", ""),
                    "body_overlap": metrics.get(selected, {}).get("body_support_overlap", ""),
                    "boundary_contact": metrics.get(selected, {}).get("boundary_contact", ""),
                    "axial_continuity": metrics.get(selected, {}).get("axial_continuity", ""),
                    "component_count": metrics.get(selected, {}).get("component_count", ""),
                    "normalized_centroid_distance": metrics.get(selected, {}).get("normalized_centroid_distance", ""),
                    "qc_status": qc_status,
                    "reviewer": "auto_rule_v1",
                    "index_metrics": index,
                    "physical_metrics": physical,
                    "error": "",
                }
            )
        except Exception as exc:
            decisions.append(
                {
                    "case_id": case_id,
                    "selected_decision": "manual_review_required",
                    "rule_triggered": "geometry_resolution_exception",
                    "preferred_candidate": "",
                    "index_score": "",
                    "physical_score": "",
                    "index_qc_pass": False,
                    "physical_qc_pass": False,
                    "volume_retention": "",
                    "body_overlap": "",
                    "boundary_contact": "",
                    "axial_continuity": "",
                    "component_count": "",
                    "normalized_centroid_distance": "",
                    "qc_status": "manual_review_required",
                    "reviewer": "auto_rule_v1",
                    "index_metrics": {},
                    "physical_metrics": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    fields = sorted({key for row in decisions for key in row})
    write_csv(AUDIT_ROOT / "geometry_decisions.csv", decisions, fieldnames=fields)
    distribution: dict[str, int] = {}
    for row in decisions:
        decision = str(row["selected_decision"])
        distribution[decision] = distribution.get(decision, 0) + 1
    summary = {
        "generated_at": utc_now(),
        "reviewer": "auto_rule_v1",
        "mismatch_cases": len(decisions),
        "decision_distribution": distribution,
        "manual_review_required": [row["case_id"] for row in decisions if row["selected_decision"] == "manual_review_required"],
    }
    write_json(AUDIT_ROOT / "geometry_decisions_summary.json", summary)
    write_text(
        AUDIT_ROOT / "GEOMETRY_DECISIONS.md",
        "# Prostate automatic geometry decisions\n\n"
        f"- Rule: `auto_rule_v1`\n- Mismatch cases: `{len(decisions)}`\n- Distribution: `{distribution}`\n\n"
        "The historical manual template is preserved; this file is the executable decision source for v1 preprocessing.\n",
    )
    return summary
