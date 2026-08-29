from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from PIL import Image
from scipy import ndimage

from .common import (
    FUNDUS_ROOT,
    MNMS_ROOT,
    PROJECT_ROOT,
    PROSTATE_ROOT,
    config_sha256,
    utc_now,
    visible_paths,
    write_csv,
    write_json,
    write_text,
)


AUDIT_ROOT = PROJECT_ROOT / "reports" / "data_audit"
MANIFEST_ROOT = PROJECT_ROOT / "manifests"
QC_ROOT = PROJECT_ROOT / "reports" / "qc_overlays"


def _array_hash(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(array, dtype=np.float64).tobytes()).hexdigest()


def _number_list(value: Any) -> list[float]:
    return [round(float(x), 7) for x in np.ravel(value).tolist()]


def _orientation(image: nib.spatialimages.SpatialImage) -> str:
    return ",".join(nib.aff2axcodes(image.affine))


def _case_stem(path: Path) -> str:
    stem = path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    return stem


def _prostate_pairs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for site_dir in visible_paths(PROSTATE_ROOT.iterdir()):
        if not site_dir.is_dir():
            continue
        files = visible_paths(site_dir.glob("*.nii.gz"))
        images = [path for path in files if "seg" not in path.name.lower()]
        labels = [path for path in files if "seg" in path.name.lower()]
        label_by_base: dict[str, Path] = {}
        for label in labels:
            base = _case_stem(label).lower().replace("_segmentation", "")
            label_by_base[base] = label
        for image_path in images:
            name = _case_stem(image_path)
            label_path = label_by_base.get(name.lower())
            rows.append(
                {
                    "case_id": f"{site_dir.name}_{name}",
                    "patient_id": f"{site_dir.name}_{name}",
                    "dataset": "prostate",
                    "site": site_dir.name,
                    "image_path_raw": str(image_path),
                    "label_path_raw": str(label_path) if label_path else "",
                }
            )
    return sorted(rows, key=lambda row: (row["site"], row["case_id"]))


def _prostate_overlay(row: dict[str, Any], image: nib.spatialimages.SpatialImage, label: nib.spatialimages.SpatialImage) -> None:
    """Render original/physical/index candidates without deciding between them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_data = np.asanyarray(image.dataobj)
    label_data = np.asanyarray(label.dataobj)
    if image_data.ndim != 3 or label_data.ndim != 3:
        return
    foreground = label_data > 0
    z = int(np.argmax(foreground.sum(axis=(0, 1)))) if foreground.any() else image_data.shape[2] // 2
    physical = resample_from_to(label, image, order=0)
    physical_data = np.asanyarray(physical.dataobj)
    index_data = label_data if label_data.shape == image_data.shape else np.zeros_like(image_data)
    low, high = np.percentile(image_data[..., z], [1, 99])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    candidates = [
        ("original mask", label_data[..., z] if label_data.shape == image_data.shape else None),
        ("physical-space candidate", physical_data[..., z]),
        ("index-preserving candidate", index_data[..., z] if label_data.shape == image_data.shape else None),
    ]
    for axis, (title, mask) in zip(axes, candidates, strict=True):
        axis.imshow(image_data[..., z].T, cmap="gray", origin="lower", vmin=low, vmax=high)
        if mask is not None:
            axis.contour(mask.T > 0, levels=[0.5], colors="lime", linewidths=0.7, origin="lower")
        axis.set_title(title)
        axis.axis("off")
    out = QC_ROOT / "prostate_geometry" / f"{row['case_id']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def audit_prostate(*, render_overlays: bool = True) -> dict[str, Any]:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    geometry_counts: Counter[str] = Counter()
    for base in _prostate_pairs():
        image_path = Path(base["image_path_raw"])
        label_path = Path(base["label_path_raw"]) if base["label_path_raw"] else None
        if label_path is None or not label_path.is_file():
            row = {**base, "qc_status": "failed", "geometry_status": "failed", "audit_error": "missing label"}
            rows.append(row)
            geometry_counts[row["geometry_status"]] += 1
            continue
        try:
            image = nib.load(str(image_path))
            label = nib.load(str(label_path))
            image_shape, label_shape = tuple(image.shape), tuple(label.shape)
            affine_equal = bool(np.allclose(image.affine, label.affine, rtol=0, atol=1e-4))
            orientation_equal = _orientation(image) == _orientation(label)
            geometry_status = "exact_match" if image_shape == label_shape and affine_equal and orientation_equal else "manual_review_required"
            label_data = np.asanyarray(label.dataobj)
            values, counts = np.unique(label_data, return_counts=True)
            label_one = label_data == 1
            label_two = label_data == 2
            one_components = int(ndimage.label(label_one)[1])
            two_components = int(ndimage.label(label_two)[1])
            if label_one.any() and label_two.any():
                c1 = np.asarray(ndimage.center_of_mass(label_one), dtype=float)
                c2 = np.asarray(ndimage.center_of_mass(label_two), dtype=float)
                centroid_distance = round(float(np.linalg.norm(c1 - c2)), 5)
                adjacency = bool(np.logical_and(ndimage.binary_dilation(label_one), label_two).any())
            else:
                centroid_distance, adjacency = "", False
            row = {
                **base,
                "shape_image": json_value(image_shape),
                "shape_label": json_value(label_shape),
                "spacing_image": json_value(image.header.get_zooms()[:3]),
                "spacing_label": json_value(label.header.get_zooms()[:3]),
                "orientation_image": _orientation(image),
                "orientation_label": _orientation(label),
                "affine_hash_image": _array_hash(image.affine),
                "affine_hash_label": _array_hash(label.affine),
                "affine_equal": affine_equal,
                "geometry_status": geometry_status,
                "label_values_raw": json_value(values.astype(float).tolist()),
                "label_voxel_counts": json_value(dict(zip(values.astype(int).tolist(), counts.astype(int).tolist(), strict=True))),
                "qc_status": "pass" if geometry_status == "exact_match" else "manual_review_required",
                "audit_error": "",
            }
            label_row = {
                "case_id": base["case_id"],
                "site": base["site"],
                "unique_labels": json_value(values.astype(float).tolist()),
                "voxels_label_1": int(label_one.sum()),
                "voxels_label_2": int(label_two.sum()),
                "components_label_1": one_components,
                "components_label_2": two_components,
                "centroid_distance_voxels": centroid_distance,
                "overlap_or_adjacency": adjacency,
            }
            if geometry_status != "exact_match":
                decision_rows.append(
                    {
                        "case_id": base["case_id"],
                        "decision": "",
                        "reviewer": "",
                        "reason": "",
                        "timestamp": "",
                        "allowed_decisions": "physical_resample|index_geometry_repair|reject",
                    }
                )
                if render_overlays:
                    _prostate_overlay(base, image, label)
            rows.append(row)
            label_rows.append(label_row)
            geometry_counts[geometry_status] += 1
        except Exception as exc:  # audit must retain a failing case as evidence
            row = {**base, "qc_status": "failed", "geometry_status": "failed", "audit_error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            geometry_counts[row["geometry_status"]] += 1
    fields = sorted({key for row in rows for key in row})
    write_csv(MANIFEST_ROOT / "prostate_cases.csv", rows, fieldnames=fields)
    write_csv(AUDIT_ROOT / "prostate_label_audit.csv", label_rows)
    write_csv(
        AUDIT_ROOT / "geometry_decisions_template.csv",
        decision_rows,
        fieldnames=["case_id", "decision", "reviewer", "reason", "timestamp", "allowed_decisions"],
    )
    summary = {
        "dataset": "prostate_six_site",
        "generated_at": utc_now(),
        "total_cases": len(rows),
        "by_site": dict(Counter(row["site"] for row in rows)),
        "geometry_status": dict(geometry_counts),
        "decision_template_count": len(decision_rows),
        "label_2_cases": sum(int(row["voxels_label_2"]) > 0 for row in label_rows),
        "blocker": "All manual_review_required cases require an explicit geometry decision; label 2 semantics must be confirmed before a binary map is enabled.",
    }
    write_json(AUDIT_ROOT / "prostate_audit_summary.json", summary)
    write_text(
        AUDIT_ROOT / "PROSTATE_AUDIT.md",
        "# Prostate audit\n\n"
        f"- Cases: `{summary['total_cases']}`\n"
        f"- Geometry status: `{summary['geometry_status']}`\n"
        f"- Pending explicit geometry decisions: `{len(decision_rows)}`\n"
        f"- Cases with raw label 2: `{summary['label_2_cases']}`\n\n"
        "No geometry repair or 0/1/2 label merge has been applied. Review the candidate overlays and fill `geometry_decisions_template.csv` before preprocessing.\n",
    )
    return summary


def _mnms_patient_dirs() -> list[tuple[str, Path]]:
    partitions = ("Training/Labeled", "Training/Unlabeled", "Validation", "Testing")
    rows: list[tuple[str, Path]] = []
    for partition in partitions:
        root = MNMS_ROOT / partition
        if root.is_dir():
            rows.extend((partition, path) for path in visible_paths(root.iterdir()) if path.is_dir())
    return rows


def _mnms_metadata() -> dict[str, dict[str, str]]:
    path = MNMS_ROOT / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {str(row.get("External code", "")).strip(): dict(row) for row in rows}


def _first_visible(paths: list[Path]) -> Path | None:
    return next((path for path in paths if not path.name.startswith((".", "._"))), None)


def audit_mnms() -> dict[str, Any]:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    metadata = _mnms_metadata()
    rows: list[dict[str, Any]] = []
    label_values: set[int] = set()
    errors: list[dict[str, str]] = []
    for partition, patient_dir in _mnms_patient_dirs():
        patient_id = patient_dir.name
        image_path = _first_visible(sorted(patient_dir.glob("*_sa.nii.gz")))
        label_path = _first_visible(sorted(patient_dir.glob("*_sa_gt.nii.gz")))
        meta = metadata.get(patient_id, {})
        vendor = str(meta.get("VendorName", "")).strip() or "UNKNOWN"
        row: dict[str, Any] = {
            "case_id": patient_id,
            "patient_id": patient_id,
            "dataset": "mnms",
            "source_partition": partition,
            "image_path_raw": str(image_path) if image_path else "",
            "label_path_raw": str(label_path) if label_path else "",
            "vendor": vendor,
            "centre": str(meta.get("Centre", "")).strip(),
            "pathology": str(meta.get("Pathology", "")).strip(),
            "sex": str(meta.get("Sex", "")).strip(),
            "ed_frame": str(meta.get("ED", "")).strip(),
            "es_frame": str(meta.get("ES", "")).strip(),
            "mapping_source": "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv" if meta else "",
            "mapping_confidence": "official_local_metadata" if meta else "missing",
            "metadata_found": bool(meta),
            "gt_provenance": "requires_research_use_confirmation" if partition == "Training/Unlabeled" else "present_on_disk",
        }
        try:
            if image_path is None or label_path is None:
                raise ValueError("missing image or label")
            image, label = nib.load(str(image_path)), nib.load(str(label_path))
            shape_equal = tuple(image.shape) == tuple(label.shape)
            affine_equal = bool(np.allclose(image.affine, label.affine, rtol=0, atol=1e-4))
            if len(image.shape) != 4:
                raise ValueError(f"expected 4-D CMR, got {image.shape}")
            frame_count = image.shape[3]
            ed, es = int(meta.get("ED", "-1")), int(meta.get("ES", "-1"))
            ed_es_valid = 0 <= ed < frame_count and 0 <= es < frame_count
            values: set[int] = set()
            if ed_es_valid:
                for frame in (ed, es):
                    arr = np.asarray(label.dataobj[..., frame])
                    values.update(np.rint(np.unique(arr)).astype(int).tolist())
            label_values.update(values)
            row.update(
                shape_image=json_value(image.shape),
                shape_label=json_value(label.shape),
                spacing_image=json_value(image.header.get_zooms()[:3]),
                spacing_label=json_value(label.header.get_zooms()[:3]),
                affine_equal=affine_equal,
                geometry_status="exact_match" if shape_equal and affine_equal else "manual_review_required",
                frame_count=frame_count,
                ed_es_valid=ed_es_valid,
                label_values_at_ed_es=json_value(sorted(values)),
                qc_status="pass" if shape_equal and affine_equal and ed_es_valid and vendor != "UNKNOWN" else "manual_review_required",
                audit_error="",
            )
        except Exception as exc:
            row.update(qc_status="failed", geometry_status="failed", audit_error=f"{type(exc).__name__}: {exc}")
            errors.append({"case_id": patient_id, "error": row["audit_error"]})
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    write_csv(MANIFEST_ROOT / "mnms_cases.csv", rows, fieldnames=fields)
    mapping_fields = [
        "patient_id", "source_partition", "vendor", "centre", "pathology", "sex", "ed_frame", "es_frame", "mapping_source", "mapping_confidence", "gt_provenance",
    ]
    write_csv(AUDIT_ROOT / "mnms_vendor_mapping.csv", rows, fieldnames=mapping_fields)
    counts = Counter(row["vendor"] for row in rows)
    summary = {
        "dataset": "mnms",
        "generated_at": utc_now(),
        "total_cases": len(rows),
        "source_partition_counts": dict(Counter(row["source_partition"] for row in rows)),
        "vendor_counts": dict(counts),
        "metadata_coverage": sum(bool(row["metadata_found"]) for row in rows),
        "gt_files_present": sum(bool(row["label_path_raw"]) for row in rows),
        "observed_labels_at_ed_es": sorted(label_values),
        "failed_cases": errors,
        "blockers": [
            "Confirm the official semantic mapping of raw labels 1/2/3 before HDF5 generation.",
            "Confirm whether the 25 source Training/Unlabeled GT files may be used for research evaluation; they must never become visible to a training unlabeled loader.",
        ],
    }
    write_json(AUDIT_ROOT / "mnms_audit_summary.json", summary)
    write_text(
        AUDIT_ROOT / "MNMS_AUDIT.md",
        "# M&Ms audit\n\n"
        f"- Patients: `{len(rows)}`\n"
        f"- Metadata coverage: `{summary['metadata_coverage']}/{len(rows)}`\n"
        f"- Vendor counts: `{summary['vendor_counts']}`\n"
        f"- Observed ED/ES label values: `{summary['observed_labels_at_ed_es']}`\n"
        f"- GT files present: `{summary['gt_files_present']}/{len(rows)}`\n\n"
        "Vendor IDs come from the local official diagnosis CSV. Raw label numbers are preserved, not semantically renamed.\n",
    )
    return summary


def _fundus_pairs(dataset: str) -> list[dict[str, str]]:
    root = FUNDUS_ROOT / dataset
    rows: list[dict[str, str]] = []
    for split_dir in visible_paths(root.iterdir()):
        if not split_dir.is_dir() or split_dir.name not in {"train", "test"}:
            continue
        images = {path.name: path for path in visible_paths((split_dir / "image").glob("*")) if path.is_file()}
        masks = {path.name: path for path in visible_paths((split_dir / "mask").glob("*")) if path.is_file()}
        for name in sorted(set(images) | set(masks)):
            case_id = f"{dataset}_{split_dir.name}_{Path(name).stem}"
            rows.append(
                {
                    "case_id": case_id,
                    "patient_id": case_id,
                    "dataset": "fundus",
                    "site": dataset,
                    "source_partition": split_dir.name,
                    "image_path_raw": str(images.get(name, "")),
                    "label_path_raw": str(masks.get(name, "")),
                }
            )
    return rows


def _fundus_crop(mask: np.ndarray, side: int = 800) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
    h, w = mask.shape[:2]
    pad_top = max((side - h) // 2, 0)
    pad_bottom = max(side - h - pad_top, 0)
    pad_left = max((side - w) // 2, 0)
    pad_right = max(side - w - pad_left, 0)
    if pad_top or pad_bottom or pad_left or pad_right:
        pad_shape = ((pad_top, pad_bottom), (pad_left, pad_right)) + ((0, 0),) * (mask.ndim - 2)
        padded = np.pad(mask, pad_shape, mode="constant")
    else:
        padded = mask
    start_y = (padded.shape[0] - side) // 2
    start_x = (padded.shape[1] - side) // 2
    crop = padded[start_y : start_y + side, start_x : start_x + side]
    original_fg = int(np.any(mask != 0, axis=-1).sum()) if mask.ndim == 3 else int((mask != 0).sum())
    crop_fg = int(np.any(crop != 0, axis=-1).sum()) if crop.ndim == 3 else int((crop != 0).sum())
    retention = 1.0 if original_fg == 0 else crop_fg / original_fg
    return crop, retention, (pad_top, pad_bottom, pad_left, pad_right)


def _fundus_overlay(row: dict[str, Any], image: np.ndarray, mask: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[0].axis("off")
    axes[1].imshow(image)
    gray = mask[..., 0] if mask.ndim == 3 and np.all(mask[..., :1] == mask[..., 1:]) else mask
    axes[1].imshow(gray, cmap="viridis", alpha=0.45, interpolation="nearest")
    axes[1].set_title("raw mask")
    axes[1].axis("off")
    out = QC_ROOT / "fundus" / row["site"] / f"{row['case_id']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    plt.close(fig)


def audit_fundus(*, qc_per_dataset: int = 20) -> dict[str, Any]:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    by_site: dict[str, dict[str, Any]] = {}
    for dataset in ("REFUGE", "RIM_ONE_r3", "Drishti_GS"):
        site_rows = _fundus_pairs(dataset)
        color_sets: Counter[str] = Counter()
        retentions: list[float] = []
        raw_areas: Counter[int] = Counter()
        rendered = 0
        for base in site_rows:
            row = dict(base)
            try:
                image_path, label_path = Path(row["image_path_raw"]), Path(row["label_path_raw"])
                if not image_path.is_file() or not label_path.is_file():
                    raise ValueError("missing image or mask pair")
                with Image.open(image_path) as pil_image, Image.open(label_path) as pil_mask:
                    image = np.asarray(pil_image.convert("RGB"))
                    mask = np.asarray(pil_mask.convert("RGB"))
                    image_mode, mask_mode = pil_image.mode, pil_mask.mode
                grayscale = mask[..., 0] if np.all(mask[..., :1] == mask[..., 1:]) else None
                if grayscale is None:
                    colors = np.unique(mask.reshape(-1, 3), axis=0)
                    values = ["/".join(map(str, color.tolist())) for color in colors]
                    foreground = np.any(mask != 0, axis=-1)
                else:
                    values = np.unique(grayscale).astype(int).tolist()
                    foreground = grayscale != 0
                    raw_areas.update({int(value): int((grayscale == value).sum()) for value in values if value != 0})
                crop, retention, pads = _fundus_crop(mask)
                color_sets[json_value(values)] += 1
                retentions.append(retention)
                row.update(
                    image_shape=json_value(image.shape),
                    mask_shape=json_value(mask.shape),
                    image_mode=image_mode,
                    mask_mode=mask_mode,
                    mask_values_raw=json_value(values),
                    grayscale_mask=grayscale is not None,
                    foreground_pixels=int(foreground.sum()),
                    center_crop_retention=round(retention, 8),
                    symmetric_padding=json_value(pads),
                    qc_status="pass" if retention >= 0.99 else "manual_review_required",
                    audit_error="",
                )
                if rendered < qc_per_dataset:
                    _fundus_overlay(row, image, mask)
                    rendered += 1
            except Exception as exc:
                row.update(qc_status="failed", audit_error=f"{type(exc).__name__}: {exc}")
                errors.append({"case_id": row["case_id"], "error": row["audit_error"]})
            rows.append(row)
        values_ranked = sorted(raw_areas.items(), key=lambda item: item[1])
        candidate_cup = values_ranked[0][0] if values_ranked else None
        candidate_rim = values_ranked[-1][0] if values_ranked else None
        by_site[dataset] = {
            "cases": len(site_rows),
            "mask_value_profiles": dict(color_sets),
            "minimum_center_crop_retention": min(retentions) if retentions else None,
            "mean_center_crop_retention": float(np.mean(retentions)) if retentions else None,
            "qc_overlays_written": rendered,
            "candidate_smaller_foreground_value": candidate_cup,
            "candidate_larger_foreground_value": candidate_rim,
        }
    fields = sorted({key for row in rows for key in row})
    write_csv(MANIFEST_ROOT / "fundus_cases.csv", rows, fieldnames=fields)
    summary = {
        "dataset": "fundus_three_site",
        "generated_at": utc_now(),
        "total_cases": len(rows),
        "by_site": by_site,
        "failed_cases": errors,
        "blocker": "The observed grayscale encoding is reported but not promoted to optic-disc-rim/optic-cup semantics until explicitly confirmed.",
    }
    write_json(AUDIT_ROOT / "fundus_audit_summary.json", summary)
    write_text(
        AUDIT_ROOT / "FUNDUS_AUDIT.md",
        "# Fundus audit\n\n"
        f"- Total paired records: `{len(rows)}`\n"
        f"- Per-site audit: `{json_value(by_site)}`\n\n"
        "The center crop is assessed using hidden GT only for offline QC. The grayscale code mapping remains intentionally unconfirmed.\n",
    )
    return summary


def json_value(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
