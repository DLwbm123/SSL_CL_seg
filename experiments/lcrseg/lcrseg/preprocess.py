from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from PIL import Image
from scipy import ndimage

from .common import (
    DATA_ROOT,
    H5_SCHEMA_VERSION,
    PROJECT_ROOT,
    config_sha256,
    ensure_no_overwrite,
    read_csv,
    require_confirmed_mapping,
    sha256_bytes,
    sha256_path,
    utc_now,
    write_csv,
    write_h5_atomically,
    write_json,
    write_text,
)


CONFIG_ROOT = PROJECT_ROOT / "configs" / "data"
MANIFEST_ROOT = PROJECT_ROOT / "manifests"
REPORT_ROOT = PROJECT_ROOT / "reports" / "preprocessing"


@dataclass(frozen=True)
class PreprocessOptions:
    output_root: Path = DATA_ROOT
    preprocess_version: str = "v1"
    compression: str = "gzip"
    compression_level: int = 4
    h5_schema_version: int = H5_SCHEMA_VERSION
    execute: bool = False


def _map_labels(data: np.ndarray, mapping: dict[Any, Any]) -> np.ndarray:
    normalized = {int(key): int(value) for key, value in mapping.items()}
    integer_data = np.rint(data).astype(int, copy=False)
    values = set(np.unique(integer_data).tolist())
    unknown = values - set(normalized)
    if unknown:
        raise ValueError(f"raw labels {sorted(unknown)} absent from confirmed mapping")
    result = np.zeros(data.shape, dtype=np.uint8)
    for source, target in normalized.items():
        result[integer_data == source] = target
    return result


def _normalize_mri(data: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    value = np.asarray(data, dtype=np.float32)
    support = value[np.isfinite(value) & (value != 0)]
    if support.size == 0:
        return np.zeros_like(value, dtype=np.float32), {"p0_5": 0.0, "p99_5": 0.0}
    low, high = np.percentile(support, [0.5, 99.5]).astype(float)
    if high <= low:
        return np.zeros_like(value, dtype=np.float32), {"p0_5": low, "p99_5": high}
    clipped = np.clip(value, low, high)
    normalized = 2.0 * (clipped - low) / (high - low) - 1.0
    return normalized.astype(np.float32), {"p0_5": low, "p99_5": high}


def _resize_zyx(array: np.ndarray, target_hw: int, *, order: int) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"expected [Z,Y,X], got {array.shape}")
    factors = (1.0, target_hw / array.shape[1], target_hw / array.shape[2])
    return ndimage.zoom(array, factors, order=order, mode="nearest", prefilter=order > 1)


def _crop_or_pad_at(array: np.ndarray, side: int, center_y: float, center_x: float) -> np.ndarray:
    """Crop a fixed in-plane field of view around an image-derived center."""
    if array.ndim != 3:
        raise ValueError(f"expected [Z,Y,X], got {array.shape}")
    output = np.zeros((array.shape[0], side, side), dtype=array.dtype)
    start_y = int(round(center_y - (side - 1) / 2.0))
    start_x = int(round(center_x - (side - 1) / 2.0))
    end_y, end_x = start_y + side, start_x + side
    source_y0, source_y1 = max(start_y, 0), min(end_y, array.shape[1])
    source_x0, source_x1 = max(start_x, 0), min(end_x, array.shape[2])
    if source_y1 <= source_y0 or source_x1 <= source_x0:
        return output
    target_y0, target_x0 = source_y0 - start_y, source_x0 - start_x
    target_y1, target_x1 = target_y0 + (source_y1 - source_y0), target_x0 + (source_x1 - source_x0)
    output[:, target_y0:target_y1, target_x0:target_x1] = array[:, source_y0:source_y1, source_x0:source_x1]
    return output


def _h5_root(options: PreprocessOptions) -> Path:
    return options.output_root / "h5" / options.preprocess_version


def _h5_paths(options: PreprocessOptions, dataset: str, group: str, stem: str) -> tuple[Path, Path]:
    root = _h5_root(options)
    return root / "images" / dataset / group / f"{stem}.h5", root / "labels" / dataset / group / f"{stem}.h5"


def _h5_relative(options: PreprocessOptions, target: Path) -> str:
    return target.relative_to(_h5_root(options)).as_posix()


def _source_digest(paths: Iterable[Path]) -> str:
    return sha256_bytes("".join(sha256_path(path) for path in paths).encode("utf-8"))


def _write_pair(
    image_target: Path,
    label_target: Path,
    image: np.ndarray,
    label: np.ndarray,
    *,
    image_attrs: dict[str, Any],
    label_attrs: dict[str, Any],
    image_chunks: tuple[int, ...],
    label_chunks: tuple[int, ...],
    options: PreprocessOptions,
) -> tuple[str, str, str]:
    config_hash = str(image_attrs["preprocess_config_sha256"])
    source_hash = str(image_attrs["source_sha256"])
    image_exists = ensure_no_overwrite(image_target, expected_config_hash=config_hash, expected_source_hash=source_hash)
    label_exists = ensure_no_overwrite(label_target, expected_config_hash=config_hash, expected_source_hash=source_hash)
    if image_exists != label_exists:
        raise FileExistsError(
            "Refusing to create an incomplete image/label HDF5 pair; repair or remove "
            f"the orphan after review: image={image_target}, label={label_target}"
        )
    if image_exists and label_exists:
        return "skipped", sha256_path(image_target), sha256_path(label_target)
    if not options.execute:
        return "planned", "", ""
    image_sha = write_h5_atomically(
        image_target,
        dataset_name="image",
        array=image,
        attrs=image_attrs,
        chunks=image_chunks,
        compression=options.compression,
        compression_level=options.compression_level,
    )
    label_sha = write_h5_atomically(
        label_target,
        dataset_name="label",
        array=label,
        attrs=label_attrs,
        chunks=label_chunks,
        compression=options.compression,
        compression_level=options.compression_level,
    )
    return "written", image_sha, label_sha


def _base_attrs(
    *,
    case_id: str,
    patient_id: str,
    dataset: str,
    site: str,
    vendor: str,
    phase: str,
    source_partition: str,
    cohort: str,
    training_role: str,
    evaluation_eligible: bool,
    original_shape: list[int],
    processed_shape: list[int],
    original_spacing: list[float] | None,
    processed_spacing: list[float] | None,
    original_affine: list[list[float]] | None,
    processed_affine: list[list[float]] | None,
    orientation: str,
    normalization: Any,
    interpolation: str,
    label_mapping: Any,
    source_sha: str,
    preprocess_hash: str,
    options: PreprocessOptions,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "patient_id": patient_id,
        "dataset": dataset,
        "site": site,
        "vendor": vendor,
        "phase": phase,
        "source_partition": source_partition,
        "cohort": cohort,
        "training_role": training_role,
        "evaluation_eligible": evaluation_eligible,
        "original_shape": original_shape,
        "processed_shape": processed_shape,
        "original_spacing": original_spacing or [],
        "processed_spacing": processed_spacing or [],
        "original_affine": original_affine or [],
        "processed_affine": processed_affine or [],
        "orientation": orientation,
        "normalization": normalization,
        "interpolation": interpolation,
        "label_mapping": label_mapping,
        "preprocess_version": options.preprocess_version,
        "h5_schema_version": options.h5_schema_version,
        "source_sha256": source_sha,
        "preprocess_config_sha256": preprocess_hash,
    }


def _failure(dataset: str, case_id: str, error: Exception | str, *, details: dict[str, Any] | None = None) -> None:
    write_json(
        REPORT_ROOT / "failures" / dataset / f"{case_id}.json",
        {"generated_at": utc_now(), "dataset": dataset, "case_id": case_id, "error": str(error), "details": details or {}},
    )


def _geometry_decisions() -> dict[str, str]:
    path = PROJECT_ROOT / "reports" / "data_audit" / "geometry_decisions.csv"
    if not path.is_file():
        raise RuntimeError("automatic geometry decisions have not been generated")
    return {row["case_id"]: row.get("selected_decision", "").strip() for row in read_csv(path)}


def _canonical_pair(image_path: Path, label_path: Path, decision: str) -> tuple[nib.spatialimages.SpatialImage, nib.spatialimages.SpatialImage]:
    image = nib.load(str(image_path))
    label = nib.load(str(label_path))
    if decision == "physical_resample":
        label = resample_from_to(label, image, order=0)
    elif decision == "index_geometry_repair":
        if image.shape != label.shape:
            raise ValueError("index geometry repair requires equal image/label shapes")
        label = nib.Nifti1Image(np.asanyarray(label.dataobj), image.affine, image.header)
    return nib.as_closest_canonical(image), nib.as_closest_canonical(label)


def preprocess_prostate(options: PreprocessOptions) -> dict[str, Any]:
    config = require_confirmed_mapping(CONFIG_ROOT / "prostate_label_map.yaml")
    rows = read_csv(MANIFEST_ROOT / "prostate_cases.csv")
    decisions = _geometry_decisions()
    pre_config = {
        "dataset": "prostate", "version": options.preprocess_version, "target_hw": 256,
        "normalization": "nonzero_p0.5_p99.5_to_minus1_1", "mapping": config["mapping"],
        "geometry_decisions_sha256": sha256_path(PROJECT_ROOT / "reports" / "data_audit" / "geometry_decisions.csv"),
    }
    preprocess_hash = config_sha256(pre_config)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        case_id = row["case_id"]
        decision = "exact_match" if row.get("geometry_status") == "exact_match" else decisions.get(case_id, "manual_review_required")
        if decision == "manual_review_required":
            _failure("prostate", case_id, "automatic geometry candidates failed QC", details={"geometry_decision": decision})
            output_rows.append({"case_id": case_id, "patient_id": row["patient_id"], "dataset": "prostate", "site": row["site"], "status": "manual_review_required", "geometry_decision": decision})
            continue
        try:
            image_path, label_path = Path(row["image_path_raw"]), Path(row["label_path_raw"])
            image, label = _canonical_pair(image_path, label_path, decision)
            image_arr = np.transpose(np.asanyarray(image.dataobj), (2, 1, 0))
            label_arr = np.transpose(np.asanyarray(label.dataobj), (2, 1, 0))
            if image_arr.shape != label_arr.shape:
                raise ValueError(f"post-alignment shape mismatch: {case_id}")
            normalized, normalization = _normalize_mri(image_arr)
            output_image = _resize_zyx(normalized, 256, order=1).astype(np.float16)
            output_label = _resize_zyx(_map_labels(label_arr, config["mapping"]), 256, order=0).astype(np.uint8)
            if not (output_label > 0).any():
                raise ValueError("empty prostate foreground after mapping")
            source_sha = _source_digest((image_path, label_path))
            zooms = image.header.get_zooms()[:3]
            attrs = _base_attrs(
                case_id=case_id, patient_id=row["patient_id"], dataset="prostate", site=row["site"], vendor="", phase="", source_partition="",
                cohort="canonical", training_role="standard", evaluation_eligible=True,
                original_shape=list(image.shape), processed_shape=list(output_image.shape), original_spacing=[float(zooms[2]), float(zooms[1]), float(zooms[0])],
                processed_spacing=[float(zooms[2]), float(zooms[1] * image_arr.shape[1] / 256), float(zooms[0] * image_arr.shape[2] / 256)],
                original_affine=image.affine.tolist(), processed_affine=[], orientation="RAS+", normalization=normalization,
                interpolation="image=linear,label=nearest", label_mapping=config["mapping"], source_sha=source_sha, preprocess_hash=preprocess_hash, options=options,
            )
            image_target, label_target = _h5_paths(options, "prostate", row["site"], case_id)
            label_attrs = {**attrs, "allowed_labels": sorted({int(value) for value in config["mapping"].values()})}
            status, image_sha, label_sha = _write_pair(image_target, label_target, output_image, output_label, image_attrs=attrs, label_attrs=label_attrs, image_chunks=(1, 256, 256), label_chunks=(1, 256, 256), options=options)
            output_rows.append({
                "case_id": case_id, "patient_id": row["patient_id"], "dataset": "prostate", "site": row["site"], "source_partition": "", "cohort": "canonical", "training_role": "standard", "evaluation_eligible": True,
                "status": status, "image_h5_relpath": _h5_relative(options, image_target), "label_h5_relpath": _h5_relative(options, label_target), "image_sha256": image_sha, "label_sha256": label_sha,
                "geometry_decision": decision, "preprocess_config_sha256": preprocess_hash,
            })
        except Exception as exc:
            _failure("prostate", case_id, exc, details={"geometry_decision": decision})
            output_rows.append({"case_id": case_id, "patient_id": row["patient_id"], "dataset": "prostate", "site": row["site"], "status": "failed", "geometry_decision": decision, "error": f"{type(exc).__name__}: {exc}"})
    write_csv(options.output_root / "manifests" / "diagnostics" / "prostate_h5.csv", output_rows, fieldnames=sorted({key for record in output_rows for key in record}))
    result = {
        "dataset": "prostate", "execute": options.execute, "rows": len(output_rows),
        "written": sum(record.get("status") == "written" for record in output_rows), "skipped": sum(record.get("status") == "skipped" for record in output_rows),
        "failed": sum(record.get("status") == "failed" for record in output_rows), "manual_review_required": sum(record.get("status") == "manual_review_required" for record in output_rows),
        "preprocess_config_sha256": preprocess_hash,
    }
    write_json(options.output_root / "reports" / "preprocessing" / "prostate_preprocess_summary.json", result)
    return result


def _mnms_metadata() -> dict[str, dict[str, str]]:
    path = Path("/Volumes/DataP/Mega/OpenDataset/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["External code"]: dict(row) for row in csv.DictReader(handle)}


def _mnms_cohort(row: dict[str, str]) -> tuple[str, str, bool]:
    if row.get("source_partition") == "Training/Unlabeled":
        return "auxiliary25", "train_unlabeled_only", False
    return "canonical320", "standard", True


def _mnms_axis_order(zooms: Sequence[float]) -> tuple[int, int, int]:
    """Return a [Z,Y,X] order using the thickest physical axis as Z."""
    slice_axis = int(np.argmax(np.asarray(zooms[:3], dtype=float)))
    inplane_axes = [axis for axis in range(3) if axis != slice_axis]
    return slice_axis, inplane_axes[0], inplane_axes[1]


def _mnms_frame_pair(
    image: nib.spatialimages.SpatialImage,
    label: nib.spatialimages.SpatialImage,
    frame: int,
    *,
    image_volume: np.ndarray | None = None,
    label_volume: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], np.ndarray]:
    if image.ndim != 4 or label.ndim != 4:
        raise ValueError(f"expected 4-D M&Ms image/label, got {image.shape}/{label.shape}")
    aligned = image.shape[:3] == label.shape[:3] and np.allclose(image.affine, label.affine, rtol=0, atol=1e-4)
    if aligned:
        zooms = image.header.get_zooms()[:3]
        raw_image = image_volume[..., frame] if image_volume is not None else np.asanyarray(image.dataobj[..., frame])
        raw_label = label_volume[..., frame] if label_volume is not None else np.asanyarray(label.dataobj[..., frame])
        affine = image.affine
    else:
        image_frame = image.slicer[:, :, :, frame]
        label_frame = label.slicer[:, :, :, frame]
        label_frame = resample_from_to(label_frame, image_frame, order=0)
        zooms = image_frame.header.get_zooms()[:3]
        raw_image = np.asanyarray(image_frame.dataobj)
        raw_label = np.asanyarray(label_frame.dataobj)
        affine = image_frame.affine
    order = _mnms_axis_order(zooms)
    image_data = np.transpose(raw_image, order)
    label_data = np.transpose(raw_label, order)
    if image_data.shape != label_data.shape:
        raise ValueError("M&Ms frame alignment did not produce equal shapes")
    return image_data, label_data, tuple(float(zooms[axis]) for axis in order), affine


def _mnms_support_center_1mm(image: nib.spatialimages.SpatialImage, *, volume: np.ndarray | None = None) -> tuple[float, float]:
    data = np.asanyarray(image.dataobj) if volume is None else volume
    support = np.isfinite(data) & (np.abs(data) > 1e-8)
    if support.ndim == 4:
        support = support.any(axis=3)
    zooms = image.header.get_zooms()[:3]
    support = np.transpose(support, _mnms_axis_order(zooms))
    points = np.argwhere(support)
    if points.size:
        center_y = float(np.mean(points[:, 1])) * float(zooms[_mnms_axis_order(zooms)[1]])
        center_x = float(np.mean(points[:, 2])) * float(zooms[_mnms_axis_order(zooms)[2]])
    else:
        order = _mnms_axis_order(zooms)
        center_y = (image.shape[order[1]] - 1.0) * float(zooms[order[1]]) / 2.0
        center_x = (image.shape[order[2]] - 1.0) * float(zooms[order[2]]) / 2.0
    return center_y, center_x


def _mnms_resample_crop(image_arr: np.ndarray, label_arr: np.ndarray, spacing_zyx: tuple[float, float, float], center_1mm_yx: tuple[float, float], fov_mm: int) -> tuple[np.ndarray, np.ndarray, float]:
    factors = (1.0, spacing_zyx[1], spacing_zyx[2])
    image_1mm = ndimage.zoom(image_arr, factors, order=1, mode="nearest")
    label_1mm = ndimage.zoom(label_arr, factors, order=0, mode="nearest")
    image_crop = _crop_or_pad_at(image_1mm, fov_mm, *center_1mm_yx)
    label_crop = _crop_or_pad_at(label_1mm, fov_mm, *center_1mm_yx)
    original_foreground = int((label_1mm > 0).sum())
    retained_foreground = int((label_crop > 0).sum())
    retention = 1.0 if original_foreground == 0 else retained_foreground / original_foreground
    return image_crop, label_crop, float(retention)


def _mnms_retention_rows(rows: Sequence[dict[str, str]], metadata: dict[str, dict[str, str]], fov_mm: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        patient_id = row["patient_id"]
        try:
            image = nib.as_closest_canonical(nib.load(row["image_path_raw"]))
            label = nib.as_closest_canonical(nib.load(row["label_path_raw"]))
            image_volume = np.asanyarray(image.dataobj)
            label_volume = np.asanyarray(label.dataobj)
            center = _mnms_support_center_1mm(image, volume=image_volume)
            meta = metadata[patient_id]
            for phase, frame in (("ED", int(meta["ED"])), ("ES", int(meta["ES"]))):
                image_arr, label_arr, spacing, _ = _mnms_frame_pair(image, label, frame, image_volume=image_volume, label_volume=label_volume)
                _, _, retention = _mnms_resample_crop(image_arr, label_arr, spacing, center, fov_mm)
                output.append({"patient_id": patient_id, "vendor": row["vendor"], "phase": phase, "fov_mm": fov_mm, "foreground_retention": retention, "status": "pass" if retention >= 0.995 else "fail", "error": ""})
        except Exception as exc:
            output.append({"patient_id": patient_id, "vendor": row.get("vendor", ""), "phase": "", "fov_mm": fov_mm, "foreground_retention": 0.0, "status": "fail", "error": f"{type(exc).__name__}: {exc}"})
    return output


def select_mnms_fov() -> dict[str, Any]:
    """Select the smallest frozen FOV that retains >=99.5% foreground globally."""
    rows = read_csv(MANIFEST_ROOT / "mnms_cases.csv")
    metadata = _mnms_metadata()
    pilot_ids: set[str] = set()
    for vendor in ("Siemens", "Philips", "GE", "Canon"):
        pilot_ids.update(row["patient_id"] for row in sorted((item for item in rows if item["vendor"] == vendor), key=lambda item: item["patient_id"])[:2])
    pilot_rows = [row for row in rows if row["patient_id"] in pilot_ids]
    all_records: list[dict[str, Any]] = []
    selected_fov: int | None = None
    selection_trace: list[dict[str, Any]] = []
    for fov in (256, 288, 320):
        pilot = _mnms_retention_rows(pilot_rows, metadata, fov)
        all_records.extend({**record, "scope": "pilot"} for record in pilot)
        pilot_min = min((float(record["foreground_retention"]) for record in pilot), default=0.0)
        trace: dict[str, Any] = {"fov_mm": fov, "pilot_min_foreground_retention": pilot_min, "pilot_pass": pilot_min >= 0.995 and all(record["status"] == "pass" for record in pilot)}
        if trace["pilot_pass"]:
            full = _mnms_retention_rows(rows, metadata, fov)
            all_records.extend({**record, "scope": "full"} for record in full)
            full_min = min((float(record["foreground_retention"]) for record in full), default=0.0)
            trace.update(full_min_foreground_retention=full_min, full_pass=full_min >= 0.995 and all(record["status"] == "pass" for record in full))
            if trace["full_pass"]:
                selected_fov = fov
                selection_trace.append(trace)
                break
        selection_trace.append(trace)
    write_csv(REPORT_ROOT / "mnms_fov_retention.csv", all_records, fieldnames=sorted({key for record in all_records for key in record}))
    payload = {
        "generated_at": utc_now(), "pilot_patient_count": len(pilot_ids), "candidate_fov_mm": [256, 288, 320],
        "selection_trace": selection_trace, "selected_fov_mm": selected_fov,
        "minimum_retention": min((float(record["foreground_retention"]) for record in all_records if record.get("scope") == "full" and record.get("fov_mm") == selected_fov), default=0.0),
    }
    write_json(REPORT_ROOT / "mnms_fov_selection.json", payload)
    if selected_fov is None:
        raise RuntimeError("No permitted M&Ms FOV retained 99.5% foreground globally")
    import yaml

    write_text(CONFIG_ROOT / "mnms_preprocess.yaml", yaml.safe_dump({"status": "FROZEN_USER_AUTHORIZED_V1", "selected_fov_mm": selected_fov, "minimum_foreground_retention": payload["minimum_retention"], "selection_trace": selection_trace}, sort_keys=False))
    return payload


def preprocess_mnms(options: PreprocessOptions, *, selected_fov_mm: int | None = None) -> dict[str, Any]:
    config = require_confirmed_mapping(CONFIG_ROOT / "mnms_label_map.yaml")
    selection = select_mnms_fov() if selected_fov_mm is None else {"selected_fov_mm": selected_fov_mm, "minimum_retention": None}
    fov_mm = int(selection["selected_fov_mm"])
    rows = read_csv(MANIFEST_ROOT / "mnms_cases.csv")
    metadata = _mnms_metadata()
    cohort_counts: dict[str, int] = {}
    for row in rows:
        cohort, _, _ = _mnms_cohort(row)
        cohort_counts[cohort] = cohort_counts.get(cohort, 0) + 1
    if cohort_counts != {"canonical320": 320, "auxiliary25": 25}:
        raise RuntimeError(f"unexpected M&Ms cohort partition: {cohort_counts}")
    pre_config = {
        "dataset": "mnms", "version": options.preprocess_version, "target_hw": 384, "target_inplane_spacing_mm": 1.0,
        "fixed_crop_fov_mm": fov_mm, "normalization": "nonzero_p0.5_p99.5_to_minus1_1", "mapping": config["mapping"],
    }
    preprocess_hash = config_sha256(pre_config)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        patient_id = row["patient_id"]
        cohort, training_role, evaluation_eligible = _mnms_cohort(row)
        try:
            image_path, label_path = Path(row["image_path_raw"]), Path(row["label_path_raw"])
            image = nib.as_closest_canonical(nib.load(str(image_path)))
            label = nib.as_closest_canonical(nib.load(str(label_path)))
            meta = metadata[patient_id]
            image_volume = np.asanyarray(image.dataobj)
            label_volume = np.asanyarray(label.dataobj)
            center = _mnms_support_center_1mm(image, volume=image_volume)
            source_sha = _source_digest((image_path, label_path))
            for phase, frame in (("ED", int(meta["ED"])), ("ES", int(meta["ES"]))):
                case_id = f"{patient_id}_{phase}"
                try:
                    image_arr, label_arr, spacing, affine = _mnms_frame_pair(image, label, frame, image_volume=image_volume, label_volume=label_volume)
                    image_crop, label_crop, retention = _mnms_resample_crop(image_arr, label_arr, spacing, center, fov_mm)
                    if retention < 0.995:
                        raise ValueError(f"foreground retention {retention:.6f} below 0.995")
                    normalized, normalization = _normalize_mri(image_crop)
                    output_image = _resize_zyx(normalized, 384, order=1).astype(np.float16)
                    output_label = _resize_zyx(_map_labels(label_crop, config["mapping"]), 384, order=0).astype(np.uint8)
                    label_nonempty = bool((output_label > 0).any())
                    if not label_nonempty and evaluation_eligible:
                        raise ValueError("empty cardiac foreground after preprocessing")
                    attrs = _base_attrs(
                        case_id=case_id, patient_id=patient_id, dataset="mnms", site=row["vendor"], vendor=row["vendor"], phase=phase, source_partition=row["source_partition"],
                        cohort=cohort, training_role=training_role, evaluation_eligible=evaluation_eligible,
                        original_shape=list(image_arr.shape), processed_shape=list(output_image.shape), original_spacing=list(spacing), processed_spacing=[spacing[0], fov_mm / 384.0, fov_mm / 384.0],
                        original_affine=affine.tolist(), processed_affine=[], orientation="RAS+", normalization=normalization,
                        interpolation="image=linear,label=nearest", label_mapping=config["mapping"], source_sha=source_sha, preprocess_hash=preprocess_hash, options=options,
                    )
                    image_target, label_target = _h5_paths(options, "mnms", row["vendor"], case_id)
                    label_attrs = {
                        **attrs,
                        "allowed_labels": sorted({int(value) for value in config["mapping"].values()}),
                        "diagnostic_label_nonempty": label_nonempty,
                        "diagnostic_label_status": "nonempty" if label_nonempty else "empty_at_official_metadata_phase",
                    }
                    status, image_sha, label_sha = _write_pair(image_target, label_target, output_image, output_label, image_attrs=attrs, label_attrs=label_attrs, image_chunks=(1, 384, 384), label_chunks=(1, 384, 384), options=options)
                    output_rows.append({
                        "case_id": case_id, "patient_id": patient_id, "dataset": "mnms", "vendor": row["vendor"], "phase": phase, "source_partition": row["source_partition"], "cohort": cohort, "training_role": training_role, "evaluation_eligible": evaluation_eligible,
                        "status": status, "foreground_retention": retention, "diagnostic_label_nonempty": label_nonempty, "image_h5_relpath": _h5_relative(options, image_target), "label_h5_relpath": _h5_relative(options, label_target), "image_sha256": image_sha, "label_sha256": label_sha, "preprocess_config_sha256": preprocess_hash,
                    })
                except Exception as exc:
                    _failure("mnms", case_id, exc, details={"patient_id": patient_id, "phase": phase})
                    output_rows.append({"case_id": case_id, "patient_id": patient_id, "dataset": "mnms", "vendor": row["vendor"], "phase": phase, "source_partition": row["source_partition"], "cohort": cohort, "training_role": training_role, "evaluation_eligible": evaluation_eligible, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:
            _failure("mnms", patient_id, exc, details={"patient_id": patient_id})
            for phase in ("ED", "ES"):
                output_rows.append({"case_id": f"{patient_id}_{phase}", "patient_id": patient_id, "dataset": "mnms", "vendor": row["vendor"], "phase": phase, "source_partition": row["source_partition"], "cohort": cohort, "training_role": training_role, "evaluation_eligible": evaluation_eligible, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    write_csv(options.output_root / "manifests" / "diagnostics" / "mnms_h5.csv", output_rows, fieldnames=sorted({key for record in output_rows for key in record}))
    result = {
        "dataset": "mnms", "execute": options.execute, "patients": len(rows), "patient_phases": len(output_rows), "fixed_crop_fov_mm": fov_mm,
        "minimum_foreground_retention": min((float(record["foreground_retention"]) for record in output_rows if record.get("status") in {"written", "skipped"}), default=0.0),
        "written": sum(record.get("status") == "written" for record in output_rows), "skipped": sum(record.get("status") == "skipped" for record in output_rows),
        "failed": sum(record.get("status") == "failed" for record in output_rows), "cohort_counts": cohort_counts, "preprocess_config_sha256": preprocess_hash,
    }
    write_json(options.output_root / "reports" / "preprocessing" / "mnms_preprocess_summary.json", result)
    return result


def _fundus_center_crop(image: np.ndarray, side: int = 800) -> np.ndarray:
    height, width = image.shape[:2]
    pad_top = max((side - height) // 2, 0)
    pad_bottom = max(side - height - pad_top, 0)
    pad_left = max((side - width) // 2, 0)
    pad_right = max(side - width - pad_left, 0)
    if pad_top or pad_bottom or pad_left or pad_right:
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)) + ((0, 0),) * (image.ndim - 2), mode="constant")
    start_y = (image.shape[0] - side) // 2
    start_x = (image.shape[1] - side) // 2
    return image[start_y : start_y + side, start_x : start_x + side]


def preprocess_fundus(options: PreprocessOptions) -> dict[str, Any]:
    config = require_confirmed_mapping(CONFIG_ROOT / "fundus_label_map.yaml")
    rows = read_csv(MANIFEST_ROOT / "fundus_cases.csv")
    pre_config = {
        "dataset": "fundus", "version": options.preprocess_version, "crop": 800, "target_hw": 384,
        "mapping": config["mapping"], "interpolation": "image=bilinear,label=nearest",
    }
    preprocess_hash = config_sha256(pre_config)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        case_id = row["case_id"]
        try:
            image_path, label_path = Path(row["image_path_raw"]), Path(row["label_path_raw"])
            with Image.open(image_path) as image_file, Image.open(label_path) as label_file:
                image = np.asarray(image_file.convert("RGB"))
                raw_mask = np.asarray(label_file.convert("RGB"))
            if not np.all(raw_mask[..., :1] == raw_mask[..., 1:]):
                raise ValueError("non-grayscale fundus encoding")
            raw_mask = raw_mask[..., 0]
            foreground_before = int((raw_mask != 255).sum())
            image_crop = _fundus_center_crop(image)
            mask_crop = _fundus_center_crop(raw_mask)
            foreground_after = int((mask_crop != 255).sum())
            retention = 1.0 if foreground_before == 0 else foreground_after / foreground_before
            image_out = np.transpose(np.asarray(Image.fromarray(image_crop).resize((384, 384), Image.Resampling.BILINEAR)), (2, 0, 1)).astype(np.uint8)
            mask_out = np.asarray(Image.fromarray(mask_crop).resize((384, 384), Image.Resampling.NEAREST))
            label_out = _map_labels(mask_out, config["mapping"])
            if set(np.unique(label_out).tolist()) - {0, 1, 2}:
                raise ValueError("unexpected fundus label after mapping")
            source_sha = _source_digest((image_path, label_path))
            attrs = _base_attrs(
                case_id=case_id, patient_id=row["patient_id"], dataset="fundus", site=row["site"], vendor="", phase="", source_partition=row["source_partition"],
                cohort="canonical", training_role="standard", evaluation_eligible=True,
                original_shape=list(image.shape), processed_shape=list(image_out.shape), original_spacing=None, processed_spacing=None,
                original_affine=None, processed_affine=None, orientation="pixel_yx", normalization="uint8_rgb_0_255; scale_to_0_1_in_loader",
                interpolation="image=bilinear,label=nearest", label_mapping=config["mapping"], source_sha=source_sha, preprocess_hash=preprocess_hash, options=options,
            )
            image_target, label_target = _h5_paths(options, "fundus", row["site"], case_id)
            label_attrs = {**attrs, "allowed_labels": [0, 1, 2], "crop_foreground_retention": retention}
            status, image_sha, label_sha = _write_pair(image_target, label_target, image_out, label_out, image_attrs=attrs, label_attrs=label_attrs, image_chunks=(3, 384, 384), label_chunks=(384, 384), options=options)
            output_rows.append({
                "case_id": case_id, "patient_id": row["patient_id"], "dataset": "fundus", "site": row["site"], "source_partition": row["source_partition"], "cohort": "canonical", "training_role": "standard", "evaluation_eligible": True,
                "status": status, "crop_foreground_retention": retention, "image_h5_relpath": _h5_relative(options, image_target), "label_h5_relpath": _h5_relative(options, label_target), "image_sha256": image_sha, "label_sha256": label_sha, "preprocess_config_sha256": preprocess_hash,
            })
        except Exception as exc:
            _failure("fundus", case_id, exc)
            output_rows.append({"case_id": case_id, "patient_id": row["patient_id"], "dataset": "fundus", "site": row["site"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    write_csv(options.output_root / "manifests" / "diagnostics" / "fundus_h5.csv", output_rows, fieldnames=sorted({key for record in output_rows for key in record}))
    result = {
        "dataset": "fundus", "execute": options.execute, "rows": len(output_rows),
        "written": sum(record.get("status") == "written" for record in output_rows), "skipped": sum(record.get("status") == "skipped" for record in output_rows),
        "failed": sum(record.get("status") == "failed" for record in output_rows),
        "minimum_crop_foreground_retention": min((float(record["crop_foreground_retention"]) for record in output_rows if record.get("status") in {"written", "skipped"}), default=0.0),
        "preprocess_config_sha256": preprocess_hash,
    }
    write_json(options.output_root / "reports" / "preprocessing" / "fundus_preprocess_summary.json", result)
    return result
