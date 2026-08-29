from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .common import PROJECT_ROOT, canonical_json, config_sha256, sha256_path, utc_now, validate_h5_file, write_csv, write_json, write_text


TRANSFER_PREFIXES = ("h5", "manifests", "splits", "reports/preprocessing")
EXCLUDED_TRANSFER_RELATIVE_PATHS = {
    "checksums/checksums.sha256",
    "manifests/transfer_manifest.json",
}

REQUIRED_H5_ATTRS = (
    "case_id",
    "patient_id",
    "dataset",
    "site",
    "vendor",
    "phase",
    "source_partition",
    "cohort",
    "training_role",
    "evaluation_eligible",
    "original_shape",
    "processed_shape",
    "original_spacing",
    "processed_spacing",
    "original_affine",
    "processed_affine",
    "orientation",
    "normalization",
    "interpolation",
    "label_mapping",
    "preprocess_version",
    "h5_schema_version",
    "source_sha256",
    "preprocess_config_sha256",
)
EXPECTED_LABELS = {"prostate": {0, 1}, "mnms": {0, 1, 2, 3}, "fundus": {0, 1, 2}}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _parse_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _h5_files(root: Path) -> list[Path]:
    h5_root = root / "h5"
    if not h5_root.is_dir():
        return []
    return sorted(
        path
        for path in h5_root.rglob("*.h5")
        if path.is_file() and not path.is_symlink() and not path.name.startswith(("._", "."))
    )


def appledouble_files(root: Path) -> list[Path]:
    h5_root = root / "h5"
    if not h5_root.is_dir():
        return []
    return sorted(path for path in h5_root.rglob("._*.h5") if path.is_file() and not path.is_symlink())


def _payload_key(root: Path, path: Path) -> tuple[str, str]:
    parts = path.relative_to(root / "h5").parts
    if len(parts) < 5 or parts[1] not in {"images", "labels"}:
        raise ValueError(f"unexpected HDF5 layout: {path}")
    kind = parts[1]
    return kind, Path(*parts[2:]).as_posix()


def _label_values_attr(handle: h5py.File) -> set[int] | None:
    value = handle.attrs.get("allowed_labels")
    if value is None:
        return None
    parsed = _parse_attr(value)
    if not isinstance(parsed, list):
        raise ValueError("allowed_labels must be a JSON list")
    return {int(item) for item in parsed}


def _shape_attr(handle: h5py.File, name: str) -> list[int] | None:
    value = _parse_attr(handle.attrs.get(name))
    if not isinstance(value, list) or not all(isinstance(item, (int, float)) for item in value):
        return None
    return [int(item) for item in value]


def _contract_errors(handle: h5py.File, *, kind: str, expected_dataset: str) -> list[str]:
    """Validate the versioned HDF5 schema independently of source-data reads."""
    errors: list[str] = []
    attrs = set(handle.attrs.keys())
    missing = sorted(set(REQUIRED_H5_ATTRS) - attrs)
    if missing:
        errors.append(f"missing required attrs {missing}")
    dataset = str(handle.attrs.get("dataset", ""))
    if dataset not in EXPECTED_LABELS:
        errors.append(f"unknown dataset attr {dataset!r}")
        return errors
    if str(handle.attrs.get("preprocess_version", "")) != "v1":
        errors.append("unexpected preprocess_version")
    if int(handle.attrs.get("h5_schema_version", -1)) != 1:
        errors.append("unexpected h5_schema_version")
    for name in ("source_sha256", "preprocess_config_sha256"):
        if not SHA256_RE.fullmatch(str(handle.attrs.get(name, ""))):
            errors.append(f"invalid {name}")
    payload = handle[expected_dataset]
    expected_dtype = np.dtype("uint8") if kind == "labels" else (np.dtype("uint8") if dataset == "fundus" else np.dtype("float16"))
    if payload.dtype != expected_dtype:
        errors.append(f"unexpected dtype {payload.dtype}; expected {expected_dtype}")
    expected_chunks = (
        (3, 384, 384) if dataset == "fundus" and kind == "images" else
        (384, 384) if dataset == "fundus" else
        (1, 256, 256) if dataset == "prostate" else
        (1, 384, 384)
    )
    if tuple(payload.chunks or ()) != expected_chunks:
        errors.append(f"unexpected chunks {payload.chunks}; expected {expected_chunks}")
    expected_shape = (
        (3, 384, 384) if dataset == "fundus" and kind == "images" else
        (384, 384) if dataset == "fundus" else
        None
    )
    if expected_shape is not None and tuple(payload.shape) != expected_shape:
        errors.append(f"unexpected shape {payload.shape}; expected {expected_shape}")
    if dataset in {"prostate", "mnms"} and (payload.ndim != 3 or payload.shape[0] < 1):
        errors.append(f"invalid MRI payload shape {payload.shape}")
    if dataset == "prostate" and tuple(payload.shape[-2:]) != (256, 256):
        errors.append(f"unexpected prostate in-plane shape {payload.shape}")
    if dataset == "mnms" and tuple(payload.shape[-2:]) != (384, 384):
        errors.append(f"unexpected M&Ms in-plane shape {payload.shape}")
    if kind == "labels":
        allowed = _label_values_attr(handle)
        if allowed != EXPECTED_LABELS[dataset]:
            errors.append(f"allowed_labels {allowed} != {EXPECTED_LABELS[dataset]}")
        values = set(np.unique(payload[...]).astype(int).tolist())
        if not values.issubset(EXPECTED_LABELS[dataset]):
            errors.append(f"label values {sorted(values)} outside {sorted(EXPECTED_LABELS[dataset])}")
        if bool(handle.attrs.get("evaluation_eligible", False)) and not (payload[...] > 0).any():
            errors.append("evaluation-eligible label has empty foreground")
    image_processed_shape = _shape_attr(handle, "processed_shape")
    if image_processed_shape is None:
        errors.append("processed_shape attr is not an integer list")
    elif kind == "images" and image_processed_shape != list(payload.shape):
        errors.append(f"processed_shape {image_processed_shape} != payload shape {list(payload.shape)}")
    if dataset in {"prostate", "mnms"}:
        for name in ("original_spacing", "processed_spacing"):
            parsed = _parse_attr(handle.attrs.get(name))
            if (
                not isinstance(parsed, list)
                or len(parsed) != 3
                or not all(isinstance(item, (int, float)) and float(item) > 0 for item in parsed)
            ):
                errors.append(f"{name} must contain three MRI spacing values")
    return errors


def validate_h5_tree(root: Path) -> dict[str, Any]:
    """Validate HDF5 payloads and image/label pairing without touching raw inputs."""
    root = root.resolve()
    errors: list[str] = []
    h5_files = _h5_files(root)
    ignored_appledouble = appledouble_files(root)
    if not h5_files:
        errors.append("no HDF5 files found")
    pairs: dict[str, dict[str, Path]] = {}
    records: list[dict[str, Any]] = []
    for path in h5_files:
        try:
            kind, key = _payload_key(root, path)
            expected_dataset = "image" if kind == "images" else "label"
            allowed_labels: set[int] | None = None
            with h5py.File(path, "r") as handle:
                if expected_dataset == "label":
                    allowed_labels = _label_values_attr(handle)
                contract = _contract_errors(handle, kind=kind, expected_dataset=expected_dataset)
                if expected_dataset == "label" and allowed_labels is None:
                    raise ValueError("label HDF5 lacks allowed_labels")
            result = validate_h5_file(path, expected_dataset=expected_dataset, allowed_labels=allowed_labels)
            record_errors = [*result["errors"], *contract]
            if record_errors:
                errors.append(f"{path}: {record_errors}")
            pairs.setdefault(key, {})[kind] = path
            records.append({"relative_path": path.relative_to(root).as_posix(), "valid": not record_errors, "errors": record_errors})
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    for key, members in sorted(pairs.items()):
        if set(members) != {"images", "labels"}:
            errors.append(f"incomplete image/label pair: {key}")
            continue
        try:
            with h5py.File(members["images"], "r") as image_file, h5py.File(members["labels"], "r") as label_file:
                image, label = image_file["image"], label_file["label"]
                required_equal_attrs = REQUIRED_H5_ATTRS
                for attr in required_equal_attrs:
                    if str(image_file.attrs.get(attr, "")) != str(label_file.attrs.get(attr, "")):
                        errors.append(f"pair metadata mismatch {key}: {attr}")
                dataset = str(image_file.attrs.get("dataset", ""))
                if dataset == "fundus":
                    shape_ok = image.ndim == 3 and label.ndim == 2 and tuple(image.shape[1:]) == tuple(label.shape)
                else:
                    shape_ok = tuple(image.shape) == tuple(label.shape)
                if not shape_ok:
                    errors.append(f"pair shape mismatch {key}: image={image.shape}, label={label.shape}")
                if _shape_attr(image_file, "processed_shape") != _shape_attr(label_file, "processed_shape"):
                    errors.append(f"pair processed_shape mismatch {key}")
                if dataset == "fundus":
                    cup = np.asarray(label[...]) == 2
                    disc = np.asarray(label[...]) > 0
                    if not np.all(~cup | disc):
                        errors.append(f"fundus cup is outside disc region: {key}")
        except Exception as exc:
            errors.append(f"pair inspection failed {key}: {type(exc).__name__}: {exc}")
    temporary = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.tmp") if path.is_file()) if root.exists() else []
    if temporary:
        errors.append(f"stale temporary files present: {temporary}")
    return {
        "generated_at": utc_now(),
        "root": str(root),
        "h5_files": len(h5_files),
        "ignored_appledouble_files": len(ignored_appledouble),
        "complete_pairs": sum(set(members) == {"images", "labels"} for members in pairs.values()),
        "errors": errors,
        "valid": not errors,
        "records": records,
    }


def build_h5_inventory(root: Path, *, validation: dict[str, Any] | None = None) -> dict[str, Any]:
    root = root.resolve()
    validation = validation or validate_h5_tree(root)
    if not validation["valid"]:
        raise RuntimeError(f"HDF5 acceptance failed: {validation['errors']}")
    entries: list[dict[str, Any]] = []
    for path in _h5_files(root):
        kind, _ = _payload_key(root, path)
        with h5py.File(path, "r") as handle:
            dataset_name = "image" if kind == "images" else "label"
            dataset = handle[dataset_name]
            entries.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "kind": kind[:-1],
                    "sha256": sha256_path(path),
                    "bytes": path.stat().st_size,
                    "case_id": str(handle.attrs["case_id"]),
                    "patient_id": str(handle.attrs["patient_id"]),
                    "dataset": str(handle.attrs["dataset"]),
                    "site": str(handle.attrs.get("site", "")),
                    "vendor": str(handle.attrs.get("vendor", "")),
                    "phase": str(handle.attrs.get("phase", "")),
                    "shape": list(dataset.shape),
                    "dtype": str(dataset.dtype),
                    "preprocess_config_sha256": str(handle.attrs["preprocess_config_sha256"]),
                }
            )
    inventory = root / "manifests" / "h5_inventory.jsonl"
    write_text(inventory, "".join(canonical_json(entry) + "\n" for entry in entries))
    csv_fields = [
        "relative_path",
        "kind",
        "sha256",
        "bytes",
        "case_id",
        "patient_id",
        "dataset",
        "site",
        "vendor",
        "phase",
        "shape",
        "dtype",
        "preprocess_config_sha256",
    ]
    csv_inventory = root / "reports" / "preprocessing" / "h5_inventory.csv"
    project_csv_inventory = PROJECT_ROOT / "reports" / "preprocessing" / "h5_inventory.csv"
    write_csv(csv_inventory, entries, fieldnames=csv_fields)
    write_csv(project_csv_inventory, entries, fieldnames=csv_fields)
    if sha256_path(csv_inventory) != sha256_path(project_csv_inventory):
        raise RuntimeError("project/DataP HDF5 inventory CSV mirror mismatch")
    summary = {
        "generated_at": utc_now(),
        "entries": len(entries),
        "pairs": len(entries) // 2,
        "inventory": inventory.relative_to(root).as_posix(),
        "inventory_sha256": sha256_path(inventory),
        "csv_inventory": csv_inventory.relative_to(root).as_posix(),
        "csv_inventory_sha256": sha256_path(csv_inventory),
        "validation": {key: validation[key] for key in ("h5_files", "complete_pairs", "valid")},
    }
    write_json(root / "manifests" / "h5_inventory_summary.json", summary)
    return summary


def _allowed_transfer_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for prefix in TRANSFER_PREFIXES:
        source = root / prefix
        if not source.is_dir():
            continue
        for path in source.rglob("*"):
            if not path.is_file() or path.is_symlink() or path.suffix == ".tmp" or path.name.startswith(("._", ".")):
                continue
            relative = path.relative_to(root).as_posix()
            if relative in EXCLUDED_TRANSFER_RELATIVE_PATHS:
                continue
            paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def generate_checksums(root: Path) -> dict[str, Any]:
    root = root.resolve()
    entries = [
        {"relative_path": path.relative_to(root).as_posix(), "sha256": sha256_path(path), "bytes": path.stat().st_size}
        for path in _allowed_transfer_paths(root)
    ]
    text = "".join(f"{entry['sha256']}  {entry['relative_path']}\n" for entry in entries)
    output = root / "checksums" / "checksums.sha256"
    write_text(output, text)
    return {
        "generated_at": utc_now(),
        "entries": len(entries),
        "checksums": output.relative_to(root).as_posix(),
        "sha256": sha256_path(output),
    }


def verify_checksums(root: Path) -> dict[str, Any]:
    """Verify every frozen transfer payload against ``checksums.sha256``."""
    root = root.resolve()
    path = root / "checksums" / "checksums.sha256"
    errors: list[str] = []
    if not path.is_file():
        return {"generated_at": utc_now(), "valid": False, "entries": 0, "errors": [f"missing checksum file: {path}"]}
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"line {line_number}: invalid checksum format")
            continue
        relative_path = Path(relative)
        if not SHA256_RE.fullmatch(digest):
            errors.append(f"line {line_number}: invalid SHA-256")
        if relative_path.is_absolute() or ".." in relative_path.parts or relative in seen:
            errors.append(f"line {line_number}: unsafe or duplicate relative path {relative!r}")
            continue
        seen.add(relative)
        target = root / relative_path
        if not target.is_file():
            errors.append(f"missing checksummed file: {relative}")
        elif sha256_path(target) != digest:
            errors.append(f"checksum mismatch: {relative}")
        entries.append({"relative_path": relative, "sha256": digest})
    expected = {path.relative_to(root).as_posix() for path in _allowed_transfer_paths(root)}
    if seen != expected:
        missing = sorted(expected - seen)
        unexpected = sorted(seen - expected)
        if missing:
            errors.append(f"checksum list missing paths: {missing}")
        if unexpected:
            errors.append(f"checksum list has unexpected paths: {unexpected}")
    return {"generated_at": utc_now(), "valid": not errors, "entries": len(entries), "errors": errors, "checksums": path.relative_to(root).as_posix()}


def _checksum_entries(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        entries.append({"relative_path": relative, "sha256": digest})
    return entries


def build_transfer_manifest(root: Path, *, validation: dict[str, Any] | None = None) -> dict[str, Any]:
    root = root.resolve()
    validation = validation or validate_h5_tree(root)
    if not validation["valid"]:
        raise RuntimeError(f"HDF5 acceptance failed: {validation['errors']}")
    inventory = root / "manifests" / "h5_inventory.jsonl"
    if not inventory.is_file():
        raise FileNotFoundError("build the HDF5 inventory before the transfer manifest")
    checksum_summary = generate_checksums(root)
    entries = _checksum_entries(root / checksum_summary["checksums"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "root_name": root.name,
        "entries": entries,
        "h5_inventory_sha256": sha256_path(inventory),
        "checksums_sha256": checksum_summary["sha256"],
        "excluded": sorted(EXCLUDED_TRANSFER_RELATIVE_PATHS),
        "raw_source_included": False,
    }
    payload["content_sha256"] = config_sha256(payload)
    write_json(root / "manifests" / "transfer_manifest.json", payload)
    return {
        "manifest": "manifests/transfer_manifest.json",
        "content_sha256": payload["content_sha256"],
        "entries": len(entries),
        "h5_pairs": validation["complete_pairs"],
    }


def verify_transfer_root(root: Path, *, h5_validation: dict[str, Any] | None = None) -> dict[str, Any]:
    root = root.resolve()
    path = root / "manifests" / "transfer_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = dict(payload)
    observed_hash = str(expected.pop("content_sha256", ""))
    errors: list[str] = []
    if config_sha256(expected) != observed_hash:
        errors.append("transfer manifest content_sha256 mismatch")
    for entry in payload.get("entries", []):
        relative = Path(entry["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"unsafe relative path: {relative}")
            continue
        target = root / relative
        if not target.is_file():
            errors.append(f"missing bundle item: {relative}")
        elif sha256_path(target) != entry["sha256"]:
            errors.append(f"checksum mismatch: {relative}")
    h5_validation = h5_validation or validate_h5_tree(root)
    if not h5_validation["valid"]:
        errors.extend(h5_validation["errors"])
    return {
        "generated_at": utc_now(),
        "root": str(root),
        "valid": not errors,
        "errors": errors,
        "entries": len(payload.get("entries", [])),
        "h5_validation": {key: h5_validation[key] for key in ("h5_files", "complete_pairs", "valid")},
    }
