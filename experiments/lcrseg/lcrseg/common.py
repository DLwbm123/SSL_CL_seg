from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/Volumes/DataP/LCRSeg")
PROSTATE_ROOT = Path("/Volumes/DataP/SMG_Learning/raw/prostate_six_site")
MNMS_ROOT = Path("/Volumes/DataP/Mega/OpenDataset")
FUNDUS_ROOT = Path("/Volumes/DataP/CL_medical_classification/Fundus")
H5_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def visible_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted(p for p in paths if not p.name.startswith((".", "._")))


def sha256_path(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def config_sha256(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(config).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n"
    _atomic_write(path, data.encode("utf-8"))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, content.encode("utf-8"))


def _atomic_write(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", newline="", delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_value(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def json_attr(value: Any) -> str:
    return canonical_json(value)


def set_h5_attrs(handle: h5py.File, attrs: dict[str, Any]) -> None:
    for key, value in attrs.items():
        if value is None:
            handle.attrs[key] = ""
        elif isinstance(value, (str, int, float, bool, np.number)):
            handle.attrs[key] = value
        else:
            handle.attrs[key] = json_attr(value)


def write_h5_atomically(
    target: Path,
    *,
    dataset_name: str,
    array: np.ndarray,
    attrs: dict[str, Any],
    chunks: tuple[int, ...],
    compression: str = "gzip",
    compression_level: int = 4,
) -> str:
    """Write one immutable HDF5 payload, validate it, then atomically expose it."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite stale temporary HDF5: {temporary}")
    with h5py.File(temporary, "w") as handle:
        dataset = handle.create_dataset(
            dataset_name,
            data=array,
            chunks=chunks,
            compression=compression,
            compression_opts=compression_level,
            shuffle=True,
            fletcher32=True,
        )
        dataset.attrs["dtype"] = str(array.dtype)
        set_h5_attrs(handle, attrs)
        handle.flush()
    result = validate_h5_file(temporary, expected_dataset=dataset_name)
    if not result["valid"]:
        raise ValueError(f"Temporary HDF5 validation failed for {temporary}: {result['errors']}")
    digest = sha256_path(temporary)
    os.replace(temporary, target)
    return digest


def validate_h5_file(path: Path, *, expected_dataset: str | None = None, allowed_labels: set[int] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    details: dict[str, Any] = {"path": str(path), "errors": errors}
    try:
        with h5py.File(path, "r") as handle:
            names = list(handle.keys())
            if expected_dataset is not None and expected_dataset not in handle:
                errors.append(f"missing dataset {expected_dataset}")
            if len(names) != 1:
                errors.append(f"expected one dataset, found {names}")
            name = (
                expected_dataset
                if expected_dataset is not None and expected_dataset in handle
                else (names[0] if names else None)
            )
            if name is not None:
                dataset = handle[name]
                details.update(
                    dataset=name,
                    shape=list(dataset.shape),
                    dtype=str(dataset.dtype),
                    chunks=list(dataset.chunks or ()),
                    compression=dataset.compression,
                    compression_opts=dataset.compression_opts,
                    shuffle=bool(dataset.shuffle),
                    fletcher32=bool(dataset.fletcher32),
                    attrs=sorted(handle.attrs.keys()),
                )
                if dataset.compression != "gzip" or dataset.compression_opts != 4:
                    errors.append("unexpected compression contract")
                if not dataset.shuffle or not dataset.fletcher32:
                    errors.append("shuffle/fletcher32 contract not met")
                if dataset.dtype.kind == "f":
                    finite = bool(np.isfinite(dataset[...]).all())
                    details["finite"] = finite
                    if not finite:
                        errors.append("NaN/Inf values")
                if allowed_labels is not None:
                    values = set(np.unique(dataset[...]).astype(int).tolist())
                    details["label_values"] = sorted(values)
                    if not values.issubset(allowed_labels):
                        errors.append(f"labels {sorted(values)} not in {sorted(allowed_labels)}")
            required = {"case_id", "dataset", "preprocess_version", "h5_schema_version", "preprocess_config_sha256"}
            missing = sorted(required - set(handle.attrs.keys()))
            if missing:
                errors.append(f"missing attrs {missing}")
    except Exception as exc:  # diagnostic boundary
        errors.append(f"{type(exc).__name__}: {exc}")
    details["valid"] = not errors
    return details


def require_confirmed_mapping(config_path: Path) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not config.get("confirmed") or not config.get("mapping"):
        state = config.get("status", "UNCONFIRMED")
        raise RuntimeError(f"BLOCKER: {config_path.name} is not confirmed ({state})")
    return config


def relative_to_data(path: Path, *, data_root: Path = DATA_ROOT) -> str:
    return path.relative_to(data_root).as_posix()


def ensure_no_overwrite(
    target: Path,
    *,
    expected_config_hash: str | None = None,
    expected_source_hash: str | None = None,
) -> bool:
    """Return True only when an existing accepted file is exactly reusable."""
    if not target.exists():
        return False
    result = validate_h5_file(target)
    if not result["valid"]:
        raise FileExistsError(f"Existing target is invalid and will not be overwritten: {target}")
    with h5py.File(target, "r") as handle:
        existing_config = str(handle.attrs.get("preprocess_config_sha256", ""))
        existing_source = str(handle.attrs.get("source_sha256", ""))
    config_matches = expected_config_hash is None or existing_config == expected_config_hash
    source_matches = expected_source_hash is None or existing_source == expected_source_hash
    if config_matches and source_matches:
        return True
    raise FileExistsError(
        "Existing accepted target conflicts with this source/configuration: "
        f"{target} (config={existing_config}, source={existing_source})"
    )
