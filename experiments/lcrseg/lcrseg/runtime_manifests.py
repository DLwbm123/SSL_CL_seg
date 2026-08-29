"""Build and validate HDF5 runtime manifests without exposing hidden labels.

The audit manifests in ``PROJECT_ROOT/manifests`` retain raw-source facts.  This
module creates a separate, portable runtime view with paths relative to
``/Volumes/DataP/LCRSeg/h5/v1``.  Training manifests deliberately remove both
the label path and label hash for every ``train_unlabeled`` row; the full label
view is retained only in diagnostics manifests.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .common import DATA_ROOT, PROJECT_ROOT, read_csv, sha256_path, utc_now, write_csv, write_json, write_text
from .splits import generate_mnms


PROJECT_MANIFEST_ROOT = PROJECT_ROOT / "manifests"
PROJECT_SPLIT_ROOT = PROJECT_ROOT / "splits"
DATA_MANIFEST_ROOT = DATA_ROOT / "manifests"
DATA_SPLIT_ROOT = DATA_ROOT / "splits"
PROJECT_REPORT_ROOT = PROJECT_ROOT / "reports" / "preprocessing"
DATA_REPORT_ROOT = DATA_ROOT / "reports" / "preprocessing"
SEEDS = (0, 1, 2)
EXPECTED_CASE_ROWS = {"prostate": 116, "fundus": 660, "mnms": 690}
EXPECTED_TOTAL_ROWS = sum(EXPECTED_CASE_ROWS.values())

COMMON_FIELDS = [
    "case_id",
    "patient_id",
    "dataset",
    "split_seed",
    "split",
    "primary_20pct_split",
    "site_or_vendor",
    "site",
    "vendor",
    "phase",
    "source_partition",
    "cohort",
    "training_role",
    "evaluation_eligible",
    "labelled_at_10pct",
    "labelled_at_20pct",
    "labelled_at_40pct",
    "image_h5_relpath",
    "label_h5_relpath",
    "image_sha256",
    "label_sha256",
    "preprocess_config_sha256",
]
DIAGNOSTIC_FIELDS = COMMON_FIELDS + [
    "source_status",
    "diagnostic_label_nonempty",
    "foreground_retention",
    "crop_foreground_retention",
    "geometry_decision",
]


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _bool_text(value: Any) -> str:
    return "true" if _bool(value) else "false"


def _path_is_h5_relative(value: str, prefix: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and path.as_posix().startswith(prefix)


def _diagnostic_h5_rows(dataset: str) -> dict[str, dict[str, str]]:
    path = DATA_MANIFEST_ROOT / "diagnostics" / f"{dataset}_h5.csv"
    rows = read_csv(path)
    expected = EXPECTED_CASE_ROWS[dataset]
    if len(rows) != expected:
        raise RuntimeError(f"{dataset}: expected {expected} HDF5 diagnostic rows, found {len(rows)}")
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = row.get("case_id", "")
        if not case_id or case_id in lookup:
            raise RuntimeError(f"{dataset}: missing or duplicate HDF5 diagnostic case_id {case_id!r}")
        if row.get("status") not in {"written", "skipped"}:
            raise RuntimeError(f"{dataset}: HDF5 row {case_id} is not accepted: {row.get('status')} {row.get('error', '')}")
        if not _path_is_h5_relative(row.get("image_h5_relpath", ""), "images/"):
            raise RuntimeError(f"{dataset}: invalid image HDF5 relative path for {case_id}")
        if not _path_is_h5_relative(row.get("label_h5_relpath", ""), "labels/"):
            raise RuntimeError(f"{dataset}: invalid label HDF5 relative path for {case_id}")
        lookup[case_id] = row
    return lookup


def _load_split(dataset: str, seed: int) -> dict[str, Any]:
    path = PROJECT_SPLIT_ROOT / f"{dataset}_seed{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError(f"invalid split payload: {path}")
    return payload


def _mirror_split(dataset: str, seed: int) -> str:
    source = PROJECT_SPLIT_ROOT / f"{dataset}_seed{seed}.json"
    target = DATA_SPLIT_ROOT / source.name
    write_text(target, source.read_text(encoding="utf-8"))
    return sha256_path(source)


def _h5_record(split_record: dict[str, Any], source_row: dict[str, str], *, seed: int) -> dict[str, str]:
    return {
        "case_id": source_row["case_id"],
        "patient_id": source_row["patient_id"],
        "dataset": source_row["dataset"],
        "split_seed": str(seed),
        "split": str(split_record["split"]),
        "primary_20pct_split": str(split_record["primary_20pct_split"]),
        "site_or_vendor": str(split_record.get("site_or_vendor", source_row.get("site") or source_row.get("vendor") or "")),
        "site": source_row.get("site", ""),
        "vendor": source_row.get("vendor", ""),
        "phase": source_row.get("phase", ""),
        "source_partition": source_row.get("source_partition", ""),
        "cohort": source_row.get("cohort", split_record.get("cohort", "canonical")),
        "training_role": source_row.get("training_role", split_record.get("training_role", "standard")),
        "evaluation_eligible": _bool_text(source_row.get("evaluation_eligible", split_record.get("evaluation_eligible", True))),
        "labelled_at_10pct": _bool_text(split_record.get("labelled_at_10pct", False)),
        "labelled_at_20pct": _bool_text(split_record.get("labelled_at_20pct", False)),
        "labelled_at_40pct": _bool_text(split_record.get("labelled_at_40pct", False)),
        "image_h5_relpath": source_row["image_h5_relpath"],
        "label_h5_relpath": source_row["label_h5_relpath"],
        "image_sha256": source_row.get("image_sha256", ""),
        "label_sha256": source_row.get("label_sha256", ""),
        "preprocess_config_sha256": source_row.get("preprocess_config_sha256", ""),
    }


def _records_for_dataset(dataset: str, seed: int, h5_rows: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    payload = _load_split(dataset, seed)
    records = payload["records"]
    output: list[dict[str, str]] = []
    if dataset == "mnms":
        if len(records) != 345:
            raise RuntimeError(f"mnms seed {seed}: expected 345 patient split records, got {len(records)}")
        for split_record in records:
            patient_id = str(split_record["patient_id"])
            for phase in ("ED", "ES"):
                case_id = f"{patient_id}_{phase}"
                try:
                    source_row = h5_rows[case_id]
                except KeyError as exc:
                    raise RuntimeError(f"mnms seed {seed}: no HDF5 row for {case_id}") from exc
                output.append(_h5_record(split_record, source_row, seed=seed))
    else:
        if len(records) != EXPECTED_CASE_ROWS[dataset]:
            raise RuntimeError(f"{dataset} seed {seed}: unexpected fixed split record count {len(records)}")
        for split_record in records:
            case_id = str(split_record["case_id"])
            try:
                source_row = h5_rows[case_id]
            except KeyError as exc:
                raise RuntimeError(f"{dataset} seed {seed}: no HDF5 row for {case_id}") from exc
            output.append(_h5_record(split_record, source_row, seed=seed))
    if len(output) != EXPECTED_CASE_ROWS[dataset]:
        raise RuntimeError(f"{dataset} seed {seed}: expected {EXPECTED_CASE_ROWS[dataset]} runtime rows, got {len(output)}")
    if len({row["case_id"] for row in output}) != len(output):
        raise RuntimeError(f"{dataset} seed {seed}: duplicate runtime case IDs")
    if set(h5_rows) != {row["case_id"] for row in output}:
        raise RuntimeError(f"{dataset} seed {seed}: split and HDF5 case-ID sets differ")
    return output


def _training_row(diagnostic_row: dict[str, str]) -> dict[str, str]:
    output = {key: diagnostic_row[key] for key in COMMON_FIELDS}
    if output["primary_20pct_split"] == "train_unlabeled":
        output["label_h5_relpath"] = ""
        output["label_sha256"] = ""
    return output


def _diagnostic_row(common_row: dict[str, str], source_row: dict[str, str]) -> dict[str, str]:
    return {
        **common_row,
        "source_status": source_row.get("status", ""),
        "diagnostic_label_nonempty": source_row.get("diagnostic_label_nonempty", ""),
        "foreground_retention": source_row.get("foreground_retention", ""),
        "crop_foreground_retention": source_row.get("crop_foreground_retention", ""),
        "geometry_decision": source_row.get("geometry_decision", ""),
    }


def _write_mirrored_csv(relative: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> dict[str, str]:
    project_path = PROJECT_MANIFEST_ROOT / relative
    data_path = DATA_MANIFEST_ROOT / relative
    write_csv(project_path, rows, fieldnames=fieldnames)
    write_csv(data_path, rows, fieldnames=fieldnames)
    project_hash = sha256_path(project_path)
    data_hash = sha256_path(data_path)
    if project_hash != data_hash:
        raise RuntimeError(f"manifest mirror hash mismatch: {relative}")
    return {"relative_path": relative.as_posix(), "sha256": project_hash, "rows": str(len(rows))}


def _summary_counts(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts["dataset"][row["dataset"]] += 1
        counts["primary_20pct_split"][row["primary_20pct_split"]] += 1
        counts["cohort"][row["cohort"]] += 1
        counts["evaluation_eligible"][row["evaluation_eligible"]] += 1
    return {name: dict(sorted(counter.items())) for name, counter in sorted(counts.items())}


def build_runtime_manifests(*, seeds: Iterable[int] = SEEDS) -> dict[str, Any]:
    """Build runtime manifests for all three frozen seeds and mirror transfer assets."""
    selected_seeds = tuple(seeds)
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("seeds must be a non-empty unique sequence")
    for seed in selected_seeds:
        generate_mnms(seed)

    h5_rows = {dataset: _diagnostic_h5_rows(dataset) for dataset in EXPECTED_CASE_ROWS}
    output_files: list[dict[str, str]] = []
    split_hashes: dict[str, str] = {}
    summaries: dict[str, Any] = {}
    for seed in selected_seeds:
        for dataset in ("prostate", "fundus", "mnms"):
            split_hashes[f"{dataset}_seed{seed}.json"] = _mirror_split(dataset, seed)
        common_rows = [
            row
            for dataset in ("prostate", "fundus", "mnms")
            for row in _records_for_dataset(dataset, seed, h5_rows[dataset])
        ]
        if len(common_rows) != EXPECTED_TOTAL_ROWS:
            raise RuntimeError(f"seed {seed}: expected {EXPECTED_TOTAL_ROWS} runtime rows, got {len(common_rows)}")
        source_by_case = {case_id: row for rows in h5_rows.values() for case_id, row in rows.items()}
        diagnostic_rows = [_diagnostic_row(row, source_by_case[row["case_id"]]) for row in common_rows]
        training_rows = [_training_row(row) for row in diagnostic_rows]
        output_files.append(_write_mirrored_csv(Path("training") / f"lcrseg_v1_seed{seed}.csv", training_rows, COMMON_FIELDS))
        output_files.append(_write_mirrored_csv(Path("diagnostics") / f"lcrseg_v1_seed{seed}.csv", diagnostic_rows, DIAGNOSTIC_FIELDS))
        summaries[f"seed{seed}"] = {
            "rows": len(common_rows),
            "training": _summary_counts(training_rows),
            "diagnostics": _summary_counts(diagnostic_rows),
        }

    summary = {
        "generated_at": utc_now(),
        "schema_version": 1,
        "h5_root_relative": "h5/v1",
        "seeds": list(selected_seeds),
        "expected_total_rows_per_seed": EXPECTED_TOTAL_ROWS,
        "split_sha256": split_hashes,
        "manifest_files": output_files,
        "summaries": summaries,
        "hidden_label_policy": "training train_unlabeled rows have empty label_h5_relpath and label_sha256; diagnostics retain label paths",
    }
    write_json(PROJECT_REPORT_ROOT / "runtime_manifest_build_summary.json", summary)
    write_json(DATA_REPORT_ROOT / "runtime_manifest_build_summary.json", summary)
    return summary


def _runtime_csv(root: Path, category: str, seed: int) -> Path:
    return root / "manifests" / category / f"lcrseg_v1_seed{seed}.csv"


def _validate_relative_path(root: Path, relative: str, prefix: str, errors: list[str], context: str) -> None:
    if not _path_is_h5_relative(relative, prefix):
        errors.append(f"{context}: invalid HDF5 relative path {relative!r}")
        return
    path = root / "h5" / "v1" / relative
    if not path.is_file():
        errors.append(f"{context}: missing HDF5 file {relative}")


def validate_runtime_manifests(*, data_root: Path = DATA_ROOT, seeds: Iterable[int] = SEEDS) -> dict[str, Any]:
    """Check HDF5-path coverage, split integrity, and hidden-label isolation."""
    selected_seeds = tuple(seeds)
    errors: list[str] = []
    summaries: dict[str, Any] = {}
    h5_root = data_root / "h5" / "v1"
    images = sorted(path for path in (h5_root / "images").rglob("*.h5") if path.is_file() and not path.name.startswith(("._", ".")))
    labels = sorted(path for path in (h5_root / "labels").rglob("*.h5") if path.is_file() and not path.name.startswith(("._", ".")))
    if len(images) != EXPECTED_TOTAL_ROWS or len(labels) != EXPECTED_TOTAL_ROWS:
        errors.append(f"HDF5 pair count mismatch: images={len(images)}, labels={len(labels)}, expected={EXPECTED_TOTAL_ROWS}")
    expected_image_rel = {path.relative_to(h5_root).as_posix() for path in images}
    expected_label_rel = {path.relative_to(h5_root).as_posix() for path in labels}

    for seed in selected_seeds:
        training_path = _runtime_csv(data_root, "training", seed)
        diagnostics_path = _runtime_csv(data_root, "diagnostics", seed)
        project_training = _runtime_csv(PROJECT_ROOT, "training", seed)
        project_diagnostics = _runtime_csv(PROJECT_ROOT, "diagnostics", seed)
        for data_path, project_path in ((training_path, project_training), (diagnostics_path, project_diagnostics)):
            if not data_path.is_file():
                errors.append(f"missing data manifest: {data_path}")
            elif not project_path.is_file():
                errors.append(f"missing project manifest mirror: {project_path}")
            elif sha256_path(data_path) != sha256_path(project_path):
                errors.append(f"manifest mirror mismatch: {data_path.name}")
        if not training_path.is_file() or not diagnostics_path.is_file():
            continue
        training = read_csv(training_path)
        diagnostics = read_csv(diagnostics_path)
        if len(training) != EXPECTED_TOTAL_ROWS or len(diagnostics) != EXPECTED_TOTAL_ROWS:
            errors.append(f"seed {seed}: manifest row counts training={len(training)}, diagnostics={len(diagnostics)}")
        for name, rows in (("training", training), ("diagnostics", diagnostics)):
            case_ids = [row.get("case_id", "") for row in rows]
            if len(case_ids) != len(set(case_ids)):
                errors.append(f"seed {seed} {name}: duplicate case_id")
            if set(case_ids) != {row.get("case_id", "") for row in diagnostics}:
                errors.append(f"seed {seed} {name}: case-ID coverage differs from diagnostics")
        diagnostics_by_case = {row["case_id"]: row for row in diagnostics}
        patient_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
        mnms_phases: dict[str, set[str]] = defaultdict(set)
        auxiliary_rows = 0
        for row in diagnostics:
            context = f"seed {seed} diagnostics {row.get('case_id', '?')}"
            _validate_relative_path(data_root, row.get("image_h5_relpath", ""), "images/", errors, context)
            _validate_relative_path(data_root, row.get("label_h5_relpath", ""), "labels/", errors, context)
            patient_splits[(row.get("dataset", ""), row.get("patient_id", ""))].add(row.get("split", ""))
            if row.get("dataset") == "mnms":
                mnms_phases[row.get("patient_id", "")].add(row.get("phase", ""))
            if row.get("cohort") == "auxiliary25":
                auxiliary_rows += 1
                if row.get("split") != "train" or row.get("primary_20pct_split") != "train_unlabeled":
                    errors.append(f"{context}: auxiliary25 leaked outside train_unlabeled")
                if _bool(row.get("evaluation_eligible")):
                    errors.append(f"{context}: auxiliary25 marked evaluation eligible")
            if row.get("split") in {"val", "test"} and not _bool(row.get("evaluation_eligible")):
                errors.append(f"{context}: non-evaluation record appears in {row.get('split')}")
        for key, values in patient_splits.items():
            if len(values) != 1:
                errors.append(f"seed {seed}: patient split overlap for {key}: {sorted(values)}")
        if len(mnms_phases) != 345 or any(phases != {"ED", "ES"} for phases in mnms_phases.values()):
            errors.append(f"seed {seed}: M&Ms ED/ES phase coverage is invalid")
        if auxiliary_rows != 50:
            errors.append(f"seed {seed}: expected 50 auxiliary25 phase rows, got {auxiliary_rows}")

        for row in training:
            context = f"seed {seed} training {row.get('case_id', '?')}"
            _validate_relative_path(data_root, row.get("image_h5_relpath", ""), "images/", errors, context)
            diagnostic = diagnostics_by_case.get(row.get("case_id", ""))
            if diagnostic is None:
                errors.append(f"{context}: no diagnostics counterpart")
                continue
            if row.get("primary_20pct_split") == "train_unlabeled":
                if row.get("label_h5_relpath") or row.get("label_sha256"):
                    errors.append(f"{context}: hidden-label leakage in train_unlabeled")
            else:
                _validate_relative_path(data_root, row.get("label_h5_relpath", ""), "labels/", errors, context)
            if "diagnostics" in row.get("image_h5_relpath", "") or "diagnostics" in row.get("label_h5_relpath", ""):
                errors.append(f"{context}: training manifest points into diagnostics")
        training_image_rel = {row.get("image_h5_relpath", "") for row in training}
        diagnostic_image_rel = {row.get("image_h5_relpath", "") for row in diagnostics}
        diagnostic_label_rel = {row.get("label_h5_relpath", "") for row in diagnostics}
        if training_image_rel != expected_image_rel or diagnostic_image_rel != expected_image_rel:
            errors.append(f"seed {seed}: image manifest/HDF5 coverage mismatch")
        if diagnostic_label_rel != expected_label_rel:
            errors.append(f"seed {seed}: diagnostic label manifest/HDF5 coverage mismatch")
        summaries[f"seed{seed}"] = {
            "training_rows": len(training),
            "diagnostic_rows": len(diagnostics),
            "auxiliary25_phase_rows": auxiliary_rows,
            "training_counts": _summary_counts(training),
        }

    return {
        "generated_at": utc_now(),
        "valid": not errors,
        "errors": errors,
        "h5_images": len(images),
        "h5_labels": len(labels),
        "expected_rows_per_seed": EXPECTED_TOTAL_ROWS,
        "summaries": summaries,
    }
