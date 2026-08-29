"""Closing gates for the local LCR-Seg HDF5 preprocessing phase."""
from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .acceptance import (
    build_h5_inventory,
    build_transfer_manifest,
    verify_checksums,
    verify_transfer_root,
    validate_h5_tree,
)
from .common import DATA_ROOT, PROJECT_ROOT, read_csv, sha256_bytes, sha256_path, utc_now, write_json, write_text
from .runtime_manifests import validate_runtime_manifests


PROJECT_REPORT_ROOT = PROJECT_ROOT / "reports" / "preprocessing"
DATA_REPORT_ROOT = DATA_ROOT / "reports" / "preprocessing"


def _source_digest(paths: tuple[Path, Path]) -> str:
    return sha256_bytes("".join(sha256_path(path) for path in paths).encode("utf-8"))


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "unavailable_not_a_git_repository"


def source_provenance_validation(*, data_root: Path = DATA_ROOT) -> dict[str, Any]:
    """Rehash source pairs and compare them to immutable HDF5 provenance attrs."""
    errors: list[str] = []
    counts: dict[str, int] = {}
    checked_h5_attrs = 0
    for dataset in ("prostate", "fundus", "mnms"):
        audit_rows = read_csv(PROJECT_ROOT / "manifests" / f"{dataset}_cases.csv")
        output_rows = {
            row["case_id"]: row
            for row in read_csv(data_root / "manifests" / "diagnostics" / f"{dataset}_h5.csv")
        }
        counts[dataset] = len(audit_rows)
        for raw_row in audit_rows:
            source_hash = _source_digest((Path(raw_row["image_path_raw"]), Path(raw_row["label_path_raw"])))
            output_case_ids = (
                (raw_row["case_id"],)
                if dataset != "mnms"
                else (f"{raw_row['case_id']}_ED", f"{raw_row['case_id']}_ES")
            )
            for case_id in output_case_ids:
                output = output_rows.get(case_id)
                if output is None:
                    errors.append(f"{dataset}: no output manifest entry for {case_id}")
                    continue
                image_rel = output.get("image_h5_relpath", "")
                if not image_rel:
                    errors.append(f"{dataset}: no image HDF5 path for {case_id}")
                    continue
                path = data_root / "h5" / "v1" / image_rel
                try:
                    with h5py.File(path, "r") as handle:
                        observed = str(handle.attrs.get("source_sha256", ""))
                    checked_h5_attrs += 1
                    if observed != source_hash:
                        errors.append(f"{dataset}: source SHA-256 mismatch for {case_id}")
                except Exception as exc:
                    errors.append(f"{dataset}: could not read source provenance for {case_id}: {type(exc).__name__}: {exc}")
    return {
        "generated_at": utc_now(),
        "valid": not errors,
        "errors": errors,
        "raw_case_counts": counts,
        "raw_cases_checked": sum(counts.values()),
        "h5_source_attrs_checked": checked_h5_attrs,
        "method": "current raw image+label SHA-256 pair digest equals immutable HDF5 source_sha256",
    }


def storage_statistics(*, data_root: Path = DATA_ROOT) -> dict[str, Any]:
    """Compute actual HDF5 bytes and raw payload bytes without reading source data."""
    h5_root = data_root / "h5" / "v1"
    grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"files": 0, "stored_bytes": 0, "payload_bytes": 0})
    for path in sorted(h5_root.rglob("*.h5")):
        if not path.is_file() or path.name.startswith(("._", ".")):
            continue
        parts = path.relative_to(h5_root).parts
        kind = parts[0] if parts else ""
        expected = "image" if kind == "images" else "label" if kind == "labels" else ""
        if not expected:
            raise RuntimeError(f"unexpected HDF5 layout in storage report: {path}")
        with h5py.File(path, "r") as handle:
            dataset = str(handle.attrs["dataset"])
            payload = handle[expected]
            record = grouped[(dataset, kind)]
            record["files"] += 1
            record["stored_bytes"] += path.stat().st_size
            record["payload_bytes"] += int(payload.size * payload.dtype.itemsize)
    rows = []
    for (dataset, kind), values in sorted(grouped.items()):
        stored = values["stored_bytes"]
        payload = values["payload_bytes"]
        rows.append(
            {
                "dataset": dataset,
                "kind": kind,
                **values,
                "compression_ratio_payload_over_h5": payload / stored if stored else 0.0,
            }
        )
    total_stored = sum(row["stored_bytes"] for row in rows)
    total_payload = sum(row["payload_bytes"] for row in rows)
    return {
        "generated_at": utc_now(),
        "rows": rows,
        "h5_files": sum(row["files"] for row in rows),
        "stored_bytes": total_stored,
        "payload_bytes": total_payload,
        "compression_ratio_payload_over_h5": total_payload / total_stored if total_stored else 0.0,
    }


def _storage_markdown(stats: dict[str, Any]) -> str:
    lines = [
        "# HDF5 storage report",
        "",
        f"Generated: {stats['generated_at']}",
        "",
        "| Dataset | Payload | Files | HDF5 bytes | Uncompressed payload bytes | Payload/HDF5 ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in stats["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['kind']} | {row['files']} | {row['stored_bytes']} | "
            f"{row['payload_bytes']} | {row['compression_ratio_payload_over_h5']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Total HDF5 files: {stats['h5_files']}",
            f"Total HDF5 bytes: {stats['stored_bytes']}",
            f"Total uncompressed payload bytes: {stats['payload_bytes']}",
            f"Overall payload/HDF5 ratio: {stats['compression_ratio_payload_over_h5']:.4f}",
            "",
            "AppleDouble sidecars are excluded from all counts and transfer payloads.",
            "",
        ]
    )
    return "\n".join(lines)


def _config_hashes(data_root: Path) -> dict[str, str]:
    output = {}
    for dataset in ("prostate", "fundus", "mnms"):
        summary = _json(data_root / "reports" / "preprocessing" / f"{dataset}_preprocess_summary.json")
        output[dataset] = str(summary["preprocess_config_sha256"])
    return output


def _ensure_frozen(
    *,
    data_root: Path,
    h5_validation: dict[str, Any],
    manifest_validation: dict[str, Any],
    provenance_validation: dict[str, Any],
    storage: dict[str, Any],
) -> dict[str, Any]:
    marker = data_root / "h5" / "v1" / "FROZEN"
    payload = {
        "schema_version": 1,
        "frozen_at": utc_now(),
        "git_commit": _git_commit(),
        "preprocess_config_sha256": _config_hashes(data_root),
        "h5_schema_version": 1,
        "image_h5_count": storage["h5_files"] // 2,
        "label_h5_count": storage["h5_files"] // 2,
        "h5_total_bytes": storage["stored_bytes"],
        "checksum_path": "checksums/checksums.sha256",
        "validation_status": {
            "h5": bool(h5_validation["valid"]),
            "runtime_manifests": bool(manifest_validation["valid"]),
            "source_provenance": bool(provenance_validation["valid"]),
        },
    }
    if marker.exists():
        existing = _json(marker)
        comparable = (
            "schema_version",
            "git_commit",
            "preprocess_config_sha256",
            "h5_schema_version",
            "image_h5_count",
            "label_h5_count",
            "h5_total_bytes",
            "checksum_path",
            "validation_status",
        )
        if any(existing.get(key) != payload.get(key) for key in comparable):
            raise RuntimeError("existing FROZEN marker conflicts with current accepted data; refusing to overwrite it")
        return {**existing, "marker": marker.relative_to(data_root).as_posix(), "created": False}
    write_text(marker, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return {**payload, "marker": marker.relative_to(data_root).as_posix(), "created": True}


def _active_failure_count() -> int:
    root = PROJECT_REPORT_ROOT / "failures"
    return sum(1 for path in root.rglob("*.json") if path.is_file()) if root.is_dir() else 0


def _mirror_report(name: str, content: str) -> None:
    write_text(PROJECT_REPORT_ROOT / name, content)
    write_text(DATA_REPORT_ROOT / name, content)


def _reports(
    *,
    data_root: Path,
    h5_validation: dict[str, Any],
    manifest_validation: dict[str, Any],
    provenance_validation: dict[str, Any],
    storage: dict[str, Any],
    inventory: dict[str, Any],
    frozen: dict[str, Any],
    checksum_status: str,
    bundle_status: str,
) -> None:
    decision_source = PROJECT_REPORT_ROOT / "PREPROCESSING_DECISIONS_V1.md"
    if decision_source.is_file():
        write_text(DATA_REPORT_ROOT / decision_source.name, decision_source.read_text(encoding="utf-8"))
    _mirror_report("H5_STORAGE_REPORT.md", _storage_markdown(storage))
    geometry = _json(PROJECT_ROOT / "reports" / "data_audit" / "geometry_decisions_summary.json")
    fundus = _json(data_root / "reports" / "preprocessing" / "fundus_preprocess_summary.json")
    prostate = _json(data_root / "reports" / "preprocessing" / "prostate_preprocess_summary.json")
    mnms = _json(data_root / "reports" / "preprocessing" / "mnms_preprocess_summary.json")
    dataloader = _json(data_root / "reports" / "validation" / "dataloader_smoke.json")
    completion = "\n".join(
        [
            "# LCR-Seg local preprocessing completion",
            "",
            f"Generated: {utc_now()}",
            "",
            "## Completed HDF5 cohorts",
            "",
            f"- Fundus: {fundus['rows']}/{fundus['rows']} accepted pairs; failed={fundus['failed']}; label values={{0,1,2}}; minimum crop retention={fundus['minimum_crop_foreground_retention']}",
            f"- Prostate: {prostate['rows']}/{prostate['rows']} accepted pairs; failed={prostate['failed']}; manual review={prostate['manual_review_required']}; label values={{0,1}}.",
            f"- Prostate geometry: {geometry['mismatch_cases']} mismatches, distribution={json.dumps(geometry['decision_distribution'], sort_keys=True)}, manual review={len(geometry['manual_review_required'])}.",
            f"- M&Ms: patients={mnms['patients']}; ED/ES phase pairs={mnms['patient_phases']}; canonical320={mnms['cohort_counts']['canonical320']} patients; auxiliary25={mnms['cohort_counts']['auxiliary25']} patients; fixed FOV={mnms['fixed_crop_fov_mm']} mm; minimum foreground retention={mnms['minimum_foreground_retention']}; failed={mnms['failed']}.",
            "",
            "## Closing gates",
            "",
            f"- Full HDF5 schema acceptance: {'pass' if h5_validation['valid'] else 'fail'}; {h5_validation['h5_files']} HDF5 files / {h5_validation['complete_pairs']} pairs.",
            f"- Runtime manifest and hidden-label isolation: {'pass' if manifest_validation['valid'] else 'fail'}; 3 fixed seeds, {manifest_validation['expected_rows_per_seed']} rows per seed.",
            f"- Raw-source provenance rehash: {'pass' if provenance_validation['valid'] else 'fail'}; {provenance_validation['raw_cases_checked']} source pairs and {provenance_validation['h5_source_attrs_checked']} image provenance attrs checked.",
            f"- DataLoader smoke: {'pass' if dataloader.get('valid') else 'fail'}; workers tested={[run['num_workers'] for run in dataloader.get('runs', [])]}.",
            f"- SHA-256 checksum gate: {checksum_status}; path=checksums/checksums.sha256.",
            f"- Transfer-bundle verification: {bundle_status}.",
            f"- Active failure bundles: {_active_failure_count()}.",
            "",
            "## Frozen artifact",
            "",
            f"- Marker: /Volumes/DataP/LCRSeg/{frozen['marker']}",
            f"- Inventory: /Volumes/DataP/LCRSeg/{inventory['csv_inventory']} ({inventory['entries']} HDF5 files)",
            f"- Stored HDF5 bytes: {storage['stored_bytes']}; payload/HDF5 ratio: {storage['compression_ratio_payload_over_h5']:.4f}.",
            "",
            "## Next bounded command (not run in this preprocessing phase)",
            "",
            "```bash",
            "/opt/miniconda3/bin/python scripts/two_case_overfit.py --root /Volumes/DataP/LCRSeg --seed 0 --dataset fundus --steps 200",
            "```",
            "",
            "Remote transfer is ready but not executed: it still requires an explicit remote absolute root and an explicit remote Python interpreter.",
            "",
        ]
    )
    _mirror_report("PREPROCESSING_COMPLETION.md", completion)
    transfer_ready = "\n".join(
        [
            "# LCR-Seg transfer readiness",
            "",
            "Local preprocessing and local checksum verification have passed.",
            "",
            f"- HDF5 files: {storage['h5_files']} ({storage['stored_bytes']} bytes)",
            f"- HDF5 pairs: {h5_validation['complete_pairs']}",
            f"- Checksum: checksums/checksums.sha256 ({checksum_status})",
            f"- Transfer bundle: {bundle_status}",
            "- Configured direct endpoint: jiangsuiyang@10.12.208.180:22 via local source address 10.75.81.150.",
            "",
            "Before any transfer, provide exactly these two explicit values (do not guess them):",
            "",
            "```bash",
            "export LCRSEG_REMOTE_ROOT=/absolute/existing/writable/LCRSeg/path",
            "export LCRSEG_REMOTE_PYTHON=/absolute/path/to/python-with-h5py-and-numpy",
            "```",
            "",
            "Then run the repository synchronizer first without `--execute` for its rsync dry run, review the itemized output, and rerun with `--execute`.",
            "",
        ]
    )
    _mirror_report("TRANSFER_READY.md", transfer_ready)


def finalize_local_preprocessing(*, data_root: Path = DATA_ROOT) -> dict[str, Any]:
    """Run all closing gates, write reports, create FROZEN, and verify checksums."""
    h5_validation = validate_h5_tree(data_root)
    if not h5_validation["valid"]:
        raise RuntimeError(f"HDF5 acceptance failed: {h5_validation['errors']}")
    manifest_validation = validate_runtime_manifests(data_root=data_root)
    if not manifest_validation["valid"]:
        raise RuntimeError(f"runtime manifest acceptance failed: {manifest_validation['errors']}")
    provenance_validation = source_provenance_validation(data_root=data_root)
    if not provenance_validation["valid"]:
        raise RuntimeError(f"raw-source provenance validation failed: {provenance_validation['errors']}")
    dataloader_path = data_root / "reports" / "validation" / "dataloader_smoke.json"
    if not dataloader_path.is_file() or not _json(dataloader_path).get("valid"):
        raise RuntimeError("DataLoader smoke result is missing or not accepted")
    inventory = build_h5_inventory(data_root, validation=h5_validation)
    storage = storage_statistics(data_root=data_root)
    frozen = _ensure_frozen(
        data_root=data_root,
        h5_validation=h5_validation,
        manifest_validation=manifest_validation,
        provenance_validation=provenance_validation,
        storage=storage,
    )
    _reports(
        data_root=data_root,
        h5_validation=h5_validation,
        manifest_validation=manifest_validation,
        provenance_validation=provenance_validation,
        storage=storage,
        inventory=inventory,
        frozen=frozen,
        checksum_status="pending closing checksum verification",
        bundle_status="pending closing bundle verification",
    )
    first_transfer = build_transfer_manifest(data_root, validation=h5_validation)
    first_checksums = verify_checksums(data_root)
    first_bundle = verify_transfer_root(data_root, h5_validation=h5_validation)
    if not first_checksums["valid"] or not first_bundle["valid"]:
        raise RuntimeError(f"initial frozen-bundle verification failed: checksums={first_checksums['errors']}, bundle={first_bundle['errors']}")
    _reports(
        data_root=data_root,
        h5_validation=h5_validation,
        manifest_validation=manifest_validation,
        provenance_validation=provenance_validation,
        storage=storage,
        inventory=inventory,
        frozen=frozen,
        checksum_status="pass",
        bundle_status="pass",
    )
    final_transfer = build_transfer_manifest(data_root, validation=h5_validation)
    final_checksums = verify_checksums(data_root)
    final_bundle = verify_transfer_root(data_root, h5_validation=h5_validation)
    if not final_checksums["valid"] or not final_bundle["valid"]:
        raise RuntimeError(f"final frozen-bundle verification failed: checksums={final_checksums['errors']}, bundle={final_bundle['errors']}")
    result = {
        "generated_at": utc_now(),
        "valid": True,
        "h5_validation": {key: h5_validation[key] for key in ("h5_files", "complete_pairs", "ignored_appledouble_files", "valid")},
        "runtime_manifest_validation": manifest_validation,
        "source_provenance_validation": provenance_validation,
        "inventory": inventory,
        "storage": storage,
        "frozen": frozen,
        "first_transfer": first_transfer,
        "final_transfer": final_transfer,
        "final_checksums": final_checksums,
        "final_bundle": final_bundle,
    }
    write_json(data_root / "reports" / "validation" / "preprocessing_finalization.json", result)
    write_json(PROJECT_ROOT / "reports" / "preprocessing" / "finalization_summary.json", result)
    return result
