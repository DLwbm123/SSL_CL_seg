from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import sha256_file


TRAIN_ROLES = frozenset({"train_labeled", "train_unlabeled"})
EVAL_ROLES = frozenset({"val", "test"})
FORBIDDEN_UNLABELED_FIELDS = ("label_h5_relpath", "label_sha256")


@dataclass(frozen=True)
class ManifestRecord:
    case_id: str
    patient_id: str
    dataset: str
    domain: str
    role: str
    image_h5_relpath: str
    label_h5_relpath: str | None


def _safe_relative(value: str, field: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe {field}: {value!r}")
    return path.as_posix()


class LCRSegManifestAdapter:
    """Read only the frozen training manifest and expose role-scoped views."""

    def __init__(self, data_root: str | Path, protocol: dict, *, seed: int, benchmark: str) -> None:
        self.data_root = Path(data_root)
        self.seed = int(seed)
        self.benchmark = benchmark
        self.protocol = protocol
        self.benchmark_protocol = protocol["benchmarks"][benchmark]
        self.domain_order = tuple(self.benchmark_protocol["domain_order"])
        self.manifest_path = self.data_root / "manifests" / "training" / f"lcrseg_v1_seed{seed}.csv"
        self.split_path = self.data_root / "splits" / f"{benchmark}_seed{seed}.json"
        self._validate_assets()
        self._rows = self._read_rows()

    def _validate_assets(self) -> None:
        frozen = self.protocol["frozen_seed_assets"][self.seed]
        expected_manifest = frozen["training_manifest_sha256"]
        expected_split = frozen[f"{self.benchmark}_split_sha256"]
        for path, expected in ((self.manifest_path, expected_manifest), (self.split_path, expected_split)):
            if not path.is_file():
                raise FileNotFoundError(path)
            observed = sha256_file(path)
            if observed != expected:
                raise RuntimeError(f"frozen asset hash mismatch: {path}: {observed} != {expected}")

    def _read_rows(self) -> list[dict[str, str]]:
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = [{key: value or "" for key, value in row.items()} for row in csv.DictReader(handle)]
        selected = [row for row in rows if row.get("dataset") == self.benchmark]
        if not selected:
            raise ValueError(f"no {self.benchmark} rows in {self.manifest_path}")
        observed_domains = {row["site_or_vendor"] for row in selected}
        unknown = observed_domains.difference(self.domain_order)
        if unknown:
            raise ValueError(f"manifest contains domains outside the frozen protocol: {sorted(unknown)}")
        return selected

    def records(self, *, domain: str, role: str, purpose: str) -> list[ManifestRecord]:
        if domain not in self.domain_order:
            raise ValueError(f"domain is not in the frozen protocol: {domain}")
        if purpose == "train" and role not in TRAIN_ROLES:
            raise RuntimeError(f"training object requested forbidden role: {role}")
        if purpose == "evaluate" and role not in EVAL_ROLES:
            raise RuntimeError(f"evaluator requested non-evaluation role: {role}")
        if purpose not in {"train", "evaluate"}:
            raise ValueError(f"unknown purpose: {purpose}")

        output: list[ManifestRecord] = []
        for row in self._rows:
            if row["site_or_vendor"] != domain or row["primary_20pct_split"] != role:
                continue
            image_rel = _safe_relative(row["image_h5_relpath"], "image_h5_relpath")
            image_path = self.data_root / "h5" / "v1" / image_rel
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if role == "train_unlabeled":
                leaked = [field for field in FORBIDDEN_UNLABELED_FIELDS if row.get(field)]
                if leaked:
                    raise RuntimeError(f"hidden GT metadata leaked for {row['case_id']}: {leaked}")
                label_rel = None
            else:
                label_rel = _safe_relative(row["label_h5_relpath"], "label_h5_relpath")
                label_path = self.data_root / "h5" / "v1" / label_rel
                if not label_path.is_file():
                    raise FileNotFoundError(label_path)
            output.append(
                ManifestRecord(
                    case_id=row["case_id"],
                    patient_id=row.get("patient_id") or row["case_id"],
                    dataset=self.benchmark,
                    domain=domain,
                    role=role,
                    image_h5_relpath=image_rel,
                    label_h5_relpath=label_rel,
                )
            )
        if not output:
            raise ValueError(f"empty manifest view: domain={domain}, role={role}, purpose={purpose}")
        return output

    def assert_current_domain_only(self, records: Iterable[ManifestRecord], current_domain: str) -> None:
        records = tuple(records)
        if not records:
            raise ValueError("empty training record collection")
        observed = {record.domain for record in records}
        roles = {record.role for record in records}
        if observed != {current_domain}:
            raise RuntimeError(f"cross-domain training leakage: expected {current_domain}, got {sorted(observed)}")
        if not roles.issubset(TRAIN_ROLES):
            raise RuntimeError(f"evaluation role entered training: {sorted(roles - TRAIN_ROLES)}")
        for record in records:
            if record.role == "train_unlabeled" and record.label_h5_relpath is not None:
                raise RuntimeError(f"hidden GT entered unlabeled record: {record.case_id}")

    def leakage_audit(self) -> dict:
        result = {"seed": self.seed, "benchmark": self.benchmark, "domains": {}, "hidden_gt_training_usage": "none"}
        for domain in self.domain_order:
            labeled = self.records(domain=domain, role="train_labeled", purpose="train")
            unlabeled = self.records(domain=domain, role="train_unlabeled", purpose="train")
            self.assert_current_domain_only([*labeled, *unlabeled], domain)
            result["domains"][domain] = {
                "train_labeled_records": len(labeled),
                "train_unlabeled_records": len(unlabeled),
                "unlabeled_records_with_label_path": sum(r.label_h5_relpath is not None for r in unlabeled),
                "observed_training_domains": sorted({r.domain for r in [*labeled, *unlabeled]}),
                "status": "PASS",
            }
        result["status"] = "PASS"
        return result
