"""M1-only canonical reader with explicit training/evaluation roles."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

MANIFEST_SHA = "0622f54f42f05d6ef87f9dc89ee9435cf8da03c6c30cd970db6ea167e00dd8a3"
SPLIT_SHA = "f250d97aea1f36f21899f5dd40bb6c9a819e7755aee458c8ee27506496b46a88"
M1_DOMAINS = ("RIM_ONE_r3", "Drishti_GS")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def role_allowed(domain: str, role: str, purpose: str, current_domain: str) -> None:
    if domain not in M1_DOMAINS or current_domain not in M1_DOMAINS:
        raise PermissionError("domain outside M1")
    if purpose == "train" and (role != "train_labeled" or domain != current_domain):
        raise PermissionError("training only accesses current-domain L")
    if purpose == "evaluate" and role != "val":
        raise PermissionError("evaluation only accesses validation")
    if purpose not in ("train", "evaluate"):
        raise PermissionError("unknown data purpose")


class CanonicalM1(Dataset):
    def __init__(self, data_root: Path, migration_manifest: Path, *, domain: str, role: str,
                 purpose: str, current_domain: str) -> None:
        role_allowed(domain, role, purpose, current_domain)
        self.root = data_root.resolve(strict=True)
        package = json.loads(migration_manifest.read_text())
        if package.get("study_id") != "SSLCL_QPROMPT_RL_V1" or package.get("phase") != "M1":
            raise PermissionError("wrong migration manifest")
        if package.get("manifest_sha256") != MANIFEST_SHA or package.get("split_sha256") != SPLIT_SHA:
            raise ValueError("canonical metadata binding differs")
        allowed = {x["source_relative"]: x for x in package["files"] if x["root_id"] == "data"}
        for name, expected in (("manifests/training/lcrseg_v1_seed0.csv", MANIFEST_SHA),
                               ("splits/fundus_seed0.json", SPLIT_SHA)):
            if digest(self._path(name)) != expected:
                raise ValueError("frozen metadata bytes changed")
        split = json.loads(self._path("splits/fundus_seed0.json").read_text())
        if split.get("seed") != 0:
            raise ValueError("wrong split")
        split_rows = {r["case_id"]: r for r in split["records"]}
        with self._path("manifests/training/lcrseg_v1_seed0.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.rows = []
        self.allowed = allowed
        self.checked: set[str] = set()
        for row in rows:
            if row["dataset"] != "fundus" or row["site_or_vendor"] != domain or row["primary_20pct_split"] != role:
                continue
            if row["split_seed"] != "0" or any(row[x] != split_rows[row["case_id"]][x]
                                                     for x in ("patient_id", "site_or_vendor", "primary_20pct_split")):
                raise ValueError("manifest/split mismatch")
            item = {}
            for kind in ("image", "label"):
                relative = row[kind + "_h5_relpath"]
                if relative not in allowed or allowed[relative]["role"] != role or allowed[relative]["domain"] != domain:
                    raise PermissionError("payload outside M1 allowlist")
                if allowed[relative]["sha256"] != row[kind + "_sha256"]:
                    raise ValueError("payload hash binding differs")
                item[kind] = relative
            self.rows.append(item)
        if not self.rows:
            raise ValueError("empty role")

    def _path(self, relative: str) -> Path:
        path = self.root / relative
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(self.root):
            raise PermissionError("payload path escaped root")
        return path

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        import h5py  # Delayed: math/CPU synthetic tests do not need HDF5.

        item = {}
        for kind, relative in self.rows[index].items():
            path = self._path(relative)
            if relative not in self.checked:
                if digest(path) != self.allowed[relative]["sha256"]:
                    raise ValueError("canonical payload changed")
                self.checked.add(relative)
            with h5py.File(path, "r") as f:
                if set(f.keys()) != {kind}:
                    raise ValueError("mixed or unexpected HDF5 payload")
                array = f[kind][...]
            if kind == "image":
                if array.shape != (3, 384, 384) or str(array.dtype) != "uint8":
                    raise ValueError("invalid canonical image")
                item[kind] = torch.as_tensor(array.copy()).float() / 255
            else:
                label = torch.as_tensor(array.copy()).long()
                if label.shape != (384, 384) or not torch.isin(label, torch.tensor([0, 1, 2, 255])).all():
                    raise ValueError("invalid canonical label")
                item[kind] = label
        return item
