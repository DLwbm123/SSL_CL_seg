"""Build a private, read-only prostate catalog for a new supervised protocol."""
import argparse
import csv
import json
import os
from pathlib import Path


SITES = ("RUNMC", "BMC", "I2CVB")


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def safe_file(root, value):
    rel = Path(value)
    if not value or rel.is_absolute() or ".." in rel.parts:
        raise ValueError("invalid HDF5 relative path")
    path = root / "h5" / "v1" / rel
    if not path.is_file():
        raise FileNotFoundError(path)
    return value


def build(data_root, scope, approved_all_train=False):
    data_root = Path(data_root)
    if scope not in ("visible", "all_train"):
        raise ValueError("scope must be visible or all_train")
    if scope == "all_train" and not approved_all_train:
        raise PermissionError("all_train requires explicit authorization")
    split_rows = json.loads((data_root / "splits" / "prostate_seed0.json").read_text())["records"]
    split = {row["case_id"]: row for row in split_rows if row["site_or_vendor"] in SITES}
    if len(split) != sum(row["site_or_vendor"] in SITES for row in split_rows):
        raise ValueError("duplicate case in frozen split")
    training = rows(data_root / "manifests" / "training" / "lcrseg_v1_seed0.csv")
    training = {row["case_id"]: row for row in training if row["dataset"] == "prostate" and row["site_or_vendor"] in SITES}
    if set(training) != set(split):
        raise ValueError("training manifest and split case sets differ")
    if any(training[k]["primary_20pct_split"] != split[k]["primary_20pct_split"] for k in split):
        raise ValueError("frozen role mismatch")
    diagnostic = {}
    if scope == "all_train":
        diagnostic = {row["case_id"]: row for row in rows(data_root / "manifests" / "diagnostics" / "lcrseg_v1_seed0.csv")
                      if row["dataset"] == "prostate" and row["site_or_vendor"] in SITES}
        if set(diagnostic) != set(split):
            raise ValueError("diagnostic and frozen split case sets differ")
    result = {site: {"train": [], "val": []} for site in SITES}
    for case_id, row in sorted(split.items()):
        site = row["site_or_vendor"]
        role = row["primary_20pct_split"]
        if row["split"] == "train":
            assert role in ("train_labeled", "train_unlabeled")
            if scope == "visible" and role != "train_labeled":
                continue
            source = diagnostic[case_id] if scope == "all_train" else training[case_id]
            if source["split"] != "train" or source["patient_id"] != row["patient_id"]:
                raise ValueError("training identity mismatch")
            bucket = "train"
        elif role == "val" and row["split"] == "val":
            source = training[case_id]
            bucket = "val"
        else:
            continue  # Test patients remain sealed.
        entry = {"case_id": case_id, "patient_id": row["patient_id"], "site": site,
                 "image": safe_file(data_root, source["image_h5_relpath"]),
                 "label": safe_file(data_root, source["label_h5_relpath"])}
        result[site][bucket].append(entry)
    ids = [r["patient_id"] for site in SITES for bucket in ("train", "val") for r in result[site][bucket]]
    if len(ids) != len(set(ids)):
        raise ValueError("patient overlap across roles or sites")
    expected = {"visible": {"RUNMC": (4, 4), "BMC": (4, 4), "I2CVB": (2, 3)},
                "all_train": {"RUNMC": (18, 4), "BMC": (18, 4), "I2CVB": (11, 3)}}[scope]
    counts = {site: (len(result[site]["train"]), len(result[site]["val"])) for site in SITES}
    if counts != expected:
        raise ValueError(f"unexpected patient counts: {counts}")
    return {"scope": scope, "seed": 0, "sites": list(SITES), "cases": result,
            "counts": {site: {"train": counts[site][0], "val": counts[site][1]} for site in SITES},
            "patient_overlap": False, "test_case_access": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scope", choices=("visible", "all_train"), required=True)
    parser.add_argument("--approved-all-train", action="store_true")
    args = parser.parse_args()
    catalog = build(args.data_root, args.scope, args.approved_all_train)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        json.dump(catalog, handle, indent=2)
    print(json.dumps({"scope": catalog["scope"], "counts": catalog["counts"], "test_case_access": 0}))


if __name__ == "__main__":
    main()
