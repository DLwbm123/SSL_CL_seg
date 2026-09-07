"""Small audited persistence and column projection helpers; no training imports."""
import hashlib
import json
from pathlib import Path

BLIND_COLUMNS = ("case_id", "patient_id", "primary_20pct_split", "image_h5_relpath", "image_sha256")
BLIND_KEYS = {"alpha", "case_id", "fold", "image_h5_relpath", "image_sha256", "patient_id", "row_index", "seed"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_path(root, relative):
    root = Path(root).resolve(); path = (root / relative).resolve()
    if not path.is_relative_to(root): raise ValueError("path escapes registered root")
    return path


def verify_files(root, files):
    for name, expected in files.items():
        if digest(safe_path(root, name)) != expected: raise ValueError("sealed file hash mismatch: " + name)


def project_training(path, columns=BLIND_COLUMNS, selected_ids=None):
    # C-parser usecols projects columns before creating Python rows. GT files are never opened here.
    import pandas as pd
    skiprows = None
    if selected_ids is not None:
        ids = pd.read_csv(path, usecols=['case_id'], dtype=str, keep_default_na=False)['case_id']
        keep = {i+1 for i, value in enumerate(ids) if value in selected_ids}
        skiprows = lambda line: line != 0 and line not in keep
    return pd.read_csv(path, usecols=list(columns), skiprows=skiprows, dtype=str, keep_default_na=False).to_dict("records")


def validate_population(rows, projected, expected_rows=198, expected_patients=177, per_seed=66):
    if len(rows) != expected_rows or any(set(r) != BLIND_KEYS for r in rows): raise ValueError("blind schema/population mismatch")
    if [r["row_index"] for r in rows] != list(range(expected_rows)): raise ValueError("row order mismatch")
    if len({(r["seed"], r["case_id"]) for r in rows}) != len(rows): raise ValueError("duplicate seed-case")
    if len({r["patient_id"] for r in rows}) != expected_patients: raise ValueError("patient count mismatch")
    mapping = {}
    for seed in (0, 1, 2):
        subset = [r for r in rows if r["seed"] == seed]
        if len(subset) != per_seed: raise ValueError("seed count mismatch")
        indexed = {}
        for row in projected[seed]: indexed.setdefault(row["case_id"], []).append(row)
        for row in subset:
            candidates = indexed.get(row["case_id"], [])
            if len(candidates) != 1: raise ValueError("manifest ambiguous/missing case")
            full = candidates[0]
            if full["primary_20pct_split"] != "train_labeled": raise ValueError("own-seed GT not authorized")
            if any(full[k] != row[k] for k in ("patient_id", "image_h5_relpath", "image_sha256")): raise ValueError("lineage mismatch")
            identity = (row["patient_id"], row["image_h5_relpath"], row["image_sha256"])
            if row["case_id"] in mapping and mapping[row["case_id"]] != identity: raise ValueError("cross-seed lineage mismatch")
            mapping[row["case_id"]] = identity
    return {"rows":len(rows), "patients":expected_patients, "unique_cases":len(mapping), "own_seed_roles":"train_labeled", "true_domain_reads":0, "GT_reads":0}
