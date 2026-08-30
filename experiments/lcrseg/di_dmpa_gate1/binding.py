"""Fail-closed identities, role isolation and append-only artifact helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

PREREG = "cfb62554f1e6a2a36850547485b1857dc9a28a20"
AUTHORIZATION = "25ec97c988af290a4fb7a637c4b7cdfe462deb87"
BRANCH = "codex/di-dmpa-gate1-diagnostics"
REMOTE = "https://github.com/DLwbm123/SSL_CL_seg.git"
FILE_HASHES = {
    "md": "32acdc5c24bcc5763daa6cb3650fea91f46da7ae3845b1fd0615c781619fbf0a",
    "json": "6f50bd9df404d987aa70e2035a5c3f3853aa59ce49d21ffface34172cf754cbf",
}
PANELS = ("B0-EMA", "B0-student", "C0-EMA", "C0-student")
ROLES = ("train_labeled", "val")


class ProtocolError(RuntimeError):
    status = "BLOCKED_PROTOCOL_OR_LEAKAGE"


class NumericalError(RuntimeError):
    status = "BLOCKED_NUMERICAL_FAILURE"


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def H(parts):
    return hashlib.sha256(compact(parts).encode("utf-8")).hexdigest()


def S(parts):
    return int(H(parts)[:8], 16) & 0x7fffffff


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_hash(path, expected):
    require(Path(path).is_file(), f"missing input: {path}")
    observed = sha256(path)
    require(observed == expected, f"SHA mismatch: {path}: {observed} != {expected}")
    return observed


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(compact(value) + "\n")
    return sha256(path)


def write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def verify_ancestor(root, ancestor, head):
    result = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, head], capture_output=True)
    require(result.returncode == 0, f"{ancestor} is not an ancestor of {head}")


def verify_registration(lcrseg_root, code_commit, *, verify_remote=True):
    root = Path(lcrseg_root)
    gitroot = root.parents[1]
    docs = root / "docs/di_dmpa_jascl"
    require(git(gitroot, "rev-parse", "HEAD") == code_commit, "execution checkout/code commit mismatch")
    require(not git(gitroot, "status", "--porcelain"), "execution source is dirty or has untracked files")
    for ancestor in (PREREG, AUTHORIZATION):
        verify_ancestor(gitroot, ancestor, code_commit)
    for suffix, digest in FILE_HASHES.items():
        path = docs / f"DI_DMPA_GATE1_PREREGISTRATION.{suffix}"
        check_hash(path, digest)
        blob = subprocess.check_output(["git", "-C", str(gitroot), "show", f"{PREREG}:{path.relative_to(gitroot)}"])
        require(hashlib.sha256(blob).hexdigest() == digest, "registration Git blob mismatch")
    for suffix in ("json", "md"):
        path = docs / f"GATE1A_EXECUTION_AUTHORIZATION.{suffix}"
        blob = subprocess.check_output(["git", "-C", str(gitroot), "show", f"{AUTHORIZATION}:{path.relative_to(gitroot)}"])
        require(hashlib.sha256(blob).hexdigest() == sha256(path), "authorization bytes changed")
    authorization = read_json(docs / "GATE1A_EXECUTION_AUTHORIZATION.json")
    require(authorization["authorization_scope"] == "GATE1A_ONLY", "wrong authorization scope")
    require(authorization["preregistration_commit"] == PREREG, "wrong authorization binding")
    remote_sha = None
    if verify_remote:
        response = subprocess.check_output(["git", "ls-remote", REMOTE, f"refs/heads/{BRANCH}"], text=True)
        remote_sha = response.split()[0] if response.split() else None
        require(remote_sha == code_commit, "remote branch is not exact diagnostic code commit")
    prereg = read_json(docs / "DI_DMPA_GATE1_PREREGISTRATION.json")
    require(all(v is False for v in prereg["method_flags"].values()), "method switch was enabled")
    require(prereg["panels"]["primary_admission_panel"] == "B0-EMA", "primary panel changed")
    return prereg, {"preregistration_git_commit": PREREG, "preregistration_remote_verified_commit": PREREG,
                    "authorization_git_commit": AUTHORIZATION, "diagnostic_code_git_commit": code_commit,
                    "remote_code_sha": remote_sha, "registration_files_verified": True,
                    "authorization_files_verified": True, "ancestor_checks": [PREREG, AUTHORIZATION]}


def safe_asset(data_root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, f"unsafe asset path: {relative}")
    return Path(data_root) / "h5/v1" / relative


def gate1a_records(data_root, prereg, seed, domain, role):
    require(role in ROLES, f"Gate1A forbidden role: {role}")
    plan = next(p for p in prereg["benchmark"]["case_plans"] if p["seed"] == seed and p["domain"] == domain)
    manifest = Path(data_root) / "manifests/training" / f"lcrseg_v1_seed{seed}.csv"
    asset = next(p for p in prereg["benchmark"]["manifest_assets"] if p["seed"] == seed)
    check_hash(manifest, asset["sha256"])
    with manifest.open(newline="") as handle:
        # No other role is constructed or resolved into image/GT access objects.
        rows = [r for r in csv.DictReader(handle) if r["dataset"] == "fundus"
                and r["site_or_vendor"] == domain and r["primary_20pct_split"] == role]
    rows.sort(key=lambda r: r["case_id"])
    require([r["case_id"] for r in rows] == plan["roles"][role], "case/role plan mismatch")
    require(not set(plan["roles"]["train_labeled"]) & set(plan["roles"]["val"]), "train/val overlap")
    for row in rows:
        require(row["label_h5_relpath"] and row["label_sha256"], "required visible label unavailable")
        safe_asset(data_root, row["image_h5_relpath"])
        safe_asset(data_root, row["label_h5_relpath"])
    return rows


def audit_inputs(lcrseg_root, data_root, prereg):
    import yaml
    root = Path(lcrseg_root)
    gitroot = root.parents[1]
    frozen = prereg["immutable_baseline"]
    checks = {}
    checks["baseline_freeze"] = check_hash(gitroot / frozen["freeze_path"], frozen["freeze_sha256"])
    protocol = prereg["benchmark"]["domain_order_source"]
    checks["domain_protocol"] = check_hash(gitroot / protocol["path"], protocol["sha256"])
    require(git(root/"third_party/JASCL_REFERENCE","rev-parse","HEAD")==frozen["upstream_jascl_commit"],"official classifier source commit changed")
    for name, cfg in frozen["configs"].items():
        path = gitroot / cfg["path"]
        check_hash(path, cfg["file_sha256"])
        parsed = yaml.safe_load(path.read_text())
        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        require(hashlib.sha256(canonical.encode()).hexdigest() == cfg["resolved_config_sha256"], "canonical config mismatch")
        checks[name] = {"file": cfg["file_sha256"], "canonical": cfg["resolved_config_sha256"]}
    for asset in prereg["benchmark"]["manifest_assets"]:
        seed = asset["seed"]
        check_hash(Path(data_root)/f"manifests/training/lcrseg_v1_seed{seed}.csv", asset["sha256"])
        check_hash(Path(data_root)/f"splits/fundus_seed{seed}.json", asset["fundus_split_sha256"])
    checks["checkpoints"] = {}
    for checkpoint in frozen["checkpoint_inputs"]:
        checks["checkpoints"][checkpoint["checkpoint_id"]] = check_hash(checkpoint["path"], checkpoint["sha256"])
    require(len(checks["checkpoints"]) == 18, "18 checkpoints required")
    checks.update(status="PASS", test_role_constructions=0, train_unlabeled_constructions=0,
                  hidden_gt_training_usage="none", test_gt_usage="none", checkpoint_tensor_schema_check="required again at read-only extraction")
    return checks


def run_metadata(prereg, receipt, sampling_sha, *, panel_id):
    require(len(sampling_sha) == 64, "actual locked sampling hash required before a worker starts")
    require(panel_id in (*PANELS, "ALL_FOUR_SEPARATE"), "unknown panel")
    return {**receipt, "preregistration_id": prereg["preregistration_id"],
            "preregistration_version": prereg["preregistration_version"],
            "preregistration_json_sha256": FILE_HASHES["json"], "preregistration_md_sha256": FILE_HASHES["md"],
            "baseline_freeze_sha256": prereg["immutable_baseline"]["freeze_sha256"],
            "input_checkpoint_sha256": {c["checkpoint_id"]: c["sha256"] for c in prereg["immutable_baseline"]["checkpoint_inputs"]},
            "manifest_sha256": {str(a["seed"]): a["sha256"] for a in prereg["benchmark"]["manifest_assets"]},
            "sampling_plan_sha256": sampling_sha, "panel_id": panel_id, "primary_admission_panel": "B0-EMA",
            "primary_feature_source": "ema_teacher", "feature_source_selection_performed": False,
            "model_optimizer_steps": 0, "transport_optimizer_steps": 0, "test_gt_usage": "none",
            "hidden_gt_training_usage": "none", "method_registered": False, "di_dmpa_training_launched": False}
