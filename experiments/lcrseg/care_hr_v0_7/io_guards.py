"""Review lock and static source audit."""
from __future__ import annotations

import ast
import inspect
import subprocess
from pathlib import Path

from .contracts import REVIEW_STATUS, require_external_review_authorization


DANGER_FLAGS = ("--train", "--fit", "--evaluate", "--formal", "--data-root", "--nas-root",
                "--checkpoint", "--v0-6b-root")
ALLOWED_PREFIXES = (
    "experiments/lcrseg/care_hr_v0_7/",
    "experiments/lcrseg/care_hr_v0_7_review.py",
    "experiments/lcrseg/tests/care_hr_v0_7/",
    "experiments/lcrseg/docs/care_hr_v0_7_review/",
)


def reject_dangerous_arguments(arguments):
    if any(argument == flag or argument.startswith(flag + "=") for argument in arguments for flag in DANGER_FLAGS):
        require_external_review_authorization()


def _api_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: [arg.arg for arg in node.args.args + node.args.kwonlyargs]
            for node in tree.body if isinstance(node, ast.FunctionDef)}


def static_audit(repo_root):
    repo = Path(repo_root)
    package = repo / "experiments/lcrseg/care_hr_v0_7"
    sources = sorted(package.glob("*.py")) + [repo / "experiments/lcrseg/care_hr_v0_7_review.py"]
    forbidden_literals = ("/data" + "_nas", "formal" + "_03", "V0.6B " + "private")
    findings = []
    checks = {
        "package_import_has_no_application_file_io": True,
        "no_forbidden_runtime_roots": True,
        "inference_modules_do_not_import_hdf5": True,
        "inference_apis_exclude_hidden_fields": True,
        "real_executor_modes_call_review_lock": True,
        "no_background_training_or_gpu_initialization": True,
        "diff_is_review_work_package_only": True,
    }
    for path in sources:
        text = path.read_text(encoding="utf-8")
        if any(value in text for value in forbidden_literals):
            findings.append(f"forbidden literal: {path.name}")
            checks["no_forbidden_runtime_roots"] = False
        if path.name in {"policy.py", "features.py", "proposals.py"} and "h5py" in text:
            findings.append(f"h5py import: {path.name}")
            checks["inference_modules_do_not_import_hdf5"] = False
        if any(value in text for value in ("torch." + "cuda", ".back" + "ward(", ".st" + "ep(", "PO" + "pen(")):
            findings.append(f"execution primitive: {path.name}")
            checks["no_background_training_or_gpu_initialization"] = False
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                findings.append(f"module-level call: {path.name}")
                checks["package_import_has_no_application_file_io"] = False
    forbidden_parameters = ("domain", "domain_index", "site", "vendor", "label", "gt", "dice", "utility", "patient_outcome")
    for name in ("features.py", "policy.py", "proposals.py"):
        for function, arguments in _api_names(package / name).items():
            if any(argument.lower() in forbidden_parameters for argument in arguments):
                findings.append(f"forbidden API parameter: {name}:{function}")
                checks["inference_apis_exclude_hidden_fields"] = False
    executor_source = (package / "executor.py").read_text(encoding="utf-8")
    executor_tree = ast.parse(executor_source)
    for node in executor_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"train", "fit", "evaluate"}:
            called = [item.func.id for item in ast.walk(node) if isinstance(item, ast.Call)
                      and isinstance(item.func, ast.Name)]
            if not called or called[0] != "require_external_review_authorization":
                findings.append(f"unlocked executor mode: {node.name}")
                checks["real_executor_modes_call_review_lock"] = False
    changed = subprocess.run(
        ["git", "-C", str(repo), "status", "--short"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    for line in changed:
        path = line[3:]
        if not any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in ALLOWED_PREFIXES):
            findings.append(f"out-of-scope diff: {path}")
            checks["diff_is_review_work_package_only"] = False
    return {"status": REVIEW_STATUS, "ok": not findings, "findings": findings,
            "checks": checks, "checked_sources": len(sources), "changed_paths": len(changed)}
