#!/usr/bin/env python3
"""One bounded original-runner replay; never patches a frozen checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
DOCS = Path("experiments/lcrseg/docs/di_dmpa_jascl")
PREREG = "05946f05484ab3bf612daf20a21e4fee541668ef"
OUTPUT_ROOT = Path("/root/LCRSeg/runs/gate1c_legacy_pas_recovery") / PREREG / "attempt1"
PLAN_NAME = "GATE1C_LEGACY_PAS_RECOVERY_PREREGISTRATION"
PLAN_HASHES = {
    "json": "152bf93deb67af18b1ac070d6b31909cc1043d4ec8984231b50701ca1a170013",
    "md": "b32cc208450d15dcf8500ab570c15a43aa19f35218455374ba4203c4408808c0",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=30).strip()


def plan():
    for suffix, expected in PLAN_HASHES.items():
        require(sha(ROOT / DOCS / f"{PLAN_NAME}.{suffix}") == expected, "recovery plan changed")
    return json.loads((ROOT / DOCS / f"{PLAN_NAME}.json").read_text())


def trace_errors(actual, expected, p):
    require(all(actual[k] == expected[k] for k in p["trace_comparison"]["exact_fields"]),
            "original trace identity mismatch")
    errors = {}
    for key in p["trace_comparison"]["numeric_fields"]:
        a, b = float(actual[key]), float(expected[key])
        require(math.isfinite(a) and math.isfinite(b) and math.isclose(
            a, b, rel_tol=p["trace_comparison"]["rtol"], abs_tol=p["trace_comparison"]["atol"]),
            f"original trace mismatch: step={actual['global_step']} field={key} actual={a} expected={b}")
        errors[key] = abs(a - b)
    return errors


def check_roles(records, require_label, p):
    require(bool(records) and require_label, "recovery permits no unlabeled batch")
    roles = {r.role for r in records}
    require(roles in ({"train_labeled"}, {"val"}), "forbidden or mixed label role")
    require({r.domain for r in records} == {p["domain"]}, "cross-domain recovery access")
    require(all(r.label_h5_relpath is not None for r in records), "missing permitted label path")


def check_capture(state, steps, p):
    require(state["stage_index"] == p["stage_index"] and state["epoch"] == p["capture_epoch"]
            and state["global_step"] == p["capture_global_step"]
            and steps == p["supervised_steps_per_replica"], "wrong recovery capture boundary")


def compare_state(a, b, path="state"):
    """Byte-exact tensor check; the existing resume comparator allows tolerances."""
    import numpy as np
    import torch
    require(type(a) is type(b), f"replica type mismatch: {path}")
    if isinstance(a, torch.Tensor):
        require(a.dtype == b.dtype and a.shape == b.shape and torch.equal(
            a.reshape(-1).contiguous().view(torch.uint8), b.reshape(-1).contiguous().view(torch.uint8)),
            f"replica tensor mismatch: {path}")
        require(not a.is_floating_point() or torch.isfinite(a).all().item(), f"nonfinite state: {path}")
    elif isinstance(a, np.ndarray):
        require(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
                f"replica array mismatch: {path}")
    elif isinstance(a, dict):
        require(a.keys() == b.keys(), f"replica keys mismatch: {path}")
        for key in a:
            compare_state(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        require(len(a) == len(b), f"replica length mismatch: {path}")
        for index, (left, right) in enumerate(zip(a, b)):
            compare_state(left, right, f"{path}[{index}]")
    else:
        require(a == b, f"replica scalar mismatch: {path}")


def run(args):
    p = plan()
    require(git(ROOT, "rev-parse", "HEAD") == args.code_commit, "helper commit mismatch")
    require(not git(ROOT, "status", "--porcelain"), "helper checkout is dirty")
    git(ROOT, "merge-base", "--is-ancestor", PREREG, args.code_commit)
    remote = git(ROOT, "ls-remote", "https://github.com/DLwbm123/SSL_CL_seg.git",
                 "refs/heads/codex/sslcl-long-running-reproduction")
    require(remote.split()[0] == args.code_commit, "exact helper code is not published")
    source = Path(args.source_root).resolve()
    require(git(source, "rev-parse", "HEAD") == p["source_commit"], "original source mismatch")
    require(not git(source, "status", "--porcelain"), "original source checkout is dirty")
    require(sys.executable == p["runtime"]["python"], "use the existing frozen Python environment")
    replica = next(r for r in p["replicas"] if r["id"] == args.replica)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(replica["physical_gpu"]), "CUDA mapping mismatch")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        require(os.environ.get(name) == "1", f"original one-thread runtime required: {name}")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUBLAS setting mismatch")

    output = Path(args.output_root).resolve() / args.replica
    require(output.parent == OUTPUT_ROOT, "only the single preregistered recovery attempt is allowed")
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source / "experiments/lcrseg"))
    import torch
    import di_dmpa_jascl.runner as original
    from di_dmpa_jascl.config import load_yaml
    from di_dmpa_jascl.metrics import write_json
    require(Path(original.__file__).resolve().is_relative_to(source), "wrong runner imported")
    require(torch.__version__ == p["runtime"]["torch"] and torch.cuda.device_count() == 1,
            "runtime or visible CUDA device count mismatch")
    require(torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 3090", "unexpected GPU model")
    audit_path = DOCS / "GATE1C_V2_LEGACY_PROTOTYPE_AVAILABILITY_AUDIT.json"
    frozen_text = git(ROOT, "show", f"{PREREG}:{audit_path}")
    require((ROOT / audit_path).read_text().strip() == frozen_text, "checkpoint availability audit changed")
    frozen = json.loads(frozen_text)
    original_hashes = {r["path"]: r["sha256"] for r in frozen["checkpoints"]}
    original_hashes[p["original_trace_path"]] = p["original_trace_sha256"]
    require(all(sha(path) == expected for path, expected in original_hashes.items()), "frozen input changed")
    payload = torch.load(p["checkpoint_path"], map_location="cpu", weights_only=False)
    require(payload["prototypes"] is None and len(payload["rng_state"]["torch_cuda"]) == 1,
            "unexpected missing-bank or saved CUDA state")
    require(sha(p["checkpoint_path"]) == p["checkpoint_sha256"], "resume checkpoint changed")
    rows = [json.loads(line) for line in Path(p["original_trace_path"]).read_text().splitlines()]
    reference = [r for r in rows if r["stage_index"] == p["stage_index"] and r["phase"] == "supervised"
                 and p["initial_epoch"] <= r["epoch"] <= p["capture_epoch"]]
    require([r["global_step"] for r in reference] == list(range(p["initial_global_step"] + 1,
            p["capture_global_step"] + 1)), "original trace coverage mismatch")
    config = load_yaml(source / "experiments/lcrseg" / p["config_path"])
    protocol = load_yaml(source / "experiments/lcrseg" / config["data"]["protocol"])
    runner = original.Gate0RepairedRunner(repo_root=source / "experiments/lcrseg", config=config,
        protocol=protocol, seed=p["seed"], output_dir=output, device="cuda")
    require(runner.config_hash == p["config_hash"], "original config changed")
    require(len(runner.adapter.records(domain=p["domain"], role="train_labeled", purpose="train"))
            == p["labeled_cases"], "original case count changed")
    metadata = json.loads((output / "run_metadata.json").read_text())
    metadata.update(recovery_protocol_id=p["protocol_id"], recovery_preregistration_commit=PREREG,
        recovery_preregistration_sha256=PLAN_HASHES, recovery_helper_commit=args.code_commit,
        recovery_helper_sha256=sha(__file__), original_checkpoint_sha256=p["checkpoint_sha256"],
        original_trace_sha256=p["original_trace_sha256"], replica=replica, command=sys.argv,
        purpose="candidate reconstruction only; not Gate1C admission", historical_bank_hash_available=False)
    write_json(output / "run_metadata.json", metadata)
    seen, matched_rows, case_hash, role_counts = [], 0, hashlib.sha256(), {}
    maximum_errors = {k: 0.0 for k in p["trace_comparison"]["numeric_fields"]}
    append_log, collate, evaluate_dataset = runner._append_log, original.collate, runner._evaluation_dataset

    def checked_log(row):
        nonlocal matched_rows
        append_log(row)
        seen.append(dict(row))  # The optimizer already stepped before this original logging hook.
        require(len(seen) <= len(reference), "recovery budget exceeded")
        errors = trace_errors(row, reference[len(seen) - 1], p)
        for key, error in errors.items():
            maximum_errors[key] = max(maximum_errors[key], error)
        matched_rows += 1

    def checked_collate(dataset, indices, *, require_label):
        indices = list(indices)
        records = [dataset.samples[i].record for i in indices]
        check_roles(records, require_label, p)  # Before any image/label payload access.
        event = {"epoch": runner.stage_state["epoch"], "phase": runner.sampler_state["phase"],
                 "role": records[0].role, "cases": [r.case_id for r in records]}
        line = json.dumps(event, sort_keys=True) + "\n"
        case_hash.update(line.encode())
        with (output / "case_order.jsonl").open("a") as handle:
            handle.write(line)
        role_counts[records[0].role] = role_counts.get(records[0].role, 0) + len(records)
        return collate(dataset, indices, require_label=require_label)

    def validation_only(domain, role):
        require(domain == p["domain"] and role == "val", "test or cross-domain evaluation forbidden")
        return evaluate_dataset(domain, role)

    def capture_before_unsupervised(*unused):
        check_capture(runner.stage_state, len(seen), p)
        bank = runner.prototypes
        require(isinstance(bank, torch.Tensor) and bank.dtype == torch.float32 and bank.shape == (3, 16)
                and torch.isfinite(bank).all().item() and (bank.norm(dim=1) > 0).all().item(), "invalid candidate bank")
        runner._save_last()
        torch.save(bank.detach().cpu().clone(), output / "legacy_pas_candidate.pt")
        return True

    runner._append_log = checked_log
    original.collate = checked_collate
    runner._evaluation_dataset = validation_only
    runner._unsupervised_phase = capture_before_unsupervised
    started = time.monotonic()
    receipt = {"status": "RUNNING", "metadata": metadata, "baseline_optimizer_steps": 0,
        "unlabeled_optimizer_steps": 0, "method_optimizer_steps": 0, "transport_optimizer_steps": 0,
        "hidden_gt_training_usage": "none", "test_gt_reads": 0, "frozen_gate1c_v2_completed": False}
    write_json(output / "RECOVERY_STATUS.json", receipt)

    def timeout_handler(*unused):
        raise TimeoutError("preregistered recovery time budget exceeded")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(p["maximum_seconds_per_replica"])
    try:
        result = runner.run(resume_path=p["checkpoint_path"])
        require(result["status"] == "INTERRUPTED" and (output / "legacy_pas_candidate.pt").is_file(),
                "original runner did not stop at the capture boundary")
        check_capture(runner.stage_state, len(seen), p)
        receipt.update(status="CAPTURED_AWAITING_REPLICA_COMPARISON", candidate_sha256=sha(output / "legacy_pas_candidate.pt"),
                       capture_state_sha256=sha(output / "last.pt"))
    except Exception as exc:
        receipt.update(status=p["failure_status"], error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        unchanged = all(sha(path) == expected for path, expected in original_hashes.items())
        receipt.update(baseline_optimizer_steps=len(seen), elapsed_seconds=time.monotonic() - started,
            original_trace_rows_compared=len(seen), original_trace_rows_matched=matched_rows,
            maximum_trace_absolute_errors=maximum_errors,
            case_order_sha256=case_hash.hexdigest(), label_reads_by_role=role_counts,
            frozen_inputs_unchanged=unchanged, source_clean=not git(source, "status", "--porcelain"))
        if not unchanged or not receipt["source_clean"]:
            receipt.update(status=p["failure_status"], integrity_error="original input/source changed")
        write_json(output / "RECOVERY_STATUS.json", receipt)
    require(receipt["status"] == "CAPTURED_AWAITING_REPLICA_COMPARISON", "recovery integrity failed")
    print(json.dumps(receipt, indent=2))


def compare(args):
    import torch
    p, root = plan(), Path(args.output_root).resolve()
    require(root == OUTPUT_ROOT, "only the preregistered recovery root may be compared")
    reports = [json.loads((root / r["id"] / "RECOVERY_STATUS.json").read_text()) for r in p["replicas"]]
    result = {"status": p["failure_status"], "preregistration_commit": PREREG, "replicas": reports,
              "historical_bank_hash_verified": False, "frozen_gate1c_v2_completed": False}
    destination = root / "RECOVERY_COMPARISON.json"
    require(not destination.exists(), "refusing to overwrite a recovery comparison")
    try:
        require(all(r["status"] == "CAPTURED_AWAITING_REPLICA_COMPARISON" and r["baseline_optimizer_steps"] == 200
                    and r["original_trace_rows_matched"] == 200
                    and r["frozen_inputs_unchanged"] and r["source_clean"] for r in reports), "one replica did not pass")
        require(reports[0]["metadata"]["recovery_helper_commit"] == reports[1]["metadata"]["recovery_helper_commit"]
                and all(r["metadata"]["recovery_preregistration_commit"] == PREREG for r in reports),
                "replica provenance mismatch")
        require(reports[0]["case_order_sha256"] == reports[1]["case_order_sha256"], "replica case order mismatch")
        for name in ("legacy_pas_candidate.pt", "last.pt"):
            key = "candidate_sha256" if name == "legacy_pas_candidate.pt" else "capture_state_sha256"
            require(all(sha(root / replica["id"] / name) == report[key]
                        for replica, report in zip(p["replicas"], reports)), "captured artifact changed")
            left, right = [torch.load(root / r["id"] / name, map_location="cpu", weights_only=False) for r in p["replicas"]]
            compare_state(left, right, name)
        result.update(status=p["success_status"], candidate_and_full_capture_state_bitwise_equal=True)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    with destination.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2))
    require(result["status"] == p["success_status"], "reconstruction not verified; no candidate may be consumed")


def self_test():
    p = plan()
    reference = {k: 1 for k in p["trace_comparison"]["exact_fields"]}
    reference.update({k: 0.1 for k in p["trace_comparison"]["numeric_fields"]})
    require(all(v == 0 for v in trace_errors(reference, reference, p).values()), "trace check broken")
    trace_errors(dict(reference, loss_total=0.10000005), reference, p)
    record = SimpleNamespace(domain=p["domain"], role="train_labeled", label_h5_relpath="permitted.h5")
    check_roles([record], True, p)
    state = {"stage_index": 1, "epoch": 25, "global_step": 3408}
    check_capture(state, 200, p)
    failures = [lambda: trace_errors(dict(reference, epoch=2), reference, p),
        lambda: trace_errors(dict(reference, loss_total=0.2), reference, p),
        lambda: trace_errors(dict(reference, loss_total=float("nan")), reference, p),
        lambda: check_capture(state, 199, p), lambda: check_capture(dict(state, epoch=24), 200, p),
        lambda: check_roles([record], False, p),
        lambda: check_roles([SimpleNamespace(domain=p["domain"], role="test", label_h5_relpath="test.h5")], True, p),
        lambda: check_roles([SimpleNamespace(domain="REFUGE", role="train_labeled", label_h5_relpath="x")], True, p)]
    for check in failures:
        try:
            check()
        except RuntimeError:
            continue
        raise AssertionError("negative guard did not reject")
    import torch
    a = {"tensor": torch.ones(2), "nested": [1, {"value": 2.0}]}
    compare_state(a, a)
    try:
        compare_state(a, {"tensor": torch.zeros(2), "nested": a["nested"]})
    except RuntimeError:
        pass
    else:
        raise AssertionError("replica mismatch did not reject")
    try:
        compare_state(torch.tensor([0.0]), torch.tensor([-0.0]))
    except RuntimeError:
        pass
    else:
        raise AssertionError("byte-exact signed-zero guard did not reject")
    print("PASS: trace tolerance, nonfinite/identity rejection, label-role/domain guards, exact capture budget, recursive replica equality")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--source-root", required=True)
    execute.add_argument("--output-root", required=True)
    execute.add_argument("--replica", choices=("replica_gpu0", "replica_gpu1"), required=True)
    execute.add_argument("--code-commit", required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--output-root", required=True)
    commands.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "compare":
        compare(args)
    else:
        self_test()
