"""Strict B0-only admission and independent stage-checkpoint/matrix verification."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess

import torch

from di_dmpa_jascl.checkpoint import capture_rng_state, load_checkpoint
from di_dmpa_jascl.config import load_yaml, resolved_config_hash
from di_dmpa_jascl.provenance import git_revision
from lcrseg.acceptance import verify_checksums
from .baseline import RegeneratedB0Runner, state_digest, verify_payload
from .durable import now, read, sha256, verify, write_new

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def admission(args):
    require(sha256(args.authorization) == args.authorization_sha256, "authorization hash mismatch")
    auth = read(args.authorization)
    require(auth["status"] == "AUTHORIZED_B0_REGENERATION_ONLY", "B0 authorization missing")
    prereg_path = args.authorization.parent / "BASELINE_V3_PREREGISTRATION.json"
    require(sha256(prereg_path) == auth["preregistration_sha256"], "preregistration hash mismatch")
    prereg = read(prereg_path)
    require(args.seed in (0, 1, 2), "unregistered seed")
    require(git_revision(ROOT) == auth["code_commit"], "wrong exact code commit")
    require(not subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True).strip(),
            "tracked code is dirty")
    for relative, digest in prereg["frozen_source_sha256"].items():
        require(sha256(ROOT / relative) == digest, "frozen source changed: " + relative)
    dest = prereg["destination"]
    require(socket.gethostname() == dest["hostname"] and os.getuid() == dest["uid"], "destination identity mismatch")
    require(args.physical_gpu in (5, 6, 7) and os.environ.get("CUDA_VISIBLE_DEVICES") == str(args.physical_gpu),
            "only one explicitly mapped physical GPU 5, 6, or 7 is authorized")
    require(not args.output.is_symlink() and args.output.resolve() == Path(dest["run_root"]) / f"B0_seed{args.seed}",
            "unexpected or symlink output root")
    require(args.output.stat().st_uid == os.getuid(), "output root owner mismatch")
    require(os.statvfs(args.output).f_bavail * os.statvfs(args.output).f_frsize >= 10 * 1024**3,
            "BLOCKED_STORAGE_OR_ARCHIVE_FAILURE: less than 10 GiB free")
    require(sha256(args.data_audit) == prereg["data_audit_sha256"], "data audit hash mismatch")
    require(read(args.data_audit)["status"] == "PASS_DATA_AND_RUNTIME_ADMISSION", "data admission did not pass")
    if args.seed != 0:
        require(args.seed0_gate is not None, "seed0 engineering and local archive gate is required")
        gate = read(args.seed0_gate)
        require(gate["status"] == "PASS_SEED0_BEFORE_SEED1_SEED2" and gate["actual_child_exit_code"] == 0,
                "seed0 gate did not pass")
        require(gate["archive_audit"]["status"] == "PASS_PRIVATE_ARCHIVE", "seed0 private archive missing")
        seed0 = Path(dest["run_root"]) / "B0_seed0"
        require(verify(seed0)["content_sha256"] == gate["archive_audit"]["content_sha256"], "seed0 bundle differs")
        require(sha256(seed0 / "BASELINE_V3_SEED_ENGINEERING_AUDIT.json") == gate["engineering_audit_sha256"],
                "seed0 engineering audit changed")
        require(read(seed0 / "BASELINE_V3_SEED_ENGINEERING_AUDIT.json")["status"] == "PASS_SEED_ENGINEERING",
                "seed0 engineering audit missing")
    config = load_yaml(ROOT / "configs/gate0_repaired_v2/fundus_pas_probmse.yaml")
    config["data"]["root"] = dest["data_root"]
    config["model"]["reference_root"] = dest["jascl_reference"]
    require(resolved_config_hash(config) == prereg["resolved_config_sha256"], "resolved frozen config changed")
    require(sha256(Path(dest["data_root"]) / "checksums/checksums.sha256") == prereg["checksums_sha256"],
            "frozen checksum list changed")
    data_check = verify_checksums(Path(dest["data_root"]))
    write_new(args.output / "BASELINE_V3_FROZEN_DATA_RECHECK.json", data_check)
    require(data_check["valid"] and data_check["entries"] == 2962, "frozen data recheck failed")
    provenance = dict(authorization_sha256=args.authorization_sha256,
                      preregistration_sha256=auth["preregistration_sha256"],
                      data_audit_sha256=prereg["data_audit_sha256"], checksums_sha256=prereg["checksums_sha256"],
                      domain_protocol_sha256=sha256(ROOT / "docs/di_dmpa_jascl/DOMAIN_PROTOCOL.yaml"))
    return config, load_yaml(ROOT / "docs/di_dmpa_jascl/DOMAIN_PROTOCOL.yaml"), provenance


def audit_seed(runner, output, config, protocol, provenance):
    training = runner.output_dir
    completion = read(training / "run_completion.json")
    expected_steps = sum(100 * math.ceil(len(runner.adapter.records(domain=d, role="train_labeled", purpose="train")) / 2)
                         + 15 * math.ceil(len(runner.adapter.records(domain=d, role="train_unlabeled", purpose="train")) / 2)
                         for d in runner.domain_order)
    require(completion["status"] == "COMPLETE" and (training / ".complete").is_file(), "training did not complete")
    require(completion["global_step"] == expected_steps, "unexpected fixed three-domain step coverage")
    require(completion["hidden_gt_training_usage"] == "none" and not completion["nan_detected"], "leakage/nonfinite completion")
    require(runner.adapter.leakage_audit()["status"] == "PASS", "final leakage audit failed")
    lines = [json.loads(line) for line in (training / "train.jsonl").read_text().splitlines()]
    require(len(lines) == expected_steps and [r["global_step"] for r in lines] == list(range(1, expected_steps + 1)),
            "missing/duplicate training steps")
    for row in lines:
        require(row["domain"] == runner.domain_order[int(row["stage_index"])] and row["hidden_gt_training_usage"] == "none",
                "training role/domain mismatch")
        require(all(not isinstance(v, float) or math.isfinite(v) for v in row.values()), "nonfinite training evidence")
    stages = []
    matrices = read(training / "stage_by_domain_matrices.json")
    for stage, domain in enumerate(runner.domain_order):
        path = training / f"stage_{stage}_{domain}" / "best.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        record = verify_payload(payload)
        require(record["binding"] == runner.provenance and payload["stage_state"]["stage_index"] == stage,
                "checkpoint binding/stage mismatch")
        channels = payload["student"]["decoder.conv_logit.mu.weight"].shape[1]
        require(payload["prototypes"].shape == (3, channels), "PAS geometry changed")
        require(torch.equal(payload["student"]["decoder.conv_logit.grad_update"], payload["gas_state"]["grad_update"]),
                "GAS does not match full classifier")
        loaded = RegeneratedB0Runner(repo_root=ROOT, config=config, protocol=protocol, seed=runner.seed,
            output_dir=output / f"independent_load_stage{stage}", device=runner.device, provenance=provenance,
            model_factory=runner.model_factory)
        load_checkpoint(path, wrapper=loaded.wrapper, optimizer=loaded.optimizer, scheduler=loaded.scheduler,
                        expected_config_hash=runner.config_hash, expected_git_commit=runner.git_commit, restore_rng=False)
        for key, value in (("student", loaded.wrapper.student.state_dict()), ("ema_teacher", loaded.wrapper.teacher.state_dict()),
                           ("optimizer", loaded.optimizer.state_dict()), ("scheduler", loaded.scheduler.state_dict())):
            require(state_digest(value) == state_digest(payload[key]), "independent load mismatch: " + key)
        before_model = state_digest(loaded.wrapper.state_dict())
        before_rng = state_digest(capture_rng_state())
        fresh = {}
        for evaluated in runner.domain_order[:stage + 1]:
            result = loaded.evaluate_domain(evaluated, "test")
            require(result["evaluation_classifier"] == "posterior_mean", "stochastic final evaluator")
            fresh[evaluated] = result
            for metric, matrix in matrices.items():
                require(set(matrix[domain]) == set(runner.domain_order[:stage + 1]), "incomplete stage/domain matrix")
                require(math.isfinite(result[metric]) and abs(result[metric] - matrix[domain][evaluated]) <= 1e-12,
                        "independent evaluation matrix mismatch")
        require(state_digest(capture_rng_state()) == before_rng, "deterministic evaluation consumed RNG")
        require(state_digest(loaded.wrapper.state_dict()) == before_model, "evaluation mutated model")
        require(all(p.grad is None for p in loaded.wrapper.parameters()), "evaluation wrote gradients")
        stages.append(dict(seed=runner.seed, stage_index=stage, domain=domain, checkpoint_id=f"B0/seed{runner.seed}/stage{stage}",
                           path=str(path), sha256=sha256(path), bytes=path.stat().st_size,
                           field_sha256=record["field_sha256"], classifier_sha256=record["classifier_sha256"],
                           legacy_pas_capture=record["legacy_pas_capture"], best_metric=payload["best_metric"],
                           selected_epoch=record["selected_checkpoint_epoch"], independent_load=True,
                           independent_evaluation=fresh, model_and_rng_unchanged=True))
        del loaded, payload
    write_new(output / "BASELINE_V3_CHECKPOINT_MANIFEST.json", dict(status="PASS", seed=runner.seed, checkpoints=stages))
    write_new(output / "BASELINE_V3_STAGE_MATRICES.json", dict(seed=runner.seed, matrices=matrices,
              lower_triangular_cells_per_metric=6, evaluation_classifier="posterior_mean", independent_reproduction=True))
    audit = dict(status="PASS_SEED_ENGINEERING", seed=runner.seed, completed_at=now(),
                 expected_steps=expected_steps, observed_steps=len(lines), domain_stages=3, epochs_per_domain=100,
                 checkpoint_count=3, stage_checkpoint_direct_pas=True, legacy_pas_reconstruction=False,
                 hidden_or_test_gt_training_usage="none", test_gt_evaluator_only=True,
                 test_gt_used_for_selection=False, deterministic_evaluation=True, independent_matrix_check=True,
                 no_nan=True, private_archive_status="PENDING_LOCAL_BYTE_VERIFICATION",
                 server_local_process_exit_status="PENDING_PARENT_EXIT_RECEIPT")
    write_new(output / "BASELINE_V3_SEED_ENGINEERING_AUDIT.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--seed0-gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    config, protocol, provenance = admission(args)
    runner = RegeneratedB0Runner(repo_root=ROOT, config=config, protocol=protocol, seed=args.seed,
        output_dir=args.output / "training", device="cuda:0", provenance=provenance)
    write_new(args.output / "BASELINE_V3_RUN_METADATA.json", dict(started_at=now(), seed=args.seed,
              provenance=runner.provenance, physical_gpu=args.physical_gpu, shared_gpu=True,
              code_root=str(ROOT), frozen_config=config, model="unchanged_repaired_Gate0_UNet_JASCL",
              method_training=False, diagnostic_forward_budget_consumed=0))
    runner.run()
    print(json.dumps(audit_seed(runner, args.output, config, protocol, provenance), sort_keys=True))


if __name__ == "__main__":
    main()
