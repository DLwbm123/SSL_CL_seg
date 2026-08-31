"""Fresh v3 authority and inputs around the unchanged Gate1C numerical engine."""
from pathlib import Path
import os
import socket
import subprocess
import sys

import torch

from di_dmpa_gate1c_v2 import binding as b
from di_dmpa_gate1c_v2.reliability import banks
from di_dmpa_jascl.provenance import git_revision
from lcrseg.acceptance import verify_checksums
from . import PROTOCOL
from .durable import read, sha256, verify

ROOT = Path(__file__).resolve().parents[1]
BUDGET = dict(validation=990, integration=75, formal=1800, total=2865)
INTEGRATION_IDS = ["B0/seed0/stage0/REFUGE/pair00", "B0/seed1/stage1/RIM_ONE_r3/pair00",
                   "B0/seed0/stage2/Drishti_GS/pair01"]


def validate_contract(p, original):
    b.require(p["protocol"] == PROTOCOL and p["budget"] == BUDGET, "wrong v3 protocol/budget")
    for key in ("primary", "method_flags", "feature_contract", "null_policy", "gate1c", "validation",
                "gradient_execution", "teacher_noise", "poe_control", "numerical", "admission"):
        b.require(p[key] == original[key], "frozen numerical contract changed: " + key)
    b.require(all(v is False for v in p["method_flags"].values()), "method flag enabled")
    for key, value in original["benchmark"].items():
        if key not in ("data_root", "hdf5_root"):
            b.require(p["benchmark"][key] == value, "frozen benchmark changed: " + key)
    for key, value in original["gradient_diagnostic"].items():
        if key != "batch_pairs":
            b.require(p["gradient_diagnostic"][key] == value, "frozen gradient contract changed: " + key)
    pairs = p["gradient_diagnostic"]["batch_pairs"]
    strip = lambda rows: [{k: v for k, v in q.items() if k != "checkpoint_sha256"} for q in rows]
    b.require(strip(pairs) == strip(original["gradient_diagnostic"]["batch_pairs"]), "fixed pairs/seeds changed")
    b.require(len(pairs) == 72 and b.H(pairs) == p["fixed_batch_pairs_sha256"], "incomplete v3 fixed pairs")
    b.require(p["integration_pair_ids"] == INTEGRATION_IDS, "original three integration pairs changed")
    checkpoints = p["immutable_baseline"]["checkpoint_inputs"]
    b.require(len(checkpoints) == 9 and {c["checkpoint_id"] for c in checkpoints}
              == {f"B0/seed{s}/stage{t}" for s in range(3) for t in range(3)}, "nine regenerated B0 checkpoints required")
    for pair in pairs:
        cp = b.checkpoint(p, pair["seed"], pair["stage_index"])
        b.require(cp["sha256"] == pair["checkpoint_sha256"], "pair bound to a different checkpoint")
    b.require("legacy_prototype_reconstruction" not in p and p["input_contract_version"] == "v3", "reconstruction forbidden")
    b.require(p["diagnostic_precision"] == "float64_shadow" and p["allowed_gpus"] == [4, 5, 6, 7], "precision/GPU scope changed")


def load_authority(args):
    b.check_hash(args.authorization, args.authorization_sha256)
    auth = read(args.authorization)
    b.require(auth["status"] == "AUTHORIZED_GATE1C_V3_CLEAN_REGEN_DIAGNOSTIC_ONLY", "missing v3 authorization")
    reg_path = args.authorization.parent / "DI_DMPA_GATE1C_V3_PREREGISTRATION.json"
    b.check_hash(reg_path, auth["preregistration_sha256"])
    p = read(reg_path)
    b.require(git_revision(ROOT) == p["code_commit"] == auth["code_commit"], "wrong exact execution code")
    for commit, filename, digest in ((auth["preregistration_commit"], "DI_DMPA_GATE1C_V3_PREREGISTRATION.json", auth["preregistration_sha256"]),
                                     (args.authorization_commit, "GATE1C_V3_EXECUTION_AUTHORIZATION.json", args.authorization_sha256)):
        import hashlib
        blob = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:experiments/lcrseg/docs/di_dmpa_jascl/{filename}"])
        b.require(hashlib.sha256(blob).hexdigest() == digest, "publication commit binding mismatch")
    b.require(not subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True).strip(),
              "dirty execution source")
    for relative, digest in p["frozen_source_sha256"].items():
        b.check_hash(ROOT / relative, digest)
    inherited = ROOT / p["inherited_numeric_contract"]["path"]
    b.check_hash(inherited, p["inherited_numeric_contract"]["sha256"])
    validate_contract(p, read(inherited))
    dest = p["destination"]
    b.require(socket.gethostname() == dest["hostname"] and os.getuid() == os.geteuid() == dest["uid"], "destination identity changed")
    b.require(sys.executable == p["runtime"]["python_executable"] and str(torch.__version__) == p["runtime"]["torch"]
              and torch.version.cuda == p["runtime"]["cuda"], "registered runtime changed")
    b.require(args.output.resolve() == Path(p["output_roots"][args.scope]) and not args.output.is_symlink()
              and args.output.stat().st_uid == os.getuid(), "unsafe/unregistered output root")
    b.check_hash(p["baseline_manifest"]["path"], p["baseline_manifest"]["sha256"])
    baseline = read(p["baseline_manifest"]["path"])
    b.require(baseline["status"] == "PASS_ALL_THREE_REGENERATED_B0_SEEDS"
              and baseline["checkpoints"] == p["immutable_baseline"]["checkpoint_inputs"], "unadmitted regenerated baseline")
    b.check_hash(p["k2_freeze"]["path"], p["k2_freeze"]["sha256"])
    freeze = read(p["k2_freeze"]["path"])
    b.require(freeze["baseline_manifest_sha256"] == p["baseline_manifest"]["sha256"], "K2 from another baseline")
    b.check_hash(p["k2_report"]["path"], p["k2_report"]["sha256"])
    b.require(read(p["k2_report"]["path"])["status"] == "K2_REPLICATION_PASS"
              and freeze["replication_report_sha256"] == p["k2_report"]["sha256"], "K2 replication did not pass")
    for seed in range(3):
        for stage in range(3):
            banks(freeze, seed, stage)
    p["_precision_contract_verified"] = True
    meta = dict(registration_id=PROTOCOL, diagnostic_code_commit=p["code_commit"], input_contract_version="v3",
                preregistration_commit=auth["preregistration_commit"], preregistration_sha256=auth["preregistration_sha256"],
                authorization_commit=args.authorization_commit, authorization_sha256=args.authorization_sha256,
                baseline_manifest_sha256=p["baseline_manifest"]["sha256"], k2_freeze_sha256=p["k2_freeze"]["sha256"],
                fixed_batch_pairs_sha256=p["fixed_batch_pairs_sha256"], execution_scope=args.scope,
                gate1_overall_status="FAIL_TRANSPORT_NOT_SUPPORTED", hidden_gt_training_usage="none", test_gt_usage="none",
                model_optimizer_steps=0, transport_optimizer_steps_this_gate=0, method_registered=False,
                next_action="REPORT_AND_HARD_STOP_NO_METHOD_IMPLEMENTATION")
    return p, freeze, meta


def audit_inputs(p, metadata):
    baseline = read(p["baseline_manifest"]["path"])
    admissions = baseline["seed_admissions"]
    b.require(len(admissions) == 3 and {a["seed"] for a in admissions} == {0, 1, 2}, "incomplete baseline admissions")
    for row in admissions:
        root = Path(row["remote_root"])
        b.require(root.resolve() == Path(p["destination"]["baseline_run_root"]) / f"B0_seed{row['seed']}", "old baseline input root forbidden")
        b.require(row["archive_audit"]["status"] == "PASS_PRIVATE_ARCHIVE" and row["actual_child_exit_code"] == 0
                  and verify(root)["content_sha256"] == row["archive_audit"]["content_sha256"], "baseline private archive mismatch")
        b.require(read(root / "PROCESS_EXIT.json")["actual_child_exit_code"] == 0
                  and read(root / "BASELINE_V3_SEED_ENGINEERING_AUDIT.json")["status"] == "PASS_SEED_ENGINEERING", "baseline exit/engineering failed")
    data = Path(p["destination"]["data_root"])
    b.check_hash(data / "checksums/checksums.sha256", p["checksums_sha256"])
    data_check = verify_checksums(data)
    b.require(data_check["valid"] and data_check["entries"] == 2962, "frozen data changed")
    units = []
    for seed in range(3):
        for stage in range(3):
            cp = b.checkpoint(p, seed, stage)
            b.check_hash(cp["path"], cp["sha256"])
            b.require(Path(cp["path"]).resolve().is_relative_to(Path(p["destination"]["baseline_run_root"])), "checkpoint outside v3 root")
            units.append(dict(seed=seed, stage_index=stage, counts={role: len(b.records(data, p, seed, stage, role))
                for role in ("train_labeled", "train_unlabeled", "val")}))
    legacy = b.legacy_input_audit(p)
    b.require(legacy["reconstructed_inputs"] == 0 and all(row["tensor_sha256"] == next(c["legacy_pas_tensor_sha256"]
              for c in p["immutable_baseline"]["checkpoint_inputs"] if c["checkpoint_id"] == row["checkpoint_id"])
              for row in legacy["checkpoints"]), "direct historical PAS bank identity changed")
    return dict(metadata=metadata, status="PASS", units=units, data_recheck=data_check, legacy_payload_readiness=legacy,
                hidden_gt_training_usage="none", test_gt_usage="none", test_role_constructions=0,
                legacy_pas_reconstruction=False, method_optimizer_steps=0, old_private_inputs_read=False)
