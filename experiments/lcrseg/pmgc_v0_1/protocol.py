"""Exact published authority, fixed metadata and NAS execution admission."""
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys

import torch

from di_dmpa_gate1c_v2 import binding as b
from di_dmpa_gate1c_v3 import durable as d
from . import REGISTRATION
from .core import require

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DOCS = ROOT/"docs/pmgc_v0_1"
BASE = "3126e59a63b205f2a075f28efa9f5d83b3911792"
REG_COMMIT = "efc187287b4b10b153867b20905cc3ddeefd94d9"
AUTH_COMMIT = "7dcac476d5ca84349d407e2f2f9ca2c8269f872e"
REG_SHA = "7731664cf715dd8b43e286f28c38a285d966b793490f07e488a9c1e22ab248aa"
AUTH_SHA = "5c179992b82f1c1fb7f58e1c0301870486cf79eb46035c3e194f82b9df0b4ad4"


def authority():
    values = []
    for commit, name, digest in ((REG_COMMIT, "PMGC_V0_1_FEASIBILITY_PREREGISTRATION.json", REG_SHA),
                                 (AUTH_COMMIT, "PMGC_V0_1_EXECUTION_AUTHORIZATION.json", AUTH_SHA)):
        path = DOCS/name
        b.check_hash(path, digest)
        blob = subprocess.check_output(["git", "-C", str(REPO), "show", commit+":"+str(path.relative_to(REPO))])
        require(hashlib.sha256(blob).hexdigest() == digest, "authority publication blob changed")
        values.append(d.read(path))
    reg, auth = values
    require(reg["registration_id"] == auth["registration_id"] == REGISTRATION and reg["base_commit"] == auth["base_commit"] == BASE, "PMGC identity/base changed")
    require(auth["preregistration_commit"] == auth["preregistration_remote_verified_commit"] == REG_COMMIT, "separate authorization missing")
    require(reg["flags"] == auth["flags"] == dict(model_optimizer_steps=0, transport_optimizer_steps=0, method_registered=False, training_launched=False), "training flag enabled")
    for spec in reg["closure"]["files"].values():
        b.check_hash(REPO/spec["path"], spec["sha256"])
    closure = d.read(REPO/reg["closure"]["files"]["json"]["path"])
    for field in ("prototype_selection_status", "prototype_weighting_status", "identity_history_status", "current_only_ranking_status", "transport_status"):
        require(closure[field] == "FAIL", "old method closure reopened")
    require(closure["relation_method_status"] == "FROZEN_FAILED" and closure["next_protocol"] == "PMGC_V0_1_FEASIBILITY", "relation closure changed")
    require(all(not closure[f"additional_{kind}_attempts_authorized"] for kind in ("selection", "weight_calibration", "transport", "relation_loss")), "closed line authorization changed")
    for path, digest in reg["frozen_source_sha256"].items():
        b.check_hash(REPO/path, digest)
    spec = reg["inputs"]
    b.check_hash(REPO/spec["inherited_path"], spec["inherited_sha256"])
    p = d.read(REPO/spec["inherited_path"])
    require(reg["inputs"]["checkpoints"] == p["immutable_baseline"]["checkpoint_inputs"], "nine checkpoint inputs changed")
    require(b.H(reg["fixed_units"]) == reg["fixed_units_sha256"], "fixed units changed")
    pairs = [pair for unit in reg["fixed_units"] for pair in unit["formal_pairs"]]
    expected = [pair for pair in p["gradient_diagnostic"]["batch_pairs"] if pair["stage_index"] in (1, 2)]
    require(pairs == expected and len(pairs) == 48, "stage1/stage2 exact48 pairs changed")
    require(len(reg["fixed_units"]) == 6 and len({u["integration_pair_id"] for u in reg["fixed_units"]}) == 6, "six integration units changed")
    return reg, p


def execution_gate(args):
    reg, p = authority()
    require(socket.gethostname() == reg["destination"]["hostname"] and os.getuid() == os.geteuid() == reg["destination"]["uid"], "wrong host/user")
    require(sys.executable == reg["runtime"]["python_executable"] and str(torch.__version__) == reg["runtime"]["torch"] and torch.version.cuda == reg["runtime"]["cuda"], "runtime changed")
    require(b.git(REPO, "rev-parse", "HEAD") == args.code_commit, "execution HEAD differs")
    require(not b.git(REPO, "status", "--porcelain", "--untracked-files=no"), "dirty execution source")
    require(all(line.startswith("A\t") for line in b.git(REPO, "diff", "--name-status", BASE, args.code_commit).splitlines()), "historical file modified")
    require(args.output.resolve().is_relative_to(Path(reg["destination"]["root"])) and not args.output.is_symlink(), "unregistered/non-NAS output")
    require(os.environ.get("SSLCL_STORAGE_ROOT") == "/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg" and Path(os.environ.get("TMPDIR", "/")).resolve().is_relative_to(Path("/data_nas")), "NAS wrapper/cache absent")
    gate = d.read(args.gate)
    require(gate["preregistration_remote_verified_commit"] == REG_COMMIT and gate["authorization_remote_verified_commit"] == AUTH_COMMIT and gate["code_commit"] == gate["remote_verified_code_commit"] == args.code_commit, "publication barrier incomplete")
    for path, digest in gate["exact_source_sha256"].items():
        b.check_hash(REPO/path, digest)
    for field in ("test_report", "call_graph"):
        b.check_hash(gate[field], gate[field+"_sha256"])
        value = d.read(gate[field])
        require(value["status"] == "PASS" and value["code_commit"] == args.code_commit, "test/compiler prerequisite failed")
        if field == "test_report":
            require(value["failures"] == value["errors"] == value["skips"] == 0, "nonclean tests")
            b.check_hash(value["junit_path"], value["junit_sha256"])
        else:
            require(value["total"] == reg["call_graph"]["total"], "compiler total differs")
    return reg, p, dict(registration_id=REGISTRATION, code_commit=args.code_commit,
                        publication_gate_sha256=d.sha256(args.gate), preregistration_commit=REG_COMMIT,
                        authorization_commit=AUTH_COMMIT, flags=reg["flags"], started_at=d.now())


def completed(root, phase, report):
    root = Path(root)
    require(d.read(root/"PROCESS_EXIT.json")["actual_child_exit_code"] == 0 and d.read(root/"EXECUTION_COMPLETION.json")["status"] == "COMMAND_COMPLETED", "phase lacks durable success", "BLOCKED_INCOMPLETE_EVIDENCE")
    d.verify(root, f"PHASE_{phase}_MANIFEST.json")
    value = d.read(root/report)
    require(value["status"].startswith("PASS"), "phase audit not admitted", "BLOCKED_INCOMPLETE_EVIDENCE")
    return value
