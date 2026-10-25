"""Paired H3 test of foreground-balanced mean risk with matched Fisher."""

import copy
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import torch

from sc_bayeseg.supervised_cl_v1 import run as data
from ..hierarchical_bayes_h1 import run as base
from ..hierarchical_bayes_h1.model import HierarchicalSegmenter, PIXELS
from ..hierarchical_bayes_h1.online import OnlineGaussian
from ..foreground_curvature_h2.curvature import balanced_curvature
from .model import MatchedRiskSegmenter


SEED = 20261029
SITES = ("RUNMC", "BMC", "I2CVB")
ARMS = ("U-FT", "risk_control", "matched_risk")
UPDATES = 2 * 8000 + 3 * 2 * 6000


def preflight():
    base.curvature = balanced_curvature
    result = base.preflight()
    ordinary = HierarchicalSegmenter().cuda()
    matched = MatchedRiskSegmenter().cuda()
    matched.load_state_dict(ordinary.state_dict())
    image = torch.randn(1, 1, 256, 256, device="cuda") * .2
    label = torch.zeros(1, 256, 256, dtype=torch.long, device="cuda")
    label[:, 80:180, 80:180] = 1
    with torch.no_grad():
        torch.manual_seed(SEED + 41)
        ordinary_logits = ordinary.deploy_logits(image)
        torch.manual_seed(SEED + 41)
        matched_logits = matched.deploy_logits(image)
    if not torch.equal(ordinary_logits, matched_logits):
        raise AssertionError("H3 deployment paths differ before training")
    torch.manual_seed(SEED + 42)
    ordinary_loss, _ = ordinary.loss(image, label)
    torch.manual_seed(SEED + 42)
    matched_loss, _ = matched.loss(image, label)
    matched_loss.backward()
    if (not torch.isfinite(ordinary_loss) or not torch.isfinite(matched_loss) or
            abs(float(ordinary_loss - matched_loss)) <= 1e-7 or
            any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                for parameter in matched.parameters())):
        raise FloatingPointError("H3 matched-risk full objective P0 failed")
    result.update({"deployment_initial_exact": True,
                   "matched_full_risk_distinct": True,
                   "matched_full_backward_finite": True})
    return result


def real_p0(models, precision, cases, root, deadline):
    if any(not torch.equal(value, models["matched_risk"].state_dict()[name])
           for name, value in models["risk_control"].state_dict().items()):
        raise AssertionError("H3 paired source models differ")
    image, label = base.batch(cases, "BMC", 1)
    with torch.no_grad():
        torch.manual_seed(base.stage_seed("BMC") + 1)
        left_field = models["risk_control"].deploy_logits(image)
        torch.manual_seed(base.stage_seed("BMC") + 1)
        right_field = models["matched_risk"].deploy_logits(image)
    if not torch.equal(left_field.detach(), right_field.detach()):
        raise AssertionError("H3 paired deployment outputs differ at BMC start")
    clone = copy.deepcopy(models["matched_risk"])
    state = OnlineGaussian(clone, precision, "F0", SEED)
    state.transition()
    torch.manual_seed(base.stage_seed("BMC") + 1)
    loss, _ = clone.loss(image, label)
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(clone.parameters(), 5.0))
    if not math.isfinite(norm):
        raise FloatingPointError("H3 full-risk gradient is non-finite")
    curve = balanced_curvature(clone, image, base.stage_seed("BMC") + 500_001, 2)
    state.update(PIXELS * len(cases) / 6000, curve)
    optimizer = torch.optim.AdamW(clone.recognition_parameters(), lr=3e-4, weight_decay=1e-5)
    optimizer.step()
    if any(not torch.isfinite(value).all() for value in clone.parameters()):
        raise FloatingPointError("H3 discarded golden update is non-finite")
    data.write(root / "P0_REAL.json", {"status": "PASS", "same_source_exact": True,
                                       "same_first_batch_deployment_exact": True,
                                       "balanced_curvature_finite": True,
                                       "discarded_engineering_updates": 1})
    base.guard(root, deadline)


def run(settings):
    os.umask(0o077)
    root = Path(settings["output"])
    if not root.is_dir() or (root / "STARTED.json").exists():
        raise FileExistsError("H3 root must be create-only and unused")
    deadline = float(settings["started_unix"]) + 23 * 3600
    if time.time() >= deadline:
        raise TimeoutError("H3 original safety clock expired")
    base.SEED = SEED
    base.MAX_BYTES = 46_000_000_000  # leaves room for immutable N1 and H1 roots
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(int(settings["gpu_index"]))
    synthetic = preflight()  # no patient data read
    catalog_bytes = Path(settings["catalog"]).read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != settings["catalog_sha256"]:
        raise ValueError("private catalog changed")
    catalog = json.loads(catalog_bytes)
    if (tuple(catalog["sites"][:3]) != SITES or catalog["scope"] != "all_train" or
            catalog["test_case_access"] != 0 or catalog["patient_overlap"]):
        raise ValueError("H3 data boundary mismatch")
    data.write(root / "STARTED.json", {"started_unix": settings["started_unix"], "seed": SEED,
                                       "sites": list(SITES), "arms": list(ARMS),
                                       "updates_planned": UPDATES, "old_train_replay": 0,
                                       "old_teacher": 0, "test_access": 0})
    data.write(root / "P0_SYNTHETIC.json", {"status": "PASS", **synthetic})
    runtime = {"stage": "source", "step": 0}
    updates = 0
    try:
        base.guard(root, deadline)
        torch.manual_seed(SEED)
        backbone = data.backbone().cuda()
        torch.manual_seed(SEED + 1)
        source_model = HierarchicalSegmenter().cuda()
        source_model.backbone.load_state_dict(backbone.state_dict())
        if any(not torch.equal(value, source_model.backbone.state_dict()[name])
               for name, value in backbone.state_dict().items()):
            raise AssertionError("H3 U/C backbone initial states differ")
        source_cases = base.current_cases(catalog, settings["data_root"], "RUNMC")
        base.train_stage(backbone, source_cases, "RUNMC", "U", "U-FT", 8000,
                         root / "source_U-FT", None, root, deadline, runtime, settings)
        updates += 8000
        source_stage = root / "source_C"
        base.train_stage(source_model, source_cases, "RUNMC", "H", "source_C", 8000,
                         source_stage, None, root, deadline, runtime, settings)
        updates += 8000
        audit_dir = root / "curvature_balanced"
        audit_dir.mkdir(exist_ok=False)
        base.curvature = balanced_curvature
        precision = base.source_precision(source_model, source_cases, audit_dir, deadline)
        del source_cases
        matched = MatchedRiskSegmenter().cuda()
        matched.load_state_dict(source_model.state_dict())
        matched.eval()
        models = {"U-FT": backbone, "risk_control": copy.deepcopy(source_model),
                  "matched_risk": matched}
        del source_model
        states = {arm: OnlineGaussian(models[arm], precision, "F0", SEED) for arm in ARMS[1:]}
        for site in SITES[1:]:
            current = base.current_cases(catalog, settings["data_root"], site)
            if site == "BMC":
                real_p0(models, precision, current, root, deadline)
            for arm in ARMS:
                base.guard(root, deadline)
                base.curvature = balanced_curvature
                state = states.get(arm)
                if state is not None and state.transition() != 0:
                    raise AssertionError("H3 F0 transition must have zero process noise")
                stage = root / f"{arm}_{site}"
                stage.mkdir(exist_ok=False)
                family = "U" if arm == "U-FT" else "H"
                base.evaluate(settings, root, stage, models[arm], family, 0, "val",
                              SITES[:SITES.index(site) + 1], site, deadline)
                base.train_stage(models[arm], current, site, family, arm, 6000, stage,
                                 state, root, deadline, runtime, settings)
                updates += 6000
            del current
        if updates != UPDATES:
            raise AssertionError("H3 formal update count mismatch")
        data.write(root / "STATUS.json", {"status": "COMPLETE", "updates": updates,
                                          "stages": 8, "elapsed_seconds": time.time() - settings["started_unix"],
                                          "bytes": data.bytes_used(root),
                                          "old_train_replay": 0, "old_teacher": 0})
        data.write(root / "EXIT.json", {"exit_code": 0})
    except BaseException:
        failure = {"stage": runtime.get("stage", "P0"), "step": runtime.get("step", 0),
                   "completed_stage_updates": updates,
                   "model": runtime["model"].state_dict() if "model" in runtime else None,
                   "optimizer": runtime["optimizer"].state_dict() if "optimizer" in runtime else None,
                   "precision": runtime["online"].snapshot() if runtime.get("online") is not None else None,
                   "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}
        torch.save(failure, root / "failure_partial.pt")
        data.write(root / "FAILURE.json", {"stage": failure["stage"], "step": failure["step"],
                                        "completed_stage_updates": updates,
                                        "traceback": traceback.format_exc()})
        data.write(root / "EXIT.json", {"exit_code": 1})
        raise


if __name__ == "__main__":
    run(json.loads(Path(os.environ["TASK_SETTINGS"]).read_text()))
