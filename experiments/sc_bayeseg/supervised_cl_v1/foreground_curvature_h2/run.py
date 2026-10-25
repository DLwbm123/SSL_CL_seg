"""Paired H2 test of foreground-balanced predictive Fisher precision."""

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
from ..hierarchical_bayes_h1.online import OnlineGaussian, curvature as standard_curvature
from .curvature import balanced_curvature


SEED = 20261028
SITES = ("RUNMC", "BMC", "I2CVB")
ARMS = ("U-FT", "standard", "balanced")
UPDATES = 2 * 8000 + 3 * 2 * 6000


def preflight():
    base.curvature = balanced_curvature
    result = base.preflight()
    probability = torch.tensor((.05, .2, .8), device="cuda")
    weight = .5 * probability / probability.sum() + .5 * (1 - probability) / (1 - probability).sum()
    if abs(float(weight.sum()) - 1) > 1e-6:
        raise AssertionError("H2 balanced Fisher weights do not sum to one")
    model = HierarchicalSegmenter().cuda()
    image = torch.randn(1, 1, 256, 256, device="cuda")
    standard = standard_curvature(model, image, SEED + 41)
    balanced = balanced_curvature(model, image, SEED + 41)
    delta = sum(float((balanced[name] - standard[name]).abs().sum()) for name in standard)
    if not math.isfinite(delta) or delta <= 0:
        raise AssertionError("H2 balanced curvature is not distinct and finite")
    result.update({"balanced_weight_total": float(weight.sum()),
                   "balanced_vs_standard_curvature_l1": delta,
                   "both_curvatures_finite": True})
    return result


def real_p0(models, precisions, cases, root, deadline):
    if any(not torch.equal(value, models["balanced"].state_dict()[name])
           for name, value in models["standard"].state_dict().items()):
        raise AssertionError("H2 paired source models differ")
    image, label = base.batch(cases, "BMC", 1)
    torch.manual_seed(base.stage_seed("BMC") + 1)
    left, _ = models["standard"].loss(image, label)
    torch.manual_seed(base.stage_seed("BMC") + 1)
    right, _ = models["balanced"].loss(image, label)
    if not torch.equal(left.detach(), right.detach()):
        raise AssertionError("H2 paired full risks differ")
    clone = copy.deepcopy(models["balanced"])
    state = OnlineGaussian(clone, precisions["balanced"], "F0", SEED)
    state.transition()
    torch.manual_seed(base.stage_seed("BMC") + 1)
    loss, _ = clone.loss(image, label)
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(clone.parameters(), 5.0))
    if not math.isfinite(norm):
        raise FloatingPointError("H2 full-risk gradient is non-finite")
    curve = balanced_curvature(clone, image, base.stage_seed("BMC") + 500_001, 2)
    state.update(PIXELS * len(cases) / 6000, curve)
    optimizer = torch.optim.AdamW(clone.recognition_parameters(), lr=3e-4, weight_decay=1e-5)
    optimizer.step()
    if any(not torch.isfinite(value).all() for value in clone.parameters()):
        raise FloatingPointError("H2 discarded golden update is non-finite")
    data.write(root / "P0_REAL.json", {"status": "PASS", "same_source_exact": True,
                                       "same_first_batch_risk_exact": True,
                                       "balanced_curvature_finite": True,
                                       "discarded_engineering_updates": 1})
    base.guard(root, deadline)


def run(settings):
    os.umask(0o077)
    root = Path(settings["output"])
    if not root.is_dir() or (root / "STARTED.json").exists():
        raise FileExistsError("H2 root must be create-only and unused")
    deadline = float(settings["started_unix"]) + 23 * 3600
    if time.time() >= deadline:
        raise TimeoutError("H2 original safety clock expired")
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
        raise ValueError("H2 data boundary mismatch")
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
            raise AssertionError("H2 U/C backbone initial states differ")
        source_cases = base.current_cases(catalog, settings["data_root"], "RUNMC")
        base.train_stage(backbone, source_cases, "RUNMC", "U", "U-FT", 8000,
                         root / "source_U-FT", None, root, deadline, runtime, settings)
        updates += 8000
        source_stage = root / "source_C"
        base.train_stage(source_model, source_cases, "RUNMC", "H", "source_C", 8000,
                         source_stage, None, root, deadline, runtime, settings)
        updates += 8000
        precision = {}
        for arm, function in (("standard", standard_curvature), ("balanced", balanced_curvature)):
            audit_dir = root / f"curvature_{arm}"
            audit_dir.mkdir(exist_ok=False)
            base.curvature = function
            precision[arm] = base.source_precision(source_model, source_cases, audit_dir, deadline)
        del source_cases
        models = {"U-FT": backbone,
                  **{arm: copy.deepcopy(source_model) for arm in ARMS[1:]}}
        del source_model
        states = {arm: OnlineGaussian(models[arm], precision[arm], "F0", SEED) for arm in ARMS[1:]}
        for site in SITES[1:]:
            current = base.current_cases(catalog, settings["data_root"], site)
            if site == "BMC":
                real_p0(models, precision, current, root, deadline)
            for arm, function in (("U-FT", standard_curvature),
                                  ("standard", standard_curvature),
                                  ("balanced", balanced_curvature)):
                base.guard(root, deadline)
                base.curvature = function
                state = states.get(arm)
                if state is not None and state.transition() != 0:
                    raise AssertionError("H2 F0 transition must have zero process noise")
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
            raise AssertionError("H2 formal update count mismatch")
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
