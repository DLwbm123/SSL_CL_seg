"""Paired H4 test of a current diagonal-plus-low-rank Gaussian precision."""

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
from .online import (LowRankOnlineGaussian, predictive_basis,
                     split_source_precision, woodbury_identity_check)


SEED = 20261030
SITES = ("RUNMC", "BMC", "I2CVB")
ARMS = ("U-FT", "diagonal", "lowrank")
UPDATES = 2 * 8000 + 3 * 2 * 6000


def preflight():
    base.curvature = balanced_curvature
    result = base.preflight()
    result.update({"woodbury_identity_error": woodbury_identity_check(),
                   "gaussian_precision_psd_by_construction": True})
    return result


def real_p0(models, states, cases, root, deadline):
    if any(not torch.equal(value, models["lowrank"].state_dict()[name])
           for name, value in models["diagonal"].state_dict().items()):
        raise AssertionError("H4 paired source models differ")
    image, label = base.batch(cases, "BMC", 1)
    with torch.no_grad():
        torch.manual_seed(base.stage_seed("BMC") + 1)
        left_field = models["diagonal"].deploy_logits(image)
        torch.manual_seed(base.stage_seed("BMC") + 1)
        right_field = models["lowrank"].deploy_logits(image)
    if not torch.equal(left_field.detach(), right_field.detach()):
        raise AssertionError("H4 paired deployment outputs differ at BMC start")
    torch.manual_seed(base.stage_seed("BMC") + 1)
    left_loss, _ = models["diagonal"].loss(image, label)
    torch.manual_seed(base.stage_seed("BMC") + 1)
    right_loss, _ = models["lowrank"].loss(image, label)
    if not torch.equal(left_loss.detach(), right_loss.detach()):
        raise AssertionError("H4 paired complete objectives differ at BMC start")
    del left_loss, right_loss
    clone = copy.deepcopy(models["lowrank"])
    state = LowRankOnlineGaussian(clone, states["lowrank"].precision,
                                  states["lowrank"].basis)
    state.transition()
    torch.manual_seed(base.stage_seed("BMC") + 1)
    loss, _ = clone.loss(image, label)
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(clone.parameters(), 5.0))
    if not math.isfinite(norm):
        raise FloatingPointError("H4 full-risk gradient is non-finite")
    curve = balanced_curvature(clone, image, base.stage_seed("BMC") + 500_001, 2)
    state.update(PIXELS * len(cases) / 6000, curve)
    optimizer = torch.optim.AdamW(clone.recognition_parameters(), lr=3e-4, weight_decay=1e-5)
    optimizer.step()
    if any(not torch.isfinite(value).all() for value in clone.parameters()):
        raise FloatingPointError("H4 discarded golden update is non-finite")
    data.write(root / "P0_REAL.json", {"status": "PASS", "same_source_exact": True,
                                       "same_first_batch_deployment_exact": True,
                                       "same_first_batch_complete_risk_exact": True,
                                       "balanced_curvature_finite": True,
                                       "woodbury_step_finite": True,
                                       "source_rank": states["lowrank"].basis.shape[1],
                                       "discarded_engineering_updates": 1})
    base.guard(root, deadline)


def run(settings):
    os.umask(0o077)
    root = Path(settings["output"])
    if not root.is_dir() or (root / "STARTED.json").exists():
        raise FileExistsError("H4 root must be create-only and unused")
    deadline = float(settings["started_unix"]) + 23 * 3600
    if time.time() >= deadline:
        raise TimeoutError("H4 original safety clock expired")
    base.SEED = SEED
    base.MAX_BYTES = 45_000_000_000  # plus existing N1/H1/H2/H3 is below 50e9
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
        raise ValueError("H4 data boundary mismatch")
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
            raise AssertionError("H4 U/C backbone initial states differ")
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
        lowrank_diagonal, source_trace = split_source_precision(source_model, precision)
        source_basis, source_audit = predictive_basis(source_model, source_cases,
                                                      "RUNMC", SEED, source_trace)
        data.write(root / "SOURCE_LOW_RANK_AUDIT.json", source_audit)
        base.guard(root, deadline)
        del source_cases
        models = {"U-FT": backbone, "diagonal": copy.deepcopy(source_model),
                  "lowrank": copy.deepcopy(source_model)}
        del source_model
        states = {"diagonal": OnlineGaussian(models["diagonal"], precision, "F0", SEED),
                  "lowrank": LowRankOnlineGaussian(models["lowrank"],
                                                   lowrank_diagonal, source_basis)}
        for site in SITES[1:]:
            current = base.current_cases(catalog, settings["data_root"], site)
            if site == "BMC":
                real_p0(models, states, current, root, deadline)
            states["lowrank"].begin_site()
            for arm in ARMS:
                base.guard(root, deadline)
                base.curvature = balanced_curvature
                state = states.get(arm)
                if state is not None and state.transition() != 0:
                    raise AssertionError("H4 F0 transition must have zero process noise")
                stage = root / f"{arm}_{site}"
                stage.mkdir(exist_ok=False)
                family = "U" if arm == "U-FT" else "H"
                base.evaluate(settings, root, stage, models[arm], family, 0, "val",
                              SITES[:SITES.index(site) + 1], site, deadline)
                base.train_stage(models[arm], current, site, family, arm, 6000, stage,
                                 state, root, deadline, runtime, settings)
                updates += 6000
            if site == "BMC":
                audit = states["lowrank"].add_current_site_basis(models["lowrank"],
                                                                   current, site, SEED)
                data.write(root / "BMC_LOW_RANK_AUDIT.json", audit)
                base.save_new(root / "BMC_CURRENT_STATE_WITH_BASIS.pt",
                              {"model": models["lowrank"].state_dict(),
                               "posterior": states["lowrank"].snapshot(),
                               "site": site})
                base.guard(root, deadline)
            del current
        if updates != UPDATES:
            raise AssertionError("H4 formal update count mismatch")
        data.write(root / "STATUS.json", {"status": "COMPLETE", "updates": updates,
                                          "stages": 8, "elapsed_seconds": time.time() - settings["started_unix"],
                                          "bytes": data.bytes_used(root),
                                          "old_train_replay": 0, "old_teacher": 0,
                                          "final_rank": states["lowrank"].basis.shape[1]})
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
