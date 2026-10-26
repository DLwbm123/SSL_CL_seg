"""Paired H5 test of anatomy-preserving acquisition orbits on current train."""

import copy
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from sc_bayeseg.supervised_cl_v1 import run as data
from ..hierarchical_bayes_h1 import run as base
from ..hierarchical_bayes_h1.model import HierarchicalSegmenter, PIXELS
from ..foreground_curvature_h2.curvature import balanced_curvature
from ..lowrank_posterior_h4.online import (LowRankOnlineGaussian, predictive_basis,
                     split_source_precision, woodbury_identity_check)
from .model import OrbitSegmenter


SEED = 20261031
SITES = ("RUNMC", "BMC", "I2CVB")
ARMS = ("U-FT", "lowrank_control", "orbit_lowrank")
UPDATES = 2 * 8000 + 3 * 2 * 6000


def preflight():
    base.curvature = balanced_curvature
    result = base.preflight()
    plain = HierarchicalSegmenter().cuda()
    orbit = OrbitSegmenter().cuda()
    orbit.load_state_dict(plain.state_dict())
    image = torch.randn(1, 1, 256, 256, device="cuda") * .2
    label = torch.zeros(1, 256, 256, dtype=torch.long, device="cuda")
    label[:, 80:170, 80:170] = 1
    raw = torch.ones(1, 6, device="cuda") * .7
    with torch.no_grad():
        transformed = orbit.forward_appearance(image, raw)
        inversion = orbit.inverse_appearance(transformed, raw)
    if (float((inversion - image).abs().max()) > 1e-5 or
            not torch.isfinite(transformed).all()):
        raise AssertionError("H5 current-image acquisition orbit is not invertible")
    torch.manual_seed(SEED + 41)
    ordinary_loss, _ = HierarchicalSegmenter.loss(orbit, image, label)
    torch.manual_seed(SEED + 41)
    orbit_loss, _ = orbit.loss(image, label)
    if not torch.isfinite(orbit_loss) or abs(float(orbit_loss - ordinary_loss)) < 1e-7:
        raise FloatingPointError("H5 orbit risk did not differ from ordinary risk")
    orbit_loss.backward()
    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
           for parameter in orbit.parameters()):
        raise FloatingPointError("H5 orbit full objective backward failed")
    result.update({"woodbury_identity_error": woodbury_identity_check(),
                   "gaussian_precision_psd_by_construction": True,
                   "orbit_inverse_max": float((inversion - image).abs().max()),
                   "orbit_full_risk_distinct": True,
                   "orbit_full_backward_finite": True})
    return result


def real_p0(models, states, cases, root, deadline):
    if any(not torch.equal(value, models["orbit_lowrank"].state_dict()[name])
           for name, value in models["lowrank_control"].state_dict().items()):
        raise AssertionError("H5 paired source models differ")
    image, label = base.batch(cases, "BMC", 1)
    with torch.no_grad():
        torch.manual_seed(base.stage_seed("BMC") + 1)
        left_field = models["lowrank_control"].deploy_logits(image)
        torch.manual_seed(base.stage_seed("BMC") + 1)
        right_field = models["orbit_lowrank"].deploy_logits(image)
    if not torch.equal(left_field.detach(), right_field.detach()):
        raise AssertionError("H5 paired deployment outputs differ at BMC start")
    torch.manual_seed(base.stage_seed("BMC") + 1)
    left_loss, _ = models["lowrank_control"].loss(image, label)
    torch.manual_seed(base.stage_seed("BMC") + 1)
    right_loss, _ = HierarchicalSegmenter.loss(models["orbit_lowrank"], image, label)
    if not torch.equal(left_loss.detach(), right_loss.detach()):
        raise AssertionError("H5 paired original objectives differ at BMC start")
    del left_loss, right_loss
    clone = copy.deepcopy(models["orbit_lowrank"])
    state = LowRankOnlineGaussian(clone, states["orbit_lowrank"].precision,
                                  states["orbit_lowrank"].basis)
    state.transition()
    torch.manual_seed(base.stage_seed("BMC") + 1)
    loss, _ = clone.loss(image, label)
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(clone.parameters(), 5.0))
    if not math.isfinite(norm):
        raise FloatingPointError("H5 full-risk gradient is non-finite")
    curve = balanced_curvature(clone, image, base.stage_seed("BMC") + 500_001, 2)
    state.update(PIXELS * len(cases) / 6000, curve)
    optimizer = torch.optim.AdamW(clone.recognition_parameters(), lr=3e-4, weight_decay=1e-5)
    optimizer.step()
    if any(not torch.isfinite(value).all() for value in clone.parameters()):
        raise FloatingPointError("H5 discarded golden update is non-finite")
    data.write(root / "P0_REAL.json", {"status": "PASS", "same_source_exact": True,
                                       "same_first_batch_deployment_exact": True,
                                       "same_first_batch_original_risk_exact": True,
                                       "orbit_risk_finite": True,
                                       "balanced_curvature_finite": True,
                                       "woodbury_step_finite": True,
                                       "source_rank": states["orbit_lowrank"].basis.shape[1],
                                       "discarded_engineering_updates": 1})
    base.guard(root, deadline)


@torch.no_grad()
def orbit_stress(model, cases, site, arm, root):
    """Frozen current-train acquisition stress check; no old-site cases."""
    was_training = model.training
    model.eval()
    scores = []
    try:
        for case in cases:
            salt = int.from_bytes(hashlib.sha256(
                f"{SEED}:{site}:{case['id']}".encode()).digest()[:8], "little")
            foreground = case["foreground"]
            if len(foreground) == 0:
                raise ValueError("H5 current patient has no foreground slice")
            indices = (int(foreground[salt % len(foreground)]),
                       int((salt // 7) % len(case["image"])))
            image = torch.stack(tuple(torch.from_numpy(case["image"][i].copy())
                                      for i in indices)).cuda()[:, None]
            label = torch.stack(tuple(torch.from_numpy(case["label"][i].copy())
                                      for i in indices)).cuda().bool()
            generator = torch.Generator(device=image.device).manual_seed(salt % (2**63))
            raw = 2 * torch.rand((2, 6), device=image.device, generator=generator) - 1
            transformed = model.forward_appearance(image, raw)
            probability = model.predict_probability(transformed, salt % (2**63))[:, 0]
            prediction = probability >= .5
            tp = (prediction & label).sum().float()
            dice = (2 * tp + 1) / (prediction.sum() + label.sum() + 1)
            scores.append(float(dice) * 100)
    finally:
        model.train(was_training)
    result = {"site": site, "arm": arm, "patients": len(cases),
              "slices_per_patient": 2, "Dice_mean_percent": float(np.mean(scores)),
              "Dice_p10_percent": float(np.quantile(scores, .1)),
              "fixed_current_train_only": True}
    data.write(root / f"{arm}_{site}_ORBIT_STRESS.json", result)
    return result


def run(settings):
    os.umask(0o077)
    root = Path(settings["output"])
    if not root.is_dir() or (root / "STARTED.json").exists():
        raise FileExistsError("H5 root must be create-only and unused")
    deadline = float(settings["started_unix"]) + 23 * 3600
    if time.time() >= deadline:
        raise TimeoutError("H5 original safety clock expired")
    base.SEED = SEED
    base.MAX_BYTES = 44_000_000_000  # plus immutable N1/H1/H2/H3/H4 stays below 50e9
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
        raise ValueError("H5 data boundary mismatch")
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
            raise AssertionError("H5 U/C backbone initial states differ")
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
        orbit = OrbitSegmenter().cuda()
        orbit.load_state_dict(source_model.state_dict())
        orbit.eval()
        models = {"U-FT": backbone, "lowrank_control": copy.deepcopy(source_model),
                  "orbit_lowrank": orbit}
        del source_model
        states = {arm: LowRankOnlineGaussian(models[arm], lowrank_diagonal, source_basis)
                  for arm in ARMS[1:]}
        for site in SITES[1:]:
            current = base.current_cases(catalog, settings["data_root"], site)
            if site == "BMC":
                real_p0(models, states, current, root, deadline)
            for arm in ARMS[1:]:
                states[arm].begin_site()
            for arm in ARMS:
                base.guard(root, deadline)
                base.curvature = balanced_curvature
                state = states.get(arm)
                if state is not None and state.transition() != 0:
                    raise AssertionError("H5 F0 transition must have zero process noise")
                stage = root / f"{arm}_{site}"
                stage.mkdir(exist_ok=False)
                family = "U" if arm == "U-FT" else "H"
                base.evaluate(settings, root, stage, models[arm], family, 0, "val",
                              SITES[:SITES.index(site) + 1], site, deadline)
                base.train_stage(models[arm], current, site, family, arm, 6000, stage,
                                 state, root, deadline, runtime, settings)
                updates += 6000
            for arm in ARMS[1:]:
                orbit_stress(models[arm], current, site, arm, root)
                base.guard(root, deadline)
            if site == "BMC":
                for arm in ARMS[1:]:
                    audit = states[arm].add_current_site_basis(models[arm], current,
                                                               site, SEED)
                    data.write(root / f"BMC_{arm}_LOW_RANK_AUDIT.json", audit)
                    base.save_new(root / f"BMC_{arm}_CURRENT_STATE_WITH_BASIS.pt",
                                  {"model": models[arm].state_dict(),
                                   "posterior": states[arm].snapshot(), "site": site})
                    base.guard(root, deadline)
            del current
        if updates != UPDATES:
            raise AssertionError("H5 formal update count mismatch")
        data.write(root / "STATUS.json", {"status": "COMPLETE", "updates": updates,
                                          "stages": 8, "elapsed_seconds": time.time() - settings["started_unix"],
                                          "bytes": data.bytes_used(root),
                                          "old_train_replay": 0, "old_teacher": 0,
                                          "final_rank": {arm: states[arm].basis.shape[1]
                                                         for arm in ARMS[1:]}})
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
