"""H1 three-site, no-replay Bayesian-prefix experiment.

Only the current site's authorized train cases enter this process. A separate
read-only evaluator opens validation cases from a sealed checkpoint.
"""

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from sc_bayeseg.supervised_cl_v1 import run as data

from .model import HierarchicalSegmenter, PIXELS, haar, inverse_haar
from .online import OnlineGaussian, curvature, initial_precision, named_prediction_parameters, scalar_conjugate_check


SEED = 20261026
SITES = ("RUNMC", "BMC", "I2CVB")
SOURCE_STEPS = 8000
DOMAIN_STEPS = 6000
ARMS = ("U-FT", "C-FT", "F0", "F_all", "F_A", "F_perm")
CHECKPOINT_STEPS = (0, 1500, 3000, 6000)
MAX_BYTES = 49_000_000_000


def save_new(path, value):
    with Path(path).open("xb") as handle:
        torch.save(value, handle)


def guard(root, deadline):
    if time.time() >= deadline:
        raise TimeoutError("H1 original 23-hour safety deadline reached")
    if data.bytes_used(root) >= MAX_BYTES:
        raise RuntimeError("H1 49 GB storage pre-stop reached")


def current_cases(catalog, data_root, site):
    # Pass only one site's train rows; no prior site's cases survive this call.
    records = catalog["cases"][site]["train"]
    return [data.read_case(data_root, record) for record in records]


def stage_seed(site):
    return SEED + 10000 * sum(ord(character) for character in site)


def batch(cases, site, step):
    image, label = data.sample_batch({site: cases}, None, step, stage_seed(site))
    return data.augment(image, label, step, stage_seed(site) + 187)


def objective(model, image, label, family):
    if family == "U":
        loss = data.loss_fn(model(image), label)
        return loss, {"task": loss.detach()}
    return model.loss(image, label)


def evaluate(settings, root, stage, model, family, step, role, sites, current_site, deadline):
    guard(root, deadline)
    mark = f"{step:04d}_{role}"
    checkpoint = stage / f"eval_{mark}.pt"
    save_new(checkpoint, {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                          "family": family, "step": step})
    output = stage / f"PRIVATE_{mark.upper()}.pt"
    eval_settings = stage / f"EVAL_SETTINGS_{mark}.json"
    data.write(eval_settings, {"catalog": settings["catalog"], "catalog_sha256": settings["catalog_sha256"],
                               "data_root": settings["data_root"], "checkpoint": str(checkpoint),
                               "output": str(output), "role": role, "sites": list(sites),
                               "current_site": current_site, "family": family,
                               "gpu_index": settings["gpu_index"]})
    environment = os.environ.copy()
    environment["TASK_EVAL_SETTINGS"] = str(eval_settings)
    with (stage / "eval.log").open("ab") as log:
        subprocess.run([sys.executable, settings["eval_entry"]], env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    if not output.is_file():
        raise RuntimeError("separate evaluator produced no patient metrics")
    guard(root, deadline)


def train_stage(model, cases, site, family, arm, steps, stage, online_state, root, deadline, runtime, settings):
    stage.mkdir(exist_ok=True)
    if (stage / "STATUS.json").exists():
        raise FileExistsError("H1 stage already entered")
    if online_state is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    else:
        optimizer = torch.optim.AdamW(model.recognition_parameters(), lr=3e-4, weight_decay=1e-5)
    runtime.update(model=model, optimizer=optimizer, online=online_state, stage=f"{arm}_{site}", step=0)
    model.train()
    losses = []
    clip_count = 0
    started = time.perf_counter()
    for step in range(1, steps + 1):
        if step % 50 == 1:
            guard(root, deadline)
        image, label = batch(cases, site, step)
        torch.manual_seed(stage_seed(site) + step)
        optimizer.param_groups[0]["lr"] = 1e-5 + (3e-4 - 1e-5) * .5 * (1 + math.cos(math.pi * (step - 1) / steps))
        model.zero_grad(set_to_none=True)
        loss, parts = objective(model, image, label, family)
        if not torch.isfinite(loss):
            raise FloatingPointError("H1 non-finite full training objective")
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        if not math.isfinite(norm):
            raise FloatingPointError("H1 non-finite gradient")
        clip_count += int(norm > 5)
        if online_state is not None:
            curve = curvature(model, image, stage_seed(site) + step + 500_000, 2)
            alpha = PIXELS * len(cases) / steps
            online_state.update(alpha, curve)
        optimizer.step()
        runtime["step"] = step
        losses.append(float(loss.detach()))
        if step % 100 == 0:
            data.append(stage / "curve.jsonl", {"step": step,
                        "loss_last100": float(np.mean(losses[-100:])),
                        "clip_count": clip_count, "elapsed_seconds": time.perf_counter() - started})
            data.write(stage / "STATUS.json", {"status": "TRAINING", "step": step, "steps": steps})
        if step % 2000 == 0 and step < steps:
            temporary = stage / "recovery_next.pt"
            torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "precision": None if online_state is None else online_state.snapshot(),
                        "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, temporary)
            temporary.replace(stage / "recovery.pt")
        if (site != "RUNMC" and step in CHECKPOINT_STEPS[1:]) or (site == "RUNMC" and step == steps):
            evaluate(settings, root, stage, model, family, step, "val", SITES[:SITES.index(site) + 1],
                     site, deadline)
        if step == steps:
            evaluate(settings, root, stage, model, family, step, "train", (site,), site, deadline)
    save_new(stage / "current.pt", {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                                    "precision": None if online_state is None else online_state.snapshot(),
                                    "arm": arm, "site": site, "step": steps})
    data.write(stage / "STATUS.json", {"status": "TRAINED", "step": steps,
                                       "steps": steps, "loss_first100": float(np.mean(losses[:100])),
                                       "loss_last100": float(np.mean(losses[-100:])),
                                       "clip_fraction": clip_count / steps,
                                       "elapsed_seconds": time.perf_counter() - started})
    return clip_count


def source_precision(model, cases, root, deadline):
    model.eval()
    named = named_prediction_parameters(model)
    average = {name: torch.zeros_like(parameter, dtype=torch.float32) for name, parameter in named}
    started = time.perf_counter()
    calls = 0
    for case in cases:
        guard(root, deadline)
        salt = int.from_bytes(hashlib.sha256(f"{SEED}:{case['id']}".encode()).digest()[:8], "little")
        generator = np.random.default_rng(salt)
        foreground = case["foreground"]
        if len(foreground) == 0:
            raise ValueError("source patient has no foreground slices for precision")
        positions = list(generator.choice(foreground, 4, replace=len(foreground) < 4))
        positions += list(generator.choice(len(case["image"]), 4, replace=len(case["image"]) < 4))
        patient = {name: torch.zeros_like(parameter, dtype=torch.float32) for name, parameter in named}
        for index, position in enumerate(positions):
            image = torch.from_numpy(case["image"][int(position)].copy()).cuda()[None, None]
            image, _ = data.augment(image, torch.zeros_like(image[:, 0], dtype=torch.long),
                                    index + 1, SEED + salt % 1_000_003)
            curve = curvature(model, image, salt % (2**31) + index, 2)
            for name in patient:
                patient[name].add_(curve[name], alpha=1 / 8)
            calls += 1
        for name in average:
            average[name].add_(patient[name], alpha=1 / len(cases))
    beta = PIXELS * len(cases)
    precision = initial_precision(named)
    for name in precision:
        precision[name].add_(average[name], alpha=beta)
    if any(not torch.isfinite(value).all() or (value <= 0).any() for value in precision.values()):
        raise FloatingPointError("source precision is non-finite")
    values = torch.cat(tuple(value.flatten() for value in precision.values()))
    data.write(root / "SOURCE_PRECISION_AUDIT.json",
               {"patients": len(cases), "slices_per_patient": 8, "curvature_calls": calls,
                "probes_per_call": 2, "beta": beta, "seconds": time.perf_counter() - started,
                "precision_quantiles": [float(item) for item in torch.quantile(values,
                                         torch.tensor((0, .1, .5, .9, 1), device=values.device))]})
    return precision


def real_p0(models, states, bmc, root, deadline):
    left, right = models["C-FT"], models["F_A"]
    if any(not torch.equal(value, right.state_dict()[name])
           for name, value in left.state_dict().items()):
        raise AssertionError("C and F_A source states are not identical")
    image, label = batch(bmc, "BMC", 1)
    clone = copy.deepcopy(right)
    state = OnlineGaussian(clone, states["F_A"].precision, "F_A", SEED)
    torch.manual_seed(stage_seed("BMC") + 1)
    l_loss, _ = left.loss(image, label)
    torch.manual_seed(stage_seed("BMC") + 1)
    r_loss, _ = clone.loss(image, label)
    if not torch.equal(l_loss.detach(), r_loss.detach()):
        raise AssertionError("paired real first-batch full objectives differ")
    r_loss.backward()
    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
           for parameter in clone.parameters()):
        raise FloatingPointError("real BMC full backward failed")
    norm = float(torch.nn.utils.clip_grad_norm_(clone.parameters(), 5.0))
    if not math.isfinite(norm):
        raise FloatingPointError("real BMC clipped gradient is non-finite")
    curve = curvature(clone, image, stage_seed("BMC") + 500_001, 2)
    state.transition()
    state.update(PIXELS * len(bmc) / DOMAIN_STEPS, curve)
    optimizer = torch.optim.AdamW(clone.recognition_parameters(), lr=3e-4, weight_decay=1e-5)
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in clone.parameters()):
        raise FloatingPointError("real BMC discarded golden update failed")
    data.write(root / "P0_REAL.json", {"status": "PASS", "paired_source_exact": True,
                                       "paired_first_batch_objective_exact": True,
                                       "full_backward_finite": True, "curvature_finite": True,
                                       "recognition_adamw_step_finite": True,
                                       "discarded_engineering_updates": 1,
                                       "source_prior_init": True,
                                       "appearance_parameters": state.appearance_count,
                                       "prediction_parameters": state.parameter_count})
    del clone
    guard(root, deadline)


def preflight():
    torch.manual_seed(SEED)
    scalar_errors = scalar_conjugate_check()
    x = torch.randn(1, 1, 256, 256, device="cuda") * .3
    c, u, v = haar(x)
    inverse_error = float((inverse_haar(c, u, v) - x).abs().max())
    model = HierarchicalSegmenter().cuda()
    a = torch.randn(1, 6, device="cuda") * .1
    appearance_error = float((model.inverse_appearance(model.forward_appearance(x, a), a) - x).abs().max())
    if inverse_error > 2e-6 or appearance_error > 2e-6:
        raise AssertionError("H1 analytic inverse P0 failed")
    label = torch.zeros(1, 256, 256, dtype=torch.long, device="cuda")
    label[:, 80:180, 80:180] = 1
    loss, _ = model.loss(x, label)
    loss.backward()
    if not torch.isfinite(loss) or any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                                       for parameter in model.parameters()):
        raise FloatingPointError("H1 complete synthetic backward failed")
    groups = (model.appearance_parameters(),
              tuple(parameter for name, parameter in named_prediction_parameters(model)
                    if not name.startswith("appearance_prior.")), model.recognition_parameters())
    ids = [id(parameter) for group in groups for parameter in group]
    if (len(ids) != len(set(ids)) or len(ids) != len(tuple(model.parameters())) or
            any(sum(float(parameter.grad.square().sum()) for parameter in group) <= 0 for group in groups)):
        raise AssertionError("H1 prediction/recognition parameter partition failed")
    named = named_prediction_parameters(model)
    prior = initial_precision(named)
    budgets = {}
    for variant in ("F0", "F_all", "F_A", "F_perm"):
        state = OnlineGaussian(model, prior, variant, SEED)
        budgets[variant] = state.transition()
    if budgets["F0"] != 0 or any(abs(budgets[variant] - 812) > .05
                                   for variant in ("F_all", "F_A", "F_perm")):
        raise AssertionError("H1 fixed process-noise budget failed")
    curve = curvature(model, x, SEED + 13, 2)
    if any(not torch.isfinite(item).all() or (item < 0).any() for item in curve.values()):
        raise FloatingPointError("H1 synthetic curvature failed")
    return {"scalar_errors": scalar_errors, "haar_inverse_max": inverse_error,
            "appearance_inverse_max": appearance_error,
            "prediction_parameters": sum(parameter.numel() for parameter in model.prediction_parameters()),
            "appearance_parameters": sum(parameter.numel() for parameter in model.appearance_parameters()),
            "parameter_partition": True, "curvature_nonnegative": True,
            "relative_variance_budgets": budgets}


def run(settings):
    os.umask(0o077)
    root = Path(settings["output"])
    if not root.is_dir() or (root / "STARTED.json").exists():
        raise FileExistsError("H1 root must be create-only and unused")
    deadline = float(settings["started_unix"]) + 23 * 3600
    if time.time() >= deadline:
        raise TimeoutError("H1 original clock expired")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(int(settings["gpu_index"]))
    synthetic = preflight()  # No patient case has been read yet.
    catalog_bytes = Path(settings["catalog"]).read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != settings["catalog_sha256"]:
        raise ValueError("private train/val catalog changed")
    catalog = json.loads(catalog_bytes)
    if (tuple(catalog["sites"][:3]) != SITES or catalog["scope"] != "all_train" or
            catalog["test_case_access"] != 0 or catalog["patient_overlap"]):
        raise ValueError("H1 frozen split and access scope mismatch")
    data.write(root / "STARTED.json", {"started_unix": settings["started_unix"], "seed": SEED,
                                       "sites": list(SITES), "arms": list(ARMS) + ["F_erase"],
                                       "updates_planned": 94000, "test_access": 0,
                                       "old_train_replay": 0, "old_teacher": 0})
    data.write(root / "P0_SYNTHETIC.json", {"status": "PASS", **synthetic})
    runtime = {"stage": "source", "step": 0}
    formal_updates = 0
    try:
        guard(root, deadline)
        torch.manual_seed(SEED)
        u = data.backbone().cuda()
        torch.manual_seed(SEED + 1)
        c = HierarchicalSegmenter().cuda()
        c.backbone.load_state_dict(u.state_dict())
        if any(not torch.equal(value, c.backbone.state_dict()[key]) for key, value in u.state_dict().items()):
            raise AssertionError("U/C backbone initial states differ")
        source = current_cases(catalog, settings["data_root"], "RUNMC")
        for arm, model, family in (("U-FT", u, "U"), ("C-FT", c, "H")):
            stage = root / f"source_{arm}"
            runtime["stage"] = f"source_{arm}"
            train_stage(model, source, "RUNMC", family, arm, SOURCE_STEPS, stage,
                        None, root, deadline, runtime, settings)
            formal_updates += SOURCE_STEPS
        source_curve = source_precision(c, source, root, deadline)
        del source
        models = {"U-FT": u, "C-FT": c}
        for arm in ARMS[2:]:
            models[arm] = copy.deepcopy(c)
        states = {arm: OnlineGaussian(models[arm], source_curve, arm, SEED) for arm in ARMS[2:]}
        for arm in ARMS[2:]:
            if any(not torch.equal(value, models[arm].state_dict()[key])
                   for key, value in c.state_dict().items()):
                raise AssertionError("filter source branch differs from C source")
        data.write(root / "SOURCE_BRANCH_AUDIT.json", {"source_backbone_exact": True,
                    "conditioned_source_state_exact": True, "filter_count": 4,
                    "source_precision_calls": 8 * len(catalog["cases"]["RUNMC"]["train"]),
                    "appearance_parameters": states["F_A"].appearance_count,
                    "prediction_parameters": states["F_A"].parameter_count})

        for site in SITES[1:]:
            current = current_cases(catalog, settings["data_root"], site)
            if site == "BMC":
                real_p0(models, states, current, root, deadline)
            active_arms = ARMS if site == "BMC" else ARMS + ("F_erase",)
            for arm in active_arms:
                guard(root, deadline)
                model = models[arm]
                family = "U" if arm == "U-FT" else "H"
                state = states.get(arm)
                if state is not None:
                    budget = state.transition()
                    expected = 0 if arm == "F0" else state.appearance_count
                    if abs(budget - expected) > .05:
                        raise AssertionError("unequal fixed relative process-noise budget")
                    data.append(root / "TRANSITIONS.jsonl", {"arm": arm, "site": site,
                                                                "relative_variance_budget": budget,
                                                                "appearance_parameters": state.appearance_count})
                stage = root / f"{arm}_{site}"
                stage.mkdir(exist_ok=False)
                evaluate(settings, root, stage, model, family, 0, "val", SITES[:SITES.index(site) + 1],
                         site, deadline)
                # train_stage uses this stage after the zero-update evaluation.
                train_stage(model, current, site, family, arm, DOMAIN_STEPS, stage,
                            state, root, deadline, runtime, settings)
                formal_updates += DOMAIN_STEPS
            if site == "BMC":
                models["F_erase"] = copy.deepcopy(models["F_A"])
                states["F_erase"] = OnlineGaussian(models["F_erase"], states["F_A"].precision,
                                                   "F_erase", SEED)
                states["F_erase"].erase_precision()
                if any(not torch.equal(value, models["F_erase"].state_dict()[key])
                       for key, value in models["F_A"].state_dict().items()):
                    raise AssertionError("F_A/F_erase branch means differ")
                data.write(root / "ERASE_BRANCH_AUDIT.json", {"same_current_model_mean": True,
                            "precision_reset_to_lambda0": True, "old_teacher": 0})
            del current
        if formal_updates != 94000:
            raise AssertionError("H1 physical update count mismatch")
        data.write(root / "STATUS.json", {"status": "PREFIX_COMPLETE", "updates": formal_updates,
                                          "stages": 2 + 6 + 7,
                                          "elapsed_seconds": time.time() - settings["started_unix"],
                                          "bytes": data.bytes_used(root)})
        data.write(root / "EXIT.json", {"exit_code": 0})
    except BaseException:
        failure = {"stage": runtime.get("stage", "P0"), "step": runtime.get("step", 0),
                   "formal_updates_completed": formal_updates,
                   "model": runtime["model"].state_dict() if "model" in runtime else None,
                   "optimizer": runtime["optimizer"].state_dict() if "optimizer" in runtime else None,
                   "precision": runtime["online"].snapshot() if runtime.get("online") is not None else None,
                   "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}
        torch.save(failure, root / "failure_partial.pt")
        data.write(root / "FAILURE.json", {"stage": failure["stage"], "step": failure["step"],
                                           "formal_updates_completed": formal_updates,
                                           "traceback": traceback.format_exc()})
        data.write(root / "EXIT.json", {"exit_code": 1})
        raise


if __name__ == "__main__":
    run(json.loads(Path(os.environ["TASK_SETTINGS"]).read_text()))
