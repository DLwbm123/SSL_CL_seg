"""Paired no-replay BayeSeg study with an old-domain gradient summary."""
import copy
import hashlib
import json
import math
import os
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from sc_bayeseg.supervised_cl_v1 import run as core


SEED = 20261027
SITES = core.SITES
ARMS = ("control", "protected")
MAX_NEW_DIRECTIONS = 8


def backbone_parameters(net):
    return tuple(net.backbone.parameters())


def flat_parameters(parameters):
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def orthogonality_error(basis):
    identity = torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)
    return float((basis.T @ basis - identity).abs().max()) if basis.shape[1] else 0.0


def projected_step(parameters, before, basis):
    """Project the actual AdamW parameter increment, including weight decay."""
    after = flat_parameters(parameters)
    delta = after - before
    if basis.shape[0] != len(delta):
        raise ValueError("gradient summary does not match backbone parameters")
    correction = basis @ (basis.T @ delta)
    updated = after - correction
    with torch.no_grad():
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.copy_(updated[offset:offset + count].view_as(parameter))
            offset += count
    residual = basis.T @ (updated - before)
    relative = float(residual.abs().max() / (delta.norm() + 1e-12)) if basis.shape[1] else 0.0
    if not math.isfinite(relative) or relative > 2e-4:
        raise FloatingPointError("protected AdamW update is not orthogonal")
    return relative


def add_current_site_basis(net, cases, prior, site):
    """Summarize only the current site's four patients and two slices each."""
    parameters = backbone_parameters(net)
    width = sum(parameter.numel() for parameter in parameters)
    basis = (torch.empty(width, 0, device="cuda") if prior is None else prior)
    if basis.shape[0] != width or basis.shape[1] > 40:
        raise ValueError("invalid old-domain gradient summary")
    chosen = sorted(cases, key=lambda case: hashlib.sha256(
        f"{SEED}:{case['id']}".encode()).digest())[:4]
    if len(chosen) != 4:
        raise ValueError("fewer than four current-site training patients")
    new = []
    was_training = net.training
    net.eval()
    try:
        for case in chosen:
            foreground_slices = case["foreground"]
            if len(foreground_slices) < 2:
                raise ValueError("too few foreground slices for fixed summary")
            positions = [int(foreground_slices[round(frac * (len(foreground_slices) - 1))])
                         for frac in (.25, .75)]
            if positions[0] == positions[1]:
                raise ValueError("fixed summary repeated a slice")
            for index in positions:
                image = torch.from_numpy(case["image"][index].copy()).cuda()[None, None]
                label = torch.from_numpy(case["label"][index].copy()).cuda().bool()
                if not label.any() or label.all():
                    raise ValueError("invalid foreground/background summary mask")
                logits = net.predict_logits(image)
                margin = logits[0, 1] - logits[0, 0]
                contrast = margin[label].mean() - margin[~label].mean()
                gradients = torch.autograd.grad(contrast, parameters)
                vector = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
                original_norm = vector.norm()
                if not torch.isfinite(original_norm) or original_norm <= 0:
                    raise FloatingPointError("non-finite or zero latent gradient")
                for _ in range(2):
                    if basis.shape[1]:
                        vector -= basis @ (basis.T @ vector)
                    if new:
                        local = torch.stack(new, dim=1)
                        vector -= local @ (local.T @ vector)
                norm = vector.norm()
                if torch.isfinite(norm) and norm >= 1e-6 * original_norm:
                    new.append(vector / norm)
        if len(new) < 4 or len(new) > MAX_NEW_DIRECTIONS:
            raise ValueError("current site supplied fewer than four independent directions")
        result = torch.cat((basis, torch.stack(new, dim=1)), dim=1)
        error = orthogonality_error(result)
        if error > 2e-4:
            raise FloatingPointError("gradient summary lost orthogonality")
        return result.detach(), {"site": site, "new_directions": len(new),
                                 "total_directions": result.shape[1],
                                 "orthogonality_error": error,
                                 "summary_forward_backward": 8}
    finally:
        net.train(was_training)


def train(net, current, basis, steps, name, root, deadline, runtime, seed):
    stage = root / name
    stage.mkdir(exist_ok=False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-5)
    runtime.update(model=net, optimizer=optimizer, stage=name, step=0)
    parameters = backbone_parameters(net)
    losses, tasks, auxiliaries = [], [], []
    projection_seconds, max_violation = 0.0, 0.0
    net.train()
    for step in range(1, steps + 1):
        if step % 50 == 1:
            core.guard(root, deadline)
        image, label = core.sample_batch(current, None, step, seed)
        image, label = core.augment(image, label, step, seed)
        optimizer.param_groups[0]["lr"] = 1e-5 + (3e-4 - 1e-5) * .5 * (1 + math.cos(math.pi * (step - 1) / steps))
        optimizer.zero_grad(set_to_none=True)
        loss, task, auxiliary = core.objectives(net, image, label, "B")
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite full BayeSeg objective")
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0))
        if not math.isfinite(norm):
            raise FloatingPointError("non-finite full BayeSeg gradient")
        before = flat_parameters(parameters) if basis is not None else None
        optimizer.step()
        if basis is not None:
            torch.cuda.synchronize()
            started = time.perf_counter()
            violation = projected_step(parameters, before, basis)
            torch.cuda.synchronize()
            projection_seconds += time.perf_counter() - started
            max_violation = max(max_violation, violation)
        runtime["step"] = step
        losses.append(float(loss.detach()))
        tasks.append(float(task.detach()))
        auxiliaries.append(float(auxiliary.detach()))
        if step % 100 == 0:
            core.append(stage / "curve.jsonl", {"step": step,
                        "loss_last100": float(np.mean(losses[-100:])),
                        "task_last100": float(np.mean(tasks[-100:])),
                        "variational_last100": float(np.mean(auxiliaries[-100:])),
                        "gradient_norm": norm})
            core.write(stage / "STATUS.json", {"status": "TRAINING", "step": step, "steps": steps})
        if step % 2000 == 0 and step < steps:
            torch.save({"step": step, "model": net.state_dict(), "optimizer": optimizer.state_dict(),
                        "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()},
                       stage / "recovery.pt")
    torch.save({"step": steps, "model": net.state_dict(), "optimizer": optimizer.state_dict()},
               stage / "endpoint.pt")
    core.write(stage / "STATUS.json", {"status": "TRAINED", "step": steps, "steps": steps,
               "loss_first100": float(np.mean(losses[:100])), "loss_last100": float(np.mean(losses[-100:])),
               "task_last100": float(np.mean(tasks[-100:])),
               "variational_last100": float(np.mean(auxiliaries[-100:])),
               "basis_rank": 0 if basis is None else basis.shape[1],
               "projection_seconds": projection_seconds,
               "max_projection_violation": max_violation})
    return stage


def probe(source_state, current, basis, root):
    left, right = core.model("B"), core.model("B")
    left.load_state_dict(source_state)
    right.load_state_dict(source_state)
    if any(not torch.equal(left.state_dict()[key], value)
           for key, value in right.state_dict().items()):
        raise ValueError("paired source weights differ")
    seed = SEED + 100003
    x1, y1 = core.sample_batch(current, None, 1, seed)
    x1, y1 = core.augment(x1, y1, 1, seed)
    x2, y2 = core.sample_batch(current, None, 1, seed)
    x2, y2 = core.augment(x2, y2, 1, seed)
    if not torch.equal(x1, x2) or not torch.equal(y1, y2):
        raise ValueError("paired first batch differs")
    torch.manual_seed(seed)
    loss1, task1, auxiliary1 = core.objectives(left, x1, y1, "B")
    torch.manual_seed(seed)
    loss2, task2, auxiliary2 = core.objectives(right, x2, y2, "B")
    if (not torch.equal(loss1.detach(), loss2.detach()) or
            not torch.equal(task1.detach(), task2.detach()) or
            not torch.equal(auxiliary1.detach(), auxiliary2.detach())):
        raise ValueError("paired first full objective differs")
    loss2.backward()
    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
           for parameter in right.parameters()):
        raise FloatingPointError("production full backward failed")
    parameters = backbone_parameters(right)
    before = flat_parameters(parameters)
    optimizer = torch.optim.AdamW(right.parameters(), lr=3e-4, weight_decay=1e-5)
    optimizer.step()  # Discard this one engineering golden update.
    violation = projected_step(parameters, before, basis)
    error = orthogonality_error(basis)
    if error > 2e-4:
        raise FloatingPointError("source gradient summary is not orthonormal")
    core.write(root / "P0.json", {"status": "PASS", "paired_source_exact": True,
               "paired_first_batch_exact": True, "paired_full_objective_exact": True,
               "full_backward_finite": True, "golden_engineering_updates_discarded": 1,
               "source_basis_rank": basis.shape[1], "orthogonality_error": error,
               "golden_actual_step_violation": violation,
               "old_train_replay": 0, "old_model_teacher": 0})
    del left, right


def run(settings):
    os.umask(0o077)
    root = Path(settings["output"])
    if not root.is_dir() or (root / "STARTED.json").exists():
        raise FileExistsError("N1 root must be new and unused")
    deadline = float(settings["started_unix"]) + 23 * 3600
    if time.time() >= deadline:
        raise TimeoutError("N1 original clock expired")
    core.SEED = SEED
    core.setup()
    torch.cuda.set_device(int(settings["gpu_index"]))
    catalog_bytes = Path(settings["catalog"]).read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != settings["catalog_sha256"]:
        raise ValueError("private train/val catalog changed")
    catalog = json.loads(catalog_bytes)
    if (tuple(catalog["sites"]) != SITES or catalog["scope"] != "all_train"
            or catalog["test_case_access"] != 0 or catalog["patient_overlap"]):
        raise ValueError("N1 frozen data scope mismatch")
    core.write(root / "STARTED.json", {"started_unix": settings["started_unix"],
               "seed": SEED, "sites": list(SITES), "arms": list(ARMS),
               "updates_planned": 68000, "old_train_replay": 0, "old_model_teacher": 0,
               "test_access": 0})
    active, runtime = {"stage": "P0"}, {}
    try:
        core.guard(root, deadline)
        torch.manual_seed(SEED)
        net = core.model("B")
        synthetic_image = torch.zeros(2, 1, 256, 256, device="cuda")
        synthetic_label = torch.zeros(2, 256, 256, dtype=torch.long, device="cuda")
        synthetic_label[:, 80:180, 80:180] = 1
        loss, _, _ = core.objectives(net, synthetic_image, synthetic_label, "B")
        loss.backward()
        if not torch.isfinite(loss) or any(parameter.grad is None or
                not torch.isfinite(parameter.grad).all() for parameter in net.parameters()):
            raise FloatingPointError("synthetic full BayeSeg objective failed")
        torch.manual_seed(SEED)
        net = core.model("B")
        source = core.load_cases(settings["data_root"], catalog, (SITES[0],))
        active["stage"] = "source"
        torch.manual_seed(SEED)
        stage = train(net, source, None, 8000, "source", root, deadline, runtime, SEED)
        core.evaluate(net, settings["data_root"], catalog, (SITES[0],),
                      stage / "PRIVATE_TRAIN.pt", "B", "train")
        core.evaluate(net, settings["data_root"], catalog, (SITES[0],),
                      stage / "PRIVATE_VAL.pt", "B")
        source_state = copy.deepcopy(net.state_dict())
        basis, audit = add_current_site_basis(net, source[SITES[0]], None, SITES[0])
        torch.save(basis.cpu(), root / "PRIVATE_BASIS_RUNMC.pt")
        core.append(root / "BASIS_AUDIT.jsonl", audit)
        del source
        weights = {arm: copy.deepcopy(source_state) for arm in ARMS}
        bmc = core.load_cases(settings["data_root"], catalog, (SITES[1],))
        active["stage"] = "P0_BMC"
        probe(source_state, bmc, basis, root)
        del bmc, source_state
        for index, site in enumerate(SITES[1:], start=1):
            current = core.load_cases(settings["data_root"], catalog, (site,))
            for arm in ARMS:
                active["stage"] = f"{arm}_{site}"
                net.load_state_dict(weights[arm])
                torch.manual_seed(SEED + index * 100003)
                stage = train(net, current, basis if arm == "protected" else None,
                              6000, active["stage"], root, deadline, runtime,
                              SEED + index * 100003)
                core.evaluate(net, settings["data_root"], catalog, (site,),
                              stage / "PRIVATE_TRAIN.pt", "B", "train")
                core.evaluate(net, settings["data_root"], catalog, SITES[:index + 1],
                              stage / "PRIVATE_VAL.pt", "B")
                weights[arm] = copy.deepcopy(net.state_dict())
                if arm == "protected" and index + 1 < len(SITES):
                    basis, audit = add_current_site_basis(net, current[site], basis, site)
                    torch.save(basis.cpu(), root / f"PRIVATE_BASIS_{site}.pt")
                    core.append(root / "BASIS_AUDIT.jsonl", audit)
            del current
        core.write(root / "STATUS.json", {"status": "COMPLETE", "stages": 11,
               "updates": 68000, "elapsed_seconds": time.time() - settings["started_unix"],
               "bytes": core.bytes_used(root), "old_train_replay": 0, "old_model_teacher": 0})
        core.write(root / "EXIT.json", {"exit_code": 0})
    except BaseException:
        torch.save({"stage": runtime.get("stage", active["stage"]),
                    "step": runtime.get("step", 0),
                    "model": runtime["model"].state_dict() if runtime else None,
                    "optimizer": runtime["optimizer"].state_dict() if runtime else None,
                    "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()},
                   root / "failure_partial.pt")
        core.write(root / "FAILURE.json", {"stage": runtime.get("stage", active["stage"]),
                   "step": runtime.get("step", 0), "traceback": traceback.format_exc()})
        core.write(root / "EXIT.json", {"exit_code": 1})
        raise


def terminate(_signal, _frame):
    raise TimeoutError("external experiment timeout requested")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminate)
    run(json.loads(Path(os.environ["TASK_SETTINGS"]).read_text()))
