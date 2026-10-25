"""Paired BayeSeg ER and posterior-preserving replay from one frozen source."""
import copy
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from sc_bayeseg.supervised_cl_v1.run import (SEED, SITES, augment, bytes_used,
                                              evaluate, guard, load_cases, loss_fn,
                                              model, replay_slices, sample_batch,
                                              setup, write, append)


STEPS = 6000
ARMS = ("er", "ppr")


def posterior(net, image, logits=None):
    if logits is None:
        logits = net.predict_logits(image)
    mean = logits[:, 1:2] - logits[:, 0:1]
    logsd = -4 + 3 * torch.tanh(net.logit_scale(logits) / 3)
    return mean, logsd


def posterior_loss(teacher, student, image, student_logits):
    with torch.no_grad():
        old_mean, old_logsd = posterior(teacher, image)
    new_mean, new_logsd = posterior(student, image, student_logits)
    kl = (new_logsd - old_logsd
          + .5 * (torch.exp(2 * (old_logsd - new_logsd))
                  + ((old_mean - new_mean) / new_logsd.exp()).square()) - .5)
    return torch.log1p(kl.clamp_min(0)).mean()


def objectives(net, teacher, x, y, arm):
    logits, variational = net(x)
    task = loss_fn(logits, y)
    retention = (posterior_loss(teacher, net, x[2:], logits[2:]) if arm == "ppr"
                 else torch.zeros_like(task))
    return task + .01 * variational + .1 * retention, task, variational, retention


def update_buffer(buffer, current, site, old_sites):
    quotas = {old: 32 // len(old_sites) + int(i < 32 % len(old_sites))
              for i, old in enumerate(old_sites)}
    updated = []
    for old in old_sites:
        items = ([item for item in buffer if item["site"] == old][:quotas[old]]
                 if old != site else replay_slices(current[site], site, quotas[site], SEED))
        if len(items) != quotas[old] or len({item["patient"] for item in items}) != 4:
            raise ValueError("fixed replay site quota or patient coverage failed")
        updated.extend(items)
    if len(updated) != 32:
        raise ValueError("fixed replay buffer exceeded 32 slices")
    return updated, quotas


def first_batch_probe(source_state, current, buffer, root):
    er, ppr, teacher = model("B"), model("B"), model("B")
    for net in (er, ppr, teacher):
        net.load_state_dict(source_state)
    teacher.eval()
    seed = SEED + sum(ord(c) for c in "BMC") * 10000
    x, y = sample_batch(current, buffer, 1, seed)
    x, y = augment(x, y, 1, seed)
    torch.manual_seed(SEED + 100003 + 303)
    loss_er, task_er, var_er, _ = objectives(er, None, x, y, "er")
    torch.manual_seed(SEED + 100003 + 303)
    loss_ppr, task_ppr, var_ppr, retention = objectives(ppr, teacher, x, y, "ppr")
    if not torch.equal(task_er.detach(), task_ppr.detach()) or not torch.equal(var_er.detach(), var_ppr.detach()):
        raise ValueError("P1 initial paired task/generative forward mismatch")
    if abs(float(retention.detach())) > 1e-6 or not torch.isfinite(loss_ppr):
        raise ValueError("P1 teacher/student initial posterior mismatch")
    gradients = torch.autograd.grad(var_ppr, ppr.backbone.parameters(), retain_graph=True,
                                    allow_unused=True)
    auxiliary_norm = float(sum(g.square().sum() for g in gradients if g is not None).sqrt())
    loss_ppr.backward()
    if (not math.isfinite(auxiliary_norm) or auxiliary_norm <= 0
            or any(p.grad is None or not torch.isfinite(p.grad).all() for p in ppr.parameters())):
        raise FloatingPointError("P1 full objective/backbone gradient admission failed")
    write(root / "P0.json", {"status": "PASS", "source_backbone_exact": True,
                             "paired_task_exact": True, "paired_variational_exact": True,
                             "initial_retention": float(retention.detach()),
                             "auxiliary_backbone_gradient_norm": auxiliary_norm,
                             "full_backward_finite": True, "updates": 0})
    del er, ppr, teacher


def train_stage(net, teacher, current, buffer, arm, site, stage, root, deadline):
    started = time.time()
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-5)
    seed = SEED + sum(ord(c) for c in site) * 10000
    index = SITES.index(site)
    torch.manual_seed(SEED + index * 100003 + 303)
    net.train()
    if teacher is not None:
        teacher.eval()
    losses, tasks, variational, retention = [], [], [], []
    for step in range(1, STEPS + 1):
        if step % 50 == 1:
            guard(root, deadline)
        x, y = sample_batch(current, buffer, step, seed)
        x, y = augment(x, y, step, seed)
        optimizer.param_groups[0]["lr"] = 1e-5 + (3e-4 - 1e-5) * .5 * (1 + math.cos(math.pi * (step - 1) / STEPS))
        optimizer.zero_grad(set_to_none=True)
        loss, task, auxiliary, keep = objectives(net, teacher, x, y, arm)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite P1 loss")
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 5.))
        if not math.isfinite(norm):
            raise FloatingPointError("non-finite P1 gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
        tasks.append(float(task.detach()))
        variational.append(float(auxiliary.detach()))
        retention.append(float(keep.detach()))
        if step % 100 == 0:
            write(stage / "STATUS.json", {"status": "TRAINING", "step": step, "steps": STEPS})
            append(stage / "curve.jsonl", {"step": step, "loss_last100": float(np.mean(losses[-100:])),
                                           "task_last100": float(np.mean(tasks[-100:])),
                                           "variational_last100": float(np.mean(variational[-100:])),
                                           "retention_last100": float(np.mean(retention[-100:])),
                                           "gradient_norm": norm})
        if step % 2000 == 0 and step < STEPS:
            torch.save({"step": step, "model": net.state_dict(), "optimizer": optimizer.state_dict(),
                        "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, stage / "recovery.pt")
    torch.save({"step": STEPS, "model": net.state_dict(), "optimizer": optimizer.state_dict()},
               stage / "endpoint.pt")
    write(stage / "STATUS.json", {"status": "TRAINED", "step": STEPS, "steps": STEPS,
                                  "train_elapsed_seconds": time.time() - started,
                                  "loss_first100": float(np.mean(losses[:100])),
                                  "loss_last100": float(np.mean(losses[-100:])),
                                  "task_last100": float(np.mean(tasks[-100:])),
                                  "variational_last100": float(np.mean(variational[-100:])),
                                  "retention_last100": float(np.mean(retention[-100:]))})


def run(settings):
    os.umask(0o077)
    root = Path(settings["output"])
    if not root.is_dir() or (root / "STARTED.json").exists():
        raise FileExistsError("P1 root must be new and unused")
    deadline = float(settings["started_unix"]) + 23 * 3600
    if time.time() >= deadline:
        raise TimeoutError("P1 original clock expired")
    setup()
    torch.cuda.set_device(int(settings["gpu_index"]))
    source_root = Path(settings["source_root"])
    catalog = json.loads((source_root / "PRIVATE_CATALOG.json").read_text())
    if (tuple(catalog["sites"]) != SITES or catalog["scope"] != "all_train"
            or catalog["test_case_access"] != 0 or catalog["patient_overlap"]):
        raise ValueError("P1 training scope mismatch")
    checkpoint = torch.load(source_root / "source" / "endpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint["step"] != 8000:
        raise ValueError("P1 source checkpoint mismatch")
    source_state = checkpoint["model"]
    write(root / "STARTED.json", {"started_unix": settings["started_unix"], "arms": list(ARMS),
                                  "source_updates_read_only": 8000, "new_updates_planned": 60000,
                                  "test_access": 0})
    active = {"stage": "P0", "step": 0}
    live = {}
    try:
        guard(root, deadline)
        source = load_cases(settings["data_root"], catalog, ("RUNMC",))
        first_buffer = replay_slices(source["RUNMC"], "RUNMC", 32, SEED)
        bmc = load_cases(settings["data_root"], catalog, ("BMC",))
        first_batch_probe(source_state, bmc, first_buffer, root)
        del bmc, source
        for arm in ARMS:
            net = model("B")
            net.load_state_dict(source_state)
            buffer = list(first_buffer)
            arm_root = root / arm
            arm_root.mkdir(exist_ok=False)
            for index, site in enumerate(SITES[1:], start=1):
                current = load_cases(settings["data_root"], catalog, (site,))
                stage = arm_root / site
                stage.mkdir(exist_ok=False)
                active.update(stage=f"{arm}_{site}", step=0)
                teacher = copy.deepcopy(net) if arm == "ppr" else None
                live.update(model=net, teacher=teacher, stage=active["stage"])
                train_stage(net, teacher, current, buffer, arm, site, stage, root, deadline)
                active["step"] = STEPS
                evaluate(net, settings["data_root"], catalog, (site,), stage / "PRIVATE_TRAIN.pt", "B", "train")
                evaluate(net, settings["data_root"], catalog, SITES[:index + 1], stage / "PRIVATE_VAL.pt", "B")
                if index + 1 < len(SITES):
                    buffer, quotas = update_buffer(buffer, current, site, SITES[:index + 1])
                    tmp = arm_root / "PRIVATE_REPLAY_next.pt"
                    torch.save(buffer, tmp)
                    tmp.replace(arm_root / "PRIVATE_REPLAY.pt")
                    append(arm_root / "REPLAY_AUDIT.jsonl", {"next_stage": index + 2,
                                                              "slices": len(buffer), "site_quotas": quotas,
                                                              "patients_per_site": 4})
                del current, teacher
            del net, buffer
        write(root / "STATUS.json", {"status": "COMPLETE", "updates": 60000,
                                     "stages": 10, "elapsed_seconds": time.time() - settings["started_unix"],
                                     "bytes": bytes_used(root)})
        write(root / "EXIT.json", {"exit_code": 0})
    except BaseException:
        torch.save({"stage": active["stage"], "step": active["step"],
                    "model": live["model"].state_dict() if live else None}, root / "failure_partial.pt")
        write(root / "FAILURE.json", {"stage": active["stage"], "step": active["step"],
                                      "traceback": traceback.format_exc()})
        write(root / "EXIT.json", {"exit_code": 1})
        raise


if __name__ == "__main__":
    run(json.loads(Path(os.environ["TASK_SETTINGS"]).read_text()))
