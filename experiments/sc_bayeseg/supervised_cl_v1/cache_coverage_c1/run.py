"""One fixed-seed U-Net control for the replay-cache coverage question."""
import copy
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from sc_bayeseg.supervised_cl_v1 import run as core


SEED = 20261026
SITES = core.SITES
ARMS = ("prefix", "coverage")


def coverage_keep(old_items, quota):
    """Select from retained items only, spreading each patient's ordered slices."""
    patients = list(dict.fromkeys(item["patient"] for item in old_items))
    if len(patients) != 4 or quota < 4:
        raise ValueError("coverage selection needs four retained patients")
    by_patient = [[item for item in old_items if item["patient"] == patient]
                  for patient in patients]
    base, extra = divmod(quota, 4)
    chosen = []
    for index, items in enumerate(by_patient):
        count = base + int(index < extra)
        if count > len(items):
            raise ValueError("cannot recover a slice that left the old cache")
        positions = [round((j + .5) * len(items) / count - .5) for j in range(count)]
        if len(set(positions)) != count:
            raise ValueError("coverage selector duplicated an old slice")
        chosen.append([items[position] for position in positions])
    result = [chosen[patient][position]
              for position in range(max(map(len, chosen)))
              for patient in range(4) if position < len(chosen[patient])]
    if len(result) != quota or not {id(item) for item in result} <= {id(item) for item in old_items}:
        raise ValueError("coverage selector accessed outside retained cache")
    return result


def update_buffer(buffer, current, site, old_sites, arm):
    quotas = {old: 32 // len(old_sites) + int(i < 32 % len(old_sites))
              for i, old in enumerate(old_sites)}
    updated = []
    for old in old_sites:
        if old == site:
            chosen = core.replay_slices(current[site], site, quotas[site], SEED)
        else:
            old_items = [item for item in buffer if item["site"] == old]
            chosen = (old_items[:quotas[old]] if arm == "prefix"
                      else coverage_keep(old_items, quotas[old]))
        if len(chosen) != quotas[old] or len({item["patient"] for item in chosen}) != 4:
            raise ValueError("fixed replay quota or patient coverage failed")
        updated.extend(chosen)
    if len(updated) != 32:
        raise ValueError("fixed replay capacity exceeded")
    return updated, quotas


def probe(source_state, current, buffer, root):
    left, right = core.model("U"), core.model("U")
    left.load_state_dict(source_state)
    right.load_state_dict(source_state)
    if any(not torch.equal(left.state_dict()[key], value)
           for key, value in right.state_dict().items()):
        raise ValueError("initial paired source states differ")
    seed = SEED + sum(ord(c) for c in "BMC") * 10000
    x1, y1 = core.sample_batch(current, buffer, 1, seed)
    x1, y1 = core.augment(x1, y1, 1, seed)
    x2, y2 = core.sample_batch(current, buffer, 1, seed)
    x2, y2 = core.augment(x2, y2, 1, seed)
    if not torch.equal(x1, x2) or not torch.equal(y1, y2):
        raise ValueError("initial patient/augmentation stream differs")
    loss1, loss2 = core.loss_fn(left(x1), y1), core.loss_fn(right(x2), y2)
    if not torch.equal(loss1.detach(), loss2.detach()) or not torch.isfinite(loss1):
        raise FloatingPointError("paired U-Net production loss mismatch")
    loss1.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in left.parameters()):
        raise FloatingPointError("U-Net production backward failed")
    retained = coverage_keep(buffer, 16)
    if len(retained) != 16 or len({item["patient"] for item in retained}) != 4:
        raise ValueError("coverage selector P0 failed")
    write_initial = {"status": "PASS", "parameters": sum(p.numel() for p in left.parameters()),
                     "source_backbone_exact": True, "paired_initial_input_exact": True,
                     "paired_task_exact": True, "full_backward_finite": True,
                     "coverage_from_retained_only": True, "optimizer_updates": 0}
    core.write(root / "P0.json", write_initial)
    del left, right


def run(settings):
    os.umask(0o077)
    root = Path(settings["output"])
    if not root.is_dir() or (root / "STARTED.json").exists():
        raise FileExistsError("C1 root must be new and unused")
    deadline = float(settings["started_unix"]) + 23 * 3600
    if time.time() >= deadline:
        raise TimeoutError("C1 original clock expired")
    core.SEED = SEED
    core.setup()
    torch.cuda.set_device(int(settings["gpu_index"]))
    catalog = json.loads(Path(settings["catalog"]).read_text())
    if (tuple(catalog["sites"]) != SITES or catalog["scope"] != "all_train"
            or catalog["test_case_access"] != 0 or catalog["patient_overlap"]):
        raise ValueError("C1 data scope mismatch")
    core.write(root / "STARTED.json", {"started_unix": settings["started_unix"],
                                       "seed": SEED, "sites": list(SITES), "arms": list(ARMS),
                                       "updates_planned": 106000, "test_access": 0})
    active = {"stage": "P0"}
    runtime = {}
    try:
        core.guard(root, deadline)
        torch.manual_seed(SEED)
        net = core.model("U")
        initial_state = copy.deepcopy(net.state_dict())
        source = core.load_cases(settings["data_root"], catalog, ("RUNMC",))
        active["stage"] = "source"
        stage = core.train(net, source, None, 8000, "source", root, deadline, runtime, "U")
        core.evaluate(net, settings["data_root"], catalog, ("RUNMC",),
                      stage / "PRIVATE_TRAIN.pt", "U", "train")
        core.evaluate(net, settings["data_root"], catalog, ("RUNMC",),
                      stage / "PRIVATE_VAL.pt", "U")
        source_state = copy.deepcopy(net.state_dict())
        first_buffer = core.replay_slices(source["RUNMC"], "RUNMC", 32, SEED)
        torch.save(first_buffer, root / "PRIVATE_REPLAY_INITIAL.pt")
        del source
        weights = {arm: copy.deepcopy(source_state) for arm in ARMS}
        buffers = {arm: list(first_buffer) for arm in ARMS}
        for arm in ARMS:
            core.append(root / f"REPLAY_AUDIT_{arm}.jsonl",
                        {"next_stage": 2, "slices": 32,
                         "site_quotas": {"RUNMC": 32}, "patients_per_site": 4})
        bmc = core.load_cases(settings["data_root"], catalog, ("BMC",))
        probe(source_state, bmc, first_buffer, root)
        del bmc
        for index, site in enumerate(SITES[1:], start=1):
            current = core.load_cases(settings["data_root"], catalog, (site,))
            for arm in ARMS:
                active["stage"] = f"{arm}_{site}"
                net.load_state_dict(weights[arm])
                torch.manual_seed(SEED + index * 100003)
                stage = core.train(net, current, buffers[arm], 6000, active["stage"], root,
                                   deadline, runtime, "U")
                core.evaluate(net, settings["data_root"], catalog, (site,),
                              stage / "PRIVATE_TRAIN.pt", "U", "train")
                core.evaluate(net, settings["data_root"], catalog, SITES[:index + 1],
                              stage / "PRIVATE_VAL.pt", "U")
                weights[arm] = copy.deepcopy(net.state_dict())
            if index + 1 < len(SITES):
                for arm in ARMS:
                    buffers[arm], quotas = update_buffer(buffers[arm], current, site,
                                                         SITES[:index + 1], arm)
                    tmp = root / f"PRIVATE_REPLAY_{arm}_next.pt"
                    torch.save(buffers[arm], tmp)
                    tmp.replace(root / f"PRIVATE_REPLAY_{arm}.pt")
                    core.append(root / f"REPLAY_AUDIT_{arm}.jsonl",
                                {"next_stage": index + 2, "slices": len(buffers[arm]),
                                 "site_quotas": quotas, "patients_per_site": 4})
            del current
        torch.manual_seed(SEED)
        joint = core.model("U")
        if any(not torch.equal(initial_state[key], value)
               for key, value in joint.state_dict().items()):
            raise ValueError("joint/source initialization differs")
        all_sites = core.load_cases(settings["data_root"], catalog, SITES)
        active["stage"] = "joint"
        stage = core.train(joint, all_sites, None, 38000, "joint", root, deadline, runtime, "U")
        core.evaluate(joint, settings["data_root"], catalog, SITES,
                      stage / "PRIVATE_TRAIN.pt", "U", "train")
        core.evaluate(joint, settings["data_root"], catalog, SITES,
                      stage / "PRIVATE_VAL.pt", "U")
        core.write(root / "STATUS.json", {"status": "COMPLETE", "stages": 12,
                                          "updates": 106000,
                                          "elapsed_seconds": time.time() - settings["started_unix"],
                                          "bytes": core.bytes_used(root)})
        core.write(root / "EXIT.json", {"exit_code": 0})
    except BaseException:
        torch.save({"stage": runtime.get("stage", active["stage"]),
                    "step": runtime.get("step", 0),
                    "model": runtime["model"].state_dict() if runtime else None,
                    "optimizer": runtime["optimizer"].state_dict() if runtime else None,
                    "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()},
                   root / "failure_partial.pt")
        core.write(root / "FAILURE.json", {"stage": runtime.get("stage", active["stage"]),
                                           "step": runtime.get("step", 0),
                                           "traceback": traceback.format_exc()})
        core.write(root / "EXIT.json", {"exit_code": 1})
        raise


if __name__ == "__main__":
    run(json.loads(Path(os.environ["TASK_SETTINGS"]).read_text()))
