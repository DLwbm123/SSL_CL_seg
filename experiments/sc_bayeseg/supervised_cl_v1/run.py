"""A fixed, fully labeled three-site prostate continual-learning comparison."""
import copy
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import torch
from monai.networks.nets import BasicUNet
from torch.nn import functional as F

from sc_bayeseg.sslcl_v1.prostate_metrics import binary_phase_metrics


SITES = ("RUNMC", "BMC", "I2CVB")
STEPS = (8000, 6000, 6000)
SEED = 20261025
BUFFER_SLICES = 32
MAX_BYTES = 49_000_000_000  # Reserve 1 GB below the strict 50 GB limit.


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def append(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def setup():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def model():
    return BasicUNet(spatial_dims=2, in_channels=1, out_channels=2,
                     features=(32, 32, 64, 128, 256, 32)).cuda()


def read_case(data_root, record):
    h5 = Path(data_root) / "h5" / "v1"
    with h5py.File(h5 / record["image"], "r") as handle:
        image = np.asarray(handle["image"][...], dtype=np.float32)
        spacing = json.loads(handle.attrs["processed_spacing"])
    with h5py.File(h5 / record["label"], "r") as handle:
        label = np.asarray(handle["label"][...], dtype=np.uint8)
    if image.shape != label.shape or image.ndim != 3 or image.shape[1:] != (256, 256):
        raise ValueError("unexpected prostate volume geometry")
    if not np.isfinite(image).all() or not np.isin(label, (0, 1)).all():
        raise ValueError("non-finite image or non-binary label")
    foreground = np.flatnonzero(label.reshape(len(label), -1).any(axis=1))
    return {"id": record["patient_id"], "site": record["site"], "image": image,
            "label": label, "spacing": spacing, "foreground": foreground}


def load_cases(data_root, catalog, sites, role="train"):
    return {site: [read_case(data_root, row) for row in catalog["cases"][site][role]] for site in sites}


def sample_one(cases, rng):
    case = cases[int(rng.integers(len(cases)))]
    choices = case["foreground"] if len(case["foreground"]) and rng.random() < .5 else range(len(case["image"]))
    index = int(choices[int(rng.integers(len(choices)))])
    return case["image"][index], case["label"][index]


def sample_batch(current, old, step, seed):
    rng = np.random.default_rng(seed + 100003 * step)
    images, labels = [], []
    for j in range(4):
        if old and j >= 2:
            item = old[int(rng.integers(len(old)))]
            image, label = item["image"], item["label"]
        else:
            site = list(current)[int(rng.integers(len(current)))]
            image, label = sample_one(current[site], rng)
        images.append(image)
        labels.append(label)
    x = torch.from_numpy(np.stack(images).copy()).float().cuda().unsqueeze(1)
    y = torch.from_numpy(np.stack(labels).copy()).long().cuda()
    return x, y


def augment(x, y, step, seed):
    rng = torch.Generator(device=x.device).manual_seed(seed + step)
    batch = len(x)
    angle = (torch.rand(batch, device=x.device, generator=rng) - .5) * (math.pi / 6)
    scale = .9 + .2 * torch.rand(batch, device=x.device, generator=rng)
    shift = (torch.rand(batch, 2, device=x.device, generator=rng) - .5) * .1
    theta = torch.zeros(batch, 2, 3, device=x.device)
    theta[:, 0, 0] = scale * angle.cos()
    theta[:, 0, 1] = -scale * angle.sin()
    theta[:, 1, 0] = scale * angle.sin()
    theta[:, 1, 1] = scale * angle.cos()
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0].long()
    for axis in (-1, -2):
        mask = torch.rand(batch, device=x.device, generator=rng) < .5
        x[mask] = x[mask].flip(axis)
        y[mask] = y[mask].flip(axis)
    gamma = .8 + .4 * torch.rand(batch, 1, 1, 1, device=x.device, generator=rng)
    gain = .9 + .2 * torch.rand(batch, 1, 1, 1, device=x.device, generator=rng)
    bias = (torch.rand(batch, 1, 1, 1, device=x.device, generator=rng) - .5) * .1
    x = x.sign() * x.abs().clamp_min(1e-6).pow(gamma) * gain + bias
    return x, y


def loss_fn(logits, y):
    # CUDA nll_loss2d has no deterministic kernel in the production runtime.
    ce = -logits.log_softmax(1).gather(1, y[:, None]).mean()
    p = logits.softmax(1)[:, 1]
    target = (y == 1).float()
    dice = (2 * (p * target).sum((1, 2)) + 1) / (p.sum((1, 2)) + target.sum((1, 2)) + 1)
    return ce + (1 - dice).mean()


def replay_slices(cases, site, num, seed):
    if num not in (16, 32) or len(cases) < 4:
        raise ValueError("fixed replay requires four source patients and 16 or 32 slices")
    ordered = sorted(cases, key=lambda c: hashlib.sha256(f"{seed}:{c['id']}".encode()).digest())
    selected = []
    per_patient = num // 4
    for case in ordered[:4]:
        choices = case["foreground"] if len(case["foreground"]) >= per_patient else np.arange(len(case["image"]))
        if len(choices) < per_patient:
            raise ValueError("patient has too few slices for fixed replay")
        indices = [int(choices[round((j + .5) * len(choices) / per_patient - .5)]) for j in range(per_patient)]
        if len(set(indices)) != per_patient:
            raise ValueError("replay selection repeated a slice")
        selected.extend({"site": site, "patient": case["id"], "image": case["image"][i].astype(np.float16),
                         "label": case["label"][i]} for i in indices)
    if len(selected) != num:
        raise ValueError("insufficient patients for fixed replay allocation")
    return selected


def evaluate(net, data_root, catalog, sites, output):
    prior = net.training
    net.eval()
    result = {}
    with torch.no_grad():
        for site in sites:
            rows = []
            for record in catalog["cases"][site]["val"]:
                case = read_case(data_root, record)
                predictions = []
                for offset in range(0, len(case["image"]), 4):
                    image = torch.from_numpy(case["image"][offset:offset + 4].copy()).cuda()[:, None]
                    predictions.append(net(image).softmax(1).cpu())
                metric = binary_phase_metrics(torch.cat(predictions), case["label"], case["spacing"])
                rows.append({"patient": case["id"], "metrics": metric})
            result[site] = rows
    torch.save(result, output)
    net.train(prior)
    return {site: len(result[site]) for site in sites}


def bytes_used(root):
    return int(subprocess.check_output(["du", "-sb", str(root)], text=True).split()[0])


def guard(root, deadline):
    if time.time() >= deadline:
        raise TimeoutError("experiment safety deadline reached")
    if bytes_used(root) >= MAX_BYTES:
        raise RuntimeError("experiment storage pre-stop threshold reached")


def train(net, current, replay, steps, name, root, deadline, runtime):
    stage = root / name
    stage.mkdir(exist_ok=False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-5)
    runtime.update(model=net, optimizer=optimizer, stage=name, step=0)
    seed = SEED + sum(ord(c) for c in name.split("_")[-1]) * 10000
    net.train()
    losses = []
    for step in range(1, steps + 1):
        if step % 50 == 1:
            guard(root, deadline)
        x, y = sample_batch(current, replay, step, seed)
        x, y = augment(x, y, step, seed)
        optimizer.param_groups[0]["lr"] = 1e-5 + (3e-4 - 1e-5) * .5 * (1 + math.cos(math.pi * (step - 1) / steps))
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(net(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite supervised loss")
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0))
        if not math.isfinite(norm):
            raise FloatingPointError("non-finite gradient")
        optimizer.step()
        runtime["step"] = step
        losses.append(float(loss.detach()))
        if step % 100 == 0:
            append(stage / "curve.jsonl", {"step": step, "loss_last100": float(np.mean(losses[-100:])),
                                           "gradient_norm": norm})
            write(stage / "STATUS.json", {"status": "TRAINING", "step": step, "steps": steps})
        if step % 2000 == 0 and step < steps:
            torch.save({"step": step, "model": net.state_dict(), "optimizer": optimizer.state_dict(),
                        "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, stage / "recovery.pt")
    torch.save({"step": steps, "model": net.state_dict(), "optimizer": optimizer.state_dict()}, stage / "endpoint.pt")
    write(stage / "STATUS.json", {"status": "TRAINED", "step": steps, "steps": steps,
                                  "loss_first100": float(np.mean(losses[:100])),
                                  "loss_last100": float(np.mean(losses[-100:]))})
    return stage


def run(settings):
    os.umask(0o077)
    root = Path(settings["output"])
    if not root.is_dir() or (root / "STARTED.json").exists():
        raise FileExistsError("experiment root must exist and be unused")
    data_root = settings["data_root"]
    catalog_bytes = Path(settings["catalog"]).read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != settings["catalog_sha256"]:
        raise ValueError("frozen private catalog changed")
    catalog = json.loads(catalog_bytes)
    if tuple(catalog["sites"]) != SITES or catalog["scope"] != settings["scope"]:
        raise ValueError("frozen catalog mismatch")
    if catalog["test_case_access"] != 0 or catalog["patient_overlap"]:
        raise ValueError("catalog leakage audit failed")
    deadline = float(settings["started_unix"]) + 23 * 3600
    if time.time() >= deadline:
        raise TimeoutError("original experiment clock expired")
    setup()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    write(root / "STARTED.json", {"started_unix": settings["started_unix"], "scope": catalog["scope"],
                                  "counts": catalog["counts"], "stages": list(SITES), "test_access": 0})
    active = {"stage": "P0", "step": 0}
    runtime = {}
    try:
        guard(root, deadline)
        net = model()
        parameter_count = sum(p.numel() for p in net.parameters())
        image = torch.zeros(2, 1, 256, 256, device="cuda")
        label = torch.zeros(2, 256, 256, dtype=torch.long, device="cuda")
        label[:, 80:180, 80:180] = 1
        probe = loss_fn(net(image), label)
        probe.backward()
        if not math.isfinite(float(probe)) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in net.parameters()):
            raise FloatingPointError("synthetic full-objective backward failed")
        net.zero_grad(set_to_none=True)
        torch.manual_seed(SEED)
        net = model()  # Synthetic P0 must not alter formal initialization.
        write(root / "P0.json", {"status": "PASS", "parameters": parameter_count,
                                 "synthetic_loss": float(probe.detach()), "full_backward": True})
        source = load_cases(data_root, catalog, (SITES[0],))
        active["stage"] = "source"
        stage = train(net, source, None, STEPS[0], "source", root, deadline, runtime)
        evaluate(net, data_root, catalog, SITES[:1], stage / "PRIVATE_VAL.pt")
        source_state = copy.deepcopy(net.state_dict())
        first_buffer = replay_slices(source[SITES[0]], SITES[0], BUFFER_SLICES, SEED)
        torch.save(first_buffer, root / "PRIVATE_REPLAY_STAGE2.pt")
        del source
        weights = {"finetune": source_state, "replay": source_state}
        second_buffer = None
        for index, site in enumerate(SITES[1:], start=1):
            current = load_cases(data_root, catalog, (site,))
            for arm in ("finetune", "replay"):
                active["stage"] = f"{arm}_{site}"
                net.load_state_dict(weights[arm])
                memory = None if arm == "finetune" else (first_buffer if index == 1 else second_buffer)
                stage = train(net, current, memory, STEPS[index], active["stage"], root, deadline, runtime)
                evaluate(net, data_root, catalog, SITES[:index + 1], stage / "PRIVATE_VAL.pt")
                weights[arm] = copy.deepcopy(net.state_dict())
            if index == 1:
                old_half = [first_buffer[8 * patient + offset] for patient in range(4) for offset in range(4)]
                if len({item["patient"] for item in old_half}) != 4:
                    raise ValueError("stage-three replay lost source patient balance")
                second_buffer = old_half + replay_slices(current[site], site, 16, SEED)
                torch.save(second_buffer, root / "PRIVATE_REPLAY_STAGE3.pt")
            del current
        active["stage"] = "joint"
        torch.manual_seed(SEED)
        joint = model()
        all_sites = load_cases(data_root, catalog, SITES)
        stage = train(joint, all_sites, None, sum(STEPS), "joint", root, deadline, runtime)
        evaluate(joint, data_root, catalog, SITES, stage / "PRIVATE_VAL.pt")
        write(root / "STATUS.json", {"status": "COMPLETE", "stages": 6, "updates": STEPS[0] + 2 * sum(STEPS[1:]) + sum(STEPS),
                                     "elapsed_seconds": time.time() - settings["started_unix"], "bytes": bytes_used(root)})
        write(root / "EXIT.json", {"exit_code": 0})
    except BaseException:
        state = {"stage": runtime.get("stage", active["stage"]), "step": runtime.get("step", 0),
                 "model": runtime["model"].state_dict() if runtime else None,
                 "optimizer": runtime["optimizer"].state_dict() if runtime else None,
                 "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}
        torch.save(state, root / "failure_partial.pt")
        write(root / "FAILURE.json", {"stage": state["stage"], "step": state["step"],
                                      "traceback": traceback.format_exc()})
        write(root / "EXIT.json", {"exit_code": 1})
        raise


if __name__ == "__main__":
    run(json.loads(Path(os.environ["TASK_SETTINGS"]).read_text()))
