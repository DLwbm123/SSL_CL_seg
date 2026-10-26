"""Two new GIN arms paired with the frozen 20261025 U/B no-replay controls."""

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F

from sc_bayeseg.supervised_cl_v1 import run as base


SITES = ("RUNMC", "BMC", "I2CVB")
STEPS = (8000, 6000, 6000)
SEED = 20261025
MAX_BYTES = 22_000_000_000
original_augment = base.augment


@torch.no_grad()
def gin_image(x, generator):
    """FOCUS Seq.+GIN default: four random grouped convolutions and Frobenius normalization."""
    batch, _, height, width = x.shape
    z = x
    for layer, (cin, cout) in enumerate(((1, 2), (2, 2), (2, 2), (2, 1))):
        kernel = (1, 3)[int(torch.randint(2, (), device=x.device, generator=generator))]
        weight = torch.randn(batch * cout, cin, kernel, kernel,
                             device=x.device, generator=generator)
        shift = torch.randn(batch * cout, 1, 1, device=x.device, generator=generator)
        z = F.conv2d(z.reshape(1, batch * cin, height, width), weight,
                     padding=kernel // 2, groups=batch).reshape(batch, cout, height, width)
        z = F.leaky_relu(z) if layer < 3 else z
    alpha = torch.rand(batch, 1, 1, 1, device=x.device, generator=generator)
    mixed = alpha * z + (1 - alpha) * x
    in_norm = torch.linalg.vector_norm(x.reshape(batch, -1), dim=1)[:, None, None, None]
    out_norm = torch.linalg.vector_norm(mixed.reshape(batch, -1), dim=1)[:, None, None, None]
    return mixed * (in_norm / (out_norm + 1e-5))


def augment(x, y, step, seed):
    x, y = original_augment(x, y, step, seed)
    generator = torch.Generator(device=x.device).manual_seed(91_000_000 + seed + step)
    if float(torch.rand((), device=x.device, generator=generator)) < .5:
        x = gin_image(x, generator)
    return x, y


def run(settings):
    os.umask(0o077)
    root = Path(settings["output"])
    if not root.is_dir() or (root / "STARTED.json").exists():
        raise FileExistsError("create-only screen root already used")
    family = settings["family"]
    if family not in ("U", "B"):
        raise ValueError("family must be U or B")
    catalog_bytes = Path(settings["catalog"]).read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != settings["catalog_sha256"]:
        raise ValueError("frozen private catalog changed")
    catalog = json.loads(catalog_bytes)
    if (tuple(catalog["sites"]) != base.SITES or catalog["scope"] != "all_train" or
            catalog["test_case_access"] != 0 or catalog["patient_overlap"]):
        raise ValueError("data boundary mismatch")
    deadline = float(settings["started_unix"]) + 23 * 3600
    if time.time() >= deadline or not torch.cuda.is_available():
        raise RuntimeError("clock or GPU admission failed")
    torch.cuda.set_device(int(settings["gpu_index"]))
    base.MAX_BYTES = MAX_BYTES
    base.setup()
    probe = torch.randn(2, 1, 32, 32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(99)
    transformed = gin_image(probe, generator)
    if (transformed.shape != probe.shape or not torch.isfinite(transformed).all() or
            torch.equal(transformed, probe) or not torch.allclose(
                torch.linalg.vector_norm(transformed.reshape(2, -1), dim=1),
                torch.linalg.vector_norm(probe.reshape(2, -1), dim=1), rtol=1e-4)):
        raise AssertionError("GIN synthetic P0 failed")
    base.setup()  # P0 cannot alter formal initialization or latent sampling.
    base.augment = augment
    base.write(root / "STARTED.json", {"started_unix": settings["started_unix"],
                                      "family": family, "seed": SEED, "sites": list(SITES),
                                      "updates_planned": sum(STEPS), "old_train_replay": 0,
                                      "old_model_or_teacher": 0, "test_access": 0})
    base.write(root / "P0.json", {"status": "PASS", "finite": True,
                                 "shape_and_norm_preserved": True})
    runtime = {}
    active = "init"
    try:
        torch.manual_seed(SEED)
        net = base.model(family)
        for index, site in enumerate(SITES):
            active = "source" if index == 0 else f"finetune_{site}"
            current = base.load_cases(settings["data_root"], catalog, (site,))
            stage = base.train(net, current, None, STEPS[index], active, root, deadline,
                               runtime, family)
            base.evaluate(net, settings["data_root"], catalog, (site,),
                          stage / "PRIVATE_TRAIN.pt", family, "train")
            base.evaluate(net, settings["data_root"], catalog, SITES[:index + 1],
                          stage / "PRIVATE_VAL.pt", family)
            del current
        base.write(root / "STATUS.json", {"status": "COMPLETE", "family": family,
                                          "updates": sum(STEPS),
                                          "elapsed_seconds": time.time() - settings["started_unix"],
                                          "bytes": base.bytes_used(root)})
        base.write(root / "EXIT.json", {"exit_code": 0})
    except BaseException:
        if runtime:
            torch.save({"stage": runtime["stage"], "step": runtime["step"],
                        "model": runtime["model"].state_dict(),
                        "optimizer": runtime["optimizer"].state_dict()}, root / "failure_partial.pt")
        base.write(root / "FAILURE.json", {"stage": active, "step": runtime.get("step", 0),
                                           "traceback": traceback.format_exc()})
        base.write(root / "EXIT.json", {"exit_code": 1})
        raise


if __name__ == "__main__":
    run(json.loads(Path(os.environ["TASK_SETTINGS"]).read_text()))
