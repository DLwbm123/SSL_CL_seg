"""Read-only, separate-process patient evaluation for H1."""

import hashlib
import json
import os
import time
from pathlib import Path

import torch

from sc_bayeseg.sslcl_v1.prostate_metrics import binary_phase_metrics
from sc_bayeseg.supervised_cl_v1 import run as data

from .model import HierarchicalSegmenter


SEED = 20261026
SITES = ("RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK")


def evaluate(settings):
    os.umask(0o077)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(int(settings["gpu_index"]))
    catalog_bytes = Path(settings["catalog"]).read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != settings["catalog_sha256"]:
        raise ValueError("private catalog changed")
    catalog = json.loads(catalog_bytes)
    if catalog["scope"] != "all_train" or catalog["test_case_access"] != 0 or catalog["patient_overlap"]:
        raise ValueError("catalog violates H1 scope")
    role = settings["role"]
    sites = tuple(settings["sites"])
    current = settings["current_site"]
    if role not in ("val", "train") or sites != SITES[:len(sites)] or current not in sites:
        raise ValueError("invalid evaluation role or sites")
    if role == "train" and sites != (current,):
        raise ValueError("old train data may not be opened")
    family = settings["family"]
    if family not in ("U", "H"):
        raise ValueError("invalid model family")
    model = (data.backbone() if family == "U" else HierarchicalSegmenter()).cuda()
    snapshot = torch.load(settings["checkpoint"], map_location="cuda", weights_only=False)
    model.load_state_dict(snapshot["model"])
    model.eval()
    result = {}
    started = time.perf_counter()
    images = 0
    with torch.no_grad():
        for site in sites:
            patients = []
            for record in catalog["cases"][site][role]:
                case = data.read_case(settings["data_root"], record)
                seed = int.from_bytes(hashlib.sha256(f"{SEED}:{case['id']}".encode()).digest()[:8], "little") % (2**63)
                predictions = []
                for offset in range(0, len(case["image"]), 4):
                    image = torch.from_numpy(case["image"][offset:offset + 4].copy()).cuda()[:, None]
                    if family == "U":
                        probability = model(image).softmax(1)
                    else:
                        foreground = model.predict_probability(image, seed + offset)
                        probability = torch.cat((1 - foreground, foreground), dim=1)
                    predictions.append(probability.cpu())
                    images += len(image)
                metrics = binary_phase_metrics(torch.cat(predictions), case["label"], case["spacing"])
                patients.append({"patient": case["id"], "metrics": metrics})
            result[site] = patients
    output = Path(settings["output"])
    with output.open("xb") as handle:
        torch.save(result, handle)
    data.write(output.with_name(output.stem + "_COST.json"),
               {"role": role, "images": images, "seconds": time.perf_counter() - started,
                "family": family, "mc_local_samples": 1 if family == "U" else 16})


if __name__ == "__main__":
    evaluate(json.loads(Path(os.environ["TASK_EVAL_SETTINGS"]).read_text()))
