"""Visible-labeled-only frozen K2 modes and whole-panel guard losses."""
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from di_dmpa_gate1.sampling import sample_layout
from di_dmpa_gate1_v2.features import split_support
from di_dmpa_gate1_v2.geometry import fit, validate_centers
from di_dmpa_gate1c_v2 import binding as b
from di_dmpa_gate1c_v3.durable import canonical
from .core import require, vector


def fit_bank(unit, caches):
    require(unit["role"] == "train_labeled", "mode fit cannot receive validation/hidden/test GT")
    require(set(caches) == {r["case_id"] for r in unit["cases"]}, "mode fit cases differ")
    banks, active, controls, rows = [], [], [], []
    for c in range(3):
        layout = sample_layout(unit, c)
        raw = []
        for case in unit["cases"]:
            coordinates = np.asarray(case["classes"][c]["coordinates"], np.int64).reshape(-1, 2)
            raw.append(caches[case["case_id"]]["features"][:, coordinates[:, 0], coordinates[:, 1]].T)
        data = split_support(np.concatenate(raw))
        entries = {}
        for K in (1, 2):
            found = fit(data["directions"], data["active_mask"], layout["weights"], K,
                        seed=unit["seed"], stage=unit["stage_index"], class_id=c, uid_rank=layout["uid_rank"])
            entries[str(K)] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in found.items()}
        banks.append(entries["2"]["centers"])
        active.append(entries["2"]["active"])
        controls.append(entries["1"]["centers"])
        rows.append(dict(class_id=c, fits=entries, UID_order_sha256=b.H(layout["uids"]),
                         sampled_coordinates_sha256=b.H(layout["uids"]), original_weights_sha256=b.array_hash(layout["weights"]),
                         original_weight_sum=float(layout["weights"].sum()), case_count=len(set(layout["case_ids"])),
                         sampled_pixels=len(layout["uids"]), null_count=int((~data["active_mask"]).sum())))
    return np.asarray(banks, np.float64), np.asarray(active, bool), np.asarray(controls, np.float64), rows


def assignments(features, labels, centers, center_active):
    features, labels = np.asarray(features), np.asarray(labels)
    require(features.ndim == 4 and features.shape[1] == 16 and labels.shape == features.shape[:1]+features.shape[2:], "mode geometry")
    require(centers.shape == (3, 2, 16) and center_active.shape == (3, 2), "K2 bank geometry")
    require(np.isin(labels, [0, 1, 2, 255]).all(), "label mapping changed")
    data = split_support(features.transpose(0, 2, 3, 1).reshape(-1, 16))
    y = labels.reshape(-1)
    modes = np.full(y.shape, -1, np.int8)
    modes[y == 255] = -2
    for c in range(3):
        validate_centers(centers[c], center_active[c])
        selected = (y == c) & data["active_mask"]
        if selected.any() and center_active[c].any():
            score = data["directions"][selected] @ centers[c].T
            score[:, ~center_active[c]] = -np.inf
            modes[selected] = (2*c + score.argmax(1)).astype(np.int8)
    return modes.reshape(labels.shape), data["active_mask"].reshape(labels.shape)


def old_correct(probability, labels):
    p = np.asarray(probability)
    b.finite(p)
    require(p.shape == (len(labels), 3, *labels.shape[1:]), "old probability/label geometry")
    require((p >= 0).all() and np.allclose(p.sum(1), 1, atol=3e-7, rtol=0), "old posterior mean probability invalid")
    return (p.argmax(1) == labels) & (labels != 255)


def support_rows(case_ids, caches, mode_maps, center_active, fits):
    rows = []
    for c in range(3):
        for k in range(2):
            index = 2*c+k
            pixels = cases = correct = nulls = active_class = 0
            uid, coordinate, null_uid = (hashlib.sha256() for _ in range(3))
            for case in case_ids:
                y = caches[case]["labels"]
                mode, active = mode_maps[case]["modes"], mode_maps[case]["active"]
                selected = mode == index
                coords = np.asarray(np.argwhere(selected), dtype="<i4")
                nullcoords = np.asarray(np.argwhere((y == c) & ~active), dtype="<i4")
                pixels += len(coords)
                cases += int(bool(len(coords)))
                correct += int((selected & mode_maps[case]["old_correct"]).sum())
                nulls += len(nullcoords)
                active_class += int(((y == c) & active).sum())
                uid.update(canonical([case, c, len(coords)])); uid.update(coords.tobytes())
                coordinate.update(canonical([case, len(coords)])); coordinate.update(coords.tobytes())
                null_uid.update(canonical([case, c, len(nullcoords)])); null_uid.update(nullcoords.tobytes())
            fitted = fits[c]["fits"]["2"]
            rows.append(dict(class_id=c, mode=k, mode_index=index, active_pixels=pixels, case_count=cases,
                active_class_pixels=active_class, occupancy=pixels/active_class if active_class else 0.0,
                old_correct_pixels=correct, KD_active=correct >= 32, center_active=bool(center_active[c, k]),
                UID_sha256=uid.hexdigest(), coordinate_sha256=coordinate.hexdigest(), null_UID_sha256=null_uid.hexdigest(),
                UID_serialization="ordered case/class/count JSON prefix then row-major little-endian int32 y,x pairs",
                prototype=fitted["centers"][k], center_norm=fitted["center_norms"][k], null_count=nulls,
                fit_converged=fitted["converged"], fit_iterations=fitted["iterations"],
                sampled_UID_sha256=fits[c]["UID_order_sha256"], finite=True, null_rule_pass=True))
    return rows


def loss_maps(logits, labels, old_probability):
    require(logits.shape == old_probability.shape and labels.shape == logits.shape[:1]+logits.shape[2:], "loss geometry")
    require(not old_probability.requires_grad and old_probability.grad is None, "old function target must be detached")
    ce = F.cross_entropy(logits, labels, ignore_index=255, reduction="none")
    kl = F.kl_div(logits.log_softmax(1), old_probability.detach().to(logits), reduction="none").sum(1)
    b.finite(ce, kl)
    return ce, kl


def guard_vjps(logits, labels, modes, correct, old_probability, named, support):
    ce, kl = loss_maps(logits, labels, old_probability)
    gradients, none_masks = [], []
    for kind, loss_map, key in (("sup", ce, "active_pixels"), ("old", kl, "old_correct_pixels")):
        for i, row in enumerate(support):
            selected = modes == i
            if kind == "old":
                selected = selected & correct
            denominator = row[key]
            numerator = loss_map[selected].sum()
            value = numerator/denominator if denominator else logits.sum()*0.0
            g, none = vector(value, named, retain=not (kind == "old" and i == 5))
            gradients.append(g)
            none_masks.append(list(none))
    return np.stack(gradients[:6]), np.stack(gradients[6:]), none_masks
