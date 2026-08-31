"""Validation labels live only here; return fixed-support arithmetic summaries."""
import numpy as np
import torch

from di_dmpa_gate1.feature_extraction import _images, seed_after_load
from di_dmpa_gate1c_v2 import binding as b, execution as e
from .core import require, functional_forward, displacement
from .modes import assignments, old_correct, loss_maps

SUM_FIELDS = ("pixels", "CE_sum", "class_pixels", "class_CE_sum", "confusion",
              "mode_pixels", "mode_CE_sum", "mode_TP", "old_correct_pixels", "old_KL_sum")


def statistics(logits, labels, modes, correct, old_probability):
    with torch.no_grad():
        ce, kl = loss_maps(logits.detach(), labels, old_probability)
        pred = logits.argmax(1)
        valid = labels != 255
        confusion = torch.bincount((labels[valid]*3 + pred[valid]).reshape(-1), minlength=9).reshape(3, 3)
        counts, sums = [], []
        for c in range(3):
            selected = labels == c
            counts.append(int(selected.sum()))
            sums.append(float(ce[selected].sum()))
        mc, ms, mtp, oc, oks = [], [], [], [], []
        for i in range(6):
            selected = modes == i
            old = selected & correct
            mc.append(int(selected.sum())); ms.append(float(ce[selected].sum()))
            mtp.append(int((selected & (pred == i//2)).sum()))
            oc.append(int(old.sum())); oks.append(float(kl[old].sum()))
        return dict(pixels=int(valid.sum()), CE_sum=float(ce[valid].sum()), class_pixels=counts, class_CE_sum=sums,
                    confusion=confusion.cpu().tolist(), mode_pixels=mc, mode_CE_sum=ms, mode_TP=mtp,
                    old_correct_pixels=oc, old_KL_sum=oks, logits_sha256=b.tensor_hash(logits),
                    labels_sha256=b.tensor_hash(labels), modes_sha256=b.tensor_hash(modes))


def aggregate(batches):
    require(bool(batches), "empty evaluator panel", "BLOCKED_INCOMPLETE_EVIDENCE")
    totals = {key: np.sum([b[key] for b in batches], axis=0).tolist() for key in SUM_FIELDS}
    require(totals["pixels"] > 0, "no valid evaluator pixels", "BLOCKED_INCOMPLETE_EVIDENCE")
    conf = np.asarray(totals["confusion"], np.int64)
    require(conf.shape == (3, 3) and int(conf.sum()) == totals["pixels"], "evaluator confusion coverage")
    require(np.array_equal(conf.sum(1), totals["class_pixels"]), "class counts/confusion disagree")
    dice = [2*int(conf[c, c])/int(conf[c].sum()+conf[:, c].sum()) if conf[c].sum()+conf[:, c].sum() else None for c in range(3)]
    divide = lambda values, counts: [float(v/n) if n else None for v, n in zip(values, counts)]
    return dict(totals=totals, CE=totals["CE_sum"]/totals["pixels"], class_CE=divide(totals["class_CE_sum"], totals["class_pixels"]),
                Dice=dice, foreground_Dice=float(np.mean(dice[1:])) if all(x is not None for x in dice[1:]) else None,
                mode_CE=divide(totals["mode_CE_sum"], totals["mode_pixels"]),
                mode_Dice=divide([2*x for x in totals["mode_TP"]], [n+x for n, x in zip(totals["mode_pixels"], totals["mode_TP"])]),
                old_KL=divide(totals["old_KL_sum"], totals["old_correct_pixels"]), batch_count=len(batches))


def prepare_validation(models, centers, active, unit, p, data_root, output, device):
    """Old-EMA strata/before predictions: evaluator only, never returned to QP."""
    result = {}
    for side in ("previous", "current"):
        panel = unit["validation"][side]
        require(panel["role"] == "val" and len(panel["batches"]) == 8, "validation panel not frozen")
        rows = {r["case_id"]: r for r in b.records(data_root, p, unit["seed"], panel["domain_stage"], "val")}
        descriptors, metrics = [], []
        for batch in panel["batches"]:
            selected = [rows[case] for case in batch["case_ids"]]
            images = _images([b.image_only(r) for r in selected], data_root).to(device)
            labels_np = np.stack(e.visible_labels(selected, data_root, role="val"))
            with torch.no_grad():
                seed_after_load(batch["old_classifier_seed"])
                old_logits, features = models["old"](images, stochastic_classifier=False)
                old_p = old_logits.float().softmax(1).cpu().numpy()
                modes, directional = assignments(features.cpu().numpy(), labels_np, centers, active)
                correct = old_correct(old_p, labels_np)
                seed_after_load(batch["student_classifier_seed"])
                logits, _ = models["gradient_student"](images.double(), stochastic_classifier=False)
                stats = statistics(logits, torch.from_numpy(labels_np).to(device), torch.from_numpy(modes).to(device),
                                   torch.from_numpy(correct).to(device), torch.from_numpy(old_p).to(device))
            arrays = b.save_arrays(output/side/f'batch{batch["batch_index"]:02d}.npz',
                                   dict(labels=labels_np.astype(np.uint8), modes=modes, active=directional,
                                        old_correct=correct, old_probability=old_p))
            descriptors.append(dict(batch=batch, arrays=arrays, rows=[b.image_only(r) for r in selected],
                                    old_features_sha256=b.tensor_hash(features), old_logits_sha256=b.tensor_hash(old_logits),
                                    GT_consumer="evaluator_only", role="val"))
            metrics.append(dict(batch_id=batch["batch_id"], case_ids=batch["case_ids"], **stats))
        result[side] = dict(descriptors=descriptors, before_batches=metrics, before=aggregate(metrics),
                            role="val", GT_consumer="evaluator_only", case_ids=panel["case_ids"])
    return result


def load_panel(panel, data_root, device, *, role):
    require(role in ("val", "train_labeled") and panel["role"] == role, "evaluator role boundary")
    loaded = []
    for desc in panel["descriptors"]:
        require(desc["role"] == role and desc["batch"]["role"] == role, "mixed-role evaluator batch")
        arrays = b.read_arrays(desc["arrays"])
        loaded.append(dict(batch=desc["batch"],
                           images=_images(desc["rows"], data_root).to(device),
                           labels=torch.as_tensor(arrays["labels"].astype(np.int64), device=device),
                           modes=torch.as_tensor(arrays["modes"], device=device),
                           correct=torch.as_tensor(arrays["old_correct"], device=device),
                           probability=torch.as_tensor(arrays["old_probability"], device=device),
                           draw=None if role == "val" else torch.as_tensor(arrays["gaussian"], device=device)))
    require([case for item in loaded for case in item["batch"]["case_ids"]] == panel["case_ids"], "evaluator case coverage/order")
    return loaded


def candidate(model, direction, raw_norm, panels):
    mapping, step = displacement(model, direction, raw_norm=raw_norm)
    result = dict(step=step, panels={})
    for name in ("previous", "current", "train_labeled"):
        metrics = []
        for item in panels[name]:
            seed_after_load(item["batch"]["student_classifier_seed"])
            logits, _ = functional_forward(model, mapping, item["images"], draw=item["draw"])
            stats = statistics(logits, item["labels"], item["modes"], item["correct"], item["probability"])
            metrics.append(dict(batch_id=item["batch"]["batch_id"], case_ids=item["batch"]["case_ids"], **stats))
        result["panels"][name] = dict(after_batches=metrics, after=aggregate(metrics), GT_consumer="evaluator_only" if name != "train_labeled" else "visible_labeled_diagnostic")
    return result
