"""Per-image foreground Dice, then equal-domain aggregation."""
from __future__ import annotations

import torch


def image_dice(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("expected paired 2D masks")
    valid = target != 255
    if not torch.isin(target[valid], torch.tensor([0, 1, 2], device=target.device)).all():
        raise ValueError("invalid target class")
    result = {}
    for name, cls in (("rim", 1), ("cup", 2)):
        gt = (target == cls) & valid
        pr = (prediction == cls) & valid
        denominator = gt.sum() + pr.sum()
        result[name] = None if not denominator else float(2 * (gt & pr).sum() / denominator)
    supported = [x for x in result.values() if x is not None]
    result["macro"] = sum(supported) / len(supported) if supported else None
    result["disc_union"] = _binary_dice(torch.isin(prediction, torch.tensor([1, 2], device=prediction.device)) & valid,
                                        torch.isin(target, torch.tensor([1, 2], device=target.device)) & valid)
    return result


def _binary_dice(prediction: torch.Tensor, target: torch.Tensor) -> float | None:
    denominator = prediction.sum() + target.sum()
    return None if not denominator else float(2 * (prediction & target).sum() / denominator)


def equal_domain_mean(domains: dict[str, list[dict]]) -> dict:
    domain_scores = {}
    support = {}
    for name, images in domains.items():
        scores = [item["macro"] for item in images if item["macro"] is not None]
        if not scores:
            raise ValueError("domain has no supported foreground image")
        domain_scores[name] = sum(scores) / len(scores)
        support[name] = len(scores)
    if not domain_scores:
        raise ValueError("no domains")
    return dict(domains=domain_scores, supported_images=support,
                Q=sum(domain_scores.values()) / len(domain_scores))
