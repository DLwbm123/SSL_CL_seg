"""Small supervised matcher and the paper-selected GRQA surrogate."""
from __future__ import annotations

from itertools import permutations
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=12)
def _assignment_options(queries: int, classes: int) -> torch.Tensor:
    return torch.tensor(list(permutations(range(queries), classes)), dtype=torch.long)


def supervised_query_loss(class_logits: torch.Tensor, mask_logits: torch.Tensor,
                          labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Match present semantic classes, including true background; 3 is no-object."""
    if class_logits.ndim != 3 or class_logits.shape[-1] != 4 or mask_logits.shape[:2] != class_logits.shape[:2]:
        raise ValueError("invalid query output")
    if labels.shape != mask_logits.shape[:1] + mask_logits.shape[-2:]:
        raise ValueError("mask and label geometry differ")
    targets = torch.full(class_logits.shape[:2], 3, device=labels.device, dtype=torch.long)
    mask_bce = mask_logits.sum() * 0
    mask_dice = mask_logits.sum() * 0
    matched = 0
    for batch in range(labels.shape[0]):
        valid = labels[batch] != 255
        if not valid.any():
            continue
        classes = torch.unique(labels[batch][valid]).tolist()
        if any(c not in (0, 1, 2) for c in classes):
            raise ValueError("invalid canonical class")
        if len(classes) > class_logits.shape[1]:
            raise ValueError("fewer queries than present classes")
        pred = mask_logits[batch, :, valid]
        binary = torch.stack([(labels[batch][valid] == c).float() for c in classes])
        with torch.no_grad():
            class_cost = -class_logits[batch].softmax(-1)[:, classes]
            bce_cost = torch.stack([
                F.binary_cross_entropy_with_logits(pred, target.expand_as(pred), reduction="none").mean(1)
                for target in binary
            ], dim=1)
            probabilities = pred.sigmoid()
            intersection = probabilities @ binary.T
            dice_cost = 1 - (2 * intersection + 1) / (
                probabilities.sum(1, keepdim=True) + binary.sum(1)[None] + 1)
            cost = class_cost + 5 * bce_cost + 5 * dice_cost
            options = _assignment_options(pred.shape[0], len(classes)).to(cost.device)
            columns = torch.arange(len(classes), device=cost.device)
            choice = options[cost[options, columns].sum(-1).argmin()].tolist()
        for j, query in enumerate(choice):
            targets[batch, query] = classes[j]
            mask_bce = mask_bce + F.binary_cross_entropy_with_logits(pred[query], binary[j])
            p = pred[query].sigmoid()
            mask_dice = mask_dice + 1 - (2 * (p * binary[j]).sum() + 1) / (p.sum() + binary[j].sum() + 1)
            matched += 1
    class_loss = F.cross_entropy(class_logits.transpose(1, 2), targets,
                                 weight=class_logits.new_tensor([1, 1, 1, 0.1]))
    total = class_loss + 5 * (mask_bce + mask_dice) / max(matched, 1)
    return dict(total=total, class_loss=class_loss, mask_bce=mask_bce / max(matched, 1),
                mask_dice=mask_dice / max(matched, 1), matched=class_logits.new_tensor(matched))


def image_prototypes(pixels: torch.Tensor, labels: torch.Tensor, *, vit_pad: bool = False
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize each pixel first, then mean current-L GT regions."""
    if pixels.ndim != 4 or labels.ndim != 3 or pixels.shape[0] != labels.shape[0]:
        raise ValueError("invalid pixel/label tensors")
    if vit_pad:
        labels = F.pad(labels, (0, 8, 0, 8), value=255)
    small = F.interpolate(labels[:, None].float(), size=pixels.shape[-2:], mode="nearest")[:, 0].long()
    normalized = F.normalize(pixels, dim=1)
    vectors = []
    supported = []
    for batch in range(labels.shape[0]):
        per_class = []
        per_support = []
        for c in range(3):
            region = small[batch] == c
            per_support.append(region.any())
            per_class.append(normalized[batch, :, region].mean(1) if region.any() else normalized[batch].sum((1, 2)) * 0)
        vectors.append(torch.stack(per_class))
        supported.append(torch.stack(per_support))
    return torch.stack(vectors), torch.stack(supported)


class PrototypeBank(nn.Module):
    def __init__(self, dimension: int, ema: float = 0.9) -> None:
        super().__init__()
        self.ema = ema
        self.register_buffer("vectors", torch.zeros(3, dimension))
        self.register_buffer("supported", torch.zeros(3, dtype=torch.bool))

    @torch.no_grad()
    def reset(self) -> None:
        self.vectors.zero_()
        self.supported.zero_()

    @torch.no_grad()
    def update(self, current: torch.Tensor, support: torch.Tensor) -> None:
        for c in range(3):
            values = current[:, c][support[:, c]]
            if not len(values):
                continue
            proposal = values.mean(0)
            if proposal.norm() < 1e-6 or not torch.isfinite(proposal).all():
                continue
            self.vectors[c] = F.normalize(self.ema * self.vectors[c] + (1 - self.ema) * proposal,
                                          dim=0) if self.supported[c] else F.normalize(proposal, dim=0)
            self.supported[c] = True


def image_alignment_loss(current: torch.Tensor, support: torch.Tensor,
                         bank: PrototypeBank) -> torch.Tensor:
    selected = support & bank.supported.unsqueeze(0)
    if not selected.any():
        return current.sum() * 0
    return ((current - bank.vectors.detach().unsqueeze(0)).square().sum(-1))[selected].mean()


def grqa_loss(queries: torch.Tensor, reference_queries: torch.Tensor,
              bank: PrototypeBank, *, clip: float = 0.1, beta: float = 0.001
              ) -> dict[str, torch.Tensor]:
    if queries.shape != reference_queries.shape or queries.shape[-1] != bank.vectors.shape[-1]:
        raise ValueError("query/reference/bank geometry mismatch")
    if not bank.supported.any():
        zero = queries.sum() * 0
        return dict(total=zero, group=zero, paper_selected_k3_surrogate=zero,
                    categorical_kl=zero, valid_queries=zero.detach())
    anchors = bank.vectors.detach()
    current = F.normalize(queries, dim=-1) @ anchors.T
    reference = F.normalize(reference_queries.detach(), dim=-1) @ anchors.T
    current = current.masked_fill(~bank.supported[None, None], -torch.inf)
    reference = reference.masked_fill(~bank.supported[None, None], -torch.inf)
    selected = current.detach().argmax(-1)
    reward = current.detach().gather(-1, selected[..., None]).squeeze(-1)
    advantage = torch.zeros_like(reward)
    for batch in range(queries.shape[0]):
        for c in range(3):
            group = selected[batch] == c
            if group.sum() > 1:
                values = reward[batch, group]
                std = values.std(unbiased=False)
                if std > 0:
                    advantage[batch, group] = (values - values.mean()) / (std + 1e-6)
    log_current = current.log_softmax(-1)
    log_reference = reference.log_softmax(-1).detach()
    log_selected_current = log_current.gather(-1, selected[..., None]).squeeze(-1)
    log_selected_reference = log_reference.gather(-1, selected[..., None]).squeeze(-1)
    ratio = (log_selected_current - log_selected_reference).exp()
    group_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
    delta = log_selected_reference - log_selected_current
    paper_k3 = (torch.expm1(delta) - delta).mean()
    lc = log_current[..., bank.supported]
    lr = log_reference[..., bank.supported]
    categorical_kl = (lc.exp() * (lc - lr)).sum(-1).mean()
    total = group_loss + beta * paper_k3
    if not (torch.isfinite(total) and torch.isfinite(categorical_kl)):
        raise FloatingPointError("non-finite GRQA; stop instead of clipping")
    return dict(total=total, group=group_loss, paper_selected_k3_surrogate=paper_k3,
                categorical_kl=categorical_kl.detach(), valid_queries=queries.new_tensor(queries.shape[0] * queries.shape[1]))
