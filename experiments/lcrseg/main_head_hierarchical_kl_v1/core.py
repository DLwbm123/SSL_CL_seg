"""Frozen main-head hierarchy; no data access or trainable parameters."""
import math
import torch
from ..five_frameworks_v1.kernels import masked_kl, _finite, _prob, _mask

ARMS = ('C0', 'C1', 'C2')


def validate(logits, target, valid):
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError('equal BCHW tensors with bg/rim/cup channels required')
    _finite(logits, 'logits'); _prob(target, 'target'); _mask(valid, logits)
    q = target.detach().to(logits)
    if not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError('target must sum to one')
    return q, valid.detach()


def parts(logits, target, valid):
    q, mask = validate(logits, target, valid)
    lp = logits.log_softmax(1)
    ld = torch.logsumexp(lp[:, 1:3], dim=1)
    dq = q[:, 1:3].sum(1)
    parent = torch.xlogy(q[:, 0], q[:, 0]) - q[:, 0]*lp[:, 0] + torch.xlogy(dq, dq) - dq*ld
    # Exact zero-mass limit, no tiny-probability division.
    conditional = (torch.xlogy(q[:, 1:3], q[:, 1:3]) - q[:, 1:3]*(lp[:, 1:3]-ld[:, None])).sum(1) - torch.xlogy(dq, dq)
    return parent, conditional


@torch.no_grad()
def weights(target, valid, arm):
    if arm not in ARMS:raise ValueError('unregistered arm')
    q, mask = validate(torch.zeros_like(target), target, valid)
    dq = q[:, 1:3].sum(1)
    eligible = mask & (dq >= .9)
    # Ratios are taken only on the registered high-parent-mass support.
    denom = torch.where(eligible, dq, torch.ones_like(dq))
    conditional = torch.where(eligible[:, None], q[:, 1:3]/denom[:, None], torch.zeros_like(q[:, 1:3]))
    entropy = -torch.xlogy(conditional, conditional).sum(1)/math.log(2)
    target_w = torch.where(eligible, 1-.5*entropy.clamp(0, 1), torch.ones_like(dq))
    if arm == 'C0':return torch.ones_like(dq)
    if arm == 'C2':return target_w
    w = torch.ones_like(dq)
    predicted = q.argmax(1)  # deterministic first-index ties; E cannot predict bg
    for image in range(q.shape[0]):
        for cls in (1, 2):
            group = eligible[image] & (predicted[image] == cls)
            mass = dq[image][group].sum()
            if bool(mass > 0):w[image][group] = (dq[image][group]*target_w[image][group]).sum()/mass
    return w


def hierarchical_kl(logits, target, valid, arm):
    if arm not in ARMS:raise ValueError('unregistered arm')
    q, mask = validate(logits, target, valid)
    # Exact original code path for the no-op, including value and gradient.
    if arm == 'C0':return masked_kl(logits, q, mask)
    w = weights(q, mask, arm)
    if bool((w == 1).all()):return masked_kl(logits, q, mask)
    parent, conditional = parts(logits, q, mask)
    return ((parent+w*conditional)*mask).sum()/mask.sum().clamp_min(1)
