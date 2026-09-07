"""Detached FP64 targets on the SAME logits used for KL; no network clones."""
import time

import torch
from torch.nn import functional as F

BLOCK = 65536
EPS_Q = 1e-8


@torch.no_grad()
def project(p, q, y, reliable, *, iterations=48, max_bracket=60):
    if p.dtype != torch.float64 or q.dtype != torch.float64:
        raise ValueError("projection requires float64 probabilities")
    if p.shape != q.shape or p.ndim != 2 or y.shape != p.shape[:1] or reliable.shape != y.shape:
        raise ValueError("projection shape mismatch")
    if not (torch.isfinite(p).all() and torch.isfinite(q).all() and (p >= 0).all() and (q > 0).all()):
        raise FloatingPointError("nonfinite or unsupported probabilities")
    if not (torch.allclose(p.sum(-1), torch.ones_like(p[:, 0]), atol=1e-12, rtol=0) and
            torch.allclose(q.sum(-1), torch.ones_like(q[:, 0]), atol=1e-12, rtol=0)):
        raise FloatingPointError("probabilities do not sum to one")
    a = p - F.one_hot(y, p.shape[-1]).double()
    conflict = reliable & (((p - q) * a).sum(-1) < 0)
    r = q.clone()
    eta = torch.zeros_like(p[:, 0])
    residual = torch.zeros_like(eta)
    brackets = 0
    ix = conflict.nonzero().flatten()
    for start in range(0, ix.numel(), BLOCK):
        j = ix[start:start + BLOCK]
        scale = a[j].abs().amax(-1)
        if (scale == 0).any():
            raise FloatingPointError("zero direction cannot be strictly conflicting")
        aa = a[j] / scale[:, None]
        b = (aa * p[j]).sum(-1)
        logq = q[j].log()
        lo, hi = torch.zeros_like(b), torch.ones_like(b)
        def target(e):
            return (logq - e[:, None] * aa).softmax(-1)
        for n in range(max_bracket + 1):
            need = (target(hi) * aa).sum(-1) > b
            if not need.any():
                brackets = max(brackets, n)
                break
            if n == max_bracket:
                raise FloatingPointError("SCD root not bracketed")
            hi = torch.where(need, hi * 2, hi)
        for _ in range(iterations):
            mid = (lo + hi) / 2
            infeasible = (target(mid) * aa).sum(-1) > b
            lo = torch.where(infeasible, mid, lo)
            hi = torch.where(infeasible, hi, mid)
        rr = target(hi)
        error = ((rr * aa).sum(-1) - b).abs()
        if not torch.isfinite(rr).all() or (error > 1e-9).any():
            raise FloatingPointError("SCD normalized residual exceeds 1e-9")
        r[j], eta[j], residual[j] = rr, hi / scale, error
    return r, conflict, eta, residual, brackets


def kd(logits, teacher_probability, reference, reliable, geometry, arm):
    if arm not in "CDE":
        raise ValueError("KD arm must be C/D/E")
    logp = logits.double().movedim(1, -1).reshape(-1, logits.shape[1]).log_softmax(-1)
    p = logp.detach().exp()
    q0 = teacher_probability.detach().double().movedim(1, -1).reshape_as(p)
    # Normalize conversion error from FP32 softmax before common positive-support smoothing.
    q0 = q0 / q0.sum(-1, keepdim=True)
    q = (1 - EPS_Q) * q0 + EPS_Q / p.shape[-1]
    y = reference.flatten().clamp(0, p.shape[-1] - 1)
    valid, rel = geometry.flatten().bool(), reliable.flatten().bool() & geometry.flatten().bool()
    a = p - F.one_hot(y, p.shape[-1]).double()
    conflict = rel & (((p - q) * a).sum(-1) < 0)
    start = time.perf_counter()
    eta, residual, brackets = torch.zeros_like(p[:, 0]), torch.zeros_like(p[:, 0]), 0
    if arm == "E":
        target, conflict, eta, residual, brackets = project(p, q, y, rel)
    else:
        target = q
    keep = valid & (~conflict if arm == "D" else torch.ones_like(valid))
    # Projected FP64 probabilities can underflow to zero. KL's continuous
    # extension is 0*log(0)=0; keep positive-target arithmetic bit-identical.
    log_target = target.log().masked_fill(target == 0, 0.)
    per = (target * (log_target - logp)).sum(-1)
    loss = (per * keep).sum() / valid.sum().clamp_min(1)
    # Small class-level sufficient statistics only; never retain pixel targets after the step.
    rows = []
    for c in range(p.shape[-1]):
        j = valid & (y == c)
        active = j & conflict
        n = int(j.sum())
        rows.append(dict(class_id=c, pixels=n, reliable=int((j & rel).sum()),
            conflicts=int(active.sum()), projected=int(active.sum()) if arm == "E" else 0,
            drop_kl_mass=float((per.detach() * active).sum()) if arm == "D" else 0.,
            drop_gradient_l2_sum=float(((p - q).norm(dim=-1) * active).sum()) if arm == "D" else 0.,
            r_q_norm_sum=float(((target - q).norm(dim=-1) * active).sum()),
            r_p_norm_sum=float(((target - p).norm(dim=-1) * active).sum()),
            nondegenerate=int((active & ((target-q).norm(dim=-1)>1e-8) & ((target-p).norm(dim=-1)>1e-8)).sum()) if arm == "E" else 0,
            eta_sum=float((eta * active).sum()), eta_max=float(eta[active].max()) if active.any() else 0.,
            residual_max=float(residual[j].max()) if n else 0.,
            smoothing_l1_sum=float(((q-q0).abs().sum(-1) * j).sum())))
    return loss, dict(classes=rows, solver_seconds=time.perf_counter()-start,
        bracket_max=brackets, target_temporary_bytes=sum(v.numel()*v.element_size() for v in (p,q0,q,target,a,eta,residual,logp)))


def supervised(logits, label):
    valid = label != 255
    # Explicit gather avoids Torch2.2 CUDA nll_loss2d atomic spatial reduction.
    pixels = -logits.log_softmax(1).gather(1,label.masked_fill(~valid,0)[:,None]).squeeze(1)
    return (pixels * valid).sum() / valid.sum().clamp_min(1)


def ssl_ce(logits, pseudo, mask, geometry):
    pixels = -logits.log_softmax(1).gather(1,pseudo.detach()[:,None]).squeeze(1)
    return (pixels * mask.detach() * geometry).sum() / geometry.sum().clamp_min(1)


@torch.no_grad()
def pas(logits, features, prototypes, supported, geometry):
    probability = logits.softmax(1)
    confidence, pseudo = probability.max(1)
    feature = F.normalize(features, dim=1).movedim(1, -1)
    similarity = (feature * prototypes[pseudo]).sum(-1)
    mask = (confidence > .7) & (similarity > .7) & supported[pseudo] & geometry
    return pseudo, mask, probability


@torch.no_grad()
def case_center(features, label, classes=3):
    centers = features.new_zeros((classes, features.shape[0]))
    supported = torch.zeros(classes, dtype=torch.bool, device=features.device)
    for c in range(classes):
        mask = label == c
        if mask.any():
            centers[c] = F.normalize(features[:, mask].mean(-1), dim=0)
            supported[c] = True
    return centers, supported


def lambda_u(epoch):
    return .5 * min(1., max(0., (epoch - 10) / 10))


def learning_rate(step, total):
    if not 0 <= step < total:
        raise ValueError("LR step outside stage")
    return .001 * (1 - step / total) ** .9
