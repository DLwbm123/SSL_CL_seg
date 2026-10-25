"""Current diagonal-plus-low-rank Gaussian precision for H4.

Only a rank-four predictive Fisher summary crosses each site boundary. The
patient images, labels and probe vectors are discarded before the next site.
"""

import hashlib
import math

import torch

from sc_bayeseg.supervised_cl_v1 import run as data
from ..hierarchical_bayes_h1.online import initial_precision, named_prediction_parameters


RANK_PER_SITE = 4
PROBES_PER_PATIENT = 2
DIAGONAL_SHARE = 0.5


def _flat(values, named):
    return torch.cat(tuple(values[name].reshape(-1) for name, _ in named))


def precision_trace(values, named):
    return float(sum(values[name].double().sum() for name, _ in named))


def split_source_precision(model, full_precision):
    """Split prior-to-source diagonal evidence equally with the rank term."""
    named = named_prediction_parameters(model)
    prior = initial_precision(named)
    diagonal = {}
    for name, _ in named:
        increment = full_precision[name] - prior[name]
        if not torch.isfinite(increment).all() or (increment < -1e-4).any():
            raise FloatingPointError("invalid source Fisher increment")
        diagonal[name] = prior[name] + DIAGONAL_SHARE * increment.clamp_min(0)
    target = (1 - DIAGONAL_SHARE) * (precision_trace(full_precision, named) -
                                      precision_trace(prior, named))
    if not math.isfinite(target) or target <= 0:
        raise FloatingPointError("source low-rank evidence trace is invalid")
    return diagonal, target


def predictive_basis(model, cases, site, seed, target_trace):
    """Top four cross-parameter Fisher directions on the current site's train.

    Each patient contributes one deterministically selected foreground slice
    and two Rademacher VJPs. The full temporary probe matrix is discarded.
    """
    named = named_prediction_parameters(model)
    parameters = tuple(parameter for _, parameter in named)
    total = sum(parameter.numel() for parameter in parameters)
    vectors = []
    was_training = model.training
    model.eval()
    try:
        for case in cases:
            foreground = case["foreground"]
            if len(foreground) == 0:
                raise ValueError("current patient has no foreground slice for H4 basis")
            salt = int.from_bytes(hashlib.sha256(
                f"{seed}:{site}:{case['id']}".encode()).digest()[:8], "little")
            position = int(foreground[salt % len(foreground)])
            image = torch.from_numpy(case["image"][position].copy()).cuda()[None, None]
            with torch.random.fork_rng(devices=[image.device.index]):
                torch.manual_seed((seed + salt) % (2**31))
                torch.cuda.manual_seed((seed + salt) % (2**31))
                logits = model.deploy_logits(image)
                probability = logits.sigmoid().detach()
                foreground_mass = probability.sum().clamp_min(1.0)
                background_mass = (1 - probability).sum().clamp_min(1.0)
                weight = .5 * probability / foreground_mass + \
                    .5 * (1 - probability) / background_mass
                coefficient = (probability * (1 - probability) * weight).sqrt()
                generator = torch.Generator(device=image.device).manual_seed(
                    (seed + salt + 1) % (2**31))
                for probe in range(PROBES_PER_PATIENT):
                    bits = torch.randint(0, 2, logits.shape, device=image.device,
                                         generator=generator)
                    direction = (bits.float() * 2 - 1) * coefficient
                    gradients = torch.autograd.grad((logits * direction).sum(), parameters,
                                                    retain_graph=probe + 1 < PROBES_PER_PATIENT,
                                                    allow_unused=False)
                    vectors.append(torch.cat(tuple(item.detach().float().reshape(-1)
                                                   for item in gradients)))
    finally:
        model.train(was_training)
    if len(vectors) != PROBES_PER_PATIENT * len(cases):
        raise AssertionError("H4 current-site probe count mismatch")
    if len(vectors) < RANK_PER_SITE:
        raise ValueError("too few current-site probes for H4 rank")
    probes = torch.stack(vectors, dim=1)
    del vectors
    if probes.shape[0] != total or not torch.isfinite(probes).all():
        raise FloatingPointError("H4 predictive probes invalid")
    gram = probes.double().T @ probes.double()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    top_values = eigenvalues[-RANK_PER_SITE:].clamp_min(0)
    total_energy = eigenvalues.clamp_min(0).sum()
    captured = top_values.sum()
    if (not torch.isfinite(captured) or captured <= 0 or
            not math.isfinite(target_trace) or target_trace <= 0):
        raise FloatingPointError("H4 low-rank predictive energy invalid")
    basis = (probes @ eigenvectors[:, -RANK_PER_SITE:].float())
    basis.mul_(math.sqrt(target_trace / float(captured)))
    if not torch.isfinite(basis).all():
        raise FloatingPointError("H4 low-rank basis invalid")
    actual_trace = float(basis.double().square().sum())
    if abs(actual_trace - target_trace) > max(1e-3, target_trace * 1e-4):
        raise AssertionError("H4 low-rank evidence trace mismatch")
    audit = {"patients": len(cases), "probes": len(cases) * PROBES_PER_PATIENT,
             "rank": RANK_PER_SITE, "captured_energy_fraction": float(captured / total_energy),
             "target_trace": target_trace, "actual_trace": actual_trace,
             "parameters": total}
    del probes, gram, eigenvalues, eigenvectors
    return basis.detach().contiguous(), audit


class LowRankOnlineGaussian:
    """q(theta)=N(current mean, [D + U U^T]^-1), updated in place."""

    def __init__(self, model, diagonal, basis):
        self.named = named_prediction_parameters(model)
        self.precision = {name: diagonal[name].clone() for name, _ in self.named}
        self.basis = basis.clone().contiguous()
        self.site_start_trace = None
        self._check()

    def _check(self):
        size = sum(parameter.numel() for _, parameter in self.named)
        if (self.basis.ndim != 2 or self.basis.shape[0] != size or
                self.basis.shape[1] > 2 * RANK_PER_SITE or
                not torch.isfinite(self.basis).all() or
                any(not torch.isfinite(item).all() or (item <= 0).any()
                    for item in self.precision.values())):
            raise FloatingPointError("invalid H4 current Gaussian state")

    def transition(self):
        return 0.0

    def begin_site(self):
        if self.site_start_trace is not None:
            raise AssertionError("H4 site trace already started")
        self.site_start_trace = precision_trace(self.precision, self.named)

    @torch.no_grad()
    def update(self, alpha, curve):
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("H4 evidence weight invalid")
        gradients = []
        for name, parameter in self.named:
            gradient = parameter.grad
            item = curve[name]
            if (gradient is None or not torch.isfinite(gradient).all() or
                    not torch.isfinite(item).all() or (item < 0).any()):
                raise FloatingPointError("H4 gradient or curvature invalid")
            new_precision = self.precision[name] + DIAGONAL_SHARE * alpha * item
            if not torch.isfinite(new_precision).all() or (new_precision <= 0).any():
                raise FloatingPointError("H4 updated diagonal invalid")
            self.precision[name] = new_precision
            gradients.append(gradient.reshape(-1))
        diagonal = _flat(self.precision, self.named)
        gradient = torch.cat(gradients)
        scaled = self.basis / diagonal[:, None]
        leading = gradient / diagonal
        small = torch.eye(self.basis.shape[1], device=diagonal.device, dtype=torch.float64) + \
            self.basis.double().T @ scaled.double()
        rhs = self.basis.double().T @ leading.double()
        correction = torch.linalg.solve(small, rhs).float()
        movement = leading - scaled @ correction
        if not torch.isfinite(movement).all():
            raise FloatingPointError("H4 Woodbury movement invalid")
        offset = 0
        for _, parameter in self.named:
            width = parameter.numel()
            parameter.add_(movement[offset:offset + width].reshape_as(parameter), alpha=-alpha)
            offset += width

    def add_current_site_basis(self, model, cases, site, seed):
        if self.site_start_trace is None:
            raise AssertionError("H4 current-site trace was not started")
        target = precision_trace(self.precision, self.named) - self.site_start_trace
        if not math.isfinite(target) or target <= 0:
            raise FloatingPointError("H4 site diagonal evidence invalid")
        addition, audit = predictive_basis(model, cases, site, seed, target)
        self.basis = torch.cat((self.basis, addition), dim=1).contiguous()
        self.site_start_trace = None
        self._check()
        return audit

    def snapshot(self):
        return {"diagonal": {name: value.cpu().clone() for name, value in self.precision.items()},
                "basis": self.basis.cpu().clone()}


def woodbury_identity_check():
    diagonal = torch.tensor((2., 3., 5., 7.), dtype=torch.float64)
    basis = torch.tensor(((1., .2), (0., .8), (.4, .5), (1., -.3)), dtype=torch.float64)
    gradient = torch.tensor((.2, -1., 3., .5), dtype=torch.float64)
    scaled = basis / diagonal[:, None]
    actual = gradient / diagonal - scaled @ torch.linalg.solve(
        torch.eye(2, dtype=torch.float64) + basis.T @ scaled,
        basis.T @ (gradient / diagonal))
    expected = torch.linalg.solve(torch.diag(diagonal) + basis @ basis.T, gradient)
    error = float((actual - expected).abs().max())
    if error > 1e-12:
        raise AssertionError("H4 Woodbury identity P0 failed")
    return error
