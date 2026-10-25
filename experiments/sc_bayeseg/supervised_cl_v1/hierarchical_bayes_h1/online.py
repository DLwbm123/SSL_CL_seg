"""One current diagonal-Gaussian state for H1's generalized online update."""

import hashlib
import math

import numpy as np
import torch

from .model import PIXELS


def named_prediction_parameters(model):
    groups = ("appearance_prior", "backbone", "refine_u", "refine_v", "shape_logsd")
    return tuple((name, parameter) for name, parameter in model.named_parameters()
                 if name.split(".")[0] in groups)


def initial_precision(named):
    result = {}
    for name, parameter in named:
        if parameter.ndim >= 2 and name.endswith("weight"):
            fan_in = parameter.shape[1] * math.prod(parameter.shape[2:])
            value = fan_in / 2
        else:
            value = 1
        result[name] = torch.full_like(parameter, value, dtype=torch.float32)
    return result


def _hash_positions(named, count, seed):
    """Choose fixed non-appearance coordinates without gradients or labels."""
    hashes, ranges = [], []
    offset = 0
    for name, parameter in named:
        if name.startswith("appearance_prior."):
            continue
        digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
        salt = np.frombuffer(digest[:8], dtype="<u8")[0]
        indices = np.arange(parameter.numel(), dtype=np.uint64)
        value = indices + salt
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xbf58476d1ce4e5b9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94d049bb133111eb)
        value ^= value >> np.uint64(31)
        hashes.append(value)
        ranges.append((name, offset, offset + parameter.numel()))
        offset += parameter.numel()
    order = np.argpartition(np.concatenate(hashes), count - 1)[:count]
    masks = {name: torch.zeros_like(parameter, dtype=torch.bool) for name, parameter in named}
    for name, start, end in ranges:
        positions = order[(order >= start) & (order < end)] - start
        if len(positions):
            masks[name].view(-1)[torch.as_tensor(positions.copy(), device=masks[name].device)] = True
    return masks


def masks(named, seed):
    appearance_count = sum(parameter.numel() for name, parameter in named
                           if name.startswith("appearance_prior."))
    if appearance_count <= 0:
        raise ValueError("appearance predictor has no parameters")
    selected = _hash_positions(named, appearance_count, seed)
    if sum(int(mask.sum()) for mask in selected.values()) != appearance_count:
        raise ValueError("non-appearance mask size mismatch")
    return selected


def curvature(model, image, seed, probes=2):
    """Diagonal GGN proxy for mean binary label CE, using random VJPs."""
    named = named_prediction_parameters(model)
    params = tuple(parameter for _, parameter in named)
    with torch.random.fork_rng(devices=[image.device.index]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        logits = model.deploy_logits(image)
        probability = logits.sigmoid().detach()
        coefficient = (probability * (1 - probability) / (len(image) * PIXELS)).sqrt()
        generator = torch.Generator(device=image.device).manual_seed(seed + 1)
        estimate = {name: torch.zeros_like(parameter, dtype=torch.float32)
                    for name, parameter in named}
        for probe in range(probes):
            bits = torch.randint(0, 2, logits.shape, device=image.device, generator=generator)
            direction = (bits.float() * 2 - 1) * coefficient
            vector = torch.autograd.grad((logits * direction).sum(), params,
                                         retain_graph=probe + 1 < probes, allow_unused=False)
            for (name, _), item in zip(named, vector):
                estimate[name].addcmul_(item.detach(), item.detach(), value=1 / probes)
    if any(not torch.isfinite(value).all() or (value < 0).any() for value in estimate.values()):
        raise FloatingPointError("non-finite or negative curvature proxy")
    return estimate


class OnlineGaussian:
    def __init__(self, model, precision, variant, seed):
        self.named = named_prediction_parameters(model)
        self.precision = {name: precision[name].clone() for name, _ in self.named}
        self.variant = variant
        self.appearance_count = sum(parameter.numel() for name, parameter in self.named
                                    if name.startswith("appearance_prior."))
        self.parameter_count = sum(parameter.numel() for _, parameter in self.named)
        self.mask = masks(self.named, seed) if variant == "F_perm" else None
        self._check()

    def _check(self):
        if any(not torch.isfinite(item).all() or (item <= 0).any()
               for item in self.precision.values()):
            raise FloatingPointError("current precision must be positive and finite")

    def transition(self):
        if self.variant == "F0":
            return 0.0
        budget = 0.0
        for name, _ in self.named:
            current = self.precision[name]
            if self.variant == "F_all":
                ratio = self.appearance_count / self.parameter_count
                release = torch.full_like(current, ratio)
            elif self.variant in ("F_A", "F_erase"):
                release = torch.ones_like(current) if name.startswith("appearance_prior.") else torch.zeros_like(current)
            elif self.variant == "F_perm":
                release = self.mask[name].float()
            else:
                raise ValueError(f"unknown online variant {self.variant}")
            budget += float(release.sum())
            self.precision[name] = current / (1 + release)
        self._check()
        return budget

    @torch.no_grad()
    def update(self, alpha, curve):
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("evidence weight must be finite and positive")
        for name, parameter in self.named:
            gradient = parameter.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise FloatingPointError("non-finite prediction gradient")
            new_precision = self.precision[name] + alpha * curve[name]
            if not torch.isfinite(new_precision).all() or (new_precision <= 0).any():
                raise FloatingPointError("non-finite updated precision")
            parameter.addcdiv_(gradient, new_precision, value=-alpha)
            self.precision[name] = new_precision
        self._check()

    def erase_precision(self):
        if self.variant != "F_erase":
            raise ValueError("precision reset belongs only to F_erase")
        self.precision = initial_precision(self.named)

    def snapshot(self):
        return {name: value.cpu().clone() for name, value in self.precision.items()}


def scalar_conjugate_check():
    results = []
    y, prior_mean, prior_precision, noise_precision, evidence = 1.7, -.4, 2.3, 1.2, 9.0
    target_precision = prior_precision + evidence * noise_precision
    target_mean = (prior_precision * prior_mean + evidence * noise_precision * y) / target_precision
    for steps in (1, 10, 100, 6000):
        mean, precision = prior_mean, prior_precision
        alpha = evidence / steps
        for _ in range(steps):
            gradient = noise_precision * (mean - y)
            precision += alpha * noise_precision
            mean -= alpha * gradient / precision
        results.append((abs(mean - target_mean), abs(precision - target_precision)))
    maximum = tuple(max(item[index] for item in results) for index in (0, 1))
    if maximum[0] > 1e-10 or maximum[1] > 1e-10:
        raise AssertionError("online Gaussian scalar update is inconsistent")
    return maximum
