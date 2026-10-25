"""Soft foreground-balanced predictive Fisher diagonal for H2."""

import torch

from ..hierarchical_bayes_h1.online import named_prediction_parameters


def balanced_curvature(model, image, seed, probes=2):
    """Give predicted foreground and background equal total Fisher mass.

    The weights are analytic functions of the current model's predictions on
    the current site's training batch. They use no old examples or labels.
    """
    named = named_prediction_parameters(model)
    params = tuple(parameter for _, parameter in named)
    with torch.random.fork_rng(devices=[image.device.index]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        logits = model.deploy_logits(image)
        probability = logits.sigmoid().detach()
        foreground = probability.sum().clamp_min(1.0)
        background = (1 - probability).sum().clamp_min(1.0)
        weight = 0.5 * probability / foreground + 0.5 * (1 - probability) / background
        coefficient = (probability * (1 - probability) * weight).sqrt()
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
        raise FloatingPointError("non-finite or negative balanced curvature")
    return estimate
