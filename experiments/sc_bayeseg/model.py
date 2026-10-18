"""Normalized SC-BayeSeg model and local proposal (design equations 45/54/73).

The residual shape/appearance inference follows the architectural idea in
obiyoag/BayeSeg, MedIA version ed6a45b. This is an independent implementation;
the upstream Gumbel probabilities, detached precision loss and Dice are not used.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def normal_log_prob(value, mean, logsd):
    return (-.5 * ((value - mean) * (-logsd).exp()).square()
            - logsd - .5 * math.log(2 * math.pi)).flatten(1).sum(1)


def student_log_prob(residual, scale, channels=1):
    """Gamma shape=2, rate=scale²; marginal variance=scale² per channel."""
    a, b = 2., scale ** 2
    d = channels
    squared = residual.square().sum(1)
    constant = (a * math.log(b) + math.lgamma(a + d / 2)
                - math.lgamma(a) - d / 2 * math.log(2 * math.pi))
    return (constant - (a + d / 2) * torch.log(b + squared / 2)).flatten(1).sum(1)


def sar(value, alpha=.9):
    neighbors = sum(torch.roll(value, shift, dim) for shift, dim in
                    [(1, -1), (-1, -1), (1, -2), (-1, -2)]) / 4
    return value - alpha * neighbors


def sar_logdet(size, alpha=.9):
    k = torch.arange(size, dtype=torch.float64)
    eigenvalues = (torch.cos(2 * math.pi * k / size)[:, None]
                   + torch.cos(2 * math.pi * k / size)[None, :]) / 2
    return float(torch.log1p(-alpha * eigenvalues).sum())


class BayesianLayer(nn.Module):
    """All parameters in the two generators have genuine Gaussian posteriors."""
    def __init__(self, incoming, outgoing, kernel=None):
        super().__init__()
        self.kernel = kernel
        shape = (outgoing, incoming) if kernel is None else (outgoing, incoming, kernel, kernel)
        fan = math.prod(shape[1:])
        self.mean_w = nn.Parameter(torch.randn(shape) * math.sqrt(2 / fan))
        self.mean_b = nn.Parameter(torch.zeros(outgoing))
        self.logsd_w = nn.Parameter(torch.full(shape, math.log(.2 / math.sqrt(fan))))
        self.logsd_b = nn.Parameter(torch.full((outgoing,), math.log(.2)))
        for part in ('w', 'b'):
            for kind in ('mean', 'logsd'):
                tensor = getattr(self, kind + '_' + part).detach().clone()
                self.register_buffer('initial_' + kind + '_' + part, tensor.clone())
                self.register_buffer('prior_' + kind + '_' + part, tensor.clone())
        self.sampled = None

    def draw(self):
        self.sampled = tuple(getattr(self, 'mean_' + p) + getattr(self, 'logsd_' + p).exp()
                             * torch.randn_like(getattr(self, 'mean_' + p)) for p in ('w', 'b'))

    def forward(self, value):
        if self.sampled is None:
            raise RuntimeError('draw generator parameters before using a probability factor')
        w, b = self.sampled
        return F.linear(value, w, b) if self.kernel is None else F.conv2d(value, w, b, padding=self.kernel // 2)

    def kl(self):
        result = self.mean_b.sum() * 0
        for p in ('w', 'b'):
            mean, logsd = getattr(self, 'mean_' + p), getattr(self, 'logsd_' + p)
            oldmean, oldsd = getattr(self, 'prior_mean_' + p), getattr(self, 'prior_logsd_' + p)
            result = result + (oldsd - logsd + .5 * (
                ((mean - oldmean) * (-oldsd).exp()).square()
                + torch.exp(2 * (logsd - oldsd)) - 1)).sum()
        return result

    @torch.no_grad()
    def set_prior(self, recursive):
        for p in ('w', 'b'):
            for kind in ('mean', 'logsd'):
                source = getattr(self, kind + '_' + p) if recursive else getattr(self, 'initial_' + kind + '_' + p)
                getattr(self, 'prior_' + kind + '_' + p).copy_(source)


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(width, width, 3, padding=1))

    def forward(self, value):
        return value + self.layers(value)


def inference_net(incoming, outgoing, width=16, blocks=4):
    return nn.Sequential(nn.Conv2d(incoming, width, 3, padding=1), nn.ReLU(),
                         *[Residual(width) for _ in range(blocks)], nn.ReLU(),
                         nn.Conv2d(width, outgoing, 3, padding=1))


class Model(nn.Module):
    num_classes = 4

    def __init__(self, size=128):
        super().__init__()
        self.size = size
        self.code_dim = 16
        self.shape_encoder = inference_net(1, 2)
        self.appearance_encoder = inference_net(1, 2, blocks=2)
        self.logit_encoder = inference_net(1, 6)
        self.code_encoder = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                                          nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 32))
        self.anatomy = nn.ModuleList([BayesianLayer(16, 16 * 4 * 4),
                                     BayesianLayer(16, 16, 3), BayesianLayer(16, 3, 3)])
        self.structure = nn.ModuleList([BayesianLayer(4, 16, 3),
                                       BayesianLayer(16, 16, 3), BayesianLayer(16, 1, 3)])
        self.register_buffer('logdet', torch.tensor(sar_logdet(size), dtype=torch.float32))
        self.calls = dict(proposal_images=0, density_images=0, generator_draws=0,
                          refinement_derivatives=0, predictive_images=0)

    def bayes_layers(self):
        return [layer for layer in self.modules() if isinstance(layer, BayesianLayer)]

    def draw(self):
        for layer in self.bayes_layers():
            layer.draw()
        self.calls['generator_draws'] += 1

    def clear_draw(self):
        for layer in self.bayes_layers():
            layer.sampled = None

    def kl(self):
        return sum(layer.kl() for layer in self.bayes_layers())

    def set_prior(self, recursive):
        for layer in self.bayes_layers():
            layer.set_prior(recursive)

    def anatomy_mean(self, c):
        value = F.relu(self.anatomy[0](c)).reshape(-1, 16, 4, 4)
        value = F.interpolate(value, (self.size, self.size), mode='bilinear', align_corners=False)
        return self.anatomy[2](F.relu(self.anatomy[1](value)))

    def structure_mean(self, probabilities):
        value = F.relu(self.structure[0](probabilities))
        return self.structure[2](F.relu(self.structure[1](value)))

    @staticmethod
    def gaussian(raw, sample=True, shift=None):
        mean, rawsd = raw.chunk(2, dim=1)
        # Smooth bounds define the proposal family; no clipping in the density.
        logsd = -4 + 3 * torch.tanh(rawsd / 3)
        if shift is not None:
            mean, logsd = mean + shift[0], logsd + shift[1]
        value = mean + logsd.exp() * torch.randn_like(mean) if sample else mean
        return value, normal_log_prob(value, mean, logsd), mean, logsd

    def proposal(self, image, shifts=None, sample=True):
        shifts = {} if shifts is None else shifts
        self.calls['proposal_images'] += len(image)
        x = self.gaussian(self.shape_encoder(image), sample, shifts.get('x'))
        m = self.gaussian(self.appearance_encoder(image), sample, shifts.get('m'))
        ell = self.gaussian(self.logit_encoder(x[0]), sample, shifts.get('ell'))
        c = self.gaussian(self.code_encoder(ell[0]), sample, shifts.get('c'))
        h = {key: value[0] for key, value in zip(('x', 'm', 'ell', 'c'), (x, m, ell, c))}
        return h, sum(value[1] for value in (x, m, ell, c))

    def log_joint(self, image, h, label=None, detach_semantics=False):
        self.calls['density_images'] += len(image)
        ell = h['ell']
        logits = torch.cat([torch.zeros_like(ell[:, :1]), ell], 1)
        probabilities = logits.softmax(1)
        semantic = probabilities.detach() if detach_semantics else probabilities
        terms = dict(
            code=normal_log_prob(h['c'], torch.zeros_like(h['c']), torch.zeros_like(h['c'])),
            appearance=normal_log_prob(h['m'], torch.zeros_like(h['m']), torch.full_like(h['m'], math.log(.5))),
            image=student_log_prob(image - h['x'] - h['m'], .1),
            shape=student_log_prob(sar(h['x'] - self.structure_mean(semantic)), .2) + self.logdet
        )
        # Each logit channel has its own independent Gamma precision.
        residual = sar(ell - self.anatomy_mean(h['c']))
        terms['logits'] = sum(student_log_prob(residual[:, j:j+1], 1.)
                              for j in range(3)) + 3 * self.logdet
        terms['labels'] = (logits.sum((1, 2, 3)) * 0 if label is None else
                           -F.cross_entropy(logits, label, reduction='none').sum((1, 2)))
        return sum(terms.values()), terms, probabilities

    def bound(self, image, label=None):
        h, logq = self.proposal(image)
        joint, terms, p = self.log_joint(image, h, label)
        diagnostics = {f'predicted_fraction_{c}': float((p.detach().argmax(1) == c).float().mean())
                       for c in range(4)}
        diagnostics.update(shape_rms=float(h['x'].detach().square().mean().sqrt()),
                           appearance_rms=float(h['m'].detach().square().mean().sqrt()))
        return joint - logq, terms, diagnostics

    def prediction(self, image, corrected=True, refine=3, parameter_samples=2, local_samples=4):
        native = image.shape[-2:]
        image = F.interpolate(image, (self.size, self.size), mode='bilinear', align_corners=False)
        self.calls['predictive_images'] += len(image)
        if not corrected:
            h, _ = self.proposal(image, sample=False)
            p = torch.cat([torch.zeros_like(h['ell'][:, :1]), h['ell']], 1).softmax(1)
            return F.interpolate(p, native, mode='bilinear', align_corners=False), {}
        flags = {name: p.requires_grad for name, p in self.named_parameters()}
        self.requires_grad_(False)
        try:
            with torch.enable_grad():
                shifts = {key: (torch.zeros(shape, device=image.device, requires_grad=True),
                                torch.zeros(shape, device=image.device, requires_grad=True))
                          for key, shape in dict(x=(len(image), 1, self.size, self.size),
                                                 m=(len(image), 1, self.size, self.size),
                                                 ell=(len(image), 3, self.size, self.size),
                                                 c=(len(image), 16)).items()}
                for _ in range(refine):
                    self.draw()
                    h, logq = self.proposal(image, shifts)
                    joint, _, _ = self.log_joint(image, h)
                    # Sum retains per-instance update scale, independent of batch size.
                    loss = -(joint - logq).sum() / (self.size ** 2)
                    flat = [value for pair in shifts.values() for value in pair]
                    gradients = torch.autograd.grad(loss, flat)
                    self.calls['refinement_derivatives'] += 1
                    updated = [(p - .05 * g).detach().requires_grad_() for p, g in zip(flat, gradients)]
                    shifts = {key: tuple(updated[2*i:2*i+2]) for i, key in enumerate(shifts)}
            weights, predictions = [], []
            with torch.no_grad():
                for _ in range(parameter_samples):
                    self.draw()
                    for _ in range(local_samples):
                        h, logq = self.proposal(image, shifts)
                        joint, _, p = self.log_joint(image, h)
                        weights.append(joint - logq)
                        predictions.append(p)
                weights = torch.stack(weights).softmax(0)
                p = (weights[:, :, None, None, None] * torch.stack(predictions)).sum(0)
                ess = 1 / weights.square().sum(0)
                aggregate = weights.reshape(parameter_samples, local_samples, -1).sum(1)
                diagnostics = dict(ess=float(ess.mean()), maximum_weight=float(weights.max(0).values.mean()),
                                   parameter_ess=float((1 / aggregate.square().sum(0)).mean()))
                return F.interpolate(p, native, mode='bilinear', align_corners=False), diagnostics
        finally:
            self.clear_draw()
            for name, p in self.named_parameters():
                p.requires_grad_(flags[name])

    def forward(self, image):
        from lcrseg.contracts import SegModelOutput
        probabilities, _ = self.prediction(image)
        return SegModelOutput(probabilities.clamp_min(1e-30).log(), probabilities)
