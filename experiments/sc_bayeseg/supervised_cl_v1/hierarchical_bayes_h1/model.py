"""Conditional appearance/anatomy hierarchy for the no-replay H1 study.

The deployment path uses only image-conditioned priors. Label-conditioned
recognition networks are used while training the current site.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from sc_bayeseg.supervised_cl_v1.bayeseg import backbone


PIXELS = 256 * 256
SDS = (.2, .1, .05)


def haar2(x):
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return ((a + b + c + d) / 2,
            torch.cat(((a - b + c - d) / 2,
                       (a + b - c - d) / 2,
                       (a - b - c + d) / 2), dim=1))


def inverse_haar2(low, high):
    h1, h2, h3 = high.chunk(3, dim=1)
    a = (low + h1 + h2 + h3) / 2
    b = (low - h1 + h2 - h3) / 2
    c = (low + h1 - h2 - h3) / 2
    d = (low - h1 - h2 + h3) / 2
    block = torch.stack((a, b, c, d), dim=2)
    return F.pixel_shuffle(block.reshape(low.shape[0], 4 * low.shape[1], *low.shape[-2:]), 2)


def haar(x):
    low128, v = haar2(x)
    c, u = haar2(low128)
    return c, u, v


def inverse_haar(c, u, v):
    return inverse_haar2(inverse_haar2(c, u), v)


def gaussian_kl(q_mean, q_logsd, p_mean, p_logsd):
    ratio = (2 * (q_logsd - p_logsd)).exp()
    distance = (q_mean - p_mean).square() * (-2 * p_logsd).exp()
    return .5 * (ratio + distance - 1 + 2 * (p_logsd - q_logsd)).flatten(1).sum(1)


def residual_head(channels, outputs):
    head = nn.Sequential(nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(16, 2 * outputs, 3, padding=1))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def refiner(channels):
    head = nn.Sequential(nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(16, 3, 3, padding=1))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class HierarchicalSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.appearance_prior = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 12))
        nn.init.zeros_(self.appearance_prior[-1].weight)
        nn.init.zeros_(self.appearance_prior[-1].bias)
        self.appearance_prior[-1].bias.data[6:] = math.log(.1)

        self.backbone = backbone()
        self.refine_u = refiner(1)
        self.refine_v = refiner(4)
        self.shape_logsd = nn.ParameterList([nn.Parameter(torch.tensor(math.log(sd))) for sd in SDS])

        self.appearance_recognition = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 12))
        nn.init.zeros_(self.appearance_recognition[-1].weight)
        nn.init.zeros_(self.appearance_recognition[-1].bias)
        self.recognition_stem = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU())
        self.recognize_c = residual_head(34, 1)
        self.recognize_u = residual_head(37, 3)
        self.recognize_v = residual_head(40, 3)

        coord = torch.linspace(-1, 1, 256)
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        basis = torch.stack((torch.cos(math.pi * xx), torch.cos(math.pi * yy),
                             torch.cos(math.pi * xx) * torch.cos(math.pi * yy)))
        self.register_buffer("image_basis", basis, persistent=False)
        self.register_buffer("coord_x", xx, persistent=False)
        self.register_buffer("coord_y", yy, persistent=False)

    def prediction_parameters(self):
        return tuple(self.appearance_prior.parameters()) + tuple(self.backbone.parameters()) + \
            tuple(self.refine_u.parameters()) + tuple(self.refine_v.parameters()) + tuple(self.shape_logsd)

    def appearance_parameters(self):
        return tuple(self.appearance_prior.parameters())

    def recognition_parameters(self):
        return tuple(self.appearance_recognition.parameters()) + tuple(self.recognition_stem.parameters()) + \
            tuple(self.recognize_c.parameters()) + tuple(self.recognize_u.parameters()) + \
            tuple(self.recognize_v.parameters())

    def image_stats(self, image):
        flat = image.flatten(1)
        quantiles = torch.quantile(flat, torch.tensor((.1, .2, .3, .5, .7, .8, .9), device=image.device), dim=1).T
        projections = (image * self.image_basis[None]).mean((-2, -1))
        return torch.cat((flat.mean(1, keepdim=True), flat.std(1, keepdim=True, unbiased=False),
                          quantiles, projections), dim=1)

    def label_stats(self, label):
        mask = label.float()[:, None]
        area = mask.mean((-2, -1))
        denom = mask.sum((-2, -1)).clamp_min(1)
        center_x = (mask * self.coord_x).sum((-2, -1)) / denom
        center_y = (mask * self.coord_y).sum((-2, -1)) / denom
        edge = ((mask[..., 1:, :] - mask[..., :-1, :]).abs().mean((-2, -1)) +
                (mask[..., :, 1:] - mask[..., :, :-1]).abs().mean((-2, -1))) / 2
        return torch.cat((area, center_x, center_y, edge), dim=1)

    def appearance_distribution(self, stats, label=None):
        raw = self.appearance_prior(stats)
        mean, logsd = raw.chunk(2, dim=1)
        logsd = logsd.clamp(-7, 0)
        if label is None:
            return mean, logsd, None
        residual = self.appearance_recognition(torch.cat((stats, self.label_stats(label)), dim=1))
        dm, ds = residual.chunk(2, dim=1)
        return mean + dm, (logsd + ds).clamp(-7, 0), (mean, logsd)

    def inverse_appearance(self, image, raw_a):
        bounded = torch.tanh(raw_a)
        alpha = bounded[:, 0:1, None, None] * math.log(2)
        kappa = bounded[:, 1:2, None, None] * math.log(2)
        delta = bounded[:, 2:3, None, None] * .25
        beta = bounded[:, 3:6, None, None] * .1
        field = alpha + (beta * self.image_basis[None]).sum(1, keepdim=True)
        return torch.sinh(torch.exp(-kappa) * torch.asinh((image - delta) * torch.exp(-field)))

    def forward_appearance(self, canonical, raw_a):
        bounded = torch.tanh(raw_a)
        alpha = bounded[:, 0:1, None, None] * math.log(2)
        kappa = bounded[:, 1:2, None, None] * math.log(2)
        delta = bounded[:, 2:3, None, None] * .25
        beta = bounded[:, 3:6, None, None] * .1
        field = alpha + (beta * self.image_basis[None]).sum(1, keepdim=True)
        return field.exp() * torch.sinh(kappa.exp() * torch.asinh(canonical)) + delta

    @staticmethod
    def sample(mean, logsd, noise=None):
        return mean + logsd.exp() * (torch.randn_like(mean) if noise is None else noise)

    def _field(self, image, raw_a, label=None, noise=None):
        canonical = self.inverse_appearance(image, raw_a)
        logits = self.backbone(canonical)
        base = logits[:, 1:2] - logits[:, 0:1]
        base_c, base_u, base_v = haar(base)
        p_c_logsd = self.shape_logsd[0].clamp(-7, 1).expand_as(base_c)
        p_u_logsd = self.shape_logsd[1].clamp(-7, 1).expand_as(base_u)
        p_v_logsd = self.shape_logsd[2].clamp(-7, 1).expand_as(base_v)
        if label is None:
            c = self.sample(base_c, p_c_logsd, None if noise is None else noise[0])
            u_mean = base_u + self.refine_u(c / 4)
            u = self.sample(u_mean, p_u_logsd, None if noise is None else noise[1])
            v_mean = base_v + self.refine_v(torch.cat((F.interpolate(c / 4, scale_factor=2),
                                                       F.interpolate(u / 2, scale_factor=2)), dim=1))
            v = self.sample(v_mean, p_v_logsd, None if noise is None else noise[2])
            return inverse_haar(c, u, v), None

        onehot = F.one_hot(label.long(), 2).permute(0, 3, 1, 2).float()
        features = self.recognition_stem(canonical)
        feature64 = F.avg_pool2d(features, 4)
        feature128 = F.avg_pool2d(features, 2)
        label64 = F.interpolate(onehot, size=(64, 64), mode="nearest")
        label128 = F.interpolate(onehot, size=(128, 128), mode="nearest")
        prob64 = F.interpolate(logits.softmax(1), size=(64, 64), mode="bilinear", align_corners=False)
        prob128 = F.interpolate(logits.softmax(1), size=(128, 128), mode="bilinear", align_corners=False)

        dc, sc = self.recognize_c(torch.cat((feature64, label64), dim=1)).chunk(2, dim=1)
        q_c_mean, q_c_logsd = base_c + dc, (p_c_logsd + sc).clamp(-7, 1)
        c = self.sample(q_c_mean, q_c_logsd)
        kl = gaussian_kl(q_c_mean, q_c_logsd, base_c, p_c_logsd)

        p_u_mean = base_u + self.refine_u(c / 4)
        du, su = self.recognize_u(torch.cat((feature64, label64, c / 4, prob64), dim=1)).chunk(2, dim=1)
        q_u_mean, q_u_logsd = p_u_mean + du, (p_u_logsd + su).clamp(-7, 1)
        u = self.sample(q_u_mean, q_u_logsd)
        kl = kl + gaussian_kl(q_u_mean, q_u_logsd, p_u_mean, p_u_logsd)

        up_c = F.interpolate(c / 4, scale_factor=2)
        up_u = F.interpolate(u / 2, scale_factor=2)
        p_v_mean = base_v + self.refine_v(torch.cat((up_c, up_u), dim=1))
        dv, sv = self.recognize_v(torch.cat((feature128, label128, up_c, up_u, prob128), dim=1)).chunk(2, dim=1)
        q_v_mean, q_v_logsd = p_v_mean + dv, (p_v_logsd + sv).clamp(-7, 1)
        v = self.sample(q_v_mean, q_v_logsd)
        kl = kl + gaussian_kl(q_v_mean, q_v_logsd, p_v_mean, p_v_logsd)
        return inverse_haar(c, u, v), kl

    def deploy_logits(self, image, noise=None):
        stats = self.image_stats(image)
        a_mean, a_logsd, _ = self.appearance_distribution(stats)
        raw_a = self.sample(a_mean, a_logsd, None if noise is None else noise[0])
        field, _ = self._field(image, raw_a, noise=None if noise is None else noise[1:])
        return field

    def loss(self, image, label):
        stats = self.image_stats(image)
        q_mean, q_logsd, p = self.appearance_distribution(stats, label)
        raw_a = self.sample(q_mean, q_logsd)
        q_field, shape_kl = self._field(image, raw_a, label)
        appearance_kl = gaussian_kl(q_mean, q_logsd, *p)
        local_nll = F.binary_cross_entropy_with_logits(q_field[:, 0], label.float())
        local = local_nll + ((shape_kl + appearance_kl) / PIXELS).mean()

        deploy_field = self.deploy_logits(image)
        deploy_ce = F.binary_cross_entropy_with_logits(deploy_field[:, 0], label.float())
        foreground = deploy_field.sigmoid()[:, 0]
        target = label.float()
        dice = (2 * (foreground * target).sum((-2, -1)) + 1) / \
            (foreground.sum((-2, -1)) + target.sum((-2, -1)) + 1)
        total = .5 * local + .5 * deploy_ce + (1 - dice).mean()
        return total, {"local": local.detach(), "deploy_ce": deploy_ce.detach(),
                       "deploy_dice_loss": (1 - dice).mean().detach()}

    @torch.no_grad()
    def predict_probability(self, image, seed):
        generator = torch.Generator(device=image.device).manual_seed(seed)
        result = 0
        for _ in range(8):
            noises = (torch.randn((len(image), 6), device=image.device, generator=generator),
                      torch.randn((len(image), 1, 64, 64), device=image.device, generator=generator),
                      torch.randn((len(image), 3, 64, 64), device=image.device, generator=generator),
                      torch.randn((len(image), 3, 128, 128), device=image.device, generator=generator))
            for sign in (1, -1):
                draw = tuple(sign * item for item in noises)
                result = result + self.deploy_logits(image, draw).sigmoid()
        return result / 16
