"""Shared spatial code, joint image/mask decoder and weighted GYRO operator."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..gyro_native_v1.module import frame


def unit(x):
    return F.normalize(x, dim=1, eps=1e-6)


class SpatialProjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Conv2d(channels, 32, 1)

    def forward(self, bottleneck):
        return unit(self.linear(F.adaptive_avg_pool2d(bottleneck, (8, 8))))


class JointGenerator(nn.Module):
    def __init__(self, image_channels):
        super().__init__()
        self.start = nn.Conv2d(32, 128, 3, padding=1)
        blocks = []
        previous = 128
        for channels in (128, 96, 64, 32, 16):
            blocks.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(previous, channels, 3, padding=1), nn.GroupNorm(4, channels), nn.SiLU()))
            previous = channels
        self.trunk = nn.Sequential(*blocks)
        self.image = nn.Conv2d(16, image_channels, 3, padding=1)
        self.mask = nn.Conv2d(16, 4, 3, padding=1)

    def forward(self, code, output_size):
        if code.ndim != 4 or code.shape[1:] != (32, 8, 8):
            raise ValueError('expected B,32,8,8 joint spatial codes')
        hidden = self.trunk(self.start(code))
        hidden = F.interpolate(hidden, size=output_size, mode='bilinear', align_corners=False)
        return self.image(hidden), self.mask(hidden).softmax(1)


def sample_positions(batch, step, device):
    generator = torch.Generator().manual_seed(163 + step)
    result = []
    for top, left in ((0, 0), (0, 4), (4, 0), (4, 4)):
        delta = torch.randint(0, 4, (batch, 2), generator=generator)
        result.append((delta[:, 0] + top) * 8 + delta[:, 1] + left)
    return torch.stack(result, 1).to(device)


def rows(field, positions):
    if field.ndim != 4 or field.shape[1:] != (32, 8, 8):
        raise ValueError('expected B,32,8,8 field')
    flattened = field.flatten(2).transpose(1, 2)
    return flattened.gather(1, positions[..., None].expand(-1, -1, 32)).reshape(-1, 32)


def centroid(weak, strong1, strong2, reference=None):
    first = weak if reference is None else reference.detach()
    return .5 * first + .25 * strong1 + .25 * strong2


def anchored_loss(c, frames, stage_ids, beta=1/3):
    if c.ndim != 2 or c.shape[1] != 32 or len(c) == 0:
        raise ValueError('expected nonempty n,32 centroid matrix')
    if stage_ids.shape != (len(c),) or not 0 <= beta < 1:
        raise ValueError('invalid stage ids or beta')
    selected = frames.detach()[stage_ids]
    indices = torch.einsum('nd,nkd->nk', c.detach(), selected).argmax(1)
    anchors = selected[torch.arange(len(c), device=c.device), indices]
    y = (1-beta) * c + beta * anchors
    rank = min(y.shape)
    kappa = torch.linalg.svdvals(y).sum() / math.sqrt(len(y) * rank)
    if not torch.isfinite(kappa):
        raise FloatingPointError('nonfinite spatial spectral objective')
    return 1-kappa, dict(kappa=kappa, anchors=anchors, mixed=y)


def rate_gap(reference, decoded, epsilon=.5):
    if reference.ndim != 2 or decoded.ndim != 2 or reference.shape[1] != decoded.shape[1]:
        raise ValueError('invalid reference/decoded matrices')
    d = reference.shape[1]
    gram_r = reference.detach().T @ reference.detach() / len(reference)
    gram_d = decoded.T @ decoded / len(decoded)
    identity = torch.eye(d, device=decoded.device, dtype=decoded.dtype)
    def rate(gram):
        chol = torch.linalg.cholesky(identity + (d / epsilon**2) * gram)
        return chol.diag().log().sum()
    return (rate((gram_r + gram_d)/2) - .5*rate(gram_r) - .5*rate(gram_d)) / (
        d/2 * math.log1p(epsilon**-2))


def retained_bytes(*modules, code):
    tensors = [code]
    for module in modules:
        tensors.extend(module.state_dict().values())
    return sum(t.numel()*t.element_size() for t in tensors)
