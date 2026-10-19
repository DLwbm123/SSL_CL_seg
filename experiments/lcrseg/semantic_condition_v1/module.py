"""Dense, identity-entry adaptation inspired by SCOPE's ProAdapter equations.

This is a convolutional adaptation, not the authors' EEG implementation.
The historical prototypes and backbone are immutable; retention is a hypothesis.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SemanticCondition(nn.Module):
    def __init__(self, prototypes, channels=16, shuffled=False):
        super().__init__()
        if prototypes.shape != (4, 3, 9 * channels):
            raise ValueError('expected four classes, three native 3x3 patch prototypes')
        if not torch.isfinite(prototypes).all() or (prototypes.norm(dim=-1) == 0).any():
            raise ValueError('invalid historical prototypes')
        self.register_buffer('prototypes', F.normalize(prototypes.detach().clone(), dim=-1))
        self.channels, self.shuffled = channels, shuffled
        self.self_path = nn.Sequential(nn.Linear(2 * channels, 32), nn.GELU(),
                                       nn.Linear(32, 2 * channels))
        self.semantic_path = nn.Sequential(nn.Conv2d(4, 32, 1), nn.GELU(),
                                           nn.Conv2d(32, 2 * channels, 1))
        for layer in (self.self_path[-1], self.semantic_path[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @torch.no_grad()
    def condition(self, features):
        b, c, h, w = features.shape
        if c != self.channels:
            raise ValueError('native feature width changed')
        patches = F.normalize(F.unfold(features, 3, padding=1), dim=1)
        scores = torch.einsum('cmd,bdn->bcmn', self.prototypes, patches)
        result = (scores.max(2).values / .1).softmax(1).reshape(b, 4, h, w)
        if self.shuffled:
            # Preserve the condition values, but break their local correspondence.
            result = result.flatten(2).roll((h * w) // 2, dims=2).reshape_as(result)
        return result

    def forward(self, features):
        condition = self.condition(features)
        mean = features.mean((-2, -1))
        std = features.var((-2, -1), unbiased=False).add(1e-6).sqrt()
        base = self.self_path(torch.cat((mean, std), 1))[:, :, None, None]
        scale, shift = (base + self.semantic_path(condition)).chunk(2, dim=1)
        # Express preactivations in residual units. This keeps the same +/-0.1
        # range and zero-entry function, with unit rather than 0.1 entry slope.
        return (1 + .1 * (scale / .1).tanh()) * features + .1 * (shift / .1).tanh()
