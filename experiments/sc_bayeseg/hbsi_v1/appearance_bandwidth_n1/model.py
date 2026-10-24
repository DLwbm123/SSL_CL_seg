"""A fixed spatial bottleneck on q(A|I) mean and scale."""
import torch
from torch import nn
from torch.nn import functional as F

from ..semantic_shape_l1.model import SemanticShape


class AppearanceFilter(nn.Module):
    def __init__(self, encoder, smooth):
        super().__init__()
        self.encoder=encoder
        self.smooth=smooth

    def forward(self, image):
        raw=self.encoder(image)
        if not self.smooth:
            return raw
        assert raw.shape[-2:]==(128,128)
        return F.interpolate(F.avg_pool2d(raw,32),size=(128,128),
                             mode='bilinear',align_corners=False)


class BandwidthModel(SemanticShape):
    def __init__(self,smooth):
        super().__init__(cut=False)
        self.base.appearance_posterior=AppearanceFilter(self.base.appearance_posterior,smooth)

    @staticmethod
    def wrapped_state(unwrapped):
        prefix='base.appearance_posterior.'
        return {(prefix+'encoder.'+k[len(prefix):] if k.startswith(prefix) else k):v
                for k,v in unwrapped.items()}

    @torch.no_grad()
    def appearance_mean(self,image):
        return self.base.appearance_posterior(image)[:,:1]
