"""Feature-only supervised readout of a full BayeSeg shape or appearance posterior."""
import torch
from torch import nn

from ..feedback_c1.model import FeedbackModel, standard_risk_from_logits


class LatentReadout(nn.Module):
    def __init__(self, use_shape):
        super().__init__()
        self.use_shape = use_shape
        self.base = FeedbackModel(cut=True)
        self.base.unet.requires_grad_(False)
        self.head = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(16, 2, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.unet.eval()
        return self

    def logits(self, image, base_logits=None):
        if base_logits is None:
            base_logits = self.base.unet(image).logits
        posterior = self.base.shape_posterior if self.use_shape else self.base.appearance_posterior
        feature = torch.tanh(posterior(image)[:, :1])
        head_logits = self.head(feature)
        fused = base_logits.detach() + .5 * torch.tanh(head_logits)
        return fused, head_logits

    def objectives(self, image, label, generator):
        _, variational, base_logits, d = self.base.objectives(image, label, generator)
        fused, head_logits = self.logits(image, base_logits)
        fused_task = standard_risk_from_logits(fused, label)
        direct_task = standard_risk_from_logits(head_logits, label)
        return fused_task, direct_task, variational, fused, d

    @torch.no_grad()
    def predict(self, image, seed, draws=1):
        return self.logits(image)[0].softmax(1)
