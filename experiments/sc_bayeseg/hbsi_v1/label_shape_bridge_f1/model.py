"""Label anchoring of the BayeSeg shape posterior with a frozen U-Net."""
import torch
from torch import nn

from ..feedback_c1.model import FeedbackModel, standard_risk_from_logits
from ..shape_fusion_e1.model import ResidualHead


class ShapeBridge(nn.Module):
    def __init__(self, label_feedback):
        super().__init__()
        self.label_feedback = label_feedback
        self.base = FeedbackModel(cut=True)
        self.base.unet.requires_grad_(False)
        self.head = ResidualHead()

    def train(self, mode=True):
        super().train(mode)
        self.base.unet.eval()
        return self

    def fused_logits(self, image, base_logits=None):
        if base_logits is None:
            base_logits = self.base.unet(image).logits
        shape_mean = self.base.shape_posterior(image)[:, :1]
        feature = shape_mean if self.label_feedback else shape_mean.detach()
        return self.head(base_logits.detach(), feature)

    def objectives(self, image, label, generator):
        _, variational, base_logits, d = self.base.objectives(image, label, generator)
        logits = self.fused_logits(image, base_logits)
        task = standard_risk_from_logits(logits, label)
        return task, variational, logits, d

    @torch.no_grad()
    def predict(self, image, seed, draws=1):
        return self.fused_logits(image).softmax(1)
