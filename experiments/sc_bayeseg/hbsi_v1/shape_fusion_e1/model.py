"""Small matched residual heads atop the frozen full BayeSeg C-Cut model."""
import torch
from torch import nn

from ..feedback_c1.model import standard_risk_from_logits


class ResidualHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 2, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, logits, feature):
        return logits + self.layers(torch.cat((logits, torch.tanh(feature)), 1))


class Fusion:
    def __init__(self, frozen, head, kind):
        assert kind in ('shape', 'raw')
        self.frozen = frozen
        self.head = head
        self.kind = kind

    def eval(self):
        self.frozen.eval()
        self.head.eval()

    def logits(self, image):
        with torch.no_grad():
            base = self.frozen.unet(image).logits
            feature = (self.frozen.shape_posterior(image)[:, :1]
                       if self.kind == 'shape' else image)
        return self.head(base, feature)

    def loss(self, image, label):
        return standard_risk_from_logits(self.logits(image), label)

    @torch.no_grad()
    def predict(self, image, seed, draws=1):
        return self.logits(image).softmax(1)
