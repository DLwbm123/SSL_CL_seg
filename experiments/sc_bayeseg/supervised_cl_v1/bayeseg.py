"""BayeSeg-inspired shape/appearance image hierarchy around the shared U-Net."""
import math

import torch
from torch import nn
from monai.networks.nets import BasicUNet

from sc_bayeseg.model import inference_net, normal_log_prob, sar, sar_logdet, student_log_prob


FEATURES = (32, 32, 64, 128, 256, 32)


def backbone():
    return BasicUNet(spatial_dims=2, in_channels=1, out_channels=2, features=FEATURES)


class BayeSeg(nn.Module):
    """Joint task and full S/A/image density; the main readout is the z mean."""

    def __init__(self):
        super().__init__()
        self.backbone = backbone()
        self.shape_posterior = inference_net(1, 2)
        self.appearance_posterior = inference_net(1, 2, blocks=2)
        self.logit_scale = nn.Conv2d(2, 1, 1)
        nn.init.zeros_(self.logit_scale.weight)
        nn.init.zeros_(self.logit_scale.bias)
        self.shape_decoder = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
                                           nn.Conv2d(16, 1, 3, padding=1))
        self.register_buffer("sar_logdet", torch.tensor(sar_logdet(256, .9), dtype=torch.float32))

    @staticmethod
    def gaussian(raw):
        mean, scale = raw.chunk(2, dim=1)
        logsd = -4 + 3 * torch.tanh(scale / 3)
        sample = mean + logsd.exp() * torch.randn_like(mean)
        return sample, normal_log_prob(sample, mean, logsd)

    def forward(self, image):
        logits = self.backbone(image)
        shape, logq_shape = self.gaussian(self.shape_posterior(image))
        appearance, logq_appearance = self.gaussian(self.appearance_posterior(image))
        z_mean = logits[:, 1:2] - logits[:, 0:1]
        z_logsd = -4 + 3 * torch.tanh(self.logit_scale(logits) / 3)
        z = z_mean + z_logsd.exp() * torch.randn_like(z_mean)
        logq_z = normal_log_prob(z, z_mean, z_logsd)
        probability = torch.cat((torch.zeros_like(z), z), dim=1).softmax(1)
        observed = student_log_prob(image - shape - appearance, .1)
        appearance_prior = normal_log_prob(appearance, torch.zeros_like(appearance),
                                           torch.full_like(appearance, math.log(.5)))
        shape_prior = student_log_prob(sar(shape - self.shape_decoder(probability), .9), .2) + self.sar_logdet
        segmentation_prior = student_log_prob(sar(z, .9), 1.) + self.sar_logdet
        variational = -(observed + appearance_prior + shape_prior + segmentation_prior
                        - logq_shape - logq_appearance - logq_z).mean() / (256 * 256)
        return logits, variational

    def predict_logits(self, image):
        return self.backbone(image)
