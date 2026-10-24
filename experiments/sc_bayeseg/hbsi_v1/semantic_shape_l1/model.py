"""A semantic shape posterior with paired generative-gradient routing."""
import math

import torch
from torch import nn

from sc_bayeseg.model import inference_net, normal_log_prob, sar, student_log_prob
from ..feedback_c1.model import FeedbackModel, standard_risk_from_logits


class SemanticShape(nn.Module):
    def __init__(self, cut):
        super().__init__()
        self.cut = cut
        self.base = FeedbackModel(cut=True)
        self.base.unet.requires_grad_(False)
        self.base.shape_posterior = inference_net(1, 2)
        nn.init.zeros_(self.base.shape_posterior[-1].weight)
        nn.init.zeros_(self.base.shape_posterior[-1].bias)
        self.shape_to_image = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1))

    def train(self, mode=True):
        super().train(mode)
        self.base.unet.eval()
        return self

    def shape_parameters(self):
        return list(self.base.shape_posterior.parameters())

    def means(self, image, unet_output=None):
        output = self.base.unet(image) if unet_output is None else unet_output
        residual, raw_scale = self.base.shape_posterior(image).chunk(2, 1)
        mean = output.logits[:, 1:2].detach() - output.logits[:, 0:1].detach() + residual
        logsd = -4 + 3 * torch.tanh(raw_scale / 3)
        return mean, logsd, output

    def logits(self, image):
        mean, _, _ = self.means(image)
        return torch.cat((torch.zeros_like(mean), mean), 1)

    def objectives(self, image, label, generator):
        mean, logsd, output = self.means(image)
        logits = torch.cat((torch.zeros_like(mean), mean), 1)
        task = standard_risk_from_logits(logits, label)
        qmean, qlogsd = ((mean.detach(), logsd.detach()) if self.cut else (mean, logsd))
        shape = qmean + qlogsd.exp() * torch.randn(qmean.shape, device=qmean.device,
                                                   dtype=qmean.dtype, generator=generator)
        logq_shape = normal_log_prob(shape, qmean, qlogsd)
        appearance, logq_appearance, _, appearance_logsd = self.base.gaussian(
            self.base.appearance_posterior(image), generator)
        z_mean = output.logits[:, 1:2].detach() - output.logits[:, 0:1].detach()
        raw_z_scale = self.base.segmentation_scale(output.decoder_features['dec1'].detach())
        z_logsd = -4 + 3 * torch.tanh(raw_z_scale / 3)
        z = z_mean + z_logsd.exp() * torch.randn(z_mean.shape, device=z_mean.device,
                                                 dtype=z_mean.dtype, generator=generator)
        logq_z = normal_log_prob(z, z_mean, z_logsd)
        probability = torch.cat((torch.zeros_like(z), z), 1).softmax(1)
        image_shape = self.shape_to_image(torch.tanh(shape))
        observed = student_log_prob(image - image_shape - appearance, .1)
        appearance_prior = normal_log_prob(appearance, torch.zeros_like(appearance),
                                           torch.full_like(appearance, math.log(.5)))
        shape_prior = student_log_prob(
            sar(shape - self.base.shape_mean(probability), .9), .2) + self.base.sar_logdet
        segmentation_prior = student_log_prob(sar(z, .9), 1.) + self.base.sar_logdet
        bound = observed + appearance_prior + shape_prior + segmentation_prior \
            - logq_shape - logq_appearance - logq_z
        variational = -bound.mean() / (self.base.size * self.base.size)
        return task, variational, logits, dict(shape=shape, appearance=appearance,
            image_shape=image_shape, shape_logsd=logsd, appearance_logsd=appearance_logsd,
            z_logsd=z_logsd)

    @torch.no_grad()
    def predict(self, image, seed, draws=1):
        return self.logits(image).softmax(1)
