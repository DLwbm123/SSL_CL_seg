"""A standard two-logit U-Net with optional hierarchical generative feedback."""
import math

import torch
from torch import nn

from sc_bayeseg.model import inference_net, normal_log_prob, sar, sar_logdet, student_log_prob
from ..unet_transfer_v45.run import unet


def standard_risk_from_logits(logits, label):
    """Match v45.risk exactly, including the empty-ground-truth Dice convention."""
    probability = logits.softmax(1)[:, 1]
    foreground = (label == 1).to(probability.dtype)
    count = foreground.sum((1, 2))
    dice = (2 * (probability * foreground).sum((1, 2)) + 1e-6) / (
        probability.sum((1, 2)) + count + 1e-6)
    return -logits.log_softmax(1).gather(1, label[:, None]).mean() + (
        (1 - dice) * (count > 0)).mean()


class FeedbackModel(nn.Module):
    def __init__(self, cut=False, size=128):
        super().__init__()
        self.cut = cut
        self.size = size
        self.unet = unet()
        torch.manual_seed(20261024)
        self.shape_posterior = inference_net(1, 2)
        self.appearance_posterior = inference_net(1, 2, blocks=2)
        self.segmentation_scale = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.segmentation_scale.weight)
        nn.init.zeros_(self.segmentation_scale.bias)
        self.shape_mean = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(16, 1, 3, padding=1))
        self.register_buffer('sar_logdet', torch.tensor(sar_logdet(size, .9)))

    @staticmethod
    def gaussian(raw, generator):
        mean, scale = raw.chunk(2, dim=1)
        logsd = -4 + 3 * torch.tanh(scale / 3)
        sample = mean + logsd.exp() * torch.randn(mean.shape, device=mean.device,
                                                  dtype=mean.dtype, generator=generator)
        return sample, normal_log_prob(sample, mean, logsd), mean, logsd

    def draw(self, image, generator, output=None):
        if image.shape[-2:] != (self.size, self.size):
            raise ValueError('wrong working image size')
        output = self.unet(image) if output is None else output
        shape, logq_shape, shape_mean, shape_logsd = self.gaussian(self.shape_posterior(image), generator)
        appearance, logq_appearance, _, appearance_logsd = self.gaussian(
            self.appearance_posterior(image), generator)
        mean = output.logits[:, 1:2] - output.logits[:, 0:1]
        feature = output.decoder_features['dec1']
        if self.cut:
            mean, feature = mean.detach(), feature.detach()
        raw_scale = self.segmentation_scale(feature)
        logsd = -4 + 3 * torch.tanh(raw_scale / 3)
        z = mean + logsd.exp() * torch.randn(mean.shape, device=mean.device,
                                             dtype=mean.dtype, generator=generator)
        logq_z = normal_log_prob(z, mean, logsd)
        probability = torch.cat((torch.zeros_like(z), z), 1).softmax(1)
        return dict(shape=shape, appearance=appearance, z=z, probability=probability,
                    logq=logq_shape + logq_appearance + logq_z,
                    shape_mean=shape_mean, shape_logsd=shape_logsd,
                    appearance_logsd=appearance_logsd, z_logsd=logsd)

    def objectives(self, image, label, generator):
        output = self.unet(image)
        task = standard_risk_from_logits(output.logits, label)
        d = self.draw(image, generator, output)
        observed = student_log_prob(image - d['shape'] - d['appearance'], .1)
        appearance_prior = normal_log_prob(d['appearance'], torch.zeros_like(d['appearance']),
                                           torch.full_like(d['appearance'], math.log(.5)))
        shape_prior = student_log_prob(
            sar(d['shape'] - self.shape_mean(d['probability']), .9), .2) + self.sar_logdet
        segmentation_prior = student_log_prob(sar(d['z'], .9), 1.) + self.sar_logdet
        bound = observed + appearance_prior + shape_prior + segmentation_prior - d['logq']
        variational = -bound.mean() / (self.size * self.size)
        return task, variational, output.logits, d

    def loss(self, image, label, generator):
        task, variational, _, d = self.objectives(image, label, generator)
        loss = task + .01 * variational
        return loss, dict(task=float(task.detach()), bound_per_pixel=float((-variational).detach()),
                          shape_logsd=float(d['shape_logsd'].detach().mean()),
                          z_logsd=float(d['z_logsd'].detach().mean()))

    @torch.no_grad()
    def predict(self, image, seed, draws=4):
        return self.unet(image).logits.softmax(1)

    @torch.no_grad()
    def predict_mc(self, image, seed, draws=4):
        generator = torch.Generator(device=image.device).manual_seed(seed)
        output = self.unet(image)
        return torch.stack([self.draw(image, generator, output)['probability']
                            for _ in range(draws)]).mean(0)
