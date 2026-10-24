"""Independent full-field BayeSeg-like variational model for the A1 gate."""
import math

import torch
from torch import nn

from lcrseg.models.unet import UNet2D
from sc_bayeseg.model import inference_net, normal_log_prob, sar, sar_logdet, student_log_prob
from ..paired_semantic_v19.method import balanced_risk


class FullFieldBayeSeg(nn.Module):
    def __init__(self, size=128):
        super().__init__()
        self.size = size
        self.shape_posterior = inference_net(1, 2)
        self.appearance_posterior = inference_net(1, 2, blocks=2)
        self.segmentation_posterior = UNet2D(1, 2, base_channels=16)
        self.segmentation_posterior.projection_head = nn.Identity()
        self.shape_mean = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(16, 1, 3, padding=1))
        self.register_buffer('sar_logdet', torch.tensor(sar_logdet(size, .9)))

    @staticmethod
    def gaussian(raw, generator):
        mean, scale = raw.chunk(2, dim=1)
        logsd = -4 + 3 * torch.tanh(scale / 3)
        sample = mean + logsd.exp() * torch.randn(mean.shape, device=mean.device,
                                                  dtype=mean.dtype, generator=generator)
        return sample, normal_log_prob(sample, mean, logsd)

    def draw(self, image, generator):
        if image.shape[-2:] != (self.size, self.size):
            raise ValueError('wrong working image size')
        shape, logq_shape = self.gaussian(self.shape_posterior(image), generator)
        appearance, logq_appearance = self.gaussian(self.appearance_posterior(image), generator)
        z, logq_z = self.gaussian(self.segmentation_posterior(shape).logits, generator)
        probability = torch.cat((torch.zeros_like(z), z), 1).softmax(1)
        return shape, appearance, z, probability, logq_shape + logq_appearance + logq_z

    def loss(self, image, label, generator):
        shape, appearance, z, probability, logq = self.draw(image, generator)
        pixels = self.size * self.size
        observed = student_log_prob(image - shape - appearance, .1)
        appearance_prior = normal_log_prob(appearance, torch.zeros_like(appearance),
                                           torch.full_like(appearance, math.log(.5)))
        shape_prior = student_log_prob(sar(shape - self.shape_mean(probability), .9), .2) + self.sar_logdet
        segmentation_prior = student_log_prob(sar(z, .9), 1.) + self.sar_logdet
        bound = observed + appearance_prior + shape_prior + segmentation_prior - logq
        truth = torch.nn.functional.one_hot(label, 2).permute(0, 3, 1, 2).to(probability.dtype)
        task = balanced_risk(probability, truth)
        loss = (task - .01 * bound / pixels).mean()
        return loss, {'task': float(task.detach().mean()), 'bound_per_pixel': float((bound / pixels).detach().mean())}

    @torch.no_grad()
    def predict(self, image, seed, draws=4):
        generator = torch.Generator(device=image.device).manual_seed(seed)
        return torch.stack([self.draw(image, generator)[3] for _ in range(draws)]).mean(0)
