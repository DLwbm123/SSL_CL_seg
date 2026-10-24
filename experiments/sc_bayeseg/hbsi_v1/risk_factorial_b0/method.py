"""Two missing risk-by-family cells; historical arms remain read-only."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from sc_bayeseg.model import normal_log_prob, sar, student_log_prob
from ..fullfield_bayeseg_a1.model import FullFieldBayeSeg
from ..paired_semantic_v19.method import balanced_risk
from ..unet_transfer_v45.run import unet


def standard_risk(logits, label):
    probability = logits.softmax(1)[:, 1]
    foreground = (label == 1).to(probability.dtype)
    count = foreground.sum((1, 2))
    dice = (2 * (probability * foreground).sum((1, 2)) + 1e-6) / (
        probability.sum((1, 2)) + count + 1e-6)
    return F.cross_entropy(logits, label, reduction='none').mean((1, 2)) + (1 - dice) * (count > 0)


class HStd(FullFieldBayeSeg):
    def loss(self, image, label, generator):
        shape, appearance, z, probability, logq = self.draw(image, generator)
        observed = student_log_prob(image - shape - appearance, .1)
        appearance_prior = normal_log_prob(appearance, torch.zeros_like(appearance),
                                           torch.full_like(appearance, math.log(.5)))
        shape_prior = student_log_prob(sar(shape - self.shape_mean(probability), .9), .2) + self.sar_logdet
        segmentation_prior = student_log_prob(sar(z, .9), 1.) + self.sar_logdet
        bound = observed + appearance_prior + shape_prior + segmentation_prior - logq
        logits = torch.cat((torch.zeros_like(z), z), 1)
        task = standard_risk(logits, label)
        loss = (task - .01 * bound / (self.size * self.size)).mean()
        return loss, {'task': float(task.detach().mean()),
                      'bound_per_pixel': float((bound / (self.size * self.size)).detach().mean())}


class UBal(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = unet()

    def loss(self, image, label, generator):
        probability = self.model(image).logits.softmax(1)
        truth = F.one_hot(label, 2).permute(0, 3, 1, 2).to(probability.dtype)
        task = balanced_risk(probability, truth)
        return task.mean(), {'task': float(task.detach().mean()), 'bound_per_pixel': 0.}

    @torch.no_grad()
    def predict(self, image, seed, draws=4):
        return self.model(image).logits.softmax(1)
