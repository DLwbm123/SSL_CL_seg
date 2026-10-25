"""Match the deployment mean risk to foreground-balanced predictive Fisher."""

import torch
from torch.nn import functional as F

from ..hierarchical_bayes_h1.model import HierarchicalSegmenter, PIXELS, gaussian_kl


class MatchedRiskSegmenter(HierarchicalSegmenter):
    def loss(self, image, label):
        stats = self.image_stats(image)
        q_mean, q_logsd, prior = self.appearance_distribution(stats, label)
        raw_a = self.sample(q_mean, q_logsd)
        q_field, shape_kl = self._field(image, raw_a, label)
        appearance_kl = gaussian_kl(q_mean, q_logsd, *prior)
        local_nll = F.binary_cross_entropy_with_logits(q_field[:, 0], label.float())
        local = local_nll + ((shape_kl + appearance_kl) / PIXELS).mean()

        logits = self.deploy_logits(image)[:, 0]
        target = label.float()
        positive, negative = target.sum(), (1 - target).sum()
        balanced_ce = (.5 * (F.softplus(-logits) * target).sum() / positive.clamp_min(1) +
                       .5 * (F.softplus(logits) * (1 - target)).sum() / negative.clamp_min(1))
        ordinary_ce = F.binary_cross_entropy_with_logits(logits, target)
        balanced_ce = torch.where((positive > 0) & (negative > 0), balanced_ce, ordinary_ce)
        foreground = logits.sigmoid()
        dice = (2 * (foreground * target).sum((-2, -1)) + 1) / \
            (foreground.sum((-2, -1)) + target.sum((-2, -1)) + 1)
        total = .5 * local + .5 * balanced_ce + (1 - dice).mean()
        return total, {"local": local.detach(), "deploy_ce": balanced_ce.detach(),
                       "deploy_dice_loss": (1 - dice).mean().detach()}
