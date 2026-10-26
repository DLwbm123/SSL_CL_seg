"""Current-image acquisition orbit for the H5 anatomy-preserving risk."""

import torch

from ..hierarchical_bayes_h1.model import HierarchicalSegmenter


class OrbitSegmenter(HierarchicalSegmenter):
    """Average the original full risk with one label-preserving orbit draw.

    The six-dimensional transform is pointwise invertible and does not move
    pixels. It acts only on images from the current training site; it never
    synthesizes or stores an old-site image.
    """

    def loss(self, image, label):
        original, original_parts = super().loss(image, label)
        with torch.no_grad():
            raw = 2 * torch.rand((len(image), 6), device=image.device) - 1
            transformed = self.forward_appearance(image, raw).detach()
        orbit, orbit_parts = super().loss(transformed, label)
        return .5 * (original + orbit), {
            "original": original.detach(), "orbit": orbit.detach(),
            "deploy_ce": .5 * (original_parts["deploy_ce"] + orbit_parts["deploy_ce"]),
            "deploy_dice_loss": .5 * (original_parts["deploy_dice_loss"] +
                                    orbit_parts["deploy_dice_loss"]),
        }
