"""Image-conditioned local segmentation posterior and its task-only control."""

import torch

from lcrseg.models.unet import UNet2D
from ..risk_factorial_b0.method import HStd, standard_risk


FIRST_WEIGHT = 'segmentation_posterior.enc1.block.0.weight'


class HIS(HStd):
    def __init__(self):
        super().__init__()
        self.segmentation_posterior = UNet2D(2, 2, base_channels=16)
        self.segmentation_posterior.projection_head = torch.nn.Identity()

    def draw(self, image, generator):
        if image.shape[-2:] != (self.size, self.size):
            raise ValueError('wrong working image size')
        shape, logq_shape = self.gaussian(self.shape_posterior(image), generator)
        appearance, logq_appearance = self.gaussian(self.appearance_posterior(image), generator)
        z, logq_z = self.gaussian(
            self.segmentation_posterior(torch.cat((shape, image), 1)).logits, generator)
        probability = torch.cat((torch.zeros_like(z), z), 1).softmax(1)
        return shape, appearance, z, probability, logq_shape + logq_appearance + logq_z


class HISNoVar(HIS):
    def loss(self, image, label, generator):
        _, _, z, _, _ = self.draw(image, generator)
        logits = torch.cat((torch.zeros_like(z), z), 1)
        task = standard_risk(logits, label)
        return task.mean(), {'task': float(task.detach().mean()), 'bound_per_pixel': 0.}


def copied_initial(model, hstd_state):
    """Copy every H-Std tensor; the new image input begins at exactly zero."""
    state = model.state_dict()
    assert set(state) == set(hstd_state)
    for name, value in state.items():
        original = hstd_state[name]
        if name == FIRST_WEIGHT:
            assert value.shape[1] == 2 and original.shape[1] == 1
            replacement = torch.zeros_like(value)
            replacement[:, :1].copy_(original)
            state[name] = replacement
        else:
            assert value.shape == original.shape, name
            state[name] = original.clone()
    model.load_state_dict(state)
    return model
