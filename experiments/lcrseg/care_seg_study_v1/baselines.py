"""Matched dense EMA control, using the same supervised and optimizer contracts."""
import copy
import torch
from torch.nn import functional as F
from lcrseg.methods.base import freeze_model
from lcrseg.methods.sequential_ssl import SequentialSSLMethod


class EMAControl(SequentialSSLMethod):
    def __init__(self, model, config):
        super().__init__(model, config=config)
        self.ema_model = freeze_model(copy.deepcopy(model))
    @torch.no_grad()
    def _weak_pseudo(self, batch):
        self.ema_model.eval()
        confidence, labels = self.ema_model(batch.weak_image).logits.softmax(1).max(1)
        return labels, confidence.ge(self.config['tau_cls'])[:,None]
    def _ssl_loss(self, logits, labels, valid, strong_valid_mask):
        # All-pixel denominator matches CARE; abstention must not amplify a tiny support.
        return (F.cross_entropy(logits, labels, reduction='none')*valid[:,0]*strong_valid_mask[:,0]).mean()
    @torch.no_grad()
    def after_optimizer_step(self):
        for target, source in zip(self.ema_model.parameters(), self.model.parameters(), strict=True):
            target.lerp_(source, 1-self.config['ema'])
        for target, source in zip(self.ema_model.buffers(), self.model.buffers(), strict=True):
            target.copy_(source)
    def method_state_dict(self):
        state = super().method_state_dict()
        state['method_statistics']['ema'] = copy.deepcopy(self.ema_model.state_dict())
        return state
    def load_method_state_dict(self,state):
        super().load_method_state_dict(state)
        self.ema_model.load_state_dict(state['method_statistics']['ema'])
