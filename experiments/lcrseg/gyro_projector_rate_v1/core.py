"""Reduce only the training projector rate; native remains fully trainable."""
from ..gyro_weight_v2 import core as weight
ARMS=weight.ARMS
STEPS=weight.STEPS
provider=weight.provider
configure_lr=weight.configure_lr
update_ema=weight.update_ema
raw_core=weight.raw_core
replay_indices=weight.replay_indices
losses=weight.losses


def setup(cfg):
    assert cfg['projector_lr']==.003
    source,model,ema,opt=weight.setup(cfg)
    assert not opt.state
    for group in opt.param_groups:
        rate=.003 if group['name'].startswith('projector') else .001
        group['lr']=group['fixed_lr']=rate
    assert all(p.requires_grad for n,p in model.parent.named_parameters() if n not in {'native.decoder.conv_logit.sigma.weight','native.decoder.conv_logit.grad_update'})
    return source,model,ema,opt
