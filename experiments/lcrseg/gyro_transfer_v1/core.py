"""Original GYRO/MMCR four-cell comparison plus matched-entry FixMatch."""
import json
from pathlib import Path
import torch
from ..gyro_source_repair_v11 import core as source_core
from ..gyro_native_v1.core import losses as original_losses
from ..gyro_native_v1.data import current,historical
from ..gyro_source_repair_v11.core import gradients,configure_lr,update_ema
from ..main_style_ssl_v1.runtime import append

ARMS=('G_U','M_U','G_noNewU','M_noNewU','dense_fixmatch')
STEPS=2000


def setup(cfg):
    assert cfg['arm'] in ARMS and cfg['budget_cap']==10160
    assert cfg['gradient_ratio_hard_stop'] is None
    source,model,ema,opt=source_core.setup(cfg)
    # source_core loads the prepared fixed student through calibrated, without optimizer history.
    assert cfg['calibrated']==cfg['prepared'] and not opt.state
    del source
    source=model.teacher()
    return source,model,ema,opt


def provider(cfg):
    data=current(cfg,'Philips',2000)
    if cfg['arm']!='dense_fixmatch':
        bank=torch.load(cfg['bank'],map_location='cpu',weights_only=False)
        assert bank['size']==16 and bank['label_fields']==[] and bank['domain']=='Siemens'
        assert bank['images'].shape[0]==16 and torch.isfinite(bank['images']).all()
        data.bank=bank['images']
    return data


def losses(arm,model,data,step):
    l,u,stats=original_losses(arm,model,data,step)
    if arm!='dense_fixmatch':u=.01*u
    groups=gradients(model,l,u)
    ratio=groups['encoder_all']['weighted_G_norm']/max(groups['encoder_all']['CE_norm'],1e-12)
    stats.update(unsupervised=float(u.detach()),gradient_groups=groups,encoder_weighted_ratio=ratio,
        geometric_images=0 if arm=='dense_fixmatch' else 24,replay_images=0 if arm=='dense_fixmatch' else 8,
        new_U_images=0 if arm.endswith('_noNewU') else 14)
    append(Path(model.cfg['output'])/'private_loss_diagnostics.jsonl',dict(step=step,arm=arm,**stats))
    return l,u,stats
