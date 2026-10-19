"""Use the v1.1 runner and qualification with explicit v1.2 contracts."""
from pathlib import Path
from unittest.mock import patch
import torch
from ..gyro_source_repair_v11 import core,run as old_run,qualification as old_q
from ..philips_transfer_v1 import run as engine
from ..main_style_ssl_v1.runtime import write
from ..five_frameworks_v1.checkpoint import atomic_save

old_setup=core.setup
old_preflight=old_q.preflight
old_main=old_run.main


def setup(cfg):
    assert cfg['arm']=='GYRO_low' and cfg['gradient_ratio_hard_stop'] is None
    assert cfg['budget_cap']==101 and cfg['checkpoint_each_step'] and cfg['save_exception_state']
    source,model,ema,opt=old_setup(cfg)
    entry=torch.load(Path(cfg['CE_low_reference'])/'initial.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v.cpu(),entry['model'][k]) for k,v in model.state_dict().items())
    assert not opt.state and not entry['optimizer']['state']
    assert cfg['data']==cfg['matched_CE_settings']['data']
    assert cfg['source_checkpoint']==cfg['matched_CE_settings']['source_checkpoint']
    assert cfg['calibrated']==cfg['matched_CE_settings']['calibrated']
    return source,model,ema,opt


def preflight(cfg,out,source,model,ema,opt,data):
    old_preflight(cfg,out,source,model,ema,opt,data)
    # Complete state at step zero is the restart point even if the first step fails.
    state=engine.snapshot(model,ema,opt,data,0)
    atomic_save(state,out/'latest.pt')
    write(out/'MATCHED_ENTRY.json',dict(status='PASS',native_projector_BN_frame_exact=True,
        both_momenta_empty=True,L_indices='69..137',CE_control_reused=True,
        CE_provider='same CurrentDomain.labeled(step+69) with per-step generator',
        ratio_guard_enabled=False,source_domain_only=True))


def main(cfg):
    with patch.object(core,'setup',setup),patch.object(old_q,'preflight',preflight):
        old_main(cfg)


def qualify(cfg):
    # Only v1.1 qualify's imported runner is redirected; its 32-call recipe is unchanged.
    with patch.object(core,'setup',setup),patch.object(old_q,'preflight',preflight),patch.object(old_run,'main',main):
        old_q.qualify(cfg)
