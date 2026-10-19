from unittest.mock import patch
from . import core
from ..philips_transfer_v1 import run as engine
from ..gyro_source_repair_v11 import qualification as old_q,run as old_run
from ..gyro_fixed_projector_v1 import run as fixed
from ..gyro_strength_v1.run import guarded_saver
from ..gyro_transfer_v1.run import finalize as target_finalize

def preflight(cfg,*args):
    if cfg['arm']=='calibration':
        with patch.object(old_q,'core',core),patch.object(old_q,'atomic_save',guarded_saver(cfg)):
            old_q.preflight(cfg,*args)
    else:
        with patch.object(fixed,'core',core):fixed.preflight(cfg,*args)

def main(cfg):
    cal=cfg['arm']=='calibration'
    with patch.multiple(engine,ARMS=core.ARMS,STEPS=1000 if cal else 2000,GPU='3',setup=core.setup,provider=core.provider,
        preflight=preflight,losses=core.losses,rate=lambda step:.03 if cal else .001,set_learning_rate=core.configure_lr,
        update_ema=core.update_ema,finalize=old_run.finalize if cal else target_finalize,atomic_save=guarded_saver(cfg)):
        engine.main(cfg)

def qualify(cfg):
    with patch.object(old_q,'core',core),patch.object(old_q,'provider',core.provider),patch.object(old_run,'main',main),patch.object(old_q,'atomic_save',guarded_saver(cfg)):
        old_q.qualify(cfg)
