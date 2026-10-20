from unittest.mock import patch
from . import core
from ..philips_transfer_v1 import run as engine
from ..gyro_fixed_projector_v1 import run as fixed
from ..gyro_strength_v1.run import guarded_saver
from ..gyro_transfer_v1.run import finalize

def preflight(cfg,*args):
    with patch.object(fixed,'core',core):fixed.preflight(cfg,*args)

def main(cfg):
    with patch.multiple(engine,ARMS=core.ARMS,STEPS=2000,GPU='3',setup=core.setup,provider=core.provider,
        preflight=preflight,losses=core.losses,rate=lambda step:1e-4,set_learning_rate=core.configure_lr,
        update_ema=core.update_ema,finalize=finalize,atomic_save=guarded_saver(cfg)):
        engine.main(cfg)

def qualify(cfg):
    from .qualification import qualify as check
    check(cfg)
