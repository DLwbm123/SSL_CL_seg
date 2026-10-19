"""Register the bounded pilot with the existing, unique training loop."""
import json
from pathlib import Path
from unittest.mock import patch
from . import core
from ..philips_transfer_v1 import run as engine
from ..main_style_ssl_v1.data import Accessor, CurrentDomain
from ..main_style_ssl_v1.runtime import permit


def provider(cfg):
    frozen=json.loads(Path(cfg['split']).read_text())
    actual=Accessor(cfg['data'],'mnms','Philips','train_labeled').patients
    if frozen!={'train':actual,'holdout':[]} or len(actual)!=13:
        raise ValueError('frozen Philips split mismatch')
    node=dict(dataset='mnms',domain='Philips',seed=163,order=1,stage=1,steps=core.STEPS)
    return CurrentDomain(cfg['data'],'mnms','Philips',163,1,1,
                         {'kind':'source_FixMatch_EMA'},'cuda',permit(node,'formal'))


def main(cfg):
    from .qualification import preflight
    with patch.multiple(engine,ARMS=core.ARMS,STEPS=core.STEPS,GPU='3',
                        setup=core.setup,rate=core.rate,losses=core.losses,
                        update_ema=core.update_ema,provider=provider,preflight=preflight):
        engine.main(cfg)
