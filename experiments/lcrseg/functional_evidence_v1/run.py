"""Use the existing unique transfer training loop, including its physical ledger."""
from unittest.mock import patch
from . import core
from ..philips_transfer_v1 import run as engine
from ..semantic_condition_v1.run import provider


def main(cfg):
    from .qualification import preflight
    # Provider's permit step count must be the current frozen count as well.
    from ..semantic_condition_v1 import core as provider_core
    with patch.object(provider_core,'STEPS',core.STEPS),patch.multiple(engine,
        ARMS=core.ARMS,STEPS=core.STEPS,GPU='3',setup=core.setup,rate=core.rate,
        losses=core.losses,update_ema=core.update_ema,provider=provider,preflight=preflight):
        engine.main(cfg)
