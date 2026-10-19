"""Register GYRO with the existing transfer engine, including its one training loop."""
from pathlib import Path
from unittest.mock import patch
import torch
from . import core
from .data import provider,ImagePool
from ..philips_transfer_v1 import run as engine
from ..five_frameworks_v1.checkpoint import atomic_save
from ..main_style_ssl_v1.runtime import write


def source_finalize(cfg,out,model,ema,committed,physical):
    # Source final student == EMA by construction; no intermediate checkpoint selection.
    old,_=engine.evaluate(ema,cfg,'Siemens','val',out/'private_Siemens_val.jsonl')
    train,_=engine.evaluate(ema,cfg,'Siemens','train_labeled',out/'private_Siemens_train.jsonl')
    delta=100*(old['primary_dice']-.8036740663770262)
    passed=delta>=-1.
    write(out/'RESULTS.json',dict(status='COMPLETE',step=138,physical_updates=138,
        Siemens=old,source_train=train,source_gate_passed=passed,delta_from_original_pp=delta,
        primary='fixed final source student',independent_test_access=False))
    if passed:
        pool=ImagePool(cfg)
        atomic_save(pool.reservoir(),Path(cfg['bank']))
        write(Path(cfg['bank']).with_suffix('.json'),dict(status='READY',size=16,
            unique_source_images=1102,labels_read_for_bank=False,optimizer_updates=0))


def main(cfg):
    from .qualification import preflight
    source=cfg['arm']=='source_prepare'
    changes=dict(ARMS=('source_prepare',)+core.ARMS,STEPS=138 if source else 2000,GPU='3',
        setup=core.setup,provider=provider,preflight=preflight,losses=core.losses,
        rate=lambda step:core.rate(step,source),update_ema=core.update_ema)
    if source: changes['finalize']=source_finalize
    with patch.multiple(engine,**changes):
        engine.main(cfg)
