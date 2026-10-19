import json
from pathlib import Path
from unittest.mock import patch
import torch
from . import core
from .data import provider
from ..philips_transfer_v1 import run as engine
from ..siemens_lonly_v1.run import aggregate,compare
from ..main_style_ssl_v1.runtime import write


def finalize(cfg,out,model,ema,committed,physical):
    if cfg['arm']=='calibration':
        initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(v.cpu(),initial['model'][k]) for k,v in model.state_dict().items() if k.startswith('parent.') or k=='frames')
        assert any(not torch.equal(v.cpu(),initial['model'][k]) for k,v in model.state_dict().items() if k.startswith('projector.'))
        result=dict(status='COMPLETE',arm=cfg['arm'],step=committed,physical_updates=physical,native_exact=True,projector_changed=True)
    else:
        val,rows=engine.evaluate(model,cfg,'Siemens','val',out/'private_student_val.jsonl')
        train,_=engine.evaluate(model,cfg,'Siemens','train_labeled',out/'private_student_train.jsonl')
        ev,_=engine.evaluate(ema,cfg,'Siemens','val',out/'private_ema_val.jsonl')
        base=[json.loads(r) for r in Path(cfg['source_val_rows']).read_text().splitlines()]
        diff=100*(val['primary_dice']-aggregate(base)['primary_dice'])
        result=dict(status='COMPLETE',arm=cfg['arm'],step=committed,physical_updates=physical,
            primary='fixed final student',Siemens=val,source_train=train,EMA_diagnostic=ev,
            delta_from_original_pp=diff,source_gate_passed=diff>=-1,paired_original=compare(base,rows))
        if cfg['arm']=='GYRO_low':
            ref=Path(cfg.get('CE_low_reference',str(Path(cfg['root'])/'jobs/CE_low')))
            low=json.loads((ref/'RESULTS.json').read_text())
            d=100*(val['primary_dice']-low['Siemens']['primary_dice'])
            result.update(delta_from_CE_low_pp=d,relative_gate_passed=d>=-.5,
                paired_CE_low=compare([json.loads(r) for r in (ref/'private_student_val.jsonl').read_text().splitlines()],rows),all_gates_passed=(d>=-.5 and diff>=-1))
    write(out/'RESULTS.json',result)


def main(cfg):
    from .qualification import preflight
    with patch.multiple(engine,ARMS=core.ARMS,STEPS=69,GPU='3',setup=core.setup,provider=provider,
        preflight=preflight,losses=core.losses,rate=lambda step:.03 if cfg['arm'] in ['calibration','CE_high'] else .001,
        set_learning_rate=core.configure_lr,update_ema=core.update_ema,finalize=finalize):
        engine.main(cfg)
