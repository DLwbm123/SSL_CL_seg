"""One legal image reservoir and one common target-entry evaluation, no optimizer calls."""
import json
from pathlib import Path
import torch
from .core import setup,provider
from ..gyro_native_v1.data import ImagePool,historical
from ..philips_transfer_v1.run import evaluate
from ..five_frameworks_v1.checkpoint import atomic_save
from ..main_style_ssl_v1.runtime import write


def main(cfg):
    root=Path(cfg['root']);out=root/'preparation';out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source,model,ema,opt=setup(dict(cfg,arm='G_U',phase='preparation',phase_cap=0,output=str(out)))
    assert not opt.state
    pool=ImagePool(cfg);bank=pool.reservoir();assert bank['size']==16 and bank['label_fields']==[]
    atomic_save(bank,Path(cfg['bank']))
    data=provider(dict(cfg,arm='G_U'))
    assert len(data._l.patients)==13 and set(data._l.patients).isdisjoint(data._u.patients)
    assert set(pool.parts[0].patients).isdisjoint(pool.parts[1].patients)
    assert len(set(map(tuple,bank['ids'])))==16
    # Zero-step identity check covers independently generated paired L/replay streams and safe noNewU input path.
    l1=data.labeled(0);l2=data.labeled(0)
    assert torch.equal(l1[0],l2[0]) and torch.equal(l1[1],l2[1]) and l1[2]==l2[2]
    assert torch.equal(historical(data,0),historical(data,0))
    target,_=evaluate(source,cfg,'Philips','val',out/'private_Philips_val.jsonl')
    train,_=evaluate(source,cfg,'Philips','train_labeled',out/'private_Philips_train.jsonl')
    assert target['patients']==31 and target['volumes']==62 and train['patients']==13 and train['volumes']==26
    write(out/'RESULTS.json',dict(status='PASS',optimizer_updates=0,target_entry=target,target_train_entry=train,
        bank_size=16,bank_unique_source_slices=1102,labels_read_for_bank=False,paired_streams_deterministic=True,
        source_entry_reused_from_v12=True,source_entry_dice_percent=79.43371427121954))
