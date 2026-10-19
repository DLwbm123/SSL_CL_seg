from pathlib import Path
import numpy as np
import torch
from ..gyro_native_v1.data import ImagePool
from ..gyro_autonomy_12h.space_guard import SpaceGuard
from ..five_frameworks_v1.checkpoint import atomic_save
from ..main_style_ssl_v1.runtime import write


def main(cfg):
    guard=SpaceGuard(cfg['storage_authorization']);guard.check(350_000_000)
    pool=ImagePool(cfg);size=256;rng=np.random.default_rng(78163);chosen=list(range(size))
    for i in range(size,1102):
        j=int(rng.integers(i+1))
        if j<size:chosen[j]=i
    assert len(set(chosen))==256
    images=torch.stack([pool.image(i) for i in chosen]);ids=[pool.items[i] for i in chosen]
    assert torch.isfinite(images).all() and images.min()>=-1.001 and images.max()<=1.001
    guard.check(2*images.numel()*images.element_size())
    atomic_save(dict(images=images,ids=ids,domain='Siemens',size=256,source_unique_slices=1102,label_fields=[]),Path(cfg['bank']))
    patients={pool.parts[a].rows[i]['patient_id'] for a,i,z in ids}
    write(Path(cfg['root'])/'BANK_PREPARED.json',dict(status='PASS',bank_size=256,patients_covered=len(patients),source_training_patients=48,
        source_L_images=sum(a==0 for a,i,z in ids),source_U_images=sum(a==1 for a,i,z in ids),labels_read=False,optimizer_updates=0,
        image_bytes=images.numel()*images.element_size(),unique_source_slices=1102,entry_evaluation_reused=True,storage=guard.check()))
