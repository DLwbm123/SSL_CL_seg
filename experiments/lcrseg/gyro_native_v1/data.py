"""Legal training-image streams; the U accessor never exposes hidden labels."""
import json
from pathlib import Path
import numpy as np
import torch
from ..main_style_ssl_v1.data import Accessor,CurrentDomain
from ..main_style_ssl_v1.runtime import permit
from ..single_teacher_scd_v0_1.data import geometry
from ..five_frameworks_v1.recipes import generator


def current(cfg,domain,steps):
    node=dict(dataset='mnms',domain=domain,seed=163,order=1,stage=1,steps=steps)
    data=CurrentDomain(cfg['data'],'mnms',domain,163,1,1,{'kind':'GYRO'},'cuda',permit(node,'formal'))
    if domain=='Philips':
        assert json.loads(Path(cfg['split']).read_text())=={'train':data._l.patients,'holdout':[]}
        assert len(data._l.patients)==13
    return data


class ImagePool:
    def __init__(self,cfg):
        self.parts=[Accessor(cfg['data'],'mnms','Siemens',r) for r in ['train_labeled','train_unlabeled']]
        self.items=[(a,i,z) for a,part in enumerate(self.parts) for i,d in enumerate(part.depths) for z in range(d)]
        assert len(self.items)==1102 and sum(self.parts[0].depths)==230
        self.cache={}
        self.orders=[np.random.default_rng(163+epoch).permutation(1102).tolist() for epoch in range(2)]

    def image(self,index):
        a,i,z=self.items[index]; access=self.parts[a]
        # This read bypasses Accessor.read for L images too: no historical labels loaded.
        key=(a,i)
        if key not in self.cache:
            import h5py
            from ..di_dmpa_gate1.binding import safe_asset,check_hash
            r=access.rows[i]; path=safe_asset(access.data,r['image_h5_relpath'])
            check_hash(path,r['image_sha256'])
            with h5py.File(path,'r') as f: self.cache[key]=torch.from_numpy(f['image'][...].astype(np.float32))
        return self.cache[key][z:z+1]

    def epoch_batch(self,step):
        epoch,within=divmod(step,69); order=self.orders[epoch]
        ids=[order[(within*16+i)%1102] for i in range(16)]
        return torch.stack([self.image(i) for i in ids]).cuda()

    def reservoir(self):
        rng=np.random.default_rng(78163); chosen=list(range(16))
        for i in range(16,1102):
            j=int(rng.integers(i+1))
            if j<16: chosen[j]=i
        return dict(images=torch.stack([self.image(i) for i in chosen]),ids=[self.items[i] for i in chosen],
                    domain='Siemens',size=16,source_unique_slices=1102,label_fields=[])


def provider(cfg):
    source=cfg['arm']=='source_prepare'
    data=current(cfg,'Siemens' if source else 'Philips',138 if source else 2000)
    if source:
        data.pool=ImagePool(cfg)
    elif cfg['arm']!='dense_fixmatch':
        bank=torch.load(cfg['bank'],map_location='cpu',weights_only=False)
        assert bank['size']==16 and bank['label_fields']==[]
        data.bank=bank['images']
    return data


def historical(data,step):
    rng=np.random.default_rng(88163+step)
    ids=rng.choice(16,8,replace=False).tolist()
    values=[]
    for pos,i in enumerate(ids):
        x=data.bank[i]; valid=torch.ones(x.shape[-2:],dtype=torch.bool)
        x,_,_=geometry(x,None,valid,generator(163,1,1,step,'gyro_replay_'+str(pos)))
        values.append(x)
    return torch.stack(values).cuda()
