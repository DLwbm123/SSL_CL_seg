"""One source-L pass, bounded deterministic per-class patch prototypes."""
from pathlib import Path
import torch
from torch.nn import functional as F
from ..space_lora_v1.core import source_model
from ..main_style_ssl_v1.data import Accessor
from ..main_style_ssl_v1.runtime import write
from ..five_frameworks_v1.checkpoint import atomic_save


def centers(x):
    # CPU spherical kmeans++, one seed, no retries/reinitialization.
    x=F.normalize(x.float(),dim=1)
    g=torch.Generator().manual_seed(163)
    chosen=[int(torch.randint(len(x),(),generator=g))]
    for _ in range(2):
        distance=(1-x@x[chosen].T).clamp_min(0).min(1).values
        if distance.sum()<=0: raise ValueError('insufficient distinct class patches')
        chosen.append(int(torch.multinomial(distance,1,generator=g)))
    c=x[chosen].clone()
    occupancy=None
    for _ in range(20):
        labels=(x@c.T).argmax(1)
        occupancy=[int((labels==i).sum()) for i in range(3)]
        if min(occupancy)==0: raise ValueError('empty spherical cluster; no retry authorized')
        c=F.normalize(torch.stack([x[labels==i].mean(0) for i in range(3)]),dim=1)
    if not torch.isfinite(c).all() or (c.norm(dim=1)<.99).any():
        raise ValueError('invalid class centers')
    return c,occupancy


@torch.inference_mode()
def main(cfg):
    output=Path(cfg['statistics'])
    marker=output.with_suffix('.started.json')
    if output.exists() or marker.exists(): raise FileExistsError('source statistics already attempted')
    write(marker,dict(status='STARTED',allowed_source_slices=230,optimizer_updates=0))
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model=source_model(cfg)
    access=Accessor(cfg['data'],'mnms','Siemens','train_labeled')
    if len(access.patients)!=10 or sum(access.depths)!=230:
        raise ValueError('source labeled population mismatch')
    candidates=[[] for _ in range(4)]; counts=[0]*4; slices=0
    for i in range(len(access.rows)):
        item=access.read(i)
        for z in range(len(item['image'])):
            h=model.parent.features(item['image'][z:z+1,None].cuda())
            labels=model.parent.output_to_feature(item['label'][z:z+1].cuda(),h.shape[-2:],categorical=True).flatten()
            patches=F.unfold(h,3,padding=1)[0].T
            if patches.shape[1]!=144: raise ValueError('unexpected native feature width')
            for c in range(4):
                where=(labels==c).nonzero().flatten(); counts[c]+=len(where)
                if len(where):
                    positions=torch.linspace(0,len(where)-1,min(32,len(where)),device=where.device).long()
                    candidates[c].append(patches[where[positions]].cpu())
            slices+=1
        print(dict(source_volumes=i+1,source_slices=slices,optimizer_updates=0),flush=True)
    if slices!=230: raise ValueError('source coverage incomplete')
    proto=[]; summary=[]
    for c,items in enumerate(candidates):
        if not items: raise ValueError('missing source class')
        x=torch.cat(items)
        if len(x)>8192: x=x[torch.linspace(0,len(x)-1,8192).long()]
        p,n=centers(x); proto.append(p)
        summary.append(dict(class_index=c,candidates=len(x),pixels=counts[c],cluster_counts=n))
    state=dict(prototypes=torch.stack(proto),source_patients=10,source_slices=slices,
               optimizer_updates=0,unlabeled_access=False,summary=summary,
               procedure='per-slice/class evenly spaced <=32; <=8192 evenly spaced cap; spherical kmeans++ seed163 20 iterations')
    atomic_save(state,output)
    write(output.with_suffix('.json'),{k:v for k,v in state.items() if k!='prototypes'})
