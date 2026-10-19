"""Create one private, fixed-identity enc1 memory from legal training accessors."""
from pathlib import Path
import torch
from torch.nn import functional as F
from .core import cached_features,contrasts,views,GRID
from .state import graph_sites
from ..space_lora_v1.core import source_model
from ..main_style_ssl_v1.data import Accessor
from ..main_style_ssl_v1.runtime import write
from ..five_frameworks_v1.checkpoint import atomic_save


@torch.no_grad()
def main(cfg):
    output=Path(cfg['bank'])
    with output.with_suffix('.started').open('x') as marker: marker.write('one create-only preparation\n')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model=source_model(cfg)
    images=[]; positions=[]; labels=[]; identities=[]; pairs=[]
    for domain,count in [('Siemens',8),('Philips',4)]:
        access=Accessor(cfg['data'],'mnms',domain,'train_labeled')
        expected=10 if domain=='Siemens' else 13
        if len(access.patients)!=expected: raise ValueError('training population changed')
        for patient in access.patients[:count]:
            index=access.by_patient[patient][0]
            item=access.read(index)
            # Label-assisted memory selection uses train L only, never U/test.
            z=int((item['label']>0).flatten(1).sum(1).argmax())
            image=item['image'][z:z+1,None].cuda(); truth=item['label'][z]
            grid=[]
            for c in range(4):
                where=(truth==c).nonzero()
                if len(where)<2: raise ValueError('selected legal L slice lacks a class; no adaptive fallback')
                grid.extend([(int(p[0]),int(p[1]),c) for p in where[torch.linspace(0,len(where)-1,2).long()]])
            slot=len(images); images.append(image)
            positions.extend([[slot,y,x] for y,x,c in grid]); labels.extend([c for y,x,c in grid])
            identities.append(dict(domain=domain,role='train_labeled',patient=patient,
                                   case=access.rows[index]['case_id'],slice=z,view='canonical'))
    access=Accessor(cfg['data'],'mnms','Philips','train_unlabeled')
    for i,patient in enumerate(access.patients[:2]):
        index=access.by_patient[patient][0]; z=access.depths[index]//2
        image=access.read(index,z)['image'][:,None].cuda()
        strong,mask=views(image,900000+i)
        where=mask[0].nonzero()
        grid=where[torch.linspace(0,len(where)-1,8).long()]
        starts=[]
        for name,current in [('weak',image),('strong',strong)]:
            slot=len(images); images.append(current)
            starts.append(len(positions))
            positions.extend([[slot,int(y),int(x)] for y,x in grid]); labels.extend([-1]*8)
            identities.append(dict(domain='Philips',role='train_unlabeled',patient=patient,
                                   case=access.rows[index]['case_id'],slice=z,view=name,
                                   augmentation_seed_step=900000+i))
        pairs.extend([[starts[0]+j,starts[1]+j] for j in range(8)])
    points=torch.tensor(positions,device='cuda',dtype=torch.long)
    assert min(positions[i][1] for i in range(len(positions)))>=0
    assert all(0<=y<GRID[0] and 0<=x<GRID[1] for _,y,x in positions)
    features=[]; semantic=[]; response=[]; keys=[]; max_error=0.
    for i,image in enumerate(images):
        assert tuple(image.shape[-2:])==GRID
        e=model.parent.native.enc1(image)
        h=cached_features(model.parent,e)
        z=model.parent.native_readout(h,image.shape[-2:])
        max_error=max(max_error,float((z-model(image)).abs().max()))
        where=(points[:,0]==i).nonzero().flatten(); y,x=points[where,1],points[where,2]
        response.append(contrasts(z)[0,:,y,x].T)
        semantic.append(F.interpolate(h,size=GRID,mode='bilinear',align_corners=True)[0,:,y,x].T)
        keys.append(F.interpolate(e,size=GRID,mode='bilinear',align_corners=True)[0,:,y,x].T)
        features.append(e.cpu())
    if max_error!=0: raise ValueError('cached native continuation is not exact')
    key=F.normalize(torch.cat(keys).double(),dim=1)
    kernel=key@key.T+.1*torch.eye(128,device='cuda',dtype=torch.double)
    covariance=torch.kron(kernel.contiguous(),torch.eye(3,device='cuda',dtype=torch.double))
    precision=torch.cholesky_inverse(torch.linalg.cholesky(covariance))
    response=torch.cat(response).double()
    relation,edges=graph_sites(torch.cat(semantic),torch.tensor(pairs,device='cuda'),True)
    state=dict(features=torch.cat(features),points=points.cpu(),labels=torch.tensor(labels),
        pairs=torch.tensor(pairs),base_precision=precision.cpu(),
        base_natural=(precision@response.flatten()).cpu(),source_response=response.cpu(),
        initial_relation=relation.cpu(),identities=identities,optimizer_updates=0,
        fixed_queries=128,source_L_queries=64,target_L_queries=32,target_U_queries=32,
        cached_continuation_max_error=max_error,semantic_edges=len(edges),hidden_gt_access=False)
    atomic_save(state,output)
    size=output.stat().st_size
    if size>256*1024**2: raise MemoryError('persistent bank exceeds frozen 256MiB budget')
    write(output.with_suffix('.json'),dict(status='PASS',optimizer_updates=0,
          fixed_queries=128,cached_images=len(images),bank_bytes=size,
          source_patients=8,target_L_patients=4,target_U_patients=2,
          L_class_query_counts=[labels.count(c) for c in range(4)],
          U_label_fields_read=False,cached_continuation_max_error=max_error,
          state_dimension=384,preparation_complete=True))
