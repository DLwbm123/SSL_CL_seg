"""Image-only U access and frozen image-group fold/schedule construction."""
import json,hashlib
from pathlib import Path
import torch
from qprompt.data import digest
from r1_12h.augmentation import geometry,strong
from r1_6_readout_v1.runtime import seed_value


class UImages:
    def __init__(self,root,domain):
        self.root=Path(root).resolve(strict=True);m=json.loads((self.root/'U_IMAGE_ONLY.private.json').read_text())
        if m.get('authorization')!='R3_U_IMAGE_ONLY' or not m.get('image_only'):raise PermissionError('U image authorization absent')
        self.rows=[r for r in m['files'] if r['domain']==domain];self.checked=set()
        for row in self.rows:
            if set(row)!={'relative','sha256','bytes','domain','case_id'}:raise PermissionError('unexpected U fields / hidden GT')
        assert len(self.rows)=={'RIM_ONE_r3':63,'Drishti_GS':41}[domain]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        import h5py
        row=self.rows[i];p=self.root/row['relative']
        if p.is_symlink() or not p.resolve(strict=True).is_relative_to(self.root):raise PermissionError('U path escape')
        if i not in self.checked:
            assert digest(p)==row['sha256'];self.checked.add(i)
        with h5py.File(p,'r') as f:
            if set(f)!={'image'}:raise PermissionError('U mixed/label payload prohibited')
            x=f['image'][...]
        assert x.shape==(3,384,384) and str(x.dtype)=='uint8'
        return torch.from_numpy(x.copy()).float()/255


def make_schedule(seed,domain,n_l,n_u):
    # Patient grouping provenance is unverified: explicitly use image groups, never claim patient independence.
    order=sorted(range(n_l),key=lambda i:seed_value(0,'bandit-fold',domain,i))
    folds=[order[i::5] for i in range(5)];assert all(folds)
    steps=[]
    for t in range(1200):
        d=t//20;online=folds[d%5];audit=folds[(d+1)%5];fit=[i for k,f in enumerate(folds) if k not in (d%5,(d+1)%5) for i in f]
        assert not(set(fit)&set(online) or set(fit)&set(audit) or set(online)&set(audit))
        g=torch.Generator().manual_seed(seed_value(seed,'bandit',domain,t,'L'))
        indices=[fit[j] for j in torch.randperm(len(fit),generator=g)[:2].tolist()]
        u=int(torch.randint(n_u,(),generator=torch.Generator().manual_seed(seed_value(seed,'bandit',domain,t,'U'))))
        item=dict(t=t,decision=d,indices=indices,online=online,audit=audit,fit=fit,u=u,geometry_seed=seed_value(seed,'bandit',domain,t,'geometry'),photometric_seed=seed_value(seed,'bandit',domain,t,'photometric'),u_geometry_seed=seed_value(seed,'bandit',domain,t,'Ugeometry'),u_photo_seed=seed_value(seed,'bandit',domain,t,'Uphoto'),actual_seed=seed_value(seed,'bandit',domain,t,'actual'),probe_seed=seed_value(seed,'bandit',domain,t,'probe'))
        item['extra']=[dict(indices=[online[(2*j+k)%len(online)] for k in range(2)],geometry_seed=seed_value(seed,'bandit',domain,t,j,'extra_geometry'),photometric_seed=seed_value(seed,'bandit',domain,t,j,'extra_photo')) for j in range(5)]
        steps.append(item)
    return dict(seed=seed,domain=domain,grouping='image; patient identity reliability unverified',folds=folds,steps=steps)


def u_views(image,item,device):
    x,_,v=geometry(image,None,torch.ones(image.shape[-2:],dtype=torch.bool),torch.Generator().manual_seed(item['u_geometry_seed']))
    weak=x[None].to(device);valid=v[None].to(device)
    augmented=strong(weak,torch.Generator(device=device).manual_seed(item['u_photo_seed']))
    return weak,augmented,valid
