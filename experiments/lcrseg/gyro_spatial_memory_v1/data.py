"""Unaugmented role-bound source or target batches; no historical target accessor."""
from pathlib import Path
import torch,h5py
from ..main_style_ssl_v1.data import Accessor
from ..five_frameworks_v1.recipes import generator

class Domain:
    def __init__(self,cfg,domain):
        assert domain in ('Siemens','Philips')
        self.domain=domain;self.arm=cfg.get('arm',domain);self.l_reads=self.u_reads=0
        self._l=Accessor(cfg['data'],'mnms',domain,'train_labeled')
        self._u=Accessor(cfg['data'],'mnms',domain,'train_unlabeled')
        assert len(self._l.patients)==(10 if domain=='Siemens' else 13)
        assert len(self._l.rows)==(20 if domain=='Siemens' else 26)
        if domain=='Philips':
            # HDF5 opens in this worker are restricted to target L/U assets only.
            allowed={str((Path(cfg['data'])/r[k]).resolve()) for a in (self._l,self._u) for r in a.rows for k in ('image_h5_relpath','label_h5_relpath') if k in r}
            real=h5py.File
            def target_only(name,*args,**kwargs):
                if str(Path(name).resolve()) not in allowed:raise PermissionError('target worker attempted non-Philips training HDF5 access')
                return real(name,*args,**kwargs)
            h5py.File=target_only
    def batch(self,role,step,n=2):
        assert n%2==0
        a=self._l if role=='L' else self._u;xs=[];ys=[];ids=[]
        for block in range(n//2):
            g=generator(163,1,0 if self.domain=='Siemens' else 1,step,role+str(block))
            for j in torch.randperm(len(a.patients),generator=g)[:2].tolist():
                patient=a.patients[j];indices=a.by_patient[patient];i=indices[int(torch.randint(len(indices),(),generator=g))]
                z=int(torch.randint(a.depths[i],(),generator=g));item=a.read(i,z)
                xs.append(item['image']);ids.append(patient)
                if role=='L':ys.append(item['label'])
            if role=='L':self.l_reads+=1
            else:self.u_reads+=1
        return torch.stack(xs).cuda(),torch.stack(ys).cuda() if ys else None,tuple(ids)
    def labeled(self,step):return self.batch('L',step)
    def unlabeled(self,step):return self.batch('U',step)
