import torch
from pathlib import Path
from ..main_style_ssl_v1.data import Accessor
from ..five_frameworks_v1.recipes import generator
from ..di_dmpa_gate1.binding import safe_asset
class Current:
    def __init__(self,cfg):
        import h5py
        self.cfg=cfg;self.arm=cfg['arm'];self.l_reads=self.u_reads=0
        from .risk import load_cache
        self.cache=load_cache(cfg)
        self.access=Accessor(cfg['data'],'mnms','Siemens' if cfg['stage']=='source' else 'Philips','train_labeled')
        assert len(self.access.patients)==(10 if cfg['stage']=='source' else 13)
        allowed={str(safe_asset(self.access.data,r[k])) for r in self.access.rows for k in ('image_h5_relpath','label_h5_relpath')};original=h5py.File
        def checked(path,*a,**kw):
            if str(Path(path).resolve()) not in allowed:raise PermissionError('HDF5 outside current training L')
            return original(path,*a,**kw)
        h5py.File=checked
    def labeled(self,step,diagnostic=False):
        self.l_reads+=1;g=generator(20271802 if diagnostic else 20271801,1,0 if self.cfg['stage']=='source' else 1,step,'ACQUISITION_ANATOMY_L')
        xx=[];yy=[];ids=[]
        for pos in torch.randperm(len(self.access.patients),generator=g)[:2].tolist():
            patient=self.access.patients[pos];ii=self.access.by_patient[patient];i=ii[int(torch.randint(len(ii),(),generator=g))];z=int(torch.randint(self.access.depths[i],(),generator=g))
            a=self.access.read(i,z);xx.append(a['image']);yy.append(a['label']);ids.append(str(i)+':'+str(z))
        return torch.stack(xx).cuda(),torch.stack(yy).cuda(),tuple(ids)

