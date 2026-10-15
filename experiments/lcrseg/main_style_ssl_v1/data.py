"""Role-bound slice access: U stores no label fields; evaluation only seen val."""
import copy,csv,json
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
import torch
from ..di_dmpa_gate1.binding import safe_asset,check_hash
from ..single_teacher_scd_v0_1.data import MANIFEST_SHA,geometry
from ..five_frameworks_v1.recipes import generator
from ..five_frameworks_v1.gate import digest
from .protocol import COUNTS,SHAPES,MAPS


def metadata(data,dataset):
    path=Path(data)/'manifests/training/lcrseg_v1_seed0.csv';check_hash(path,MANIFEST_SHA)
    with path.open() as f:rows=[r for r in csv.DictReader(f) if r['dataset']==dataset and r['evaluation_eligible']=='true']
    roles={};cases=set()
    for r in rows:
        if r['split_seed']!='0' or r['case_id'] in cases:raise ValueError('invalid frozen manifest')
        cases.add(r['case_id']);key=(r['primary_20pct_split'],r['site_or_vendor'])
        if roles.setdefault(r['patient_id'],key)!=key:raise ValueError('patient crosses role/domain')
        if r['primary_20pct_split']=='train_unlabeled' and (r['label_h5_relpath'] or r['label_sha256']):raise PermissionError('U GT metadata')
        if dataset=='mnms' and r['cohort']!='canonical320':raise ValueError('auxiliary cohort admitted')
    return rows


class Accessor:
    def __init__(self,data,dataset,domain,role):
        if role not in ('train_labeled','train_unlabeled','val'):raise PermissionError('no test accessor')
        self.data=Path(data);self.dataset=dataset;self.role=role;self.rows=[];self.checked=set()
        for r in metadata(data,dataset):
            if r['site_or_vendor']!=domain or r['primary_20pct_split']!=role:continue
            item={k:r[k] for k in ('case_id','patient_id','phase','image_h5_relpath','image_sha256')}
            if role!='train_unlabeled':item.update({k:r[k] for k in ('label_h5_relpath','label_sha256')})
            self.rows.append(item)
        self.rows.sort(key=lambda r:r['case_id'])
        if not self.rows:raise ValueError('empty dataset')
        self.by_patient=defaultdict(list);self.depths=[]
        for i,r in enumerate(self.rows):
            self.by_patient[r['patient_id']].append(i)
            with h5py.File(safe_asset(self.data,r['image_h5_relpath']),'r') as f:
                shape=f['image'].shape
            expected=SHAPES[dataset]
            if dataset=='fundus' and shape!=expected:raise ValueError('RGB shape mismatch')
            if dataset!='fundus' and shape[1:]!=expected[1:]:raise ValueError('volume shape mismatch')
            self.depths.append(1 if dataset=='fundus' else shape[0])
        self.patients=sorted(self.by_patient)
    def read(self,index,slice_index=None):
        r=self.rows[index];out={}
        for key in (('image',) if self.role=='train_unlabeled' else ('image','label')):
            path=safe_asset(self.data,r[key+'_h5_relpath'])
            if (index,key) not in self.checked:
                check_hash(path,r[key+'_sha256']);self.checked.add((index,key))
            with h5py.File(path,'r') as f:
                array=f[key][...] if slice_index is None or self.dataset=='fundus' else f[key][slice_index]
                if key=='image':out['spacing']=json.loads(f.attrs['processed_spacing'])
            if key=='image':
                if self.dataset=='fundus':
                    if array.dtype!=np.uint8:raise ValueError('fundus dtype')
                    array=array.astype(np.float32)/255
                else:
                    if array.dtype!=np.float16:raise ValueError('MRI dtype')
                    array=array.astype(np.float32)
                    if slice_index is not None:array=array[None]
                if not np.isfinite(array).all() or array.min()<-1.001 or array.max()>1.001:raise ValueError('normalized image range')
                out[key]=torch.from_numpy(array)
            else:
                if not np.isin(array,range(len(MAPS[self.dataset]))).all():raise ValueError('label map mismatch')
                out[key]=torch.from_numpy(array.astype(np.int64))
        return out


class CurrentDomain:
    def __init__(self,data,dataset,domain,seed,order,stage,stage_source,device,permit):
        permit.validate();self.dataset,self.domain=dataset,domain
        self.seed,self.order,self.stage=seed,order,stage;self.device=device
        self.stage_source=copy.deepcopy(stage_source);self.l_reads=self.u_reads=0
        self._identity=dict(dataset=dataset,domain=domain,seed=seed,order=order,stage=stage,manifest=MANIFEST_SHA)
        if digest(self._identity) not in permit.bindings['authorized_manifest_digests']:raise PermissionError('outside execution permit')
        self._l=Accessor(data,dataset,domain,'train_labeled');self._u=Accessor(data,dataset,domain,'train_unlabeled')
        actual=(sum(self._l.depths),sum(self._u.depths))
        if actual!=COUNTS[dataset][domain]:raise ValueError(f'frozen slice budget mismatch: {actual}')
        if min(len(self._l.patients),len(self._u.patients))<2:raise ValueError('two distinct patients required')
    def semantic_metadata(self):
        return {**self._identity,'stage_source':self.stage_source,'shape':SHAPES[self.dataset],
                'sampling':'uniform patient -> phase -> image-only uniform slice; distinct paired patients',
                'batch_size':2,'normalization':'uint8/255' if self.dataset=='fundus' else 'stored [-1,1] cast FP32'}
    def _batch(self,access,step,stream):
        g=generator(self.seed,self.order,self.stage,step,stream)
        selected=torch.randperm(len(access.patients),generator=g)[:2].tolist();values=[];ids=[]
        for pos in selected:
            patient=access.patients[pos];ii=access.by_patient[patient]
            index=ii[int(torch.randint(len(ii),(),generator=g))]
            z=int(torch.randint(access.depths[index],(),generator=g));item=access.read(index,z)
            image=item['image'];valid=torch.ones(image.shape[-2:],dtype=torch.bool)
            x,y,v=geometry(image,item.get('label'),valid,g)
            values.append((x,y,v));ids.append(patient)
        return torch.stack([v[0] for v in values]).to(self.device),torch.stack([v[1] if access.role=='train_labeled' else v[2] for v in values]).to(self.device),tuple(ids)
    def labeled(self,step,stream='labeled'):
        self.l_reads+=1;return self._batch(self._l,step,stream)
    def unlabeled(self,step):
        self.u_reads+=1;return self._batch(self._u,step,'unlabeled')
