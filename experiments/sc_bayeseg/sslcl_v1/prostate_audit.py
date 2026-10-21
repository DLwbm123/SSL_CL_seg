"""Read frozen split metadata and visible training labels; no test or U labels."""
import csv,json,os
from pathlib import Path
from collections import Counter
import h5py
import numpy as np
from .data import Schedule,style
from sc_bayeseg.window_v1.r_start import write

def run(s):
    data=Path(s['data']);out=Path(s['root'])
    reference={r['patient_id']:r for r in json.loads(Path(s['split']).read_text())['records']}
    with (data/'manifests/training/lcrseg_v1_seed0.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['dataset']=='prostate' and r['site_or_vendor'] in s['domains']]
    assert len({r['patient_id'] for r in rows})==len(rows)
    expected={k for k,r in reference.items() if r['site_or_vendor'] in s['domains']}
    assert {r['patient_id'] for r in rows}==expected
    report={};paths=set();spacings=[];shapes=[];label_values=set()
    for site in s['domains']:
        counts=Counter(r['primary_20pct_split'] for r in rows if r['site_or_vendor']==site)
        assert counts['train_labeled']==4 and counts['train_unlabeled']==14 and counts['val']==4
        report[site]=dict(counts)
    for r in rows:
        ref=reference[r['patient_id']];role=r['primary_20pct_split']
        assert (r['site_or_vendor'],role)==(ref['site_or_vendor'],ref['primary_20pct_split'])
        if role=='test':continue
        p=data/'h5/v1'/r['image_h5_relpath'];assert p not in paths;paths.add(p)
        with h5py.File(p,'r') as f:
            assert 'image' in f and 'label' not in f
            shape=f['image'].shape;spacing=json.loads(f.attrs['processed_spacing'])
            assert len(shape)==3 and shape[1:]==(256,256) and min(shape)>0
            assert len(spacing)==3 and np.isfinite(spacing).all() and min(spacing)>0
            if role.startswith('train_'):assert np.isfinite(f['image'][0]).all()
            shapes.append(shape);spacings.append(spacing)
        if role=='train_unlabeled':
            assert not r['label_h5_relpath'] and not r.get('label_sha256','')
        elif role=='train_labeled':
            with h5py.File(data/'h5/v1'/r['label_h5_relpath'],'r') as f:y=f['label'][...]
            assert y.shape==shape and np.isin(y,[0,1]).all() and (y==1).any()
            label_values.update(np.unique(y).tolist())
        elif role=='val':assert (data/'h5/v1'/r['label_h5_relpath']).is_file()
        else:raise AssertionError('unexpected role')
    styles={d:style(Schedule(s['data'],d,dataset='prostate')) for d in s['domains']}
    assert all(np.isfinite(v).all() and v[1]>0 for v in styles.values())
    s['styles']=styles;write(Path(os.environ['TASK_SETTINGS']),s)
    write(out/'DATA_AUDIT.json',dict(status='PASS',dataset='prostate',domains=report,split_matches=True,patient_roles_disjoint=True,unique_image_paths=True,visible_train_label_values=sorted(label_values),input_channels=1,num_classes=2,shape_min=np.min(shapes,axis=0).tolist(),shape_max=np.max(shapes,axis=0).tolist(),spacing_min=np.min(spacings,axis=0).tolist(),spacing_max=np.max(spacings,axis=0).tolist(),styles=styles,test_files_opened=0,hidden_U_labels_opened=0,val_labels_opened=0,identity_limit='Frozen patient IDs and unique paths checked; no content hashing or clinical identity linkage audit.'))
if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
