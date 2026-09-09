import argparse,os
from pathlib import Path
from . import core as c
from .contract import verify,DOC,protocol
from experiments.lcrseg.di_dmpa_gate1.binding import safe_asset,sha256

def preflight(base,data):
    source=verify();base=Path(base)
    if not str(base.resolve()).startswith('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/'):raise PermissionError('NAS required')
    st=os.statvfs(base.parent)
    if st.f_bavail*st.f_frsize<10*1024**3:raise RuntimeError('10GiB reserve required')
    base.mkdir();allrows=c.metadata(data);selected=[];counts={}
    for domain in c.DOMAINS:
        counts[domain]={}
        for role,n in zip(('train_labeled','train_unlabeled','val'),c.COUNTS[domain]):
            rows=sorted((r for r in allrows if r['site_or_vendor']==domain and r['primary_20pct_split']==role),key=lambda r:r['case_id'])
            if len(rows)!=n or len({r['patient_id'] for r in rows})!=n:raise ValueError('patient count contract')
            counts[domain][role]=n
            for r in rows:
                item={k:r[k] for k in ('case_id','patient_id','site_or_vendor','primary_20pct_split','image_h5_relpath','image_sha256')}
                if role!='train_unlabeled':item.update({k:r[k] for k in ('label_h5_relpath','label_sha256')})
                for kind in ('image',) if role=='train_unlabeled' else ('image','label'):
                    if not safe_asset(data,item[kind+'_h5_relpath']).is_file():raise FileNotFoundError('authorized asset missing')
                selected.append(item)
    c.write_json(base/'private_allowlist.json',selected)
    result=dict(status='PASS',source=source,counts=counts,rows=len(selected),manifest_sha=c.MANIFEST_SHA,split_sha=c.SPLIT_SHA,protocol_sha=sha256(DOC/'PROTOCOL.md'),config_sha=sha256(DOC/'PROTOCOL.json'),allowlist_sha=sha256(base/'private_allowlist.json'),GT_reads=0,image_reads=0,old_checkpoint_dependency=False,stage_boundary='new source student only; reset EMA/Adam/RNG',same_patient_pairing_verified=True)
    c.write_json(base/'SOURCE_AND_INPUT_LINEAGE.json',result);print(result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--data',required=True);preflight(**vars(p.parse_args()))
