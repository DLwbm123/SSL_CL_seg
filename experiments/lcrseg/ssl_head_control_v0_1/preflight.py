import argparse,json,os
from pathlib import Path
from experiments.lcrseg.single_teacher_scd_v0_1.data import metadata,MANIFEST_SHA,SPLIT_SHA
from experiments.lcrseg.di_dmpa_gate1.binding import safe_asset,sha256
from .core import *
from .freeze import verify,DOC

def preflight(base,data):
    source=verify();base=Path(base)
    if not str(base.resolve()).startswith('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/'):raise PermissionError('NAS required')
    stat=os.statvfs(base.parent)
    if stat.f_bavail*stat.f_frsize<10*1024**3:raise RuntimeError('10 GiB reserve required')
    base.mkdir();rows=metadata(data);selected=[];counts={}
    for domain in DOMAINS:
        counts[domain]=[]
        for role,n in zip(('train_labeled','train_unlabeled','val'),COUNTS[domain]):
            rr=sorted([r for r in rows if r['site_or_vendor']==domain and r['primary_20pct_split']==role],key=lambda r:r['case_id'])
            if len(rr)!=n or len({r['patient_id'] for r in rr})!=n:raise RuntimeError('count/patient identity mismatch')
            counts[domain].append(len(rr))
            for r in rr:
                item={k:r[k] for k in ('case_id','patient_id','site_or_vendor','primary_20pct_split','image_h5_relpath','image_sha256')}
                if role!='train_unlabeled':item.update({k:r[k] for k in ('label_h5_relpath','label_sha256')})
                for name in (('image',) if role=='train_unlabeled' else ('image','label')):
                    path=safe_asset(data,item[name+'_h5_relpath'])
                    if not path.is_file():raise FileNotFoundError('authorized asset missing')
                selected.append(item)
    write_json(base/'private_allowlist.json',selected)
    receipt=dict(status='PASS',source=source,manifest_sha=MANIFEST_SHA,split_sha=SPLIT_SHA,protocol_sha=sha256(DOC/'PREREGISTRATION.md'),config_sha=sha256(DOC/'PREREGISTRATION.json'),private_allowlist_sha=sha256(base/'private_allowlist.json'),counts=counts,selected_rows=len(selected),path_metadata_only=True,test_file_reads=0,U_GT_metadata_constructed=False,random_initialization_only=True)
    receipt.update(reference_audit=old_reference_audit())
    write_json(base/'INPUT_LINEAGE.json',receipt);print(json.dumps(receipt))
def old_reference_audit():
    import csv
    old=Path('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/ssl_foundation_v0_1_20260908_01')
    table=list(csv.DictReader(Path('experiments/lcrseg/docs/ssl_foundation_v0_1/SINGLE_DOMAIN_METRICS.csv').open()))
    evidence=[]
    for d in DOMAINS:
        for a in base.ARMS:
            root=old/'seed11'/d/a;rec=json.loads((root/'receipt.json').read_text());dep=json.loads((root/'deployment.json').read_text());metric=list(csv.DictReader((root/'eval100/metrics.csv').open()))
            assert rec['source']=='d9542bd6f7b4acfe351bdaa8c8107bb875d40717' and rec['status']=='TRAINING_COMPLETE'
            assert dep['status']=='PASS' and rec['student_hash']==dep['student_hash']
            public=next(r for r in table if r['arm']==a and r['domain']==d)
            private=next(r for r in metric if r['model']=='student')
            for k in ('macro_fg_dice','rim_dice','cup_dice'):assert float(public[k])==float(private[k])
            evidence.append(dict(arm=a,domain=d,source=rec['source'],student_hash=rec['student_hash'],ema_hash=rec['ema_hash'],initialization=rec['initialization'],matched_public=True,P0_checkpoint_available=(root/'latest.pt').is_file() if a.startswith('MT_PAS') else None))
    return evidence

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--data',required=True);preflight(**vars(p.parse_args()))
