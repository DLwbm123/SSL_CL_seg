import argparse,csv,json,os
from pathlib import Path
from experiments.lcrseg.di_dmpa_gate1.binding import check_hash,sha256
from experiments.lcrseg.single_teacher_scd_v0_1.data import metadata,COUNTS,DOMAINS
from experiments.lcrseg.single_teacher_scd_v0_1.engine import write_json
from .freeze import verify
OLD=Path('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/single_teacher_scd_v0_1_20260908_01/run_01')
R1=Path('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/single_teacher_r1_20260908_01')
PUBLIC=Path('experiments/lcrseg/docs/single_teacher_scd_r1')
def run(base,data):
    source=verify();base=Path(base)
    stat=os.statvfs(base.parent)
    if stat.f_bavail*stat.f_frsize<5*1024**3:raise RuntimeError('less than 5 GiB output/cache reserve')
    base.mkdir()
    allrows=metadata(data)
    assert [[sum(x['site_or_vendor']==d and x['primary_20pct_split']==role for x in allrows) for role in ('train_labeled','train_unlabeled')] for d in DOMAINS]==[list(x) for x in COUNTS]
    public=list(csv.DictReader((PUBLIC/'STAGE_DOMAIN_MATRIX.csv').open()));reused=[]
    for arm in ('common','S','L05','D','U0','L10','E_R1'):
        root=OLD/arm if arm in ('common','S','D') else R1/('E_run' if arm=='E_R1' else 'ablations')/arm
        for stage in ((0,) if arm=='common' else (1,2)):
            sr=root/f'stage{stage}'
            rec=json.loads((sr/('receipt.json' if arm in ('common','S','D') else 'r1_receipt.json')).read_text())
            val=json.loads((sr/'val.json').read_text())
            assert rec['status']=='COMPLETE' and val['student_hash']==rec['student_hash']
            assert rec['source']==('057ce07fa7bd02f8319e0c2bdd9d4adbceff23d8' if arm in ('common','S','D') else 'f94be07a8a3db1c062035a9bc2e542af9d4b156b')
            for row in val['rows']:
                ref=next(x for x in public if x['arm']==('S' if arm=='common' else arm) and int(x['stage'])==stage and int(x['domain_index'])==row['domain_index'])
                for k in ('macro_fg_dice','rim_dice','cup_dice','background_dice'):assert float(ref[k])==row[k]
            if arm=='common' or (arm in ('S','L05') and stage==2):check_hash(sr/'student_latest.pt',rec['checkpoint_sha256'])
            if stage==2:
                dep=json.loads((sr/'deployment.json').read_text());assert dep['status']=='PASS' and dep['models']==1 and dep['student_hash']==rec['student_hash']
            reused.append(dict(arm=arm,stage=stage,source=rec['source'],student_hash=rec['student_hash'],checkpoint_sha256=rec['checkpoint_sha256'],val_sha256=sha256(sr/'val.json'),private_public_exact=True))
    out=dict(status='PASS',source=source,data_split_seed=0,optimization_seeds=[0,1,2],common_seed0_checkpoint=str(OLD/'common/stage0/student_latest.pt'),reused=reused,old_formal_updates=57208,baseline_csv_sha256=sha256(PUBLIC/'STAGE_DOMAIN_MATRIX.csv'),formal_new_GT_accesses=0)
    write_json(base/'INPUT_LINEAGE.json',out);print(json.dumps(out))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--data',required=True);run(**vars(p.parse_args()))
