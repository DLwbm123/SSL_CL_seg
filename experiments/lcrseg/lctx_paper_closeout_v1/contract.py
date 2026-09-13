"""Fixed 30-target matrix and NAS-only admission; no performance gates."""
import hashlib,json,subprocess
from pathlib import Path
from experiments.lcrseg.lctx_weight_memory_v0_1.contract import read,nas
from experiments.lcrseg.dpr_sign_replication_v0_1.contract import neutral_subprocess_paths

DOC=Path('experiments/lcrseg/docs/lctx_paper_closeout_v1')
PARENT_SOURCE='87bb70ee05609e77935b1bab5b9a44c9532dc491'
PARENT_RUN='dpr_sign_replication_v0_1_20260913_01'
PREFIX='LCTX_PAPER_CLOSEOUT_V1'

def protocol():return read(DOC/'PROTOCOL.json')

def verify():
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():raise RuntimeError('modified tracked runtime source')
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()

def task_by_id(tid):
    rows=[t for t in protocol()['tasks'] if t['task_id']==tid]
    if len(rows)!=1:raise PermissionError('unregistered task')
    return rows[0]

def admit(base,tid,source):
    r=read(nas(base)/'reservation.json');p=protocol()
    if r['source']!=source or r['tasks']!=p['tasks'] or r['formal_updates']!=79500:raise PermissionError('reservation mismatch')
    return task_by_id(tid)

def nested(rows,domain):
    rows=sorted((r for r in rows if r['site_or_vendor']==domain and r['primary_20pct_split']=='train_labeled'),key=lambda r:r['case_id'])
    if len({r['patient_id'] for r in rows})!=len(rows) or len(rows)<2:raise ValueError('unique patients required')
    ranked=sorted(rows,key=lambda r:(hashlib.sha256((PREFIX+'|'+domain+'|'+r['patient_id']).encode()).hexdigest(),r['case_id']))
    chosen=sorted(ranked[:max(2,len(rows)//2)],key=lambda r:r['case_id'])
    return rows,chosen

def prepare(base,data,parent):
    from . import core as c
    from experiments.lcrseg.single_teacher_scd_v0_1.data import metadata
    b=nas(base);b.mkdir(exist_ok=True);old=Path(parent)
    if any((b/n).exists() for n in ('private_inputs.json','reservation.json','tasks')):raise FileExistsError('run already prepared')
    if old.name!=PARENT_RUN or read(old/'GPU67_CONTROLLER_STATUS.json')['status']!='COMPLETE':raise PermissionError('completed SIGN source run required')
    rows=metadata(data);private={};public={}
    for d in c.DOMAINS:
        full,low=nested(rows,d)
        if len(full)!=c.COUNTS[d][0]:raise ValueError('actual L count mismatch')
        private[d]={budget:[dict(case_id=r['case_id'],patient_id=r['patient_id']) for r in group] for budget,group in [('standard',full),('low',low)]}
        canonical=json.dumps(private[d]['low'],sort_keys=True,separators=(',',':')).encode()
        public[d]=dict(standard_L=len(full),low_L=len(low),manifest_sha256=hashlib.sha256(canonical).hexdigest(),hash_encoding='UTF-8 compact sorted-key JSON, selected rows in case_id order')
    binding={'sources':{},'baselines':{}}
    for order in ('O1','O2'):
        for seed in (71,72,73):
            sid=f'{order}_s{seed}_SRC_CE';tid=f'{order}_s{seed}_F_CONV'
            for name,ident in [('sources',sid),('baselines',tid)]:
                root=old/'tasks'/ident;r=read(root/'receipt.json');e=read(root/'evaluation/receipt.json')
                if r['source']!=PARENT_SOURCE or r['status']!='TRAINING_COMPLETE' or r['student_hash']!=e['student_hash'] or e['status']!='COMPLETE':raise PermissionError('reuse receipt mismatch')
                if not (root/'deploy_student.pt').is_file():raise FileNotFoundError('missing reused deployment')
                binding[name][ident]=dict(source=r['source'],student_hash=r['student_hash'],seed=seed,task=r['task'],updates=r['updates'])
    c.write_json(b/'private_inputs.json',dict(data=str(Path(data).resolve()),parent=str(old.resolve())))
    c.write_json(b/'private_subsets.json',private)
    c.write_json(b/'SUBSET_MANIFEST.json',dict(rule=PREFIX+'|domain|patient_id',domains=public,selection_uses_scores=False,patient_ids_public=False))
    c.write_json(b/'REUSE_BINDING.json',binding)
    from collections import Counter
    roles=Counter((r['site_or_vendor'],r['primary_20pct_split']) for r in rows)
    c.write_json(b/'DATA_SCOPE.json',dict(manifest_role_counts=[dict(domain=d,role=r,patients=n) for (d,r),n in sorted(roles.items())],independent_patient_cohort='NOT_ESTABLISHED',scope_check='metadata and existing source/evaluation receipts only; no test or hidden GT access',third_training_domain='NOT_INCLUDED',published_method_comparisons='NOT_INCLUDED'))
    return b
