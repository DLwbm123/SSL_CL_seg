"""Recovery-only admission, frozen training kernels and original scientific matrix."""
import subprocess
from pathlib import Path
from experiments.lcrseg.di_dmpa_gate1.binding import sha256,check_hash
from experiments.lcrseg.lctx_weight_memory_v0_1.contract import read,protocol,nas,task_by_id

BASE='d614de8aaeccc08a8448f0e6460e8a59ed9395c3'
OLD_SOURCE='50e094dd188c90c0117b62bb63bf961c096ea37d'
DOC=Path('experiments/lcrseg/docs/lctx_weight_memory_v0_1_1')
OLD_DOC=Path('experiments/lcrseg/docs/lctx_weight_memory_v0_1')
EDITED=tuple('experiments/lcrseg/lctx_weight_memory_v0_1/'+n+'.py' for n in ('engine','evaluate','results'))
PREFIXES=('experiments/lcrseg/lctx_weight_memory_v0_1_1/',str(DOC)+'/')

def verify():
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('dirty source')
    for p,h in read(OLD_DOC/'SOURCE_FREEZE.json')['files'].items():
        if p not in EDITED:check_hash(p,h)
    for p,h in read(DOC/'SOURCE_FREEZE.json')['files'].items():check_hash(p,h)
    changed=subprocess.check_output(['git','diff','--name-only',BASE,'HEAD'],text=True).splitlines()
    if any(p not in EDITED and not p.startswith(PREFIXES) for p in changed):raise RuntimeError('outside diagnostic recovery boundary')
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()

def admit(base,tid,source):
    b=nas(base);r=read(b/'reservation.json');m=read(b/'RECOVERY_MANIFEST.json')
    if r['source']!=source or r['new_updates']!=49900 or tid not in r['new_task_ids']:raise PermissionError('recovery reservation')
    for n,h in r['input_hashes'].items():check_hash(b/n,h)
    if tid!='O1_s62_LR_SRC_AB' and read(b/'RECOVERY_420_GATE.json')['status']!='PASS':raise PermissionError('420 engineering gate required')
    return task_by_id(tid)

def task_root(base,tid):
    b=Path(base);m=read(b/'RECOVERY_MANIFEST.json')
    return (Path(m['old_run']) if tid in m['reused_task_ids'] else b)/'tasks'/tid

def task_source(base,tid):
    return OLD_SOURCE if tid in read(Path(base)/'RECOVERY_MANIFEST.json')['reused_task_ids'] else verify()
