import json,subprocess
from pathlib import Path
from experiments.lcrseg.di_dmpa_gate1.binding import check_hash
BASE='671d0e57c8a2b56b5162933be4973ea0f538f58c'
PARENT_SOURCE='f174cf2a68212cf04ad19b18d8203a6455c94aee'
DOC=Path('experiments/lcrseg/docs/lctx_weight_memory_v0_1')
PREFIXES=('experiments/lcrseg/lctx_weight_memory_v0_1/',str(DOC)+'/')
def read(p):return json.loads(Path(p).read_text())
def protocol():return read(DOC/'PROTOCOL.json')
def verify():
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('dirty source')
    for p,h in read(DOC/'SOURCE_FREEZE.json')['files'].items():check_hash(p,h)
    if any(not p.startswith(PREFIXES) for p in subprocess.check_output(['git','diff','--name-only',BASE,'HEAD'],text=True).splitlines()):raise RuntimeError('inherited namespace modified')
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def nas(base):
    b=Path(base)
    if not str(b.resolve()).startswith('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/'):raise PermissionError('NAS required')
    return b
def task_by_id(tid):
    rows=[t for t in protocol()['tasks'] if t['task_id']==tid]
    if len(rows)!=1:raise PermissionError('task is not registered')
    return rows[0]
def admit(base,tid,source):
    b=nas(base);r=read(b/'reservation.json')
    if r['source']!=source or r['tasks']!=protocol()['tasks']:raise PermissionError('reservation identity')
    for name,h in r['input_hashes'].items():check_hash(b/name,h)
    return task_by_id(tid)
