import json,subprocess
from pathlib import Path
from experiments.lcrseg.di_dmpa_gate1.binding import check_hash
BASE='77af53864fa116030c832d69e7a404c79de2456e'
PREFIXES=('experiments/lcrseg/ams_seq_transfer_v0_1/','experiments/lcrseg/tests/ams_seq_transfer_v0_1/','experiments/lcrseg/docs/ams_seq_transfer_v0_1/')
DOC=Path(PREFIXES[2])
def read(p):return json.loads(Path(p).read_text())
def protocol():return read(DOC/'PROTOCOL.json')
def verify():
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('dirty source')
    for p,h in read(DOC/'SOURCE_FREEZE.json')['files'].items():check_hash(p,h)
    changed=subprocess.check_output(['git','diff','--name-only',BASE,'HEAD'],text=True).splitlines()
    if any(not p.startswith(PREFIXES) for p in changed):raise RuntimeError('inherited source modified')
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def task_by_id(task_id):
    matches=[r for r in protocol()['tasks'] if r['task_id']==task_id]
    if len(matches)!=1:raise PermissionError('task not frozen')
    return matches[0]
def admit(base,task_id,source):
    b=Path(base).resolve()
    if not str(b).startswith('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/'):raise PermissionError('NAS required')
    reservation=read(b/'reservation.json');task=task_by_id(task_id)
    if reservation['source']!=source or reservation['tasks']!=protocol()['tasks']:raise PermissionError('reservation differs')
    return task
