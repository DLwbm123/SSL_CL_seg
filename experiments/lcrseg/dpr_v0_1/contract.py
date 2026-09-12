"""Create-only fixed matrix, original source identities and current-domain capability."""
import subprocess
from pathlib import Path
from experiments.lcrseg.di_dmpa_gate1.binding import check_hash,sha256
from experiments.lcrseg.lctx_weight_memory_v0_1.contract import read,nas

BASE='8610343c08de61f639844aa069ceecaec0de2b49'
PARENT_SOURCE='f174cf2a68212cf04ad19b18d8203a6455c94aee'
DOC=Path('experiments/lcrseg/docs/dpr_v0_1')
PREFIXES=('experiments/lcrseg/dpr_v0_1/','experiments/lcrseg/tests/dpr_v0_1/',str(DOC)+'/')

def protocol():return read(DOC/'PROTOCOL.json')

def verify():
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('dirty source')
    for path,checksum in read(DOC/'SOURCE_FREEZE.json')['files'].items():check_hash(path,checksum)
    if any(not p.startswith(PREFIXES) for p in subprocess.check_output(['git','diff','--name-only',BASE,'HEAD'],text=True).splitlines()):raise RuntimeError('inherited namespace changed')
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()

def task_by_id(tid):
    rows=[t for t in protocol()['tasks'] if t['task_id']==tid]
    if len(rows)!=1:raise PermissionError('unregistered task')
    return rows[0]

def admit(base,tid,source):
    b=nas(base);r=read(b/'reservation.json')
    if r['source']!=source or r['tasks']!=protocol()['tasks'] or r['formal_updates']!=47700:raise PermissionError('reservation identity')
    for n,h in r['input_hashes'].items():check_hash(b/n,h)
    return task_by_id(tid)

# Frozen libraries use git -C for a read-only upstream check. Keep paths out of child argv.
def neutral_subprocess_paths():
    original=subprocess.Popen
    def popen(args,*a,**kw):
        if isinstance(args,(list,tuple)) and len(args)>3 and args[0]=='git' and args[1]=='-C':
            if 'cwd' in kw:raise ValueError('ambiguous git working directory')
            kw['cwd']=args[2];args=[args[0],*args[3:]]
        return original(args,*a,**kw)
    subprocess.Popen=popen
