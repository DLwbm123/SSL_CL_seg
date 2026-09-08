import json,subprocess
from pathlib import Path
from experiments.lcrseg.di_dmpa_gate1.binding import check_hash
BASE='f2c7bd366cf9bdf80ae52c35798ece6be2e3a60d'
PREFIXES=('experiments/lcrseg/l05_ssl_replication/','experiments/lcrseg/tests/l05_ssl_replication/','experiments/lcrseg/docs/l05_ssl_replication/')
def verify():
    root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('dirty execution checkout')
    frozen=json.loads((root/PREFIXES[2]/'SOURCE_FREEZE.json').read_text())
    for p,h in frozen['files'].items():check_hash(root/p,h)
    changed=subprocess.check_output(['git','diff','--name-only',BASE,'HEAD'],text=True).splitlines()
    if any(not p.startswith(PREFIXES) for p in changed):raise RuntimeError('historical source changed')
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
