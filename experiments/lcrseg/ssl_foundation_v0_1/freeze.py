import json,subprocess
from pathlib import Path
from experiments.lcrseg.di_dmpa_gate1.binding import check_hash
BASE='6eb48fd845f6192698c4d39893b2524447afb3df'
PREFIXES=('experiments/lcrseg/ssl_foundation_v0_1/','experiments/lcrseg/tests/ssl_foundation_v0_1/','experiments/lcrseg/docs/ssl_foundation_v0_1/')
DOC=Path(PREFIXES[2])
def verify():
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('dirty execution source')
    frozen=json.loads((DOC/'SOURCE_FREEZE.json').read_text())
    for path,sha in frozen['files'].items():check_hash(path,sha)
    changed=subprocess.check_output(['git','diff','--name-only',BASE,'HEAD'],text=True).splitlines()
    if any(not p.startswith(PREFIXES) for p in changed):raise RuntimeError('historical source changed')
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
