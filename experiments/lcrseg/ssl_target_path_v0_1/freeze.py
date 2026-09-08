import json,subprocess
from pathlib import Path
from experiments.lcrseg.di_dmpa_gate1.binding import check_hash
BASE='60843e85c110f31863299639822b080cf4c4ba9b'
PREFIXES=('experiments/lcrseg/ssl_target_path_v0_1/','experiments/lcrseg/tests/ssl_target_path_v0_1/','experiments/lcrseg/docs/ssl_target_path_v0_1/')
DOC=Path(PREFIXES[2])
def verify():
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('dirty execution source')
    frozen=json.loads((DOC/'SOURCE_FREEZE.json').read_text())
    for path,sha in frozen['files'].items():check_hash(path,sha)
    changed=subprocess.check_output(['git','diff','--name-only',BASE,'HEAD'],text=True).splitlines()
    if any(not p.startswith(PREFIXES) for p in changed):raise RuntimeError('historical source changed')
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
