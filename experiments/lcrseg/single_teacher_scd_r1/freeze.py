import json,subprocess
from pathlib import Path
from experiments.lcrseg.di_dmpa_gate1.binding import check_hash
BASE="bd29494153aa0175b5b52d360f87a44255293d9f"
PREFIXES=("experiments/lcrseg/single_teacher_scd_r1/","experiments/lcrseg/tests/single_teacher_scd_r1/","experiments/lcrseg/docs/single_teacher_scd_r1/")
def verify():
 root=Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip())
 if subprocess.check_output(["git","status","--porcelain"],text=True).strip():raise RuntimeError("dirty R1 execution checkout")
 frozen=json.loads((root/PREFIXES[2]/"SOURCE_FREEZE.json").read_text())
 for path,digest in frozen["files"].items():check_hash(root/path,digest)
 changed=subprocess.check_output(["git","diff","--name-only",BASE,"HEAD"],text=True).splitlines()
 if any(not any(p.startswith(x) for x in PREFIXES) for p in changed):raise RuntimeError("historical file changed")
 return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
