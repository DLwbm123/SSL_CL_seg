"""Exact frozen scientific files, clean Git and immutable historical namespace."""
import json
from pathlib import Path
import subprocess
from experiments.lcrseg.di_dmpa_gate1.binding import check_hash


def verify():
    root=Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip())
    document=root/"experiments/lcrseg/docs/single_teacher_scd_v0_1/SOURCE_FREEZE.json"
    frozen=json.loads(document.read_text())
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():raise RuntimeError("execution checkout is dirty")
    for name,digest in frozen["files"].items():check_hash(root/name,digest)
    paths=subprocess.check_output(["git","diff","--name-only",frozen["base_commit"],"HEAD"],text=True).splitlines()
    if any(not any(p.startswith(prefix+"/") for prefix in frozen["allowed_prefixes"]) for p in paths):
        raise RuntimeError("historical namespace modified")
    return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
