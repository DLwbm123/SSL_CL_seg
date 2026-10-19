"""Allocated disk growth guard for the user-authorized 12-hour campaign."""
import json
import subprocess
from pathlib import Path


def allocated(path):
    return int(subprocess.check_output(['du','-s','-B1',str(path)],text=True).split()[0])


class SpaceGuard:
    def __init__(self,authorization):
        self.path=Path(authorization)
        self.rules=json.loads(self.path.read_text())

    def check(self,planned_extra_bytes=0):
        r=self.rules
        # Reserve small local/source/operational metadata, kept outside model-output directories.
        used=100_000_000+max(0,allocated(r['current_run'])-r['current_run_allocated_baseline_bytes'])+allocated(r['new_run_parent'])
        peak=used+planned_extra_bytes+r['peak_reserve_bytes']
        if used>=r['operational_stop_new_bytes'] or peak>r['hard_cap_new_bytes']:
            raise OSError(f'authorized storage growth guard: allocated={used}, projected_peak={peak}')
        return dict(accounted_new_bytes=used,projected_peak_bytes=peak,hard_cap=r['hard_cap_new_bytes'])
