"""Independent original-clock timeout; only registered PID+birth identities."""
import json,os,time,signal
from datetime import datetime
from pathlib import Path
from ..main_style_ssl_v1.runtime import write

def alive(pid,birth):
    try:return Path('/proc/'+str(pid)+'/stat').read_text().split()[21]==str(birth)
    except FileNotFoundError:return False

def main(cfg):
    root=Path(cfg['root']);deadline=datetime.fromisoformat(cfg['clock_start']).timestamp()+23.75*3600
    while time.time()<deadline:
        status=root/'STATUS.json'
        if status.exists() and json.loads(status.read_text()).get('status') in ('COMPLETE','STOPPED'):return
        time.sleep(min(30,max(0,deadline-time.time())))
    targets=[]
    for path in (root/'ACTIVE.json',root/'CONTROLLER_STARTED.json'):
        if path.exists():
            item=json.loads(path.read_text())
            if alive(item['pid'],item['birth']):
                os.killpg(item['pid'],signal.SIGTERM);targets.append(item)
    write(root/'TIMEOUT.json',dict(status='TIME_LIMIT',targets=targets,time=time.time()))
    time.sleep(30)
    for item in targets:
        if alive(item['pid'],item['birth']):os.killpg(item['pid'],signal.SIGKILL)
