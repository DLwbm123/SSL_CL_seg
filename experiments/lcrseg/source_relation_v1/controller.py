"""Two matched finite arms; no restart, retry, or adaptive continuation."""
import json
import os
import subprocess
import time
from pathlib import Path
from .method import ARMS,STEPS
from ..main_style_ssl_v1.runtime import write


def main(cfg):
    root=Path(cfg['root'])
    workers=[]
    for arm in ARMS:
        q=Path(cfg['qualification_root'])/arm
        for name in ('PREFLIGHT.json','BASELINE_IDENTITY.json'):
            record=json.loads((q/name).read_text())
            if record['status']!='PASS' or record['optimizer_updates']!=0:
                raise PermissionError('zero-update qualification required')
        settings=root/('SETTINGS_'+arm+'.json');worker=json.loads(settings.read_text())
        if worker['arm']!=arm or worker.get('preflight_only') or Path(worker['output']).exists():
            raise PermissionError('invalid or existing worker output')
        workers.append((arm,settings,worker))
    with (root/'CONTROLLER_STARTED.json').open('x') as f:
        json.dump(dict(started=time.time(),physical_cap=2*STEPS),f)
    running=[]
    for arm,settings,worker in workers:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',TASK_SETTINGS=str(settings),SSLCL_STORAGE_ROOT=cfg['storage_root'])
        env.pop('TASK_MODE',None)
        with (root/(arm+'.log')).open('x') as log:
            p=subprocess.Popen(['bash',cfg['wrapper'],cfg['runtime'],cfg['entry']],env=env,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active=dict(arm=arm,pid=p.pid,birth=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],
                    started=time.time(),gpu=2,output=worker['output'],physical_cap=STEPS)
        write(root/('ACTIVE_'+arm+'.json'),active);running.append((arm,p))
    write(root/'STATUS.json',dict(status='RUNNING',physical_cap=2*STEPS))
    results=[dict(arm=arm,exit_code=p.wait()) for arm,p in running]
    write(root/'STATUS.json',dict(status='COMPLETE' if all(r['exit_code']==0 for r in results) else 'TERMINAL_WITH_FAILURE',results=results))
