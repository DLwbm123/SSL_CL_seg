"""One create-only launch per authorized arm; no retries or continuation."""
import json
import os
import subprocess
import time
from pathlib import Path
from .model import ARMS
from ..main_style_ssl_v1.runtime import write


def main(cfg):
    root=Path(cfg['root'])
    with (root/'CONTROLLER_STARTED.json').open('x') as f:
        json.dump(dict(started=time.time(),physical_cap=30000),f)
    processes=[]
    for arm in ARMS:
        qualification=json.loads((root/'qualification'/arm/'PREFLIGHT.json').read_text())
        if qualification['status']!='PASS' or qualification['optimizer_updates']!=0:
            raise PermissionError('missing zero-update qualification')
        settings=root/('SETTINGS_'+arm+'.json'); worker=json.loads(settings.read_text())
        if worker['arm']!=arm or worker.get('preflight_only') or Path(worker['output']).exists():
            raise PermissionError('output exists or frozen arm changed')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',TASK_SETTINGS=str(settings),SSLCL_STORAGE_ROOT=cfg['storage_root'])
        env.pop('TASK_MODE',None)
        with (root/(arm+'.log')).open('x') as log:
            process=subprocess.Popen(['bash',cfg['wrapper'],cfg['runtime'],cfg['entry']],env=env,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active=dict(arm=arm,pid=process.pid,birth=Path('/proc/'+str(process.pid)+'/stat').read_text().split()[21],
                    started=time.time(),gpu=2,physical_cap=10000,output=worker['output'])
        write(root/('ACTIVE_'+arm+'.json'),active); processes.append((arm,process))
    write(root/'STATUS.json',dict(status='RUNNING_PARALLEL',arms=list(ARMS),physical_cap=30000))
    results=[dict(arm=arm,exit_code=process.wait()) for arm,process in processes]
    write(root/'STATUS.json',dict(status='COMPLETE' if all(r['exit_code']==0 for r in results) else 'TERMINAL_WITH_FAILURE',results=results))


if __name__=='__main__': main(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
