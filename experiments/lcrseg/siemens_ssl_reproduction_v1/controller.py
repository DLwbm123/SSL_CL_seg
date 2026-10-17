"""Detached sequential execution; never retry or overwrite an existing arm."""
import json
import os
import subprocess
import time
from pathlib import Path
from ..main_style_ssl_v1.runtime import write


def main(cfg):
    root=Path(cfg['root'])
    with (root/'CONTROLLER_STARTED.json').open('x') as f:
        json.dump({'started':time.time(),'new_physical_cap':130800},f)
    results=[]
    for method in ('fixmatch','cps'):
        qualification=json.loads((Path(cfg['qualification_root'])/method/'PREFLIGHT.json').read_text())
        if qualification['status']!='PASS' or qualification['optimizer_updates']!=0:
            raise PermissionError('zero-update native qualification missing')
        settings=root/('SETTINGS_'+method+'.json')
        worker=json.loads(settings.read_text())
        if worker['method']!=method or worker.get('preflight_only') or Path(worker['output']).exists():
            raise ValueError('worker configuration changed or output already exists')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',TASK_SETTINGS=str(settings),SSLCL_STORAGE_ROOT=cfg['storage_root'])
        env.pop('TASK_MODE',None)
        with (root/(method+'.log')).open('x') as log:
            process=subprocess.Popen(['bash',cfg['wrapper'],cfg['runtime'],cfg['entry']],env=env,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active=dict(method=method,pid=process.pid,birth=Path('/proc/'+str(process.pid)+'/stat').read_text().split()[21],
                    started=time.time(),gpu=2,output=worker['output'],physical_cap=43600 if method=='fixmatch' else 87200)
        write(root/('ACTIVE_'+method+'.json'),active)
        write(root/'STATUS.json',dict(status='RUNNING',active=active,completed=results,
                                     queued=['cps'] if method=='fixmatch' else []))
        code=process.wait()
        results.append(dict(method=method,exit_code=code))
    write(root/'STATUS.json',dict(status='COMPLETE' if all(r['exit_code']==0 for r in results) else 'TERMINAL_WITH_FAILURE',results=results))


if __name__=='__main__':main(json.loads(Path(os.environ['TASK_CONTROLLER_SETTINGS']).read_text()))
