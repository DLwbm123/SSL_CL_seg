"""A create-only, bounded queue: statistics, qualification, three frozen arms."""
import json
import os
import subprocess
import time
from pathlib import Path
from .core import ARMS
from ..main_style_ssl_v1.runtime import write


def main(cfg):
    root=Path(cfg['root'])
    with (root/'CONTROLLER_STARTED.json').open('x') as f:
        json.dump(dict(started=time.time(),formal_cap=12000,engineering_cap=96),f)
    def launch(mode,settings,logname):
        if time.time()>=cfg['deadline_unix']: raise PermissionError('authorization window ended')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',TASK_MODE=mode,TASK_SETTINGS=str(settings),
                 SSLCL_STORAGE_ROOT=cfg['storage_root'])
        with (root/logname).open('x') as log:
            process=subprocess.Popen(['bash',cfg['wrapper'],cfg['runtime'],cfg['entry']],env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/('ACTIVE_'+logname+'.json'),dict(pid=process.pid,gpu=3,mode=mode,
              birth=Path('/proc/'+str(process.pid)+'/stat').read_text().split()[21],started=time.time()))
        return process
    try:
        write(root/'STATUS.json',dict(status='SOURCE_STATISTICS',formal_updates=0))
        p=launch('statistics',root/'SETTINGS_dense_fixmatch.json','statistics.log')
        if p.wait()!=0: raise RuntimeError('source statistics failed; no training launched')
        for arm in ARMS:
            write(root/'STATUS.json',dict(status='QUALIFYING',arm=arm))
            p=launch('qualify',root/('QUAL_'+arm+'.json'),'qualification_'+arm+'.log')
            if p.wait()!=0: raise RuntimeError('qualification failed: '+arm)
        processes=[]
        for arm in ARMS:
            qualification=json.loads((root/'qualification'/arm/'ENGINEERING.json').read_text())
            if qualification['status']!='PASS' or qualification['physical_optimizer_updates']!=32:
                raise PermissionError('engineering receipt mismatch')
            settings=root/('SETTINGS_'+arm+'.json'); worker=json.loads(settings.read_text())
            if Path(worker['output']).exists(): raise FileExistsError(worker['output'])
            processes.append((arm,launch('train',settings,arm+'.log')))
        write(root/'STATUS.json',dict(status='RUNNING_PARALLEL',arms=list(ARMS),formal_cap=12000,engineering_cap=96))
        results=[dict(arm=arm,exit_code=p.wait()) for arm,p in processes]
        write(root/'STATUS.json',dict(status='COMPLETE' if all(r['exit_code']==0 for r in results)
              else 'TERMINAL_WITH_FAILURE',results=results))
    except BaseException as exc:
        write(root/'CONTROLLER_FAILURE.json',dict(error=str(exc),time=time.time()))
        write(root/'STATUS.json',dict(status='BLOCKED',error=str(exc))); raise
