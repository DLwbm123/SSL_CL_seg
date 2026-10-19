"""Create-only, one-shot source qualification/preparation, then five sequential arms."""
import json
import os
from pathlib import Path
import subprocess
import time
from .core import ARMS
from ..main_style_ssl_v1.runtime import write


def main(cfg):
    root=Path(cfg['root'])
    receipts=root/cfg['attempt'] if cfg.get('attempt') else root
    if receipts!=root: receipts.mkdir(exist_ok=False)
    with (receipts/'CONTROLLER_STARTED.json').open('x') as f:
        json.dump(dict(started=time.time(),budget_cap=10330,automatic_retry=False),f)
    def launch(mode,settings,name):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',TASK_MODE=mode,TASK_SETTINGS=str(settings),
                 CUBLAS_WORKSPACE_CONFIG=':4096:8',SSLCL_STORAGE_ROOT=cfg['storage_root'])
        with (receipts/(name+'.log')).open('x') as log:
            p=subprocess.Popen(['bash',cfg['wrapper'],cfg['runtime'],cfg['entry']],env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(receipts/('ACTIVE_'+name+'.json'),dict(pid=p.pid,birth=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],mode=mode,gpu=3,time=time.time()))
        if p.wait()!=0: raise RuntimeError('worker failed: '+name)
    def stage(status,arm):
        write(root/'STATUS.json',dict(status=status,arm=arm,time=time.time()))
    try:
        if (root/'SYNTHETIC_CHECKS.json').exists():
            assert json.loads((root/'SYNTHETIC_CHECKS.json').read_text())['status']=='PASS'
        else:
            launch('checks',root/'CONTROLLER.json','checks')
        stage('QUALIFYING','source_prepare')
        launch('qualify',root/'QUAL_source_prepare.json','qual_source_prepare')
        stage('SOURCE_PREPARATION','source_prepare')
        launch('train',root/'SETTINGS_source_prepare.json','source_prepare')
        result=json.loads((root/'source/RESULTS.json').read_text())
        if not result['source_gate_passed']:
            write(root/'STATUS.json',dict(status='SCIENTIFIC_STOP_SOURCE_GATE',source=result,time=time.time()))
            return
        # One worker at a time; all formal arms start from identical immutable prepared state.
        for arm in ARMS:
            stage('QUALIFYING',arm)
            launch('qualify',root/('QUAL_'+arm+'.json'),'qual_'+arm)
        for arm in ARMS:
            q=json.loads((root/'qualification'/arm/'ENGINEERING.json').read_text())
            free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
            if free<q['cuda_peak_allocated_bytes']*1.25+1024**3: raise MemoryError('GPU3 admission insufficient')
            stage('TRAINING',arm)
            launch('train',root/('SETTINGS_'+arm+'.json'),arm)
        write(root/'STATUS.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as exc:
        write(receipts/'CONTROLLER_FAILURE.json',dict(error=str(exc),time=time.time()))
        write(root/'STATUS.json',dict(status='BLOCKED',error=str(exc),time=time.time()));raise
