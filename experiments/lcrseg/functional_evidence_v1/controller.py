"""Create-only bounded controller: qualify, plasticity gate, then mechanism pilot."""
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
        json.dump(dict(time=time.time(),formal_cap=2000,engineering_cap=160),f)
    def launch(mode,settings,name):
        env=dict(os.environ,TASK_MODE=mode,TASK_SETTINGS=str(settings),CUDA_VISIBLE_DEVICES='3',
                 SSLCL_STORAGE_ROOT=cfg['storage_root'],CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (receipts/(name+'.log')).open('x') as log:
            p=subprocess.Popen(['bash',cfg['wrapper'],cfg['runtime'],cfg['entry']],env=env,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(receipts/('ACTIVE_'+name+'.json'),dict(pid=p.pid,
            birth=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],time=time.time(),mode=mode))
        return p
    try:
        control=Path(os.environ['TASK_SETTINGS'])
        if not cfg.get('checks_ready'):
            p=launch('checks',control,'checks')
            if p.wait()!=0: raise RuntimeError('CPU state contract checks failed')
        else:
            assert json.loads((root/'SYNTHETIC_CHECKS.json').read_text())['status']=='PASS'
        p=launch('bank',control,'bank')
        if p.wait()!=0: raise RuntimeError('one-time memory preparation failed')
        peaks=[]
        for arm in ARMS:
            write(root/'STATUS.json',dict(status='QUALIFYING',arm=arm))
            p=launch('qualify',root/('QUAL_'+arm+'.json'),'qualification_'+arm)
            if p.wait()!=0: raise RuntimeError('qualification failed: '+arm)
            receipt=json.loads((root/'qualification'/arm/'ENGINEERING.json').read_text())
            if receipt['status']!='PASS' or receipt['physical_optimizer_updates']!=32:
                raise ValueError('qualification receipt mismatch')
            peaks.append(receipt['cuda_peak_allocated_bytes'])
        for i,arm in enumerate(ARMS):
            free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'],text=True).strip())*1024**2
            if free<peaks[i]*1.25+1024**3: raise MemoryError('insufficient GPU3 memory')
            write(root/'STATUS.json',dict(status='TRAINING',arm=arm,steps=400))
            p=launch('train',root/('SETTINGS_'+arm+'.json'),arm)
            if p.wait()!=0: raise RuntimeError('formal worker failed: '+arm)
            if arm=='native_lonly':
                a=json.loads((root/'jobs/dense_lonly/RESULTS.json').read_text())
                b=json.loads((root/'jobs/native_lonly/RESULTS.json').read_text())
                differences={key:100*(b[key]['primary_dice']-a[key]['primary_dice']) for key in ('train','Philips')}
                passed=min(differences.values())>=-1.
                write(root/'PLASTICITY_GATE.json',dict(status='PASS' if passed else 'FAIL',
                     native_minus_dense_pp=differences,steps_per_arm=400))
                if not passed:
                    write(root/'STATUS.json',dict(status='SCIENTIFIC_STOP_PLASTICITY',
                         reason='Frozen enc1 misses predeclared 1pp training/validation gate',unused_formal_updates=1200))
                    return
        write(root/'STATUS.json',dict(status='COMPLETE',formal_updates=2000,engineering_updates=160))
    except BaseException as exc:
        write(receipts/'CONTROLLER_FAILURE.json',dict(error=str(exc),time=time.time()))
        write(root/'STATUS.json',dict(status='BLOCKED',error=str(exc)))
        raise
