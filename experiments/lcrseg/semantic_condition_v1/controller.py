"""Create-only queue, five matched arms, no automatic retry or tuning."""
import json
import os
import subprocess
import time
from pathlib import Path
from .core import ARMS
from ..main_style_ssl_v1.runtime import write


def main(cfg):
    root=Path(cfg['root'])
    settings_prefix=cfg.get('settings_prefix','SETTINGS_')
    qualification_prefix=cfg.get('qualification_prefix','QUAL_')
    receipts=root/cfg['attempt'] if cfg.get('attempt') else root
    if receipts!=root: receipts.mkdir(exist_ok=False)
    with (receipts/'CONTROLLER_STARTED.json').open('x') as f:
        json.dump(dict(started=time.time(),formal_cap=17500,
                       engineering_cap=cfg.get('engineering_cap',160),round_cap=20000),f)

    def launch(mode,settings,logname):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',TASK_MODE=mode,
                 TASK_SETTINGS=str(settings),SSLCL_STORAGE_ROOT=cfg['storage_root'],
                 CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (receipts/logname).open('x') as log:
            p=subprocess.Popen(['bash',cfg['wrapper'],cfg['runtime'],cfg['entry']],env=env,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(receipts/('ACTIVE_'+logname+'.json'),dict(pid=p.pid,gpu=3,mode=mode,
              birth=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],started=time.time()))
        return p

    try:
        if cfg.get('statistics_ready'):
            settings=json.loads((root/(settings_prefix+'dense_fixmatch.json')).read_text())
            stats=Path(settings['statistics'])
            receipt=json.loads(stats.with_suffix('.json').read_text())
            if not stats.exists() or receipt['source_slices']!=230 or receipt['optimizer_updates']!=0:
                raise PermissionError('existing source preparation receipt mismatch')
        else:
            write(root/'STATUS.json',dict(status='SOURCE_STATISTICS',formal_updates=0))
            p=launch('statistics',root/(settings_prefix+'dense_fixmatch.json'),'statistics.log')
            if p.wait()!=0: raise RuntimeError('source statistics failed; no training launched')
        peaks=[]
        for arm in ARMS:
            write(root/'STATUS.json',dict(status='QUALIFYING',arm=arm))
            if arm not in cfg.get('qualification_ready_arms',[]):
                p=launch('qualify',root/(qualification_prefix+arm+'.json'),'qualification_'+arm+'.log')
                if p.wait()!=0: raise RuntimeError('qualification failed: '+arm)
            qualification=json.loads((root/(qualification_prefix+arm+'.json')).read_text())
            q=json.loads((Path(qualification['output'])/'ENGINEERING.json').read_text())
            if q['status']!='PASS' or q['physical_optimizer_updates']!=32:
                raise PermissionError('engineering receipt mismatch')
            peaks.append(q['cuda_peak_allocated_bytes'])
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free',
                     '--format=csv,noheader,nounits'],text=True).strip())*1024**2
        # Preflight includes the real U14 backward and nonzero deployment/EMA.
        # Include allocator/context overhead and 25% headroom before sharing.
        required=sum(peaks)*1.25+len(ARMS)*1024**3
        if free<required:
            raise MemoryError('parallel admission insufficient; preserve qualification, no blind launch')
        processes=[]
        for arm in ARMS:
            settings=root/(settings_prefix+arm+'.json'); worker=json.loads(settings.read_text())
            if Path(worker['output']).exists(): raise FileExistsError(worker['output'])
            processes.append((arm,launch('train',settings,arm+'.log')))
        write(root/'STATUS.json',dict(status='RUNNING_PARALLEL',arms=list(ARMS),
              formal_cap=17500,engineering_cap=cfg.get('engineering_cap',160),free_bytes_at_admission=free,
              estimated_required_bytes=required))
        results=[dict(arm=arm,exit_code=p.wait()) for arm,p in processes]
        write(root/'STATUS.json',dict(status='COMPLETE' if all(r['exit_code']==0 for r in results)
              else 'TERMINAL_WITH_FAILURE',results=results))
    except BaseException as exc:
        write(receipts/'CONTROLLER_FAILURE.json',dict(error=str(exc),time=time.time()))
        write(root/'STATUS.json',dict(status='BLOCKED',error=str(exc))); raise
