import json,os,subprocess,time
from pathlib import Path
from .core import ARMS
from ..main_style_ssl_v1.runtime import write


def main(cfg):
    root=Path(cfg['root'])
    with (root/'CONTROLLER_STARTED.json').open('x') as f:json.dump(dict(time=time.time(),cap=10160,no_later_domains=True),f)
    def launch(mode,path,name):
        env=dict(os.environ,TASK_MODE=mode,TASK_SETTINGS=str(path),CUDA_VISIBLE_DEVICES='3',SSLCL_STORAGE_ROOT=cfg['storage_root'],CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (root/(name+'.log')).open('x') as log:
            p=subprocess.Popen(['bash',cfg['wrapper'],cfg['runtime'],cfg['entry']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/('ACTIVE_'+name+'.json'),dict(pid=p.pid,birth=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],mode=mode,gpu=3,time=time.time()))
        if p.wait()!=0:raise RuntimeError('worker failed: '+name)
    try:
        launch('checks',root/'CONTROLLER.json','checks')
        write(root/'STATUS.json',dict(status='PREPARING',time=time.time()))
        launch('prepare',root/'CONTROLLER.json','prepare')
        for arm in ARMS:
            write(root/'STATUS.json',dict(status='QUALIFYING',arm=arm,time=time.time()))
            launch('qualify',root/('QUAL_'+arm+'.json'),'qual_'+arm)
        for arm in ARMS:
            q=json.loads((root/'qualification'/arm/'ENGINEERING.json').read_text())
            assert q['status']=='PASS' and q['actual_calls']==32
            free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
            if free<q['cuda_peak_allocated_bytes']*1.25+1024**3:raise MemoryError('GPU3 memory admission')
            write(root/'STATUS.json',dict(status='TRAINING',arm=arm,time=time.time()))
            launch('train',root/('SETTINGS_'+arm+'.json'),arm)
        from .decision import decide
        write(root/'DECISION.json',decide(root))
        write(root/'STATUS.json',dict(status='COMPLETE_NO_LATER_DOMAINS',time=time.time()))
    except BaseException as exc:
        write(root/'CONTROLLER_FAILURE.json',dict(error=str(exc),time=time.time()))
        write(root/'STATUS.json',dict(status='BLOCKED',error=str(exc),time=time.time()));raise
