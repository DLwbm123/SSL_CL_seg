import json,os,subprocess,time
from pathlib import Path
from .core import ARMS
from ..main_style_ssl_v1.runtime import write


def main(cfg):
    root=Path(cfg['root'])
    with (root/'CONTROLLER_STARTED.json').open('x') as f:json.dump(dict(started=time.time(),cap=404,no_target=True),f)
    def launch(mode,settings,name):
        env=dict(os.environ,TASK_MODE=mode,TASK_SETTINGS=str(settings),CUDA_VISIBLE_DEVICES='3',SSLCL_STORAGE_ROOT=cfg['storage_root'],CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (root/(name+'.log')).open('x') as log:
            p=subprocess.Popen(['bash',cfg['wrapper'],cfg['runtime'],cfg['entry']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/('ACTIVE_'+name+'.json'),dict(pid=p.pid,birth=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],mode=mode,gpu=3,time=time.time()))
        if p.wait()!=0:raise RuntimeError('worker failed: '+name)
    try:
        launch('checks',root/'CONTROLLER.json','checks')
        for arm in ARMS:
            write(root/'STATUS.json',dict(status='QUALIFYING',arm=arm,time=time.time()))
            launch('qualify',root/('QUAL_'+arm+'.json'),'qual_'+arm)
            q=json.loads((root/'qualification'/arm/'ENGINEERING.json').read_text())
            free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
            if free<q['cuda_peak_allocated_bytes']*1.25+1024**3:raise MemoryError('GPU3 memory admission')
            write(root/'STATUS.json',dict(status='TRAINING',arm=arm,time=time.time()))
            launch('train',root/('SETTINGS_'+arm+'.json'),arm)
        result=json.loads((root/'jobs/GYRO_low/RESULTS.json').read_text())
        write(root/'STATUS.json',dict(status='COMPLETE_SOURCE_GATES_PASS_NO_TARGET' if result['all_gates_passed'] else 'SCIENTIFIC_STOP_SOURCE_GATE',result=result,time=time.time()))
    except BaseException as exc:
        write(root/'CONTROLLER_FAILURE.json',dict(error=str(exc),time=time.time()))
        write(root/'STATUS.json',dict(status='BLOCKED',error=str(exc),time=time.time()));raise
