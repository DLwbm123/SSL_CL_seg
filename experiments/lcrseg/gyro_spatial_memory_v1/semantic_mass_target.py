"""One fixed R0 transfer with repaired memory and the unchanged target engine."""
import json,os,time,subprocess,traceback
from pathlib import Path
from . import target,runtime

def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main(cfg):
    root=Path(cfg['root'])
    def guard(c,planned=0,new_stage=False):
        elapsed=time.time()-cfg['started_epoch']
        if elapsed>=6300 or (new_stage and elapsed>=5400):raise TimeoutError('target original 2h clock')
        used=int(subprocess.check_output(['du','-s','-B1',str(root)],text=True).split()[0])+2_000_000
        if used+planned>=800_000_000:raise OSError('target 800MB protection plus 200MB reserve')
        return dict(elapsed_seconds=elapsed,allocated_bytes_including_metadata=used)
    runtime.guard=guard;target.guard=guard
    mode=os.environ.get('TASK_MODE','controller')
    if mode!='controller':
        import torch
        torch.set_num_threads(4);torch.manual_seed(163);torch.cuda.manual_seed_all(163);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        {'qualify':target.qualify,'train':target.main,'evaluate':target.evaluate}[mode](cfg);return
    try:
        for mode,phase in [('qualify','engineering'),('train','formal'),('evaluate','formal')]:
            guard(cfg,new_stage=True)
            out=root/('qualification/A' if phase=='engineering' else 'jobs/A')
            current=dict(cfg,arm='A',phase=phase,stage='A',output=str(out),preflight_only=False,checkpoint_each_step=True,save_exception_state=True,diagnostic_holdout_excluded=False,audit_source_training=True)
            path=root/('SETTINGS_'+mode+'.private.json');write(path,current)
            free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=14000,'GPU3 free admission'
            with (root/(mode+'.log')).open('x') as log:
                p=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(path),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
            active=dict(pid=p.pid,birth=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],mode=mode,phase=phase)
            write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',mode=mode,phase=phase))
            code=p.wait();write(root/('EXIT_'+mode+'.json'),dict(exit_code=code))
            if code:raise RuntimeError('target stage failed: '+mode)
        write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException as e:
        write(root/'FAILURE.json',dict(error=str(e),traceback=traceback.format_exc()));write(root/'STATUS.json',dict(status='FAILED'));raise
