"""Independent clocks, physical update ledger and forward caps for new frozen rounds."""
import fcntl,json,subprocess,time
from pathlib import Path
from . import runtime
from ..main_style_ssl_v1.runtime import write,append


def install(cfg):
    def guard(c,planned=0,new_stage=False):
        elapsed=time.time()-cfg['started_epoch']
        if elapsed>=cfg['stop_seconds'] or (new_stage and elapsed>=cfg['no_new_seconds']):
            raise TimeoutError('original independent clock exhausted')
        used=int(subprocess.check_output(['du','-s','-B1',cfg['root']],text=True).split()[0])+2_000_000
        if used+planned>=cfg['protect_bytes']:raise OSError('independent storage protection exhausted')
        return dict(elapsed_seconds=elapsed,allocated_bytes_including_metadata=used)
    def event(c,kind,stage,phase):
        root=Path(cfg['root']);key=c.get('group',stage)+':'+phase
        with (root/'budget.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            p=root/'budget_counter.json'
            d=json.loads(p.read_text()) if p.exists() else dict(total=0,optimizer=0,functional=0,pending=0,by_phase={})
            if kind=='reserved':
                assert d['pending']==0
                if d['total']>=cfg['update_cap'] or d['by_phase'].get(key,0)>=cfg['phase_caps'][key]:raise PermissionError('frozen update cap')
                d['pending']=1
            else:
                assert kind=='optimizer' and d['pending']==1
                d['pending']=0;d['total']+=1;d['optimizer']+=1;d['by_phase'][key]=d['by_phase'].get(key,0)+1
            append(root/'budget_events.jsonl',dict(kind=kind,key=key,time=time.time(),total=d['total']))
            write(p,d)
    runtime.guard=guard;runtime.event=event
    return guard


class ForwardBudget:
    def __init__(self,cfg,caps):self.cfg=cfg;self.caps=caps;self.count={k:0 for k in caps};self.pending=None
    def call(self,kind,n,fn,*args):
        if self.pending is not None or self.count[kind]+n>self.caps[kind]:raise PermissionError('frozen forward cap')
        runtime.guard(self.cfg);self.pending=dict(kind=kind,samples=n)
        self.save();result=fn(*args);self.count[kind]+=n;self.pending=None;self.save();return result
    def save(self):
        write(Path(self.cfg['root'])/'forward_counter.json',dict(executed=self.count,caps=self.caps,pending=self.pending))
