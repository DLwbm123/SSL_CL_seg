"""Operation-boundary counters, persisted at updates/snapshots and on every exit."""
import contextlib,json,time
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import h5py,torch
from . import core
from experiments.lcrseg.di_dmpa_jascl.modeling import LCRSegUNet2DJASCL
from experiments.lcrseg.single_teacher_scd_v0_1.data import CurrentData

ACTIVE=None
def flush(event):
    if ACTIVE is not None:ACTIVE.flush(event)

class Operations:
    def __init__(self,root,update_cap=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.counts=Counter();self.cap=update_cap
    def flush(self,event):
        with (self.root/'operation_events.jsonl').open('a') as f:f.write(json.dumps(dict(event=event,counts=dict(self.counts),time=time.time()))+'\n')
    def __enter__(self):
        global ACTIVE
        if ACTIVE is not None:raise RuntimeError('nested counter scope')
        ACTIVE=self;self.stack=contextlib.ExitStack()
        def instrument(owner,name,label,pre=None,after=None):
            original=getattr(owner,name)
            def call(*a,**kw):
                if pre:pre(*a,**kw)
                self.counts[label+'_attempts']+=1
                result=original(*a,**kw);self.counts[label]+=1
                if after:after(*a,**kw)
                return result
            self.stack.enter_context(patch.object(owner,name,call))
        def budget(*a,**kw):
            if self.cap is not None and self.counts['optimizer_steps']>=self.cap:raise RuntimeError('qualification update cap')
        instrument(torch.optim.Adam,'step','optimizer_steps',budget,lambda *a,**kw:self.flush('optimizer_step'))
        instrument(torch.Tensor,'backward','backward')
        instrument(torch.autograd,'grad','autograd_grad')
        instrument(LCRSegUNet2DJASCL,'forward','model_forward',after=lambda m,x,**kw:self.counts.update(image_forward=len(x)))
        def io_before(ds,i):
            self.counts['sample_'+ds.role+'_attempts']+=1
        def io_after(ds,i):
            self.counts['sample_'+ds.role]+=1
        instrument(CurrentData,'__getitem__','sample_access',io_before,io_after)
        def h5_before(name,*a,**kw):
            mode=a[0] if a else kw.get('mode','r');self.counts['hdf5_mode_'+str(mode)]+=1
        instrument(h5py,'File','hdf5_open',h5_before)
        # Imported aliases are patched where used; each call still invokes the original once.
        for module in (core,core.predecessor,core.predecessor.old):
            instrument(module,'update_ema','ema_updates')
        self.flush('started');return self
    def __exit__(self,kind,error,tb):
        global ACTIVE
        self.stack.close();self.flush('failed' if error else 'complete')
        (self.root/'operation_counts.json').write_text(json.dumps(dict(status='FAIL' if error else 'PASS',error=repr(error) if error else None,counts=dict(self.counts)),indent=2)+'\n')
        ACTIVE=None
