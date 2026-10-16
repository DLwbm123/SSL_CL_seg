"""Exactly 32 current-training-L batches at one copied final anchor; no optimizer."""
import copy
import random
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from ..main_style_ssl_v1.data import Accessor, CurrentDomain
from .memory import seal_site, finite
from .protocol import MEMORY_BATCHES


def collect(deployment,anchor,node,data,progress=None):
    if node['dataset']!='mnms':raise PermissionError('M&Ms training L only')
    device=next(deployment.parameters()).device
    # A private frozen copy and independent stateless stream leave training state intact.
    model=copy.deepcopy(deployment).eval().requires_grad_(False)
    state={n:v.detach().clone().to(device).requires_grad_(True) for n,v in anchor.items()}
    overrides={('transform' if n=='__F__' else 'parent.native.'+n):v for n,v in state.items()}
    access=Accessor(data,'mnms',node['domain'],'train_labeled')
    provider=SimpleNamespace(seed=node['seed'],order=node['order'],stage=node['stage'],dataset='mnms',device=device)
    observations=[];python_rng=random.getstate()
    devices=[] if device.type=='cpu' else [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=devices):
            for cursor in range(MEMORY_BATCHES):
                x,y,ids=CurrentDomain._batch(provider,access,cursor,'history_final')
                if len(set(ids))!=2:raise ValueError('two distinct current-L patients required')
                logits=torch.func.functional_call(model,overrides,(x,))
                logp=logits.log_softmax(1);batch=[]
                for c in range(4):
                    mask=y==c;pixels=int(mask.sum())
                    if not pixels:continue
                    loss=-(logp[:,c]*mask).sum()/pixels
                    gradients=torch.autograd.grad(loss,list(state.values()),allow_unused=True,retain_graph=True)
                    batch.append((c,float(loss.detach()),{n:torch.zeros_like(v,device='cpu') if g is None else g.detach().cpu() for (n,v),g in zip(state.items(),gradients)},pixels))
                if not finite(batch):raise FloatingPointError('nonfinite final-anchor observation')
                observations.append(batch)
                if progress:progress(dict(batch=cursor+1,L_image_exposures=2*(cursor+1),optimizer_calls=0))
                del logits,logp,batch,x,y
    finally:random.setstate(python_rng)
    site=seal_site(node['domain'],anchor,observations)
    return site,dict(L_batch_reads=32,L_image_exposures=64,U_reads=0,val_reads=0,test_reads=0,optimizer_calls=0,stream='history_final',fixed_anchor=True)
