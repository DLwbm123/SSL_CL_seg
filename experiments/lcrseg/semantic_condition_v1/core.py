"""Bounded semantic conditioning; native readout and backbone remain intact."""
import copy
import fcntl
import json
import math
import os
import time
from pathlib import Path
import torch
from torch import nn
from .module import SemanticCondition
from ..space_lora_v1.core import source_model, setup as dense_setup
from ..siemens_ssl_reproduction_v1.methods import optimizer, pixel_ce, strong_view

ARMS = ('dense_fixmatch', 'aligned_lonly', 'aligned_ssl', 'shuffled_lonly', 'shuffled_ssl')
STEPS = 3500


class Conditioned(nn.Module):
    def __init__(self, source, prototypes, shuffled):
        super().__init__()
        self.parent = copy.deepcopy(source.parent).eval().requires_grad_(False)
        self.conditioner = SemanticCondition(prototypes, shuffled=shuffled)

    def forward(self, x):
        h = self.parent.features(x)
        return self.parent.native_readout(self.conditioner(h), x.shape[-2:])

    def teacher(self):
        return copy.deepcopy(self).eval().requires_grad_(False)

    def deploy(self):
        # Spatial nonlinear conditioning cannot be merged into a fixed conv.
        return self.teacher()


def budget_hooks(opt, cfg):
    """Precharge under a shared lock; interrupted calls remain conservatively charged.

    Each worker also has a create-only run directory and its own physical ledger.
    No optimizer call can pass either the total cap or its preallocated arm cap.
    """
    root = Path(cfg['budget_root'])
    eventfile = root/'budget_events.jsonl'
    key = cfg['phase']+':'+cfg['arm']
    pending = []

    def event(kind):
        with (root/'budget.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            counterfile=root/'budget_counter.json'
            counter=json.loads(counterfile.read_text()) if counterfile.exists() else dict(total=0,keys={},offset=0)
            # Reconcile only the tail after a crash between append and counter commit.
            if eventfile.exists():
                with eventfile.open() as f:
                    f.seek(counter['offset'])
                    for line in f:
                        r=json.loads(line)
                        if r['kind']=='reserved':
                            counter['total']+=1
                            counter['keys'][r['key']]=counter['keys'].get(r['key'],0)+1
            if kind == 'reserved':
                if counter['total']>=cfg.get('budget_cap',20000) or counter['keys'].get(key,0)>=cfg['phase_cap']:
                    raise PermissionError('shared actual-call budget exhausted')
                token = counter['total']+1
                counter['total']+=1; counter['keys'][key]=counter['keys'].get(key,0)+1
                pending.append(token)
            else:
                token = pending.pop()
            row = dict(kind=kind, token=token, key=key, pid=os.getpid(), time=time.time())
            with eventfile.open('a') as f:
                f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())
                counter['offset']=f.tell()
            temporary=counterfile.with_suffix('.tmp')
            temporary.write_text(json.dumps(counter))
            os.replace(temporary,counterfile)
    opt.register_step_pre_hook(lambda *_: event('reserved'))
    opt.register_step_post_hook(lambda *_: event('executed'))


def setup(cfg):
    if cfg['arm'] not in ARMS: raise PermissionError('unregistered arm')
    torch.manual_seed(163); torch.cuda.manual_seed_all(163)
    if cfg['arm']=='dense_fixmatch':
        source, model, ema, opt = dense_setup(cfg)
    else:
        source = source_model(cfg)
        state = torch.load(cfg['statistics'], map_location='cpu', weights_only=False)
        if state['source_slices']!=230 or state['optimizer_updates']!=0:
            raise ValueError('prototype preparation receipt mismatch')
        model = Conditioned(source, state['prototypes'], cfg['arm'].startswith('shuffled')).cuda()
        model.train(); model.parent.eval().requires_grad_(False)
        ema = model.teacher()
        opt = optimizer(model, 'fixmatch')
    budget_hooks(opt, cfg)
    return source, model, ema, opt


def rate(step):
    if not 0<=step<STEPS: raise PermissionError('outside frozen pilot')
    return .03*math.cos(7*math.pi*step/(16*53300))


@torch.no_grad()
def update_ema(ema, model):
    if isinstance(model, Conditioned):
        # Update trainable residuals only: immutable floating buffers/weights
        # must not accumulate EMA roundoff drift.
        current = dict(model.conditioner.named_parameters())
        for name, value in ema.conditioner.named_parameters():
            value.mul_(.999).add_(current[name], alpha=.001)
    else:
        model.parent.update_dense_ema(ema.parent, decay=.999)


def losses(arm, model, data, step):
    x, y, _ = data.labeled(step)
    l = pixel_ce(model(x), y).mean()
    u = l*0
    counts = [0]*4; pixels = 0
    if not arm.endswith('_lonly'):
        images = torch.cat([data.unlabeled(step*7+i)[0] for i in range(7)])
        with torch.no_grad():
            confidence, labels = model(images).softmax(1).max(1)
            keep = confidence>=.95
        u = (pixel_ce(model(strong_view(images,step)), labels)*keep).mean()
        counts = [int(((labels==c)&keep).sum()) for c in range(4)]
        pixels = keep.numel()
    return l,u,dict(supervised=float(l.detach()),unsupervised=float(u.detach()),
                    accepted_class_pixels=counts,U_pixels=pixels)
