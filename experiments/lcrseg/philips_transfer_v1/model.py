import copy
import math
import torch
from ..main_style_ssl_v1.model import Parent
from ..five_frameworks_v1.model import Model
from ..siemens_ssl_reproduction_v1.methods import Segmenter, optimizer, pixel_ce, strong_view

ARMS = ('dense_fixmatch', 'cl_lonly', 'cl_fixmatch')
STEPS = 10000


def setup(cfg):
    if cfg['arm'] not in ARMS:
        raise PermissionError('unregistered arm')
    torch.manual_seed(163); torch.cuda.manual_seed_all(163)
    source = Segmenter(cfg['reference']).cuda()
    saved = torch.load(cfg['source_checkpoint'], map_location='cpu', weights_only=False)
    if saved['method'] != 'fixmatch' or saved['step'] != 43600 or saved['physical_updates'] != 43600:
        raise ValueError('wrong completed source')
    source.load_state_dict(saved['ema'], strict=True)
    source.eval().requires_grad_(False)
    dense = cfg['arm'] == 'dense_fixmatch'
    parent = Parent(copy.deepcopy(source.parent.native), 163,
                    {'kind': 'completed_FixMatch_EMA', 'domain': 'Siemens'}, source=dense)
    model = Model(parent, 'MAIN_SOURCE' if dense else 'F5', .25).cuda()
    if model.sidecar is not None:
        model.sidecar.q = model.sidecar.q.contiguous()
    model.configure_training()
    ema = model.teacher()
    return source, model, ema, optimizer(model, 'fixmatch')


def rate(step):
    if not 0 <= step < STEPS:
        raise PermissionError('outside truncated schedule')
    return .03 * math.cos(7 * math.pi * step / (16 * 53300))


@torch.no_grad()
def update_ema(ema, model):
    model.parent.update_dense_ema(ema.parent, decay=.999)
    if model.sidecar is not None:
        model.sidecar.update_teacher(ema.sidecar, decay=.999)


def losses(arm, model, provider, step):
    x, y, _ = provider.labeled(step)
    supervised = pixel_ce(model(x), y).mean()
    unsupervised = supervised * 0
    counts = [0]*4; pixels = 0
    if arm != 'cl_lonly':
        u = torch.cat([provider.unlabeled(step*7+i)[0] for i in range(7)])
        with torch.no_grad():
            confidence, labels = model(u).softmax(1).max(1)
            keep = confidence >= .95
        unsupervised = (pixel_ce(model(strong_view(u, step)), labels)*keep).mean()
        counts = [int(((labels == c) & keep).sum()) for c in range(4)]
        pixels = keep.numel()
    return supervised, unsupervised, dict(supervised=float(supervised.detach()),
        unsupervised=float(unsupervised.detach()), accepted_class_pixels=counts, U_pixels=pixels)
