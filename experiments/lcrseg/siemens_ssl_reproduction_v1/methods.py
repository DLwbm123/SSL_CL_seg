import math
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageEnhance, ImageOps
from ..five_frameworks_v1.native_parent import build
from ..main_style_ssl_v1.model import Parent
from ..di_dmpa_jascl.data import stable_seed

STEPS = 43600


def pixel_ce(logits, labels):
    if labels.dtype != torch.long or labels.shape != logits.shape[:1]+logits.shape[2:]:
        raise ValueError('invalid CE label grid')
    # CUDA's fused spatial NLL mean lacks deterministic support on this runtime.
    return -logits.log_softmax(1).gather(1, labels[:,None])[:,0]


class Segmenter(nn.Module):
    def __init__(self, reference, branch=0):
        super().__init__()
        init_seed = 163 if branch == 0 else stable_seed(163, 'CPS', 'branch2')
        native = build(reference, torch.device('cuda'), init_seed, input_channels=1, num_classes=4, domain='Siemens')
        self.parent = Parent(native, init_seed, {'kind': 'plain_supervised_backbone'}, source=True)
        self.parent.configure_stage_training()
        if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.modules()):
            raise ValueError('separate weak forwards require the frozen non-BatchNorm backbone')

    def forward(self, x):
        return self.parent.native_readout(self.parent.features(x), x.shape[-2:])


def fixmatch_loss(strong_logits, weak_logits):
    if strong_logits.shape != weak_logits.shape:
        raise ValueError('weak/strong output grids differ')
    confidence, labels = weak_logits.detach().softmax(1).max(1)
    keep = confidence >= .95
    loss = (pixel_ce(strong_logits, labels)*keep).mean()
    return loss, float(keep.float().mean())


def cps_loss(left, right):
    if left.shape != right.shape:
        raise ValueError('CPS branch output grids differ')
    return pixel_ce(left, right.detach().argmax(1)).mean() + pixel_ce(right, left.detach().argmax(1)).mean()


def strong_view(images, step):
    """Two grayscale photometric RandAugment operations plus half-width Cutout.

    Spatial changes are shared in the weak view so dense pseudo labels stay aligned.
    This excludes classification RandAugment's independent geometric operations.
    """
    rng = np.random.default_rng(stable_seed(163, step, 'fixmatch_strong'))
    result = []
    for image in images.detach().cpu():
        pixels = ((image[0].numpy()+1)*127.5).clip(0,255).round().astype(np.uint8)
        im = Image.fromarray(pixels)
        for _ in range(2):
            op = int(rng.integers(8)); level = float(rng.uniform(0,1)); sign = int(rng.choice([-1,1]))
            if op == 1: im = ImageOps.autocontrast(im)
            elif op == 2: im = ImageOps.equalize(im)
            elif op == 3: im = ImageEnhance.Brightness(im).enhance(1+sign*.9*level)
            elif op == 4: im = ImageEnhance.Contrast(im).enhance(1+sign*.9*level)
            elif op == 5: im = ImageEnhance.Sharpness(im).enhance(1+sign*.9*level)
            elif op == 6: im = ImageOps.posterize(im, max(1,8-round(4*level)))
            elif op == 7: im = ImageOps.solarize(im, round(256*(1-level)))
        a = np.asarray(im).copy(); h,w = a.shape; side = min(h,w)//2
        cy,cx = int(rng.integers(h)),int(rng.integers(w))
        a[max(0,cy-side//2):min(h,cy+side//2),max(0,cx-side//2):min(w,cx+side//2)] = 128
        result.append(torch.from_numpy(a.astype(np.float32)/127.5-1)[None])
    return torch.stack(result).to(images.device)


def learning_rate(method, step):
    if not 0 <= step < STEPS: raise ValueError('step outside frozen schedule')
    return .03*math.cos(7*math.pi*step/(16*STEPS)) if method == 'fixmatch' else .005*(1-step/STEPS)**.9


def optimizer(model, method):
    decay = 5e-4 if method == 'fixmatch' else 1e-4
    groups = [dict(params=[p for p in model.parameters() if p.requires_grad and p.ndim>1],weight_decay=decay),
              dict(params=[p for p in model.parameters() if p.requires_grad and p.ndim<=1],weight_decay=0.)]
    return torch.optim.SGD(groups, lr=learning_rate(method,0), momentum=.9, nesterov=method=='fixmatch')


@torch.no_grad()
def update_ema(ema, model):
    for name, value in ema.state_dict().items():
        current = model.state_dict()[name]
        if value.is_floating_point(): value.mul_(.999).add_(current,alpha=.001)
        else: value.copy_(current)


def losses(method, models, provider, step):
    x,y,_ = provider.labeled(step)
    if method == 'fullsupervised':
        batches=[provider.unlabeled(step*7+i) for i in range(7)]
        u=torch.cat([b[0] for b in batches]); truth=torch.cat([b[1] for b in batches])
        supervised=pixel_ce(models[0](x),y).mean()
        additional=pixel_ce(models[0](strong_view(u,step)),truth).mean()
        return supervised+additional, dict(supervised=float(supervised.detach()),additional_gt=float(additional.detach()),supervision_fraction=1.)
    if method == 'fixmatch':
        u = torch.cat([provider.unlabeled(step*7+i)[0] for i in range(7)])
        with torch.no_grad(): weak = models[0](u)
        supervised = pixel_ce(models[0](x),y).mean()
        unsupervised, kept = fixmatch_loss(models[0](strong_view(u,step)),weak)
        return supervised+unsupervised, dict(supervised=float(supervised.detach()),unsupervised=float(unsupervised.detach()),accepted_fraction=kept)
    u,_,_ = provider.unlabeled(step)
    inputs = torch.cat([x,u]); left,right = [m(inputs) for m in models]
    supervised = pixel_ce(left[:len(x)],y).mean()+pixel_ce(right[:len(x)],y).mean()
    cross = cps_loss(left,right)
    return supervised+1.5*cross, dict(supervised=float(supervised.detach()),cps_unweighted=float(cross.detach()))
