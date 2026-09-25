"""Fixed R1 math, state, and durable accounting; no historical launcher imports."""
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from qprompt.models import UNetBody, UNetDense, UNetQuery, DINOv2Adapter, load_pinned_dinov2
from qprompt.losses import (PrototypeBank, supervised_query_loss, image_prototypes,
                           image_alignment_loss, grqa_loss)
from qprompt.state import make_reference, ema_update

CONFIG = json.loads(Path(__file__).with_name('EXECUTION_AUTHORIZATION.json').read_text())
SOURCE_COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
WEIGHT_SHA = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'


def seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps([261, *parts]).encode()).digest()[:8], 'big') % (2**63 - 1)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic(path, value, *, binary=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.write-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            if binary:
                torch.save(value, stream)
            else:
                stream.write((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
            stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def append(path, item):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as f:
        f.write(json.dumps(dict(timestamp=time.time(), **item), allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def events(path):
    if not Path(path).exists(): return []
    lines = Path(path).read_text().splitlines()
    output = []
    for i, line in enumerate(lines):
        try: output.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1: raise
    return output


def build(backbone, dense, assets, source):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed(backbone, 'dense-head' if dense else 'query-head'))
        if backbone == 'UNET_QUERY_128':
            model = UNetDense() if dense else UNetQuery()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed(backbone, 'body'))
                model.body.load_state_dict(UNetBody().state_dict())
            body = model.body
        else:
            body = load_pinned_dinov2(source / 'third_party', assets / 'dinov2_vits14_pretrain.pth',
                                     commit=SOURCE_COMMIT, weight_sha256=WEIGHT_SHA)
            model = DINOv2Adapter(body, query=not dense)
        digest = tensor_sha(body.state_dict())
    return model, digest


def optimizer_for(model, backbone):
    if backbone == 'UNET_QUERY_128':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        body = list(model.backbone.parameters()); ids = {id(p) for p in body}
        optimizer = torch.optim.AdamW([dict(params=body, lr=1e-5),
            dict(params=[p for p in model.parameters() if id(p) not in ids], lr=1e-4)], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: max(0., 1 - n / 3000) ** .9)
    return optimizer, scheduler


def dense_loss(logits, label):
    # Port of the frozen canonical supervised objective at f6b9697, objectives.py:106.
    valid = label != 255
    pixels = -logits.log_softmax(1).gather(1, label.masked_fill(~valid, 0)[:, None]).squeeze(1)
    return (pixels * valid).sum() / valid.sum().clamp_min(1)


def direct_alignment(queries, reference_queries, bank):
    if not bank.supported.any(): return queries.sum() * 0
    similarities = F.normalize(queries.float(), dim=-1) @ bank.vectors.detach().float().T
    similarities = similarities.masked_fill(~bank.supported[None, None], -torch.inf)
    selected = similarities.detach().argmax(-1, keepdim=True)
    return -similarities.gather(-1, selected).mean()


def losses(model, image, label, arm, bank=None, reference=None):
    output = model(image)
    if arm == 'D0':
        seg = dense_loss(output.float(), label)
        return seg, dict(Lseg=seg), None, None, None
    # Matching and probabilities use FP32 even when the backbone uses BF16.
    with torch.autocast(image.device.type, enabled=False):
        seg = supervised_query_loss(output['class_logits'].float(), output['mask_logits'].float(), label)['total']
        terms = dict(Lseg=seg); total = seg; current = support = ref_q = None
        if arm in ('Q1', 'Q2', 'Q3', 'Q4'):
            current, support = image_prototypes(output['pixels'].float(), label,
                                                vit_pad=isinstance(model, DINOv2Adapter))
            terms['Limg'] = image_alignment_loss(current, support, bank)
            total = total + 10 * terms['Limg']
    if arm in ('Q2', 'Q3', 'Q4'):
        with torch.no_grad(): ref_q = reference(image)['queries'].float()
        with torch.autocast(image.device.type, enabled=False):
            gr = grqa_loss(output['queries'].float(), ref_q, bank, beta=0 if arm == 'Q2' else .001)
            terms.update(group=gr['group'], k3=gr['paper_selected_k3_surrogate'], categorical_kl=gr['categorical_kl'])
            if arm == 'Q4':
                terms['direct'] = direct_alignment(output['queries'].float(), ref_q, bank)
                terms['regularizer'] = 5 * terms['direct'] + .005 * gr['paper_selected_k3_surrogate']
            else:
                terms['regularizer'] = 5 * gr['total']
            total = total + terms['regularizer']
    return total, terms, current, support, (output['queries'].float(), ref_q)


@torch.no_grad()
def diagnostics(query, reference, bank):
    if reference is None or not bank.supported.any(): return {}
    s = F.normalize(query, dim=-1) @ bank.vectors.T
    r = F.normalize(reference, dim=-1) @ bank.vectors.T
    s = s.masked_fill(~bank.supported[None, None], -torch.inf)
    r = r.masked_fill(~bank.supported[None, None], -torch.inf)
    selected = s.argmax(-1); reward = s.gather(-1, selected[..., None]).squeeze(-1)
    groups = []; advantages = torch.zeros_like(reward)
    for b in range(query.shape[0]):
        for c in range(3):
            mask = selected[b] == c; n = int(mask.sum())
            if n:
                groups.append(n)
                if n > 1:
                    v = reward[b, mask]; advantages[b, mask] = (v-v.mean())/(v.std(unbiased=False)+1e-6)
    ratio = (s.log_softmax(-1)-r.log_softmax(-1)).gather(-1, selected[..., None]).exp()
    return dict(query_utilization=[int((selected == c).sum()) for c in range(3)],
        group_sizes=groups, singleton_fraction=sum(n == 1 for n in groups)/max(len(groups),1),
        reward_mean=float(reward.mean()), reward_std=float(reward.std(unbiased=False)),
        advantage_std=float(advantages.std(unbiased=False)), ratio_mean=float(ratio.mean()),
        ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
        clip_fraction=float(((ratio<.9)|(ratio>1.1)).float().mean()))


def snapshot(model, optimizer, scheduler, reference, bank, step, **metadata):
    return dict(student=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
        reference=None if reference is None else reference.state_dict(), bank=None if bank is None else bank.state_dict(),
        python_rng=random.getstate(), numpy_rng=np.random.get_state(), cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(), global_step=step, data_cursor=step,
        precision=CONFIG['precision'], **metadata)


def restore(state, model, optimizer, scheduler, reference=None, bank=None):
    model.load_state_dict(state['student']); optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    for name, module in [('reference', reference), ('bank', bank)]:
        if state[name] is not None and module is not None: module.load_state_dict(state[name])
    random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['cpu_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['global_step']


def grad_norm(loss, model):
    grads = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True, allow_unused=True)
    return float(sum(g.detach().float().square().sum() for g in grads if g is not None).sqrt())
