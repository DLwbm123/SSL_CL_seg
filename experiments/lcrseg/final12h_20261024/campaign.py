"""Bounded orchestration; production optimizer steps use the shared Trainer."""
import copy
import csv
import json
import math
import os
import random
import signal
import time
import traceback
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from lcrseg.contracts import LabeledBatch, UnlabeledBatch, MethodStepOutput
from lcrseg.models.unet import UNet2D
from lcrseg.engine.trainer import Trainer, TrainerState
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from existing_schedule import Schedule

COUNTS = dict(native=0, reverse=0, importance_reverse=0, optimizer=0)
CFG = {}
STOP = False


def write(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)); t.replace(p)


def save(path, value):
    p = Path(path); t = p.with_suffix('.tmp'); torch.save(value, t); t.replace(p)


def allocated(root):
    return sum(os.stat(Path(d)/n, follow_symlinks=False).st_blocks*512
               for d, _, files in os.walk(root, followlinks=False) for n in files)


def guard():
    if STOP or time.time() >= CFG['deadline_epoch']-60:
        raise TimeoutError('Original 12h deadline; no restart or clock reset')
    if COUNTS['optimizer'] > 80096 or COUNTS['native'] >= 2500000 or COUNTS['reverse'] >= 81000 or COUNTS['importance_reverse'] >= 500:
        raise RuntimeError('Frozen physical call cap')
    if allocated(CFG['root']) >= 45000000000:
        raise RuntimeError('45GB pre-stop below strict 50GB window cap')


class Network(nn.Module):
    def __init__(self, dataset):
        super().__init__(); self.dataset = dataset
        self.net = UNet2D(1 if dataset == 'mnms' else 3, 4 if dataset == 'mnms' else 3)
        self.net.projection_head = nn.Identity()
        if dataset == 'mnms':
            self.net.segmentation_head = nn.Conv2d(16, 4, 3, padding=0, bias=False)

    def forward(self, x):
        COUNTS['native'] += len(x)
        out = self.net(x)
        z = out.logits
        if z.shape[-2:] != x.shape[-2:]:
            z = F.interpolate(z, x.shape[-2:], mode='bilinear', align_corners=True)
        return z


def warm(dataset):
    model = Network(dataset).cuda()
    ck = torch.load(CFG['warm'][dataset], map_location='cpu', weights_only=False)
    if dataset == 'mnms':
        assert (ck['method'], ck['step'], ck['physical_updates']) == ('fixmatch', 43600, 43600)
        state = {}
        for k, v in ck['ema'].items():
            if not k.startswith('parent.native.'): continue
            key = k[len('parent.native.'):]
            if key.startswith('decoder.conv_logit.'):
                if key == 'decoder.conv_logit.mu.weight': state['segmentation_head.weight'] = v
                continue
            if key.startswith('decoder.'): key = key[len('decoder.'):]
            state[key] = v
    else:
        assert ck['step'] == 5000
        state = {k[len('backbone.'):]: v for k, v in ck['teacher'].items() if k.startswith('backbone.')}
    model.net.load_state_dict(state, strict=True)
    return model


def supervised(logits, y, dataset):
    ce = -logits.log_softmax(1).gather(1, y[:, None])[:, 0].mean()
    p = logits.softmax(1)
    if dataset == 'fundus':
        p = torch.stack((p[:, 1]+p[:, 2], p[:, 2]), 1)
        truth = torch.stack((y > 0, y == 2), 1).to(p)
    else:
        p = p[:, 1:]; truth = torch.stack([y == c for c in (1, 2, 3)], 1).to(p)
    support = truth.sum((2, 3)); present = support > 0
    dice = (2*(p*truth).sum((2, 3))+1e-6)/(p.sum((2, 3))+support+1e-6)
    return ce + (((1-dice)*present).sum(1)/present.sum(1).clamp_min(1)).mean()


def moments(x, dataset):
    mask = (x.amax(1, keepdim=True) > 5/255).to(x) if dataset == 'fundus' else torch.ones_like(x[:, :1])
    den = mask.sum((2, 3), keepdim=True).clamp_min(1)
    mean = (x*mask).sum((2, 3), keepdim=True)/den
    sd = (((x-mean).square()*mask).sum((2, 3), keepdim=True)/den).sqrt().clamp_min(1e-4)
    return mean, sd, mask.bool()


def strong(x, dataset, bank):
    mu, sd, mask = moments(x, dataset)
    target_mu, target_sd = mu.clone(), sd.clone()
    if bank:
        for i in range(len(x)):
            if torch.rand((), device=x.device) < .5:
                record = bank[int(torch.randint(len(bank), (), device=x.device))]
                target_mu[i] = torch.tensor(record['mean'], device=x.device)[:, None, None]
                target_sd[i] = torch.tensor(record['sd'], device=x.device)[:, None, None]
    factor = .9+.2*torch.rand_like(sd); offset = -.05+.1*torch.rand_like(mu)
    z = (x-mu)/sd*target_sd*factor + target_mu + target_sd*offset
    z = z.clamp(0, 1) if dataset == 'fundus' else z.clamp(-1, 1)
    return torch.where(mask, z, x) if dataset == 'fundus' else z


def kl(student, teacher, mask):
    q = teacher.detach().softmax(1)
    per = (q*(q.clamp_min(1e-8).log()-student.log_softmax(1))).sum(1)
    return (per*mask.detach()).mean()


class Method(nn.Module):
    def __init__(self, model, dataset, arm, old=None, importance=None, bank=None):
        super().__init__(); self.model = model; self.dataset = dataset; self.arm = arm
        self.teacher = copy.deepcopy(model).eval().requires_grad_(False)
        self.old_model = old.eval().requires_grad_(False) if old is not None else None
        self.importance = importance or {}; self.bank = bank or []
        self.old_weights = {k: v.detach().clone() for k, v in old.named_parameters()} if old is not None else {}
        self.versions = [p._version for p in old.parameters()] if old is not None else []
        self.anchor_den = sum(float(v.sum()) for v in self.importance.values()) or 1.
        self.last = {}

    def assert_old_state_unchanged(self):
        if self.old_model is not None:
            assert self.versions == [p._version for p in self.old_model.parameters()]
            assert all(p.grad is None and not p.requires_grad for p in self.old_model.parameters())

    def training_step(self, labeled, unlabeled, global_step, site_step):
        x, y, u = labeled.image, labeled.label, unlabeled.weak_image
        self.teacher.eval()
        history = self.bank if self.arm == 'C' and self.old_model is not None else []
        z = self.model(x); zs = self.model(strong(x, self.dataset, history))
        sup = .5*(supervised(z, y, self.dataset)+supervised(zs, y, self.dataset))
        with torch.no_grad(): q = self.teacher(u)
        zu = self.model(strong(u, self.dataset, history))
        conf = q.softmax(1).max(1).values >= .7
        con = kl(zu, q, conf)
        keep = z.sum()*0; anchor = z.sum()*0
        if self.arm == 'C' and self.old_model is not None:
            with torch.no_grad():
                old_l = self.old_model(x); old_u = self.old_model(u)
                pl = old_l.softmax(1); pu = old_u.softmax(1)
                ml = (pl.max(1).values >= .7) & (pl.argmax(1) == y)
                mu = pu.max(1).values >= .7
            # Reference and student see exactly the same current weak U.
            weak_u = self.model(u)
            keep = .5*(kl(z, old_l, ml)+kl(weak_u, old_u, mu))
            anchor = sum((self.importance[k]*(p-self.old_weights[k]).square()).sum()
                         for k, p in self.model.named_parameters())/self.anchor_den
        total = sup + min((site_step+1)/1000, 1.)*con + .5*keep + 1000*anchor
        losses = dict(supervised=sup, consistency=con, history=keep, anchor=anchor)
        return MethodStepOutput(total, losses, dict(accepted=float(conf.float().mean())))

    @torch.no_grad()
    def after_optimizer_step(self):
        for q, p in zip(self.teacher.parameters(), self.model.parameters(), strict=True): q.mul_(.999).add_(p, alpha=.001)
        for q, p in zip(self.teacher.buffers(), self.model.buffers(), strict=True): q.copy_(p)


def batches(l, u, step, dataset):
    x, y, _ = l.batch(step); ux, _, _ = u.batch(step)
    n, m = len(x), len(ux)
    lb = LabeledBatch(x, y, torch.ones_like(y[:, None], dtype=torch.bool), ['private']*n, ['private']*n, [l.ds.records[0]['site_or_vendor']]*n, [None]*n)
    ub = UnlabeledBatch(ux, ux, torch.ones_like(ux[:, :1], dtype=torch.bool), ['private']*m, ['private']*m, [u.ds.records[0]['site_or_vendor']]*m, [None]*m, [{}]*m)
    assert not hasattr(ub, 'label')
    return lb, ub


def trainer(method, lr, steps, bucket):
    opt = torch.optim.Adam(method.model.parameters(), lr=lr, weight_decay=1e-5)
    def pre(*args):
        if COUNTS['optimizer'] >= 80096: raise RuntimeError('Frozen optimizer cap before an additional update')
    opt.register_step_pre_hook(pre)
    def post(*args):
        COUNTS['optimizer'] += 1
        with (Path(CFG['root'])/'physical_optimizer.jsonl').open('a') as f:
            f.write(json.dumps(dict(call=COUNTS['optimizer'], bucket=bucket))+'\n')
    opt.register_step_post_hook(post)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: .1+.9*.5*(1+math.cos(math.pi*min(k, steps)/steps)))
    return Trainer(method, optimizer=opt, scheduler=sch, device='cuda', amp=False, grad_clip_norm=5.)


def snapshot(method, tr, step, bank, importance):
    return dict(model=method.model.state_dict(), teacher=method.teacher.state_dict(), trainer=tr.state_dict(),
                rng=capture_rng_state(), step=step, bank=bank, importance=importance, counts=dict(COUNTS))


def qualify(method, l, u, out):
    start = copy.deepcopy(method.model.state_dict()); teacher = copy.deepcopy(method.teacher.state_dict())
    rng = capture_rng_state(); tr = trainer(method, 1e-4, 8, out.name+':qualification')
    lb, ub = batches(l, u, 900000, method.dataset)
    saved = copy.deepcopy(tr.state_dict()); saved_rng = capture_rng_state()
    def one():
        COUNTS['reverse'] += 1
        return tr.train_step(lb, ub, state=TrainerState()).total_loss.detach().clone()
    a = one(); first = copy.deepcopy(method.model.state_dict()); first_ema = copy.deepcopy(method.teacher.state_dict())
    method.model.load_state_dict(start); method.teacher.load_state_dict(teacher); tr.load_state_dict(saved); restore_rng_state(saved_rng)
    b = one()
    assert torch.equal(a, b)
    assert all(torch.equal(first[k], v) for k, v in method.model.state_dict().items())
    assert all(torch.equal(first_ema[k], v) for k, v in method.teacher.state_dict().items())
    before = float(supervised(method.model(lb.image), lb.label, method.dataset).detach())
    for j in range(6):
        COUNTS['reverse'] += 1
        tr.train_step(lb, ub, state=TrainerState(site_step=j+2, global_step=j+2))
    after = float(supervised(method.model(lb.image), lb.label, method.dataset).detach())
    assert math.isfinite(after) and after < before, (before, after)
    method.assert_old_state_unchanged()
    method.model.load_state_dict(start); method.teacher.load_state_dict(teacher); method.model.zero_grad(set_to_none=True)
    restore_rng_state(rng)
    write(out/'QUALIFICATION.json', dict(status='PASS', physical_optimizer=8, golden_exact=True, restored=True, before=before, after=after))


def patient_scores(rows):
    groups = defaultdict(list)
    for row in rows: groups[row['key']].append(row['dice'])
    return {k: float(np.mean(v)) for k, v in groups.items()}


def contrast(a, b):
    assert a.keys() == b.keys()
    delta = np.array([b[k]-a[k] for k in sorted(a)])*100
    rng = np.random.default_rng(163)
    means = np.mean(delta[rng.integers(len(delta), size=(10000, len(delta)))], 1)
    return dict(mean_pp=float(delta.mean()), CI95_pp=np.quantile(means, [.025, .975]).tolist(),
                over1=int((delta < -1).sum()), over2=int((delta < -2).sum()), p10_pp=float(np.quantile(delta, .1)), n=len(delta))


@torch.no_grad()
def evaluate(model, dataset, sites, out):
    model.eval(); root = Path(CFG['data']); allrows = []; summary = {}
    with (root/'manifests/training/lcrseg_v1_seed0.csv').open() as f: manifest = list(csv.DictReader(f))
    for site in sites:
        rows = []
        for r in manifest:
            if r['dataset'] != dataset or r['site_or_vendor'] != site or r['primary_20pct_split'] != 'val': continue
            guard()
            with h5py.File(root/'h5/v1'/r['image_h5_relpath']) as f: image = f['image'][...]
            with h5py.File(root/'h5/v1'/r['label_h5_relpath']) as f: label = f['label'][...]
            if dataset == 'fundus': image = image[None].astype(np.float32)/255; label = label[None]
            else: image = image[:, None].astype(np.float32)
            predictions = []
            for j in range(0, len(image), 4): predictions.append(model(torch.from_numpy(image[j:j+4]).cuda()).argmax(1).cpu().numpy())
            pred = np.concatenate(predictions)
            assert pred.shape == label.shape
            masks = [(pred == c, label == c) for c in (1, 2, 3)] if dataset == 'mnms' else [(pred == 2, label == 2), (pred > 0, label > 0)]
            dice = [2*int((a & b).sum())/(int(a.sum())+int(b.sum())) if a.any() or b.any() else 1. for a, b in masks]
            rows.append(dict(site=site, key=r['patient_id'], phase=r.get('phase', ''), dice=dice))
        assert rows
        scores = patient_scores(rows)
        summary[site] = dict(Dice_percent=100*float(np.mean(list(scores.values()))), classes_percent=(100*np.mean([r['dice'] for r in rows], 0)).tolist(),
                             n=len(scores), p10_Dice_percent=100*float(np.quantile(list(scores.values()), .1)))
        allrows.extend(rows)
    write(out/'METRICS.json', summary); write(out/'PRIVATE_ROWS.json', allrows)
    return summary, allrows


def summarize_current(method, l, u, old_importance):
    measure = copy.deepcopy(method.teacher).eval().requires_grad_(True)
    params = dict(measure.named_parameters()); importance = {k: torch.zeros_like(p) for k, p in params.items()}
    ms = []; ss = []
    for j in range(32):
        guard(); lb, ub = batches(l, u, 800000+j, method.dataset)
        x = torch.cat((lb.image, ub.weak_image))
        mu, sd, _ = moments(x, method.dataset); ms.append(mu.mean(0).flatten()); ss.append(sd.mean(0).flatten())
        p = measure(x).softmax(1); COUNTS['importance_reverse'] += 1
        grads = torch.autograd.grad(p.square().mean(), list(params.values()), allow_unused=False)
        for (k, _), g in zip(params.items(), grads, strict=True): importance[k].add_(g.detach().abs()/32)
    scale = sum(float(v.sum()) for v in importance.values())/sum(v.numel() for v in importance.values())
    assert math.isfinite(scale) and scale > 0
    importance = {k: v/scale for k, v in importance.items()}
    if old_importance: importance = {k: (v+old_importance[k].to(v))*.5 for k, v in importance.items()}
    del measure
    return importance, dict(mean=torch.stack(ms).mean(0).cpu().tolist(), sd=torch.stack(ss).mean(0).cpu().tolist())


def run_stage(dataset, arm, index, previous=None):
    site = CFG['sites'][dataset][index]; name = f'{dataset}_{index}_{arm}'
    out = Path(CFG['root'])/name; out.mkdir(exist_ok=False)
    guard(); torch.manual_seed(20272401+index); np.random.seed(20272401+index); random.seed(20272401+index)
    model = warm(dataset); bank = []; importance = {}; old = None
    if previous:
        ck = torch.load(previous/'final.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ck['teacher']); old = copy.deepcopy(model).eval().requires_grad_(False)
        bank = ck['bank']; importance = {k: v.cuda() for k, v in ck['importance'].items()}
    method = Method(model, dataset, arm, old, importance, bank)
    l = Schedule(CFG['data'], site, seed=20272401, dataset=dataset, image_size=384, batch_size=2)
    u = Schedule(CFG['data'], site, True, seed=20272401, dataset=dataset, image_size=384, batch_size=4)
    source_rows = None
    if index == 0:
        base = out/'warm_reference'; base.mkdir(); _, source_rows = evaluate(method.teacher, dataset, [site], base)
    qualify(method, l, u, out)
    steps = (12000 if dataset == 'mnms' else 8000) if index == 0 else 6000
    tr = trainer(method, 5e-4 if index == 0 else 1e-4, steps, name+':formal')
    save(out/'initial.pt', snapshot(method, tr, 0, bank, importance)); completed = 0
    try:
        for k in range(steps):
            if k % 50 == 0: guard()
            lb, ub = batches(l, u, k+index*100000, dataset)
            COUNTS['reverse'] += 1
            result = tr.train_step(lb, ub, state=TrainerState(global_step=k, site_step=k))
            assert not result.scalars['optimizer_step_skipped']
            completed = k+1
            if completed % 50 == 0 or completed == 1:
                info = dict(status='TRAINING', task=name, step=completed, total_steps=steps, counts=dict(COUNTS),
                            loss=float(result.total_loss.detach()), components={k: float(v.detach()) for k, v in result.losses.items()},
                            allocated_bytes=allocated(CFG['root']), time_epoch=time.time(), **result.scalars)
                write(out/'STATUS.json', info); write(Path(CFG['root'])/'STATUS.json', info)
                with (out/'LOG.jsonl').open('a') as f: f.write(json.dumps(info)+'\n')
            if completed % 1000 == 0: save(out/'latest.pt', snapshot(method, tr, completed, bank, importance))
        metrics, rows = evaluate(method.teacher, dataset, CFG['sites'][dataset][:index+1], out)
        if index == 0:
            c = contrast(patient_scores(source_rows), patient_scores(rows))
            threshold = .8006740663770262 if dataset == 'mnms' else .8929755965874404
            passed = metrics[site]['Dice_percent']/100 >= threshold and c['CI95_pp'][0] >= -.3 and c['over2'] == 0 and c['p10_pp'] >= -.5
            write(out/'SOURCE_GATE.json', dict(status='PASS' if passed else 'SOURCE_STOP', contrast=c, min_Dice_percent=threshold*100))
        else: passed = True
        importance, current = summarize_current(method, l, u, importance if arm == 'C' else {})
        bank = bank+[current]
        save(out/'final.pt', snapshot(method, tr, completed, bank, importance))
        method.assert_old_state_unchanged()
        write(out/'EXIT.json', dict(exit_code=0, source_pass=passed, step=completed, counts=dict(COUNTS)))
        return out, passed
    except BaseException:
        save(out/'failure_partial.pt', snapshot(method, tr, completed, bank, importance))
        write(out/'FAILURE.json', dict(step=completed, counts=dict(COUNTS), traceback=traceback.format_exc()))
        write(out/'EXIT.json', dict(exit_code=1)); raise


def audit_data():
    report = {}; seen = {}
    with (Path(CFG['data'])/'manifests/training/lcrseg_v1_seed0.csv').open() as f: rows = list(csv.DictReader(f))
    for dataset, sites in CFG['sites'].items():
        report[dataset] = {}
        for site in sites:
            counts = defaultdict(set)
            for r in rows:
                if r['dataset'] != dataset or r['site_or_vendor'] != site: continue
                role = r['primary_20pct_split']; assert role in ('train_labeled', 'train_unlabeled', 'val')
                if dataset == 'mnms': assert r['cohort'] == 'canonical320'
                key = (dataset, r['patient_id']); assert key not in seen or seen[key] == (site, role)
                seen[key] = (site, role); counts[role].add(r['patient_id'])
                if role == 'train_unlabeled': assert not r['label_h5_relpath'] and not r['label_sha256']
                assert (Path(CFG['data'])/'h5/v1'/r['image_h5_relpath']).is_file()
                if role != 'train_unlabeled': assert (Path(CFG['data'])/'h5/v1'/r['label_h5_relpath']).is_file()
            report[dataset][site] = {k: len(v) for k, v in counts.items()}
    write(Path(CFG['root'])/'DATA_AUDIT.json', dict(status='PASS', domains=report, test_access=0, U_label_access=0, fundus_missing_second_camera=True))


def selfcheck():
    initial_native = COUNTS['native']
    x = torch.rand(2, 1, 32, 32, device='cuda')*2-1; y = torch.randint(4, (2, 32, 32), device='cuda')
    n = Network('mnms').cuda(); z = n(x); loss = supervised(z, y, 'mnms'); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in n.parameters())
    identity = strong(torch.ones_like(x)*.2, 'mnms', [])
    assert torch.isfinite(identity).all() and identity.shape == x.shape
    n = Network('fundus').cuda(); z = n(torch.rand(2, 3, 32, 32, device='cuda')); supervised(z, torch.randint(3, (2, 32, 32), device='cuda'), 'fundus').backward()
    for dataset in ('mnms', 'fundus'):
        model = Network(dataset).cuda(); old = copy.deepcopy(model).eval().requires_grad_(False)
        imp = {k: torch.ones_like(p) for k, p in model.named_parameters()}
        channels = 1 if dataset == 'mnms' else 3
        method = Method(model, dataset, 'C', old, imp, [dict(mean=[.2]*channels, sd=[.1]*channels)])
        xx = torch.rand(2, channels, 32, 32, device='cuda'); yy = torch.randint(4 if dataset == 'mnms' else 3, (2, 32, 32), device='cuda')
        lb = LabeledBatch(xx, yy, torch.ones_like(yy[:, None], dtype=torch.bool), [], [], [], [])
        ub = UnlabeledBatch(xx, xx, torch.ones_like(xx[:, :1], dtype=torch.bool), [], [], [], [], [])
        result = method.training_step(lb, ub, 0, 1000); result.total_loss.backward()
        assert torch.isfinite(result.total_loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        method.assert_old_state_unchanged()
    a = {'a': .8, 'b': .9}; assert contrast(a, a)['over2'] == 0
    write(Path(CFG['root'])/'SYNTHETIC.json', dict(status='PASS', samples=COUNTS['native']-initial_native, reverse=4, optimizer=0, native_geometry=True, cup_disc_union=True, complete_target_backward=True, reference_unchanged=True))


def main(cfg):
    global CFG, STOP
    CFG = cfg
    def stop(*args):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    root = Path(cfg['root']); root.mkdir(exist_ok=True)
    audit_data(); selfcheck(); failed = {}; current = {}; source_ok = {}
    for dataset in ('mnms', 'fundus'):
        try:
            path, ok = run_stage(dataset, 'SOURCE', 0); source_ok[dataset] = ok
            current[(dataset, 'B')] = current[(dataset, 'C')] = path
        except BaseException:
            source_ok[dataset] = False; failed[dataset+'_source'] = traceback.format_exc()
            write(root/'INCIDENTS.json', failed)
    for depth in (1, 2, 3):
        for dataset in ('mnms', 'fundus'):
            if not source_ok.get(dataset) or depth >= len(cfg['sites'][dataset]): continue
            for arm in ('B', 'C'):
                if dataset+'_'+arm in failed: continue
                try:
                    path, _ = run_stage(dataset, arm, depth, current[(dataset, arm)])
                    current[(dataset, arm)] = path
                except BaseException:
                    failed[dataset+'_'+arm] = traceback.format_exc(); write(root/'INCIDENTS.json', failed)
    write(root/'TERMINAL.json', dict(status='FINISHED_WITH_STOPS' if failed or not all(source_ok.values()) else 'COMPLETE',
                                   source_ok=source_ok, failed=failed, counts=dict(COUNTS), end_epoch=time.time(),
                                   elapsed_seconds=time.time()-cfg['started_epoch'], allocated_bytes=allocated(root)))
    write(root/'STATUS.json', json.loads((root/'TERMINAL.json').read_text()))


if __name__ == '__main__': main(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
