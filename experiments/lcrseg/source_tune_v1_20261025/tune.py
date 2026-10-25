"""Four frozen source configurations; all optimizer updates use the existing Trainer."""
import copy
import json
import math
import random
import shutil
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from lcrseg.contracts import MethodStepOutput, LabeledBatch, UnlabeledBatch
import campaign as c


def geometry(x, y=None, *, identity=False):
    n = len(x)
    theta = torch.zeros(n, 2, 3, device=x.device)
    angle = torch.zeros(n, device=x.device) if identity else (torch.rand(n, device=x.device)*2-1)*math.pi/12
    scale = torch.ones(n, device=x.device) if identity else .9+.2*torch.rand(n, device=x.device)
    theta[:, 0, 0] = angle.cos()/scale; theta[:, 0, 1] = -angle.sin()/scale
    theta[:, 1, 0] = angle.sin()/scale; theta[:, 1, 1] = angle.cos()/scale
    if not identity: theta[:, :, 2] = (torch.rand(n, 2, device=x.device)*2-1)*.05
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    z = F.grid_sample(x+1, grid, mode='bilinear', padding_mode='zeros', align_corners=False)-1
    target = None if y is None else F.grid_sample(y[:, None].float(), grid, mode='nearest', padding_mode='zeros', align_corners=False)[:, 0].long()
    return z, target


class Method(c.Method):
    def training_step(self, labeled, unlabeled, global_step, site_step):
        x, y, u = labeled.image, labeled.label, unlabeled.weak_image
        if c.CFG['arm']['geometry']:
            x, y = geometry(x, y); u, _ = geometry(u)
        self.teacher.eval()
        z = self.model(x); zs = self.model(c.strong(x, 'mnms', []))
        sup = .5*(c.supervised(z, y, 'mnms')+c.supervised(zs, y, 'mnms'))
        with torch.no_grad(): q = self.teacher(u)
        zu = self.model(c.strong(u, 'mnms', []))
        mask = q.softmax(1).max(1).values >= .9
        con = c.kl(zu, q, mask)
        total = sup+c.CFG['arm']['u_weight']*min((site_step+1)/1000, 1.)*con
        return MethodStepOutput(total, dict(supervised=sup, consistency=con), dict(accepted=float(mask.float().mean())))


base_trainer = c.trainer
def trainer(method, lr, steps, bucket):
    tr = base_trainer(method, lr, steps, bucket)
    for group in tr.optimizer.param_groups: group['weight_decay'] = 1e-4
    return tr


@torch.no_grad()
def diagnostic(method, l, u, out, step):
    rng = c.capture_rng_state(); rows = []
    try:
        torch.manual_seed(20272599); torch.cuda.manual_seed_all(20272599)
        for j in range(4):
            lb, ub = c.batches(l, u, 770001+j, 'mnms')
            result = method.training_step(lb, ub, 0, step-1)
            rows.append(float(result.total_loss))
        p = out/'TRAINING_DIAGNOSTIC.json'
        saved = json.loads(p.read_text()) if p.exists() else []
        saved.append(dict(step=step, fixed_four_batch_objectives=rows, mean=float(np.mean(rows)), not_convergence_certificate=True))
        c.write(p, saved)
    finally: c.restore_rng_state(rng)


def phase_summary(reference, rows):
    result = {}
    for phase in ('ED', 'ES', 'ED_ES'):
        a = reference if phase == 'ED_ES' else [r for r in reference if r['phase'] == phase]
        b = rows if phase == 'ED_ES' else [r for r in rows if r['phase'] == phase]
        x, y = c.patient_scores(a), c.patient_scores(b)
        result[phase] = dict(Dice_percent=100*float(np.mean(list(y.values()))),
                            reference_Dice_percent=100*float(np.mean(list(x.values()))),
                            classes_percent=(100*np.mean([r['dice'] for r in b], axis=0)).tolist(),
                            contrast=c.contrast(x, y))
    return result


def selfcheck():
    start = c.COUNTS['native']; rng = c.capture_rng_state()
    x = torch.rand(2, 1, 64, 64, device='cuda')*2-1
    y = torch.randint(4, (2, 64, 64), device='cuda')
    a, b = geometry(x, y, identity=True)
    assert torch.allclose(a, x, atol=2e-5) and torch.equal(b, y)
    for geo in (False, True):
        c.CFG['arm'] = dict(geometry=geo, u_weight=1., lr=1e-5)
        model = c.Network('mnms').cuda(); method = Method(model, 'mnms', 'SOURCE')
        lb = LabeledBatch(x, y, torch.ones_like(y[:, None], dtype=torch.bool), [], [], [], [])
        ub = UnlabeledBatch(x, x, torch.ones_like(x, dtype=torch.bool), [], [], [], [], [])
        result = method.training_step(lb, ub, 0, 1000); result.total_loss.backward()
        assert torch.isfinite(result.total_loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(p.grad is None for p in method.teacher.parameters())
    c.restore_rng_state(rng)
    c.write(Path(c.CFG['campaign_root'])/'SYNTHETIC.json', dict(status='PASS', native=c.COUNTS['native']-start, reverse=2, optimizer=0, affine_identity=True, complete_objective_backward=True))


def main(cfg):
    c.CFG = cfg
    signal.signal(signal.SIGTERM, lambda *args: setattr(c, 'STOP', True))
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    c.Method = Method; c.trainer = trainer; c.record_training_diagnostic = diagnostic
    c.summarize_current = lambda *args: ({}, {})
    root = Path(cfg['campaign_root']); c.audit_data(); selfcheck()
    # Verify checkpoint identity without another medical forward or evaluation.
    baseline = torch.load(Path(cfg['source_reference'])/'mnms_0_SOURCE/final.pt', map_location='cpu', weights_only=False)
    assert baseline['step'] == 0
    model = c.warm('mnms')
    assert all(torch.equal(v.cpu(), baseline['teacher'][k]) for k, v in model.state_dict().items())
    del model, baseline
    result = {}; failed = {}
    for arm in cfg['arms']:
        if c.STOP or time.time() >= cfg['deadline_epoch']-120: break
        folder = root/arm['name']; folder.mkdir(exist_ok=False)
        cfg['root'] = str(folder); cfg['arm'] = arm
        try:
            path, passed = c.run_stage('mnms', 'SOURCE', 0)
            ref = json.loads((path/'warm_reference/PRIVATE_ROWS.json').read_text())
            rows = json.loads((path/'PRIVATE_ROWS.json').read_text())
            result[arm['name']] = dict(config=arm, phase=phase_summary(ref, rows), source_capability_pass=passed)
            c.write(path/'PHASE_SUMMARY.json', result[arm['name']])
        except BaseException:
            failed[arm['name']] = traceback.format_exc(); c.write(root/'INCIDENTS.private.json', failed)
        c.write(root/'STATUS.json', dict(status='RUNNING', completed=list(result), failed=list(failed), counts=dict(c.COUNTS), time_epoch=time.time()))
    eligible = [(n, r) for n, r in result.items() if r['source_capability_pass']]
    eligible.sort(key=lambda t: (-t[1]['phase']['ED_ES']['Dice_percent'], t[1]['phase']['ED_ES']['contrast']['over1'], t[0]))
    selected = eligible[0][0] if eligible else None
    terminal = dict(status='COMPLETE' if len(result)==len(cfg['arms']) else 'INCOMPLETE', results=result, failed=list(failed),
                    selected_development_candidate=selected, adaptation_started=False, independent_test=False,
                    counts=dict(c.COUNTS), synthetic_reverse=2, end_epoch=time.time(), elapsed_seconds=time.time()-cfg['started_epoch'],
                    allocated_bytes=c.allocated(root))
    c.write(root/'TERMINAL.json', terminal); c.write(root/'STATUS.json', terminal)

