"""One frozen stage/arm; the external controller owns its time/storage budget."""
import copy
import json
import os
import time
import traceback
import subprocess
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from lcrseg.data.h5_dataset import load_training_records
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from sc_bayeseg.window_v1.r_start import write, append, save
from sc_bayeseg.window_v1.patient_validation import phase_metrics
from .data import Schedule, view
from .model import initialize, supervised
from .prostate_metrics import binary_phase_metrics


def setup():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(163); np.random.seed(163)


def optimizer(model):
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4, betas=(.9, .999), eps=1e-8, weight_decay=0, foreach=False)


@torch.no_grad()
def ema(teacher, model):
    for a, b in zip(teacher.parameters(), model.parameters()): a.mul_(.99).add_(b, alpha=.01)
    for a, b in zip(teacher.buffers(), model.buffers()):
        if a.is_floating_point(): a.mul_(.99).add_(b, alpha=.01)
        else: a.copy_(b)


def step_loss(model, teacher, x, y, u, current, history, step, seed=163, history_l=True, history_u=True, style_mode="history", fixed_jl=None, ju_weight=0.005, consistency_weight=1.0, history_strength=1.0):
    strong = view(x, current, history if history_l else None, step, seed, style_mode if history_l else "history", history_strength)
    l1, i1 = model(x); l2, i2 = model(strong)
    sup = (supervised(l1, y) + supervised(l2, y)) / 2
    img = (i1 + i2) / 2; labeled_img = img; unlabeled_img = sup * 0; consistency = sup * 0
    if u is not None:
        with torch.no_grad(): target = teacher.probability(u)
        lu, iu = model(view(u, current, history if history_u else None, 500000 + step, seed, style_mode if history_u else "history", history_strength))
        p = torch.stack([v.softmax(1) for v in lu]).mean(0).clamp_min(1e-8)
        consistency = (target * (target.clamp_min(1e-8).log() - p.log())).sum(1).mean()
        unlabeled_img = iu
        img = (img + iu) / 2
    loss = sup + .01 * img + min(step / 1000, 1) * consistency
    if fixed_jl is not None:
        assert u is not None and fixed_jl >= 0 and ju_weight >= 0 and consistency_weight >= 0
        loss = sup + fixed_jl * labeled_img + ju_weight * unlabeled_img + consistency_weight * min(step / 1000, 1) * consistency
    return loss, dict(labeled_image_objective=float(labeled_img.detach()), unlabeled_image_objective=float(unlabeled_img.detach()), supervised=float(sup.detach()), image_objective=float(img.detach()), soft_KL=float(consistency.detach()))


def aggregate(rows, num_classes=4, expected_phases=2):
    groups = defaultdict(list)
    for r in rows: groups[r['patient']].append(r['metrics'])
    patients = []
    for patient, phases in sorted(groups.items()):
        assert len(phases) == expected_phases
        class_values = []
        for k in range(num_classes - 1):
            valid = [p['classes'][k] for p in phases if not p['classes'][k]['GT_absent']]
            class_values.append(float(np.mean([c['Dice'] for c in valid])) if valid else None)
        phase_dice = [np.mean([c['Dice'] for c in p['classes'] if not c['GT_absent']]) for p in phases]
        patients.append(dict(patient=patient, Dice=float(np.mean(phase_dice)), class_Dice=class_values,
                             NLL=float(np.mean([p['NLL'] for p in phases])), Brier=float(np.mean([p['Brier'] for p in phases]))))
    cells = [c for r in rows for c in r['metrics']['classes']]
    return dict(mean_Dice=float(np.mean([p['Dice'] for p in patients])), patients=patients,
                class_Dice=[float(np.mean([p['class_Dice'][k] for p in patients if p['class_Dice'][k] is not None])) for k in range(num_classes - 1)],
                NLL=float(np.mean([p['NLL'] for p in patients])), Brier=float(np.mean([p['Brier'] for p in patients])),
                HD95_penalized_mm=float(np.mean([c['HD95_penalized_mm'] for c in cells if not c['GT_absent']])),
                empty_prediction_failures=[sum(c['empty_prediction_failure'] for c in cells if c['class_index'] == k) for k in range(1, num_classes)])


@torch.no_grad()
def evaluate(model, s, out):
    previous = model.training; model.eval(); results = {}; private = {}
    rng = capture_rng_state()
    try:
        for domain in s.get('domains', ('Siemens', 'Philips')):
            records = load_training_records(Path(s['data']), seed=0, dataset=s.get('dataset', 'mnms'), sites=[domain], roles=['val'])
            rows = []
            for i, row in enumerate(sorted(records, key=lambda r: (r['patient_id'], r['case_id']))):
                assert time.time() - s['original_started_unix'] < 85200
                with h5py.File(Path(s['data']) / 'h5/v1' / row['image_h5_relpath'], 'r') as f:
                    volume = f['image'][...]; spacing = json.loads(f.attrs['processed_spacing'])
                predictions = []
                for j in range(0, len(volume), 2):
                    torch.manual_seed(s.get('evaluation_seed', 620163) + 1000 * i + j)
                    x = torch.from_numpy(np.ascontiguousarray(volume[j:j+2, None])).float().cuda()
                    small = F.interpolate(x, (128, 128), mode='bilinear', align_corners=False)
                    p = model.probability(small)
                    p = F.interpolate(p, x.shape[-2:], mode='bilinear', align_corners=False); p = p / p.sum(1, keepdim=True)
                    assert torch.isfinite(p).all()
                    predictions.append(p.cpu())
                # Complete this phase's image-only prediction before opening its GT.
                with h5py.File(Path(s['data']) / 'h5/v1' / row['label_h5_relpath'], 'r') as f: y = f['label'][...]
                rows.append(dict(patient=row['patient_id'], case=row['case_id'], metrics=(phase_metrics if s.get('num_classes', 4) == 4 else binary_phase_metrics)(torch.cat(predictions), y, spacing)))
                if i % 10 == 0: write(out / 'STATUS.json', dict(status='EVALUATING', domain=domain, phase=i))
            private[domain] = rows; results[domain] = aggregate(rows, s.get('num_classes', 4), 1 if s.get('dataset') == 'prostate' else 2)
        save(out / 'PRIVATE_METRICS.pt', private)
        public = {d: {k: v for k, v in r.items() if k != 'patients'} for d, r in results.items()}
        write(out / 'METRICS.json', public)
        save(out / 'PRIVATE_PATIENT_METRICS.pt', results)
        return public
    finally:
        restore_rng_state(rng); model.train(previous)


def run(s):
    device = os.environ['CUDA_VISIBLE_DEVICES']
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    free = {uuid.strip(): int(memory) for uuid, memory in (line.split(',') for line in devices.splitlines())}
    assert free[device] >= 4096, 'GPU admission: require 4 GiB free including headroom'
    setup(); out = Path(s['output']); out.mkdir(exist_ok=False)
    write(out / 'CONTEXT.json', s)
    model = initialize(s, s['family']); teacher = copy.deepcopy(model).eval().requires_grad_(False)
    opt = optimizer(model); completed = 0; source = None
    if s['stage'] == 'target':
        source = torch.load(s['source'], map_location='cpu', weights_only=False)
        assert source['step'] == 5000
        model.load_state_dict(source['model']); teacher.load_state_dict(source['teacher']); restore_rng_state(source['rng'])
        write(out / 'ARRIVAL.json', json.loads((Path(s['source']).parent / 'METRICS.json').read_text()))
    seed = s.get('training_seed', s.get('schedule_seed', 163))
    if 'training_seed' in s:
        torch.manual_seed(seed); np.random.seed(seed)
    schedule = Schedule(s['data'], s['domain'], seed=seed, dataset=s.get('dataset', 'mnms'))
    unlabeled = Schedule(s['data'], s['domain'], True, seed=seed, dataset=s.get('dataset', 'mnms')) if s['use_u'] else None
    current = s['styles'][s['domain']]
    history = s['styles'][s.get('source_domain', 'Siemens')] if s['history'] else None
    def snapshot():
        return dict(step=completed, model=model.state_dict(), teacher=teacher.state_dict(), optimizer=opt.state_dict(), rng=capture_rng_state(), family=s['family'], original_started_unix=s['original_started_unix'])
    try:
        save(out / 'S0000.pt', snapshot())
        started = time.time()
        for step in range(1, 5001):
            assert time.time() - s['original_started_unix'] < 85200
            absolute = step + (5000 if s['stage'] == 'target' else 0)
            x, y, indices = schedule.batch(absolute)
            u = unlabeled.batch(absolute)[0] if unlabeled else None
            opt.zero_grad(set_to_none=True)
            loss, parts = step_loss(model, teacher, x, y, u, current, history, step, seed, s.get("history_l", True), s.get("history_u", True), s.get("style_mode", "history"), s.get("fixed_jl"), s.get("ju_weight", .005), s.get("consistency_weight", 1.0), s.get("history_strength", 1.0))
            assert torch.isfinite(loss), parts
            loss.backward()
            params = [p for p in model.parameters() if p.requires_grad]
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1.)); opt.step(); ema(teacher, model); completed = step
            append(out / 'steps.jsonl', dict(step=step, loss=float(loss.detach()), grad_norm=norm, elapsed=time.time()-started, **parts))
            if step % 50 == 0:
                write(out / 'STATUS.json', dict(status='TRAINING', step=step, elapsed=time.time()-started))
            if step % 1000 == 0: save(out / f'S{step:04}.pt', snapshot())
        assert all(int(v['step']) == 5000 for v in opt.state.values())
        metrics = evaluate(model, s, out)
        write(out / 'RESULTS.json', dict(status='COMPLETE', updates=completed, metrics=metrics, use_u=s['use_u'], history=s['history'], test_access=False, elapsed_seconds=time.time()-started))
        write(out / 'STATUS.json', dict(status='COMPLETE', step=completed)); write(out / 'EXIT.json', dict(exit_code=0))
    except BaseException:
        save(out / 'failure_partial.pt', snapshot())
        write(out / 'FAILURE.json', dict(step=completed, traceback=traceback.format_exc()))
        write(out / 'EXIT.json', dict(exit_code=1)); raise


if __name__ == '__main__': run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
