"""Dataset/epoch adapter around the existing supervised method and Trainer."""
import csv
import json
import math
import os
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..di_dmpa_gate1.binding import safe_asset
from ..five_frameworks_v1.checkpoint import atomic_save, PhysicalLedger
from ..lcrseg.contracts import LabeledBatch
from ..lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from ..lcrseg.engine.trainer import Trainer, TrainerState, build_optimizer, seed_everything
from ..lcrseg.methods.supervised import SupervisedMethod
from ..lcrseg.models import UNet2D
from ..main_style_ssl_v1.runtime import append, write
from .protocol import PROTOCOL as P


def records(data, role):
    if role not in ('train_labeled', 'val'):
        raise PermissionError('only current training-L and validation are authorized')
    with (Path(data) / 'manifests/training/lcrseg_v1_seed0.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['dataset'] == 'mnms'
                and r['site_or_vendor'] == 'Siemens' and r['primary_20pct_split'] == role]
    expected = (10, 20) if role == 'train_labeled' else (23, 46)
    patients = {r['patient_id'] for r in rows}
    if (len(patients), len(rows)) != expected:
        raise ValueError('frozen patient/phase counts differ')
    for patient in patients:
        if sorted(r['phase'] for r in rows if r['patient_id'] == patient) != ['ED', 'ES']:
            raise ValueError('patient must have exactly ED and ES')
    if any(r['split_seed'] != '0' or r['cohort'] != 'canonical320'
           or r['evaluation_eligible'] != 'true' or not r['label_h5_relpath'] for r in rows):
        raise ValueError('invalid frozen role binding')
    return sorted(rows, key=lambda r: r['case_id'])


def read_volume(data, row):
    with h5py.File(safe_asset(data, row['image_h5_relpath']), 'r') as f:
        x = f['image'][...]
    with h5py.File(safe_asset(data, row['label_h5_relpath']), 'r') as f:
        y = f['label'][...]
    if x.dtype != np.float16 or x.ndim != 3 or x.shape[1:] != (384, 384) or y.shape != x.shape:
        raise ValueError('unexpected frozen MRI layout')
    if not np.isfinite(x).all() or x.min() < -1.001 or x.max() > 1.001 or not np.isin(y, [0, 1, 2, 3]).all():
        raise ValueError('invalid image intensity or label values')
    return torch.from_numpy(x.astype(np.float32))[:, None], torch.from_numpy(y.astype(np.uint8))


def crop_input(x, y=None):
    top, left = P['crop_yx']; n = P['crop_pixels']
    if x.shape[-2:] != (384, 384):
        raise ValueError('crop expects the existing physical canvas')
    x = F.interpolate(x[..., top:top+n, left:left+n], (384, 384), mode='bilinear', align_corners=False)
    if y is None:
        return x
    y = F.interpolate(y[:, None, top:top+n, left:left+n].float(), (384, 384), mode='nearest')[:, 0].long()
    return x, y


def restore_prediction(logits):
    top, left = P['crop_yx']; n = P['crop_pixels']
    prob = F.interpolate(logits.softmax(1), (n, n), mode='bilinear', align_corners=False)
    pred = torch.zeros((len(logits), 384, 384), dtype=torch.long, device=logits.device)
    pred[:, top:top+n, left:left+n] = prob.argmax(1)
    return pred


def augment(x, y, seed):
    g = torch.Generator().manual_seed(seed)
    cfg = P['augmentation']; b = len(x)
    signs = torch.where(torch.rand(b, 2, generator=g) < cfg['flip_probability_each_axis'], -1., 1.)
    angle = (torch.rand(b, generator=g)*2-1) * math.radians(cfg['rotation_degrees'])
    angle *= (torch.rand(b, generator=g) < cfg['rotation_probability'])
    theta = torch.zeros(b, 2, 3)
    theta[:, 0, 0] = angle.cos()*signs[:, 0]; theta[:, 0, 1] = -angle.sin()*signs[:, 1]
    theta[:, 1, 0] = angle.sin()*signs[:, 0]; theta[:, 1, 1] = angle.cos()*signs[:, 1]
    grid = F.affine_grid(theta.to(x.device), x.shape, align_corners=False)
    x = F.grid_sample(x+1, grid, mode='bilinear', padding_mode='zeros', align_corners=False)-1
    y = F.grid_sample(y[:, None].float(), grid, mode='nearest', padding_mode='zeros', align_corners=False)[:, 0].long()
    lo, hi = cfg['gamma_range']
    gamma = lo+(hi-lo)*torch.rand(b, generator=g)
    gamma = torch.where(torch.rand(b, generator=g) < cfg['gamma_probability'], gamma, 1.)
    x = 2*((x+1)/2).clamp(0, 1).pow(gamma.to(x.device)[:, None, None, None])-1
    return x, y


def make_trainer(seed, device, ledger):
    seed_everything(seed)
    model = UNet2D(1, 4)
    model.projection_head = nn.Identity()
    model.to(device)
    method = SupervisedMethod(model, config={'lambda_dice': 1., 'lambda_anchor_sup': 0.})
    method.begin_site('Siemens', None, P['steps_per_seed'])
    optimizer = build_optimizer(method, lr=P['lr'], weight_decay=P['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    trainer = Trainer(method, optimizer=optimizer, scheduler=scheduler, device=device, amp=False)
    state = TrainerState()
    def finite_gradients(opt, args, kwargs):
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for group in opt.param_groups for p in group['params']):
            raise FloatingPointError('nonfinite gradient before Adam')
    optimizer.register_step_pre_hook(finite_gradients)
    optimizer.register_step_post_hook(lambda opt, args, kwargs: ledger.append(state.global_step+1))
    return trainer, state


def batch(x, y):
    n = len(x)
    return LabeledBatch(x, y, torch.ones_like(y[:, None], dtype=torch.bool),
                        ['private']*n, ['private']*n, ['Siemens']*n, [None]*n)


def step(trainer, state, x, y):
    result = trainer.train_step(batch(x, y), None, state=state)
    if result.scalars['optimizer_step_skipped']:
        raise RuntimeError('FP32 optimizer unexpectedly skipped')
    state.global_step += 1; state.site_step += 1
    return {k: float(result.losses[k].detach()) for k in ('loss_sup', 'loss_seg_ce', 'loss_seg_dice')}


def snapshot(trainer, state, seed):
    return dict(protocol=P, seed=seed, step=state.global_step, epoch=state.epoch,
                model=trainer.method.model.state_dict(), trainer=trainer.state_dict(), rng=capture_rng_state())


def restore(trainer, state, payload, seed):
    if payload['protocol'] != P or payload['seed'] != seed:
        raise ValueError('checkpoint protocol/seed mismatch')
    trainer.method.model.load_state_dict(payload['model'])
    trainer.load_state_dict(payload['trainer']); restore_rng_state(payload['rng'])
    state.global_step = state.site_step = payload['step']; state.epoch = payload['epoch']


@torch.no_grad()
def evaluate(model, data, rows, out, epoch):
    model.eval(); by_patient = {}; counts = 0; retained = 0; foreground = 0
    top, left = P['crop_yx']; n = P['crop_pixels']; device = next(model.parameters()).device
    for row in rows:
        x, y = read_volume(data, row); predictions = []
        for i in range(0, len(x), P['batch_size']):
            logits = model(crop_input(x[i:i+P['batch_size']].to(device))).logits
            predictions.append(restore_prediction(logits).cpu())
        pred = torch.cat(predictions); dices = []
        for c in (1, 2, 3):
            pp, tt = pred == c, y == c; den = int(pp.sum()+tt.sum())
            dices.append(1. if den == 0 else 2*int((pp & tt).sum())/den)
        by_patient.setdefault(row['patient_id'], []).append(dices); counts += len(x)
        foreground += int((y > 0).sum()); retained += int((y[:, top:top+n, left:left+n] > 0).sum())
        append(out/'private_validation.jsonl', dict(epoch=epoch, patient=row['patient_id'], phase=row['phase'], dice=dices))
    if len(by_patient) != 23 or any(len(v) != 2 for v in by_patient.values()):
        raise ValueError('incomplete patient-paired validation')
    classes = np.mean([np.mean(v, axis=0) for v in by_patient.values()], axis=0)
    return dict(epoch=epoch, primary_dice=float(classes.mean()), classes=dict(zip(('LV', 'MYO', 'RV'), classes.tolist())),
                patients=23, phases=46, image_forward_slices=counts, foreground_retention=retained/foreground)


def load_training(data):
    rows = records(data, 'train_labeled'); images = []; labels = []
    for row in rows:
        x, y = read_volume(data, row); images.append(x); labels.append(y)
    x, y = torch.cat(images), torch.cat(labels)
    if len(x) != P['train_slices']:
        raise ValueError('unexpected slice budget')
    return x, y, rows


def run_seed(config, seed):
    out = Path(config['root'])/'jobs'/f'seed{seed}'; out.mkdir(parents=True, exist_ok=False)
    started = time.time(); ledger = PhysicalLedger(out/'physical_updates.jsonl'); state = TrainerState()
    train_exposure = val_exposure = 0; best = -1.; best_epoch = None
    try:
        x, y, train_rows = load_training(config['data']); val_rows = records(config['data'], 'val')
        if {r['patient_id'] for r in train_rows} & {r['patient_id'] for r in val_rows}:
            raise ValueError('patient leakage')
        trainer, state = make_trainer(seed, 'cuda', ledger); torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, P['epochs']+1):
            order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed+10000*epoch))
            sums = {k: 0. for k in ('loss_sup', 'loss_seg_ce', 'loss_seg_dice')}
            for start in range(0, len(order), P['batch_size']):
                idx = order[start:start+P['batch_size']]
                bx, by = crop_input(x[idx].cuda(), y[idx].cuda())
                bx, by = augment(bx, by, seed+1000000*epoch+start)
                losses = step(trainer, state, bx, by); train_exposure += len(idx)
                for key in sums: sums[key] += losses[key]*len(idx)
            state.epoch = epoch
            record = dict(epoch=epoch, step=state.global_step, **{k: v/len(x) for k, v in sums.items()}, elapsed_seconds=time.time()-started)
            append(out/'train.jsonl', record); print(json.dumps(dict(seed=seed, **record)), flush=True)
            write(out/'STATUS.json', dict(status='RUNNING', **record, train_image_exposure=train_exposure))
            if epoch % P['validation_every_epochs'] == 0:
                atomic_save(snapshot(trainer, state, seed), out/'latest.pt')
                write(out/'STATUS.json', dict(status='VALIDATING', epoch=epoch, step=state.global_step))
                val = evaluate(trainer.method.model, config['data'], val_rows, out, epoch)
                val_exposure += val['image_forward_slices']; append(out/'validation.jsonl', val)
                if val['primary_dice'] > best:
                    best, best_epoch = val['primary_dice'], epoch
                    atomic_save(snapshot(trainer, state, seed), out/'best.pt')
        assert state.global_step == P['steps_per_seed'] and train_exposure == 46000
        atomic_save(snapshot(trainer, state, seed), out/'final.pt')
        write(out/'VALIDATION.json', dict(final=val, best_secondary=dict(epoch=best_epoch, primary_dice=best)))
        write(out/'COSTS.json', dict(**ledger.summary(state.global_step), train_image_exposure=train_exposure,
              val_image_forward_slices=val_exposure, epochs=state.epoch, elapsed_seconds=time.time()-started,
              peak_memory_bytes=torch.cuda.max_memory_allocated(), training_patients=10, U_image_exposure=0, test_access=0))
        write(out/'STATUS.json', dict(status='COMPLETE', epoch=state.epoch, step=state.global_step, validation=val))
        write(out/'EXIT.json', dict(exit_code=0))
        return dict(seed=seed, status='COMPLETE', final=val, best_secondary=dict(epoch=best_epoch, primary_dice=best))
    except BaseException as exc:
        write(out/'COSTS.json', dict(**ledger.summary(state.global_step), train_image_exposure=train_exposure,
              val_image_forward_slices=val_exposure, elapsed_seconds=time.time()-started))
        write(out/'STATUS.json', dict(status='FAILED', epoch=state.epoch, step=state.global_step, error=str(exc), traceback=traceback.format_exc()))
        write(out/'EXIT.json', dict(exit_code=1, error=str(exc)))
        return dict(seed=seed, status='FAILED', error=str(exc))


def qualify(config):
    out = Path(config['root'])/'engineering'; out.mkdir(exist_ok=False)
    ledger = PhysicalLedger(out/'physical_updates.jsonl'); state = TrainerState()
    try:
        rows = records(config['data'], 'train_labeled'); patients = sorted({r['patient_id'] for r in rows})[:2]
        xs, ys = [], []
        for p in patients:
            row = next(r for r in rows if r['patient_id'] == p and r['phase'] == 'ED')
            x, y = read_volume(config['data'], row)
            ids = (y > 0).flatten(1).sum(1).argsort(descending=True)[:3]
            xs.append(x[ids]); ys.append(y[ids])
        x, y = crop_input(torch.cat(xs).cuda(), torch.cat(ys).cuda())
        trainer, state = make_trainer(163, 'cuda', ledger); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(): initial = float(trainer.method.training_step(batch(x, y), None, 0, 0).total_loss)
        for _ in range(1):
            losses = step(trainer, state, x, y)
            append(out/'fit.jsonl', dict(step=state.global_step, **losses))
        with torch.no_grad(): final = float(trainer.method.training_step(batch(x, y), None, 1, 1).total_loss)
        write(out/'FIT_RESULT.json', dict(initial_loss=initial, final_loss=final, steps=state.global_step))
        atomic_save(snapshot(trainer, state, 163), out/'fit_final.pt')
        atomic_save(snapshot(trainer, state, 163), out/'restore.pt')
        first = step(trainer, state, x, y)
        expected = {k: v.detach().clone() for k, v in trainer.method.model.state_dict().items()}
        restore(trainer, state, torch.load(out/'restore.pt', map_location='cpu', weights_only=False), 163)
        second = step(trainer, state, x, y)
        if first != second or any(not torch.equal(v, expected[k]) for k, v in trainer.method.model.state_dict().items()):
            raise AssertionError('checkpoint replay is not exact')
        count = ledger.summary(state.global_step)['physical_optimizer_updates']; assert count == 3
        write(out/'RESULT.json', dict(status='PASS', initial_loss=initial, final_loss=final, physical_optimizer_updates=count,
              training_L_forward_exposure=30, two_patients=True, exact_restore=True,
              peak_memory_bytes=torch.cuda.max_memory_allocated(), val_access=0, U_access=0, test_access=0))
    except BaseException as exc:
        write(out/'RESULT.json', dict(status='FAILED', physical_optimizer_updates=ledger.summary(state.global_step)['physical_optimizer_updates'],
              initial_loss=locals().get('initial'), final_loss=locals().get('final'),
              error=str(exc), traceback=traceback.format_exc()))
        raise


def main():
    config = json.loads(Path(os.environ['TASK_SETTINGS']).read_text()); root = Path(config['root'])
    if config['protocol'] != P or config.get('execution_authorized') is not True:
        raise PermissionError('missing exact frozen authority')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0': raise PermissionError('GPU0 only')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if config['action'] == 'qualify': return qualify(config)
    if config['action'] != 'train': raise ValueError('unknown action')
    if json.loads((root/'engineering/RESULT.json').read_text())['status'] != 'PASS':
        raise PermissionError('qualification incomplete')
    results = []
    for seed in P['seeds']:
        write(root/'STATUS.json', dict(status='RUNNING', active_seed=seed, completed=results))
        results.append(run_seed(config, seed)); torch.cuda.empty_cache()
    complete = all(r['status'] == 'COMPLETE' for r in results)
    summary = dict(status='COMPLETE' if complete else 'TERMINAL_WITH_FAILURE', results=results,
                   mean_final_dice=float(np.mean([r['final']['primary_dice'] for r in results])) if complete else None)
    write(root/'SUMMARY.json', summary); write(root/'STATUS.json', summary)


if __name__ == '__main__': main()
