"""RGB Fundus views and image-level cup/disc metrics; no target-hidden labels."""
from collections import defaultdict
from pathlib import Path
import json
import time
import h5py
import numpy as np
import torch
from scipy import ndimage
from lcrseg.data.h5_dataset import load_training_records
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from sc_bayeseg.window_v1.r_start import write, save


def fov(x):
    return x.amax(dim=-3, keepdim=True) > 5 / 255


def moments(x):
    mask = fov(x).to(x.dtype)
    count = mask.sum((-2, -1), keepdim=True)
    assert (count > 0).all()
    mean = (x * mask).sum((-2, -1), keepdim=True) / count
    sd = (((x - mean).square() * mask).sum((-2, -1), keepdim=True) / count).sqrt().clamp_min(1e-4)
    return mean, sd


@torch.no_grad()
def rgb_style(schedule):
    patients = []
    for phases in schedule.groups.values():
        cases = []
        for ids in phases.values():
            values = []
            for i in ids:
                x = schedule.ds._read_image(schedule.ds.samples[i])
                mean, sd = moments(x)
                values.append(torch.cat([mean.flatten(), sd.flatten()]))
            cases.append(torch.stack(values).mean(0))
        patients.append(torch.stack(cases).mean(0))
    return torch.stack(patients).mean(0).tolist()


def rgb_view(x, current, history, step, seed=163, mode='history', strength=1.):
    assert x.shape[1] == 3 and mode in ('history', 'wide') and strength == 1.
    g = torch.Generator(device=x.device).manual_seed(seed * 1000 + step)
    r = torch.rand((len(x), 16, 1, 1), device=x.device, generator=g)
    cur = x.new_tensor(current).reshape(1, 6, 1, 1)
    mean, sd = cur[:, :3], cur[:, 3:]
    selected = r[:, :1] < .5
    if mode == 'wide':
        assert history is None  # Generic arm cannot inspect historical magnitude.
        target_mean = mean + .75 * sd * (2 * r[:, 10:13] - 1)
        target_sd = sd * torch.exp(np.log(1.5) * (2 * r[:, 13:16] - 1))
        mean = torch.where(selected, target_mean, mean)
        sd = torch.where(selected, target_sd, sd)
    elif history is not None:
        hist = x.new_tensor(history).reshape(1, 6, 1, 1)
        mean = torch.where(selected, hist[:, :3], mean)
        sd = torch.where(selected, hist[:, 3:], sd)
    own_mean, own_sd = moments(x)
    axis = torch.linspace(0, 2 * torch.pi, x.shape[-1], device=x.device)
    bias = (torch.cos(axis)[None, None, :, None] + torch.cos(axis)[None, None, None, :]) / 2
    z = (x - own_mean) / own_sd
    z = z * sd * (.8 + .4 * r[:, 1:4]) + mean + sd * (.1 * r[:, 4:7] - .05) + sd * (.1 * r[:, 7:10] - .05) * bias
    return torch.where(fov(x), z.clamp(0, 1), x)


def image_metrics(p, y):
    assert p.shape == (3, *y.shape) and np.isin(y, [0, 1, 2]).all()
    pred = p.argmax(0).numpy(); rows = {}
    for name, a, b in [('cup', pred == 2, y == 2), ('disc', pred > 0, y > 0), ('rim', pred == 1, y == 1)]:
        pn, gn = int(a.sum()), int(b.sum()); hd = None
        if pn and gn:
            sa = a & ~ndimage.binary_erosion(a, border_value=0)
            sb = b & ~ndimage.binary_erosion(b, border_value=0)
            hd = float(np.percentile(np.r_[ndimage.distance_transform_edt(~sb)[sa], ndimage.distance_transform_edt(~sa)[sb]], 95))
        rows[name] = dict(Dice=2 * int((a & b).sum()) / (pn + gn) if pn + gn else None,
                          HD95_pixels=hd, HD95_penalized_pixels=hd if hd is not None else float(np.linalg.norm(np.array(y.shape)-1)) if pn or gn else None,
                          GT_absent=gn == 0, empty_prediction_failure=gn > 0 and pn == 0)
    yt = torch.from_numpy(y.astype(np.int64)); one = torch.nn.functional.one_hot(yt, 3).permute(2, 0, 1)
    rows['NLL'] = float(-p.double().clamp_min(1e-30).log().gather(0, yt[None]).mean())
    rows['Brier'] = float((p.double() - one).square().sum(0).mean())
    return rows


def aggregate_fundus(rows):
    groups = defaultdict(list)
    for r in rows: groups[r['patient']].append(r['metrics'])
    patients = []
    for key, items in groups.items():
        entry = {'patient': key}
        for name in ('cup', 'disc', 'rim'):
            valid = [v[name] for v in items if not v[name]['GT_absent']]
            entry[name] = dict(Dice=float(np.mean([v['Dice'] for v in valid])) if valid else None,
                               HD95_penalized_pixels=float(np.mean([v['HD95_penalized_pixels'] for v in valid])) if valid else None)
        entry['mean_Dice'] = float(np.mean([entry[n]['Dice'] for n in ('cup', 'disc') if entry[n]['Dice'] is not None]))
        for n in ('NLL', 'Brier'): entry[n] = float(np.mean([v[n] for v in items]))
        patients.append(entry)
    out = dict(mean_Dice=float(np.mean([p['mean_Dice'] for p in patients])),
               lower_decile_Dice=float(np.quantile([p['mean_Dice'] for p in patients], .1, method='linear')),
               images=len(rows), patient_keys=len(patients), identity_caveat='Manifest keys; cross-eye patient linkage unverified.')
    for name in ('cup', 'disc', 'rim'):
        valid = [p[name] for p in patients if p[name]['Dice'] is not None]
        out[name] = dict(Dice=float(np.mean([p['Dice'] for p in valid])) if valid else None,
                         HD95_penalized_pixels=float(np.mean([p['HD95_penalized_pixels'] for p in valid])) if valid else None,
                         empty_prediction_failures=sum(r['metrics'][name]['empty_prediction_failure'] for r in rows),
                         GT_absent=sum(r['metrics'][name]['GT_absent'] for r in rows))
    for n in ('NLL', 'Brier'): out[n] = float(np.mean([p[n] for p in patients]))
    return out, patients


@torch.no_grad()
def evaluate_fundus(model, s, out):
    previous = model.training; rng = capture_rng_state(); model.eval(); results = {}; private = {}
    try:
        for domain in s['domains']:
            records = load_training_records(Path(s['data']), seed=0, dataset='fundus', sites=[domain], roles=['val'])
            rows = []
            for i, row in enumerate(sorted(records, key=lambda r: (r['patient_id'], r['case_id']))):
                assert time.time() - s['original_started_unix'] < 85200
                with h5py.File(Path(s['data']) / 'h5/v1' / row['image_h5_relpath']) as f:
                    x = torch.from_numpy(f['image'][...].astype(np.float32) / 255)[None].cuda()
                assert x.shape == (1, 3, 384, 384)
                p = model.probability(x)[0].cpu()
                assert torch.isfinite(p).all()
                with h5py.File(Path(s['data']) / 'h5/v1' / row['label_h5_relpath']) as f: y = f['label'][...]
                rows.append(dict(patient=row['patient_id'], case=row['case_id'], metrics=image_metrics(p, y)))
                if i % 20 == 0: write(out/'STATUS.json', dict(status='EVALUATING', domain=domain, image=i))
            results[domain], patients = aggregate_fundus(rows)
            private[domain] = dict(rows=rows, patients=patients)
        write(out/'METRICS.json', results); save(out/'PRIVATE_PATIENT_METRICS.pt', private)
        return results
    finally:
        restore_rng_state(rng); model.train(previous)


def source_gate(root, tasks, domain):
    rows = {}
    for task in tasks:
        m = json.loads((Path(root)/task/'METRICS.json').read_text())[domain]
        rows[task] = dict(cup=m['cup']['Dice'], disc=m['disc']['Dice'], empty=[m[n]['empty_prediction_failures'] for n in ('cup', 'disc')])
        rows[task]['pass'] = (m['cup']['Dice'] is not None and m['disc']['Dice'] is not None and m['cup']['Dice'] >= .65 and m['disc']['Dice'] >= .85 and all(m[n]['empty_prediction_failures'] == 0 for n in ('cup', 'disc')))
    return dict(status='PASS' if all(r['pass'] for r in rows.values()) else 'FAIL', seeds=rows)
