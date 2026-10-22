"""Zero-update paired patient diagnostics; exports anonymous row numbers only."""
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch


def patients(path, domain, class_index):
    rows = torch.load(path / 'PRIVATE_METRICS.pt', map_location='cpu', weights_only=False)[domain]
    grouped = defaultdict(list)
    for row in rows:
        c = next(c for c in row['metrics']['classes'] if c['class_index'] == class_index)
        if not c['GT_absent']:
            grouped[row['patient']].append(c)
    out = {}
    for key, cells in grouped.items():
        out[key] = {name: float(np.mean([c[name] for c in cells])) for name in ('Dice', 'HD95_penalized_mm')}
        out[key].update({name: sum(c[name] for c in cells) for name in ('GT', 'predicted', 'TP', 'FP', 'FN')})
        out[key]['phases'] = len(cells)
    return out


def compare(arms, comparisons, domain, class_index):
    data = {name: patients(path, domain, class_index) for name, path in arms.items()}
    keys = sorted(next(iter(data.values())))
    assert all(set(v) == set(keys) for v in data.values())
    anonymous = []
    for i, key in enumerate(keys):
        row = dict(anonymous_patient=f'case_{i+1:02}', arms={n: v[key] for n, v in data.items()})
        assert len({v[key]['GT'] for v in data.values()}) == 1
        anonymous.append(row)
    result = dict(n_patients=len(keys), class_index=class_index, rows=anonymous, comparisons={})
    for candidate, baseline in comparisons:
        deltas = {metric: np.array([data[candidate][k][metric]-data[baseline][k][metric] for k in keys]) for metric in ('Dice', 'HD95_penalized_mm', 'FP', 'FN', 'predicted')}
        d = deltas['Dice'] * 100; h = deltas['HD95_penalized_mm']
        loo = [float(np.delete(d, i).mean()) for i in range(len(d))]
        result['comparisons'][candidate+'-'+baseline] = dict(
            mean_Dice_delta_pp=float(d.mean()), Dice_improved=int((d>0).sum()),
            mean_HD95_delta_mm=float(h.mean()), HD95_improved=int((h<0).sum()),
            Dice_leave_one_out_pp=loo, Dice_leave_one_out_range_pp=[min(loo), max(loo)],
            HD95_leave_one_out_mm=[float(np.delete(h,i).mean()) for i in range(len(h))],
            Dice_down_HD95_improved=int(((d<0)&(h<0)).sum()),
            Dice_down_FN_up=int(((d<0)&(deltas['FN']>0)).sum()),
            Dice_down_predicted_volume_down=int(((d<0)&(deltas['predicted']<0)).sum()),
            total_FP_change=int(deltas['FP'].sum()), total_FN_change=int(deltas['FN'].sum()),
            total_predicted_voxels_change=int(deltas['predicted'].sum()))
    return result


def run(base):
    b = base/'sslcl_b11_tuning_20261022'; t = base/'sslcl_b11_mechanism_tuning_20261022'
    result = dict(status='COMPLETE', optimizer_updates=0, new_model_forwards=0,
                  volume_unit='stored-grid voxel counts; summed across phases, not physical mL',
                  patient_ids_exported=False, test_access=False, prostate={}, cardiac={})
    for domain in ('RUNMC','BMC'):
        result['prostate'][domain] = compare(dict(default=b/'P2', anchor=b/'P1', selected=t/'P01'), [('selected','default'),('selected','anchor')], domain, 1)
    for domain in ('Siemens','Philips'):
        result['cardiac'][domain] = compare(dict(default=b/'C2', diagnostic=t/'C01'), [('diagnostic','default')], domain, 3)
    return result

if __name__ == '__main__':
    print(json.dumps(run(Path(os.environ['RUNS_ROOT'])), indent=2))
