"""Verify matched source states, real updates and sealed supervised metrics."""
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset, collate_labeled
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.r_start import metrics, write
from sc_bayeseg.window_v1.fullview import flip


def audit(s):
    torch.set_num_threads(4)
    root = Path(s['output'])
    load = lambda name: json.loads((root / name).read_text())
    pt = lambda p: torch.load(p, map_location='cpu', weights_only=False)
    assert load('EXIT.json')['exit_code'] == 0 and load('RESULTS.json')['status'] == 'COMPLETE'
    assert load('STATUS.json')['status'] == 'COMPLETE' and not (root / 'FAILURE.json').exists()
    assert load('LOSS_CHECK.json')['status'] == 'PASS'
    repaired = bool(s.get('reuse_CE'))
    factor = 1 if repaired else 2
    assert load('RESULTS.json')['costs'] == dict(optimizer=1000*factor, reverse=1000*factor, train_image_forward=8000*factor,
                 prediction_image_forward=32*factor, generator_forward=0, qualification_optimizer=0, qualification_reverse=0)
    physical = [json.loads(x) for x in (root / 'physical_updates.jsonl').read_text().splitlines()]
    steps = [json.loads(x) for x in (root / 'steps.jsonl').read_text().splitlines()]
    arms = ('CLASS_MEAN_CE',) if repaired else ('CE', 'CLASS_MEAN_CE')
    expected = [(arm, step) for arm in arms for step in range(1, 1001)]
    assert [(r['arm'], r['step']) for r in physical] == expected
    assert [(r['arm'], r['step']) for r in steps] == expected
    assert [r['index'] for r in physical] == list(range(1, 1000*factor+1))
    assert all(r['total_interface_step'] == r['step'] + 600 for r in physical)
    source = pt(s['interface_source'])
    h = {k: v for k, v in source['model'].items() if k.startswith(('hc.', 'hr.'))}
    for arm in ('CE', 'CLASS_MEAN_CE'):
        armroot = Path(s['reuse_CE'])/'CE' if repaired and arm=='CE' else root/arm
        assert json.loads((armroot/'EXIT.json').read_text())['exit_code'] == 0
        for step in range(0, 1001, 100):
            state = pt(armroot / f'S{step:04}.pt')
            assert state['step'] == step and state['total_interface_step'] == 600 + step
            assert all(int(v['step']) == 600 + step for v in state['optimizer']['state'].values())
            assert same(state['rng'], source['rng'])
            assert all(same(v, state['model'][k]) for k, v in h.items())
            if step == 0:
                assert same(state['model'], source['model']) and same(state['optimizer'], source['optimizer'])
    sel = json.loads((Path(s['origin']) / 'PRIVATE_SELECTION.json').read_text())
    ds = H5LabeledDataset(s['data'], seed=0, dataset='mnms', sites=['Siemens'], transform=None)
    lb = collate_labeled([ds[i] for i in sel['train_indices']])
    labels = F.interpolate(lb.label[:, None].float(), (128, 128), mode='nearest')[:, 0].long()
    labels = torch.cat([flip(labels, k) for k in range(4)])
    assert torch.bincount(labels.flatten(), minlength=4).tolist() == load('DESIGN.json')['class_pixel_counts_all_8_views']
    rows = load('VALID_METRICS.json') if repaired else [json.loads(x) for x in (root / 'metrics.jsonl').read_text().splitlines()]
    assert [(r['arm'], r['step']) for r in rows] == [(arm, step) for arm in ('CE', 'CLASS_MEAN_CE') for step in (0, 250, 500, 1000)]
    source_probability = pt(Path(s['interface_source']).parent / 'PRED_0600.pt')['direct']
    metric_error = [0.]
    def compare(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                compare(a[k], b[k])
        elif isinstance(a, list):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                compare(x, y)
        elif isinstance(a, float):
            metric_error[0] = max(metric_error[0], abs(a-b))
            assert abs(a-b) < 2e-6
        else:
            assert a == b
    for row in rows:
        armroot = Path(s['reuse_CE'])/'CE' if repaired and row['arm']=='CE' else root/row['arm']
        artifact = pt(armroot / f"PRED_{row['step']:04}.pt")
        assert list(artifact) == ['probability']
        prob = artifact['probability']
        compare(row['metrics'], metrics(prob, labels))
        if row['step'] == 0:
            assert same(prob, source_probability)
    original = json.loads((Path(s['interface_source']).parent.parent / 'CONTEXT.json').read_text())
    assert s['original_started_unix'] == original['original_started_unix']
    assert load('RESULTS.json')['original_elapsed_seconds'] < 86400
    if repaired:
        failed = Path(s['reuse_CE'])
        assert json.loads((failed/'EXIT.json').read_text())['exit_code'] == 1
        assert json.loads((failed/'FAILURE.json').read_text())['costs']['optimizer'] == 2000
        oldupdates = [json.loads(x) for x in (failed/'physical_updates.jsonl').read_text().splitlines()]
        assert [(r['arm'],r['step']) for r in oldupdates] == [(a,j) for a in ('CE','CLASS_MEAN_CE') for j in range(1,1001)]
        for step, expected_adam in ((0,1600),(1000,2600)):
            invalid = pt(failed/'CLASS_MEAN_CE'/f'S{step:04}.pt')
            assert all(int(v['step'])==expected_adam for v in invalid['optimizer']['state'].values())
        assert load('RESULTS.json')['costs_including_invalid_attempt'] == dict(optimizer=3000,reverse=3000,
            train_image_forward=24000,prediction_image_forward=96,generator_forward=0,
            valid_optimizer=2000,invalid_optimizer=1000,invalid_Adam_steps=[1600,2600])
    result = dict(status='PASS', optimizer=3000 if repaired else 2000, reverse=3000 if repaired else 2000,
                  valid_optimizer=2000, invalid_optimizer=1000 if repaired else 0, complete_step_coverage=True,
                  full_state_checkpoints=22, sealed_predictions_checked=8,
                  exact_source_model_Adam_RNG=True, source_prediction_exact=True,
                  unused_H_frozen=True, RNG_unchanged=True, Adam_steps=[600, 1600],
                  original_clock_preserved=True, generator_calls=0, no_generalization_claim=True)
    result['maximum_metric_recompute_error'] = metric_error[0]
    write(root / 'AUDIT.json', result)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
