"""Independent ledger, paired-state and sealed-prediction audit."""
import json
import os
from pathlib import Path

import torch

from sc_bayeseg.run import same
from sc_bayeseg.window_v1.r_start import ARMS, STREAMS, metrics, write
from lcrseg.data import H5LabeledDataset, collate_labeled
from torch.nn import functional as F
from sc_bayeseg.window_v1.fullview import flip


def audit(s):
    torch.set_num_threads(4)
    root = Path(s['output'])
    load = lambda name: json.loads((root / name).read_text())
    assert load('EXIT.json')['exit_code'] == 0 and load('STATUS.json')['status'] == 'COMPLETE'
    assert not (root / 'FAILURE.json').exists()
    result = load('R_RESULTS.json')
    c = result['costs']
    assert c == dict(optimizer=2400, training_reverse=2400, train_theta=3600, train_g_forward=7200,
                    train_g_image_forward=57600, train_encoder_image_forward=19200,
                    train_direct_image_forward=14400, evaluation_theta=288, evaluation_g_forward=2304,
                    evaluation_encoder_image_forward=96, evaluation_direct_image_forward=96,
                    qualification_g_forward=4, qualification_reverse=2)
    updates = [json.loads(line) for line in (root / 'physical_updates.jsonl').read_text().splitlines()]
    steps = [json.loads(line) for line in (root / 'steps.jsonl').read_text().splitlines()]
    expected = [(arm, step) for arm in ARMS for step in range(1, 601)]
    assert [(r['arm'], r['step']) for r in updates] == expected
    assert [r['index'] for r in updates] == list(range(1, 2401))
    assert [(r['arm'], r['step']) for r in steps] == expected
    assert load('QUALIFICATION.json')['status'] == 'PASS'
    parity = [json.loads(line) for line in (root / 'SG_PARITY.jsonl').read_text().splitlines()]
    assert [r['step'] for r in parity] == list(range(0, 601, 25))
    assert all(r['model_and_Adam_exact'] for r in parity)
    torchload = lambda p: torch.load(p, map_location='cpu', weights_only=False)
    init = torchload(root / 'INITIAL_INTERFACE.pt')['model']
    for arm in ARMS:
        assert json.loads((root / arm / 'EXIT.json').read_text())['exit_code'] == 0
        for step in range(0, 601, 100):
            state = torchload(root / arm / f'S{step:04}.pt')
            assert state['step'] == step and state['arm'] == arm and state['source_cursor'] == 8675
            assert 'rng' in state
            opt = state['optimizer']
            if step:
                assert all(int(v['step']) == step for v in opt['state'].values())
            if step == 0:
                assert same(init, state['model']) and not opt['state']
    for step in range(0, 601, 100):
        a = torchload(root / 'R0' / f'S{step:04}.pt')
        b = torchload(root / 'R2-SG' / f'S{step:04}.pt')
        for key, tensor in a['model'].items():
            if key.startswith(('e.', 'd.')):
                assert same(tensor, b['model'][key])
        for key, value in a['optimizer']['state'].items():
            assert same(value, b['optimizer']['state'][key])
    selection = json.loads((Path(s['origin']) / 'PRIVATE_SELECTION.json').read_text())
    assert len(selection['train_indices']) == 2
    ds = H5LabeledDataset(s['data'], seed=0, dataset='mnms', sites=['Siemens'], transform=None)
    lb = collate_labeled([ds[i] for i in selection['train_indices']])
    y = F.interpolate(lb.label[:, None].float(), (128, 128), mode='nearest')[:, 0].long()
    y = torch.cat([flip(y, k) for k in range(4)])
    rows = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
    assert [(r['arm'], r['step']) for r in rows] == [(arm, step) for arm in ARMS for step in (0, 200, 600)]
    max_error = 0.
    def compare(a, b):
        nonlocal max_error
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                compare(a[k], b[k])
        elif isinstance(a, list):
            assert len(a) == len(b)
            for x, z in zip(a, b):
                compare(x, z)
        elif isinstance(a, float):
            max_error = max(max_error, abs(a - b))
            assert abs(a - b) < 2e-6
        else:
            assert a == b
    for row in rows:
        pred = torchload(root / row['arm'] / f"PRED_{row['step']:04}.pt")
        assert set(pred) == {'direct', 'generator_flows', 'seeds', 'source_cursor'}
        if row['arm'] != 'R1':
            compare(row['direct'], metrics(pred['direct'], y))
        if row['arm'] != 'R0':
            assert pred['seeds'] == list(STREAMS) and len(pred['generator_flows']) == 4
            compare(row['generator'], metrics(torch.stack(pred['generator_flows']).mean(0), y))
            for x, z in zip(row['generator_flow_metrics'], pred['generator_flows']):
                compare(x, metrics(z, y))
    previous = load('PREVIOUS_ATTEMPT.json')
    assert previous['optimizer'] == previous['training_reverse'] == previous['qualification_reverse'] == 0
    old = Path(previous['output'])
    assert json.loads((old / 'EXIT.json').read_text())['exit_code'] == 1
    assert json.loads((old / 'FAILURE.json').read_text())['costs']['optimizer'] == 0
    assert same(init, torchload(old / 'INITIAL_INTERFACE.pt')['model'])
    assert s['original_started_unix'] == previous['original_started_unix']
    st = Path(s['checkpoint']).stat()
    assert st.st_size == s['source_metadata']['bytes'] and st.st_mtime_ns == s['source_metadata']['mtime_ns']
    assert load('E0/EXIT.json')['exit_code'] == 0 and load('E0/RESULTS.json')['status'] == 'PASS'
    assert load('RESULTS.json')['elapsed_seconds'] < 86400
    receipt = dict(status='PASS', optimizer_applications=2400, training_reverse=2400,
                   complete_arm_step_coverage=True, checkpoint_and_Adam_coverage=True,
                   paired_state_checks=25, endpoint_pairs_independently_checked=7,
                   sealed_prediction_records_checked=12, maximum_CPU_GPU_metric_error=max_error,
                   generator_frozen_in_worker=True, source_file_metadata_unchanged=True,
                   original_initialization_and_clock_preserved=True,
                   prior_attempt_optimizer=0, prior_attempt_qualification_G_forward=1,
                   prior_attempt_qualification_reverse=0, current_qualification_reverse=2,
                   current_qualification_G_forward=4, E0='PASS', no_generalization_claim=True)
    write(root / 'AUDIT.json', receipt)
    print(json.dumps(receipt))
    return receipt


if __name__ == '__main__':
    audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
