"""Finish the interrupted B train evaluation without replaying training or A inference."""
import json
import os
from pathlib import Path
import torch
from .run import NODE, MAX_STEPS, aggregate, evaluate, existing_validation, compare
from .run import build, Parent, Deployment, write


def saved_rows(path, patients):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    summary = aggregate(rows)
    if summary['patients'] != patients:
        raise ValueError('incomplete saved evaluation')
    return {**summary, 'reused_completed_inference': True}, rows


def main(cfg):
    source = Path(cfg['output'])
    out = source.parent/'evaluation_recovery'
    costs = json.loads((source/'COSTS.json').read_text())
    if costs['scientific_step'] != MAX_STEPS or costs['cumulative_physical_updates'] != 43601:
        raise ValueError('training is not complete at the authorized budget')
    if (source/'private_B_train.jsonl').exists():
        raise ValueError('B evaluation may already have started')
    atrain, atrain_rows = saved_rows(source/'private_A_train.jsonl', 10)
    aval, aval_rows = saved_rows(source/'private_A_val.jsonl', 23)
    bval, bval_rows = existing_validation(cfg)
    compare(aval_rows, bval_rows)
    out.mkdir(exist_ok=False)
    write(out/'STATUS.json', {'status': 'EVALUATING', 'optimizer_updates': 0})
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    payload = torch.load(Path(cfg['reference_run'])/'sealed.pt', map_location='cpu', weights_only=False)
    if not payload['sealed'] or any(payload[k] != NODE[k] for k in NODE) or payload['updates'] != MAX_STEPS:
        raise ValueError('incorrect existing SSL source')
    native = build(cfg['reference'], torch.device('cuda'), 163, input_channels=1, num_classes=4, domain='Siemens')
    native.load_state_dict(payload['native'], strict=True)
    model = Deployment(Parent(native, 163, {'kind': 'existing_source'}, source=True), payload['transform'].cuda()).cuda().eval().requires_grad_(False)
    btrain, btrain_rows = evaluate(model, cfg, 'train_labeled', out/'private_B_train.jsonl')
    result = dict(status='COMPLETE', seed=163, A={'train': atrain, 'val': aval}, B={'train': btrain, 'val': bval},
                  paired_train=compare(atrain_rows, btrain_rows), paired_val=compare(aval_rows, bval_rows),
                  new_optimizer_updates=43601, scientific_step=MAX_STEPS, recovery_optimizer_updates=0,
                  independent_test_access=False, checkpoint_selection='fixed final',
                  interpretation='Selected development seed; historical B used different GPU/backend. A inference and B validation reused; only unfinished B train inference executed.')
    write(out/'RESULTS.json', result)
    write(out/'COSTS.json', {**costs, 'evaluation_recovery_optimizer_updates': 0})
    lines = ['# Siemens L-only versus existing SSL, seed163', '',
             '|Arm|Train Dice (%)|Validation Dice (%)|Train-val gap (pp)|', '|---|---:|---:|---:|']
    for arm in ('A', 'B'):
        tr, va = result[arm]['train']['primary_dice'], result[arm]['val']['primary_dice']
        lines.append(f'|{arm}|{100*tr:.4f}|{100*va:.4f}|{100*(tr-va):.4f}|')
    lines += ['', f'Existing SSL minus L-only validation: {result["paired_val"]["delta_B_minus_A_pp"]:.4f}pp.',
              'A is lambda_U=0 with the original labeled objective, initialization, sampling and schedule.',
              'The B-loader device-type error occurred after A inference; recovery reused A rows and ran only B train inference.',
              '43,600 scientific steps; 43,601 cumulative optimizer calls including the earlier failed call; recovery added zero.',
              'No independent test, U ground truth or checkpoint selection. Historical B differs in GPU/backend.',
              'Classes, ED/ES and paired-patient summaries are in RESULTS.json. Publication verification is separate.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'STATUS.json', {'status': 'COMPLETE', 'optimizer_updates': 0})
    write(out/'EXIT.json', {'exit_code': 0})


if __name__ == '__main__':
    main(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
