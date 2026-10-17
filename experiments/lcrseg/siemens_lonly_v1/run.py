"""One fixed source L-only run, followed by paired final Dice evaluation."""
import gc
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from ..main_style_ssl_v1.data import Accessor
from ..main_style_ssl_v1.model import Parent
from ..main_style_ssl_v1.protocol import nodes, options
from ..main_style_ssl_v1.runtime import setup, seal, write, append
from ..five_frameworks_v1.checkpoint import save, restore, PhysicalLedger
from ..five_frameworks_v1.model import Deployment
from ..five_frameworks_v1.native_parent import build

NODE = next(n for n in nodes() if n['id'] == 'mnms_s163_source')
CLASSES = ('LV', 'MYO', 'RV')
MAX_STEPS = 43600


def remaining_target(step, prior_physical, physical_limit):
    if any(type(v) is not int for v in (step, prior_physical, physical_limit)) or not 0 <= step <= prior_physical < physical_limit:
        raise ValueError('invalid cumulative physical budget')
    return min(MAX_STEPS, step + physical_limit-prior_physical)


def dice(pred, truth):
    if pred.shape != truth.shape or not np.isin(pred, range(4)).all() or not np.isin(truth, range(4)).all():
        raise ValueError('invalid prediction/label grid')
    cm = np.bincount((truth.astype(np.int64)*4+pred).ravel(), minlength=16).reshape(4, 4)
    den = cm.sum(0)+cm.sum(1)
    return np.divide(2*np.diag(cm), den, out=np.ones(4), where=den != 0)[1:]


def aggregate(rows):
    patients = {}
    phases = {'ED': [], 'ES': []}
    for row in rows:
        key, phase = row['patient'], row['phase']
        if phase in patients.setdefault(key, {}):
            raise ValueError('duplicate patient phase')
        patients[key][phase] = row['dice']
        phases[phase].append(row['dice'])
    if not patients or any(set(v) != {'ED', 'ES'} for v in patients.values()):
        raise ValueError('incomplete ED/ES pairs')
    values = np.array([np.mean(list(v.values()), axis=0) for v in patients.values()])
    return dict(primary_dice=float(values.mean()), classes=dict(zip(CLASSES, values.mean(0).tolist())),
                phases={k: dict(primary_dice=float(np.mean(v)), classes=dict(zip(CLASSES, np.mean(v, axis=0).tolist()))) for k, v in phases.items()},
                patients=len(patients), volumes=len(rows))


@torch.inference_mode()
def evaluate(model, cfg, role, output):
    if role not in ('train_labeled', 'val'):
        raise PermissionError('only current training L and validation evaluation authorized')
    access = Accessor(cfg['data'], 'mnms', 'Siemens', role)
    if len(access.patients) != (10 if role == 'train_labeled' else 23):
        raise ValueError('patient count changed')
    model.eval().requires_grad_(False)
    rows, count = [], 0
    for i, meta in enumerate(access.rows):
        item = access.read(i)
        x, truth = item['image'], item['label'].numpy()
        pred = np.concatenate([model(x[z:z+2, None].cuda()).argmax(1).cpu().numpy()
                               for z in range(0, len(x), 2)])
        row = dict(patient=meta['patient_id'], phase=meta['phase'], dice=dice(pred, truth).tolist())
        rows.append(row)
        append(output, row)
        count += len(x)
    return {**aggregate(rows), 'forward_slices': count}, rows


def existing_validation(cfg):
    root = Path(cfg['reference_run'])
    rows = []
    for line in (root/'private_validation.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['domain'] != 'Siemens':
            raise ValueError('reference validation contains another domain')
        rows.append(dict(patient=r['patient'], phase=r['phase'], dice=[r['metrics'][c]['Dice'] for c in CLASSES]))
    summary = aggregate(rows)
    original = json.loads((root/'VALIDATION.json').read_text())['Siemens']['primary_dice']
    if abs(summary['primary_dice']-original) > 1e-12 or summary['patients'] != 23:
        raise ValueError('existing reference validation does not reproduce its published aggregate')
    return {**summary, 'forward_slices': 0, 'reused': True}, rows


def compare(a, b):
    aa = {(r['patient'], r['phase']): np.array(r['dice']) for r in a}
    bb = {(r['patient'], r['phase']): np.array(r['dice']) for r in b}
    if aa.keys() != bb.keys():
        raise ValueError('comparison patients differ')
    differences = {}
    for (patient, phase), value in aa.items():
        differences.setdefault(patient, []).append(bb[patient, phase]-value)
    values = np.array([np.mean(v, axis=0) for v in differences.values()])
    per_patient = values.mean(1)
    return dict(delta_B_minus_A_pp=float(values.mean()*100), classes_pp=dict(zip(CLASSES, (values.mean(0)*100).tolist())),
                patients=len(values), increased=int((per_patient > 0).sum()), decreased=int((per_patient < 0).sum()),
                unchanged=int((per_patient == 0).sum()), median_patient_delta_pp=float(np.median(per_patient)*100))


def main(cfg):
    out = Path(cfg['output'])
    out.mkdir(exist_ok=False)
    started = time.time()
    trainer = None
    committed = 0
    start_step = 0
    ledger = PhysicalLedger(out/'physical_updates.jsonl')
    try:
        if NODE['steps'] != MAX_STEPS or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise PermissionError('frozen budget/device changed')
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        write(out/'STATUS.json', dict(status='PREFLIGHT', seed=163, max_updates=MAX_STEPS))
        bval, bval_rows = existing_validation(cfg)
        trainer = setup(NODE, cfg['data'], cfg['reference'], 'cuda', source_labeled_only=True)
        expected = options(MAX_STEPS, True)
        if any(trainer.options[k] != (0. if k == 'lambda_U' else v) for k, v in expected.items()):
            raise ValueError('configuration mismatch beyond U coefficient')
        # Match the historical first supervised objective before any optimizer call.
        loss, u, pending = trainer.losses()
        if u is not None or pending is not None or abs(float(loss.detach())-2.063466310157642) > 1e-5:
            raise ValueError('source initialization / first labeled objective mismatch')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainer.model.parameters()):
            raise ValueError('nonfinite preflight gradient')
        write(out/'PREFLIGHT.json', dict(status='PASS', optimizer_updates=0, first_labeled_loss=float(loss.detach()),
                                       expected_first_labeled_loss=2.063466310157642,
                                       L_image_exposure=2*trainer.provider.l_reads, U_reads=trainer.provider.u_reads,
                                       telemetry=trainer.telemetry, backwards=1))
        del loss
        trainer = None
        gc.collect()
        torch.cuda.empty_cache()
        trainer = setup(NODE, cfg['data'], cfg['reference'], 'cuda', source_labeled_only=True)
        identity = dict(dataset='mnms', seed=163, order=1, stage=0, domain='Siemens', arm='L_ONLY')
        prior_physical = cfg.get('prior_physical_updates', 0)
        physical_limit = cfg.get('max_physical_updates', MAX_STEPS)
        if cfg.get('resume_checkpoint'):
            restore(trainer, cfg['resume_checkpoint'], identity)
            committed = trainer.step
            start_step = trainer.step
        target = remaining_target(trainer.step, prior_physical, physical_limit)
        def limit(opt, args, kwargs):
            if prior_physical+trainer.physical_optimizer_updates >= physical_limit or trainer.step >= target:
                raise PermissionError('optimizer budget exhausted')
        trainer.optimizer.register_step_pre_hook(limit)
        save(trainer, out/'initial.pt', identity)
        while trainer.step < target:
            trainer.update(physical=ledger.append)
            committed = trainer.step
            if trainer.provider.u_reads != 0 or trainer.last['active_U']:
                raise PermissionError('L-only run read U')
            if trainer.step == 1 or trainer.step % 100 == 0:
                row = dict(status='RUNNING', step=trainer.step, physical_updates=trainer.physical_optimizer_updates,
                           labeled_loss=trainer.last['labeled_loss'], U_reads=trainer.provider.u_reads,
                           elapsed_seconds=time.time()-started)
                write(out/'STATUS.json', row)
                print(json.dumps(row), flush=True)
            if trainer.step % 2000 == 0:
                save(trainer, out/'latest.pt', identity)
        save(trainer, out/'final.pt', identity)
        model = seal(trainer, {**NODE, 'arm': 'L_ONLY', 'study': 'SIEMENS_LONLY_V1'}, out/'sealed.pt')
        costs = {**ledger.summary(trainer.step-start_step), 'scientific_step': trainer.step, 'resumed_from_step': start_step, 'telemetry': trainer.telemetry,
                 'L_batch_reads': trainer.provider.l_reads, 'U_batch_reads': trainer.provider.u_reads,
                 'L_image_exposure': 2*trainer.provider.l_reads, 'U_image_exposure': 2*trainer.provider.u_reads,
                 'engineering_optimizer_updates': 0, 'prior_attempt_physical_updates': prior_physical,
                 'cumulative_physical_updates': prior_physical+trainer.physical_optimizer_updates,
                 'physical_limit': physical_limit}
        if costs['cumulative_physical_updates'] > physical_limit or trainer.step != target:
            raise ValueError('physical update accounting mismatch')
        write(out/'COSTS.json', costs)
        write(out/'STATUS.json', dict(status='EVALUATING', step=trainer.step))
        del trainer
        trainer = None
        gc.collect()
        torch.cuda.empty_cache()
        atrain, atrain_rows = evaluate(model, cfg, 'train_labeled', out/'private_A_train.jsonl')
        aval, aval_rows = evaluate(model, cfg, 'val', out/'private_A_val.jsonl')
        del model
        gc.collect()
        torch.cuda.empty_cache()
        payload = torch.load(Path(cfg['reference_run'])/'sealed.pt', map_location='cpu', weights_only=False)
        if not payload['sealed'] or any(payload[k] != NODE[k] for k in NODE) or payload['updates'] != MAX_STEPS:
            raise ValueError('incorrect existing SSL source')
        native = build(cfg['reference'], torch.device('cuda'), 163, input_channels=1, num_classes=4, domain='Siemens')
        native.load_state_dict(payload['native'], strict=True)
        model = Deployment(Parent(native, 163, {'kind': 'existing_source'}, source=True), payload['transform'].cuda()).cuda().eval().requires_grad_(False)
        btrain, btrain_rows = evaluate(model, cfg, 'train_labeled', out/'private_B_train.jsonl')
        result = dict(status='COMPLETE', seed=163, A={'train': atrain, 'val': aval}, B={'train': btrain, 'val': bval},
                      paired_train=compare(atrain_rows, btrain_rows), paired_val=compare(aval_rows, bval_rows),
                      new_optimizer_updates=costs['cumulative_physical_updates'], scientific_step=target, engineering_optimizer_updates=0,
                      evaluation_forward_slices=atrain['forward_slices']+aval['forward_slices']+btrain['forward_slices'],
                      independent_test_access=False, checkpoint_selection='fixed final', elapsed_seconds=time.time()-started,
                      interpretation='One selected development seed; no significance or independent-test claim. Historical B used different GPU/backend during training.')
        write(out/'RESULTS.json', result)
        lines = ['# Siemens L-only versus existing SSL, seed163', '',
                 '|Arm|Train Dice (%)|Validation Dice (%)|Train-val gap (pp)|', '|---|---:|---:|---:|']
        for arm in ('A', 'B'):
            tr, va = result[arm]['train']['primary_dice'], result[arm]['val']['primary_dice']
            lines.append(f'|{arm}|{100*tr:.4f}|{100*va:.4f}|{100*(tr-va):.4f}|')
        delta = result['paired_val']['delta_B_minus_A_pp']
        verdict = 'higher' if delta > 1 else ('lower' if delta < -1 else 'within the descriptive +/-1pp band')
        lines += ['', f'Existing SSL validation is {verdict} versus L-only; B-A = {delta:.4f}pp.',
                  'The band is descriptive, not a significance threshold. Detailed classes, ED/ES and paired patient counts are in RESULTS.json.',
                  'Only lambda_U was changed to zero; labeled sampling, LCTX schedule, hierarchy, initialization and optimizer settings were retained.',
                  'B validation was reused; A train/val and B train were evaluated once. No test data or U ground truth was accessed.',
                  'The historical B training crossed GPU/software environments, so this is not a bitwise matched hardware comparison.',
                  f'A scientific steps: {target}; cumulative physical optimizer calls: {costs["cumulative_physical_updates"]}. Prior failed calls remain charged.',
                  'Engineering recovery is authorized within the agreed budget; no second seed or tuning. GitHub delivery remains pending local verification.']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
        write(out/'STATUS.json', dict(status='COMPLETE', step=target, results='RESULTS.json'))
        write(out/'EXIT.json', dict(exit_code=0))
        print(json.dumps(dict(status='COMPLETE', validation_delta_pp=delta)), flush=True)
    except BaseException as exc:
        step = committed if trainer is None else trainer.step
        write(out/'FAILURE.json', dict(error=str(exc), traceback=traceback.format_exc(), step=step, resumed_from_step=start_step,
                                     **ledger.summary(max(0,step-start_step))))
        write(out/'STATUS.json', dict(status='FAILED', step=step, error=str(exc), retry=False))
        write(out/'EXIT.json', dict(exit_code=1))
        raise


if __name__ == '__main__':
    main(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
