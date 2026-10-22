"""Explicitly scoped real-data feasibility study; never issues a learning PASS.

Uses the existing training engine and density. MCMC and formal-source admission
are deliberately separate from this new, user-authorized diagnostic protocol.
"""
import csv
import io
import json
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .budgets import Counters, ResourceLedger
from .config import Config
from .data_adapter import CurrentTrainingData, confined, restore_probability
from .local_inference import Refiner
from .objective import SequentialObjective
from .predictive import Predictor, InferenceBudget
from .qualification import code_fingerprint, fingerprint
from .runner import storage_check
from .stream_state import StreamState
from .synthetic import to_json
from .training import run_stage


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(to_json(value), f, indent=2, allow_nan=False)


def validate_protocol(p):
    required = dict(schema=1, purpose='REAL_TASK_FEASIBILITY', dataset='prostate',
                    domain='RUNMC', arms=['L', 'LU'], steps=400, batch_size=1,
                    use_mcmc_teacher=False, permit_formal_promotion=False,
                    device='cuda', data_seed=163000, authorization='USER_20261022_RUNMC_PILOT',
                    checkpoint_stride=100, preflight_id='preflight02')
    if any(p.get(k) != v for k, v in required.items()):
        raise PermissionError('not the authorized single-domain diagnostic protocol')
    if not p.get('data_version') or not p.get('data_root') or p.get('checkpoint'):
        raise PermissionError('fresh source and explicit data version required')
    cfg = Config(**p['config'])
    if (cfg.channels, cfg.classes, cfg.height, cfg.width, cfg.inner_steps) != (1, 2, 128, 128, 3):
        raise PermissionError('diagnostic configuration mismatch')
    return cfg


def prepare_data(p, cfg, arm):
    if arm not in p['arms']:
        raise ValueError('unknown arm')
    prepared = CurrentTrainingData.inspect_manifest(p['data_root'], 'prostate', 'RUNMC',
        p['data_version'], 0, roles=('L',) if arm == 'L' else ('L', 'U'))
    data = CurrentTrainingData(p['data_root'], 'prostate', 'RUNMC', arm, cfg,
                               data_version=p['data_version'], prepared=prepared)
    return data


def audit(p, cfg):
    import h5py
    import numpy as np
    root = Path(p['data_root'])
    records = [r for r in csv.DictReader((root/'manifests/training/lcrseg_v1_seed0.csv').open())
               if r['dataset'] == 'prostate' and r['site_or_vendor'] == 'RUNMC']
    patients = {}; paths = set(); summary = {}; val_slices = 0
    for role, expected in [('train_labeled', 4), ('train_unlabeled', 14), ('val', 4), ('test', 8)]:
        rows = [r for r in records if r['primary_20pct_split'] == role]
        ids = {r['patient_id'] for r in rows}
        if len(ids) != expected or len(rows) != expected:
            raise ValueError('frozen patient/volume count mismatch')
        if any(ids & previous for previous in patients.values()):
            raise ValueError('patient role overlap')
        patients[role] = ids
        summary[role] = dict(patients=len(ids), slices=0)
        if role == 'test':
            continue  # Metadata count only; no test path resolution or file open.
        for r in rows:
            path = confined(root/'h5/v1', r['image_h5_relpath'])
            if path in paths:
                raise ValueError('duplicate image path')
            paths.add(path)
            with h5py.File(path, 'r') as f:
                shape = f['image'].shape
                if len(shape) != 3 or shape[1:] != (256, 256):
                    raise ValueError('unexpected input geometry')
                spacing = json.loads(f.attrs['processed_spacing'])
                if len(spacing) != 3 or min(spacing) <= 0:
                    raise ValueError('invalid spacing')
                if role == 'train_unlabeled' and (r['label_h5_relpath'] or 'label' in f):
                    raise PermissionError('U label capability')
                if role != 'val' and not np.isfinite(f['image'][...]).all():
                    raise ValueError('nonfinite training input')
            summary[role]['slices'] += shape[0]
            if role == 'val':
                val_slices += shape[0]
            if role == 'train_labeled':
                with h5py.File(confined(root/'h5/v1', r['label_h5_relpath']), 'r') as f:
                    y = f['label'][...]
                    if y.shape != shape or not np.isin(y, [0, 1]).all():
                        raise ValueError('binary label/geometry mismatch')
    return dict(roles=summary, validation_slices=val_slices, test_files_opened=0,
                validation_labels_opened=0, hidden_U_labels_opened=0,
                independence='frozen patient IDs; no clinical cross-site identity claim')


def preflight(root, p, cfg, ledger, counter):
    from copy import deepcopy
    from .tests import rejected
    from .tests import run_tests
    from .revision_checks import run_checks
    for field, value in [('purpose','FORMAL_TRAINING'), ('domain','BMC'),
                         ('use_mcmc_teacher',True), ('permit_formal_promotion',True), ('steps',401)]:
        invalid=deepcopy(p);invalid[field]=value
        rejected(lambda:validate_protocol(invalid),PermissionError)
    # Relevant boundary, budget, evidence and resume regressions, no synthetic performance gate.
    checks = run_tests(ledger, 'boundary02', ['T16', 'T17'])
    checks += run_checks(ledger, 'revision02', ['R01', 'R02', 'R04'])
    # Exercise the actual CUDA + sparse-retention + independent data stream resume path.
    from functools import partial
    from unittest.mock import patch
    from . import revision_checks
    original_begin = StreamState.begin_task
    def gpu_begin(state, *args, **kwargs):
        original_begin(state, *args, **kwargs)
        state.to_device('cuda')
    with patch.object(StreamState, 'begin_task', gpu_begin), patch.object(revision_checks, 'run_stage',
            partial(run_stage, data_seed=p['data_seed'], checkpoint_stride=2)):
        checks += run_checks(ledger, 'gpu_resume02', ['R04'])
    retained = root/'gpu_resume02_R04/budget/full_attempt001'
    assert not (retained/'ACTIVE_STEP_000001.pt').exists()
    assert (retained/'ACTIVE_STEP_000002.pt').is_file() and (retained/'ACTIVE_STEP_000004.pt').is_file()
    if not all(r['status'] == 'PASS' for r in checks):
        raise RuntimeError('required interface regression failed')
    with ledger.attempt(p['preflight_id'], seconds=300, updates=0) as out:
        data_audit = audit(p, cfg)
        data = {arm: prepare_data(p, cfg, arm) for arm in p['arms']}
        assert data['L'].counts['U'] == 0 and data['LU'].counts['U'] > 0
        assert set(data['L'].evidence['events']) < set(data['LU'].evidence['events'])
        # L schedules must match exactly, regardless of U/model RNG consumption.
        for step in (0, 1, 399):
            aa = data['L'].batch('L', 1, torch.Generator().manual_seed(p['data_seed'] + 2*step))
            bb = data['LU'].batch('L', 1, torch.Generator().manual_seed(p['data_seed'] + 2*step))
            assert torch.equal(aa['x'], bb['x'])
            assert torch.equal(aa['observations']['y_obs'], bb['observations']['y_obs'])
        timings = {}; checkpoint_bytes = 0
        for arm in p['arms']:
            state = StreamState(cfg)
            state.begin_task(arm, data[arm].manifest_hash, data[arm].evidence)
            L = data[arm].batch('L', 1, torch.Generator().manual_seed(p['data_seed']))
            U = data[arm].batch('U', 1, torch.Generator().manual_seed(p['data_seed']+1))
            objective = SequentialObjective(state, Refiner(state.model, state.phi, counter))
            # Matched RNG and real minibatch: CPU/CUDA loss and all gradient vectors.
            cpu_rng = state.generator.get_state()
            cpu_loss, _ = objective.estimate(L, U, data[arm].counts, state.generator)
            cpu_loss.backward()
            cpu_grads = [v.grad.detach().clone() for group in state.optimizer.param_groups for v in group['params']]
            state.optimizer.zero_grad();state.generator.set_state(cpu_rng)
            state.to_device(p['device'])
            for batch in (L,U):
                if batch is not None:
                    batch['x'] = batch['x'].to(p['device'])
                    if 'observations' in batch:
                        batch['observations']={k:v.to(p['device']) for k,v in batch['observations'].items()}
            torch.cuda.synchronize()
            start = time.monotonic()
            loss, parts = objective.estimate(L, U, data[arm].counts, state.generator)
            loss.backward()  # No optimizer step; real-data smoke and cost only.
            torch.cuda.synchronize()
            assert all(v.grad is not None and torch.isfinite(v.grad).all()
                       for group in state.optimizer.param_groups for v in group['params'])
            timings[arm] = time.monotonic() - start
            torch.testing.assert_close(loss.cpu(), cpu_loss, atol=1e-5, rtol=1e-8)
            gpu_grads=[v.grad.cpu() for group in state.optimizer.param_groups for v in group['params']]
            for cpu,gpu in zip(cpu_grads,gpu_grads):
                torch.testing.assert_close(cpu,gpu,atol=1e-5,rtol=1e-6)
            buffer = io.BytesIO(); torch.save(state.payload(), buffer)
            # Adam's two moment buffers will be materialized at the first real update.
            moments = sum(2*v.numel()*v.element_size() for g in state.optimizer.param_groups for v in g['params'])
            checkpoint_bytes = max(checkpoint_bytes, buffer.tell() + moments + 32768)
            write(out/f'{arm}_BACKWARD.json', dict(seconds=timings[arm], parts=parts, optimizer_updates=0))
        state = StreamState(cfg)
        state.begin_task('initial', data['LU'].manifest_hash, data['LU'].evidence)
        state.finalize_task()
        state.save(out/'PRIVATE_INITIAL.pt')
        state.to_device(p['device'])
        image = data['L'].batch('L', 1, torch.Generator().manual_seed(p['data_seed']))['x'].to(p['device'])
        prediction_seconds = {}
        for K in (0, 3):
            torch.cuda.synchronize();start = time.monotonic()
            result = Predictor(counter).predict_image_only(image, state, InferenceBudget(K=K), return_repeats=True)
            torch.cuda.synchronize();prediction_seconds[str(K)] = time.monotonic() - start
            assert torch.isfinite(result['repeat_probabilities']).all()
            assert torch.equal(result['probability'], result['repeat_probabilities'][0])
        # Five full validation evaluations: initial K0, two final arms each K0/K3.
        predicted_compute = 400*sum(timings.values()) + data_audit['validation_slices']*(
            3*prediction_seconds['0'] + 2*prediction_seconds['3'])
        predicted_bytes = 20*checkpoint_bytes + 5*data_audit['validation_slices']*3*2*128*128*8
        result = dict(state='COST_REVIEW_REQUIRED', scientific_state='NOT_QUALIFIED',
            config=cfg.to_dict(), protocol_fingerprint=fingerprint(p), code_fingerprint=code_fingerprint(),
            evidence={a:d.evidence['fingerprint'] for a,d in data.items()}, data_audit=data_audit,
            train_step_seconds=timings, inference_slice_seconds=prediction_seconds,
            estimated_compute_seconds=predicted_compute, estimated_bytes=predicted_bytes,
            checkpoint_upper_estimate=checkpoint_bytes, initial_checkpoint=str(out/'PRIVATE_INITIAL.pt'),
            real_optimizer_updates=0, tests=checks)
        write(root/'PREFLIGHT02.json', result)


def require_ready(root, p, cfg):
    ready = json.loads((root/'READY.json').read_text())
    pre = json.loads((root/'PREFLIGHT02.json').read_text())
    if (ready.get('state') != 'REAL_DIAGNOSTIC_READY' or ready.get('preflight_fingerprint') != fingerprint(pre)
            or pre['protocol_fingerprint'] != fingerprint(p) or fingerprint(pre['config']) != fingerprint(cfg.to_dict())
            or pre['code_fingerprint'] != code_fingerprint()):
        raise PermissionError('missing or mismatched real diagnostic admission')
    return pre


def train(root, p, cfg, ledger, counter, arm):
    pre = require_ready(root, p, cfg)
    data = prepare_data(p, cfg, arm)
    if pre['evidence'][arm] != data.evidence['fingerprint']:
        raise PermissionError('training evidence changed after audit')
    state = StreamState(cfg)
    state.begin_task(arm, data.manifest_hash, data.evidence)
    state.to_device(p['device'])
    run_stage(state, data, ledger, counter, arm, 400, p['training_seconds_per_arm'],
              fingerprint(p), batch_size=1, data_seed=p['data_seed'], checkpoint_stride=p['checkpoint_stride'])


def evaluate(root, p, cfg, ledger, counter, arm, K):
    import h5py
    import numpy as np
    from sc_bayeseg.sslcl_v1.prostate_metrics import binary_phase_metrics
    pre = require_ready(root, p, cfg)
    checkpoint = pre['initial_checkpoint'] if arm == 'initial' else root/f'{arm}_attempt001/PRIVATE_SEALED_STATE.pt'
    state = StreamState.load(checkpoint).to_device(p['device'])
    if state.active is not None or state.config != cfg:
        raise PermissionError('evaluation requires matching sealed state')
    data_root = Path(p['data_root'])
    records = [r for r in csv.DictReader((data_root/'manifests/training/lcrseg_v1_seed0.csv').open())
               if r['dataset'] == 'prostate' and r['site_or_vendor'] == 'RUNMC' and r['primary_20pct_split'] == 'val']
    rows = []
    with ledger.attempt(f'eval_{arm}_K{K}', seconds=p['evaluation_seconds_per_cell']) as out:
        for i,r in enumerate(records):
            ledger.check()
            with h5py.File(confined(data_root/'h5/v1', r['image_h5_relpath']), 'r') as f:
                x = torch.from_numpy(f['image'][...].copy())[:,None].to(getattr(torch,cfg.dtype))
                spacing = json.loads(f.attrs['processed_spacing'])
            small = F.interpolate(x, (128,128), mode='bilinear', align_corners=False).to(p['device'])
            prediction = Predictor(counter).predict_image_only(small, state,
                InferenceBudget(K=K), return_repeats=True)
            probabilities = prediction.pop('repeat_probabilities').cpu()
            prediction.pop('probability')
            torch.save(dict(probability=probabilities, diagnostics=to_json(prediction)), out/f'PRIVATE_PREDICTION_{i}.pt')
            # Validation GT is opened only after the image-only predictions are sealed.
            with h5py.File(confined(data_root/'h5/v1', r['label_h5_relpath']), 'r') as f:
                y = f['label'][...]
            metrics = [binary_phase_metrics(restore_probability(q, x.shape[-2:]),y,spacing) for q in probabilities]
            rows.append(dict(patient=r['patient_id'], metrics=metrics, inference=to_json(prediction)))
            write(out/f'PRIVATE_METRICS_{i}.json', rows[-1])
            counter.add('validation_images', len(x))
        summary = []
        for repeat in range(3):
            mm=[r['metrics'][repeat] for r in rows]
            cells=[m['classes'][0] for m in mm]
            summary.append(dict(Dice=float(np.mean([c['Dice'] for c in cells])),
                NLL=float(np.mean([m['NLL'] for m in mm])), Brier=float(np.mean([m['Brier'] for m in mm])),
                HD95_penalized_mm=float(np.mean([c['HD95_penalized_mm'] for c in cells])),
                empty_predictions=sum(c['empty_prediction_failure'] for c in cells)))
        write(out/'SUMMARY.json', dict(arm=arm,K=K,patients=len(rows),repeats=summary,
            scientific_state='DIAGNOSTIC_ONLY',complete=True,teacher_used=False,
            inference_unqualified=sum(r['inference']['status']=='INFERENCE_UNQUALIFIED' for r in rows)))


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['preflight','train','evaluate'])
    parser.add_argument('--arm',choices=['L','LU','initial'])
    parser.add_argument('--K',type=int,choices=[0,3],default=3)
    args=parser.parse_args()
    root=Path(os.environ['RUN_ROOT'])/'results'
    storage_check(root)
    p=json.loads((root.parent/'PRIVATE_PROTOCOL.json').read_text());cfg=validate_protocol(p)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    counter=Counters();ledger=ResourceLedger(root)
    start=time.time()
    try:
        if args.command=='preflight':preflight(root,p,cfg,ledger,counter)
        elif args.command=='train':train(root,p,cfg,ledger,counter,args.arm)
        else:evaluate(root,p,cfg,ledger,counter,args.arm,args.K)
    finally:
        write(root/f'COUNTERS_{args.command}_{args.arm}_{args.K}_{time.time_ns()}.json',
              dict(seconds=time.time()-start,counters=counter.export()))


if __name__=='__main__':
    main()
