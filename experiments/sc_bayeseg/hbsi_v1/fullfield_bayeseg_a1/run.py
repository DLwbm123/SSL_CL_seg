"""Frozen A1 source-to-target supervised qualification; private artifacts stay on NAS."""
import csv
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from ..budgets import ResourceLedger
from ..config import Config
from ..data_adapter import confined, restore_probability
from ..paired_semantic_v19.run import data
from ..real_diagnostic import write
from ...sslcl_v1.prostate_metrics import binary_phase_metrics
from .model import FullFieldBayeSeg


def batch(d, step, seed):
    b = d.batch('L', 1, torch.Generator().manual_seed(seed + 2 * step))
    return b['x'].cuda(), b['observations']['y_obs'].long().argmax(1).cuda()


def records(p, domain, split):
    rows = [r for r in csv.DictReader((Path(p['data_root']) / 'manifests/training/lcrseg_v1_seed0.csv').open())
            if r['dataset'] == 'prostate' and r['site_or_vendor'] == domain
            and r['primary_20pct_split'] == split]
    assert len(rows) == 4
    return rows


def evaluate(p, model, domain, split, out):
    out.mkdir(exist_ok=False)
    model.eval()
    summaries = []
    for i, row in enumerate(records(p, domain, split)):
        with h5py.File(confined(Path(p['data_root']) / 'h5/v1', row['image_h5_relpath']), 'r') as f:
            image = torch.from_numpy(f['image'][...].copy())[:, None].double()
            spacing = json.loads(f.attrs['processed_spacing'])
        with torch.no_grad():
            probability = torch.cat([
                model.predict(F.interpolate(v[None], (128, 128), mode='bilinear', align_corners=False).cuda(),
                              seed=1101 + 10000 * i + 100 * j, draws=4).cpu()
                for j, v in enumerate(image)
            ])
        torch.save(probability, out / f'PRIVATE_PREDICTION_{i}.pt')
        with h5py.File(confined(Path(p['data_root']) / 'h5/v1', row['label_h5_relpath']), 'r') as f:
            truth = f['label'][...]
        metrics = binary_phase_metrics(restore_probability(probability, image.shape[-2:]), truth, spacing)
        write(out / f'PRIVATE_METRICS_{i}.json', dict(patient=row['patient_id'], metrics=metrics))
        summaries.append(metrics)
    classes = [m['classes'][0] for m in summaries]
    result = dict(domain=domain, split=split, patients=4, draws=4, complete=True,
                  predictor='fullfield_local_posterior_mean',
                  Dice=float(np.mean([v['Dice'] for v in classes])),
                  precision=float(np.mean([v['TP'] / (v['TP'] + v['FP']) if v['TP'] + v['FP'] else 0
                                           for v in classes])),
                  recall=float(np.mean([v['TP'] / (v['TP'] + v['FN']) if v['TP'] + v['FN'] else 0
                                        for v in classes])),
                  HD95_penalized_mm=float(np.mean([v['HD95_penalized_mm'] for v in classes])),
                  NLL=float(np.mean([v['NLL'] for v in summaries])),
                  Brier=float(np.mean([v['Brier'] for v in summaries])),
                  empty_predictions=sum(v['empty_prediction_failure'] for v in classes))
    write(out / 'SUMMARY.json', result)
    return result


def train(model, optimizer, d, seed, steps, name, ledger, seconds):
    model.train()
    with ledger.attempt(name, seconds=seconds, updates=steps) as out:
        def save(step):
            temporary = out / 'checkpoint.tmp'
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step), temporary)
            temporary.replace(out / 'PRIVATE_CHECKPOINT.pt')
        save(0)
        started = time.monotonic()
        with (out / 'PRIVATE_TRAIN_LOG.jsonl').open('x') as log:
            for step in range(steps):
                ledger.check()
                x, y = batch(d, step, seed)
                generator = torch.Generator(device='cuda').manual_seed(54000000 + (seed - 163000) * 10000 + step)
                optimizer.zero_grad(set_to_none=True)
                loss, parts = model.loss(x, y, generator)
                loss.backward()
                assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all()
                                                   for v in model.parameters())
                optimizer.step()
                log.write(json.dumps(dict(step=step + 1, loss=float(loss.detach()), **parts)) + '\n')
                log.flush()
                if (step + 1) % 400 == 0:
                    save(step + 1)
        write(out / 'RECEIPT.json', dict(status='COMPLETE', steps=steps,
                                         seconds=time.monotonic() - started,
                                         formal_backward_calls=steps))
    return out / 'PRIVATE_CHECKPOINT.pt'


def main():
    root = Path(os.environ['RUN_ROOT'])
    protocol = json.loads((root / 'PRIVATE_PROTOCOL.json').read_text())
    assert protocol['purpose'] == 'FULLFIELD_BAYESEG_A1'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = replace(Config(**protocol['config']), seed=20261023)
    ledger = ResourceLedger(root / 'results', max_seconds=12600, max_updates=2400,
                            max_bytes=3_000_000_000)

    def status(stage, **extra):
        temporary = root / 'STATUS.tmp'
        temporary.write_text(json.dumps(dict(stage=stage, time=time.time(), **extra), indent=2))
        temporary.replace(root / 'STATUS.json')

    try:
        source = data(protocol, cfg, 'RUNMC', 'L')
        target = data(protocol, cfg, 'BMC', 'L')
        assert source.counts == {'L': 80, 'U': 0} and target.counts == {'L': 134, 'U': 0}
        assert cfg.height == cfg.width == 128 and cfg.dtype == 'float64' and cfg.outer_lr == 3e-4
        assert all(len(records(protocol, domain, 'val')) == 4 for domain in ('RUNMC', 'BMC'))
        status('PREFLIGHT')
        with ledger.attempt('preflight', seconds=300, updates=0) as out:
            torch.manual_seed(20261023)
            model = FullFieldBayeSeg().double().cuda()
            probes = {}
            initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
            for domain, d, seed in [('RUNMC', source, 163000), ('BMC', target, 263000)]:
                x, y = batch(d, 0, seed)
                generator = torch.Generator(device='cuda').manual_seed(54000000)
                loss, parts = model.loss(x, y, generator)
                loss.backward()
                assert torch.isfinite(loss)
                assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
                           for p in model.parameters())
                prediction = model.predict(x, seed=1101)
                torch.testing.assert_close(prediction.sum(1), torch.ones_like(prediction[:, 0]))
                probes[domain] = dict(loss=float(loss.detach()), **parts)
                model.zero_grad(set_to_none=True)
            assert all(torch.equal(v, model.state_dict()[k]) for k, v in initial.items())
            checkpoint = out / 'PRIVATE_PREFLIGHT.pt'
            torch.save(model.state_dict(), checkpoint)
            restored = FullFieldBayeSeg().double().cuda()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))
            assert all(torch.equal(v, restored.state_dict()[k]) for k, v in initial.items())
            write(root / 'PREFLIGHT.json', dict(status='PASS', real_source_target_backward=2,
                                                 optimizer_updates=0, finite_nonzero_all_parameter_grads=True,
                                                 checkpoint_roundtrip=True, data_counts=dict(RUNMC=80, BMC=134),
                                                 probes=probes, parameters=sum(p.numel() for p in model.parameters())))
            del restored
        cells = {}
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)
        status('TRAIN_SOURCE')
        source_checkpoint = train(model, optimizer, source, 163000, 1600, 'source', ledger, 4000)
        for domain in ('RUNMC', 'BMC'):
            status('EVALUATE_SOURCE', domain=domain)
            with ledger.attempt('source_' + domain, seconds=900) as out:
                cells['source_' + domain] = evaluate(protocol, model, domain, 'val', out / 'validation')
        sealed = torch.load(source_checkpoint, map_location='cpu', weights_only=True)
        assert sealed['step'] == 1600
        model.load_state_dict(sealed['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)
        assert not optimizer.state
        status('TRAIN_TARGET')
        train(model, optimizer, target, 263000, 800, 'target', ledger, 2400)
        status('EVALUATE_TARGET_TRAIN')
        with ledger.attempt('target_train', seconds=900) as out:
            cells['target_BMC_train'] = evaluate(protocol, model, 'BMC', 'train_labeled', out / 'validation')
        for domain in ('RUNMC', 'BMC'):
            status('EVALUATE_TARGET', domain=domain)
            with ledger.attempt('target_' + domain, seconds=900) as out:
                cells['target_' + domain] = evaluate(protocol, model, domain, 'val', out / 'validation')
        write(root / 'RESULTS.json', dict(cells=cells, formal_updates=2400,
                                          source_updates=1600, target_updates=800,
                                          preflight_updates=0, training_seed=20261023))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
