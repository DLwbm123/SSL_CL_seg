"""Unlabeled BMC generative adaptation with and without shape-posterior updates."""
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from sc_bayeseg.model import normal_log_prob, sar, student_log_prob
from ..budgets import ResourceLedger
from ..config import Config
from ..fullfield_bayeseg_a1.run import batch, evaluate
from ..paired_semantic_v19.run import data
from ..real_diagnostic import write
from ..label_shape_bridge_f1.model import ShapeBridge


def generative(model, image, generator):
    base = model.base
    output = base.unet(image)
    d = base.draw(image, generator, output)
    observed = student_log_prob(image - d['shape'] - d['appearance'], .1)
    appearance_prior = normal_log_prob(d['appearance'], torch.zeros_like(d['appearance']),
                                       torch.full_like(d['appearance'], math.log(.5)))
    shape_prior = student_log_prob(
        sar(d['shape'] - base.shape_mean(d['probability']), .9), .2) + base.sar_logdet
    segmentation_prior = student_log_prob(sar(d['z'], .9), 1.) + base.sar_logdet
    bound = observed + appearance_prior + shape_prior + segmentation_prior - d['logq']
    return -bound.mean() / (base.size * base.size), d


def main():
    root = Path(os.environ['RUN_ROOT'])
    p = json.loads((root / 'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose'] == 'TARGET_GENERATIVE_ADAPT_G1'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = replace(Config(**p['config']), seed=20261023)
    ledger = ResourceLedger(root / 'results', max_seconds=36000, max_updates=1600,
                            max_bytes=3_000_000_000)

    def status(stage, **extra):
        tmp = root / 'STATUS.tmp'
        tmp.write_text(json.dumps(dict(stage=stage, time=time.time(), **extra), indent=2))
        tmp.replace(root / 'STATUS.json')

    f1 = Path(p['f1_root'])
    entry = torch.load(f1 / 'results/Anchor/PRIVATE_CHECKPOINT_1600.pt',
                       map_location='cpu', weights_only=True)
    assert entry['step'] == 1600
    initial = entry['model']

    def fresh(adapt_shape):
        model = ShapeBridge(True).double().cuda()
        model.load_state_dict(initial)
        model.base.unet.requires_grad_(False)
        model.head.requires_grad_(False)
        if not adapt_shape:
            model.base.shape_posterior.requires_grad_(False)
        return model

    def frozen_error(model):
        current = model.state_dict()
        return max(float((current[k].detach().cpu() - v).abs().max())
                   for k, v in initial.items()
                   if k.startswith(('base.unet.', 'head.')))

    def score(arm, model, domain, split='val'):
        name = f'{arm}_{domain}' + ('' if split == 'val' else '_train')
        status('EVALUATE', arm=arm, domain=domain, split=split)
        with ledger.attempt(name, seconds=900) as out:
            result = evaluate(p, model, domain, split, out / 'validation')
            result['predictor'] = 'fixed_head_shape_posterior_mean'
            result['draws'] = 1
            (out / 'validation/SUMMARY.json').write_text(json.dumps(result, indent=2))
        return result

    try:
        assert json.loads((f1 / 'EXIT.json').read_text())['exit_code'] == 0
        target = data(p, cfg, 'BMC', 'U')
        source = data(p, cfg, 'RUNMC', 'L')
        assert target.counts['U'] > 0 and source.counts == {'L': 80, 'U': 0}
        assert cfg.height == cfg.width == 128 and cfg.dtype == 'float64' and cfg.outer_lr == 3e-4
        status('PREFLIGHT')
        with ledger.attempt('preflight', seconds=300, updates=0) as out:
            a = fresh(False); s = fresh(True)
            assert all(torch.equal(a.state_dict()[k], s.state_dict()[k])
                       for k in a.state_dict())
            assert frozen_error(a) == frozen_error(s) == 0
            a.train(); s.train()
            image = target.batch('U', 1, torch.Generator().manual_seed(173000))['x'].cuda()
            assert image.shape == (1, 1, 128, 128)
            av, ad = generative(a, image, torch.Generator(device='cuda').manual_seed(64000000))
            sv, sd = generative(s, image, torch.Generator(device='cuda').manual_seed(64000000))
            torch.testing.assert_close(av, sv, rtol=0, atol=0)
            assert torch.isfinite(av)
            for key in ('shape', 'appearance', 'shape_logsd', 'appearance_logsd'):
                torch.testing.assert_close(ad[key], sd[key], rtol=0, atol=0)
            av.backward(); sv.backward()
            assert all(v.grad is None for v in a.base.shape_posterior.parameters())
            assert any(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.norm() > 0
                       for v in s.base.shape_posterior.parameters())
            assert all(v.grad is None for v in s.base.unet.parameters())
            assert all(v.grad is None for v in s.head.parameters())
            x, y = batch(source, 0, 163000)
            _, old_v, _, _ = s.base.objectives(x, y,
                torch.Generator(device='cuda').manual_seed(64000001))
            new_v, _ = generative(s, x, torch.Generator(device='cuda').manual_seed(64000001))
            torch.testing.assert_close(new_v, old_v, rtol=0, atol=0)
            with torch.no_grad():
                torch.testing.assert_close(a.predict(image, 0), s.predict(image, 0), rtol=0, atol=0)
            assert frozen_error(a) == frozen_error(s) == 0
            write(root / 'PREFLIGHT.json', dict(status='PASS', production_updates=0,
                complete_generator_exact=True, target_labels_not_loaded=True,
                paired_initial_state_equal=True, A_only_shape_gradient_zero=True,
                S_plus_A_shape_gradient_nonzero=True, frozen_predictor_exact=True))
            del a, s

        cells = {}
        for arm, adapt_shape in (('A_only', False), ('S_plus_A', True)):
            model = fresh(adapt_shape)
            model.train()
            optimizer = torch.optim.Adam([v for v in model.parameters() if v.requires_grad],
                                         lr=cfg.outer_lr)
            status('TRAIN', arm=arm)
            with ledger.attempt(arm, seconds=12000, updates=800) as out:
                with (out / 'PRIVATE_TRAIN_LOG.jsonl').open('x') as log:
                    for step in range(800):
                        ledger.check()
                        image = target.batch('U', 1,
                            torch.Generator().manual_seed(173000 + 2 * step))['x'].cuda()
                        generator = torch.Generator(device='cuda').manual_seed(64000000 + step)
                        optimizer.zero_grad(set_to_none=True)
                        variational, d = generative(model, image, generator)
                        loss = .01 * variational
                        assert torch.isfinite(loss)
                        loss.backward()
                        assert all(v.grad is None or torch.isfinite(v.grad).all()
                                   for v in model.parameters())
                        assert all(v.grad is None for v in model.base.unet.parameters())
                        assert all(v.grad is None for v in model.head.parameters())
                        if not adapt_shape:
                            assert all(v.grad is None for v in model.base.shape_posterior.parameters())
                        optimizer.step()
                        assert frozen_error(model) == 0
                        row = dict(step=step + 1, variational=float(variational.detach()),
                                   reconstruction_mse=float((image - d['shape'] - d['appearance'])
                                                            .detach().square().mean()),
                                   shape_logsd=float(d['shape_logsd'].detach().mean()),
                                   appearance_logsd=float(d['appearance_logsd'].detach().mean()))
                        log.write(json.dumps(row) + '\n'); log.flush()
                        if step + 1 in (400, 800):
                            tmp = out / 'checkpoint.tmp'
                            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                            step=step + 1), tmp)
                            tmp.replace(out / f'PRIVATE_CHECKPOINT_{step + 1}.pt')
                write(out / 'RECEIPT.json', dict(status='COMPLETE', steps=800,
                                                 frozen_predictor_max_error=0))
            for domain in ('RUNMC', 'BMC'):
                cells[f'{arm}_{domain}'] = score(arm, model, domain)
            cells[f'{arm}_RUNMC_train'] = score(arm, model, 'RUNMC', 'train_labeled')
            del model, optimizer
        write(root / 'RESULTS.json', dict(cells=cells, formal_updates=1600,
            target_U_unlabeled_only=True, BMC_L_training=False, source_L_training=False))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
