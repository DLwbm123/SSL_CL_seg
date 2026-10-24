"""Fixed source-only shape-posterior fusion comparison; private outputs stay on NAS."""
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..budgets import ResourceLedger
from ..config import Config
from ..feedback_c1.model import FeedbackModel
from ..fullfield_bayeseg_a1.run import batch, evaluate
from ..paired_semantic_v19.run import data
from ..real_diagnostic import write
from .model import Fusion, ResidualHead


def main():
    root = Path(os.environ['RUN_ROOT'])
    p = json.loads((root / 'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose'] == 'SHAPE_POSTERIOR_FUSION_E1'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = replace(Config(**p['config']), seed=20261023)
    ledger = ResourceLedger(root / 'results', max_seconds=36000, max_updates=3200,
                            max_bytes=3_000_000_000)

    def status(stage, **extra):
        tmp = root / 'STATUS.tmp'
        tmp.write_text(json.dumps(dict(stage=stage, time=time.time(), **extra), indent=2))
        tmp.replace(root / 'STATUS.json')

    def frozen():
        model = FeedbackModel(cut=True).double().cuda()
        state = torch.load(Path(p['c1_root']) / 'results/Cut_source/PRIVATE_CHECKPOINT.pt',
                           map_location='cpu', weights_only=True)
        assert state['step'] == 1600
        model.load_state_dict(state['model'])
        model.requires_grad_(False)
        model.eval()
        return model

    def head():
        model = ResidualHead().double().cuda()
        model.load_state_dict(torch.load(root / 'results/preflight/PRIVATE_HEAD_INITIAL.pt',
                                         map_location='cpu', weights_only=True))
        return model

    def score(arm, fusion, domain, split='val'):
        name = f'{arm}_{domain}' + ('' if split == 'val' else '_train')
        status('EVALUATE', arm=arm, domain=domain, split=split)
        with ledger.attempt(name, seconds=900) as out:
            result = evaluate(p, fusion, domain, split, out / 'validation')
            result['predictor'] = 'frozen_UNet_plus_posterior_shape_mean' if arm == 'Shape' else 'frozen_UNet_plus_raw_image'
            result['draws'] = 1
            (out / 'validation/SUMMARY.json').write_text(json.dumps(result, indent=2))
        return result

    try:
        c1 = Path(p['c1_root'])
        assert json.loads((c1 / 'EXIT.json').read_text())['exit_code'] == 0
        assert json.loads((c1 / 'STATUS.json').read_text())['stage'] == 'SOURCE_GATE_FAILED'
        source = data(p, cfg, 'RUNMC', 'L')
        assert source.counts == {'L': 80, 'U': 0}
        assert cfg.height == cfg.width == 128 and cfg.dtype == 'float64' and cfg.outer_lr == 3e-4
        status('PREFLIGHT')
        with ledger.attempt('preflight', seconds=300, updates=0) as out:
            base = frozen()
            old = torch.load(Path(p['unet_root']) / 'results/U1600/PRIVATE_CHECKPOINT.pt',
                             map_location='cpu', weights_only=True)['model']
            error = max(float((base.unet.state_dict()[k].cpu() - v).abs().max()) for k, v in old.items())
            assert error <= 1e-10
            torch.manual_seed(20261025)
            initial = ResidualHead().double().cuda()
            torch.save(initial.state_dict(), out / 'PRIVATE_HEAD_INITIAL.pt')
            raw, shape = head(), head()
            assert all(torch.equal(raw.state_dict()[k], shape.state_dict()[k]) for k in raw.state_dict())
            image, label = batch(source, 0, 163000)
            with torch.no_grad():
                task, variational, base_logits, d = base.objectives(
                    image, label, torch.Generator(device='cuda').manual_seed(54000000))
                assert torch.isfinite(task) and torch.isfinite(variational)
                assert torch.isfinite(d['shape']).all() and torch.isfinite(d['appearance']).all()
                assert torch.isfinite(base.shape_posterior(image)).all()
                assert torch.isfinite(base.appearance_posterior(image)).all()
                for kind, model in (('raw', raw), ('shape', shape)):
                    output = Fusion(base, model, kind).logits(image)
                    torch.testing.assert_close(output, base_logits, rtol=0, atol=0)
            rg = torch.autograd.grad(Fusion(base, raw, 'raw').loss(image, label), list(raw.parameters()))
            sg = torch.autograd.grad(Fusion(base, shape, 'shape').loss(image, label), list(shape.parameters()))
            assert all(torch.isfinite(g).all() for g in rg + sg)
            assert any(g.norm() > 0 for g in rg) and any(g.norm() > 0 for g in sg)
            assert all(v.grad is None for v in base.parameters())
            write(root / 'PREFLIGHT.json', dict(status='PASS', production_updates=0,
                                                C1_U1600_max_tensor_error=error,
                                                initial_heads_equal=True, initial_prediction_exact_U=True,
                                                full_shape_appearance_generator_finite=True,
                                                frozen_model_no_gradient=True,
                                                source_labeled_slices=source.counts['L']))
            del initial, raw, shape, base
        cells = {}
        for arm, kind in (('Raw', 'raw'), ('Shape', 'shape')):
            base = frozen()
            model = head()
            fusion = Fusion(base, model, kind)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)
            status('TRAIN', arm=arm)
            with ledger.attempt(arm, seconds=12000, updates=1600) as out:
                with (out / 'PRIVATE_TRAIN_LOG.jsonl').open('x') as log:
                    for step in range(1600):
                        ledger.check()
                        image, label = batch(source, step, 163000)
                        optimizer.zero_grad(set_to_none=True)
                        loss = fusion.loss(image, label)
                        loss.backward()
                        assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all()
                                                            for v in model.parameters())
                        assert all(v.grad is None for v in base.parameters())
                        optimizer.step()
                        log.write(json.dumps(dict(step=step + 1, task=float(loss.detach()))) + '\n')
                        log.flush()
                        if (step + 1) % 400 == 0:
                            temp = out / 'checkpoint.tmp'
                            torch.save(dict(head=model.state_dict(), optimizer=optimizer.state_dict(),
                                            step=step + 1), temp)
                            temp.replace(out / 'PRIVATE_CHECKPOINT.pt')
                write(out / 'RECEIPT.json', dict(status='COMPLETE', steps=1600,
                                                 frozen_generator=True, frozen_unet=True))
            for domain in ('RUNMC', 'BMC'):
                cells[f'{arm}_{domain}'] = score(arm, fusion, domain)
            cells[f'{arm}_RUNMC_train'] = score(arm, fusion, 'RUNMC', split='train_labeled')
            del base, model, fusion, optimizer
        write(root / 'RESULTS.json', dict(cells=cells, formal_updates=3200,
                                          frozen_C1_source_steps=1600, training_seed=20261023,
                                          target_not_run=True))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
