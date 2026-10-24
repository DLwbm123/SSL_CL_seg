"""Frozen B1 paired chains, with exact initial-function control."""
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..budgets import ResourceLedger
from ..config import Config
from ..fullfield_bayeseg_a1.run import batch, evaluate, train
from ..paired_semantic_v19.run import data
from ..real_diagnostic import write
from ..risk_factorial_b0.method import HStd
from .method import FIRST_WEIGHT, HIS, HISNoVar, copied_initial


def main():
    root = Path(os.environ['RUN_ROOT'])
    protocol = json.loads((root / 'PRIVATE_PROTOCOL.json').read_text())
    assert protocol['purpose'] == 'IMAGE_CONDITIONED_B1'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = replace(Config(**protocol['config']), seed=20261023)
    ledger = ResourceLedger(root / 'results', max_seconds=12600, max_updates=4800,
                            max_bytes=3_000_000_000)

    def status(stage, **extra):
        p = root / 'STATUS.tmp'
        p.write_text(json.dumps(dict(stage=stage, time=time.time(), **extra), indent=2))
        p.replace(root / 'STATUS.json')

    def initial(cls):
        state = torch.load(Path(protocol['b0_root']) / 'results/preflight/PRIVATE_H_Std_INITIAL.pt',
                           map_location='cpu', weights_only=True)
        return copied_initial(cls().double().cuda(), state)

    try:
        old = Path(protocol['b0_root'])
        assert json.loads((old / 'EXIT.json').read_text())['exit_code'] == 0
        assert json.loads((old / 'RESULTS.json').read_text())['formal_updates'] == 4800
        assert json.loads((Path(protocol['a1_root']) / 'EXIT.json').read_text())['exit_code'] == 0
        assert json.loads((Path(protocol['unet_root']) / 'EXIT.json').read_text())['exit_code'] == 0
        source = data(protocol, cfg, 'RUNMC', 'L')
        target = data(protocol, cfg, 'BMC', 'L')
        assert source.counts == {'L': 80, 'U': 0} and target.counts == {'L': 134, 'U': 0}
        assert cfg.height == cfg.width == 128 and cfg.dtype == 'float64' and cfg.outer_lr == 3e-4
        status('PREFLIGHT')
        with ledger.attempt('preflight', seconds=300, updates=0) as out:
            assert (root / 'source/sc_bayeseg/hbsi_v1/risk_factorial_b0/method.py').read_bytes() == (
                old / 'source/sc_bayeseg/hbsi_v1/risk_factorial_b0/method.py').read_bytes()
            std = HStd().double().cuda()
            std.load_state_dict(torch.load(old / 'results/preflight/PRIVATE_H_Std_INITIAL.pt',
                                           map_location='cpu', weights_only=True))
            full, task = initial(HIS), initial(HISNoVar)
            for k, v in full.state_dict().items():
                torch.testing.assert_close(v, task.state_dict()[k], rtol=0, atol=0)
                if k == FIRST_WEIGHT:
                    torch.testing.assert_close(v[:, :1], std.state_dict()[k], rtol=0, atol=0)
                    assert torch.count_nonzero(v[:, 1:]).item() == 0
                else:
                    torch.testing.assert_close(v, std.state_dict()[k], rtol=0, atol=0)
            probes = {}
            for domain, d, seed in (('RUNMC', source, 163000), ('BMC', target, 263000)):
                image, label = batch(d, 0, seed)
                noise = 54000000
                outputs = [m.draw(image, torch.Generator(device='cuda').manual_seed(noise))
                           for m in (std, full, task)]
                for i in range(5):
                    torch.testing.assert_close(outputs[0][i], outputs[1][i], rtol=0, atol=1e-12)
                    torch.testing.assert_close(outputs[1][i], outputs[2][i], rtol=0, atol=1e-12)
                losses = [m.loss(image, label, torch.Generator(device='cuda').manual_seed(noise))
                          for m in (std, full, task)]
                torch.testing.assert_close(losses[0][0], losses[1][0], rtol=0, atol=1e-12)
                torch.testing.assert_close(losses[1][0] - losses[2][0],
                                           -.01 * torch.as_tensor(losses[1][1]['bound_per_pixel'],
                                                                   device='cuda', dtype=torch.float64),
                                           rtol=0, atol=1e-10)
                for arm, model in (('H_IS', full), ('H_IS_NoVar', task)):
                    model.zero_grad(set_to_none=True)
                    model.loss(image, label, torch.Generator(device='cuda').manual_seed(noise))[0].backward()
                    grads = [name for name, p in model.named_parameters() if p.grad is not None]
                    assert grads and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                    assert model.segmentation_posterior.enc1.block[0].weight.grad[:, 1:].norm() > 0
                    if arm == 'H_IS':
                        assert len(grads) == len(list(model.parameters()))
                    else:
                        assert any('segmentation_posterior' in n for n in grads)
                        assert any('shape_posterior' in n for n in grads)
                        assert not any('appearance_posterior' in n or 'shape_mean' in n for n in grads)
                    probes[arm + '_' + domain] = dict(
                        loss=float(losses[1 if arm == 'H_IS' else 2][0].detach()),
                        gradient_parameter_count=len(grads))
                    model.zero_grad(set_to_none=True)
            for arm, model in (('H_IS', full), ('H_IS_NoVar', task)):
                checkpoint = out / ('PRIVATE_' + arm + '_INITIAL.pt')
                torch.save(model.state_dict(), checkpoint)
                restored = initial(type(model))
                restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
                assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
            write(root / 'PREFLIGHT.json', dict(status='PASS', HStd_initial_function_exact=True,
                                             arms_initial_function_exact=True, real_backward_calls=4,
                                             optimizer_updates=0, image_channel_initial_zero=True,
                                             probes=probes))
            del std, full, task
        cells = {}
        for arm, cls in (('H_IS', HIS), ('H_IS_NoVar', HISNoVar)):
            model = initial(cls)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)
            status('TRAIN_SOURCE', arm=arm)
            source_checkpoint = train(model, optimizer, source, 163000, 1600,
                                      arm + '_source', ledger, 3000)
            for domain in ('RUNMC', 'BMC'):
                status('EVALUATE_SOURCE', arm=arm, domain=domain)
                with ledger.attempt(arm + '_source_' + domain, seconds=800) as out:
                    cells[arm + '_source_' + domain] = evaluate(protocol, model, domain, 'val', out / 'validation')
            sealed = torch.load(source_checkpoint, map_location='cpu', weights_only=True)
            assert sealed['step'] == 1600
            model.load_state_dict(sealed['model'])
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)
            assert not optimizer.state
            status('TRAIN_TARGET', arm=arm)
            train(model, optimizer, target, 263000, 800, arm + '_target', ledger, 1800)
            status('EVALUATE_TARGET_TRAIN', arm=arm)
            with ledger.attempt(arm + '_target_train', seconds=800) as out:
                cells[arm + '_target_BMC_train'] = evaluate(protocol, model, 'BMC', 'train_labeled',
                                                            out / 'validation')
            for domain in ('RUNMC', 'BMC'):
                status('EVALUATE_TARGET', arm=arm, domain=domain)
                with ledger.attempt(arm + '_target_' + domain, seconds=800) as out:
                    cells[arm + '_target_' + domain] = evaluate(protocol, model, domain, 'val',
                                                                out / 'validation')
            del model, optimizer
        write(root / 'RESULTS.json', dict(cells=cells, formal_updates=4800,
                                          source_updates_per_arm=1600, target_updates_per_arm=800,
                                          preflight_updates=0, training_seed=20261023))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
