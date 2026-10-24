"""Frozen B0 two-arm source-to-target training and full endpoints."""
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..budgets import ResourceLedger
from ..config import Config
from ..fullfield_bayeseg_a1.model import FullFieldBayeSeg
from ..fullfield_bayeseg_a1.run import batch, evaluate as a1_evaluate, train
from ..paired_semantic_v19.run import data
from ..real_diagnostic import write
from .method import HStd, UBal


def evaluate(p, model, domain, split, out, arm):
    summary = a1_evaluate(p, model, domain, split, out)
    summary['predictor'] = 'deterministic_standard_UNet' if arm == 'U_Bal' else 'fullfield_local_posterior_mean'
    write(out / 'SUMMARY.json', summary)
    return summary


def main():
    root = Path(os.environ['RUN_ROOT'])
    protocol = json.loads((root / 'PRIVATE_PROTOCOL.json').read_text())
    assert protocol['purpose'] == 'RISK_FACTORIAL_B0'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = replace(Config(**protocol['config']), seed=20261023)
    ledger = ResourceLedger(root / 'results', max_seconds=12600, max_updates=4800,
                            max_bytes=3_000_000_000)

    def status(stage, **extra):
        temporary = root / 'STATUS.tmp'
        temporary.write_text(json.dumps(dict(stage=stage, time=time.time(), **extra), indent=2))
        temporary.replace(root / 'STATUS.json')

    def initial(arm):
        if arm == 'U_Bal':
            return UBal()
        model = HStd().double().cuda()
        initial_path = Path(protocol['a1_root']) / 'results/preflight/PRIVATE_PREFLIGHT.pt'
        model.load_state_dict(torch.load(initial_path, map_location='cpu', weights_only=True))
        return model

    try:
        assert json.loads((Path(protocol['a1_root']) / 'EXIT.json').read_text())['exit_code'] == 0
        assert json.loads((Path(protocol['unet_root']) / 'EXIT.json').read_text())['exit_code'] == 0
        a1_result = json.loads((Path(protocol['a1_root']) / 'RESULTS.json').read_text())
        u_result = json.loads((Path(protocol['unet_root']) / 'RESULTS.json').read_text())
        assert a1_result['formal_updates'] == 2400 and u_result['cells']['U_B800_BMC']['complete']
        source = data(protocol, cfg, 'RUNMC', 'L')
        target = data(protocol, cfg, 'BMC', 'L')
        assert source.counts == {'L': 80, 'U': 0} and target.counts == {'L': 134, 'U': 0}
        assert cfg.height == cfg.width == 128 and cfg.dtype == 'float64' and cfg.outer_lr == 3e-4
        status('PREFLIGHT')
        with ledger.attempt('preflight', seconds=300, updates=0):
            for relative in ('lcrseg/models/unet.py', 'sc_bayeseg/hbsi_v1/unet_transfer_v45/run.py',
                             'sc_bayeseg/hbsi_v1/paired_semantic_v19/method.py'):
                assert (root / 'source' / relative).read_bytes() == (Path(protocol['unet_root']) / 'source' / relative).read_bytes()
            h = initial('H_Std')
            historic = FullFieldBayeSeg().double().cuda()
            historic.load_state_dict(torch.load(Path(protocol['a1_root']) / 'results/preflight/PRIVATE_PREFLIGHT.pt',
                                                map_location='cpu', weights_only=True))
            assert all(torch.equal(v, historic.state_dict()[k]) for k, v in h.state_dict().items())
            first, labels = batch(source, 0, 163000)
            same_seed = 54000000
            p_h = h.draw(first, torch.Generator(device='cuda').manual_seed(same_seed))[3]
            p_old = historic.draw(first, torch.Generator(device='cuda').manual_seed(same_seed))[3]
            torch.testing.assert_close(p_h, p_old, rtol=0, atol=0)
            hb_loss, hb_parts = historic.loss(first, labels, torch.Generator(device='cuda').manual_seed(same_seed))
            hs_loss, hs_parts = h.loss(first, labels, torch.Generator(device='cuda').manual_seed(same_seed))
            assert abs(hb_parts['bound_per_pixel'] - hs_parts['bound_per_pixel']) < 1e-12
            assert abs(float((hs_loss - hb_loss).detach()) - (hs_parts['task'] - hb_parts['task'])) < 1e-12
            del historic, p_h, p_old, hb_loss, hs_loss, first, labels
            probes = {}
            for arm in ('U_Bal', 'H_Std'):
                model = initial(arm)
                original = {k: v.detach().clone() for k, v in model.state_dict().items()}
                for domain, d, seed in (('RUNMC', source, 163000), ('BMC', target, 263000)):
                    image, label = batch(d, 0, seed)
                    generator = torch.Generator(device='cuda').manual_seed(54000000)
                    loss, parts = model.loss(image, label, generator)
                    loss.backward()
                    assert torch.isfinite(loss)
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
                               for p in model.parameters())
                    prediction = model.predict(image, seed=1101)
                    torch.testing.assert_close(prediction.sum(1), torch.ones_like(prediction[:, 0]))
                    probes[arm + '_' + domain] = dict(loss=float(loss.detach()), **parts)
                    model.zero_grad(set_to_none=True)
                assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
                checkpoint = root / 'results/preflight' / ('PRIVATE_' + arm + '_INITIAL.pt')
                torch.save(model.state_dict(), checkpoint)
                test = initial(arm)
                test.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
                assert all(torch.equal(v, test.state_dict()[k]) for k, v in original.items())
                del model, test
            write(root / 'PREFLIGHT.json', dict(status='PASS', old_source_and_target_complete=True,
                                                 L_slices=dict(RUNMC=80, BMC=134), H_initial_state_exact=True,
                                                 H_initial_probability_exact=True,
                                                 H_variational_term_exact=True,
                                                 U_historical_initial_state_not_saved=True,
                                                 U_historical_source_code_exact=True,
                                                 real_backward_calls=4, optimizer_updates=0,
                                                 probes=probes))
        cells = {}
        for arm in ('U_Bal', 'H_Std'):
            model = initial(arm)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)
            status('TRAIN_SOURCE', arm=arm)
            source_checkpoint = train(model, optimizer, source, 163000, 1600,
                                      arm + '_source', ledger, 3000)
            for domain in ('RUNMC', 'BMC'):
                status('EVALUATE_SOURCE', arm=arm, domain=domain)
                with ledger.attempt(arm + '_source_' + domain, seconds=800) as out:
                    cells[arm + '_source_' + domain] = evaluate(protocol, model, domain, 'val',
                                                                 out / 'validation', arm)
            checkpoint = torch.load(source_checkpoint, map_location='cpu', weights_only=True)
            assert checkpoint['step'] == 1600
            model.load_state_dict(checkpoint['model'])
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)
            assert not optimizer.state
            status('TRAIN_TARGET', arm=arm)
            train(model, optimizer, target, 263000, 800, arm + '_target', ledger, 1800)
            status('EVALUATE_TARGET_TRAIN', arm=arm)
            with ledger.attempt(arm + '_target_train', seconds=800) as out:
                cells[arm + '_target_BMC_train'] = evaluate(protocol, model, 'BMC', 'train_labeled',
                                                            out / 'validation', arm)
            for domain in ('RUNMC', 'BMC'):
                status('EVALUATE_TARGET', arm=arm, domain=domain)
                with ledger.attempt(arm + '_target_' + domain, seconds=800) as out:
                    cells[arm + '_target_' + domain] = evaluate(protocol, model, domain, 'val',
                                                                 out / 'validation', arm)
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
