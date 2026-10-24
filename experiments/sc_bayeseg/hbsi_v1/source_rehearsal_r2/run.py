"""Paired source rehearsal during target-stage hierarchical feedback."""
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..budgets import ResourceLedger
from ..config import Config
from ..feedback_c1.model import FeedbackModel, standard_risk_from_logits
from ..feedback_c1.run import evaluate, exact_unet
from ..fullfield_bayeseg_a1.run import batch
from ..paired_semantic_v19.run import data
from ..real_diagnostic import write


def source_task(model, image, label):
    return standard_risk_from_logits(model.unet(image).logits, label)


def main():
    root = Path(os.environ['RUN_ROOT'])
    p = json.loads((root / 'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose'] == 'SOURCE_REHEARSAL_R2'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = replace(Config(**p['config']), seed=20261023)
    ledger = ResourceLedger(root / 'results', max_seconds=12600,
                            max_updates=1600, max_bytes=3_000_000_000)

    def status(stage, **extra):
        tmp = root / 'STATUS.tmp'
        tmp.write_text(json.dumps(dict(stage=stage, time=time.time(), **extra), indent=2))
        tmp.replace(root / 'STATUS.json')

    c1 = Path(p['c1_root'])
    u = Path(p['unet_root'])
    source_checkpoint = c1 / 'results/Cut_source/PRIVATE_CHECKPOINT.pt'

    def fresh(cut):
        model = FeedbackModel(cut=cut).double().cuda()
        saved = torch.load(source_checkpoint, map_location='cpu', weights_only=True)
        assert saved['step'] == 1600
        model.load_state_dict(saved['model'])
        return model

    def score(arm, model, domain, split='val'):
        name = f'{arm}_{domain}' + ('' if split == 'val' else '_train')
        status('EVALUATE', arm=arm, domain=domain, split=split)
        with ledger.attempt(name, seconds=900, updates=0) as out:
            return evaluate(p, model, domain, split, out / 'validation', 'mean')

    try:
        assert json.loads((c1 / 'EXIT.json').read_text())['exit_code'] == 0
        assert json.loads((c1 / 'SOURCE_GATE.json').read_text())['status'] == 'FAILED'
        assert json.loads((u / 'EXIT.json').read_text())['exit_code'] == 0
        source = data(p, cfg, 'RUNMC', 'L')
        target = data(p, cfg, 'BMC', 'L')
        assert source.counts == {'L': 80, 'U': 0}
        assert target.counts == {'L': 134, 'U': 0}
        assert cfg.height == cfg.width == 128 and cfg.dtype == 'float64'
        assert cfg.outer_lr == 3e-4
        status('PREFLIGHT')
        with ledger.attempt('preflight', seconds=450, updates=0):
            cut = fresh(True)
            joint = fresh(False)
            assert exact_unet(cut, u / 'results/U1600/PRIVATE_CHECKPOINT.pt') == 0
            assert all(torch.equal(cut.state_dict()[k], joint.state_dict()[k])
                       for k in cut.state_dict())
            sx, sy = batch(source, 0, 163000)
            tx, ty = batch(target, 0, 263000)
            seed = 54000000 + (263000 - 163000) * 10000
            ct, cv, cl, cd = cut.objectives(
                tx, ty, torch.Generator(device='cuda').manual_seed(seed))
            jt, jv, jl, jd = joint.objectives(
                tx, ty, torch.Generator(device='cuda').manual_seed(seed))
            cs = source_task(cut, sx, sy)
            js = source_task(joint, sx, sy)
            for a, b in ((ct, jt), (cv, jv), (cl, jl), (cs, js)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for key in ('shape', 'appearance', 'z', 'probability', 'logq'):
                torch.testing.assert_close(cd[key], jd[key], rtol=0, atol=0)
            cp = list(cut.unet.parameters())
            jp = list(joint.unet.parameters())
            sc = torch.autograd.grad(cs, cp, retain_graph=True)
            sj = torch.autograd.grad(js, jp, retain_graph=True)
            assert any(torch.isfinite(g).all() and g.norm() > 0 for g in sc)
            for a, b in zip(sc, sj):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            gc = torch.autograd.grad(cv, cp, retain_graph=True, allow_unused=True)
            gj = torch.autograd.grad(jv, jp, retain_graph=True, allow_unused=True)
            assert all(g is None for g in gc)
            assert any(g is not None and torch.isfinite(g).all() and g.norm() > 0
                       for g in gj)
            state = {k: v.detach().clone() for k, v in cut.state_dict().items()}
            for model, st, tt, var in ((cut, cs, ct, cv), (joint, js, jt, jv)):
                (.5 * st + .5 * tt + .01 * var).backward()
                assert all(v.grad is None or torch.isfinite(v.grad).all()
                           for v in model.parameters())
                model.zero_grad(set_to_none=True)
                assert all(torch.equal(state[k], model.state_dict()[k]) for k in state)
            write(root / 'PREFLIGHT.json', dict(status='PASS', formal_updates=0,
                source_U_exact=True, paired_initial_state=True,
                paired_source_target_task_bound_draws=True,
                source_supervised_U_gradients_equal_nonzero=True,
                cut_generator_U_gradient_zero=True,
                joint_generator_U_gradient_nonzero=True,
                production_backward_finite=True, state_restored=True))
            del cut, joint

        cells = {}
        for arm, cut_flag in (('ReplayCut', True), ('ReplayJoint', False)):
            model = fresh(cut_flag)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)
            assert not optimizer.state
            status('TRAIN_TARGET_WITH_SOURCE', arm=arm)
            with ledger.attempt(arm, seconds=5000, updates=800) as out:
                def save(step):
                    tmp = out / 'checkpoint.tmp'
                    torch.save(dict(model=model.state_dict(),
                                    optimizer=optimizer.state_dict(), step=step), tmp)
                    tmp.replace(out / f'PRIVATE_CHECKPOINT_{step}.pt')
                save(0)
                model.train()
                started = time.monotonic()
                with (out / 'PRIVATE_TRAIN_LOG.jsonl').open('x') as log:
                    for step in range(800):
                        ledger.check()
                        sx, sy = batch(source, step, 163000)
                        tx, ty = batch(target, step, 263000)
                        generator = torch.Generator(device='cuda').manual_seed(
                            54000000 + (263000 - 163000) * 10000 + step)
                        optimizer.zero_grad(set_to_none=True)
                        tt, var, _, d = model.objectives(tx, ty, generator)
                        st = source_task(model, sx, sy)
                        loss = .5 * st + .5 * tt + .01 * var
                        loss.backward()
                        assert torch.isfinite(loss) and all(
                            v.grad is None or torch.isfinite(v.grad).all()
                            for v in model.parameters())
                        optimizer.step()
                        log.write(json.dumps(dict(step=step + 1,
                            source_task=float(st.detach()), target_task=float(tt.detach()),
                            bound_per_pixel=float((-var).detach()),
                            shape_logsd=float(d['shape_logsd'].detach().mean()),
                            z_logsd=float(d['z_logsd'].detach().mean()))) + '\n')
                        log.flush()
                        if step + 1 in (400, 800):
                            save(step + 1)
                write(out / 'RECEIPT.json', dict(status='COMPLETE', steps=800,
                    source_supervised_images=800, target_supervised_images=800,
                    target_generator_images=800,
                    seconds=time.monotonic() - started))
            cells[f'{arm}_RUNMC_train'] = score(arm, model, 'RUNMC', 'train_labeled')
            cells[f'{arm}_BMC_train'] = score(arm, model, 'BMC', 'train_labeled')
            for domain in ('RUNMC', 'BMC'):
                cells[f'{arm}_{domain}'] = score(arm, model, domain)
            del model, optimizer
        write(root / 'RESULTS.json', dict(cells=cells, formal_updates=1600,
            source_training_reused='C1_Cut_source1600',
            source_replay_supervised_images_per_arm=800,
            target_supervised_images_per_arm=800,
            target_generator_images_per_arm=800,
            target_unlabeled_used=False))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
