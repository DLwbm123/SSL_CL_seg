"""Paired BayeSeg shape-label bridge training with the full generator retained."""
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..budgets import ResourceLedger
from ..config import Config
from ..fullfield_bayeseg_a1.run import batch, evaluate
from ..paired_semantic_v19.run import data
from ..real_diagnostic import write
from .model import ShapeBridge


def grad_audit(task, variational, parameters):
    tg = torch.autograd.grad(task, parameters, retain_graph=True, allow_unused=True)
    vg = torch.autograd.grad(variational, parameters, retain_graph=True, allow_unused=True)
    zero = torch.zeros((), device='cuda', dtype=torch.float64)
    tn = sum((g.square().sum() for g in tg if g is not None), zero).sqrt()
    vn = sum((g.square().sum() for g in vg if g is not None), zero).sqrt()
    dot = sum((a.mul(b).sum() for a, b in zip(tg, vg) if a is not None and b is not None), zero)
    cosine = float(dot / (tn * vn)) if tn > 0 and vn > 0 else None
    return dict(task_grad_norm=float(tn), weighted_variational_grad_norm=float(.01 * vn),
                cosine=cosine, task_gradient_parameters=sum(g is not None for g in tg),
                variational_gradient_parameters=sum(g is not None for g in vg))


def main():
    root = Path(os.environ['RUN_ROOT'])
    p = json.loads((root / 'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose'] == 'LABEL_SHAPE_BRIDGE_F1'
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

    def fresh(label_feedback):
        model = ShapeBridge(label_feedback).double().cuda()
        model.load_state_dict(torch.load(root / 'results/preflight/PRIVATE_INITIAL.pt',
                                         map_location='cpu', weights_only=True))
        model.base.unet.requires_grad_(False)
        return model

    def unet_error(model):
        old = torch.load(Path(p['unet_root']) / 'results/U1600/PRIVATE_CHECKPOINT.pt',
                         map_location='cpu', weights_only=True)['model']
        new = model.base.unet.state_dict()
        assert set(new) == set(old)
        return max(float((new[k].detach().cpu() - old[k]).abs().max()) for k in old)

    def score(arm, model, domain, split='val'):
        name = f'{arm}_{domain}' + ('' if split == 'val' else '_train')
        status('EVALUATE', arm=arm, domain=domain, split=split)
        with ledger.attempt(name, seconds=900) as out:
            result = evaluate(p, model, domain, split, out / 'validation')
            result['predictor'] = 'image_only_BayeSeg_shape_mean_bridge'
            result['draws'] = 1
            (out / 'validation/SUMMARY.json').write_text(json.dumps(result, indent=2))
        return result

    try:
        c1, e1 = Path(p['c1_root']), Path(p['e1_root'])
        assert json.loads((c1 / 'EXIT.json').read_text())['exit_code'] == 0
        assert json.loads((e1 / 'EXIT.json').read_text())['exit_code'] == 0
        source = data(p, cfg, 'RUNMC', 'L')
        assert source.counts == {'L': 80, 'U': 0}
        assert cfg.height == cfg.width == 128 and cfg.dtype == 'float64' and cfg.outer_lr == 3e-4
        status('PREFLIGHT')
        with ledger.attempt('preflight', seconds=300, updates=0) as out:
            anchor = ShapeBridge(True).double().cuda()
            c1_state = torch.load(c1 / 'results/Cut_source/PRIVATE_CHECKPOINT.pt',
                                  map_location='cpu', weights_only=True)
            assert c1_state['step'] == 1600
            anchor.base.load_state_dict(c1_state['model'])
            anchor.head.load_state_dict(torch.load(e1 / 'results/preflight/PRIVATE_HEAD_INITIAL.pt',
                                                   map_location='cpu', weights_only=True))
            anchor.base.unet.requires_grad_(False)
            assert unet_error(anchor) <= 1e-10
            torch.save(anchor.state_dict(), out / 'PRIVATE_INITIAL.pt')
            cut = fresh(False)
            anchor = fresh(True)
            assert all(torch.equal(cut.state_dict()[k], anchor.state_dict()[k])
                       for k in cut.state_dict())
            image, label = batch(source, 0, 163000)
            cut.train(); anchor.train()
            before = {k: v.detach().clone() for k, v in anchor.base.unet.state_dict().items()}
            noise = 54000000
            ct, cv, cl, cd = cut.objectives(image, label,
                                            torch.Generator(device='cuda').manual_seed(noise))
            at, av, al, ad = anchor.objectives(image, label,
                                               torch.Generator(device='cuda').manual_seed(noise))
            for a, b in ((ct, at), (cv, av), (cl, al), (cd['shape'], ad['shape']),
                         (cd['appearance'], ad['appearance'])):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            with torch.no_grad():
                base_logits = anchor.base.unet(image).logits
                torch.testing.assert_close(al, base_logits, rtol=0, atol=0)
            assert torch.isfinite(ct) and torch.isfinite(cv)
            assert all(torch.equal(v, anchor.base.unet.state_dict()[k]) for k, v in before.items())
            # A reversible fixed perturbation checks the path hidden by the zero residual initialization.
            with torch.no_grad():
                for model in (cut, anchor):
                    model.head.layers[-1].weight[0].fill_(.001)
                    model.head.layers[-1].weight[1].fill_(-.001)
            ct2, _, _, _ = cut.objectives(image, label,
                                           torch.Generator(device='cuda').manual_seed(noise))
            at2, _, _, _ = anchor.objectives(image, label,
                                              torch.Generator(device='cuda').manual_seed(noise))
            cp = list(cut.base.shape_posterior.parameters())
            ap = list(anchor.base.shape_posterior.parameters())
            cg = torch.autograd.grad(ct2, cp, allow_unused=True, retain_graph=True)
            ag = torch.autograd.grad(at2, ap, allow_unused=True, retain_graph=True)
            assert all(g is None for g in cg)
            assert any(g is not None and torch.isfinite(g).all() and g.norm() > 0 for g in ag)
            qag = torch.autograd.grad(at2, list(anchor.base.appearance_posterior.parameters()),
                                      allow_unused=True, retain_graph=True)
            assert all(g is None for g in qag)
            vg = torch.autograd.grad(av, ap, allow_unused=True)
            assert any(g is not None and torch.isfinite(g).all() and g.norm() > 0 for g in vg)
            cut.load_state_dict(torch.load(out / 'PRIVATE_INITIAL.pt', map_location='cpu', weights_only=True))
            anchor.load_state_dict(torch.load(out / 'PRIVATE_INITIAL.pt', map_location='cpu', weights_only=True))
            assert all(torch.equal(cut.state_dict()[k], anchor.state_dict()[k])
                       for k in cut.state_dict())
            assert unet_error(cut) == unet_error(anchor) == 0
            write(root / 'PREFLIGHT.json', dict(status='PASS', production_updates=0,
                                                initial_state_equal=True, initial_output_equal_U=True,
                                                complete_generator_finite=True,
                                                frozen_U_exact=True, cut_label_to_shape_zero=True,
                                                anchor_label_to_shape_nonzero_on_reversible_probe=True,
                                                label_to_appearance_direct_zero=True,
                                                generator_to_shape_nonzero=True,
                                                probe_state_restored=True))
            del cut, anchor
        cells = {}
        for arm, label_feedback in (('Cut', False), ('Anchor', True)):
            model = fresh(label_feedback)
            model.train()
            optimizer = torch.optim.Adam([v for v in model.parameters() if v.requires_grad],
                                         lr=cfg.outer_lr)
            status('TRAIN', arm=arm)
            with ledger.attempt(arm, seconds=12000, updates=1600) as out:
                with (out / 'PRIVATE_TRAIN_LOG.jsonl').open('x') as log:
                    for step in range(1600):
                        ledger.check()
                        image, label = batch(source, step, 163000)
                        generator = torch.Generator(device='cuda').manual_seed(54000000 + step)
                        optimizer.zero_grad(set_to_none=True)
                        task, variational, _, d = model.objectives(image, label, generator)
                        loss = task + .01 * variational
                        audit = (grad_audit(task, variational,
                                            list(model.base.shape_posterior.parameters()))
                                 if step + 1 in (1, 400, 800, 1200, 1600) else None)
                        loss.backward()
                        assert torch.isfinite(loss)
                        assert all(v.grad is None for v in model.base.unet.parameters())
                        assert all(v.grad is None or torch.isfinite(v.grad).all()
                                   for v in model.parameters())
                        optimizer.step()
                        row = dict(step=step + 1, task=float(task.detach()),
                                   variational=float(variational.detach()),
                                   shape_logsd=float(d['shape_logsd'].detach().mean()),
                                   appearance_logsd=float(d['appearance_logsd'].detach().mean()),
                                   reconstruction_mse=float((image-d['shape']-d['appearance']).detach().square().mean()))
                        if audit is not None:
                            row['gradient_audit_pre_update'] = audit
                        log.write(json.dumps(row) + '\n'); log.flush()
                        if audit is not None:
                            temp = out / 'checkpoint.tmp'
                            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                            step=step + 1, audit=audit), temp)
                            temp.replace(out / f'PRIVATE_CHECKPOINT_{step + 1}.pt')
                assert unet_error(model) == 0
                write(out / 'RECEIPT.json', dict(status='COMPLETE', steps=1600,
                                                 unet_max_tensor_error=0,
                                                 audit_steps=[1, 400, 800, 1200, 1600]))
            for domain in ('RUNMC', 'BMC'):
                cells[f'{arm}_{domain}'] = score(arm, model, domain)
            cells[f'{arm}_RUNMC_train'] = score(arm, model, 'RUNMC', split='train_labeled')
            del model, optimizer
        write(root / 'RESULTS.json', dict(cells=cells, formal_updates=3200,
                                          reused_C1_source_steps=1600,
                                          training_seed=20261023, target_not_run=True))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
