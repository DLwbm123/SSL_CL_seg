"""Paired target adaptation after source-only hierarchy training."""
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..budgets import ResourceLedger
from ..config import Config
from ..feedback_c1.model import FeedbackModel
from ..feedback_c1.run import evaluate, exact_unet
from ..fullfield_bayeseg_a1.run import batch, train
from ..paired_semantic_v19.run import data
from ..real_diagnostic import write


def main():
    root = Path(os.environ['RUN_ROOT'])
    p = json.loads((root / 'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose'] == 'STAGED_TARGET_FEEDBACK_R1'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = replace(Config(**p['config']), seed=20261023)
    ledger = ResourceLedger(root / 'results', max_seconds=12600, max_updates=1600,
                            max_bytes=3_000_000_000)

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
            result = evaluate(p, model, domain, split, out / 'validation', 'mean')
        return result

    try:
        assert json.loads((c1 / 'EXIT.json').read_text())['exit_code'] == 0
        assert json.loads((c1 / 'SOURCE_GATE.json').read_text())['status'] == 'FAILED'
        assert json.loads((c1 / 'SOURCE_GATE.json').read_text())['gates']['Cut_exact_v45']
        assert json.loads((u / 'EXIT.json').read_text())['exit_code'] == 0
        source = data(p, cfg, 'RUNMC', 'L')
        target = data(p, cfg, 'BMC', 'L')
        assert source.counts == {'L': 80, 'U': 0}
        assert target.counts == {'L': 134, 'U': 0}
        assert cfg.height == cfg.width == 128 and cfg.dtype == 'float64'
        assert cfg.outer_lr == 3e-4
        status('PREFLIGHT')
        with ledger.attempt('preflight', seconds=400, updates=0):
            cut = fresh(True)
            joint = fresh(False)
            assert all(torch.equal(cut.state_dict()[k], joint.state_dict()[k])
                       for k in cut.state_dict())
            source_unet_error = exact_unet(cut,
                u / 'results/U1600/PRIVATE_CHECKPOINT.pt')
            image, label = batch(target, 0, 263000)
            noise = torch.Generator(device='cuda').manual_seed(54000000)
            cut_task, cut_var, cut_logits, cut_draw = cut.objectives(image, label, noise)
            noise = torch.Generator(device='cuda').manual_seed(54000000)
            joint_task, joint_var, joint_logits, joint_draw = joint.objectives(image, label, noise)
            for a,b in ((cut_task,joint_task),(cut_var,joint_var),
                        (cut_logits,joint_logits)):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            for key in ('shape','appearance','z','probability','logq'):
                torch.testing.assert_close(cut_draw[key],joint_draw[key],rtol=0,atol=0)
            cp = list(cut.unet.parameters())
            jp = list(joint.unet.parameters())
            vg_cut = torch.autograd.grad(cut_var, cp, retain_graph=True,
                                         allow_unused=True)
            vg_joint = torch.autograd.grad(joint_var, jp, retain_graph=True,
                                           allow_unused=True)
            assert all(g is None for g in vg_cut)
            assert any(g is not None and torch.isfinite(g).all() and g.norm()>0
                       for g in vg_joint)
            tg_cut = torch.autograd.grad(cut_task, cp, retain_graph=True)
            tg_joint = torch.autograd.grad(joint_task, jp, retain_graph=True)
            for a,b in zip(tg_cut,tg_joint):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            initial = {k:v.detach().clone() for k,v in cut.state_dict().items()}
            for model,task,var in ((cut,cut_task,cut_var),
                                   (joint,joint_task,joint_var)):
                (task+.01*var).backward()
                assert all(v.grad is None or torch.isfinite(v.grad).all()
                           for v in model.parameters())
                model.zero_grad(set_to_none=True)
            assert all(torch.equal(initial[k],cut.state_dict()[k])
                       and torch.equal(initial[k],joint.state_dict()[k])
                       for k in initial)
            write(root / 'PREFLIGHT.json', dict(status='PASS', formal_updates=0,
                c1_source_gate_failed_preserved=True, C1_Cut_source_step=1600,
                source_U_max_tensor_error=source_unet_error,
                paired_initial_state_equal=True,paired_task_bound_logits_draws_equal=True,
                cut_generator_to_U_zero=True,joint_generator_to_U_nonzero=True,
                task_gradients_equal=True,full_backward_finite=True,
                preflight_state_restored=True,
                joint_generator_U_gradient_norm=float(sum((g.square().sum()
                    for g in vg_joint if g is not None),
                    torch.zeros((),device='cuda',dtype=torch.float64)).sqrt())))
            del cut,joint

        cells = {}
        for arm, cut_flag in (('RemainCut', True), ('SwitchJoint', False)):
            model = fresh(cut_flag)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)
            assert not optimizer.state
            status('TRAIN_TARGET', arm=arm)
            train(model, optimizer, target, 263000, 800, arm, ledger, 3500)
            if arm == 'RemainCut':
                target_error = exact_unet(model,
                    u / 'results/U_B800/PRIVATE_CHECKPOINT.pt')
                write(root / 'CUT_TARGET_MATCH.json', dict(status='PASS',
                      max_tensor_error=target_error,
                      comparator='v45_U_B800_same_source_and_target_exposure'))
            cells[f'{arm}_BMC_train'] = score(arm,model,'BMC','train_labeled')
            for domain in ('RUNMC','BMC'):
                cells[f'{arm}_{domain}'] = score(arm,model,domain)
            del model,optimizer
        write(root / 'RESULTS.json', dict(cells=cells,formal_updates=1600,
            source_training_reused='C1_Cut_source1600',target_training_slices=134,
            target_unlabeled_used=False,comparator_exact_U_B800=True))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__ == '__main__':
    main()
