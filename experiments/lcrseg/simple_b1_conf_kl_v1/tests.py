import copy
from pathlib import Path
from unittest.mock import patch
import torch
from . import protocol
from .run import BaselineTrainer, construct, engine
from ..five_frameworks_v1.model import Model
from ..five_frameworks_v1.train_stage import StageTrainer
from ..five_frameworks_v1.kernels import masked_kl
from ..five_frameworks_v1.native_parent import build, NativeLRParent
from ..five_frameworks_v1.native_runner import Counter
from ..five_frameworks_v1 import checkpoint
from ..f5_module_pilots_v1.tests import GeneratedDomain, snapshot, equal


def math_checks(counter):
    target = torch.tensor([[[[.8, .6, .61, .7]], [[.1, .3, .19, .2]], [[.1, .1, .2, .1]]]], dtype=torch.double, requires_grad=True)
    geometry = torch.tensor([[[True, True, False, True]]])
    valid = geometry & (target.detach().max(1).values > .6)
    assert torch.equal(valid, torch.tensor([[[True, False, False, True]]]))
    logits = torch.zeros_like(target, requires_grad=True)
    loss = masked_kl(logits, target, valid)
    grad, teacher_grad = torch.autograd.grad(loss, (logits, target), allow_unused=True)
    expected = (logits.detach().softmax(1)-target.detach())*valid[:, None]/valid.sum()
    assert torch.allclose(grad, expected, atol=1e-14, rtol=1e-14) and teacher_grad is None
    empty = masked_kl(logits, target, torch.zeros_like(valid))
    assert empty == 0 and torch.count_nonzero(torch.autograd.grad(empty, logits)[0]) == 0
    fit = torch.nn.Parameter(torch.zeros(2, 3, 1, 1, dtype=torch.double))
    desired = torch.tensor([[.8, .1, .1], [.1, .2, .7]], dtype=torch.double)[:, :, None, None]
    optimizer = torch.optim.Adam([fit], lr=.15); counter.wrap(optimizer)
    mask = torch.ones(2, 1, 1, dtype=torch.bool)
    initial = float(masked_kl(fit, desired, mask))
    for _ in range(32):
        optimizer.zero_grad(); objective = masked_kl(fit, desired, mask)
        objective.backward(); optimizer.step()
    final = float(masked_kl(fit, desired, mask))
    assert final < initial*.05
    return dict(strict_confidence_geometry=True, detached_teacher=True, empty_graph_zero=True,
                gradient_max_error=float((grad-expected).abs().max()), two_case_initial=initial, two_case_final=final)


def qualify(config):
    root = Path(config['run_root'])
    cpu = Counter(root/'qualification_cpu_physical.jsonl', 96)
    cuda = Counter(root/'qualification_cuda_physical.jsonl', 32)
    smoke = Counter(root/'smoke_physical.jsonl', 4)
    if (root/'QUALIFICATION.json').exists():
        raise FileExistsError('qualification already attempted')
    try:
        torch.set_num_threads(2); device = torch.device('cuda:0'); torch.cuda.set_device(device)
        permit = engine.admission(config, 'qualification'); result = math_checks(cpu)
        options = {**protocol.OPTIONS, 'total_steps':4, 'warmup_fraction':.25, 'U_ramp_fraction':.25,
                   'PAS_confidence':0., 'PAS_cosine':.7}

        def generated():
            provider = GeneratedDomain(seed=163, size=384, device=device,
                                       stage_source=dict(kind='generated', seed=163, domain='REFUGE'))
            native = build(config['reference'], device, 163)
            from ..f5_confirmation_v1.native_qualification import foreground_fixture
            foreground_fixture(native)
            model = Model(NativeLRParent(native, 163, provider.stage_source), protocol.B1).to(device)
            return BaselineTrainer(model, provider, options, execution=permit)

        a = generated()
        old_path = Path(config['baseline_parity_code'])/'experiments/lcrseg/five_frameworks_v1/train_stage.py'
        namespace = dict(__name__='experiments.lcrseg.five_frameworks_v1.b1_frozen_stage',
                         __package__='experiments.lcrseg.five_frameworks_v1')
        exec(compile(old_path.read_text(), str(old_path), 'exec'), namespace)
        b = namespace['StageTrainer'](copy.deepcopy(a.model), copy.deepcopy(a.provider), options, execution=permit)
        cuda.wrap(a.optimizer); cuda.wrap(b.optimizer)
        for _ in range(2):
            a.update(); b.update(); equal(snapshot(a), snapshot(b)); equal(a.last, b.last)
        del a, b
        t = generated(); cuda.wrap(t.optimizer); t.update(); gradients = t.update()
        assert t.model.sidecar is None and t.last['active_U'] and t.last['accepted'] > 0
        groups = t.model.parent.parameter_groups()
        for name in ('input_factors', 'output_factors'):
            assert any(gradients[id(p)]['U'] is not None and gradients[id(p)]['U'].abs().sum() > 0 for p in groups[name])
        assert not t.ema.training and all(not p.requires_grad and p.grad is None for p in t.ema.parameters())
        # An all-rejected prototype mask cannot select B1 pixels.
        captured = {}; original = t.fine_kl
        def capture(logits, target, valid):
            captured.update(target=target.detach().clone(), valid=valid.detach().clone())
            return original(logits, target, valid)
        t.fine_kl = capture
        with patch.object(t.prototypes, 'admission', return_value=(torch.zeros(2,384,384,device=device,dtype=torch.bool), {})):
            _, unlabeled, _ = t.losses()
        del t.fine_kl
        expected = captured['target'].max(1).values > 0.
        assert torch.equal(captured['valid'], expected) and expected.any() and t.module_stats['teacher_detached']
        assert unlabeled.isfinite() and t.module_stats['prototype_filter_used'] is False
        golden = dict(supported_pixels=int(expected.sum()), actual_A_B_U_nonzero=True,
                      rejected_prototype_mask_ignored=True, teacher_detached=True)
        del captured, unlabeled
        identity = dict(family=protocol.B1, seed=163, order=1, stage=1, arm=protocol.B1)
        checkpoint.save(t, root/'qualification.pt', identity)
        restored = BaselineTrainer.for_resume(copy.deepcopy(t.model), copy.deepcopy(t.provider), options, execution=permit)
        restored.entry_fingerprint = t.entry_fingerprint
        checkpoint.restore(restored, root/'qualification.pt', identity); cuda.wrap(restored.optimizer)
        t.update(); restored.update(); equal(snapshot(t), snapshot(restored)); assert t.telemetry == restored.telemetry
        del restored
        wrong_model = copy.deepcopy(t.model); wrong_model.family = 'B2_PARENT_PAS_KL'
        wrong = BaselineTrainer.for_resume(wrong_model, copy.deepcopy(t.provider), options, execution=permit)
        wrong.entry_fingerprint = t.entry_fingerprint
        try:
            checkpoint.restore(wrong, root/'qualification.pt', identity)
        except ValueError as error:
            assert 'semantic' in str(error)
        else:
            raise AssertionError('changed baseline family accepted')
        del wrong, wrong_model
        t.model.eval().requires_grad_(False); t.model.role = 'eval'
        images = torch.zeros(2,3,384,384,device=device)
        with torch.no_grad():
            before = t.model(images); deployed = t.model.deploy(); after = deployed(images)
        assert torch.allclose(before, after, atol=2e-6, rtol=1e-5)
        assert torch.equal(deployed.transform, torch.eye(16, device=device))
        provider = GeneratedDomain(seed=163, stage=2, size=384, device=device,
                                   stage_source=dict(kind='own_generated_predecessor', seed=163, domain='RIM_ONE_r3'))
        model = Model(NativeLRParent(deployed.parent.native,163,provider.stage_source), protocol.B1,
                      previous=deployed.transform.detach().clone()).to(device)
        next_stage = BaselineTrainer(model, provider, options, execution=permit)
        assert next_stage.model.sidecar is None and not next_stage.optimizer.state
        assert all(a.new and a.b.count_nonzero() == 0 for a in next_stage.model.parent.adapters)
        cuda.wrap(next_stage.optimizer); next_stage.update(); del next_stage, model, provider, deployed
        t.model.configure_training()
        try:
            t.update(fault='after_optimizer')
        except RuntimeError as error:
            assert str(error) == 'injected after_optimizer'
        else:
            raise AssertionError('expected deliberate failure')
        assert t.requires_restore and t.step == 3
        try:
            t.update()
        except RuntimeError as error:
            assert 'requires checkpoint restore' in str(error)
        else:
            raise AssertionError('failed trainer continued')
        del t
        current = construct(config, protocol.jobs()[0], 1, device, engine.admission(config, 'smoke'), allow_u=False)
        smoke.wrap(current.optimizer); current.update(); current.update(); assert current.provider.u_reads == 0
        del current
        assert (cpu.count, cuda.count, smoke.count) == (32, 10, 2)
        protocol.write(root/'QUALIFICATION.json', dict(status='PASS', execution_commit=config['execution_commit'],
            plan_id=protocol.plan()['plan_id'], costs=dict(cpu_optimizer_updates=cpu.count,
            synthetic_cuda_updates=cuda.count, real_L_smoke_updates=smoke.count), formal_advancement=True,
            math_result=result, native_golden=golden, extra_generated_loss_calls=1,
            checks=['archived built-in B1 native full state/loss/update parity',
                    'strict confidence and geometry, empty KL, detached teacher, two-case fit',
                    'prototype-mask bypass and actual nonzero native A/B U gradients',
                    'frozen EMA, actual checkpoint continuation and changed-family rejection',
                    'identity deployment and own-prefix next stage',
                    'deliberate failed-call ledger and real current-L zero-U smoke']))
    except BaseException as error:
        protocol.write(root/'QUALIFICATION.json', dict(status='FAIL', execution_commit=config['execution_commit'],
            costs=dict(cpu_optimizer_updates=cpu.count, synthetic_cuda_updates=cuda.count, real_L_smoke_updates=smoke.count)))
        protocol.write(root/'QUALIFICATION_FAILURE.private.json', dict(error=repr(error), traceback=__import__('traceback').format_exc()))
        raise
