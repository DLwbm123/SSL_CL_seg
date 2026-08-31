"""Small synthetic end-to-end probes plus the complete evidence compiler."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from di_dmpa_gate1_v2.features import ImmutableModels
from di_dmpa_gate1c_v2 import binding as b, execution as e, gradients as g, reliability as r, metrics as m, reporting as rep
from .test_core import Tiny, contract, toy_scores, evidence


def check_supervised_mean_reference(device, ignore_pattern):
    """CPU native CE is an independent value/gradient reference; no real data."""
    old = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(219); model = Tiny().eval().to(device); parts = g.partition(model)
        images = torch.linspace(-1, 1, 2*3*8*8, device=device).reshape(2, 3, 8, 8)
        labels = (torch.arange(128, device=device).reshape(2, 8, 8) % 3).long()
        if ignore_pattern == 'mixed': labels.reshape(-1)[::3] = 255
        if ignore_pattern in ('single_valid', 'all_ignored'):
            labels.fill_(255)
            if ignore_pattern == 'single_valid': labels[1, 3, 2] = 2
        with b.no_updates():
            logits, _ = model(images, stochastic_classifier=False)
            if ignore_pattern == 'all_ignored':
                with pytest.raises(b.ProtocolError, match='labeled reference shape/support'):
                    g.supervised_gradient(logits, labels, parts)
                return dict(device=device, ignore_pattern=ignore_pattern, all_ignored_rejected=True)
            reference_logits = logits.detach().cpu().requires_grad_(True)
            reference = torch.nn.functional.cross_entropy(reference_logits, labels.cpu(), ignore_index=255)
            logit_gradient, = torch.autograd.grad(reference, reference_logits)
            expected = torch.autograd.grad(logits, parts['params'], grad_outputs=logit_gradient.to(device),
                retain_graph=True, allow_unused=True)
            expected_vector = g.vectors(expected, parts)['global']
            loss, vector = g.supervised_gradient(logits, labels, parts)
            assert loss == pytest.approx(float(reference.detach()), abs=2e-7, rel=1e-6)
            np.testing.assert_allclose(vector['global'], expected_vector, atol=1e-7, rtol=1e-5)
            repeated_logits, _ = model(images, stochastic_classifier=False)
            repeated_loss, repeated_vector = g.supervised_gradient(repeated_logits, labels, parts)
            assert loss == repeated_loss and np.array_equal(vector['global'], repeated_vector['global'])
            assert all(parameter.grad is None for parameter in model.parameters())
            return dict(device=device, ignore_pattern=ignore_pattern, bitwise_repeat=True,
                loss_abs_error=abs(loss-float(reference.detach())),
                gradient_max_abs_error=float(np.max(np.abs(vector['global']-expected_vector))))
    finally:
        torch.use_deterministic_algorithms(old, warn_only=old_warn)


@pytest.mark.parametrize('ignore_pattern', ['none', 'mixed', 'single_valid', 'all_ignored'])
def test_supervised_reference_is_mean_ce_over_nonignored_pixels(ignore_pattern):
    check_supervised_mean_reference('cpu', ignore_pattern)


def test_all_four_probe_phases_share_forward_and_never_update(monkeypatch, tmp_path):
    p, _, _, _, freeze = contract(); pair = next(q for q in p['gradient_diagnostic']['batch_pairs'] if q['stage_index'] == 1)
    torch.manual_seed(71); student = Tiny().eval(); teacher = copy.deepcopy(student).requires_grad_(False).eval()
    models = dict(student=student, ema_teacher=teacher); legacy = torch.ones(3, 16)
    current, history = r.banks(freeze, 0, 1)
    generator = torch.Generator().manual_seed(19)
    xu = torch.randn(2, 3, 8, 8, generator=generator); xl = torch.randn(2, 3, 8, 8, generator=generator)
    labels = torch.arange(128).reshape(2, 8, 8) % 3
    monkeypatch.setattr(e, 'pair_inputs', lambda *args: (xu.clone(), xl.clone(), labels.clone()))
    cp = tmp_path/'synthetic_only.pt'; torch.save(student.state_dict(), cp)
    cpdesc = dict(path=str(cp), sha256=b.sha256(cp), checkpoint_id='synthetic_only')
    before = (b.array_hash(current), b.array_hash(history)); metadata = {'scope': 'SYNTHETIC_ONLY_NO_FROZEN_TENSOR_READ'}
    phases = {}
    with b.no_updates():
        for phase in ('draw0', 'noise', 'posterior', 'poe'):
            with ImmutableModels(models, cpdesc, tmp_path/'audit'/phase, metadata):
                phases[phase] = e.gradient_pair(models, legacy, current, history, p, pair, tmp_path, tmp_path/'run', metadata, phase=phase, device='cpu')
                assert g.isolation(models, legacy, before, current, history)['teacher_gradients'] == 'None'
    assert phases['noise']['teacher_draw_seeds'] == pair['teacher_draw_seeds']
    assert phases['noise']['teacher_forwards'] == 7 and phases['poe']['teacher_forwards'] == 0
    assert phases['noise']['alignment'][:56] == phases['draw0']['alignment']
    assert len({x['student_logits_sha256'] for x in phases.values()}) == 1
    assert len({x['supervised_gradient_sha256'] for x in phases.values()}) == 1
    assert phases['posterior']['teacher_stochastic_classifier'] is False
    assert phases['poe']['same_detached_R3_weight'] and phases['poe']['own_predicted_class_strata']
    for phase in phases:
        assert b.read_json(tmp_path/'audit'/phase/'immutability/synthetic_only.json')['bitwise_unchanged']


@pytest.mark.parametrize('input_contract', ['v2', 'v2.1', 'v2.2'])
def test_complete_report_compiler_with_all_72_pairs(tmp_path, monkeypatch, input_contract):
    p = contract()[0]
    metadata = dict(preregistration_commit=b.PREREG, authorization_commit=b.AUTH, diagnostic_code_commit='synthetic-no-real-checkpoints',
        model_optimizer_steps=0, transport_optimizer_steps_this_gate=0, hidden_gt_training_usage='none', test_gt_usage='none', selected_K=2, R4_available=False)
    if input_contract in ('v2.1', 'v2.2'):
        metadata.update(input_contract_version='v2.1', preregistration_commit=b.PREREG_V21,
            original_gate1c_v2_completed=False, historical_bank_hash_verified=False,
            execution_scope='GATE1C_V21_ONLY', next_action='ANALYZE_VERSIONED_RESULT_WITHIN_LONG_RUNNING_SCOPE')
        if input_contract == 'v2.2':
            metadata.update(diagnostic_version='v2.2_fp64_full', execution_scope='GATE1C_V22_FULL')
        b.write_json(tmp_path/'GATE1C_V2_RUN_METADATA.json', metadata)
    scores, _ = toy_scores(null=True); cache = {k: scores[k] for k in e.CACHE_FIELDS}
    labels = np.repeat(np.arange(3), 4).reshape(3, 4)
    for seed in range(3):
        for stage in range(3):
            for poe in (False, True):
                result = m.evaluate(seed, stage, ['synthetic'], [cache], [labels], include_poe=poe, height=3, width=4)
                result['metadata'] = metadata
                b.write_json(tmp_path/('poe_validation' if poe else 'reliability_units')/f'seed{seed}_stage{stage}.json', result)
    b.write_json(tmp_path/'GATE1C_V2_INPUT_AUDIT.json', dict(status='PASS', hidden_gt_training_usage='none', test_gt_usage='none', test_role_constructions=0))
    _, _, template = evidence()
    inventory = g.partition(Tiny())['inventory']
    for phase in ('draw0', 'noise', 'posterior', 'poe'):
        for pair in p['gradient_diagnostic']['batch_pairs']:
            candidates = ('PoE',) if phase == 'poe' else ('R0', 'R1', 'R2', 'R3')
            base_rows = [x for x in template if x['batch_id'] == pair['batch_id'] and x['candidate'] in candidates]
            rows = [dict(x, draw_index=d, teacher_kind='posterior_mean' if phase == 'posterior' else 'stochastic')
                    for d in (range(8) if phase in ('noise', 'poe') else [0]) for x in base_rows]
            components = [dict(x, class_id=c, component_sum_pass=True) for x in base_rows for c in range(3)] if phase in ('draw0', 'poe') else []
            result = dict(metadata=metadata, pair=pair, phase=phase, alignment=rows, class_contribution=components,
                parameter_inventory=inventory, no_optimizer=True, no_backward=True, no_parameter_grad_writes=True,
                target_probability_variance=.001, weight_variance={c: .001 for c in r.CANDIDATES}, predicted_class_change_rate=.1,
                any_draw_class_change_rate=.2, changed_predictions=[], teacher_draw_seeds=pair['teacher_draw_seeds'])
            b.write_json(tmp_path/'probes'/phase/e.pair_name(pair)/'result.json', result)
    audit = dict(status='PASS', guard_count=297, new_probe_guard_count=288, reused_validation_guard_count=9)
    status = rep.compile_report(tmp_path, p, metadata, audit)
    assert status['validation_units_completed'] == 9 and status['gradient_pairs_completed'] == 72
    assert status['teacher_draw_records_completed'] == 576 and status['gate1_overall_status'] == 'FAIL_TRANSPORT_NOT_SUPPORTED'
    assert status['method_registered'] is False and status['next_action'] == metadata.get('next_action', 'STOP_FOR_INDEPENDENT_REVIEW')
    if input_contract == 'v2.1':
        assert b.read_json(tmp_path/'GATE1C_V21_STATUS.json')['input_contract_version'] == 'v2.1'
        assert 'original v2 attempt remains incomplete' in (tmp_path/'GATE1C_V21_FINAL_REPORT.md').read_text()
        native_read = rep.read_json
        def mixed_read(path):
            value = native_read(path)
            if Path(path).parent.name == 'reliability_units':
                value['metadata']['input_contract_version'] = 'v2'
            return value
        with monkeypatch.context() as patcher:
            patcher.setattr(rep, 'read_json', mixed_read)
            with pytest.raises(b.ProtocolError, match='mixed validation provenance'):
                rep.compile_report(tmp_path, p, metadata, audit)
    if input_contract == 'v2.2':
        assert b.read_json(tmp_path/'GATE1C_V22_STATUS.json') == status
        assert status['new_probe_guards'] == 288 and status['reused_validation_guards'] == 9
        assert status['reused_validation_forwards'] == 990 and status['new_validation_forwards'] == 0
        assert status['old_gate1c_v21_status'] == 'BLOCKED_INCOMPLETE_EVIDENCE'
        assert not (tmp_path/'GATE1C_V21_STATUS.json').exists()
        assert 'v2.2 with unchanged v2.1 inputs' in (tmp_path/'GATE1C_V22_FINAL_REPORT.md').read_text()
    for name in ('RELIABILITY_DIAGNOSTIC_V2.json', 'GRADIENT_CONFLICT_DIAGNOSTIC_V2.json',
                 'TEACHER_TARGET_STOCHASTICITY_DIAGNOSTIC_V2.json', 'POE_TARGET_DIAGNOSTIC_V2.json', 'GATE1C_V2_FINAL_REPORT.md'):
        assert (tmp_path/name).is_file()
