"""Small synthetic end-to-end probes plus the complete evidence compiler."""
import copy
from pathlib import Path

import numpy as np
import torch

from di_dmpa_gate1_v2.features import ImmutableModels
from di_dmpa_gate1c_v2 import binding as b, execution as e, gradients as g, reliability as r, metrics as m, reporting as rep
from .test_core import Tiny, contract, toy_scores, evidence


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


def test_complete_report_compiler_with_all_72_pairs(tmp_path):
    p = contract()[0]
    metadata = dict(preregistration_commit=b.PREREG, authorization_commit=b.AUTH, diagnostic_code_commit='synthetic-no-real-checkpoints',
        model_optimizer_steps=0, transport_optimizer_steps_this_gate=0, hidden_gt_training_usage='none', test_gt_usage='none', selected_K=2, R4_available=False)
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
    audit = dict(status='PASS', guard_count=297)
    status = rep.compile_report(tmp_path, p, metadata, audit)
    assert status['validation_units_completed'] == 9 and status['gradient_pairs_completed'] == 72
    assert status['teacher_draw_records_completed'] == 576 and status['gate1_overall_status'] == 'FAIL_TRANSPORT_NOT_SUPPORTED'
    assert status['method_registered'] is False and status['next_action'] == 'STOP_FOR_INDEPENDENT_REVIEW'
    for name in ('RELIABILITY_DIAGNOSTIC_V2.json', 'GRADIENT_CONFLICT_DIAGNOSTIC_V2.json',
                 'TEACHER_TARGET_STOCHASTICITY_DIAGNOSTIC_V2.json', 'POE_TARGET_DIAGNOSTIC_V2.json', 'GATE1C_V2_FINAL_REPORT.md'):
        assert (tmp_path/name).is_file()
