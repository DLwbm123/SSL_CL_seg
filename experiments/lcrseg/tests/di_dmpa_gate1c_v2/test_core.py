"""Synthetic mathematical, contract, leakage and zero-update checks."""
import ast
import copy
import csv
import inspect
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from scipy.special import expit, logsumexp

from di_dmpa_gate1_v2.features import ImmutableModels, split_support
from di_dmpa_jascl.modeling import pas_probability_objective
from di_dmpa_gate1c_v2 import binding as b, reliability as r, metrics as m, gradients as g, execution as e, reporting as rep

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT/'docs/di_dmpa_jascl'


def contract():
    return tuple(b.read_json(DOCS/f) for f in ('DI_DMPA_GATE1C_V2_PREREGISTRATION.json', 'DI_DMPA_GATE1_PREREGISTRATION.json',
        'GATE1C_V2_EXECUTION_AUTHORIZATION.json', 'GATE1B_V2_FREEZE.json', 'GATE1A_V2_FREEZE.json'))


def toy_scores(*, history=True, null=False):
    z = np.eye(16)[:3].repeat(4, axis=0)
    if null:
        z[0] = 0
    p = np.full((12, 3), .1)
    p[np.arange(12), np.repeat(np.arange(3), 4)] = .8
    current = np.repeat(np.eye(16)[:3, None], 2, axis=1)
    past = current if history else np.empty((3, 0, 16))
    return r.score_arrays(p, z, np.ones(12, bool), current, past), (p, z, current, past)


class Head(nn.Module):
    def __init__(self):
        super().__init__(); self.mu = nn.Conv2d(16, 3, 1, bias=False); self.sigma = nn.Conv2d(16, 3, 1, bias=False)
        self.grad_update = nn.Parameter(torch.zeros_like(self.mu.weight))
    def forward(self, x, stochastic):
        weight = self.mu.weight+.02*torch.randn_like(self.mu.weight) if stochastic else self.mu.weight
        return torch.nn.functional.conv2d(x, weight)


class Tiny(nn.Module):
    """Same parameter prefixes/inactive contract, entirely synthetic tensors."""
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 16, 1); self.enc2 = nn.Conv2d(16, 16, 1); self.enc3 = nn.Conv2d(16, 16, 1)
        self.bottleneck = nn.Conv2d(16, 16, 1); self.decoder = nn.Module()
        self.decoder.dec3 = nn.Conv2d(16, 16, 1); self.decoder.dec2 = nn.Conv2d(16, 16, 1)
        self.decoder.dec1 = nn.Conv2d(16, 16, 1); self.decoder.conv_logit = Head()
    def forward(self, x, *, stochastic_classifier):
        for layer in (self.enc1, self.enc2, self.enc3, self.bottleneck, self.decoder.dec3, self.decoder.dec2, self.decoder.dec1):
            x = torch.tanh(layer(x))
        return self.decoder.conv_logit(x, stochastic_classifier), x


def gradient_example():
    torch.manual_seed(7); model = Tiny().eval(); parts = g.partition(model)
    logits, _ = model(torch.randn(2, 3, 8, 8), stochastic_classifier=True)
    p = logits.softmax(1); target = torch.softmax(torch.randn_like(logits), 1).detach()
    weights = torch.ones((2, 8, 8), dtype=torch.float64); predicted = target.argmax(1)
    return model, parts, p, target, weights, predicted


def evidence():
    classwise = []; guards = []; gradients = []
    for candidate in ('R0', 'R1', 'R2', 'R3', 'PoE'):
        for s in range(3):
            for t in range(3):
                for c in (1, 2):
                    classwise.append(dict(candidate=candidate, seed=s, stage_index=t, class_id=c, common_upper_bound=.5,
                        common_support_AURC=.8, reference_common_support_AURC=1., actual_shared_points=list(m.POINTS),
                        precision_points=[dict(requested=x, precision=.8) for x in m.POINTS],
                        reference_points=[dict(requested=x, precision=.7) for x in m.POINTS]))
                    guards.append(dict(candidate=candidate, seed=s, stage_index=t, class_id=c, global_operating_point=.5,
                        candidate_retained_fraction=1., reference_retained_fraction=1., pass_=True))
                for i in range(8):
                    for norm in g.NORMALIZATIONS:
                        for block in ('global', *g.BLOCKS):
                            cos = (-.2 if i < 4 else .2) if candidate == 'R1' else .2
                            gradients.append(dict(candidate=candidate, normalization=norm, seed=s, stage_index=t, pair_index=i,
                                block=block, cosine=cos, batch_id=f'B0/seed{s}/stage{t}/{b.DOMAINS[t]}/pair{i:02d}', draw_index=0,
                                teacher_kind='stochastic', domain=b.DOMAINS[t], zero_gradient=False, supervised_zero=False,
                                unsupervised_zero=False, supervised_norm=1., unsupervised_norm=1., norm_ratio=.5,
                                negative_cosine=cos < 0, undefined_reason=None, loss=1.))
    return classwise, guards, gradients


def condition(data=None, norm='pixel_normalized'):
    return rep.candidate_conditions(*(evidence() if data is None else data), 'R3', norm)


def test_01_gate1b_freeze_report_receipt_binding():
    for name, sha in b.FROZEN_HASHES.items():
        b.check_hash(DOCS/name, sha)
    p, old, auth, frozen, af = contract(); b.validate_contract(p, old, auth, frozen, af)
    assert frozen['B1_B7'] == b.read_json(DOCS/'GATE1B_V2_STATUS.json')['B1_B7']


@pytest.mark.parametrize('field,value', [('transport_status', 'PASS'), ('transport_optimizer_steps', 6001),
    ('further_transport_attempts_authorized', True), ('T1_rescue_allowed', True), ('drift_calibrated_claim_allowed', True)])
def test_02_transport_failure_immutable(field, value):
    p, old, auth, frozen, af = contract(); frozen[field] = value
    with pytest.raises(b.ProtocolError): b.validate_contract(p, old, auth, frozen, af)


def test_03_r4_unavailable():
    p, old, auth, frozen, af = contract(); p['gate1c']['candidates']['R4'] = 'forbidden'
    with pytest.raises(b.ProtocolError): b.validate_contract(p, old, auth, frozen, af)


@pytest.mark.parametrize('transform', ['T1', 'T2', 'learned', 'identity_then_T2'])
def test_04_transformed_banks_rejected(transform):
    with pytest.raises(b.ProtocolError): r.banks(contract()[-1], 0, 1, transform=transform)


@pytest.mark.parametrize('k', [1, 3, 5])
def test_05_selected_k_exactly_two(k):
    af = contract()[-1]; af['selected_K'] = k
    with pytest.raises(b.ProtocolError): r.banks(af, 0, 0)


@pytest.mark.parametrize('field,value', [('panel', 'B0-student'), ('baseline', 'C0'), ('training_source', 'val'),
    ('feature_source', 'student'), ('operational_refit_allowed', True), ('active_mask', [True, False])])
def test_06_b0_ema_original_bank_only(field, value):
    af = contract()[-1]; af['prototype_records'][0][field] = value
    with pytest.raises(b.ProtocolError): r.banks(af, 0, 0)


def test_07_bank_membership_and_original_order():
    af = contract()[-1]
    for stage in range(3):
        current, history = r.banks(af, 0, stage)
        assert current.shape == (3, 2, 16) and history.shape == (3, 2*stage, 16)
        ref = next(x for x in af['prototype_records'] if (x['seed'], x['stage_index'], x['class_id']) == (0, stage, 1))
        assert np.array_equal(current[1], ref['centers']) and not current.flags.writeable


def test_08_stage0_r3_exactly_r2():
    scores, _ = toy_scores(history=False, null=True)
    assert np.array_equal(scores['R2'], scores['R3'])


def test_09_stage1_one_historical_domain():
    af = contract()[-1]
    assert r.bank_identity(af, 2, 1)['history'] == ['REFUGE']
    assert r.banks(af, 2, 1)[1].shape == (3, 2, 16)


def test_10_stage2_two_historical_domains():
    af = contract()[-1]
    assert r.bank_identity(af, 1, 2)['history'] == ['REFUGE', 'RIM_ONE_r3']
    assert np.array_equal(r.banks(af, 1, 2)[1][:, :2], r.banks(af, 1, 0)[0])


def test_11_logmeanexp_count_invariance():
    scores, (_, z, current, _) = toy_scores()
    x, _ = r.class_scores(z, current); y, _ = r.class_scores(z, np.repeat(current, 3, axis=1))
    np.testing.assert_allclose(x, y, atol=2e-14, rtol=0)


def test_12_null_ema_zero_prototype_weights():
    scores, _ = toy_scores(null=True)
    assert scores['R0'][0] > 0 and scores['R1'][0]
    assert scores['R2'][0] == scores['R3'][0] == 0 and not scores['prototype_valid'][0]
    for field in ('current_scores', 'history_scores', 'current_margin', 'history_margin', 'history_gate'):
        assert np.isnan(scores[field][0]).all()


def test_13_null_not_normalized_or_epsilon_shifted():
    z = np.zeros((3, 16)); z[1, 0] = 1e-13; z[2, 0] = 2e-12
    a = split_support(z)
    assert a['active_mask'].tolist() == [False, False, True]
    assert not a['directions'][:2].any() and a['directions'][2, 0] == 1


def test_14_null_remains_in_coverage_denominator():
    mass = m.case_weights(np.zeros(4, int), np.ones(4, bool))
    curve = m.ranked_curve(np.array([0, 1, 1, 1]), np.ones(4, bool), mass, m.tie_keys(0, 0, ['x'], 1, 4))
    assert curve['maximum'] == .75 and mass.sum() == 1


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('field', ['feature', 'logit'])
def test_15_nonfinite_evidence_blocks(field, bad):
    if field == 'feature':
        _, (p, z, cur, past) = toy_scores(); z[0, 0] = bad
        with pytest.raises(b.NonfiniteEvidence): r.score_arrays(p, z, np.ones(12, bool), cur, past)
    else:
        logit = torch.zeros(1, 3, 2, 2); logit[0, 0, 0, 0] = bad
        with pytest.raises(b.NonfiniteEvidence): r.legacy_pas(logit, torch.ones(1, 16, 2, 2), logit, torch.ones(1, 16, 2, 2), torch.ones(3, 16))


def test_16_r0_formula():
    scores, (p, _, _, _) = toy_scores()
    assert np.array_equal(scores['R0'], p.max(1).astype(np.float32))


class Cached(nn.Module):
    def __init__(self, outputs): super().__init__(); self.outputs = outputs
    def forward(self, images, *, stochastic_classifier):
        assert stochastic_classifier is True
        return self.outputs


def test_17_r1_exact_frozen_gate0_pas_regression():
    torch.manual_seed(2); sl = torch.randn(2, 3, 8, 8, requires_grad=True); tl = torch.randn(2, 3, 8, 8)
    sf = torch.randn(2, 16, 8, 8); tf = torch.randn(2, 16, 8, 8); legacy = torch.randn(3, 16)
    _, _, _, official = pas_probability_objective(Cached((sl, sf)), Cached((tl, tf)), None, legacy)
    assert torch.equal(r.legacy_pas(sl, sf, tl, tf, legacy), official)


@pytest.mark.parametrize('confidence,similarity,expected', [(.7, 1., False), (.700001, .7, False), (.700001, .700001, True)])
def test_18_r1_strict_greater_than_boundaries(confidence, similarity, expected):
    logits = torch.zeros(1, 3, 1, 1); features = torch.ones(1, 16, 1, 1); legacy = torch.ones(3, 16)
    p = torch.tensor([confidence, (1-confidence)/2, (1-confidence)/2]).reshape(1, 3, 1, 1)
    with patch.object(torch.Tensor, 'softmax', return_value=p), patch('torch.nn.functional.cosine_similarity', return_value=torch.tensor([[[similarity]]])):
        assert bool(r.legacy_pas(logits, features, logits, features, legacy).item()) == expected


def test_19_r2_formula():
    scores, (p, _, _, _) = toy_scores()
    np.testing.assert_array_equal(scores['R2'], p.max(1)*expit(scores['current_margin']/.1))


def test_20_r3_formula():
    scores, _ = toy_scores(); gate = expit((scores['history_similarity']-.3)/.1)
    np.testing.assert_array_equal(scores['history_gate'], gate)
    np.testing.assert_array_equal(scores['R3'], scores['R2']*((1-gate)+gate*expit(scores['history_margin']/.1)))


def test_21_history_gate_bounded():
    scores, _ = toy_scores(); assert np.all((scores['history_gate'] >= 0) & (scores['history_gate'] <= 1))


def test_22_r3_not_above_r2():
    scores, _ = toy_scores(null=True); assert np.all(scores['R3'] <= scores['R2'])


def test_23_missing_history_is_neutral():
    _, (p, z, cur, past) = toy_scores()
    scores = r.score_arrays(p, z, np.ones(12, bool), cur, past, history_valid=np.zeros((3, 2), bool))
    assert np.array_equal(scores['R3'], scores['R2']) and np.all(scores['history_gate'] == 0)


def test_24_reliability_bounded_and_missing_current_invalid():
    _, (p, z, cur, past) = toy_scores(); valid = np.ones((3, 2), bool); valid[0] = False
    scores = r.score_arrays(p, z, np.ones(12, bool), cur, past, current_valid=valid)
    assert not scores['prototype_valid'][:4].any() and not scores['R2'][:4].any()
    assert all(np.all((scores[k] >= 0) & (scores[k] <= 1)) for k in r.CANDIDATES)


def test_25_tie_break_full_sha_deterministic():
    keys = m.tie_keys(1, 2, ['case"with_quote'], 2, 3)
    assert [row.tobytes().hex() for row in keys] == [b.H(['reliability-tie-v1', 1, 2, 'case"with_quote', y, x]) for y in range(2) for x in range(3)]
    curve = m.ranked_curve(np.ones(6), np.arange(6) % 2 == 0, np.ones(6)/6, keys)
    assert curve['pos'].tolist() == sorted(range(6), key=lambda i: keys[i].tobytes())


def test_26_unsupported_coverage_explicit():
    curve = m.ranked_curve(np.array([1., 0.]), np.ones(2, bool), np.ones(2)/2, m.tie_keys(0, 0, ['x'], 1, 2))
    assert m.point(curve, .6)['precision'] is None and m.point(curve, .6)['reason'] == 'OUTSIDE_POSITIVE_SUPPORT'


def test_27_right_continuous_weighted_common_aurc():
    curve = m.ranked_curve(np.array([3., 2., 1.]), np.array([1, 0, 1], bool), np.array([.2, .3, .5]), m.tie_keys(0, 0, ['x'], 1, 3))
    assert m.aurc(curve, .4) == pytest.approx((.2*0+.2*.6)/.4)
    assert m.aurc(curve, 1.) == pytest.approx(.3*.6+.5*.3)


def test_28_no_coverage_extrapolation_or_structural_null_imputation():
    curve = m.ranked_curve(np.zeros(2), np.ones(2, bool), np.ones(2)/2, m.tie_keys(0, 0, ['x'], 1, 2))
    assert m.aurc(curve, .05) is None and m.point(curve, .05)['reason'] == 'EMPTY_POSITIVE_SUPPORT'
    assert m.composition(curve, .05, np.zeros(2), np.zeros(2), np.ones(2)/2)[0]['predicted_retained_fraction'] is None


def test_29_class_case_balancing():
    ci = np.array([0, 0, 1, 1, 1, 1]); w = m.case_weights(ci, np.ones(6, bool))
    assert w[:2].sum() == w[2:].sum() == .5
    w = m.case_weights(ci, np.array([1, 0, 1, 1, 1, 1], bool))
    assert w[0] == .5 and w[1] == 0 and w[2:].sum() == .5


def test_30_ece_fifteen_bins_and_empty_bins():
    ece, bins = m.calibration(np.array([0., .5, 1.]), np.array([0, 1, 1]), np.ones(3)/3)
    assert len(bins) == 15 and sum(x['empty'] for x in bins) == 12
    assert ece == pytest.approx(1/6) and bins[-1]['accuracy'] == 1


def test_31_multiclass_brier_not_binary():
    p = np.array([[.8, .1, .1], [.2, .3, .5]])
    assert m.brier(p, np.array([0, 1]), np.ones(2)/2) == pytest.approx((.06+.78)/2)


def test_32_exact_72_pair_case_checkpoint_and_seed_identity():
    p, old, auth, frozen, af = contract(); pairs = p['gradient_diagnostic']['batch_pairs']
    assert len(pairs) == 72 and pairs == old['gradient_diagnostic']['batch_pairs']
    pairs[0]['labeled_case_ids'].reverse()
    with pytest.raises(b.ProtocolError): b.validate_contract(p, old, auth, frozen, af)


def test_33_shared_target_for_all_candidates():
    model, parts, ps, target, weights, pred = gradient_example(); original = target.clone()
    sup = g.vectors(g.grad(ps.square().sum(), parts), parts)
    rows, _, _ = g.consistency_gradients(ps, target, {c: weights.numpy().reshape(-1) for c in r.CANDIDATES}, parts, sup)
    assert len(rows) == 4*2*7 and torch.equal(target, original) and all(p.grad is None for p in model.parameters())


def test_34_teacher_no_grad():
    model, parts, ps, target, weights, pred = gradient_example(); leaf = target.clone().requires_grad_()
    g.grad(g.objective(ps, leaf, weights, pred, 'pixel_normalized'), parts)
    assert leaf.grad is None


def test_35_prototype_and_history_no_grad():
    af = contract()[-1]; cur, hist = r.banks(af, 0, 2)
    assert not cur.flags.writeable and not hist.flags.writeable
    with pytest.raises(ValueError): hist[0, 0, 0] = 0


def test_36_student_consistency_gradient_exists():
    model, parts, ps, target, weights, pred = gradient_example()
    values = g.grad(g.objective(ps, target, weights, pred, 'pixel_normalized'), parts)
    assert np.linalg.norm(g.vectors(values, parts)['global']) > 0


def test_37_zero_weight_is_graph_connected():
    model, parts, ps, target, weights, pred = gradient_example()
    loss = g.objective(ps, target, weights*0, pred, 'pixel_normalized'); assert loss.requires_grad and loss.item() == 0
    assert not g.vectors(g.grad(loss, parts), parts)['global'].any()


def test_38_undefined_zero_gradient_retained():
    x = g.alignment(np.ones(4), np.zeros(4)); assert x['cosine'] is None and x['zero_gradient'] and x['norm_ratio'] == 0
    assert g.summary([None, .2])['population_variance'] is None


def test_39_block_partition_complete_or_block():
    model = Tiny(); parts = g.partition(model)
    assert sorted(i for block in parts['blocks'].values() for i in block) == parts['active']
    model.unregistered_branch = nn.Linear(2, 2)
    with pytest.raises(b.GradientPartitionError): g.partition(model)


def test_40_inactive_parameters_inventoried():
    model, parts, ps, target, weights, pred = gradient_example(); values = g.grad(ps.sum(), parts)
    assert {parts['names'][i] for i, v in enumerate(values) if v is None} == g.INACTIVE
    assert len(parts['inventory']) == len(list(model.named_parameters()))


def test_41_no_backward_optimizer_or_parameter_grad_write():
    with b.no_updates():
        with pytest.raises(b.ProtocolError): torch.ones(1, requires_grad=True).sum().backward()
        with pytest.raises(b.ProtocolError): torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=.1)
        model, parts, ps, target, weights, pred = gradient_example(); g.grad(ps.sum(), parts)
        assert all(p.grad is None for p in model.parameters())
    for file in (ROOT/'di_dmpa_gate1c_v2').glob('*.py'):
        calls = [n for n in ast.walk(ast.parse(file.read_text())) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        assert not any(n.func.attr in ('backward', 'step', 'Adam', 'SGD', 'update_teacher') for n in calls)


def test_42_class_gradient_vectors_decompose():
    _, parts, ps, target, weights, pred = gradient_example(); sup = g.vectors(g.grad(ps.square().sum(), parts), parts)
    rows, components, _ = g.consistency_gradients(ps, target, {'R3': weights.numpy().reshape(-1)}, parts, sup, candidates=('R3',), decompose=True)
    assert len(components) == 2*7*3 and all(x['component_sum_pass'] for x in components)
    assert all(x['total_norm'] <= x['nonadditive_norm_sum']+1e-12 for x in components)


def test_43_class_balanced_formula_and_zero():
    _, _, ps, target, weights, pred = gradient_example(); weights = weights.clone(); weights[pred == 0] = .3
    loss = (ps-target).square().sum(1)
    expected = torch.stack([(weights[pred == c]*loss[pred == c]).sum()/(weights[pred == c].sum()+1e-12) for c in range(3) if (pred == c).any()]).mean()
    assert torch.equal(g.objective(ps, target, weights, pred, 'class_balanced'), expected)
    assert g.objective(ps, target, weights*0, pred, 'class_balanced').requires_grad


def test_44_exact_eight_teacher_draw_seeds():
    p = contract()[0]
    for pair in p['gradient_diagnostic']['batch_pairs']:
        assert pair['teacher_draw_seeds'] == [b.S(['teacher-draw-v1', pair['batch_id'], i]) for i in range(8)]


def test_45_posterior_mean_not_primary():
    p = contract()[0]
    assert p['teacher_noise']['future_posterior_mean_target_requires_extra_baseline']
    assert p['gradient_execution']['primary_draw'] == 0
    assert 'cannot replace B0 or rescue R3' in p['teacher_noise']['posterior_mean']


def test_46_poe_formula_exponents_and_null():
    scores, _ = toy_scores(null=True); ctrl = r.poe_target(scores); active = scores['active_mask']
    pc = np.exp(scores['current_scores'][active]-logsumexp(scores['current_scores'][active], axis=1, keepdims=True))
    ph = np.exp(scores['history_scores'][active]-logsumexp(scores['history_scores'][active], axis=1, keepdims=True))
    expected = scores['teacher_probability'][active]*pc**.5*ph**(.25*scores['history_gate'][active, None]); expected /= expected.sum(1, keepdims=True)
    np.testing.assert_allclose(ctrl['probability'][active], expected, atol=2e-15, rtol=1e-14)
    assert not ctrl['valid'][0] and ctrl['weights'][0] == 0 and not ctrl['probability'][0].any()


def test_47_poe_cannot_rescue():
    a = dict(candidate='R3', all_pass=False)
    assert rep.select(a, a, {'PoE': True})['reliability_status'].startswith('FAIL_')


def test_48_current_only_cannot_rescue():
    a = dict(candidate='R3', all_pass=False)
    assert rep.select(a, a, {'R2': True})['reduced_candidate_status'] == 'NOT_ELIGIBLE'


def test_49_class_balanced_independent_admission():
    no = dict(candidate='R3', all_pass=False); yes = dict(candidate='R3', all_pass=True)
    assert rep.select(no, yes)['selected_normalization'] == 'CLASS_BALANCED'
    assert rep.select(yes, yes)['selected_normalization'] == 'PIXEL_NORMALIZED'
    assert rep.select(yes, yes)['gate1_overall_status'] == 'FAIL_TRANSPORT_NOT_SUPPORTED'


@pytest.mark.parametrize('value,expected', [(.9, True), (np.nextafter(.9, 1.), False)])
def test_50a_c1_exact_aurc_boundary(value, expected):
    data = evidence()
    for x in data[0]:
        if x['candidate'] == 'R3':
            x['common_support_AURC'] = value; x['precision_points'] = copy.deepcopy(x['reference_points'])
    assert condition(data)['C1_C8']['C1']['pass_'] == expected


@pytest.mark.parametrize('improvements,expected', [(12, True), (11, False)])
def test_50b_c2_exact_count(improvements, expected):
    data = evidence(); units = [x for x in data[0] if x['candidate'] == 'R3']
    for x in units[improvements:]:
        x['common_support_AURC'] = 1.; x['precision_points'] = copy.deepcopy(x['reference_points'])
    assert condition(data)['C1_C8']['C2']['pass_'] == expected


@pytest.mark.parametrize('value,expected', [(.8, True), (np.nextafter(.8, 0.), False)])
def test_50c_c3_retained_boundary_recomputed(value, expected):
    data = evidence(); next(x for x in data[1] if x['candidate'] == 'R3')['candidate_retained_fraction'] = value
    assert condition(data)['C1_C8']['C3']['pass_'] == expected


@pytest.mark.parametrize('neg,expected', [(28, True), (29, False)])
def test_50d_c4_negative_fraction_boundary(neg, expected):
    data = evidence(); rows = [x for x in data[2] if x['candidate'] == 'R3' and x['normalization'] == 'pixel_normalized' and x['block'] == 'global']
    for i, x in enumerate(rows): x['cosine'] = -.1 if i < neg else .2
    assert condition(data)['C1_C8']['C4']['pass_'] == expected


def test_50e_reference_zero_negative_fraction_never_improves():
    data = evidence()
    for x in data[2]:
        if x['candidate'] == 'R1': x['cosine'] = .1
    assert not condition(data)['C1_C8']['C4']['pass_']


@pytest.mark.parametrize('value,expected', [(.05, True), (np.nextafter(.05, 0.), False)])
def test_50f_c5_median_boundary(value, expected):
    data = evidence()
    for x in data[2]:
        if x['candidate'] == 'R3': x['cosine'] = value
    assert condition(data)['C1_C8']['C5']['pass_'] == expected


@pytest.mark.parametrize('value,expected', [(-.05, True), (np.nextafter(-.05, -1.), False)])
def test_50g_c6_domain_boundary(value, expected):
    data = evidence()
    for x in data[2]:
        if x['candidate'] == 'R3' and x['stage_index'] == 1: x['cosine'] = value
    assert condition(data)['C1_C8']['C6']['pass_'] == expected


def test_50h_c7_c8_and_required_undefined_fail():
    data = evidence(); a = rep.candidate_conditions(*data, 'R3', 'pixel_normalized', immutable=False, leakage=False)
    assert not a['C1_C8']['C7']['pass_'] and not a['C1_C8']['C8']['pass_']
    next(x for x in data[2] if x['candidate'] == 'R3' and x['normalization'] == 'pixel_normalized' and x['block'] == 'global')['cosine'] = None
    a = condition(data)
    assert not any(a['C1_C8'][k]['pass_'] for k in ('C4', 'C5', 'C6'))


def test_51_incomplete_unit_fails_closed():
    data = evidence(); data[0].remove(next(x for x in data[0] if x['candidate'] == 'R3'))
    with pytest.raises(b.IncompleteEvidence): condition(data)


def test_52_test_and_hidden_roles_forbidden(tmp_path):
    with pytest.raises(b.ProtocolError): b.records(tmp_path, contract()[0], 0, 0, 'test')
    with pytest.raises(b.ProtocolError): e.visible_labels([], tmp_path, role='train_unlabeled')
    with pytest.raises(b.ProtocolError): b.safe_asset(tmp_path, '../hidden_GT')


def test_53_gt_free_builder_and_image_objects():
    assert not {'GT', 'label', 'labels', 'target_gt'} & set(inspect.signature(r.build).parameters)
    row = dict(case_id='x', image_h5_relpath='x.h5', image_sha256='a'*64, label_h5_relpath='SECRET', label_sha256='b'*64)
    assert set(b.image_only(row)) == {'case_id', 'image_h5_relpath', 'image_sha256'}
    assert 'visible_labels' not in inspect.getsource(e.validation_unit)


def test_54_model_checkpoint_immutability(tmp_path):
    model = Tiny(); cp = tmp_path/'synthetic.pt'; torch.save(model.state_dict(), cp)
    desc = dict(path=str(cp), sha256=b.sha256(cp), checkpoint_id='synthetic')
    with ImmutableModels({'student': model}, desc, tmp_path/'good', {}): pass
    with pytest.raises(b.ModelMutation):
        with ImmutableModels({'student': model}, desc, tmp_path/'bad', {}):
            with torch.no_grad(): next(model.parameters()).add_(1)
    assert b.read_json(tmp_path/'bad/immutability/synthetic.json')['status'] == 'BLOCKED_MODEL_MUTATION'


def test_55_artifact_manifest_and_cache_tamper(tmp_path):
    desc = b.save_arrays(tmp_path/'a.npz', {'x': np.arange(5)})
    assert np.array_equal(b.read_arrays(desc)['x'], np.arange(5))
    manifest = rep.artifact_manifest(tmp_path)
    assert manifest['file_count'] == 1 and manifest['artifacts'][0]['sha256'] == desc['sha256']
    desc['sha256'] = 'f'*64
    with pytest.raises(b.ProtocolError): b.read_arrays(desc)


def test_56_report_compiler_primary_cannot_be_promoted_from_control():
    a = condition(); assert a['all_pass']
    assert rep.select(a, a)['selected_reliability'] == 'R3_IDENTITY_HISTORY_WEIGHT_ONLY'
    with pytest.raises(b.ProtocolError): rep.select(dict(candidate='R2', all_pass=True), a)


def test_empty_shared_precision_branch_cannot_pass():
    data = evidence()
    for x in data[0]:
        if x['candidate'] == 'R3': x['actual_shared_points'] = []; x['common_support_AURC'] = 1.
    a = condition(data)['C1_C8']['C1']; assert a['shared_points'] == [] and not a['pass_']


def test_evaluator_all_rows_null_denominator_and_no_gt_ties():
    scores, _ = toy_scores(null=True)
    cache = {k: scores[k] for k in e.CACHE_FIELDS}
    labels = np.repeat(np.arange(3), 4).reshape(3, 4)
    out = m.evaluate(0, 1, ['case'], [cache], [labels], include_poe=True, height=3, width=4)
    row = next(x for x in out['classwise'] if x['candidate'] == 'R3' and x['class_id'] == 'overall')
    assert row['full_valid_pixel_count'] == 12 and row['null_count'] == 1 and row['null_case_balanced_mass'] == pytest.approx(1/12)
    assert row['maximum_supported_coverage'] == pytest.approx(11/12)
    assert out['poe']['available_mass'] == pytest.approx(11/12)
