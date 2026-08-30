"""Synthetic reference checks; no real images, checkpoints or forward seeds."""
import copy

import numpy as np
import pytest
import torch

from scripts import reference_gate1c_fp64 as ref
from di_dmpa_gate1c_v2 import binding as b, gradients as g
from .test_core import Tiny, DOCS


def test_prospective_reference_contract():
    spec = b.read_json(DOCS/f'{ref.NAME}.json')
    assert spec['fixed_inputs']['maximum_model_forward_calls_total'] == 8
    assert spec['reference']['predicate'] == dict(atol=1e-6, rtol=1e-4, reference_operand='class_component_sum', unchanged=True)
    assert not spec['boundaries']['full_gate_retry_authorized'] and not spec['boundaries']['method_registered']
    for suffix, digest in ref.HASHES.items(): b.check_hash(DOCS/f'{ref.NAME}.{suffix}', digest)


def test_observer_preserves_native_draws_and_rng():
    like = torch.zeros(3, 16, 3, 3)
    torch.manual_seed(83); expected = [torch.randn_like(like) for _ in range(3)]; after = ref.rng_hash()
    torch.manual_seed(83)
    with ref.observe_draws(like.shape) as captured:
        returned = [torch.randn_like(like) for _ in range(3)]
        with pytest.raises(b.ProtocolError): torch.randn_like(like)
    assert ref.rng_hash() == after
    assert all(torch.equal(x, y) and torch.equal(x, z) for x, y, z in zip(expected, returned, captured))


def test_replay_is_same_values_not_a_new_float64_draw():
    draw = torch.randn(3, 16, 3, 3); before = ref.rng_hash()
    with ref.replay_draw(draw):
        actual = torch.randn_like(draw.double())
        with pytest.raises(b.ProtocolError): torch.randn_like(draw.double())
    assert torch.equal(actual, draw.double()) and before == ref.rng_hash()
    with pytest.raises(b.ProtocolError):
        with ref.replay_draw(draw): pass
    with pytest.raises(b.ProtocolError):
        with ref.replay_draw(draw): torch.randn_like(draw)


def test_isolated_reference_uses_original_objective_and_never_mutates():
    torch.manual_seed(61); student = Tiny().eval(); x = torch.randn(2, 3, 8, 8)
    with ref.observe_draws(student.decoder.conv_logit.mu.weight.shape) as draws:
        logits, features = student(x, stochastic_classifier=True)
        with torch.no_grad():
            target = student(x, stochastic_classifier=True)[0].softmax(1)
            student(x, stochastic_classifier=True)
    parts = g.partition(student); probability = logits.softmax(1)
    weights = torch.linspace(.1, 1., 128, dtype=torch.float64).reshape(2, 8, 8)
    predicted = target.argmax(1)
    kwargs = dict(probability=probability, target=target, weights=weights, predicted=predicted, normalization='class_balanced')
    vector = g.vectors(g.grad(g.objective(**kwargs), parts), parts)
    components = [g.vectors(g.grad(g.objective(**kwargs, class_component=c), parts), parts) for c in range(3)]
    native = dict(student_probability=probability, target=target, weights=weights, predicted=predicted, vector=vector, class_vectors=components)
    with b.no_updates():
        details = ref.reference(dict(models={'student': student}, xu=x, sl=logits, sf=features), native, draws[0])
    assert details['rng_before'] == details['rng_after']
    assert details['shadow_state_before'] == details['shadow_state_after']
    assert set(details['block_details']) == {'global', *g.BLOCKS}
    assert all(x['original_predicate_pass'] for x in details['block_details'].values())
    assert all(p.grad is None for p in student.parameters())
    spec = b.read_json(DOCS/f'{ref.NAME}.json')
    assert ref.interpretation(details, spec) == 'SAME_PAIR_FP64_NUMERICAL_REFERENCE_SUPPORTED'
    changed = copy.deepcopy(details); changed['native_reference_gradient_comparisons']['global']['total']['relative_l2'] = .01
    assert ref.interpretation(changed, spec) == 'HIGH_PRECISION_DECOMPOSITION_ONLY_NONCOMPARABLE'
    changed = copy.deepcopy(details); changed['block_details']['encoder']['original_predicate_pass'] = False
    assert ref.interpretation(changed, spec) == 'FP64_REFERENCE_NOT_SUPPORTING_HYPOTHESIS'


def test_exception_path_bank_and_teacher_checks_are_explicit():
    models = {'student': Tiny(), 'ema_teacher': Tiny().requires_grad_(False)}
    legacy = torch.ones(3, 16); current = np.ones((3, 2, 16)); history = np.ones((3, 4, 16))
    unit = dict(models=models, legacy=legacy, current=current, history=history,
        legacy_before=b.tensor_hash(legacy), before=(b.array_hash(current), b.array_hash(history)))
    receipt = ref.after_error_isolation(unit)
    assert receipt['legacy_before'] == receipt['legacy_after']
    legacy[0, 0] = 2
    with pytest.raises(b.ProtocolError): ref.after_error_isolation(unit)
    legacy[0, 0] = 1; current[0, 0, 0] = 2
    with pytest.raises(b.ProtocolError): ref.after_error_isolation(unit)
