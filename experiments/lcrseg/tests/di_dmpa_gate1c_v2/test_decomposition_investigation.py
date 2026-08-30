"""Synthetic-only inspector checks: a native failure remains a failure."""
import numpy as np
import pytest
import torch

from scripts import inspect_gate1c_decomposition as inspector
from di_dmpa_gate1c_v2 import binding as b, gradients as g
from .test_core import gradient_example, DOCS


def test_registered_pair_and_error_tolerance_are_original():
    spec = b.read_json(DOCS/(inspector.NAME+'.json'))
    parent = b.read_json(DOCS/'DI_DMPA_GATE1C_V2_PREREGISTRATION.json')
    assert spec['pair'] in parent['gradient_diagnostic']['batch_pairs']
    assert spec['native_probe']['replays_per_gpu'] == 1
    assert (spec['native_probe']['atol'], spec['native_probe']['rtol']) == (1e-6, 1e-4)
    for suffix, digest in inspector.HASHES.items():
        assert b.sha256(DOCS/f'{inspector.NAME}.{suffix}') == digest


def test_summary_keeps_failure_and_coordinate():
    row = inspector.decomposition_summary([1., 2.], [[.25, .5], [.25, .5], [.5, 1.1]])
    assert not row['original_predicate_pass'] and row['violating_coordinates'] == 1
    assert row['worst_flat_index'] == 1 and row['max_tolerance_ratio'] > 1
    assert row['atol'] == 1e-6 and row['rtol'] == 1e-4
    exact = inspector.decomposition_summary([1., 2.], [[.25, .5], [.25, .5], [.5, 1.]])
    assert exact['original_predicate_pass'] and exact['max_abs_error'] == 0
    with pytest.raises(b.ProtocolError):
        inspector.decomposition_summary([1., 2.], [[1.], [1.], [1.]])
    assert inspector.capture_failure(ValueError('unrelated failure')) is None


@pytest.mark.parametrize('normalization', g.NORMALIZATIONS)
def test_leaf_algebra_is_separate_and_detached(normalization):
    model, parts, probability, target, weights, predicted = gradient_example()
    rows = inspector.leaf_checks(dict(student_probability=probability, target=target,
        weights=weights, predicted=predicted, normalization=normalization))
    assert [r['dtype'] for r in rows] == ['torch.float32', 'torch.float64']
    assert all(r['gradient_sum']['original_predicate_pass'] for r in rows)
    assert all(r['gradient_receiver'] == 'detached_probability_leaf_only' and r['model_forwards'] == 0 for r in rows)
    assert all(p.grad is None for p in model.parameters())


def test_capture_observes_original_exception_without_bypassing(monkeypatch):
    _, parts, probability, target, weights, _ = gradient_example()
    supervised = g.vectors(g.grad(probability.square().sum(), parts), parts)
    original = g.vectors; calls = 0

    def deliberately_inconsistent(values, inventory):
        nonlocal calls
        calls += 1
        result = original(values, inventory)
        if calls == 1:
            result = {key: value+1.0 for key, value in result.items()}
        return result

    monkeypatch.setattr(g, 'vectors', deliberately_inconsistent)
    with pytest.raises(b.ProtocolError, match=inspector.ERROR) as caught:
        g.consistency_gradients(probability, target, {'R3': weights.numpy().reshape(-1)},
            parts, supervised, candidates=('R3',), decompose=True)
    captured = inspector.capture_failure(caught.value)
    assert captured['candidate'] == 'R3' and captured['normalization'] == 'pixel_normalized'
    assert captured['first_failed_block'] == 'global'
    assert not captured['block_details']['global']['original_predicate_pass']
    assert set(captured['block_details']) == {'global', *g.BLOCKS}
    assert captured['original_guard_raised'] and not captured['original_guard_replaced']
    assert not captured['native_failure_rescued']
    assert all(p.grad is None for p in parts['params'])
