"""Synthetic-only precision, frozen-native parity and failure-audit checks."""
import copy
import subprocess
import types
from unittest.mock import patch

import numpy as np
import pytest
import torch

from di_dmpa_gate1.feature_extraction import state_hash
from di_dmpa_gate1_v2.features import ImmutableModels
from di_dmpa_jascl.modeling import pas_probability_objective
from di_dmpa_gate1c_v2 import binding as b, execution as e, gradients as g, precision as pr, precision_pilot as pilot, reliability as r
from .test_core import Tiny, contract, ROOT, DOCS


def test_pilot_contract_and_exact_code_receipt(record_testsuite_property):
    spec = b.read_json(DOCS/f'{pilot.NAME}.json')
    for suffix, digest in pilot.HASHES.items(): b.check_hash(DOCS/f'{pilot.NAME}.{suffix}', digest)
    assert spec['pilot']['phases'] == list(pilot.PHASES)
    assert len(spec['pilot']['pairs']) == 3 and all(q in contract()[0]['gradient_diagnostic']['batch_pairs'] for q in spec['pilot']['pairs'])
    assert [sum(x[i] for x in pilot.COUNTS.values())*3 for i in range(4)] == [51, 24, 276, 366]
    assert spec['scope'] == 'PILOT_ONLY_NOT_FULL_GATE1C' and not spec['boundaries']['method_registered']
    assert not spec['boundaries']['full_gate_retry_authorized'] and not spec['inherited']['scientific_admission_in_pilot']
    record_testsuite_property('diagnostic_code_commit', b.git(ROOT.parents[1], 'rev-parse', 'HEAD'))
    record_testsuite_property('source_clean', str(not b.git(ROOT.parents[1], 'status', '--porcelain')).lower())


def setup_pair(monkeypatch, *, shadow):
    p, _, _, _, freeze = contract(); pair = next(q for q in p['gradient_diagnostic']['batch_pairs'] if q['stage_index'] == 1)
    torch.manual_seed(71); student = Tiny().eval(); teacher = copy.deepcopy(student).requires_grad_(False).eval()
    models = dict(student=student, ema_teacher=teacher); legacy = torch.ones(3, 16)
    generator = torch.Generator().manual_seed(19)
    xu = torch.randn(2, 3, 8, 8, generator=generator); xl = torch.randn(2, 3, 8, 8, generator=generator)
    labels = torch.arange(128).reshape(2, 8, 8) % 3
    monkeypatch.setattr(e, 'pair_inputs', lambda *args: (xu.clone(), xl.clone(), labels.clone()))
    if shadow:
        p.update(diagnostic_precision='float64_shadow', _precision_contract_verified=True)
        pr.attach_gradient_student(models, p)
    return p, pair, freeze, models, legacy, (xu, xl, labels)


def test_shadow_same_draw_state_rng_and_authority(monkeypatch):
    p, pair, _, models, _, inputs = setup_pair(monkeypatch, shadow=False)
    with pytest.raises(b.ProtocolError, match='unregistered'):
        pr.attach_gradient_student(models, {'diagnostic_precision': 'float64_shadow'})
    state = state_hash(models['student'].state_dict()); rng = pr.rng_hash()
    p.update(diagnostic_precision='float64_shadow', _precision_contract_verified=True)
    pr.attach_gradient_student(models, p)
    assert state == state_hash(models['student'].state_dict()) and rng == pr.rng_hash()
    assert all(a.data_ptr() != z.data_ptr() for a, z in zip(models['student'].parameters(), models['gradient_student'].parameters()))
    seed = pair['forward_seeds']['student_unlabeled']
    old_logits, old_features, _, _ = pr.student_forward({'student': models['student']}, inputs[0], seed)
    native_rng = pr.rng_hash()
    logits, features, shadow_logits, receipt = pr.student_forward(models, inputs[0], seed)
    assert torch.equal(logits, old_logits) and torch.equal(features, old_features) and native_rng == pr.rng_hash()
    assert shadow_logits.dtype == torch.float64 and receipt['rng_before_shadow'] == receipt['rng_after_shadow']
    assert all(x.grad is None for model in models.values() for x in model.parameters())
    with pytest.raises(b.ProtocolError, match='duplicate'): pr.attach_gradient_student(models, p)


def test_poe_target_keeps_native_float32_class_strata(monkeypatch):
    _, pair, _, models, _, inputs = setup_pair(monkeypatch, shadow=True)
    sl, _, dl, _ = pr.student_forward(models, inputs[0], pair['forward_seeds']['student_unlabeled'])
    native_parts = g.partition(models['student']); parts = g.partition(models['gradient_student'])
    shape = sl.shape; target = torch.full(shape, .2, dtype=torch.float64)
    target[:, 0] = .4; target[:, 1] = .4+1e-10
    assert bool((target.argmax(1) == 1).all()) and bool((target.float().argmax(1) == 0).all())
    dummy = {k: np.ones(sum(parts['params'][i].numel() for i in v)) for k, v in parts['blocks'].items()}
    dummy['global'] = np.concatenate(list(dummy.values()))
    native = dict(probability=sl.softmax(1), parts=native_parts, supervised=dummy, comparisons=[])
    _, components, _ = g.consistency_gradients(dl.softmax(1), target, {'PoE': np.ones(128)}, parts, dummy,
        candidates=('PoE',), decompose=True, native_reference=native)
    assert len(components) == 42 and all(x['component_sum_pass'] for x in components)
    assert all(x['class_strata_sha256'] == b.tensor_hash(target.float().argmax(1)) for x in native['comparisons'])
    assert all(x['target_float32_sha256'] == b.tensor_hash(target.float()) for x in native['comparisons'])


def test_all_precision_phases_complete_and_native_scoring_unchanged(monkeypatch, tmp_path):
    p, pair, freeze, models, legacy, _ = setup_pair(monkeypatch, shadow=True)
    current, history = r.banks(freeze, 0, 1); before = (b.array_hash(current), b.array_hash(history))
    cp = tmp_path/'synthetic.pt'; torch.save(models['student'].state_dict(), cp)
    cpdesc = dict(path=str(cp), sha256=b.sha256(cp), checkpoint_id='synthetic_only')
    metadata = {'scope': 'SYNTHETIC_ONLY_NO_FROZEN_TENSOR_READ'}
    original_forward = Tiny.forward; original_grad = torch.autograd.grad; original_build = e.build
    phases = {}; parity_calls = 0

    def observed_forward(model, *args, **kwargs):
        counts['native_forwards' if next(model.parameters()).dtype == torch.float32 else 'shadow_forwards'] += 1
        return original_forward(model, *args, **kwargs)

    def observed_grad(loss, inputs, **kwargs):
        counts['native_autograd' if inputs[0].dtype == torch.float32 else 'shadow_autograd'] += 1
        return original_grad(loss, inputs, **kwargs)

    def checked_build(sl, sf, tl, tf, *banks):
        nonlocal parity_calls
        result = original_build(sl, sf, tl, tf, *banks)
        assert sl.dtype == sf.dtype == tl.dtype == tf.dtype == torch.float32
        _, _, _, valid = pas_probability_objective(lambda *a, **kw: (sl, sf), lambda *a, **kw: (tl, tf), None, banks[0])
        assert np.array_equal(result['R1'], valid.numpy().reshape(-1)); parity_calls += 1
        return result

    with b.no_updates(), patch.object(Tiny, 'forward', observed_forward), patch.object(torch.autograd, 'grad', observed_grad), patch.object(e, 'build', checked_build):
        for phase in pilot.PHASES:
            counts = dict.fromkeys(pilot.COUNT_KEYS, 0)
            with ImmutableModels(models, cpdesc, tmp_path/'audit'/phase, metadata):
                result = e.gradient_pair(models, legacy, current, history, p, pair, tmp_path, tmp_path/'run', metadata, phase=phase, device='cpu')
                g.isolation(models, legacy, before, current, history)
            assert counts == dict(zip(pilot.COUNT_KEYS, pilot.COUNTS[phase]))
            pilot.validate_result(result, pair, phase, metadata); phases[phase] = result
            assert b.read_json(tmp_path/'audit'/phase/'immutability/synthetic_only.json')['bitwise_unchanged']
    assert parity_calls == 10
    assert phases['noise']['native_precision_comparisons'][:56] == phases['draw0']['native_precision_comparisons']
    assert len({b.H(x['student_draw_replay']) for x in phases.values()}) == 1
    for field in ('alignment', 'class_contribution', 'native_precision_comparisons'):
        bad = copy.deepcopy(phases['draw0']); bad[field][0] = copy.deepcopy(bad[field][1])
        with pytest.raises(b.ProtocolError, match='incomplete'): pilot.validate_result(bad, pair, 'draw0', metadata)
    bad = copy.deepcopy(phases['draw0']); bad['native_precision_comparisons'][0]['relative_l2'] = .1
    with pytest.raises(b.ProtocolError, match='FAIL_NUMERIC'): pilot.validate_result(bad, pair, 'draw0', metadata)


def test_default_engine_matches_frozen_native_all_phases(monkeypatch, tmp_path):
    frozen = {}
    for name in ('gradients', 'execution'):
        path = f'experiments/lcrseg/di_dmpa_gate1c_v2/{name}.py'
        source = subprocess.check_output(['git', '-C', str(ROOT.parents[1]), 'show', '44a25254697fa535d2b48b64e27ecb226436f7d0:'+path])
        module = types.ModuleType('di_dmpa_gate1c_v2.frozen_'+name); module.__package__ = 'di_dmpa_gate1c_v2'
        exec(compile(source, 'frozen_44a2525/'+path, 'exec'), module.__dict__); frozen[name] = module
    old = frozen['execution']
    for name in ('partition', 'supervised_gradient', 'consistency_gradients', 'isolation'): setattr(old, name, getattr(frozen['gradients'], name))
    p, pair, freeze, models, legacy, inputs = setup_pair(monkeypatch, shadow=False)
    monkeypatch.setattr(old, 'pair_inputs', lambda *args: tuple(x.clone() for x in inputs))
    current, history = r.banks(freeze, 0, 1); results = {}
    for label, engine in (('frozen', old), ('current', e)):
        local_models = copy.deepcopy(models); results[label] = {}
        with b.no_updates():
            for phase in pilot.PHASES:
                result = engine.gradient_pair(local_models, legacy, current, history, p, pair, tmp_path, tmp_path/label, {}, phase=phase, device='cpu')
                for key in ('primary_cache', 'teacher_cache'):
                    if key in result:
                        b.read_arrays(result[key]); result[key]['path'] = '<same-native-cache>'
                results[label][phase] = result
    assert results['frozen'] == results['current']
    assert all('diagnostic_precision' not in x for x in results['current'].values())


def test_probe_exception_keeps_bank_and_model_audits(monkeypatch, tmp_path):
    p, pair, freeze, models, legacy, _ = setup_pair(monkeypatch, shadow=False)
    p.update(diagnostic_precision='float64_shadow', _precision_contract_verified=True)
    cp = tmp_path/'synthetic.pt'; torch.save(models['student'].state_dict(), cp)
    cpdesc = dict(path=str(cp), sha256=b.sha256(cp), checkpoint_id='synthetic_only')
    monkeypatch.setattr(e, 'checkpoint', lambda *args: cpdesc)
    monkeypatch.setattr(e, 'load_b0', lambda *args: (models, legacy))
    def fail(*args, **kwargs): raise b.ProtocolError('synthetic failure preserved')
    monkeypatch.setattr(e, 'gradient_pair', fail)
    with pytest.raises(b.ProtocolError, match='synthetic failure preserved'):
        e.probe_unit(ROOT, tmp_path, p, freeze, {}, pair['seed'], pair['stage_index'], tmp_path/'run', 'cpu', 'draw0', pair_indices=[pair['pair_index']])
    iso = b.read_json(tmp_path/'run/probes/draw0'/e.pair_name(pair)/'isolation.json')
    guard = b.read_json(tmp_path/'run/probe_models/draw0'/e.pair_name(pair)/'immutability/synthetic_only.json')
    assert iso['legacy_prototypes_unchanged'] and iso['current_history_banks_unchanged']
    assert guard['bitwise_unchanged'] and not guard['extraction_completed'] and guard['error'] == 'ProtocolError: synthetic failure preserved'
    assert set(guard['before']) == {'student', 'ema_teacher', 'gradient_student'}


def test_zero_comparison_cannot_become_scientific_credit():
    zero = np.zeros(5); one = np.ones(5)
    assert pr.comparable(pr.compare(zero, zero))
    assert not pr.comparable(pr.compare(zero, one)) and not pr.comparable(pr.compare(one, zero))
    assert g.alignment(zero, zero)['cosine'] is None


def test_cli_refusals_never_mutate_sealed_or_occupied_outputs(monkeypatch, tmp_path):
    args = types.SimpleNamespace(action='report', phase='draw0', gpu=0, code_commit='synthetic', tests=tmp_path/'tests.xml')
    meta = dict(diagnostic_code_commit='synthetic', numeric_preregistration_commit=pilot.PREREG)
    def forbidden(*args): raise AssertionError('refused command reached an engine action')
    for action in ('prepare', 'worker', 'barrier', 'report'): monkeypatch.setattr(pilot, action, forbidden)
    scenarios = [('sealed', action) for action in ('prepare', 'worker', 'barrier', 'report')]
    scenarios += [('occupied', 'prepare'), ('started', 'worker'), ('phase_sealed', 'barrier'),
                  ('incomplete', 'barrier'), ('incomplete', 'report'), ('wrong_code', 'barrier')]
    for index, (state, action) in enumerate(scenarios):
        output = tmp_path/str(index); monkeypatch.setattr(pilot, 'OUTPUT', output)
        b.write_json(output/'RUN_METADATA.json', meta); b.write_json(output/'INPUT_AUDIT.json', dict(metadata=meta, status='PASS'))
        if state == 'sealed':
            for name in ('PILOT_STATUS.json', 'PILOT_ARTIFACT_MANIFEST.json'): b.write_json(output/name, {'synthetic': True})
        if state == 'started': b.write_json(output/'WORKER_draw0_gpu0_START.json', {'synthetic': True})
        if state == 'phase_sealed': b.write_json(output/'PHASE_draw0.json', {'synthetic': True})
        args.action = action; args.code_commit = 'wrong' if state == 'wrong_code' else 'synthetic'
        before = {str(p.relative_to(output)): b.sha256(p) for p in output.rglob('*') if p.is_file()}
        with pytest.raises(b.ProtocolError): pilot.dispatch(args)
        assert before == {str(p.relative_to(output)): b.sha256(p) for p in output.rglob('*') if p.is_file()}


def test_cli_preserves_genuine_failure_but_not_prepare_race(monkeypatch, tmp_path):
    output = tmp_path/'failure'; monkeypatch.setattr(pilot, 'OUTPUT', output)
    meta = dict(diagnostic_code_commit='synthetic', numeric_preregistration_commit=pilot.PREREG)
    b.write_json(output/'RUN_METADATA.json', meta); b.write_json(output/'INPUT_AUDIT.json', dict(metadata=meta, status='PASS'))
    for gpu in (0, 1): b.write_json(output/f'WORKER_draw0_gpu{gpu}.json', {'synthetic': True})
    def failure(*args): raise b.ProtocolError('synthetic fresh-action failure')
    monkeypatch.setattr(pilot, 'barrier', failure)
    args = types.SimpleNamespace(action='barrier', phase='draw0', gpu=None, code_commit='synthetic', tests=None)
    with pytest.raises(b.ProtocolError, match='fresh-action failure'): pilot.dispatch(args)
    assert b.read_json(output/'FAILURE_barrier_draw0.json')['error'] == 'synthetic fresh-action failure'
    prior = b.sha256(output/'FAILURE_barrier_draw0.json')
    with pytest.raises(b.ProtocolError, match='already failed'): pilot.dispatch(args)
    assert b.sha256(output/'FAILURE_barrier_draw0.json') == prior
    output = tmp_path/'prepare_race'; monkeypatch.setattr(pilot, 'OUTPUT', output)
    def race(*args):
        b.write_json(output/'OTHER_OWNER.json', {'synthetic': True})
        raise FileExistsError('synthetic concurrent create-only winner')
    monkeypatch.setattr(pilot, 'prepare', race); args.action = 'prepare'; args.tests = tmp_path/'tests.xml'
    with pytest.raises(FileExistsError): pilot.dispatch(args)
    assert {p.name for p in output.iterdir()} == {'OTHER_OWNER.json'}
