"""Synthetic-only version/provenance, complete-shard, budget and refusal gates."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from di_dmpa_gate1c_v2 import binding as b, execution as e, precision_pilot as pilot, runner
from di_dmpa_gate1c_v2 import full_precision as f, cache_reuse as c
from .test_core import ROOT, DOCS, Tiny, contract, toy_scores
from .test_precision_pilot import setup_pair


def test_v22_contract_assignment_and_exact_suite(record_testsuite_property):
    spec = b.read_json(DOCS/f'{f.NAME}.json'); p = contract()[0]
    for suffix, digest in f.HASHES.items(): b.check_hash(DOCS/f'{f.NAME}.{suffix}', digest)
    for suffix, digest in f.AUTH_HASHES.items(): b.check_hash(DOCS/f'GATE1C_V22_EXECUTION_AUTHORIZATION.{suffix}', digest)
    for name, digest in spec['core_byte_identity']['files'].items(): b.check_hash(ROOT/'di_dmpa_gate1c_v2'/name, digest)
    for shard in (0, 1):
        assert f.selected_pairs(spec, p, 'full', shard) == p['gradient_diagnostic']['batch_pairs'][shard::2]
        assert len(f.selected_pairs(spec, p, 'full', shard)) == 36
    assert [len(f.selected_pairs(spec, p, 'integration', s)) for s in (0, 1)] == [2, 1]
    assert {k: dict(zip(pilot.COUNT_KEYS, v)) for k, v in pilot.COUNTS.items()} == spec['execution']['counters_per_pair_phase']
    record_testsuite_property('v22_synthetic_contract', f.PREREG)


@pytest.mark.parametrize('bad', [None, 'case', 'seed', 'hash', 'bytes', 'path', 'dtype', 'shape', 'field', 'nan', 'null', 'GT'])
def test_cache_cases_are_exact_and_null_aware(tmp_path, bad):
    scores, _ = toy_scores(null=True)
    arrays = {k: scores[k].copy() for k in e.CACHE_FIELDS}
    arrays['teacher_probability'] = arrays['teacher_probability'].astype('float32')
    arrays['R0'] = arrays['teacher_probability'].max(1)
    support = e.validate_scores(arrays, 1, 12)
    if bad == 'dtype': arrays['R1'] = arrays['R1'].astype('int8')
    if bad == 'shape': arrays['R2'] = arrays['R2'][:11]
    if bad == 'field': arrays.pop('R3')
    if bad == 'nan': arrays['R0'][1] = np.nan
    if bad == 'null': arrays['R2'][0] = .1
    desc = b.save_arrays(tmp_path/'synthetic.npz', arrays)
    row = dict(case_id='synthetic', image_sha256='synthetic-image')
    plan = dict(case_id='synthetic', teacher_draw0_seed=11, student_seed=13)
    case = dict(row, **{k:v for k,v in plan.items() if k != 'case_id'}, arrays=desc, support=support,
                first_null_coordinates=[[0, 0]], GT_received_by_builder=False)
    manifest = dict(path='synthetic.npz', sha256=desc['sha256'], bytes=desc['bytes'])
    if bad == 'case': case['case_id'] = 'replacement'
    if bad == 'seed': case['student_seed'] += 1
    if bad == 'hash': desc['sha256'] = 'wrong'
    if bad == 'bytes': desc['bytes'] += 1
    if bad == 'path': desc['path'] = str(tmp_path.parent/'escaped.npz')
    if bad == 'GT': case['GT_received_by_builder'] = True
    if bad is None:
        assert c.validate_case(case, row, plan, manifest, tmp_path, 1, pixels=12) == support
    else:
        with pytest.raises((b.ProtocolError, b.NonfiniteEvidence)):
            c.validate_case(case, row, plan, manifest, tmp_path, 1, pixels=12)


@pytest.mark.parametrize('bad', [None, 'source_code', 'role', 'bank', 'cp', 'legacy', 'guard', 'grad', 'updates', 'optimizer', 'backward'])
def test_original_unit_guard_and_wrapper_provenance(bad):
    old = dict(diagnostic_code_commit='synthetic-old'); new = dict(diagnostic_code_commit='synthetic-new')
    cp = dict(checkpoint_id='B0/seed0/stage1', sha256='synthetic-cp'); bank = {'source': 'synthetic-original-fit'}
    context = dict(old, seed=0, stage_index=1, role='val', bank=bank, legacy_prototypes_sha256='legacy')
    iso = dict.fromkeys(('teacher_gradients', 'prototype_gradients', 'history_bank_gradients', 'student_parameter_grad_fields'), 'None')
    iso.update(optimizer_constructed=False, backward_called=False, model_optimizer_steps=0, transport_optimizer_steps_this_gate=0)
    original = dict(metadata=copy.deepcopy(context), checkpoint_sha256=cp['sha256'], legacy_prototypes_sha256_after='legacy',
        seed=0, stage_index=1, read_only=True, current_domain_only=True, all_scores_finite_or_structural_null=True, isolation=iso, cases=[])
    guard = dict(metadata=copy.deepcopy(context), status='PASS', extraction_completed=True, bitwise_unchanged=True,
        before={'student':'A', 'ema_teacher':'B'}, after={'student':'A', 'ema_teacher':'B'}, checkpoint_id=cp['checkpoint_id'],
        checkpoint_sha256_before=cp['sha256'], checkpoint_sha256_after=cp['sha256'])
    if bad == 'source_code': original['metadata']['diagnostic_code_commit'] = 'synthetic-new'
    if bad == 'role': original['metadata']['role'] = 'train_unlabeled'
    if bad == 'bank': original['metadata']['bank'] = {'source':'different'}
    if bad == 'cp': original['checkpoint_sha256'] = 'different'
    if bad == 'legacy': original['legacy_prototypes_sha256_after'] = 'different'
    if bad == 'guard': guard['after']['student'] = 'changed'
    if bad == 'grad': original['isolation']['teacher_gradients'] = 'Tensor'
    if bad == 'updates': original['isolation']['model_optimizer_steps'] = 1
    if bad == 'optimizer': original['isolation']['optimizer_constructed'] = True
    if bad == 'backward': original['isolation']['backward_called'] = True
    if bad is not None:
        with pytest.raises(b.ProtocolError): c.validate_unit(original, guard, old, 0, 1, cp, 'legacy', bank)
        return
    assert c.validate_unit(original, guard, old, 0, 1, cp, 'legacy', bank) == context
    source = dict(path='synthetic-original.json', sha256='synthetic-hash', bytes=100)
    derived = c.derived_unit(original, source, old, new)
    assert derived['cases'] == original['cases'] and derived['source_validation_unit']['original_metadata'] == context
    c.validate_derived(derived, original, source, old, new)
    derived['source_validation_unit']['original_metadata']['diagnostic_code_commit'] = 'new'
    with pytest.raises(b.ProtocolError): c.validate_derived(derived, original, source, old, new)


def test_shared_observer_preserves_four_phases_and_caps_before_extra_compute(monkeypatch, tmp_path):
    p, pair, freeze, models, legacy, inputs = setup_pair(monkeypatch, shadow=True)
    monkeypatch.setattr(pilot, 'LCRSegUNet2DJASCL', Tiny)
    current, history = e.banks(freeze, pair['seed'], pair['stage_index']); parity = []
    with b.no_updates():
        for phase in pilot.PHASES:
            counts = dict.fromkeys(pilot.COUNT_KEYS, 0); limits = dict(zip(pilot.COUNT_KEYS, pilot.COUNTS[phase]))
            with pilot.observe_pair(pair, counts, parity, limits=limits):
                result = e.gradient_pair(models, legacy, current, history, p, pair, tmp_path, tmp_path/'run', {}, phase=phase, device='cpu')
            assert counts == limits; pilot.validate_result(result, pair, phase, {})
        assert len(parity) == 10
        before = torch.get_rng_state().clone(); counts = dict.fromkeys(pilot.COUNT_KEYS, 0)
        with pilot.observe_pair(pair, counts, [], limits=counts.copy()), pytest.raises(b.ProtocolError, match='budget'):
            models['student'](inputs[0], stochastic_classifier=True)
        assert torch.equal(before, torch.get_rng_state()) and not any(counts.values())
        with f.forbid_forwards(), pytest.raises(b.ProtocolError, match='forward forbidden'):
            models['student'](inputs[0], stochastic_classifier=True)


def test_budget_and_owned_launcher_cleanup(tmp_path, monkeypatch):
    spec = b.read_json(DOCS/f'{f.NAME}.json')
    monkeypatch.setattr(f, 'PREFIX', tmp_path); monkeypatch.setattr(f, 'DATA_ROOT', tmp_path)
    monkeypatch.setattr(f.shutil, 'disk_usage', lambda p: SimpleNamespace(free=2**40))
    assert f.resource_guard(f.output_path('integration'), 'integration', spec, prepare=True)['output_bytes'] == 0
    monkeypatch.setattr(f.shutil, 'disk_usage', lambda p: SimpleNamespace(free=spec['storage']['minimum_root_reserve_bytes']-1))
    with pytest.raises(b.ProtocolError, match='storage'): f.resource_guard(f.output_path('integration'), 'integration', spec)
    args = SimpleNamespace(output=tmp_path/'launcher', code_commit='synthetic', data_root=tmp_path,
        input_contract='v2.1', execution_version='v2.2', scope='integration', deadline_monotonic=10**30)
    class Owned:
        pid = 123; code = None; terminated = 0
        def poll(self): return self.code
        def terminate(self): self.terminated += 1; self.code = -15
        def wait(self, timeout=None): return self.code
        def kill(self): raise AssertionError('no broad/extra kill required')
    owned = Owned(); commands = []
    def launch(command, **kwargs):
        commands.append(command)
        if len(commands) == 2: raise RuntimeError('synthetic second launch failure')
        return owned
    monkeypatch.setattr(runner.subprocess, 'Popen', launch); monkeypatch.setattr(runner, 'SHUTDOWN_TIMEOUT_SECONDS', 0)
    with pytest.raises(RuntimeError, match='second launch'): runner.gpu_workers(args, 'draw0')
    assert owned.terminated == 1 and b.read_json(args.output/'PROCESS_EXIT_draw0.json')['exit_codes'] == [-15]
    assert all('--execution-version' in command for command in commands)


@pytest.mark.parametrize('phase', pilot.PHASES)
@pytest.mark.parametrize('bad', [None, 'counter', 'pair', 'guard', 'PAS', 'start', 'exit', 'missing', 'extra', 'numeric'])
def test_full_72_pair_phase_barrier(tmp_path, monkeypatch, phase, bad):
    """Exercise real orchestration coverage; math is independently tested above."""
    p = contract()[0]; spec = b.read_json(DOCS/f'{f.NAME}.json')
    meta = dict(diagnostic_code_commit='synthetic', controller_pid=99)
    args = SimpleNamespace(output=tmp_path/'run', scope='full')
    pairs = f.selected_pairs(spec, p, 'full')
    fake_root = tmp_path/'repo/experiments/lcrseg'; monkeypatch.setattr(f, 'ROOT', fake_root)
    b.write_text(tmp_path/'repo/golden', 'synthetic-only')
    b.write_json(fake_root/'docs/di_dmpa_jascl'/f'{pilot.NAME}.json', dict(native_goldens=[],
        failed_pair_reference=dict(native_outcome_path='golden', native_outcome_sha256=b.sha256(tmp_path/'repo/golden'))))
    monkeypatch.setattr(pilot, 'golden_checks', lambda *a: None)
    def numeric(result, pair, requested, metadata):
        b.require(result['metadata'] == metadata and result['pair'] == pair and result['phase'] == requested and not result.get('numeric_failure'), 'numeric/identity failure')
        return dict(alignment_rows={'draw0':56, 'noise':448, 'posterior':56, 'poe':112}[phase],
            global_comparisons={'draw0':8, 'noise':64, 'posterior':8, 'poe':16}[phase],
            class_components={'draw0':168, 'noise':0, 'posterior':0, 'poe':42}[phase], supervised_global_comparisons=1)
    monkeypatch.setattr(pilot, 'validate_result', numeric)
    monkeypatch.setattr(f.reporting, 'validate_probe_results', lambda p, rows, phase: b.require(len(rows) == 72, 'missing rows'))
    for shard in (0, 1):
        assigned = f.selected_pairs(spec, p, 'full', shard); ids = [q['batch_id'] for q in assigned]
        start = dict(metadata=meta, phase=phase, shard=shard, parent_pid=99, physical_gpu=shard, pid=shard+10)
        worker = dict(metadata=meta, status='PASS', phase=phase, shard=shard, pairs=ids,
            counts={k:v*36 for k,v in zip(pilot.COUNT_KEYS, pilot.COUNTS[phase])}, all9_checkpoint_hashes_unchanged=True,
            per_pair_resources=[dict(batch_id=q) for q in ids], R1_parity=[dict(batch_id=q, exact_R1_parity=True, pixels=294912)
                for q in ids for _ in range({'draw0':1, 'noise':8, 'posterior':1, 'poe':0}[phase])])
        if shard == 0:
            if bad == 'counter': worker['counts']['native_forwards'] += 1
            if bad == 'pair': worker['pairs'] = worker['pairs'][:-1]+[worker['pairs'][0]]
            if bad == 'PAS': worker['R1_parity'].append(dict(batch_id=ids[0], exact_R1_parity=False, pixels=1))
            if bad == 'start': start['parent_pid'] = 98
        b.write_json(args.output/f'WORKER_{phase}_gpu{shard}_START.json', start)
        b.write_json(args.output/f'WORKER_{phase}_gpu{shard}.json', worker)
    for i, pair in enumerate(pairs):
        result = dict(metadata=meta, pair=pair, phase=phase, numeric_failure=bad == 'numeric' and i == 0)
        if bad == 'missing' and i == 0: continue
        folder = args.output/'probes'/phase/e.pair_name(pair)
        b.write_json(folder/'result.json', result)
        iso = dict(metadata=meta, legacy_prototypes_unchanged=True, current_history_banks_unchanged=True,
            optimizer_constructed=False, backward_called=False, **dict.fromkeys(('teacher_gradients', 'prototype_gradients',
            'history_bank_gradients', 'student_parameter_grad_fields'), 'None'))
        b.write_json(folder/'isolation.json', iso)
        states = {'student':'A', 'ema_teacher':'B', 'gradient_student':'C'}
        guard = dict(metadata=meta, status='PASS', bitwise_unchanged=True, extraction_completed=True, before=states,
            after=dict(states, student='changed') if bad == 'guard' and i == 0 else states,
            checkpoint_id=pair['checkpoint_id'], checkpoint_sha256_before=pair['checkpoint_sha256'], checkpoint_sha256_after=pair['checkpoint_sha256'])
        b.write_json(args.output/'probe_models'/phase/e.pair_name(pair)/'immutability'/f'B0_seed{pair["seed"]}_stage{pair["stage_index"]}.json', guard)
    if bad == 'extra': b.write_json(args.output/'probes'/phase/'extra/result.json', {})
    b.write_json(args.output/f'PROCESS_EXIT_{phase}.json', dict(exit_codes=[0,1] if bad == 'exit' else [0,0],
        diagnostic_code_commit='synthetic', worker_pids=[10,11]))
    b.write_json(args.output/f'INPUT_REFERENCES_{phase}.json', dict(metadata=meta, status='PASS'))
    if bad is None:
        f.phase_barrier(args, spec, p, meta, phase)
        assert len(b.read_json(args.output/f'PHASE_{phase}.json')['guards']) == 72
    else:
        with pytest.raises((b.ProtocolError, FileNotFoundError)): f.phase_barrier(args, spec, p, meta, phase)
        assert not (args.output/f'PHASE_{phase}.json').exists()


@pytest.mark.parametrize('marker', ['partial', 'GATE1C_V22_STATUS.json', 'EXECUTION_COMPLETION.json', 'GATE1C_V2_ARTIFACT_MANIFEST.json'])
def test_occupied_cli_and_repeated_reports_never_write(tmp_path, monkeypatch, marker):
    monkeypatch.setattr(f, 'PREFIX', tmp_path)
    output = f.output_path('integration'); b.write_json(output/marker, {'synthetic': True})
    args = SimpleNamespace(action='run', input_contract='v2.1', scope='integration', output=output,
        data_root=Path('/root/LCRSeg'), code_commit='synthetic', tests=tmp_path/'none')
    before = {str(p):b.sha256(p) for p in output.rglob('*') if p.is_file()}
    with pytest.raises(b.ProtocolError): f.dispatch(args)
    assert before == {str(p):b.sha256(p) for p in output.rglob('*') if p.is_file()}
    b.write_json(output/'PHASE_draw0.json', {'synthetic':True})
    with pytest.raises(b.ProtocolError, match='already sealed'): f.phase_barrier(args, {}, {}, {}, 'draw0')
    b.write_json(output/'NUMERICAL_COMPARISON_AUDIT.json', {'synthetic':True})
    with pytest.raises(b.ProtocolError, match='already attempted'): f.summarize_execution(args, {}, {}, {})


def test_unapproved_and_changed_reference_or_test_receipt_rejected(tmp_path):
    ref = tmp_path/'synthetic'; b.write_text(ref, 'original')
    receipt = dict(status='PASS', cache_reuse_approved=True, references=[c.checked_file(ref, b.sha256(ref))])
    c.recheck_references(receipt)
    with pytest.raises(b.ProtocolError): c.recheck_references(dict(receipt, cache_reuse_approved=False))
    with pytest.raises(b.ProtocolError): c.recheck_references(dict(receipt, references=receipt['references']*2))
    ref.write_text('changed')
    with pytest.raises(b.ProtocolError): c.recheck_references(receipt)
    xml = tmp_path/'synthetic.xml'
    xml.write_text('<testsuites><testsuite tests="200" failures="0" errors="0" skipped="0"><properties>'
        '<property name="diagnostic_code_commit" value="synthetic"/><property name="source_clean" value="true"/>'
        '<property name="v22_synthetic_contract" value="'+f.PREREG+'"/></properties></testsuite></testsuites>')
    assert f.test_receipt(xml, 'synthetic', 138)['tests'] == 200
    with pytest.raises(b.ProtocolError): f.test_receipt(xml, 'wrong-code', 138)
    xml.write_text(xml.read_text().replace('skipped="0"', 'skipped="1"'))
    with pytest.raises(b.ProtocolError): f.test_receipt(xml, 'synthetic', 138)
