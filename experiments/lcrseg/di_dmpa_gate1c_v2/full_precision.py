"""Conditionally authorized v2.2 orchestration around the shared frozen engine."""
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import platform
import shutil
import shlex
import signal
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr, nullcontext
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch

from scripts.audit_gate1c_v22_sources import audit as source_audit, identical_definitions
from . import binding as b, execution as e, cache_reuse as cache, precision_pilot as pilot
from . import runner, reporting

ROOT = Path(__file__).resolve().parents[1]
PREREG = '9593908bd36f7f833e385a70b2b772b7a8c84d22'
AUTH = 'aabef38c473f281bef7717e77fa326a542266d76'
NAME = 'DI_DMPA_GATE1C_V22_EXECUTION_PREREGISTRATION'
HASHES = {'md': '01a24ebfbb92db87a00263f1fd9e84262f730a8d64e128a7cd4e1cb72227246a',
          'json': 'a7ed480aac09cbf9fdf5fe723f4d236ec128e4619eb490a9005c32b81beba6f0'}
AUTH_HASHES = {'md': 'c59123607189f5ff2bad5206bf5b9fc792f93e045f318f223457b8b4e6c99715',
               'json': 'ff1cb6b4175f176457b9f60cd12d5d62c74daaa84b31bf7e4f42c31d72734b60'}
PREFIX = Path('/root/LCRSeg/runs/di_dmpa_gate1c_v22')/PREREG
DATA_ROOT = Path('/root/LCRSeg')
PHASES = pilot.PHASES
METRICS = ('validation_metrics', 'poe_metrics')


def output_path(scope):
    b.require(scope in ('integration', 'full'), 'unknown v2.2 scope')
    return PREFIX/('integration_attempt1' if scope == 'integration' else 'attempt1')


def selected_pairs(spec, p, scope, shard=None):
    b.require(scope in ('integration', 'full') and shard in (None, 0, 1), 'unknown pair assignment')
    pairs = p['gradient_diagnostic']['batch_pairs'] if scope == 'full' else spec['exact_code_real_integration']['pairs']
    if shard is None:
        return pairs
    if scope == 'full':
        return [pair for index, pair in enumerate(pairs) if index % 2 == shard]
    assigned = spec['exact_code_real_integration']['assignment'][str(shard)]
    return [pair for pair in pairs if pair['batch_id'] in assigned]


def verify_code(code, scope, *, remote):
    repo = ROOT.parents[1]; docs = ROOT/'docs/di_dmpa_jascl'
    for name, commit, hashes in ((NAME, PREREG, HASHES), ('GATE1C_V22_EXECUTION_AUTHORIZATION', AUTH, AUTH_HASHES)):
        b.verify_ancestor(repo, commit, code)
        for suffix, digest in hashes.items():
            path = docs/f'{name}.{suffix}'; b.check_hash(path, digest)
            blob = subprocess.check_output(['git', '-C', str(repo), 'show', f'{commit}:{path.relative_to(repo)}'])
            b.require(hashlib.sha256(blob).hexdigest() == digest, 'v2.2 publication blob changed')
    spec = b.read_json(docs/f'{NAME}.json'); authorization = b.read_json(docs/'GATE1C_V22_EXECUTION_AUTHORIZATION.json')
    b.require(authorization['preregistration']['commit'] == PREREG and authorization['preregistration']['sha256'] == HASHES and
        authorization['authorization_status'] == 'CONDITIONAL_EXECUTION_NOT_LAUNCH_READY' and
        not authorization['method_registered'] and not authorization['launch_ready_at_authorization'], 'wrong v2.2 authority')
    for record in (spec['authority'], authorization['live_precheck']):
        b.check_hash(repo/record['path'], record['sha256'])
    source = source_audit()
    b.require(source['worktree_clean'] and source['candidate_commit'] == code, 'dirty v2.2 source')
    path = 'experiments/lcrseg/di_dmpa_gate1c_v2/reporting.py'
    reference = subprocess.check_output(['git', '-C', str(repo), 'show', spec['core_byte_identity']['reference_commit']+':'+path]).decode()
    identical_definitions(reference, (repo/path).read_text(), spec['protected_reporting_functions'])
    upstream = ROOT/'third_party/JASCL_REFERENCE'
    b.require(not b.git(upstream, 'diff', '--name-only', 'HEAD'), 'official tracked source changed')
    p, freeze, meta = b.verify(ROOT, code, remote=remote, input_contract='v2.1')
    b.require(b.H(p['gradient_diagnostic']['batch_pairs']) == spec['execution']['fixed_batch_pairs_sha256'] and
        len(selected_pairs(spec, p, 'full', 0)) == len(selected_pairs(spec, p, 'full', 1)) == 36 and
        all(q in p['gradient_diagnostic']['batch_pairs'] for q in spec['exact_code_real_integration']['pairs']), 'v2.2 pair plan changed')
    b.require({phase: dict(zip(pilot.COUNT_KEYS, pilot.COUNTS[phase])) for phase in PHASES} ==
              spec['execution']['counters_per_pair_phase'], 'compute counters differ from registration')
    p.update(diagnostic_precision='float64_shadow', _precision_contract_verified=True)
    meta.update(input_registration_id=meta['registration_id'], input_preregistration_commit=meta['preregistration_commit'], input_preregistration_file_sha256=meta['preregistration_file_sha256'],
        input_authorization_commit=meta['authorization_commit'], input_authorization_file_sha256=meta['authorization_file_sha256'])
    meta.update(registration_id=spec['registration_id'], preregistration_commit=PREREG, preregistration_file_sha256=HASHES, authorization_commit=AUTH,
        authorization_file_sha256=AUTH_HASHES, diagnostic_version=spec['diagnostic_version'], execution_scope='GATE1C_V22_'+scope.upper(),
        numeric_engine_commit=spec['core_byte_identity']['reference_commit'], numeric_engine_sha256=spec['core_byte_identity']['files'],
        validation_reuse=dict(source_root=spec['validation_reuse']['source_root'], source_code_commit=spec['validation_reuse']['source_code_commit'],
            source_manifest_sha256=spec['validation_reuse']['source_artifact_manifest_sha256'], original_forwards=990, new_forwards=0,
            reused_guards=9, source_metadata_preserved=True), old_gate1c_v21_status='BLOCKED_INCOMPLETE_EVIDENCE',
        next_action='ANALYZE_VERSIONED_RESULT_WITHIN_CURRENT_METHOD_SCOPE', method_flags=p['method_flags'])
    return spec, p, freeze, meta, source


def test_receipt(path, code, minimum):
    path = Path(path); suites = list(ET.parse(path).getroot().iter('testsuite'))
    b.require(suites and sum(int(s.attrib['tests']) for s in suites) >= minimum and
        all(int(s.attrib[k]) == 0 for s in suites for k in ('failures', 'errors', 'skipped')), 'exact synthetic suite failed/incomplete')
    props = {r.attrib['name']: r.attrib['value'] for s in suites for r in s.findall('properties/property')}
    b.require(props.get('diagnostic_code_commit') == code and props.get('source_clean') == 'true' and
              props.get('v22_synthetic_contract') == PREREG, 'tests not from clean exact code/new suite')
    return dict(path=str(path), sha256=b.sha256(path), tests=sum(int(s.attrib['tests']) for s in suites), failures=0, errors=0, skipped=0,
                diagnostic_code_commit=code, source_clean=True)


def resource_guard(output, scope, spec, *, prepare=False):
    storage = spec['storage']; output = Path(output)
    b.require(output == output_path(scope) and output.resolve().is_relative_to(DATA_ROOT.resolve()), 'unregistered output path')
    free = shutil.disk_usage(DATA_ROOT).free
    minimum = storage['minimum_free_root_bytes_at_prepare'] if prepare else storage['minimum_root_reserve_bytes']+storage['per_pair_headroom_bytes']
    used = sum(path.stat().st_size for path in output.rglob('*') if path.is_file()) if output.exists() else 0
    budget = storage['integration_artifact_byte_budget'] if scope == 'integration' else storage['full_new_artifact_byte_budget']
    b.require(free >= minimum and used <= budget, 'registered storage budget/reserve exceeded')
    return dict(available_bytes=free, output_bytes=used, required_free_bytes=minimum, output_byte_budget=budget)


def evidence_barrier(output, name, meta):
    value = b.read_json(Path(output)/name)
    b.require(value['status'] == 'PASS' and value['metadata'] == meta, 'phase barrier provenance failed')
    for path, digest in value['evidence_sha256'].items():
        target = Path(output)/path
        b.require(target.resolve().is_relative_to(Path(output).resolve()), 'phase evidence escaped run')
        b.check_hash(target, digest)
    return value


def load_run(args):
    spec, p, freeze, verified, source = verify_code(args.code_commit, args.scope, remote=False)
    meta = b.read_json(args.output/'GATE1C_V2_RUN_METADATA.json')
    for key, value in verified.items():
        if key not in ('remote_verified_code_commit', 'publication_verification_url', 'publication_verification'):
            b.require(meta[key] == value, 'mixed v2.2 run metadata: '+key)
    b.require(meta['remote_verified_code_commit'] == args.code_commit, 'missing exact-code publication proof')
    return spec, p, freeze, meta


def record_failure(output, name, meta, error, **details):
    path = Path(output)/name
    if not path.exists():
        b.write_json(path, dict(metadata=meta, status=getattr(error, 'status', 'BLOCKED_INCOMPLETE_EVIDENCE'),
            error=str(error), traceback=traceback.format_exc(), observed_at_utc=datetime.now(timezone.utc).isoformat(),
            new_optimizer_updates=0, method_registered=False, **details))


def no_prior_failure(output):
    b.require(not list(Path(output).glob('FAILURE_*.json')) and not (Path(output)/'STOP_REQUESTED.json').exists(), 'prior failure; no automatic replay')


def dispatch(args):
    """All occupied/partial/duplicate command refusals happen before logging."""
    b.require(args.action in ('run', 'worker') and args.input_contract == 'v2.1', 'invalid v2.2 command')
    b.require(args.output == output_path(args.scope) and args.data_root == Path('/root/LCRSeg'), 'unregistered v2.2 paths')
    b.require(not any((args.output/n).exists() for n in ('EXECUTION_COMPLETION.json', 'GATE1C_V2_ARTIFACT_MANIFEST.json', 'GATE1C_V22_STATUS.json')), 'run sealed; read-only inspection only')
    if args.action == 'run':
        b.require(args.tests is not None and not args.output.exists(), 'occupied run or missing tests; no replay')
        run(args)
    else:
        b.require(args.phase in (*PHASES, *METRICS) and args.shard in (0, 1), 'invalid worker phase/shard')
        no_prior_failure(args.output)
        label = 'cpu' if args.phase in METRICS else 'gpu'
        b.require(not (args.output/f'WORKER_{args.phase}_{label}{args.shard}_START.json').exists(), 'worker already attempted')
        meta = b.read_json(args.output/'GATE1C_V2_RUN_METADATA.json')
        b.require(meta['diagnostic_code_commit'] == args.code_commit and meta['controller_pid'] == os.getppid(), 'worker is not owned by this controller')
        audit = b.read_json(args.output/'CACHE_REUSE_AUDIT.json')
        b.require(audit['metadata'] == meta and audit['status'] == 'PASS' and audit['cache_reuse_approved'], 'cache preparation incomplete')
        worker(args)


def worker(args):
    spec, p, freeze, meta = load_run(args)
    cpu = args.phase in METRICS; label = 'cpu' if cpu else 'gpu'
    if cpu:
        b.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU evaluator exposed a GPU')
    else:
        b.require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.shard) and torch.cuda.device_count() == 1, 'GPU assignment changed')
    previous = {'noise': 'draw0', 'posterior': 'noise', 'poe': 'posterior', 'poe_metrics': 'posterior'}
    if args.phase in previous:
        evidence_barrier(args.output, 'PHASE_'+previous[args.phase]+'.json', meta)
    if args.phase == 'draw0' and args.scope == 'full':
        evidence_barrier(args.output, 'PHASE_validation_metrics.json', meta)
    torch.set_num_threads(1)
    counts = dict.fromkeys(pilot.COUNT_KEYS, 0); parity = []; per_pair = []; tasks = []
    limit = spec['execution']['cpu_metric_phase_timeout_seconds'] if cpu else (
        spec['exact_code_real_integration']['maximum_worker_phase_seconds'] if args.scope == 'integration' else spec['execution']['full_gpu_worker_phase_timeout_seconds'])
    remaining = meta['deadline_monotonic']-time.monotonic()
    b.require(remaining > 0, 'registered overall time budget exhausted')
    started = time.monotonic()
    b.write_json(args.output/f'WORKER_{args.phase}_{label}{args.shard}_START.json', dict(metadata=meta, phase=args.phase,
        shard=args.shard, pid=os.getpid(), parent_pid=os.getppid(), physical_gpu=None if cpu else args.shard,
        device_name='CPU' if cpu else torch.cuda.get_device_name(0), exact_command=sys.argv,
        started_at_utc=datetime.now(timezone.utc).isoformat(), started_monotonic=started))

    def timeout(signum, frame):
        raise TimeoutError('registered worker/phase time budget exceeded')

    signal.signal(signal.SIGALRM, timeout); signal.alarm(max(1, int(min(limit, remaining))))
    try:
        with b.no_updates(), (forbid_forwards() if cpu else nullcontext()):
            if cpu:
                b.require(args.scope == 'full', 'integration must not become a metric selection run')
                for index, (seed, stage) in enumerate((s, t) for s in range(3) for t in range(3)):
                    if index % 2 != args.shard:
                        continue
                    no_prior_failure(args.output); resource_guard(args.output, args.scope, spec)
                    tasks.append(e.evaluate_unit((args.data_root, p, args.output, seed, stage, args.phase == 'poe_metrics')))
            else:
                for pair in selected_pairs(spec, p, args.scope, args.shard):
                    no_prior_failure(args.output); space = resource_guard(args.output, args.scope, spec)
                    limits = {k: counts[k]+n for k, n in zip(pilot.COUNT_KEYS, pilot.COUNTS[args.phase])}
                    with pilot.observe_pair(pair, counts, parity, limits=limits):
                        e.probe_unit(ROOT, args.data_root, p, freeze, meta, pair['seed'], pair['stage_index'], args.output,
                            'cuda:0', args.phase, pair_indices=[pair['pair_index']])
                    b.require(counts == limits, 'per-pair forward/autograd count mismatch')
                    result = b.read_json(args.output/'probes'/args.phase/e.pair_name(pair)/'result.json')
                    pilot.validate_result(result, pair, args.phase, meta)
                    per_pair.append(dict(batch_id=pair['batch_id'], before_resource=space, after_resource=resource_guard(args.output, args.scope, spec)))
        if not cpu:
            flags = dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                autocast=torch.is_autocast_enabled())
            b.require(flags == dict(deterministic_algorithms=True, cudnn_deterministic=True, cudnn_benchmark=False,
                cudnn_allow_tf32=False, matmul_allow_tf32=False, autocast=False), 'native backend flags changed')
        else:
            flags = dict(model_forwards=0, evaluator_only=True, threads=torch.get_num_threads())
        runner.disk_hashes(p)
        b.write_json(args.output/f'WORKER_{args.phase}_{label}{args.shard}.json', dict(metadata=meta, status='PASS',
            phase=args.phase, shard=args.shard, counts=counts, R1_parity=parity, per_pair_resources=per_pair, metric_tasks=tasks,
            pairs=[] if cpu else [q['batch_id'] for q in selected_pairs(spec, p, args.scope, args.shard)],
            backend_flags=flags, all9_checkpoint_hashes_unchanged=True, elapsed_seconds=time.monotonic()-started,
            completed_at_utc=datetime.now(timezone.utc).isoformat()))
    except Exception as error:
        record_failure(args.output, f'FAILURE_{args.phase}_{label}{args.shard}.json', meta, error, counts=counts,
            R1_parity=parity, completed_pair_resources=per_pair, exact_command=sys.argv)
        runner.request_stop(args.output, 'v2.2 worker failed; preserve evidence, no replay')
        raise
    finally:
        signal.alarm(0)


def validate_isolation(output, pair, phase, meta):
    folder = Path(output)/'probes'/phase/e.pair_name(pair)
    iso_path = folder/'isolation.json'; iso = b.read_json(iso_path)
    b.require(iso['legacy_prototypes_unchanged'] and iso['current_history_banks_unchanged'] and all(iso[k] == 'None' for k in
        ('teacher_gradients', 'prototype_gradients', 'history_bank_gradients', 'student_parameter_grad_fields')) and
        not iso['optimizer_constructed'] and not iso['backward_called'] and all(iso['metadata'][k] == v for k, v in meta.items()), 'new probe isolation failure')
    path = Path(output)/'probe_models'/phase/e.pair_name(pair)/'immutability'/f'B0_seed{pair["seed"]}_stage{pair["stage_index"]}.json'
    guard = b.read_json(path)
    b.require(guard['bitwise_unchanged'] and guard['extraction_completed'] and guard['before'] == guard['after'] and
        set(guard['before']) == {'student', 'ema_teacher', 'gradient_student'} and guard['status'] == 'PASS' and
        guard['checkpoint_id'] == pair['checkpoint_id'] and guard['checkpoint_sha256_before'] == guard['checkpoint_sha256_after'] == pair['checkpoint_sha256'] and
        all(guard['metadata'][k] == v for k, v in meta.items()), 'new probe guard incomplete')
    return iso_path, path, dict(path=str(path.relative_to(output)), sha256=b.sha256(path), checkpoint_id=pair['checkpoint_id'],
                               before=guard['before'], after=guard['after'], reused=False, newly_executed=True)


def phase_barrier(args, spec, p, meta, phase):
    destination = args.output/f'PHASE_{phase}.json'
    b.require(not destination.exists(), 'phase already sealed')
    no_prior_failure(args.output)
    evidence = {}; results = []; guards = []
    totals = dict(alignment_rows=0, global_comparisons=0, class_components=0, supervised_global_comparisons=0)
    counts = dict.fromkeys(pilot.COUNT_KEYS, 0); pas_calls = 0
    for shard in (0, 1):
        path = args.output/f'WORKER_{phase}_gpu{shard}.json'; w = b.read_json(path)
        start_path = args.output/f'WORKER_{phase}_gpu{shard}_START.json'; start = b.read_json(start_path)
        b.require(start['metadata'] == meta and start['parent_pid'] == meta['controller_pid'] and
                  start['physical_gpu'] == shard and start['phase'] == phase and start['shard'] == shard, 'unowned worker start')
        evidence[start_path.name] = b.sha256(start_path)
        pairs = selected_pairs(spec, p, args.scope, shard)
        expected_counts = dict(zip(pilot.COUNT_KEYS, [n*len(pairs) for n in pilot.COUNTS[phase]]))
        b.require(w['metadata'] == meta and w['status'] == 'PASS' and w['phase'] == phase and w['shard'] == shard and
            w['pairs'] == [q['batch_id'] for q in pairs] and w['counts'] == expected_counts and w['all9_checkpoint_hashes_unchanged'] and
            [r['batch_id'] for r in w['per_pair_resources']] == w['pairs'], 'worker evidence incomplete')
        multiplier = {'draw0': 1, 'noise': 8, 'posterior': 1, 'poe': 0}[phase]
        b.require(len(w['R1_parity']) == multiplier*len(pairs) and all(r['exact_R1_parity'] and r['pixels'] == 294912 for r in w['R1_parity']) and
            [r['batch_id'] for r in w['R1_parity']] == [q['batch_id'] for q in pairs for _ in range(multiplier)], 'PAS parity coverage changed')
        pas_calls += len(w['R1_parity'])
        for key, value in w['counts'].items(): counts[key] += value
        evidence[str(path.relative_to(args.output))] = b.sha256(path)
    for pair in selected_pairs(spec, p, args.scope):
        path = args.output/'probes'/phase/e.pair_name(pair)/'result.json'; result = b.read_json(path)
        values = pilot.validate_result(result, pair, phase, meta); results.append(result)
        for key, value in values.items(): totals[key] += value
        iso, guard_path, guard = validate_isolation(args.output, pair, phase, meta); guards.append(guard)
        for item in (path, iso, guard_path): evidence[str(item.relative_to(args.output))] = b.sha256(item)
        for field in ('primary_cache', 'teacher_cache'):
            if field in result:
                desc = result[field]; item = Path(desc['path'])
                b.require(item.resolve().is_relative_to(args.output.resolve()), 'new probe cache escaped output')
                b.read_arrays(desc); cache.checked_file(item, desc['sha256'], desc['bytes'])
                evidence[str(item.relative_to(args.output))] = desc['sha256']
    b.require({path.name for path in (args.output/'probes'/phase).iterdir()} == {e.pair_name(q) for q in selected_pairs(spec, p, args.scope)}, 'extra probe pair')
    if args.scope == 'full': reporting.validate_probe_results(p, results, phase)
    if phase == 'draw0':
        pilot_spec = b.read_json(ROOT/'docs/di_dmpa_jascl'/f'{pilot.NAME}.json')
        for golden in pilot_spec['native_goldens']: b.check_hash(golden['path'], golden['sha256'])
        ref = pilot_spec['failed_pair_reference']
        b.check_hash(ROOT.parents[1]/ref['native_outcome_path'], ref['native_outcome_sha256'])
        pilot.golden_checks(pilot_spec, results)
    exit_path = args.output/f'PROCESS_EXIT_{phase}.json'; exits = b.read_json(exit_path)
    b.require(exits['exit_codes'] == [0, 0] and exits['diagnostic_code_commit'] == meta['diagnostic_code_commit'], 'worker exit failure')
    b.require(exits['worker_pids'] == [b.read_json(args.output/f'WORKER_{phase}_gpu{s}_START.json')['pid'] for s in (0, 1)], 'worker exit identity mismatch')
    evidence[exit_path.name] = b.sha256(exit_path)
    input_path = args.output/f'INPUT_REFERENCES_{phase}.json'
    checked = b.read_json(input_path)
    b.require(checked['status'] == 'PASS' and checked['metadata'] == meta, 'phase input check missing/mixed')
    evidence[input_path.name] = b.sha256(input_path)
    b.write_json(destination, dict(metadata=meta, status='PASS', phase=phase, totals=totals, counts=counts,
        native_PAS_calls=pas_calls, guards=guards, evidence_sha256=evidence, numerical_checks_complete=True))


def metric_barrier(args, meta, *, poe=False):
    phase = 'poe_metrics' if poe else 'validation_metrics'; destination = args.output/f'PHASE_{phase}.json'
    b.require(not destination.exists(), 'metric phase already sealed'); no_prior_failure(args.output)
    paths = sorted((args.output/('poe_validation' if poe else 'reliability_units')).glob('*.json'))
    units = [b.read_json(path) for path in paths]
    b.require(len(units) == 9 and {(u['seed'], u['stage_index']) for u in units} == {(s,t) for s in range(3) for t in range(3)}, 'metric units incomplete')
    for unit in units:
        b.require(all(unit['metadata'][k] == v for k, v in meta.items()), 'metric provenance changed')
    for shard in (0, 1):
        path = args.output/f'WORKER_{phase}_cpu{shard}.json'; worker = b.read_json(path)
        start_path = args.output/f'WORKER_{phase}_cpu{shard}_START.json'; start = b.read_json(start_path)
        b.require(start['metadata'] == meta and start['parent_pid'] == meta['controller_pid'] and
                  start['physical_gpu'] is None and start['phase'] == phase and start['shard'] == shard, 'unowned CPU worker start')
        paths.append(start_path)
        expected = [(s,t) for i,(s,t) in enumerate((s,t) for s in range(3) for t in range(3)) if i % 2 == shard]
        b.require(worker['status'] == 'PASS' and worker['metadata'] == meta and
            [(r['seed'], r['stage_index']) for r in worker['metric_tasks']] == expected and
            all(r['complete'] for r in worker['metric_tasks']) and worker['counts'] == dict.fromkeys(pilot.COUNT_KEYS, 0), 'metric worker incomplete')
        paths.append(path)
    path = args.output/f'PROCESS_EXIT_{phase}.json'
    exits = b.read_json(path)
    b.require(exits['exit_codes'] == [0, 0] and exits['diagnostic_code_commit'] == meta['diagnostic_code_commit'] and
              exits['worker_pids'] == [b.read_json(args.output/f'WORKER_{phase}_cpu{s}_START.json')['pid'] for s in (0, 1)], 'metric worker exit failure')
    paths.append(path)
    b.write_json(destination, dict(metadata=meta, status='PASS', phase=phase, units=9,
        evidence_sha256={str(path.relative_to(args.output)): b.sha256(path) for path in paths}))


def forbid_forwards():
    return patch.object(pilot.LCRSegUNet2DJASCL, 'forward', side_effect=b.ProtocolError('model forward forbidden in input/cache/metric audit'))


def verify_manifest(output):
    output = Path(output); path = output/'GATE1C_V2_ARTIFACT_MANIFEST.json'
    b.check_hash(path, (output/'GATE1C_V2_ARTIFACT_MANIFEST.sha256').read_text().split()[0])
    manifest = b.read_json(path); files = manifest['artifacts']
    names = [r['path'] for r in files]
    excluded = {path.name, 'GATE1C_V2_ARTIFACT_MANIFEST.sha256'}
    observed = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p.name not in excluded}
    b.require(len(names) == len(set(names)) == manifest['file_count'] and set(names) == observed and
        sum(r['bytes'] for r in files) == manifest['total_bytes'], 'sealed output inventory incomplete')
    for row in files:
        target = output/row['path']
        b.require(target.resolve().is_relative_to(output.resolve()), 'sealed artifact escaped output')
        cache.checked_file(target, row['sha256'], row['bytes'])
    return dict(path=str(path), sha256=b.sha256(path), file_count=manifest['file_count'], total_bytes=manifest['total_bytes'])


def require_integration(code):
    output = output_path('integration'); no_prior_failure(output)
    manifest = verify_manifest(output)
    observation_path = PREFIX/'INTEGRATION_PROCESS_EXIT.json'; observation = b.read_json(observation_path)
    b.require(observation['diagnostic_code_commit'] == code and observation['output'] == str(output) and
        observation['manifest_sha256'] == manifest['sha256'] and observation['exit_code'] == 0 and
        observation['exit_observation'] == 'DIRECT_SSH_COMMAND_RETURN' and observation['process_exited'] is True,
        'actual integration process exit not independently observed')
    meta = b.read_json(output/'GATE1C_V2_RUN_METADATA.json'); status = b.read_json(output/'GATE1C_V22_STATUS.json')
    done = b.read_json(output/'EXECUTION_COMPLETION.json')
    b.require(meta['diagnostic_code_commit'] == meta['remote_verified_code_commit'] == code and
        meta['preregistration_commit'] == PREREG and meta['preregistration_file_sha256'] == HASHES and
        meta['authorization_commit'] == AUTH and meta['authorization_file_sha256'] == AUTH_HASHES and
        meta['execution_scope'] == 'GATE1C_V22_INTEGRATION', 'integration from another version/code')
    b.require(status['metadata'] == done['metadata'] == meta and status['status'] == 'PASS_EXACT_CODE_REAL_INTEGRATION' and
        done['status'] == 'COMPLETE' and done['controller_result_code'] == 0 and status['scientific_admission'] is None and
        status['counts'] == dict(native_forwards=51, shadow_forwards=24, native_autograd=276, shadow_autograd=366) and
        status['new_probe_guards'] == 12 and not status['method_registered'], 'new integration incomplete')
    for phase in PHASES: evidence_barrier(output, f'PHASE_{phase}.json', meta)
    receipt = b.read_json(output/'CACHE_REUSE_AUDIT.json')
    b.require(receipt['metadata'] == meta and receipt['known_real_null']['null'] == 1, 'integration cache/null audit missing')
    cache.recheck_references(receipt)
    b.require(b.read_json(output/'INPUT_REFERENCES_after.json')['status'] == 'PASS', 'integration post-audit incomplete')
    return dict(manifest, process_exit_observation=cache.checked_file(observation_path, b.sha256(observation_path)),
                status_sha256=b.sha256(output/'GATE1C_V22_STATUS.json'),
                completion_sha256=b.sha256(output/'EXECUTION_COMPLETION.json'), new_model_forwards=75)


def check_inputs(args, spec, p, meta, label):
    b.require(time.monotonic() < args.deadline_monotonic, 'overall execution budget exhausted')
    _, _, _, verified, source = verify_code(args.code_commit, args.scope, remote=False)
    b.require(all(meta[k] == v for k, v in verified.items() if not k.startswith('publication_verification') and
                  k != 'remote_verified_code_commit'), 'execution source binding changed')
    receipt = b.read_json(args.output/'CACHE_REUSE_AUDIT.json')
    b.require(receipt['metadata'] == meta, 'mixed reference audit metadata')
    cache.recheck_references(receipt); runner.disk_hashes(p)
    index = b.read_json(args.output/'RELIABILITY_CACHE_MANIFEST.json')
    b.require(index['metadata'] == meta and len(index['units']) == index['unit_count'] == 9 and index['case_count'] == 495,
              'new reference index incomplete')
    for desc, source_desc in zip(index['units'], receipt['units']):
        b.check_hash(desc['path'], desc['sha256']); unit = b.read_json(desc['path']); original = b.read_json(source_desc['path'])
        b.require(desc['cases'] == original['cases'] and desc['source_validation_unit'] == unit['source_validation_unit'], 'new index changed source cases')
        cache.validate_derived(unit, original, source_desc, receipt['source_metadata'], meta)
    b.require(time.monotonic() < args.deadline_monotonic, 'reference audit exceeded execution budget')
    b.write_json(args.output/f'INPUT_REFERENCES_{label}.json', dict(metadata=meta, status='PASS', label=label,
        source=source, reference_count=len(receipt['references']), all_source_cache_input_hashes_unchanged=True,
        all_new_reference_wrappers_match_original=True, resource=resource_guard(args.output, args.scope, spec),
        checkpoint_tensor_loads=0, cache_array_loads=0, model_forwards=0,
        checked_at_utc=datetime.now(timezone.utc).isoformat()))


def summarize_execution(args, spec, p, meta):
    b.require(not (args.output/'NUMERICAL_COMPARISON_AUDIT.json').exists() and not (args.output/'GATE1C_V22_STATUS.json').exists(), 'report already attempted; read-only inspection only')
    no_prior_failure(args.output)
    counts = dict.fromkeys(pilot.COUNT_KEYS, 0)
    coverage = dict(alignment_rows=0, global_comparisons=0, class_components=0, supervised_global_comparisons=0)
    guards = []; comparisons = []; supervised = []; components = []; pas = 0; phases = []
    pairs = selected_pairs(spec, p, args.scope)
    for phase in PHASES:
        value = evidence_barrier(args.output, f'PHASE_{phase}.json', meta); phases.append(value)
        for key, n in value['counts'].items(): counts[key] += n
        for key, n in value['totals'].items(): coverage[key] += n
        pas += value['native_PAS_calls']; guards.extend(value['guards'])
        for pair in pairs:
            result = b.read_json(args.output/'probes'/phase/e.pair_name(pair)/'result.json')
            pilot.validate_result(result, pair, phase, meta)
            comparisons.extend(r for r in result['native_precision_comparisons'] if r['block'] == 'global')
            supervised.append(result['supervised_precision_comparisons']['global'])
            components.extend(result['class_contribution'])
    expected = spec['execution']['full_count_totals'] if args.scope == 'full' else spec['exact_code_real_integration']['counts']
    b.require(counts == {k: expected[k] for k in counts} and counts['native_forwards']+counts['shadow_forwards'] == expected['total_real_forwards'],
        'total execution counter mismatch')
    b.require(coverage == dict(zip(coverage, (len(pairs)*672, len(pairs)*96, len(pairs)*210, len(pairs)*4))) and
        len(guards) == len(pairs)*4 == len(list((args.output/'probe_models').rglob('immutability/*.json'))) and
        len(list((args.output/'probes').glob('*/*/result.json'))) == len(guards) and
        {d.name for d in (args.output/'probes').iterdir()} == set(PHASES) and pas == len(pairs)*10, 'full phase/guard/PAS coverage incomplete')
    b.require(len(comparisons) == coverage['global_comparisons'] and len(supervised) == coverage['supervised_global_comparisons'] and
        len(components) == coverage['class_components'], 'missing raw numerical comparison')
    numbers = dict(metadata=meta, status='PASS', counts=counts, new_model_forwards=expected['total_real_forwards'],
        coverage=coverage, native_PAS_calls=pas, native_PAS_pixel_call_comparisons=pas*294912,
        maximum_objective_relative_l2=max((r['relative_l2'] for r in comparisons if r['relative_l2'] is not None), default=None),
        minimum_objective_cosine=min((r['cosine'] for r in comparisons if r['cosine'] is not None), default=None),
        maximum_supervised_relative_l2=max((r['relative_l2'] for r in supervised if r['relative_l2'] is not None), default=None),
        minimum_supervised_cosine=min((r['cosine'] for r in supervised if r['cosine'] is not None), default=None),
        maximum_component_sum_abs_residual=max(r['component_sum_max_abs_error'] for r in components),
        both_zero_global_comparisons=sum(r['native_l2_norm'] == r['reference_l2_norm'] == 0 for r in comparisons),
        optimizer_updates=0, all_numerical_checks_passed_before_scientific_compilation=True)
    b.write_json(args.output/'NUMERICAL_COMPARISON_AUDIT.json', numbers)
    reused = b.read_json(args.output/'CACHE_REUSE_AUDIT.json')['original_validation_guards']
    b.require(len(reused) == 9 and all(r['reused'] and not r['newly_executed'] for r in reused) and
              all(not r['reused'] and r['newly_executed'] for r in guards), 'old/new guard counts relabeled')
    audit = dict(metadata=meta, status='PASS', guard_count=len(guards)+len(reused), new_probe_guard_count=len(guards),
        reused_validation_guard_count=9, guards=guards, original_validation_guards=reused,
        checkpoint_disk_sha256=runner.disk_hashes(p), all9_B0_checkpoints_unchanged=True,
        all_model_states_bitwise_unchanged=True, all_grad_fields_None=True, model_optimizer_steps=0, transport_optimizer_steps_this_gate=0)
    b.write_json(args.output/'GATE1C_V2_MODEL_IMMUTABILITY_AUDIT.json', audit)
    if args.scope == 'full':
        for phase in METRICS: evidence_barrier(args.output, f'PHASE_{phase}.json', meta)
        status = reporting.compile_report(args.output, p, meta, audit)
    else:
        status = dict(metadata=meta, status='PASS_EXACT_CODE_REAL_INTEGRATION', counts=counts, coverage=coverage,
            new_probe_guards=12, reused_validation_guards=9, scientific_admission=None, method_registered=False,
            gate1_overall_status='FAIL_TRANSPORT_NOT_SUPPORTED', hidden_gt_training_usage='none', test_gt_usage='none')
        b.write_json(args.output/'GATE1C_V22_STATUS.json', status)
        b.write_text(args.output/'GATE1C_V22_FINAL_REPORT.md', '# Gate1C v2.2 exact-code real integration\n\n'
            'PASS_EXACT_CODE_REAL_INTEGRATION. Three fixed pairs, four phases, 75 new forwards and 12 new model guards. '
            'All native/FP64 hash goldens, PAS and numerical comparisons passed. The nine validation guards and 990 validation '
            'forwards are explicitly reused historical evidence; 495 caches were independently audited. '
            'No scientific C1-C8 admission is made by this integration. No optimizer or method registration.\n')
    return status


def run(args):
    # All code/test/integration/resource checks precede atomic ownership of output.
    spec, p, freeze, meta, source = verify_code(args.code_commit, args.scope, remote=True)
    tests = test_receipt(args.tests, args.code_commit, spec['minimum_existing_synthetic_test_count'])
    integration = require_integration(args.code_commit) if args.scope == 'full' else None
    resource = resource_guard(args.output, args.scope, spec, prepare=True)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'], text=True, timeout=15).strip()
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=15).strip()
    b.require(not occupied and len(gpu.splitlines()) == 2 and all('RTX 3090' in line for line in gpu.splitlines()), 'existing GPUs not idle/compatible')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); args.deadline_monotonic = started+spec['execution']['full_timeout_seconds']
    meta.update(started_at_utc=datetime.now(timezone.utc).isoformat(), started_monotonic=started,
        deadline_monotonic=args.deadline_monotonic, controller_pid=os.getpid(), exact_controller_command=sys.argv,
        python=sys.version, python_executable=sys.executable, torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        platform=platform.platform(), hostname=platform.node(), physical_gpus=[0, 1], gpu_identity=gpu,
        gpu_workers=2, cpu_metric_workers=2, synthetic_tests=tests, integration_evidence=integration, prepare_resource=resource)
    error = None; status = None
    with (args.output/'controller.log').open('x') as log, redirect_stdout(log), redirect_stderr(log):
        try:
            b.write_json(args.output/'GATE1C_V2_RUN_METADATA.json', meta)
            b.write_json(args.output/'SOURCE_AUDIT.json', source)
            b.write_text(args.output/'pytest.xml', Path(args.tests).read_text())
            b.write_text(args.output/'EXACT_COMMANDS.md', '# Exact execution\n\n```sh\ncd '+shlex.quote(str(ROOT))+'\n'+
                'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/lib/x86_64-linux-gnu '
                'CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH='+shlex.quote(str(ROOT))+' '+shlex.join([sys.executable,
                '-m', 'di_dmpa_gate1c_v2.runner', 'run', '--code-commit', args.code_commit, '--output', str(args.output),
                '--tests', str(args.tests), '--data-root', str(args.data_root), '--input-contract', 'v2.1',
                '--execution-version', 'v2.2', '--scope', args.scope])+'\n```\n')
            with b.no_updates(), forbid_forwards():
                fresh_input = runner.input_audit(ROOT, args.data_root, p, meta)
                b.write_json(args.output/'GATE1C_V2_INPUT_AUDIT.json', fresh_input)
                receipt, units, census = cache.audit_sources(ROOT, args.data_root, spec, p, freeze, fresh_input)
                receipt['metadata'] = meta
                b.write_json(args.output/'CACHE_REUSE_AUDIT.json', receipt)
                cache.write_references(args.output, meta, receipt, units, census)
                del units, census
            if args.scope == 'full':
                check_inputs(args, spec, p, meta, 'validation_metrics')
                runner.gpu_workers(args, 'validation_metrics', cpu=True); metric_barrier(args, meta)
            for phase in PHASES:
                check_inputs(args, spec, p, meta, phase)
                if phase == 'poe' and args.scope == 'full':
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        gpu_job = pool.submit(runner.gpu_workers, args, phase)
                        cpu_job = pool.submit(runner.gpu_workers, args, 'poe_metrics', cpu=True)
                        gpu_job.result(); cpu_job.result()
                    metric_barrier(args, meta, poe=True)
                else:
                    runner.gpu_workers(args, phase)
                phase_barrier(args, spec, p, meta, phase)
                print('phase complete', phase, flush=True)
            check_inputs(args, spec, p, meta, 'after')
            status = summarize_execution(args, spec, p, meta)
            space = resource_guard(args.output, args.scope, spec)
            b.require(time.monotonic() < args.deadline_monotonic, 'overall completion budget exhausted')
            b.write_json(args.output/'EXECUTION_COMPLETION.json', dict(metadata=meta, status='COMPLETE', controller_result_code=0,
                scientific_status=status.get('reliability_status'), completed_at_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=time.monotonic()-started, final_resource=space, model_optimizer_steps=0))
            print('execution complete', status.get('reliability_status', status.get('status')), flush=True)
        except BaseException as failure:
            error = failure
            runner.request_stop(args.output, 'v2.2 controller failed; preserve evidence, no replay')
            record_failure(args.output, 'FAILURE_controller.json', meta, failure, exact_command=sys.argv)
            try:
                check_inputs(args, spec, p, meta, 'after_error')
            except Exception as audit_error:
                record_failure(args.output, 'FAILURE_after_error_audit.json', meta, audit_error)
            if not (args.output/'GATE1C_V22_STATUS.json').exists():
                b.write_json(args.output/'GATE1C_V22_STATUS.json', dict(metadata=meta, status=getattr(failure, 'status', 'BLOCKED_INCOMPLETE_EVIDENCE'),
                    error=str(failure), controller_result_code=1, method_registered=False, gate1_overall_status='FAIL_TRANSPORT_NOT_SUPPORTED'))
    # The controller log is closed before sealing; no logger may mutate it later.
    reporting.artifact_manifest(args.output)
    if error is not None: raise error
    print(status.get('reliability_status', status.get('status')), flush=True)
