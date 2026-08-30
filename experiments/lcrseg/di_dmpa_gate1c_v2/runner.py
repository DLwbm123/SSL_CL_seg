"""One create-only run in the preregistered phase order, then stop."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import torch
import yaml

from di_dmpa_gate1.recovery import request_stop, stop_requested, SHUTDOWN_TIMEOUT_SECONDS
from .binding import (PREREG, verify, require, complete, ModelMutation, checkpoint, records, read_json, write_json,
    write_text, sha256, check_hash, git, read_arrays, no_updates, legacy_input_audit)
from .execution import validation_unit, evaluate_unit, probe_unit, validate_scores, unit_name, pair_name
from .reporting import validate_probe_results, compile_report, artifact_manifest

ROOT = Path(__file__).resolve().parents[1]


def disk_hashes(p):
    result = {c['checkpoint_id']: check_hash(c['path'], c['sha256']) for c in p['immutable_baseline']['checkpoint_inputs']}
    require(len(result) == 9 and all(k.startswith('B0/') for k in result), 'nine B0 checkpoint inputs required')
    return result


def input_audit(root, data_root, p, metadata):
    root = Path(root); repo = root.parents[1]; base = p['immutable_baseline']; bench = p['benchmark']
    check_hash(repo/base['freeze_path'], base['freeze_sha256'])
    check_hash(repo/bench['domain_order_source']['path'], bench['domain_order_source']['sha256'])
    require(git(root/'third_party/JASCL_REFERENCE', 'rev-parse', 'HEAD') == base['upstream_jascl_commit'], 'official classifier commit changed')
    cfg = base['configs']['B0']; check_hash(repo/cfg['path'], cfg['file_sha256'])
    value = yaml.safe_load((repo/cfg['path']).read_text())
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    require(digest == cfg['resolved_config_sha256'], 'B0 config changed')
    units = []
    for a in bench['manifest_assets']:
        check_hash(Path(data_root)/f'manifests/training/lcrseg_v1_seed{a["seed"]}.csv', a['sha256'])
        check_hash(Path(data_root)/f'splits/fundus_seed{a["seed"]}.json', a['fundus_split_sha256'])
    for seed in range(3):
        for stage in range(3):
            counts = {role: len(records(data_root, p, seed, stage, role)) for role in ('train_labeled', 'train_unlabeled', 'val')}
            units.append(dict(seed=seed, stage_index=stage, counts=counts, current_domain_only=True, hidden_label_fields_empty=True))
    return dict(metadata=metadata, status='PASS', checkpoints=disk_hashes(p), units=units,
        legacy_payload_readiness=legacy_input_audit(p),
        manifests_and_splits_unchanged=True, label_roles=('train_labeled_supervised_reference', 'val_evaluator_only'),
        hidden_gt_training_usage='none', test_gt_usage='none', test_role_constructions=0,
        historical_raw_data_loaded=False, T1_T2_output_reads=0)


def model_audit(output, p, metadata, *, final):
    output = Path(output); expected = {}
    if p.get('legacy_prototype_reconstruction'):
        spec = p['legacy_prototype_reconstruction']; check_hash(spec['bank_path'], spec['bank_sha256'])
    for seed in range(3):
        for stage in range(3):
            cp = checkpoint(p, seed, stage)
            expected[f'validation_models/{unit_name(seed,stage)}/immutability/B0_seed{seed}_stage{stage}.json'] = cp['checkpoint_id']
    if final:
        for phase in ('draw0', 'noise', 'posterior', 'poe'):
            for pair in p['gradient_diagnostic']['batch_pairs']:
                expected[f'probe_models/{phase}/{pair_name(pair)}/immutability/B0_seed{pair["seed"]}_stage{pair["stage_index"]}.json'] = pair['checkpoint_id']
    observed = {str(path.relative_to(output)) for folder in ('validation_models', 'probe_models') for path in (output/folder).rglob('*.json')}
    complete(observed == set(expected), 'missing/extra model immutability audit')
    checks = []
    for rel, cp in expected.items():
        r = read_json(output/rel)
        if not r['bitwise_unchanged'] or r['before'] != r['after'] or r['checkpoint_sha256_before'] != r['checkpoint_sha256_after']:
            raise ModelMutation(rel)
        require(r['checkpoint_id'] == cp and r['status'] == 'PASS' and r['extraction_completed'], 'incomplete model guard')
        require(all(r['metadata'][k] == v for k, v in metadata.items()), 'mixed model provenance')
        checks.append(dict(path=rel, sha256=sha256(output/rel), checkpoint_id=cp, before=r['before'], after=r['after']))
    if final:
        for phase in ('draw0', 'noise', 'posterior', 'poe'):
            for pair in p['gradient_diagnostic']['batch_pairs']:
                r = read_json(output/'probes'/phase/pair_name(pair)/'isolation.json')
                require(r['teacher_gradients'] == r['prototype_gradients'] == r['history_bank_gradients'] == r['student_parameter_grad_fields'] == 'None', 'gradient isolation changed')
                require(r['legacy_prototypes_unchanged'] and r['current_history_banks_unchanged'] and not r['optimizer_constructed'] and not r['backward_called'], 'diagnostic state changed')
    return dict(metadata=metadata, status='PASS', guard_count=len(checks), guards=checks, checkpoint_disk_sha256=disk_hashes(p),
        all9_B0_checkpoints_unchanged=True, all_model_states_bitwise_unchanged=True, all_grad_fields_None=True,
        model_optimizer_steps=0, transport_optimizer_steps_this_gate=0)


def cache_audit(output, p, metadata):
    output = Path(output); units = []; census = []
    for plan in p['validation']['plans']:
        path = output/'validation_units'/(unit_name(plan['seed'], plan['stage_index'])+'.json'); r = read_json(path)
        require([c['case_id'] for c in r['cases']] == [c['case_id'] for c in plan['cases']], 'validation cache completeness')
        require(all(r['metadata'][k] == v for k, v in metadata.items()), 'mixed cache metadata')
        for c in r['cases']:
            a = read_arrays(c['arrays']); support = validate_scores(a, plan['stage_index'], 384*384)
            require(support == c['support'] and not c['GT_received_by_builder'], 'cache support/GT mismatch')
            census.append(dict(seed=plan['seed'], stage_index=plan['stage_index'], case_id=c['case_id'], **support,
                first_null_coordinates=c['first_null_coordinates'], raw_all_pixel_null_fraction=support['null']/support['rows']))
        units.append(dict(seed=plan['seed'], stage_index=plan['stage_index'], path=str(path), sha256=sha256(path), cases=r['cases']))
    complete(len(units) == 9 and len(census) == 495, '9/495 cache records required')
    write_json(output/'RELIABILITY_CACHE_MANIFEST.json', dict(metadata=metadata, status='PASS', unit_count=9, case_count=495, units=units))
    write_json(output/'RELIABILITY_SUPPORT_CENSUS.json', dict(metadata=metadata, status='PASS', cases=census,
        rows=sum(r['rows'] for r in census), active=sum(r['active'] for r in census), null=sum(r['null'] for r in census),
        null_UIDs_preserved=True, null_denominator_policy='all non-ignore mass retained by separate evaluator', GT_received_by_builder=False))


def phase_receipt(output, p, metadata, phase):
    output = Path(output)
    folder = output/('reliability_units' if phase == 'validation_metrics' else 'probes/'+phase)
    paths = sorted(folder.glob('*.json') if phase == 'validation_metrics' else folder.glob('*/result.json'))
    if phase == 'validation_metrics':
        complete(len(paths) == 9, 'incomplete validation metrics')
    else:
        validate_probe_results(p, [read_json(path) for path in paths], phase)
    result = dict(metadata=metadata, status='PASS', phase=phase, completed=len(paths),
        evidence_sha256={str(path.relative_to(output)): sha256(path) for path in paths})
    write_json(output/('PHASE_COMPLETION_'+phase+'.json'), result)


def worker(args):
    p, freeze, verified = verify(ROOT, args.code_commit, remote=False, input_contract=args.input_contract)
    meta = read_json(args.output/'GATE1C_V2_RUN_METADATA.json')
    require(meta['diagnostic_code_commit'] == args.code_commit, 'worker code changed')
    require(meta['input_contract_version'] == args.input_contract and
            meta['preregistration_commit'] == verified['preregistration_commit'] and
            meta['preregistration_file_sha256'] == verified['preregistration_file_sha256'], 'mixed input contract')
    write_json(args.output/'shards'/f'{args.phase}_{args.shard}_metadata.json', dict(meta, phase=args.phase, shard=args.shard,
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), device='cuda:0'))
    previous = {'draw0': 'validation_metrics', 'noise': 'draw0', 'posterior': 'noise', 'poe': 'posterior'}
    if args.phase in previous:
        receipt = read_json(args.output/('PHASE_COMPLETION_'+previous[args.phase]+'.json'))
        require(receipt['status'] == 'PASS', 'phase order barrier failed')
        for path, digest in receipt['evidence_sha256'].items():
            check_hash(args.output/path, digest)
    try:
        torch.set_num_threads(1)
        with no_updates():
            for index, (seed, stage) in enumerate((s, t) for s in range(3) for t in range(3)):
                if index % 2 != args.shard:
                    continue
                require(not stop_requested(args.output), 'another worker failed; no continuation')
                if args.phase == 'validation':
                    validation_unit(ROOT, args.data_root, p, freeze, meta, seed, stage, args.output, 'cuda:0')
                else:
                    probe_unit(ROOT, args.data_root, p, freeze, meta, seed, stage, args.output, 'cuda:0', args.phase)
    except Exception as error:
        write_json(args.output/'failures'/f'{args.phase}_shard{args.shard}.json', dict(status=getattr(error, 'status', 'BLOCKED_INCOMPLETE_EVIDENCE'),
            error=str(error), traceback=traceback.format_exc(), metadata=meta))
        request_stop(args.output, 'Gate1C worker failed; no automatic retry'); raise


def gpu_workers(args, phase):
    processes = []; handles = []; failed_at = None
    try:
        for gpu in (0, 1):
            path = args.output/'logs'/f'{phase}_gpu{gpu}.log'; path.parent.mkdir(parents=True, exist_ok=True)
            log = path.open('x'); handles.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONPATH=str(ROOT))
            command = [sys.executable, '-m', 'di_dmpa_gate1c_v2.runner', 'worker', '--code-commit', args.code_commit,
                '--output', str(args.output), '--data-root', str(args.data_root), '--phase', phase, '--shard', str(gpu),
                '--input-contract', args.input_contract]
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                if failed_at is None:
                    failed_at = time.monotonic(); request_stop(args.output, 'worker exit failure')
                if time.monotonic()-failed_at > SHUTDOWN_TIMEOUT_SECONDS:
                    for process in processes:
                        if process.poll() is None:
                            process.terminate()
                    break
            time.sleep(.2)
        complete(all(p.wait() == 0 for p in processes), f'{phase}: worker failure; see preserved logs')
    finally:
        for handle in handles:
            handle.close()


def metric_workers(args, p, *, poe=False):
    tasks = [(args.data_root, p, args.output, s, t, poe) for s in range(3) for t in range(3)]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        list(pool.map(evaluate_unit, tasks))


def run(args):
    if args.input_contract == 'v2.1':
        require(args.output.resolve().is_relative_to(Path('/root/LCRSeg/runs/di_dmpa_gate1c_v21')), 'v2.1 requires a separate output root')
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {}; started = datetime.now(timezone.utc).isoformat()
    try:
        p, freeze, metadata = verify(ROOT, args.code_commit, input_contract=args.input_contract)
        tests = read_json(args.tests/'GATE1C_V2_UNIT_INTEGRATION_TEST_REPORT.json')
        require(tests['status'] == 'PASS' and tests['diagnostic_code_commit'] == args.code_commit and tests['tests_failed'] == 0 and
                tests['tests_skipped'] == 0 and tests['tests_passed'] >= 56 and tests['real_integration_status'] == 'PASS', 'exact-code tests/integration barrier failed')
        require(tests.get('input_contract_version', 'v2') == args.input_contract, 'tests used a different input contract')
        if args.input_contract == 'v2.1':
            require(tests['preregistration_commit'] == metadata['preregistration_commit'] and
                    tests['preregistration_file_sha256'] == metadata['preregistration_file_sha256'], 'test registration mismatch')
        for name, digest in tests['files_sha256'].items():
            check_hash(args.tests/name, digest)
        for name in ('pytest.xml', 'pytest_output.txt', 'GATE1C_V2_UNIT_INTEGRATION_TEST_REPORT.json', 'GATE1C_V2_REAL_INTEGRATION.json'):
            shutil.copy2(args.tests/name, args.output/name)
        metadata.update(started_at_utc=started, python=sys.version, torch_version=torch.__version__, platform=platform.platform(),
            gpu_workers=2, cpu_metric_workers=2, physical_gpus=[0, 1],
            execution_scope='GATE1C_V21_ONLY' if args.input_contract == 'v2.1' else 'GATE1C_V2_ONLY', method_flags=p['method_flags'])
        write_json(args.output/'GATE1C_V2_RUN_METADATA.json', metadata)
        write_json(args.output/'GATE1C_V2_INPUT_AUDIT.json', input_audit(ROOT, args.data_root, p, metadata))
        gpu_workers(args, 'validation'); cache_audit(args.output, p, metadata)
        initial_audit = model_audit(args.output, p, metadata, final=False)
        write_json(args.output/'VALIDATION_MODEL_IMMUTABILITY_AUDIT.json', initial_audit)
        write_json(args.output/'VALIDATION_CACHE_BARRIER.json', dict(metadata=metadata, status='PASS', validation_units=9,
            evidence_sha256={n: sha256(args.output/n) for n in ('RELIABILITY_CACHE_MANIFEST.json', 'RELIABILITY_SUPPORT_CENSUS.json', 'VALIDATION_MODEL_IMMUTABILITY_AUDIT.json')}))
        metric_workers(args, p); phase_receipt(args.output, p, metadata, 'validation_metrics')
        for phase in ('draw0', 'noise', 'posterior'):
            gpu_workers(args, phase); phase_receipt(args.output, p, metadata, phase)
        with ThreadPoolExecutor(max_workers=1) as pool:
            gpu = pool.submit(gpu_workers, args, 'poe'); metric_workers(args, p, poe=True); gpu.result()
        phase_receipt(args.output, p, metadata, 'poe')
        audit = model_audit(args.output, p, metadata, final=True)
        write_json(args.output/'GATE1C_V2_MODEL_IMMUTABILITY_AUDIT.json', audit)
        status = compile_report(args.output, p, metadata, audit)
        write_text(args.output/'GATE1C_V2_EXACT_COMMANDS.md', '# Gate 1C '+args.input_contract+' exact runtime command\n\n'
            'Existing Python environment; two GPU workers; no training/optimizer. See the report publication record for preparation/test commands.\n\n```sh\n'
            'cd '+str(ROOT)+'\nOMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/lib/x86_64-linux-gnu '
            'CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH='+str(ROOT)+' '+sys.executable+' -m di_dmpa_gate1c_v2.runner run '
            '--code-commit '+args.code_commit+' --output '+str(args.output)+' --tests '+str(args.tests)+' --data-root '+str(args.data_root)
            +' --input-contract '+args.input_contract+'\n```\n\n'
            'All phases are create-only and outcome-independent. No automatic retry. STOP_FOR_INDEPENDENT_REVIEW.\n')
        write_json(args.output/'EXECUTION_COMPLETION.json', dict(metadata=metadata, status='COMPLETE', started_at_utc=started,
            completed_at_utc=datetime.now(timezone.utc).isoformat(), scientific_status=status['reliability_status'], model_optimizer_steps=0, transport_optimizer_steps_this_gate=0))
        artifact_manifest(args.output)
        print(status['reliability_status'], flush=True)
    except Exception as error:
        status = getattr(error, 'status', 'BLOCKED_INCOMPLETE_EVIDENCE')
        failure = dict(metadata=metadata, status=status, error=str(error), traceback=traceback.format_exc(),
            model_optimizer_steps=0, transport_optimizer_steps_this_gate=0, method_registered=False, di_dmpa_training_launched=False,
            gate1_overall_status='FAIL_TRANSPORT_NOT_SUPPORTED', next_action='STOP_FOR_INDEPENDENT_REVIEW')
        write_json(args.output/'GATE1C_V2_FAILURE.json', failure)
        if not (args.output/'GATE1C_V2_STATUS.json').exists():
            write_json(args.output/'GATE1C_V2_STATUS.json', failure)
        if args.input_contract == 'v2.1' and not (args.output/'GATE1C_V21_STATUS.json').exists():
            write_json(args.output/'GATE1C_V21_STATUS.json', dict(failure, input_contract_version='v2.1', original_gate1c_v2_completed=False))
        if not (args.output/'GATE1C_V2_ARTIFACT_MANIFEST.json').exists():
            artifact_manifest(args.output)
        raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('run', 'worker'))
    parser.add_argument('--code-commit', required=True); parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--data-root', default='/root/LCRSeg', type=Path); parser.add_argument('--tests', type=Path)
    parser.add_argument('--input-contract', choices=('v2', 'v2.1'), default='v2')
    parser.add_argument('--phase', choices=('validation', 'draw0', 'noise', 'posterior', 'poe')); parser.add_argument('--shard', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.action == 'run':
        require(args.tests is not None, 'tests receipt required'); run(args)
    else:
        require(args.phase is not None and args.shard is not None, 'worker phase/shard required'); worker(args)


if __name__ == '__main__':
    main()
