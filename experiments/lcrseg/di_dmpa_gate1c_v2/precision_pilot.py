"""Published, create-only three-pair engineering pilot; no scientific admission."""
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import traceback
from unittest.mock import patch

import numpy as np
import torch

from di_dmpa_jascl.modeling import LCRSegUNet2DJASCL, pas_probability_objective
from . import binding as b, execution as e, gradients as g
from .runner import input_audit, disk_hashes
from .precision import comparable

ROOT = Path(__file__).resolve().parents[1]
PREREG = '6357317749b0ff904e3acd39023b86430d6263ee'
NAME = 'DI_DMPA_GATE1C_V22_PRECISION_PREREGISTRATION'
HASHES = {'md': '2ec5ba23d8136bbb3870776345a0be229e6071ba9a1e58604884dda42b2a433a',
          'json': '2ceb37fc571b17373261fe631c8a2e416130912e2e882461b9e42795d495aeca'}
PHASES = ('draw0', 'noise', 'posterior', 'poe')
COUNTS = {'draw0': (3, 2, 9, 33), 'noise': (9, 2, 57, 57), 'posterior': (3, 2, 9, 9), 'poe': (2, 2, 17, 23)}
COUNT_KEYS = ('native_forwards', 'shadow_forwards', 'native_autograd', 'shadow_autograd')


def verify(code, *, remote):
    repo = ROOT.parents[1]; docs = ROOT/'docs/di_dmpa_jascl'
    for suffix, digest in HASHES.items():
        path = docs/f'{NAME}.{suffix}'; b.check_hash(path, digest)
        blob = subprocess.check_output(['git', '-C', str(repo), 'show', f'{PREREG}:{path.relative_to(repo)}'])
        b.require(hashlib.sha256(blob).hexdigest() == digest, 'precision preregistration blob changed')
    b.verify_ancestor(repo, PREREG, code)
    spec = b.read_json(docs/f'{NAME}.json')
    for path, digest in ((spec['authority']['path'], spec['authority']['sha256']),
                         (spec['evidence']['report_path'], spec['evidence']['report_sha256']),
                         (spec['inherited']['json_path'], spec['inherited']['json_sha256'])):
        b.check_hash(repo/path, digest)
    protected = ['experiments/lcrseg/'+x for x in ('di_dmpa_jascl', 'di_dmpa_gate1', 'di_dmpa_gate1_v2', 'di_dmpa_gate1b_v2')]
    protected += ['experiments/lcrseg/di_dmpa_gate1c_v2/'+x for x in ('binding.py', 'reliability.py', 'metrics.py', 'reporting.py', 'runner.py')]
    b.require(not b.git(repo, 'diff', spec['evidence']['formal_code_commit'], 'HEAD', '--', *protected), 'unregistered engine change')
    upstream = ROOT/'third_party/JASCL_REFERENCE'
    b.require(b.git(upstream, 'rev-parse', 'HEAD') == '3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53' and
        not b.git(upstream, 'diff', '--name-only', 'HEAD'), 'official tracked source changed')
    p, freeze, meta = b.verify(ROOT, code, remote=remote, input_contract='v2.1')
    b.require(len(spec['pilot']['pairs']) == 3 and all(q in p['gradient_diagnostic']['batch_pairs'] for q in spec['pilot']['pairs']), 'pilot pairs changed')
    for golden in spec['native_goldens']: b.check_hash(golden['path'], golden['sha256'])
    ref = spec['failed_pair_reference']
    b.check_hash(repo/ref['native_outcome_path'], ref['native_outcome_sha256'])
    b.check_hash(ref['reference_details_path'], ref['reference_details_sha256'])
    p.update(diagnostic_precision=spec['repair']['mode'], _precision_contract_verified=True)
    meta.update(numeric_preregistration_commit=PREREG, numeric_preregistration_file_sha256=HASHES,
        numeric_version=spec['repair']['numeric_version'], execution_scope=spec['scope'],
        scientific_admission=None, method_registered=False, full_gate_retry_authorized=False)
    output = Path(spec['execution']['output_root_prefix'])/PREREG/spec['execution']['attempt']
    return spec, p, freeze, meta, output


def load_run(code):
    spec, p, freeze, verified, output = verify(code, remote=False)
    meta = b.read_json(output/'RUN_METADATA.json')
    for key in ('diagnostic_code_commit', 'numeric_preregistration_commit', 'numeric_preregistration_file_sha256', 'numeric_version'):
        b.require(meta[key] == verified[key], 'mixed precision run provenance')
    b.require(meta['remote_verified_code_commit'] == code, 'missing published-code barrier')
    return spec, p, freeze, meta, output


def prepare(code, tests):
    spec, p, freeze, meta, output = verify(code, remote=True)
    import xml.etree.ElementTree as ET
    suites = list(ET.parse(tests).getroot().iter('testsuite'))
    b.require(suites and sum(int(s.attrib['tests']) for s in suites) >= 126 and
        all(int(s.attrib[k]) == 0 for s in suites for k in ('failures', 'errors', 'skipped')), 'synthetic test barrier failed')
    properties = {x.attrib['name']: x.attrib['value'] for s in suites for x in s.findall('properties/property')}
    b.require(properties.get('diagnostic_code_commit') == code and properties.get('source_clean') == 'true', 'tests not bound to clean exact code')
    b.require(shutil.disk_usage('/root/LCRSeg').free >= spec['execution']['minimum_available_root_bytes'], 'pilot disk headroom')
    output.mkdir(parents=True, exist_ok=False)
    meta.update(started_at_utc=datetime.now(timezone.utc).isoformat(), synthetic_junit_path=str(tests),
                synthetic_junit_sha256=b.sha256(tests), torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                physical_gpus=[0, 1], exact_prepare_command=sys.argv)
    b.write_json(output/'RUN_METADATA.json', meta)
    b.write_json(output/'INPUT_AUDIT.json', input_audit(ROOT, '/root/LCRSeg', p, meta))
    print(dict(output=str(output), status='PREPARED_NO_FORWARDS'), flush=True)


def worker(code, phase, gpu):
    spec, p, freeze, meta, output = load_run(code)
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu) and torch.cuda.device_count() == 1, 'GPU assignment changed')
    previous = PHASES.index(phase)-1
    if previous >= 0:
        receipt = b.read_json(output/f'PHASE_{PHASES[previous]}.json')
        b.require(receipt['status'] == 'PASS', 'pilot phase barrier not passed')
        for path, digest in receipt['evidence_sha256'].items(): b.check_hash(output/path, digest)
    torch.set_num_threads(1)
    selected = [q for q in spec['pilot']['pairs'] if q['batch_id'] in spec['pilot']['assignment'][str(gpu)]]
    start = output/f'WORKER_{phase}_gpu{gpu}_START.json'
    b.require(not start.exists(), 'worker already attempted; no automatic replay')
    b.write_json(start, dict(metadata=meta, phase=phase, gpu=gpu, pid=os.getpid(), device_name=torch.cuda.get_device_name(0),
        exact_command=sys.argv, started_at_utc=datetime.now(timezone.utc).isoformat()))
    counts = dict.fromkeys(COUNT_KEYS, 0); parity = []; original_build = e.build
    original_forward = LCRSegUNet2DJASCL.forward; original_grad = torch.autograd.grad

    def observed_forward(model, *args, **kwargs):
        dtype = next(model.parameters()).dtype
        b.require(dtype in (torch.float32, torch.float64), 'unexpected forward dtype')
        counts['native_forwards' if dtype == torch.float32 else 'shadow_forwards'] += 1
        return original_forward(model, *args, **kwargs)

    def observed_grad(loss, inputs, **kwargs):
        dtype = inputs[0].dtype
        b.require(dtype in (torch.float32, torch.float64), 'unexpected gradient dtype')
        counts['native_autograd' if dtype == torch.float32 else 'shadow_autograd'] += 1
        return original_grad(loss, inputs, **kwargs)

    def checked_build(sl, sf, tl, tf, legacy, current, history):
        result = original_build(sl, sf, tl, tf, legacy, current, history)
        with torch.no_grad():
            _, _, _, valid = pas_probability_objective(lambda *a, **kw: (sl, sf), lambda *a, **kw: (tl, tf), None, legacy)
        b.require(np.array_equal(result['R1'], valid.cpu().numpy().reshape(-1)), 'native Gate0 R1 parity failed')
        e.validate_scores({k: result[k] for k in e.CACHE_FIELDS}, active_pair['stage_index'], valid.numel())
        if active_pair['stage_index'] == 0: b.require(np.array_equal(result['R2'], result['R3']), 'stage0 R3/R2 mismatch')
        parity.append(dict(batch_id=active_pair['batch_id'], pixels=valid.numel(), exact_R1_parity=True,
                           null_pixels=int((~result['active_mask']).sum())))
        return result

    def timeout(signum, frame):
        raise TimeoutError('registered pilot worker/phase time budget exceeded')

    signal.signal(signal.SIGALRM, timeout); signal.alarm(spec['execution']['worker_minutes_per_phase']*60)
    try:
        with b.no_updates(), patch.object(e, 'build', checked_build), patch.object(LCRSegUNet2DJASCL, 'forward', observed_forward), patch.object(torch.autograd, 'grad', observed_grad):
            for active_pair in selected:
                b.require(not list(output.glob('FAILURE_*.json')), 'another pilot worker failed')
                e.probe_unit(ROOT, '/root/LCRSeg', p, freeze, meta, active_pair['seed'], active_pair['stage_index'], output,
                             'cuda:0', phase, pair_indices=[active_pair['pair_index']])
        expected = dict(zip(COUNT_KEYS, [n*len(selected) for n in COUNTS[phase]]))
        b.require(counts == expected, 'pilot forward/autograd count outside registration')
        flags = dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            autocast=torch.is_autocast_enabled())
        b.require(flags == dict(deterministic_algorithms=True, cudnn_deterministic=True, cudnn_benchmark=False,
            cudnn_allow_tf32=False, matmul_allow_tf32=False, autocast=False), 'native backend flags changed')
        disk_hashes(p)
        b.write_json(output/f'WORKER_{phase}_gpu{gpu}.json', dict(metadata=meta, status='PASS', phase=phase, gpu=gpu,
            completed_at_utc=datetime.now(timezone.utc).isoformat(), counts=counts, R1_parity=parity, backend_flags=flags,
            pairs=[q['batch_id'] for q in selected], all9_checkpoint_hashes_unchanged=True))
        print(dict(phase=phase, gpu=gpu, status='PASS', counts=counts), flush=True)
    except Exception as error:
        b.write_json(output/f'FAILURE_{phase}_gpu{gpu}.json', dict(metadata=meta, error=str(error), traceback=traceback.format_exc(), counts=counts,
            failed_at_utc=datetime.now(timezone.utc).isoformat(), scientific_admission=None))
        raise
    finally:
        signal.alarm(0)


def validate_result(result, pair, phase, meta):
    b.require(result['metadata'] == meta and result['pair'] == pair and result['phase'] == phase, 'pilot result identity changed')
    b.require(result['diagnostic_precision'] == 'float64_shadow' and result['no_optimizer'] and result['no_backward'] and result['no_parameter_grad_writes'], 'pilot mode/updates changed')
    candidates = ('PoE',) if phase == 'poe' else ('R0', 'R1', 'R2', 'R3')
    draws = range(8) if phase in ('noise', 'poe') else [0]
    expected = {(c, n, d, block) for c in candidates for n in g.NORMALIZATIONS for d in draws for block in ('global', *g.BLOCKS)}
    for field in ('alignment', 'native_precision_comparisons'):
        rows = result[field]
        b.require(len(rows) == len(expected) and {(r['candidate'], r['normalization'], r['draw_index'], r['block']) for r in rows} == expected, 'incomplete/duplicate pilot rows')
    expected_classes = {(c, n, 0, block, k) for c in candidates for n in g.NORMALIZATIONS for block in ('global', *g.BLOCKS) for k in range(3)} if phase in ('draw0', 'poe') else set()
    components = result['class_contribution']
    b.require(len(components) == len(expected_classes) and
        {(r['candidate'], r['normalization'], r['draw_index'], r['block'], r['class_id']) for r in components} == expected_classes and
        all(r['component_sum_pass'] for r in components), 'pilot class decomposition incomplete')
    for row in result['alignment']+result['native_precision_comparisons']+components:
        b.require(all(row[k] == pair[k] for k in ('batch_id', 'seed', 'stage_index', 'domain', 'pair_index')) and
            row['teacher_kind'] == ('posterior_mean' if phase == 'posterior' else 'stochastic'), 'pilot row provenance changed')
    for row in result['alignment']+components:
        b.require((row['cosine'] is None) == row['zero_gradient'], 'zero-gradient null changed')
        b.finite([x for x in (row['cosine'], row['norm_ratio'], row['supervised_norm'], row['unsupervised_norm']) if x is not None])
    supervised = result['supervised_precision_comparisons']
    b.require(set(supervised) == {'global', *g.BLOCKS} and
        result['supervised_precision_comparable'] == comparable(supervised['global']), 'supervised comparison schema changed')
    b.require(result['supervised_precision_comparable'] and all(r['precision_comparable'] == comparable(r) for r in result['native_precision_comparisons']) and
        all(r['precision_comparable'] for r in result['native_precision_comparisons'] if r['block'] == 'global'), 'FAIL_NUMERIC_COMPARABILITY')
    inventory = result['parameter_inventory']
    b.require(all(r['dtype'] == 'torch.float64' and r['expected_gradient'] == ('Tensor' if r['active'] else 'None') for r in inventory) and
        {r['name'] for r in inventory if not r['active']} == g.INACTIVE, 'shadow parameter inventory changed')
    replay = result['student_draw_replay']
    b.require(set(replay) == {'unlabeled', 'labeled'} and all(r['rng_before_shadow'] == r['rng_after_shadow'] and
        r['seed'] == pair['forward_seeds']['student_'+role] and r['native_forwards'] == r['shadow_forwards'] == 1
        for role, r in replay.items()), 'shadow RNG/forward identity changed')
    b.require(result['teacher_forwards'] == {'draw0': 1, 'noise': 7, 'posterior': 1, 'poe': 0}[phase], 'teacher forward count changed')
    if phase in ('noise', 'poe'): b.require(result['teacher_draw_seeds'] == pair['teacher_draw_seeds'], 'teacher seeds changed')
    if phase == 'posterior': b.require(not result['teacher_stochastic_classifier'] and not result['baseline_replacement'], 'posterior control changed')
    if phase == 'poe': b.require(result['same_detached_R3_weight'] and result['own_predicted_class_strata'] and result['cached_teacher_draws_used'] == 8, 'PoE control changed')
    return dict(alignment_rows=len(expected), global_comparisons=len(expected)//7, class_components=len(expected_classes), supervised_global_comparisons=1)


def golden_checks(spec, results):
    by_id = {r['pair']['batch_id']: r for r in results}
    for golden in spec['native_goldens']:
        old = b.read_json(golden['path']); new = by_id[golden['batch_id']]
        for key in ('student_logits_sha256', 'student_features_sha256', 'labeled_logits_sha256', 'teacher_features_sha256', 'teacher_probability_sha256', 'R1_validity_sha256'):
            b.require(new[key] == old[key], 'native golden forward/PAS changed: '+key)
        b.require(new['native_supervised_gradient_sha256'] == old['supervised_gradient_sha256'], 'native supervised golden changed')
        for row in new['native_precision_comparisons']:
            if row['block'] == 'global':
                b.require(row['native_sha256'] == old['gradient_hashes']['0'][row['candidate']+'/'+row['normalization']], 'native gradient golden changed')
    ref = spec['failed_pair_reference']; result = by_id[ref['batch_id']]
    old = b.read_json(ROOT.parents[1]/ref['native_outcome_path'])['details']
    row = next(r for r in result['native_precision_comparisons'] if (r['candidate'], r['normalization'], r['block']) == ('R2', 'class_balanced', 'global'))
    b.require(result['native_student_probability_sha256'] == old['probability_sha256'] and
        row['native_sha256'] == old['block_details']['global']['total_sha256'] and
        row['target_float32_sha256'] == old['target_sha256'] and row['weights_sha256'] == old['weights_sha256'] and row['class_strata_sha256'] == old['predicted_sha256'], 'failed native golden changed')
    b.require(result['gradient_hashes']['0']['R2/class_balanced'] == ref['expected_R2_class_balanced_FP64_global_sha256'], 'published FP64 golden changed')


def barrier(code, phase):
    spec, p, freeze, meta, output = load_run(code)
    b.require(not list(output.glob('FAILURE_*.json')), 'pilot worker failure')
    results = []; evidence = {}; totals = dict(alignment_rows=0, global_comparisons=0, class_components=0, supervised_global_comparisons=0)
    for gpu in (0, 1):
        path = output/f'WORKER_{phase}_gpu{gpu}.json'; worker_receipt = b.read_json(path)
        assigned = spec['pilot']['assignment'][str(gpu)]
        b.require(worker_receipt['status'] == 'PASS' and worker_receipt['metadata'] == meta and
            worker_receipt['phase'] == phase and worker_receipt['gpu'] == gpu and worker_receipt['pairs'] == assigned and
            worker_receipt['counts'] == dict(zip(COUNT_KEYS, [n*len(assigned) for n in COUNTS[phase]])) and
            worker_receipt['all9_checkpoint_hashes_unchanged'], 'missing/changed worker receipt')
        evidence[str(path.relative_to(output))] = b.sha256(path)
    for pair in spec['pilot']['pairs']:
        path = output/'probes'/phase/e.pair_name(pair)/'result.json'; result = b.read_json(path)
        counts = validate_result(result, pair, phase, meta); results.append(result)
        for key, value in counts.items(): totals[key] += value
        iso_path = path.with_name('isolation.json'); iso = b.read_json(iso_path)
        b.require(iso['legacy_prototypes_unchanged'] and iso['current_history_banks_unchanged'] and all(iso[k] == 'None' for k in
            ('teacher_gradients', 'prototype_gradients', 'history_bank_gradients', 'student_parameter_grad_fields')) and
            not iso['optimizer_constructed'] and not iso['backward_called'] and all(iso['metadata'][k] == v for k, v in meta.items()), 'pilot isolation failure')
        guard_path = output/'probe_models'/phase/e.pair_name(pair)/'immutability'/f'B0_seed{pair["seed"]}_stage{pair["stage_index"]}.json'
        guard = b.read_json(guard_path)
        b.require(guard['bitwise_unchanged'] and guard['extraction_completed'] and guard['before'] == guard['after'] and
            set(guard['before']) == {'student', 'ema_teacher', 'gradient_student'} and guard['status'] == 'PASS' and
            guard['checkpoint_id'] == pair['checkpoint_id'] and guard['checkpoint_sha256_before'] == guard['checkpoint_sha256_after'] == pair['checkpoint_sha256'] and
            all(guard['metadata'][k] == v for k, v in meta.items()), 'pilot model guard failure')
        for item in (path, iso_path, guard_path): evidence[str(item.relative_to(output))] = b.sha256(item)
        for field in ('primary_cache', 'teacher_cache'):
            if field in result:
                desc = result[field]; b.read_arrays(desc)
                item = Path(desc['path']); b.require(item.stat().st_size == desc['bytes'], 'cache size changed')
                evidence[str(item.relative_to(output))] = desc['sha256']
    b.require({q.name for q in (output/'probes'/phase).iterdir()} == {e.pair_name(q) for q in spec['pilot']['pairs']}, 'extra pilot pair')
    if phase == 'draw0': golden_checks(spec, results)
    path = output/f'PHASE_{phase}.json'; b.require(not path.exists(), 'phase already sealed')
    b.write_json(path, dict(metadata=meta, status='PASS', phase=phase, totals=totals, evidence_sha256=evidence))
    print(dict(phase=phase, status='PASS', **totals), flush=True)


def report(code):
    spec, p, freeze, meta, output = load_run(code)
    b.require(not list(output.glob('FAILURE_*.json')), 'pilot has a failure marker')
    totals = dict.fromkeys(COUNT_KEYS, 0); coverage = dict(alignment_rows=0, global_comparisons=0, class_components=0, supervised_global_comparisons=0); files = []
    for phase in PHASES:
        receipt = b.read_json(output/f'PHASE_{phase}.json')
        b.require(receipt['status'] == 'PASS' and receipt['phase'] == phase and receipt['metadata'] == meta, 'unsealed pilot phase')
        for path, digest in receipt['evidence_sha256'].items(): b.check_hash(output/path, digest)
        for key, value in receipt['totals'].items(): coverage[key] += value
        for gpu in (0, 1):
            row = b.read_json(output/f'WORKER_{phase}_gpu{gpu}.json')
            for key, value in row['counts'].items(): totals[key] += value
    b.require(totals == dict(zip(COUNT_KEYS, (51, 24, 276, 366))), 'total pilot budget mismatch')
    b.require(coverage == dict(alignment_rows=2016, global_comparisons=288, class_components=630, supervised_global_comparisons=12) and
        len(list((output/'probe_models').rglob('immutability/*.json'))) == 12 and
        len(list((output/'probes').glob('*/*/result.json'))) == 12 and
        {q.name for q in (output/'probes').iterdir()} == set(PHASES), 'pilot coverage incomplete')
    disk_hashes(p)
    for path in sorted(output.rglob('*')):
        if path.is_file(): files.append(dict(path=str(path.relative_to(output)), sha256=b.sha256(path), bytes=path.stat().st_size))
    b.require(sum(x['bytes'] for x in files) <= spec['execution']['pilot_artifact_byte_budget'], 'pilot artifact budget exceeded')
    b.require(not (output/'PILOT_STATUS.json').exists(), 'pilot already reported')
    b.write_json(output/'PILOT_STATUS.json', dict(metadata=meta, status='PASS_NUMERIC_PRECISION_PILOT', counts=totals,
        pairs=3, phases=4, coverage=coverage, model_immutability_guards=12, scientific_admission=None, full_gate_retry_authorized=False,
        method_registered=False, model_optimizer_steps=0, transport_optimizer_steps=0,
        hidden_gt_training_usage='none', test_gt_usage='none', all9_checkpoint_hashes_unchanged=True,
        old_Gate1C_v21_status='BLOCKED_INCOMPLETE_EVIDENCE', gate1_overall_status='FAIL_TRANSPORT_NOT_SUPPORTED',
        completed_at_utc=datetime.now(timezone.utc).isoformat(), next_action='REPORT_ALL_EVIDENCE_THEN_NEW_FINITE_PLAN'))
    path = output/'PILOT_STATUS.json'; files.append(dict(path=path.name, sha256=b.sha256(path), bytes=path.stat().st_size))
    b.write_json(output/'PILOT_ARTIFACT_MANIFEST.json', dict(files=files, total_bytes=sum(x['bytes'] for x in files)))
    print(dict(output=str(output), status='PASS_NUMERIC_PRECISION_PILOT', counts=totals), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'worker', 'barrier', 'report'))
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--tests', type=Path)
    parser.add_argument('--phase', choices=PHASES); parser.add_argument('--gpu', type=int, choices=(0, 1))
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            b.require(args.tests is not None, 'exact-code JUnit required'); prepare(args.code_commit, args.tests)
        elif args.action == 'worker':
            b.require(args.phase is not None and args.gpu is not None, 'phase/GPU required'); worker(args.code_commit, args.phase, args.gpu)
        elif args.action == 'barrier':
            b.require(args.phase is not None, 'phase required'); barrier(args.code_commit, args.phase)
        else:
            report(args.code_commit)
    except Exception as error:
        output = Path('/root/LCRSeg/runs/gate1c_v22_precision_pilot')/PREREG/'attempt1'
        failure = output/f'FAILURE_{args.action}_{args.phase or "all"}.json'
        if args.action != 'worker' and output.is_dir() and not failure.exists():
            b.write_json(failure, dict(diagnostic_code_commit=args.code_commit, exact_command=sys.argv,
                error=str(error), traceback=traceback.format_exc(), failed_at_utc=datetime.now(timezone.utc).isoformat(), scientific_admission=None))
        raise
