"""Byte-only postrun verification and summaries of existing v2 JSON results.

Does not import Torch, load feature tensors, fit centers, rerun diagnostics,
or write inside any original attempt directory.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

CODE = '8ae5d7532f90aee5d53c0d966706ef64c18a19ac'
PREREG = 'eaae37bbaa7546679d9e6893023afbeeef0ab5c6'
V1 = 'cfb62554f1e6a2a36850547485b1857dc9a28a20'
PLAN = '96277c841165d8cd84a63d3d4a9301b27c50dcdd87d0b0d215c162d95c345e24'
REPO = Path('/root/SSL_CL_gate1_v2')
DOCS = REPO / 'experiments/lcrseg/docs/di_dmpa_jascl'
PARENT = Path('/root/LCRSeg/runs/di_dmpa_gate1_v2') / PREREG
FORMAL = PARENT / f'gate1a_v2_{CODE}_attempt1'
OUTPUT = PARENT / 'postrun_8ae5d75_attempt1'
PANELS = ('B0-EMA', 'B0-student', 'C0-EMA', 'C0-student')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(name, payload):
    with (OUTPUT / name).open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def command(args):
    return subprocess.check_output(args, text=True, timeout=60).strip()


def manifest_audit(root, expected_sha=None):
    path = root / 'GATE1A_ARTIFACT_MANIFEST.json'
    observed_sha = sha(path)
    manifest = read(path)
    errors = []
    if expected_sha and observed_sha != expected_sha:
        errors.append({'manifest': 'sha256 mismatch'})
    for item in manifest['files']:
        target = root / item['path']
        if not target.is_file():
            errors.append({'path': item['path'], 'error': 'missing'})
        elif target.stat().st_size != item['size_bytes'] or sha(target) != item['sha256']:
            errors.append({'path': item['path'], 'error': 'size/sha256 mismatch'})
    return dict(status='PASS' if not errors else 'FAIL', root=str(root),
                manifest_sha256=observed_sha, files_verified=len(manifest['files']),
                bytes_verified=sum(f['size_bytes'] for f in manifest['files']), errors=errors)


def main():
    completion = read(PARENT / 'formal_8ae5d75_attempt1.completion.json')
    assert completion['exit_code'] == 0, 'formal attempt must finish before postrun verification'
    OUTPUT.mkdir(exist_ok=False)
    old = read(DOCS / 'DI_DMPA_GATE1_PREREGISTRATION.json')
    status = read(FORMAL / 'GATE1A_V2_STATUS.json')
    metadata = read(FORMAL / 'GATE1A_V2_RUN_METADATA.json')
    manifest = manifest_audit(FORMAL)
    manifest['v2_named_copy_byte_identical'] = sha(FORMAL / 'GATE1A_V2_ARTIFACT_MANIFEST.json') == manifest['manifest_sha256']
    history = []
    historical = (
        ('gate1a_formal_8f4a71a_attempt1', 'c26edceea102da568421e0327a7cc10fabb2ceee16fc936dd8adeb439eab8ee9'),
        ('gate1a_formal_a89716ddbd2eccbe76c574e97e520d424aa923ab_attempt2', '15e7beaf67ad55bd1c18b494f53c7f06fc6ea92ff161f99ccf62b6a648736ea3'),
    )
    for name, expected in historical:
        root = Path('/root/LCRSeg/runs/di_dmpa_gate1') / V1 / name
        entry = manifest_audit(root, expected)
        state = read(root / 'GATE1A_STATUS.json')
        entry['original_state'] = state
        entry['observed_geometry_files'] = len(list((root / 'geometry_units').glob('*.json')))
        history.append(entry)
    checkpoints = []
    for cp in old['immutable_baseline']['checkpoint_inputs']:
        observed = sha(cp['path'])
        checkpoints.append(dict(checkpoint_id=cp['checkpoint_id'], expected=cp['sha256'],
                                observed=observed, unchanged=observed == cp['sha256']))
    feature_files = sorted((FORMAL / 'feature_units').glob('*.json'))
    geometry_files = sorted((FORMAL / 'geometry_units').glob('*.json'))
    feature_keys = []
    for path in feature_files:
        e = read(path)
        feature_keys.append((e['panel_id'], e['seed'], e['stage_index'], e['role']))
    expected_features = {(p,s,t,r) for p in PANELS for s in range(3) for t in range(3) for r in ('train_labeled','val')}
    expected_geometry = {(p,s,t,c,k) for p in PANELS for s in range(3) for t in range(3) for c in range(3) for k in (1,2,3,5)}
    rows = [read(path) for path in geometry_files]
    geometry_keys = [(r['panel_id'],r['seed'],r['stage_index'],r['class_id'],r['K']) for r in rows]
    all_fits = []
    restart_warnings = []
    for r in rows:
        ids = {k:r[k] for k in ('panel_id','seed','stage_index','domain','class_id','K')}
        for replicate, fit in [(-1,r['fit'])] + [(b['replicate'],b['fit']) for b in r['bootstrap']]:
            row = dict(ids, replicate=replicate, selected_restart=fit['selected_restart'],
                       converged=fit['converged'], iterations=fit['iterations'],
                       directional_support=fit['directional_support'], inactive_slots=sum(not a for a in fit['active']))
            all_fits.append(row)
            for restart in fit['restarts']:
                if not restart['converged']:
                    restart_warnings.append(dict(ids,replicate=replicate,**restart))
    census = read(FORMAL / 'GATE1A_V2_FEATURE_SUPPORT_CENSUS.json')
    diagnostic = read(FORMAL / 'PROTOTYPE_GEOMETRY_DIAGNOSTIC_V2.json')
    barrier = FORMAL / 'GEOMETRY_START_BARRIER.json'
    barrier_data = read(barrier)
    tests = read(FORMAL / 'GATE1A_V2_UNIT_INTEGRATION_TEST_REPORT.json')
    immutability = read(FORMAL / 'GATE1A_V2_MODEL_IMMUTABILITY_AUDIT.json')
    raw_array_files = list((FORMAL / 'features').rglob('*.npy'))
    checks = dict(
        formal_exit_zero=completion['exit_code']==0,
        complete72_unique_features=len(feature_keys)==72 and set(feature_keys)==expected_features,
        complete432_unique_geometry=len(geometry_keys)==432 and set(geometry_keys)==expected_geometry,
        cache_arrays648=len(raw_array_files)==648,
        every_geometry_has_five_bootstraps=all(len(r['bootstrap'])==5 for r in rows),
        every_fit_has_five_restarts=all(len(fit['restarts'])==5 for r in rows for fit in [r['fit']]+[b['fit'] for b in r['bootstrap']]),
        geometry_metadata_bound=all(r['metadata']['diagnostic_code_git_commit']==CODE and r['metadata']['v2_preregistration_git_commit']==PREREG and r['metadata']['sampling_plan_sha256']==PLAN for r in rows),
        all72_features_written_before_barrier=max(p.stat().st_mtime_ns for p in feature_files)<=barrier.stat().st_mtime_ns,
        all18_immutability_audits_before_barrier=max(p.stat().st_mtime_ns for p in (FORMAL/'immutability').glob('*.json'))<=barrier.stat().st_mtime_ns,
        geometry_files_after_barrier=min(p.stat().st_mtime_ns for p in geometry_files)>=barrier.stat().st_mtime_ns,
        census_hash_matches_barrier=sha(FORMAL/'GATE1A_V2_FEATURE_SUPPORT_CENSUS.json')==barrier_data['census_sha256'],
        barrier_requires72_and_zero_jobs=barrier_data['feature_units']==72 and barrier_data['clustering_jobs_started_before_this_barrier']==0,
        all18_checkpoint_disk_hashes_unchanged=all(c['unchanged'] for c in checkpoints),
        all18_model_audits_unchanged=len(immutability['checkpoints'])==18 and all(c['bitwise_unchanged'] for c in immutability['checkpoints']),
        old_and_v2_sampling_plan_identical=sha(FORMAL/'SHARED_GEOMETRY_SAMPLING_PLAN.json')==PLAN,
        both_v1_artifact_manifests_unchanged=all(h['status']=='PASS' and h['observed_geometry_files']==0 for h in history),
        formal_manifest_all_bytes_unchanged=manifest['status']=='PASS' and manifest['v2_named_copy_byte_identical'],
        test98pass=tests['status']=='PASS' and tests['passed']==98 and tests['failures']==tests['skipped']==0,
        no_optimizer_no_training=metadata['model_optimizer_steps']==metadata['transport_optimizer_steps']==0 and not metadata['method_registered'] and not metadata['di_dmpa_training_launched'],
        no_hidden_or_test_gt=metadata['hidden_gt_training_usage']==metadata['test_gt_usage']=='none',
        no_downstream=not metadata['Gate1B'] and not metadata['Gate1C'],
        all_registered_counts_conserved=census['registered_count']==census['active_count']+census['null_count'],
        status_census_byte_equivalent=status['support_census']==census,
        rows_in_consolidated_diagnostic_match=sorted(diagnostic['units'],key=lambda r:(r['panel_id'],r['seed'],r['stage_index'],r['class_id'],r['K']))==rows,
    )
    # Lexical filenames sort K numerically for the registered values 1,2,3,5.
    source_diff = command(['git','-C',str(REPO),'diff','--name-only','606a5c53a37d0e4c9605415e8b38a1f177d1604f',CODE,'--',
        'experiments/lcrseg/di_dmpa_gate1','experiments/lcrseg/docs/di_dmpa_jascl/gate1a_results',
        'experiments/lcrseg/docs/di_dmpa_jascl/gate1a_recovery_results'])
    checks['v1_source_and_archives_unchanged'] = not source_diff
    checks['execution_checkout_clean_exact'] = command(['git','-C',str(REPO),'rev-parse','HEAD'])==CODE and not command(['git','-C',str(REPO),'status','--porcelain'])
    remote = command(['git','ls-remote','https://github.com/DLwbm123/SSL_CL_seg.git',
                      'refs/heads/codex/gate1a-v2-null-aware-sphere','refs/heads/main'])
    checks['remote_exact_code_before_report'] = f'{CODE}\trefs/heads/codex/gate1a-v2-null-aware-sphere' in remote
    checks['main_unchanged'] = '46e892960240543c946c570a9378d409b226384b\trefs/heads/main' in remote
    warning_summary = dict(fits=len(all_fits), restarts=len(all_fits)*5,
        nonconverged_selected_fits=sum(not f['converged'] for f in all_fits),
        nonconverged_restarts=len(restart_warnings), fits_with_inactive_slots=sum(f['inactive_slots']>0 for f in all_fits),
        fits_without_directional_support=sum(f['directional_support']=='NONE' for f in all_fits),
        total_inactive_slots=sum(f['inactive_slots'] for f in all_fits),
        policy='All frozen 100-iteration-cap fits/restarts and inactive slots retained; no reroll or post-hoc change.',
        selected_fit_warnings=[f for f in all_fits if not f['converged'] or f['inactive_slots'] or f['directional_support']=='NONE'],
        nonconverged_restart_records=restart_warnings)
    write('GATE1A_V2_FIT_WARNINGS_AUDIT.json',warning_summary)
    audit = dict(status='PASS' if all(checks.values()) else 'FAIL', checked_at_utc=datetime.now(timezone.utc).isoformat(),
        operation='BYTE_CHECKSUM_AND_EXISTING_JSON_SUMMARY_ONLY_NO_FORWARD_NO_FIT',
        diagnostic_code_git_commit=CODE,v2_preregistration_git_commit=PREREG,checks=checks,
        manifest=manifest,historical_attempts=history,checkpoints=checkpoints,
        feature_units=len(feature_files),geometry_jobs=len(geometry_files),raw_array_files=len(raw_array_files),
        registered_count=census['registered_count'],active_count=census['active_count'],null_count=census['null_count'],
        panel_support=census['panels'],null_identity_checks=len(diagnostic['verdict']['null_identity_checks']),
        maximum_Q_identity_absolute_error=max(r['absolute_error'] for r in diagnostic['verdict']['null_identity_checks']),
        runtime_outputs_read_only=True,remote_before_report=remote,source_diff=source_diff,
        gpu_snapshot=command(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv']),
        compute_process_snapshot=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
        source_checkpoint_tensors_loaded=0,model_forward_calls=0,clustering_jobs=0,model_optimizer_steps=0,transport_optimizer_steps=0)
    write('GATE1A_V2_POSTRUN_INTEGRITY_AUDIT.json',audit)
    print(json.dumps(dict(status=audit['status'],checks=checks,warning_counts={k:v for k,v in warning_summary.items() if not isinstance(v,list)},
                         registered_count=census['registered_count'],active_count=census['active_count'],null_count=census['null_count']),indent=2))
    if audit['status']!='PASS':raise SystemExit(2)


if __name__=='__main__':
    main()
