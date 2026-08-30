"""Read-only, stdlib independent audit of the sealed precision pilot."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''): digest.update(chunk)
    return digest.hexdigest()


root = Path(sys.argv[1]).resolve(); spec_path = Path(sys.argv[2])
assert sha(spec_path) == '2ceb37fc571b17373261fe631c8a2e416130912e2e882461b9e42795d495aeca'
read = lambda path: json.loads(Path(path).read_text())
spec = read(spec_path); manifest_path = root/'PILOT_ARTIFACT_MANIFEST.json'; manifest = read(manifest_path)
indexed = [x['path'] for x in manifest['files']]
assert len(indexed) == len(set(indexed)) and set(indexed) == {str(x.relative_to(root)) for x in root.rglob('*') if x.is_file() and x != manifest_path}
for row in manifest['files']:
    path = (root/row['path']).resolve()
    assert path.is_relative_to(root) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
assert sum(x['bytes'] for x in manifest['files']) == manifest['total_bytes'] <= 536870912
assert not list(root.glob('FAILURE_*.json'))
meta = read(root/'RUN_METADATA.json'); status = read(root/'PILOT_STATUS.json')
assert meta['diagnostic_code_commit'] == meta['remote_verified_code_commit'] == '7fdd4312278eb64dbfb471107bb47e6b897c6859'
assert meta['numeric_preregistration_commit'] == '6357317749b0ff904e3acd39023b86430d6263ee'
assert status['metadata'] == meta and status['status'] == 'PASS_NUMERIC_PRECISION_PILOT' and status['scientific_admission'] is None and not status['method_registered']
assert status['gate1_overall_status'] == 'FAIL_TRANSPORT_NOT_SUPPORTED' and status['old_Gate1C_v21_status'] == 'BLOCKED_INCOMPLETE_EVIDENCE'
assert status['model_optimizer_steps'] == status['transport_optimizer_steps'] == 0 and status['hidden_gt_training_usage'] == status['test_gt_usage'] == 'none'
input_audit = read(root/'INPUT_AUDIT.json')
assert input_audit['metadata'] == meta and input_audit['status'] == 'PASS' and len(input_audit['units']) == len(input_audit['checkpoints']) == 9
assert input_audit['test_role_constructions'] == 0 and input_audit['hidden_gt_training_usage'] == input_audit['test_gt_usage'] == 'none'
totals = dict.fromkeys(('native_forwards', 'shadow_forwards', 'native_autograd', 'shadow_autograd'), 0)
comparisons = []; supervised = []; components = []; per_phase = []; per_pair = []; workers = []; guards = []
for phase in spec['pilot']['phases']:
    receipt = read(root/f'PHASE_{phase}.json')
    assert receipt['metadata'] == meta and receipt['status'] == 'PASS' and receipt['phase'] == phase
    for path, digest in receipt['evidence_sha256'].items(): assert sha(root/path) == digest
    per_phase.append(dict(phase=phase, **receipt['totals']))
    for gpu in (0, 1):
        start = read(root/f'WORKER_{phase}_gpu{gpu}_START.json'); end = read(root/f'WORKER_{phase}_gpu{gpu}.json')
        assert start['metadata'] == end['metadata'] == meta and end['status'] == 'PASS'
        assert end['pairs'] == spec['pilot']['assignment'][str(gpu)]
        assert len(end['R1_parity']) == {'draw0': 1, 'noise': 8, 'posterior': 1, 'poe': 0}[phase]*len(end['pairs'])
        assert all(x['exact_R1_parity'] and x['pixels'] == 294912 for x in end['R1_parity'])
        for key, value in end['counts'].items(): totals[key] += value
        seconds = (datetime.fromisoformat(end['completed_at_utc'])-datetime.fromisoformat(start['started_at_utc'])).total_seconds()
        workers.append(dict(phase=phase, gpu=gpu, seconds=seconds, pairs=end['pairs'], counts=end['counts']))
    for pair in spec['pilot']['pairs']:
        name = f'seed{pair["seed"]}_stage{pair["stage_index"]}_pair{pair["pair_index"]:02d}'
        result = read(root/'probes'/phase/name/'result.json')
        assert result['pair'] == pair and result['metadata'] == meta and result['phase'] == phase
        values = [r for r in result['native_precision_comparisons'] if r['block'] == 'global']
        comparisons.extend(values); supervised.append(result['supervised_precision_comparisons']['global']); components.extend(result['class_contribution'])
        per_pair.append(dict(phase=phase, batch_id=pair['batch_id'], global_comparisons=len(values),
            max_relative_l2=max(r['relative_l2'] for r in values if r['relative_l2'] is not None),
            min_cosine=min(r['cosine'] for r in values if r['cosine'] is not None),
            max_class_sum_abs_residual=max((r['component_sum_max_abs_error'] for r in result['class_contribution']), default=None),
            null_ema_count_draw0=result.get('null_ema_count')))
        guard = read(root/'probe_models'/phase/name/'immutability'/f'B0_seed{pair["seed"]}_stage{pair["stage_index"]}.json')
        assert guard['status'] == 'PASS' and guard['bitwise_unchanged'] and guard['extraction_completed'] and guard['before'] == guard['after']
        assert set(guard['before']) == {'student', 'ema_teacher', 'gradient_student'}
        assert guard['checkpoint_sha256_before'] == guard['checkpoint_sha256_after'] == pair['checkpoint_sha256']
        guards.append(guard)
assert totals == status['counts'] == dict(native_forwards=51, shadow_forwards=24, native_autograd=276, shadow_autograd=366)
assert len(comparisons) == 288 and len(supervised) == len(guards) == 12 and len(components) == 630
assert sum(r['alignment_rows'] for r in per_phase) == 2016 and all(r['component_sum_pass'] for r in components)
for row in comparisons+supervised:
    assert ((row['native_l2_norm'] == row['reference_l2_norm'] == 0) or
        (row['relative_l2'] is not None and row['cosine'] is not None and row['relative_l2'] <= .001 and row['cosine'] >= .9999))
print(json.dumps(dict(status='PASS_INDEPENDENT_SEALED_ARTIFACT_AUDIT', root=str(root), audited_at_utc=datetime.now(timezone.utc).isoformat(),
    audit_script_sha256=sha(__file__), code_commit=meta['diagnostic_code_commit'], numeric_preregistration_commit=meta['numeric_preregistration_commit'],
    manifest_sha256=sha(manifest_path), pilot_status_sha256=sha(root/'PILOT_STATUS.json'),
    manifest_file_count=len(indexed), manifest_total_bytes=manifest['total_bytes'],
    total_files_including_manifest=len(indexed)+1, total_bytes_including_manifest=manifest['total_bytes']+manifest_path.stat().st_size,
    counts=totals, coverage=status['coverage'], per_phase=per_phase, per_pair=per_pair, workers=workers,
    maximum_objective_relative_l2=max(r['relative_l2'] for r in comparisons if r['relative_l2'] is not None),
    minimum_objective_cosine=min(r['cosine'] for r in comparisons if r['cosine'] is not None),
    maximum_supervised_relative_l2=max(r['relative_l2'] for r in supervised if r['relative_l2'] is not None),
    minimum_supervised_cosine=min(r['cosine'] for r in supervised if r['cosine'] is not None),
    maximum_component_sum_abs_residual=max(r['component_sum_max_abs_error'] for r in components),
    both_zero_global_comparisons=sum(r['native_l2_norm'] == r['reference_l2_norm'] == 0 for r in comparisons),
    model_guard_count=len(guards), model_optimizer_steps=0, transport_optimizer_steps=0, scientific_admission=None), indent=2))
