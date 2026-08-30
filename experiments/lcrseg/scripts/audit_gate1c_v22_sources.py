"""Read-only local source/inventory preflight; never certifies remote caches."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
FORMAL = '44a25254697fa535d2b48b64e27ecb226436f7d0'
PILOT = '7fdd4312278eb64dbfb471107bb47e6b897c6859'
MANIFEST_SHA = '0d652551711e0a3ceff6ac8bdb0001355f4ec6083882460d740784ee837420d9'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args])


def definition(source, name):
    nodes = [n for n in ast.parse(source).body if
             isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(nodes) != 1:
        raise ValueError('missing/duplicate source definition: '+name)
    return ast.dump(nodes[0], include_attributes=False)


def identical_definitions(before, after, names):
    result = {}
    for name in names:
        left, right = definition(before, name), definition(after, name)
        if left != right:
            raise ValueError('native validation definition changed: '+name)
        result[name] = digest(right.encode())
    return result


def audit():
    protected = ['experiments/lcrseg/'+p for p in (
        'di_dmpa_jascl', 'di_dmpa_gate1', 'di_dmpa_gate1_v2', 'di_dmpa_gate1b_v2',
        'di_dmpa_gate1c_v2/binding.py', 'di_dmpa_gate1c_v2/reliability.py', 'di_dmpa_gate1c_v2/metrics.py')]
    if git('diff', FORMAL, '--', *protected).strip():
        raise ValueError('native data/model/binding/scoring/evaluator source changed')
    code = 'experiments/lcrseg/di_dmpa_gate1c_v2/'
    path = code+'execution.py'
    functions = identical_definitions(git('show', FORMAL+':'+path).decode(), (REPO/path).read_text(),
        ('CACHE_FIELDS', 'unit_name', 'visible_labels', 'validate_scores', 'validation_unit', 'evaluate_unit'))
    isolation = identical_definitions(git('show', FORMAL+':'+code+'gradients.py').decode(),
        (REPO/code/'gradients.py').read_text(), ('isolation',))
    numeric = {}
    for name in ('execution.py', 'gradients.py', 'precision.py'):
        current = (REPO/code/name).read_bytes()
        if current != git('show', PILOT+':'+code+name):
            raise ValueError('tested numeric engine changed: '+name)
        numeric[name] = digest(current)
    docs = ROOT/'docs/di_dmpa_jascl'
    manifest_path = docs/'gate1c_v21_failure/44a2525_attempt1/GATE1C_V2_ARTIFACT_MANIFEST.json'
    raw = manifest_path.read_bytes()
    if digest(raw) != MANIFEST_SHA:
        raise ValueError('published historical artifact index changed')
    manifest = json.loads(raw)
    prereg_path = docs/'DI_DMPA_GATE1C_V2_PREREGISTRATION.json'
    raw_prereg = prereg_path.read_bytes()
    if digest(raw_prereg) != '8b8dc8c56b60e27e3e1521053cd9307bf65d017ec9343476857b9508721c2f57':
        raise ValueError('frozen validation plan changed')
    p = json.loads(raw_prereg)
    expected = {f'validation_cache/seed{u["seed"]}_stage{u["stage_index"]}/{c["case_id"]}.npz'
                for u in p['validation']['plans'] for c in u['cases']}
    arrays = [r for r in manifest['artifacts'] if r['path'].startswith('validation_cache/')]
    if len(arrays) != 495 or len(expected) != 495 or {r['path'] for r in arrays} != expected:
        raise ValueError('published validation case inventory differs from frozen plan')
    return dict(status='PASS_LOCAL_SOURCE_AND_PUBLISHED_INVENTORY_ONLY',
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version, ast_hashes_interpreter_specific=True,
        candidate_commit=git('rev-parse', 'HEAD').decode().strip(),
        worktree_clean=not bool(git('status', '--porcelain').strip()),
        auditor_sha256=digest(Path(__file__).read_bytes()), old_formal_code_commit=FORMAL,
        pilot_numeric_code_commit=PILOT, protected_paths=protected,
        protected_source_diff_empty=True, native_validation_ast_sha256=functions,
        isolation_ast_sha256=isolation, unchanged_numeric_files_sha256=numeric,
        historical_manifest_sha256=MANIFEST_SHA,
        published_inventory_case_count=len(arrays), published_inventory_cache_bytes=sum(r['bytes'] for r in arrays),
        published_inventory_unit_count=len(p['validation']['plans']),
        published_inventory_pixels=495*384*384, remote_cache_revalidated=False,
        checkpoint_tensors_loaded=False, cache_arrays_loaded=False, labels_loaded=False,
        new_model_forwards=0, optimizer_steps=0, cache_reuse_approved=False,
        next_required='LIVE_REMOTE_FILES_METADATA_INPUTS_AND_NUMERICAL_CACHE_AUDIT')


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
