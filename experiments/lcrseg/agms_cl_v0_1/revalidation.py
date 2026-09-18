"""Narrow R1 composition of immutable native CPU evidence and zero-update repair checks."""
import hashlib
from .protocol import DOC, read, digest, code_manifest, execution_plan

BASE_COMMIT='ac01ab6de6250e11877fe9000a30665c2245292d'
BASE_TREE='e1949dab81db5643378805cd7a30419dc44d1ca6c269aa7301dc8770b3656e0a'
ALLOWED={f'experiments/lcrseg/agms_cl_v0_1/{n}.py' for n in
         ('p0','reporting','authority','revalidation','review_r1_regression')}
CHECKS=('old_nested_failure_and_cleanup','prefixes_exit_before_P0','sealed_idempotency',
        'prefix_failure_no_L_or_P0','partial_P0_refused','payload_identity_hash_rechecked',
        'coverage_partitions_and_export','readonly_state_and_RNG','missing_and_old_authority_refused',
        'composite_binding_rejects_drift','zero_optimizer_and_real_IO')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline():
    folder=DOC/'REVIEW_R1'
    for name,expected in (
        ('BASE_CODE_MANIFEST.json','8f355df6411cb4bd8a26cec7355fdb6973f37d3efb8db742f072a6aed10f73a7'),
        ('BASE_CPU_FILES.json','51cf3cb4711b1d07b20c9ff98826e06e8066ffa155581790a4ee3c64b0096ac0')):
        if file_hash(folder/name)!=expected:raise PermissionError('changed R1 baseline evidence')
    manifest=read(folder/'BASE_CODE_MANIFEST.json');cpu_files=read(folder/'BASE_CPU_FILES.json')
    if manifest['code_tree_sha256']!=BASE_TREE or digest(manifest['files'])!=BASE_TREE:
        raise PermissionError('wrong baseline code manifest')
    actual={str(p.relative_to(DOC)):file_hash(p) for p in sorted((DOC/'CPU').rglob('*')) if p.is_file()}
    if actual!=cpu_files:raise PermissionError('immutable native CPU evidence changed')
    cpu=read(DOC/'CPU/TEST_REPORT.json')
    if (cpu['status']!='PASS' or not cpu['baseline_equivalence'] or cpu['code_tree_sha256']!=BASE_TREE
            or file_hash(DOC/'CPU/TEST_REPORT.json')!='7de5b20e6029a156a8536105bebc098a6c93884781e43a55d416e5946b5f00ce'):
        raise PermissionError('wrong native CPU source')
    return manifest,cpu,cpu_files


def binding(plan,current=None):
    current=code_manifest() if current is None else current
    base,cpu,cpu_files=baseline()
    if digest(current['files'])!=current['code_tree_sha256']:raise PermissionError('current manifest mismatch')
    changed={p:{'baseline':base['files'].get(p),'current':current['files'].get(p)}
             for p in sorted(set(base['files'])|set(current['files'])) if base['files'].get(p)!=current['files'].get(p)}
    if set(changed)!=ALLOWED or any(v['current'] is None for v in changed.values()):
        raise PermissionError('R1 scope exceeded: unknown or missing local changes')
    if cpu['plan_sha256']!=plan['plan_sha256'] or cpu['execution_sha256']!=digest(execution_plan()):
        raise PermissionError('R1 science or execution drift')
    return dict(kind='SCOPED_ZERO_UPDATE_REVALIDATION',baseline_commit=BASE_COMMIT,baseline_code_tree_sha256=BASE_TREE,
                baseline_cpu_files=cpu_files,code_tree_sha256=current['code_tree_sha256'],
                plan_sha256=plan['plan_sha256'],execution_sha256=digest(execution_plan()),changed_files=changed,
                protected_files={p:h for p,h in base['files'].items() if p not in changed},
                native_suite_rerun=False,new_optimizer_calls=0)


def validate_composite(plan,report,regression,current=None):
    expected=binding(plan,current)
    if any(report.get(k)!=v for k,v in expected.items()):raise PermissionError('composite binding mismatch')
    if (report.get('status')!='PASS' or report.get('regression_sha256')!=digest(regression)
            or regression.get('status')!='PASS' or regression.get('code_tree_sha256')!=expected['code_tree_sha256']
            or regression.get('checks')!={k:'PASS' for k in CHECKS}
            or regression.get('optimizer_calls')!=0 or regression.get('optimizer_attempts')!=0
            or regression.get('real_prefix_reads')!=0 or regression.get('patient_reads')!=0
            or regression.get('native_model_runs')!=0 or regression.get('CUDA_runs')!=0
            or regression.get('attempt') not in (1,2)):
        raise PermissionError('current zero-update regression required')
    return report


def validate_cpu(plan,tree):
    # Current-tree checking remains mandatory; old native PASS alone cannot qualify a repair.
    current=code_manifest()
    if tree!=current['code_tree_sha256']:raise PermissionError('current code mismatch')
    folder=DOC/'REVIEW_R1_REGRESSION'
    return validate_composite(plan,read(folder/'COMPOSITE_REPORT.json'),read(folder/'REPORT.json'),current)
