"""Independent canonical semantics; only pinned public aggregate inputs."""
import copy
import hashlib
from pathlib import Path
from ..f5_confirmation_v1.protocol import read, write, digest, MANIFEST, SPLIT
from .anchors import ANCHORS
ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / 'experiments/lcrseg/docs/agms_cl_v0_1'
STUDY = 'AGMS_CL_V0_1'
BASE = 'd557bd245eb204ec3ca377a5d868a03dc31436a5'


def anchored(name):
    path = DOC / 'inputs' / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != ANCHORS[name]:
        raise ValueError('changed pinned input: ' + name)
    return read(path)


PROPOSAL = anchored('05_PROPOSED_PLAN.json')
ARMS = copy.deepcopy(PROPOSAL['arms'])
CAPS = copy.deepcopy(PROPOSAL['budgets'])
# Explicit user amendment after the original three attempts, CPU preparation only.
CPU_AMENDMENT = dict(id='CPU_REPAIR_1', user_instruction='我都授权， 你尽快解决',
    context='one additional 28-call CPU generated suite after 52 calls / 3 attempts',
    previous_attempt_cap=3, attempt_cap=4, additional_planned_calls=28,
    cumulative_optimizer_cap=96, production_authorized=False)
CAPS['new_CPU_attempt_cap'] = CPU_AMENDMENT['attempt_cap']
METHOD = copy.deepcopy(PROPOSAL['options'])
METHOD.pop('fully_bound_b2_options')
GEOMETRY = dict(labels={'background': 0, 'rim': 1, 'cup': 2, 'ignore': 255},
               parent=[1, 2], main='unchanged valid3x3 then bilinear align_corners=True',
               aux='dec3/dec2 native grid to output: bilinear align_corners=True; no valid crop',
               LCTX='original complementary mask; collect anchor after auxiliary log_softmax')


def canonical_plan():
    proposal = anchored('05_PROPOSED_PLAN.json')
    result = copy.deepcopy(proposal)
    result['budgets'] = copy.deepcopy(CAPS)
    result['preparation_amendment'] = copy.deepcopy(CPU_AMENDMENT)
    result['options'] = anchored('BASE_FROZEN_OPTIONS.json')
    result['method'] = copy.deepcopy(METHOD)
    result['geometry'] = copy.deepcopy(GEOMETRY)
    result['environment'] = anchored('BASE_ENVIRONMENT_BINDING.json')
    result['manifest_sha256'], result['split_sha256'] = MANIFEST, SPLIT
    public = anchored('B2_PUBLIC_RESULTS.json')
    result['prefixes'] = [p for p in anchored('BASE_PREFIX_BINDINGS.json') if p['identity']['seed'] == 163]
    for p in result['prefixes']:
        row = next(r for r in public['rows'] if r['node_id'] == p['node_id'])
        proof = public['integrity'][p['node_id']]
        if proof['receipt_sha256'] != p['receipt_sha256'] or row['identity'] != p['identity'] or proof['status'] != 'VERIFIED' or proof['student_hash'] != p['student_sha256']:
            raise ValueError('BASELINE_REUSE_BLOCKED: prefix metadata')
    for n in result['nodes']:
        p = next(p for p in result['prefixes'] if p['node_id'] == n['prefix_node'])
        n['prefix_binding_sha256'] = p['binding_sha256']
        n['options_sha256'] = digest(result['options'][n['domain']])
        n['method_sha256'] = digest({'arm': ARMS[n['arm']], 'method': METHOD, 'geometry': GEOMETRY})
    for item in result['imports']:
        row = next(r for r in public['rows'] if r['node_id'] == item['node_id'])
        proof = public['integrity'][item['node_id']]
        if proof['status'] != 'VERIFIED' or row['identity']['seed'] != 163 or row['identity']['family'] != 'B2_PARENT_PAS_KL':
            raise ValueError('BASELINE_REUSE_BLOCKED: import')
        item.pop('receipt_hash_status')
        item.update(receipt_sha256=proof['receipt_sha256'], row_sha256=digest(row), proof_sha256=digest(proof), identity=row['identity'],
                    student_sha256=proof['student_hash'], diagnostics='NOT_MEASURED_HISTORICAL')
    result['plan_sha256'] = digest(result)
    return result


def validate_plan(plan):
    if plan != canonical_plan():
        raise ValueError('noncanonical AGMS plan, including rehashed changes')
    return plan


def validate_prefix(node, record, receipt):
    p = canonical_plan()
    if node not in (p['nodes'] + smoke_nodes()) or record not in p['prefixes'] or node['prefix_node'] != record['node_id']:
        raise PermissionError('prefix outside canonical scope')
    if (node['prefix_binding_sha256'] != record['binding_sha256'] or receipt['identity'] != record['identity']
            or digest(receipt) != record['receipt_sha256'] or receipt['status'] != 'SEALED'
            or receipt['student_hash'] != record['student_sha256']):
        raise ValueError('PREFIX_REUSE_BLOCKED')


def code_manifest():
    paths = sorted((ROOT / 'experiments').rglob('*.py'))
    files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if '__pycache__' not in p.parts}
    return dict(files=files, code_tree_sha256=digest(files))


def execution_plan():
    cpu = dict(math_metadata=0, six_arms=12, continuation=10, baseline=4, failures=2, integrity_reports=0)
    return dict(study_id=STUDY, science_sha256=canonical_plan()['plan_sha256'], caps=CAPS,
                CPU=dict(cases=cpu, planned_calls=28, attempt_cap=CAPS['new_CPU_attempt_cap'], per_attempt_cap=32, cap=96),
                CUDA=dict(cases=[dict(arm=a, kind='continuation', calls=5) for a in ARMS] +
                     [dict(kind='baseline', calls=4), dict(kind='failures', calls=2)], planned_calls=36, cap=36),
                smoke=dict(cases=[dict(arm=a, calls=4, L_only=True, discarded=True) for a in ARMS], planned_calls=24, cap=24),
                diagnostics=PROPOSAL['diagnostics'], P0=dict(image_cap=32, per_order=16, updates=0),
                future_status='PENDING', automatic_retry=False)


def freeze():
    p = canonical_plan()
    for name, value in [('PLAN', p), ('PREFIX_BINDINGS', p['prefixes']), ('IMPORT_BINDINGS', p['imports']),
                        ('ENVIRONMENT_BINDING', p['environment']), ('FROZEN_OPTIONS', {'B2': p['options'], 'AGMS': METHOD}),
                        ('GEOMETRY_CONTRACT', GEOMETRY), ('RISK_STATE_CONTRACT', METHOD['risk']),
                        ('EXECUTION_PLAN', execution_plan()), ('CPU_REPAIR_AUTHORIZATION', CPU_AMENDMENT), ('CODE_MANIFEST', code_manifest())]:
        write(DOC / (name + '.json'), value)
    return p


def smoke_nodes():
    plan = canonical_plan()
    nodes = [copy.deepcopy(next(n for n in plan['nodes'] if n['arm']==a and n['order']==1)) for a in ARMS if a!='A0']
    zero = copy.deepcopy(nodes[0]);zero.update(arm='A0',id='SMOKE__A0__S163__O1__STAGE2',phase='smoke')
    zero['method_sha256']=digest({'arm':ARMS['A0'],'method':METHOD,'geometry':GEOMETRY})
    return [zero]+nodes
