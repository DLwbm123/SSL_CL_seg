"""Independent canonical semantics; only pinned public aggregate inputs."""
import copy
import hashlib
from pathlib import Path
from ..f5_confirmation_v1.protocol import read, write, digest, MANIFEST, SPLIT
from .anchors import ANCHORS
ROOT = Path(__file__).resolve().parents[3]
BASE_DOC = ROOT / 'experiments/lcrseg/docs/agms_cl_v0_1'
DOC = ROOT / 'experiments/lcrseg/docs/agms_ds_half_v1'
STUDY = 'AGMS_DS_HALF_V1'
BASE = 'd557bd245eb204ec3ca377a5d868a03dc31436a5'


def anchored(name):
    path = BASE_DOC / 'inputs' / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != ANCHORS[name]:
        raise ValueError('changed pinned input: ' + name)
    return read(path)


PROPOSAL = anchored('05_PROPOSED_PLAN.json')
ARMS = copy.deepcopy(PROPOSAL['arms'])
CAPS = copy.deepcopy(PROPOSAL['budgets'])
# Independent development study; prior CPU cost is immutable and not reusable capacity.
CAPS = dict(new_CPU_optimizer_cap=24, new_CPU_attempt_cap=1, CUDA_optimizer_cap=11,
            smoke_optimizer_cap=8, formal_optimizer_cap=5300, real_optimizer_cap=5308,
            source_updates=0, first_target_updates=0, diagnostic_vjps=32)
CPU_AMENDMENT = dict(id='NEW_STUDY_USER_DELEGATION', prior_AGMS_CPU=80, prior_earlier_CPU=134,
                    production_authority='fresh external review plus current user delegation; old approval refused')
METHOD = copy.deepcopy(PROPOSAL['options'])
METHOD.pop('fully_bound_b2_options')
METHOD['lambda_DS'] = .125
GEOMETRY = dict(labels={'background': 0, 'rim': 1, 'cup': 2, 'ignore': 255},
               parent=[1, 2], main='unchanged valid3x3 then bilinear align_corners=True',
               aux='dec3/dec2 native grid to output: bilinear align_corners=True; no valid crop',
               LCTX='original complementary mask; collect anchor after auxiliary log_softmax')


def _base_plan():
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


CONTROL_PUBLIC_SHA256 = 'a9256908ed1c726151d648f0ab77345e492610d536f160e5f85241e3168c814d'
CONTROL_PUBLIC_COMMIT = 'ef6888dc81519ce9e9ca13ea497bbf6397747d70'

def controls():
    path=BASE_DOC/'results_20260919/PUBLIC_RESULTS.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CONTROL_PUBLIC_SHA256:
        raise ValueError('prior complete result changed')
    value=read(path)
    if (value['status']!='COMPLETE_AGMS_P1_AWAITING_SCIENTIFIC_REVIEW'
        or value['execution_commit']!='89dc7657c4f0b7c6f51de339d0f6bc2a621da55f'
        or value['costs']['physical']!={'formal':26500,'smoke':24,'synthetic_cuda':36}
        or len(value['integrity'])!=10 or any(p['status']!='VERIFIED' for p in value['integrity'].values())):
        raise ValueError('prior full matrix acceptance required')
    return value


def canonical_plan():
    result=_base_plan();result.pop('plan_sha256')
    result.update(study_id=STUDY,status='PREPARED_PENDING_FRESH_REVIEW',primary_candidate='A5_DS_HALF',
                  controls_public_commit=CONTROL_PUBLIC_COMMIT,controls_public_sha256=CONTROL_PUBLIC_SHA256,
                  evidence_use='development tuning; not independent confirmation')
    result['nodes']=[n for n in result['nodes'] if n['arm']=='A5']
    for n in result['nodes']:
        n.update(id=n['id'].replace('AGMS__','DS_HALF__'),study=STUDY)
    prior=controls()
    result['control_imports']=[dict(node_id=x['node_id'],order=x['identity']['order'],arm='A5_CONTROL',
         row_sha256=digest(x),proof_sha256=digest(prior['integrity'][x['node_id']]))
         for x in prior['rows'] if x['arm']=='A5']
    result['arms']={k:{**v,'formal':k=='A5'} for k,v in ARMS.items()}
    result['reports']=dict(final_rows=6,domain_rows=18,paired_rows=9,diagnostic_points=8)
    result['future']={}
    result.pop('gates')
    result['advancement_gates']=dict(mean_Final_vs_A5_at_least=.003,mean_Old_vs_A5_at_least=.003,
                                    each_Incoming_vs_A5_at_least=-.005,each_Final_vs_A5_at_least=-.002)
    result['success_interpretation']='DS-pressure hypothesis supported only if all new gates pass; separately report absolute A0 deficit'
    result['plan_sha256']=digest(result)
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
    return dict(study_id=STUDY,science_sha256=canonical_plan()['plan_sha256'],caps=CAPS,
       CPU=dict(planned_calls=21,cap=24,attempt_cap=1,cases=dict(A5_native=2,continuation=5,baseline=4,failures=2,overfit=8)),
       CUDA=dict(cases=[dict(arm='A5',kind='continuation',calls=5),dict(kind='baseline',calls=4),dict(kind='failures',calls=2)],planned_calls=11,cap=11),
       smoke=dict(cases=[dict(arm='A5',order=o,calls=4,L_only=True,discarded=True) for o in (1,2)],planned_calls=8,cap=8),
       diagnostics=PROPOSAL['diagnostics'],P0=dict(mode='import_prior_26_L_screen',images=0,updates=0),
       future_status='PENDING',automatic_retry=False)


def freeze():
    p = canonical_plan()
    for name, value in [('PLAN', p), ('PREFIX_BINDINGS', p['prefixes']), ('IMPORT_BINDINGS', {'A0':p['imports'],'A5_CONTROL':p['control_imports']}),
                        ('ENVIRONMENT_BINDING', p['environment']), ('FROZEN_OPTIONS', {'B2': p['options'], 'AGMS': METHOD}),
                        ('GEOMETRY_CONTRACT', GEOMETRY), ('RISK_STATE_CONTRACT', METHOD['risk']),
                        ('EXECUTION_PLAN', execution_plan()), ('PREPARATION_AUTHORITY', CPU_AMENDMENT), ('CODE_MANIFEST', code_manifest())]:
        write(DOC / (name + '.json'), value)
    return p


def smoke_nodes():
    nodes=copy.deepcopy(canonical_plan()['nodes'])
    for n in nodes:n.update(id='SMOKE__'+n['id'],phase='smoke')
    return nodes
