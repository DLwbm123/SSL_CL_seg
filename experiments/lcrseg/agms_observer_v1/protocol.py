"""Independent observer study; immutable public inputs, finite proposal only."""
import copy
import hashlib
from ..agms_cl_v0_1 import protocol as prior
from ..f5_confirmation_v1.protocol import read, write, digest
ROOT = prior.ROOT
DOC = ROOT / 'experiments/lcrseg/docs/agms_observer_v1'
STUDY = 'AGMS_OBSERVER_V1'
anchored = prior.anchored
controls = prior.controls
CAPS = dict(new_CPU_optimizer_cap=32,new_CPU_attempt_cap=2,CPU_attempt_optimizer_cap=16,
            CUDA_optimizer_cap=10,smoke_optimizer_cap=8,formal_optimizer_cap=5300,
            real_optimizer_cap=5308,source_updates=0,first_target_updates=0,diagnostic_vjps=32)
CPU_AMENDMENT = dict(prior_CPU=235,prior_breakdown={'earlier':134,'AGMS':80,'HALF':21},
                    authority='attached preparation prompt; generated CPU only; fresh external review AND separate user launch for production')
ARMS = {'OBS0':dict(prior.ARMS['A5']), 'SHADOW':{**prior.ARMS['A5'],'H':False}}
METHOD = {**copy.deepcopy(prior.METHOD),'lambda_DS':.25,'aux_features_detached':True,
          'observer_contract':'same_state_counts_v1_L_strata_no_feedback'}
GEOMETRY = copy.deepcopy(prior.GEOMETRY)
HALF_SHA = 'c2fe1daea849b95b108256c3895ac49f1bb73463fa8aa4f576bff2c6f6cfdde2'
DIAGNOSTIC = dict(points={'Drishti_GS':[525,1050,1575,2100],'RIM_ONE_r3':[800,1600,2400,3200]},
    all_active_U='same teacher-U state; main/equal/risk selection; no labels; additive counts and ratios',
    L='current training L only; parent Brier and conditional rim/cup Brier/error; class-balanced counts; GT boundary strata diagnostic only',
    ranking='top quartile vs entropy: identical pixel coverage, boundary stratum, class; deterministic index tie-break',
    gradient='4 VJP at 8 points; DS upstream None becomes zero; never reconnect',
    KL='numeric decomposition log only, w=1, production masked_kl unchanged',feedback=False,extra_backbone_forwards=0)


def half_results():
    path=prior.DOC/'results_20260919/PUBLIC_RESULTS.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=HALF_SHA:raise ValueError('HALF public input changed')
    v=read(path)
    if v['status']!='COMPLETE_DS_HALF_AWAITING_SCIENTIFIC_REVIEW' or len(v['integrity'])!=2 or any(x['status']!='VERIFIED' for x in v['integrity'].values()):
        raise ValueError('HALF acceptance missing')
    return v


def canonical_plan():
    p=copy.deepcopy(prior.canonical_plan());p.pop('plan_sha256')
    p.update(study_id=STUDY,status='PREPARED_PENDING_FRESH_REVIEW',primary_candidate='OBS0',
             budgets=CAPS,preparation_amendment=CPU_AMENDMENT,method=METHOD,geometry=GEOMETRY,
             arms={k:{**v,'formal':k=='OBS0'} for k,v in ARMS.items()},diagnostic_contract=DIAGNOSTIC,
             reports=dict(final_rows=10,domain_rows=30,paired_rows=12,diagnostic_points=8),future={},
             advancement_gates={'each_order_Final_vs_B2':-.001,'each_old_domain_macro_vs_B2':-.002,'each_order_Incoming_vs_B2':-.005},
             success_interpretation='descriptive recovery only, not statistical noninferiority or new mechanism success',
             half_public_commit='6e65a9b987f8818f7b10fa11a2cc205d11283779',half_public_sha256=HALF_SHA)
    for n in p['nodes']:
        n.update(id=n['id'].replace('DS_HALF__A5','OBSERVER__OBS0'),arm='OBS0',study=STUDY,
                 method_sha256=digest({'arm':ARMS['OBS0'],'method':METHOD,'geometry':GEOMETRY}))
    p['control_imports']=[]
    for public,aliases,commit in [(controls(),{'A1':'A1','A5':'A5_CONTROL'},prior.CONTROL_PUBLIC_COMMIT),
                                  (half_results(),{'A5':'HALF_CONTROL'},p['half_public_commit'])]:
        for row in public['rows']:
            if row['arm'] not in aliases:continue
            proof=public['integrity'][row['node_id']]
            p['control_imports'].append(dict(node_id=row['node_id'],order=row['identity']['order'],arm=aliases[row['arm']],
                  source_commit=commit,row_sha256=digest(row),proof_sha256=digest(proof)))
    assert len(p['nodes'])==2 and sum(n['updates'] for n in p['nodes'])==5300
    assert len(p['imports'])+len(p['control_imports'])==8
    p['plan_sha256']=digest(p);return p


def validate_plan(plan):
    if plan!=canonical_plan():raise ValueError('noncanonical observer plan, including rehashed changes')
    return plan


def validate_prefix(node,record,receipt):
    p=canonical_plan()
    if node not in p['nodes']+smoke_nodes() or record not in p['prefixes'] or node['prefix_node']!=record['node_id']:
        raise PermissionError('prefix outside observer scope')
    if node['prefix_binding_sha256']!=record['binding_sha256'] or receipt['identity']!=record['identity'] or digest(receipt)!=record['receipt_sha256'] or receipt['status']!='SEALED' or receipt['student_hash']!=record['student_sha256']:
        raise ValueError('PREFIX_REUSE_BLOCKED')

code_manifest = prior.code_manifest


def execution_plan():
    p=canonical_plan()
    cpu=dict(native=2,continuation=5,shadow=4,failure=1)
    cuda=[dict(arm='OBS0',kind='continuation',calls=5),dict(kind='baseline',calls=4),dict(kind='failures',calls=1)]
    smoke=[dict(arm='OBS0',order=o,calls=4,L_only=True,discarded=True) for o in (1,2)]
    assert sum(cpu.values())==12<=16 and sum(x['calls'] for x in cuda)==10 and sum(x['calls'] for x in smoke)==8
    return dict(study_id=STUDY,science_sha256=p['plan_sha256'],caps=CAPS,
      CPU=dict(planned_calls=12,cap=32,attempt_cap=2,per_attempt_cap=16,cases=cpu),
      CUDA=dict(cases=cuda,planned_calls=10,cap=10),smoke=dict(cases=smoke,planned_calls=8,cap=8),
      diagnostics=DIAGNOSTIC['points'],P0=dict(mode='historical only',images=0,updates=0),
      future_status='PENDING_FRESH_REVIEW_AND_USER_LAUNCH',automatic_retry=False)


def smoke_nodes():
    nodes=copy.deepcopy(canonical_plan()['nodes'])
    for n in nodes:n.update(id='SMOKE__'+n['id'],phase='smoke')
    return nodes


def freeze():
    p=canonical_plan()
    values={'PLAN':p,'METHOD':METHOD,'PREFIX_BINDINGS':p['prefixes'],
      'IMPORT_BINDINGS':{'A0':p['imports'],'controls':p['control_imports']},'ENVIRONMENT_BINDING':p['environment'],
      'FROZEN_OPTIONS':{'B2':p['options'],'OBSERVER':METHOD},'GEOMETRY_CONTRACT':GEOMETRY,
      'RISK_STATE_CONTRACT':METHOD['risk'],'DIAGNOSTIC_CONTRACT':DIAGNOSTIC,
      'RESUME_CONTRACT':dict(study=STUDY,detach=True,full_state=['main','aux','EMA','risk','optimizers','scheduler','RNG','prototypes','diagnostics','support_summary','physical_calls'],reject=['old_study','wrong_prefix','non_detach','lost_physical_tail'],deployment='merged main only; discard aux/risk'),
      'EXECUTION_PLAN':execution_plan(),'PREPARATION_AUTHORITY':CPU_AMENDMENT,'CODE_MANIFEST':code_manifest()}
    for k,v in values.items():write(DOC/(k+'.json'),v)
    return p
