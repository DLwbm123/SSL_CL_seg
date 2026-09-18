"""Pure canonical dose semantics anchored to immutable published metadata."""
import copy,hashlib
from pathlib import Path
from ..native_key_alignment_v0_1 import protocol as old
from ..f5_confirmation_v1.protocol import read,write,digest
ROOT=old.ROOT
DOC=ROOT/'experiments/lcrseg/docs/nka_dose_v0_1'
STUDY='NKA_DOSE_V0_1'
BASE='145b3c1b3ab301031ea379c9d67452e57b28ae1d'
RESULTS='0da596f4050a3701f41ff74248ca9d4ba21222e2'
ARMS={a:{k:v for k,v in old.ARMS[a].items() if k!='lambda_align'} for a in ('C2','C3')}
DOSES={'L0p5':.5,'L2p0':2.}
CAPS=dict(formal_scientific=42400,formal_physical=42400,new_source=0,new_first_target=0,
          new_cpu_generated=96,new_cpu_invocations=3,new_cpu_per_invocation_max=32,
          synthetic_cuda_planned=40,synthetic_cuda_cap=40,real_smoke=32,real_total=42432)
ANCHORS={'PUBLIC_RESULTS.json':'c412b2856cb0dc082d62c2a361daee9bee24cb3009fd8f560915696d5a17b0d2',
 'PUBLIC_COMPLETION_PROOFS.json':'e09197c7c5c9671904693271d17ea0261e9e3ce26dba94c54e865a4b285f64c6',
 'RUNTIME_ENVIRONMENT.json':'96e0173414f9324dfea6d50f5053dfa640ef7a50c2a5b49c1aa4f42121e3a823'}
ADAM_SOURCE='2a28ca25790a8b9b3921ae5c0b2d9a05926aa8096390be5933608c113e628f76'
METER_VERSION='adam_local_fp32_v1'
TOLERANCE=dict(parameter_atol=2e-7,increment_atol=1e-9,increment_rtol=2e-5,ratio_resolution_ulps=4,ratio_floor=1e-12)


def anchored(name):
    p=DOC/'inputs'/name
    if hashlib.sha256(p.read_bytes()).hexdigest()!=ANCHORS[name]:raise ValueError('BASELINE_REUSE_BLOCKED: changed '+name)
    return read(p)


def canonical_plan():
    prior=old.canonical_plan();published=anchored('PUBLIC_RESULTS.json');proofs=anchored('PUBLIC_COMPLETION_PROOFS.json')
    env=anchored('RUNTIME_ENVIRONMENT.json')
    if env['sha256']!=digest(env['fingerprint']) or published['execution_commit']!=BASE or published['status']!='COMPLETE_D1_AWAITING_SCIENTIFIC_REVIEW':raise ValueError('BASELINE_REUSE_BLOCKED')
    # The full baseline transitive code is retained; new package is additive only.
    baseline=read(old.DOC/'CODE_MANIFEST.json')
    if baseline['code_tree_sha256']!='59f51bc04182892116b728078e471b8c89ee59373b1831cd8eaa1a5a2ff0df07' or digest(baseline['files'])!=baseline['code_tree_sha256']:raise ValueError('baseline code manifest changed')
    for path,sha in baseline['files'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:raise ValueError('BASELINE_REUSE_BLOCKED: base code changed')
    imports=[];nodes=[]
    for row in published['rows']:
        i=row['identity'];arm=i['arm']
        if arm not in ('C0','C2','C3'):continue
        n=next(n for n in prior['nodes'] if n['id']==row['node_id'])
        proof=next(p for p in proofs['targets'] if p['node_id']==row['node_id'])
        if (i['prefix_binding_sha256']!=n['prefix_binding_sha256'] or row['resolved_options']!=prior['options'][n['domain']]
            or proof['status']!='VERIFIED' or proof['student_hash']!=row['student_hash'] or proof['receipt_sha256']!=digest(row)
            or i['execution_commit']!=BASE):raise ValueError('BASELINE_REUSE_BLOCKED: historical row/proof')
        item=dict(node_id=row['node_id'],origin='historical_import',source_commit=RESULTS,source_identity=i,
                  lambda_align=0. if arm=='C0' else .05,row_sha256=digest(row),proof_sha256=digest(proof),adam_diagnostics='NOT_MEASURED_HISTORICAL')
        imports.append(item)
    if len(imports)!=12:raise ValueError('historical coverage')
    for seed in (163,164):
        for order in (1,2):
            prefix=next(p for p in prior['prefixes'] if p['identity']['seed']==seed and p['identity']['order']==order)
            domain='Drishti_GS' if order==1 else 'RIM_ONE_r3'
            for arm in ARMS:
                for dose,weight in DOSES.items():
                    nodes.append(dict(id=f'DOSE__{arm}__{dose}__S{seed}__O{order}__STAGE2',phase='DOSE',arm=arm,dose_id=dose,lambda_align=weight,
                        seed=seed,order=order,stage=2,domain=domain,updates=prior['options'][domain]['total_steps'],
                        prefix_node=prefix['node_id'],prefix_binding_sha256=prefix['binding_sha256'],options_sha256=digest(prior['options'][domain]),executable=False))
    result=dict(study_id=STUDY,state='STOP_AWAITING_EXTERNAL_CODE_REVIEW',base_code=BASE,base_results=RESULTS,prefix_results=old.RESULTS,
        nodes=nodes,prefixes=prior['prefixes'],imports=imports,environment=env,options=prior['options'],arms=ARMS,doses=DOSES,
        parent=prior['parent'],auxiliary=prior['auxiliary'],auxiliary_namespace='NATIVE_KEY_ALIGNMENT_V0_1',
        manifest_sha256=prior['manifest_sha256'],split_sha256=prior['split_sha256'],caps=CAPS,
        diagnostics=dict(Drishti_GS=[525,1050,1575,2100],RIM_ONE_r3=[800,1600,2400,3200],points=64,VJPs=256,candidates=128,clean_U_forwards=33920),
        meter=dict(version=METER_VERSION,adam_source_sha256=ADAM_SOURCE,tolerance=TOLERANCE,upstream_layers='fixed native LAYERS[:-1]',
                   optimizer='torch2.2.1 Adam coupled L2; FP32; scalar lr; amsgrad/capturable/differentiable/maximize/fused false',
                   gradient_order='same graph gS/gK/gA/gU; actual original split_gradients untouched'),
        analysis=dict(performance_final_min=.005,each_seed_final_gt=0.,O2_Old_min=0.,Incoming_each_order_min=-.005,key_final_min=.001,
            key_each_seed_gt=0.,eligible_lambdas=[.5,2.],tie='smaller .5',automatic_followup=False,
            both_perf_fail='STOP_DOSE_AND_CURRENT_NKA_SWD',pairing='seed/order then order within seed then across seeds',
            report_rows=dict(final=28,domain=84,paired=63,adam=64),future_nodes=[]))
    result['plan_sha256']=digest(result);return result


def validate_plan(plan):
    if plan!=canonical_plan():raise ValueError('canonical dose semantics mismatch')
    return plan


def validate_prefix(node,record,receipt):
    if node not in canonical_plan()['nodes']:raise PermissionError('dose node outside scope')
    return old.validate_prefix({**node,'phase':'D1'},record,receipt)


def code_manifest():return old.code_manifest()


def freeze():
    plan=canonical_plan()
    for name,value in [('PLAN',plan),('PREFIX_BINDINGS',plan['prefixes']),('IMPORT_BINDINGS',plan['imports']),('ENVIRONMENT_BINDING',plan['environment']),('FROZEN_OPTIONS',plan['options'])]:write(DOC/(name+'.json'),value)
    return plan
