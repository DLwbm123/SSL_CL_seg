"""Frozen metadata-only D1 plan. No production launcher or old authorization reuse."""
import copy
import hashlib
import json
from pathlib import Path
from ..f5_confirmation_v1.protocol import read,write,digest,MANIFEST,SPLIT

ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'experiments/lcrseg/docs/native_key_alignment_v0_1'
STUDY='NATIVE_KEY_ALIGNMENT_V0_1'
RUNTIME='667f178c3b80183fd80809760ff31ec5f9a14e25'
RESULTS='3034b2199aa7d6e54ea67499f391a0dd4ea3d21e'
B2='B2_PARENT_PAS_KL'
ARMS={'C0':dict(coordinate='no_op',D=None,lambda_align=0.),
      'C1':dict(coordinate='full_patch',D=144,lambda_align=.05),
      'C2':dict(coordinate='fixed_random_key',D=8,lambda_align=.05),
      'C3':dict(coordinate='native_entry_V',D=8,lambda_align=.05)}


def freeze():
    results=read(ROOT/'results/f5_confirmation_v1_p1/PUBLIC_RESULTS.json')
    old=read(ROOT/'experiments/lcrseg/docs/f5_confirmation_v1/PLAN.json')
    options=copy.deepcopy(old['options'][B2])
    prefixes=[];nodes=[]
    for seed in (163,164):
        for order in (1,2):
            ident=f'P1__B2_PARENT_PAS_KL_C06__S{seed}__O{order}__STAGE1'
            row=next(r for r in results['rows'] if r['node_id']==ident)
            proof=results['integrity'][ident]
            assert row['identity']['stage']==1 and row['identity']['family']==B2 and proof['status']=='VERIFIED'
            prefix=dict(node_id=ident,identity=row['identity'],receipt_sha256=proof['receipt_sha256'],
                student_sha256=proof['student_hash'],transform_sha256=proof['transform_hash'],file_sha256=proof['file_sha256'],
                origin_results_commit=RESULTS,origin_execution_commit=RUNTIME,
                permission='diagnostic_common_prefix_only',runtime_tensor_acceptance='PENDING_FRESH_REVIEW_AND_LAUNCH')
            prefix['binding_sha256']=digest(prefix);prefixes.append(prefix)
            domain='Drishti_GS' if order==1 else 'RIM_ONE_r3';steps=2100 if order==1 else 3200
            for arm in ARMS:
                nodes.append(dict(id=f'D1__{arm}__S{seed}__O{order}__STAGE2',phase='D1',arm=arm,
                    seed=seed,order=order,stage=2,domain=domain,updates=steps,prefix_node=ident,
                    prefix_binding_sha256=prefix['binding_sha256'],options_sha256=digest(options[domain]),
                    initialization='independent fresh adapters/EMA/prototypes/optimizer/warmup from bound B2 prefix',
                    executable=False))
    plan=dict(study_id=STUDY,state='CODE_READY_FOR_REVIEW',parent='NATIVE_LR_SRC_A_3DOMAIN_V1',
        base_code=RUNTIME,base_results=RESULTS,arms=ARMS,nodes=nodes,prefixes=prefixes,
        options=options,manifest_sha256=MANIFEST,split_sha256=SPLIT,
        counts=dict(target_stages=16,formal_updates=42400,new_source_updates=0,new_first_target_updates=0),
        auxiliary=dict(layer='decoder.dec1.merge.block.3',input_channels=16,kernel=[3,3],stride=[1,1],padding=[1,1],
            dilation=[1,1],patch_dimension=144,key_dimension=8,normalize_eps=1e-6,directions=32,min_samples=8,
            max_samples=64,class_ids=[1,2],class_weighting='equal over valid classes',scale='D/8',
            teacher_reference='current clean L only; stop-gradient',student='one additional current clean U feature forward',
            key='existing student entry V; same read-only key for teacher; no extra SVD',
            mapping='same-grid selected convolution input; no final-readout crop/rescale applied',
            border='exclude boundary and any invalid pixel in 3x3 patch; PAS at U center',
            diagnostics='ceil(25/50/75/100 percent total updates); extra VJPs counted; no online adaptation'),
        analysis=dict(primary='C3',pairing='arm minus C0 per seed/order, then two orders within seed',
            gate=dict(each_seed_Final_gt=0.,mean_Final_ge=.005,O2_Old_ge=0.,each_order_Incoming_ge=-.005),
            controls='report C3-C2, C3-C1 per seed and overall; no direction-consistent advantage means no key-specific claim',
            forget_identity='common-prefix DeltaForget=-DeltaOld, not independent evidence',
            complete_all_nodes_before_decision=True),
        future=dict(D2={'executable_nodes':[],'max_targets':24,'max_target_updates':63600,'requires':'new review and authorization'},
                    D3={'executable_nodes':[],'candidate_seeds_unverified':[165,166,167],
                        'source_updates':24000,'target_updates':95400,'requires':'new frozen protocol, review and authorization'}),
        allowed_now=['metadata','bounded_CPU_generated_tests'],forbidden_now=['CUDA','real_smoke','formal_training','monitoring'],
        CPU_caps=dict(optimizer_calls=32,invocations=2),production_approval='NONE; old F5 approvals invalid here')
    plan['plan_sha256']=digest(plan)
    validate_plan(plan)
    write(DOC/'D1_MATRIX.json',plan);write(DOC/'FROZEN_OPTIONS.json',options);write(DOC/'PREFIX_BINDINGS.json',prefixes)
    return plan


def validate_plan(plan):
    if plan['study_id']!=STUDY or digest({k:v for k,v in plan.items() if k!='plan_sha256'})!=plan['plan_sha256']:
        raise ValueError('plan binding mismatch')
    nodes=plan['nodes']
    if (len(nodes)!=16 or sum(n['updates'] for n in nodes)!=42400
            or {(n['arm'],n['seed'],n['order'],n['stage']) for n in nodes}!=
               {(a,s,o,2) for a in ARMS for s in (163,164) for o in (1,2)}
            or any(n['executable'] or n['phase']!='D1' for n in nodes)
            or any(p['executable_nodes'] for p in plan['future'].values())):
        raise ValueError('D1 scope mismatch')
    for n in nodes:
        prefix=next(p for p in plan['prefixes'] if p['node_id']==n['prefix_node'])
        if n['prefix_binding_sha256']!=digest({k:v for k,v in prefix.items() if k!='binding_sha256'}):
            raise ValueError('prefix binding mismatch')
        if n['options_sha256']!=digest(plan['options'][n['domain']]):raise ValueError('options mismatch')
    return plan


def validate_prefix(node,record,receipt):
    """Narrow metadata authority for a future tensor gate, never change old identity.

    Returning metadata is not tensor acceptance and cannot launch training.
    """
    expected=f"P1__B2_PARENT_PAS_KL_C06__S{node['seed']}__O{node['order']}__STAGE1"
    if (node['phase']!='D1' or node['stage']!=2 or node['arm'] not in ARMS
            or record['node_id']!=expected or node['prefix_node']!=expected
            or record['permission']!='diagnostic_common_prefix_only'
            or record['binding_sha256']!=node['prefix_binding_sha256']
            or digest({k:v for k,v in record.items() if k!='binding_sha256'})!=record['binding_sha256']
            or receipt['identity']!=record['identity'] or digest(receipt)!=record['receipt_sha256']
            or receipt['status']!='SEALED' or receipt['student_hash']!=record['student_sha256']):
        raise PermissionError('common prefix outside diagnostic binding')
    return dict(kind='diagnostic_common_prefix',original_identity=copy.deepcopy(record['identity']),
                new_node=node['id'],tensor_acceptance='PENDING')


def code_manifest():
    paths=sorted((ROOT/'experiments').rglob('*.py'))
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if '__pycache__' not in p.parts}
    return dict(files=files,code_tree_sha256=digest(files))
