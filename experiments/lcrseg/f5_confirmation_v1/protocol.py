"""Metadata-only frozen matrix, evidence binding, and paired analysis."""
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / 'experiments/lcrseg/docs/f5_confirmation_v1'
OLD = ROOT / 'experiments/lcrseg/docs/five_frameworks_v1/native_execution'
ANCHOR = '96ba0e29f601589a069bcbfdf50b2f4aafc1d883'
WORKER = 'cc21871a43e3cd105cddaf831f5c3ea2fef59b9e'
PARENT = 'NATIVE_LR_SRC_A_3DOMAIN_V1'
B0, B2, F5 = 'B0_PARENT_LCTX', 'B2_PARENT_PAS_KL', 'F5'
CANDIDATES = {B0: 'P1', B2: 'B2_PARENT_PAS_KL_C06', F5: 'F5_C02'}
SEEDS = {B0: (163, 164), B2: (162, 163, 164), F5: (163, 164)}
ORDERS = (('REFUGE', 'RIM_ONE_r3', 'Drishti_GS'), ('REFUGE', 'Drishti_GS', 'RIM_ONE_r3'))
STEPS = {'RIM_ONE_r3': 3200, 'Drishti_GS': 2100}
MANIFEST = '0622f54f42f05d6ef87f9dc89ee9435cf8da03c6c30cd970db6ea167e00dd8a3'
SPLIT = 'f250d97aea1f36f21899f5dd40bb6c9a819e7755aee458c8ee27506496b46a88'
CAPS = dict(formal_scientific=74200, formal_physical=74200, source=0,
            real_L_smoke=24, real_total=74224, synthetic_cuda=60)
METRICS = ('Final', 'Old', 'Incoming', 'Forget')


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def matrix():
    nodes = []
    for family, seeds in SEEDS.items():
        for seed in seeds:
            for order, domains in enumerate(ORDERS, 1):
                seq = f'P1__{CANDIDATES[family]}__S{seed}__O{order}'
                for stage in (1, 2):
                    parent = f'SOURCE_S{seed}' if stage == 1 else seq + '__STAGE1'
                    nodes.append(dict(id=f'{seq}__STAGE{stage}', kind='target', phase='P1', family=family,
                                      candidate_id=CANDIDATES[family], seed=seed, order=order, stage=stage,
                                      sequence_id=seq, domain=domains[stage], updates=STEPS[domains[stage]],
                                      parent_checkpoint=parent, dependencies=[parent]))
    return nodes


def selected_rows(evidence):
    result = []
    for r in evidence['index']:
        i = r.get('identity', {})
        if i.get('family') not in CANDIDATES:
            continue
        selected = CANDIDATES[i['family']]
        if i.get('seed') == 161 and i.get('candidate_id') == selected:
            result.append(r)
        elif i.get('seed') == 162 and r.get('phase') == 'D' and i['family'] in (B0, F5):
            result.append(r)
    if len(result) != 20:
        raise ValueError('CONFIG_BINDING_MISMATCH: expected 20 selected historical target receipts')
    return result


def bound_options(evidence):
    """Use complete receipts, checking every selected receipt, not guessed defaults."""
    manifests = OLD.parent / 'delivery/manifests'
    baseline = next(r for r in read(manifests/'BASELINE_CANDIDATES.json') if r['candidate_id'] == CANDIDATES[B2])
    with (manifests/'SEARCH_CANDIDATES.csv').open() as f:
        f5 = next(r for r in csv.DictReader(f) if r['candidate_id'] == CANDIDATES[F5])
    expected = {B0: {}, B2: baseline['hyperparameters'], F5: json.loads(f5['hyperparameters'])}
    common = dict(lr=.001, parent_lr_multiplier=.5, lr_B_over_A=1., weight_decay=4e-5,
                  warmup_fraction=.2, U_ramp_fraction=.2, feature_lr_multiplier=1., feature_weight_decay_multiplier=1.)
    result = {}
    for family, cid in CANDIDATES.items():
        if evidence['selections']['SELECT_'+family]['candidate_id'] != cid:
            raise ValueError('CONFIG_BINDING_MISMATCH: frozen selection '+family)
        result[family] = {}
    if evidence['selections']['SELECT_PARENT']['candidate_id'] != 'P1':
        raise ValueError('CONFIG_BINDING_MISMATCH: parent selection')
    for r in selected_rows(evidence):
        i, o = r['identity'], r['resolved_options']
        f, d = i['family'], i['domain']
        want = {**common, **expected[f], 'total_steps': STEPS[d]}
        if f == F5:
            want.update(PAS_confidence=.7, PAS_cosine=.5)
        conflicts = {k: {'receipt': o.get(k), 'expected': v} for k, v in want.items() if o.get(k) != v}
        if d in result[f] and result[f][d] != o:
            conflicts['complete_options'] = {'receipt': o, 'previous': result[f][d]}
        if conflicts:
            raise ValueError('CONFIG_BINDING_MISMATCH '+r['node_id']+' '+json.dumps(conflicts))
        result[f][d] = o
    return result


def source_reuse(metadata):
    rows = metadata['sources']
    if {r['node_id'] for r in rows} != {f'SOURCE_S{s}' for s in (162, 163, 164)} or len(rows) != 3:
        raise ValueError('SOURCE_REUSE_BLOCKED: source coverage')
    for r in rows:
        i = r['identity']
        if (i != dict(kind='source', seed=int(r['node_id'][8:]), domain='REFUGE', node_id=r['node_id'], execution_commit=WORKER)
                or r['status'] != 'SEALED' or r['step'] != 8000 or not r['student_exists']
                or r['student_bytes'] <= 0 or len(r['student_hash_declared']) != 64 or r['U_batches'] != 0
                or r['physical_optimizer_calls'] != 8000
                or any(c not in '0123456789abcdef' for c in r['student_hash_declared'])):
            raise ValueError('SOURCE_REUSE_BLOCKED: '+r['node_id'])
    return dict(status='METADATA_VERIFIED_TENSOR_PREFLIGHT_PENDING', sources=rows,
                checked_utc=metadata['checked_utc'], parent=PARENT, network='LCRSegUNet2DJASCL', head='LINEAR',
                manifest_sha256=MANIFEST, split_sha256=SPLIT, source_training='8000 REFUGE CE updates; no U',
                provenance='receipt identity + pinned native_runner.source_task/build + PARENT_BINDING',
                tensor_hash='PENDING', architecture_tensor_schema='PENDING', synthetic_forward='PENDING',
                fresh_source_training=0)


def make_plan():
    from .tests import TEST_NAMES,CPU_PHYSICAL_CAP,MAX_INVOCATIONS
    from .native_qualification import qualification_plan
    evidence = read(OLD/'FINAL_RESULTS_REDUCED.json')
    options = bound_options(evidence)
    imports = [r['node_id'] for r in selected_rows(evidence) if r['identity']['seed'] == 162]
    binding = read(OLD/'PARENT_BINDING.json')
    plan = dict(schema=1, study_id='F5_CONFIRMATION_V1', state='STOP_AWAITING_EXTERNAL_CODE_REVIEW',
                results_anchor=ANCHOR, training_implementation=WORKER, parent=PARENT, nodes=matrix(),
                historical_target_imports=imports, reused_sources=[f'SOURCE_S{s}' for s in (162,163,164)],
                counts=dict(trajectories=14,new_target_stages=28,imported_target_stages=8,reused_sources=3,new_sources=0),
                options=options, caps=CAPS, orders=[list(o) for o in ORDERS], primary_seeds=[163,164], supplementary_seeds=[162,163,164],
                development_seeds=[161], optimizer=dict(name='Adam', betas=[.9,.999], eps=1e-8, polynomial_power=.9,
                                                       initial_lr_A=.0005,initial_lr_B=.0005,initial_lr_R=.001),
                data=dict(manifest_sha256=MANIFEST,split_sha256=SPLIT,split='primary_20pct_split',size=[384,384],batch=2,
                          labels=dict(background=0,rim=1,cup=2,ignore=255),precision='FP32; no autocast; TF32 off; deterministic',
                          geometry='native 3x3 valid readout, bilinear align_corners=True; original augmentation',
                          evaluation='sealed-stage student only, seen val only; no test/future; no score feedback'),
                F5=dict(loss='L_supervised + L_parent_constraint + ramp * 0.25 * (KL + 0.2 * SWD)',
                        SWD_full_ramp_coefficient=.05,d=16,k=4,R_shape=[4,4],F_shape=[16,16],directions=32,
                        samples_max=64,samples_min=8,normalization='original class_swd; equal eligible rim/cup classes',
                        U_direct_gradients='R only',L_direct_gradients='allowed A/B and R',Q='native readout C F_prev',
                        teacher='reset current dense EMA .99',memory='merged weights and cumulative F only'),
                statistics=dict(unit='seed: pair by order, average 2 orders within seed, then average seeds',
                                Final='mean final three domain macro Dice; macro=(rim+cup)/2',
                                Old='mean final first two domains',Incoming='final last domain',
                                Forget='mean(source early-final, first target early-final); not max history',
                                inference='descriptive on development patients; not independent patients or six independent seeds'),
                gates=dict(G1='both new-seed mean F5-B0 Final >0 and mean >=0.005',
                           G2='each order across new seeds: F5-B0 Old >=-0.005 and Forget <=0.005',
                           G3='new seeds and orders mean F5-B2 Final >0; report all tradeoffs',
                           action='budget decisions only after all 28 nodes; all pass recommends P2 review, never executes'),
                P2=dict(status='PLAN_ONLY',executable_nodes=[],conditional_proposal='F5_L_ONLY/F5_NO_SWD/F5_RANDOM_Q; separate review'),
                cpu_tests=dict(tests=list(TEST_NAMES),physical_cap=CPU_PHYSICAL_CAP,invocation_cap=MAX_INVOCATIONS,
                               accounting='durable cumulative CPU_PHYSICAL.jsonl; all attempts retained'),
                resume='all original checkpoint state; ledger must equal checkpoint scientific step; otherwise ENGINEERING_STOP',
                diagnostics='detached existing receipt/last only; unavailable fields NA; no new VJP/forward/RNG',
                evidence_digests=dict(aggregate=digest(evidence),parent_binding=digest(binding),
                                      source_reuse=digest(source_reuse(read(DOC/'REUSE_METADATA.json')))))
    plan['native_qualification'] = qualification_plan()
    plan['integrity_schema'] = 'R1_actual_tensor_and_file_v1; all 28 targets; stat freshness for metadata-only report'
    plan['report_schema'] = 'R1_six_artifacts_v1; COMPLETE_P1_AWAITING_SCIENTIFIC_REVIEW; publication separate'
    plan['plan_sha256'] = digest(plan)
    return plan


def validate_plan(plan):
    if plan != make_plan():
        raise ValueError('CONFIG_BINDING_MISMATCH: plan differs from exact frozen expansion/evidence')
    if len(plan['nodes']) != 28 or sum(n['updates'] for n in plan['nodes']) != 74200:
        raise ValueError('invalid matrix budget')
    return plan


def metrics(receipt):
    i = receipt['identity']; domains = ORDERS[i['order']-1]
    def score(x, d):
        v = x[d]; m = (v['rim']+v['cup'])/2
        if not math.isfinite(m) or abs(m-v['macro_Dice']) > 1e-12:
            raise ValueError('invalid macro Dice')
        return m
    final = [score(receipt['scores'], d) for d in domains]
    return dict(Final=sum(final)/3, Old=sum(final[:2])/2, Incoming=final[2],
                Forget=(score(receipt['timeline']['0'],domains[0])-final[0]+
                        score(receipt['timeline']['1'],domains[1])-final[1])/2)


def paired(rows, seeds):
    table = {}
    for r in rows:
        i = r['identity']
        if i.get('stage') != 2 or i['seed'] not in seeds:
            continue
        key = (i['family'], i['seed'], i['order'])
        if key in table:
            raise ValueError('duplicate paired trajectory')
        table[key] = metrics(r)
    expected = {(f,s,o) for f in CANDIDATES for s in seeds for o in (1,2)}
    if set(table) != expected:
        raise ValueError('incomplete paired matrix')
    result = {}
    for ref in (B0,B2):
        per_order = {f'{s}/O{o}':{m:table[F5,s,o][m]-table[ref,s,o][m] for m in METRICS} for s in seeds for o in (1,2)}
        per_seed = {str(s):{m:sum(per_order[f'{s}/O{o}'][m] for o in (1,2))/2 for m in METRICS} for s in seeds}
        result[ref] = dict(per_order=per_order,per_seed=per_seed,
                           mean={m:sum(r[m] for r in per_seed.values())/len(seeds) for m in METRICS})
    return result


def decisions(primary):
    a,b=primary[B0],primary[B2]
    g1=all(a['per_seed'][str(s)]['Final']>0 for s in (163,164)) and a['mean']['Final']>=.005
    order={str(o):{m:sum(a['per_order'][f'{s}/O{o}'][m] for s in (163,164))/2 for m in METRICS} for o in (1,2)}
    g2=all(v['Old']>=-.005 and v['Forget']<=.005 for v in order.values())
    g3=b['mean']['Final']>0
    return dict(G1=g1,G2=g2,G3=g3,order_tradeoffs=order,
                recommendation='P2_REVIEW_ONLY' if g1 and g2 and g3 else 'STOP_NO_ADDITIONAL_EXPERIMENTS')


def manifest():
    # Bind all repository Python used transitively, including native parent/data dependencies.
    paths = sorted(p for p in (ROOT/'experiments').rglob('*.py') if '__pycache__' not in p.parts)
    files = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return dict(files=files,code_tree_sha256=digest(files),excludes='mutable reports, private payloads, review/launch receipts')


def p0_export():
    evidence=read(OLD/'FINAL_RESULTS_REDUCED.json');rows=selected_rows(evidence)
    extra=read(DOC/'REUSE_METADATA.json')['spectral']
    fields=['node_id','cohort','family','candidate_id','seed','order','stage','training_domain','domain','rim','cup','disc_union','macro_Dice']
    with (DOC/'P0_STAGE_METRICS.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in rows:
            i=r['identity']
            for d,v in r['scores'].items():
                w.writerow(dict(node_id=r['node_id'],cohort='DEVELOPMENT' if i['seed']==161 else 'ALREADY_OBSERVED',
                                **{k:i[k] for k in ('family','candidate_id','seed','order','stage')},training_domain=i['domain'],domain=d,**v))
    write(DOC/'P0_EVIDENCE.json',{'rows':[{**r,'spectral':extra.get(r['node_id'],'NA')} for r in rows],
                                'missing':['full stepwise spectral/prototype history','B2 seed162 targets','new seed163/164 targets']})
    lines=['# P0: read-only historical audit','',
           'Seed 161 is DEVELOPMENT; seed 162 is ALREADY_OBSERVED. No new confirmation results exist.',
           'Existing aggregate receipts only; no patient payload, model load, re-evaluation, or optimizer update.', '',
           '## O2 old-domain breakdown', '',
           '| Seed | Method | Domain | Early | Final | Early − final | F5 − B0 final |',
           '|---:|---|---|---:|---:|---:|---:|']
    for s in (161,162):
        group={r['identity']['family']:r for r in rows if r['identity']['seed']==s and r['identity']['order']==2 and r['identity']['stage']==2}
        for f,r in group.items():
            for d,t in [('REFUGE','0'),('Drishti_GS','1')]:
                early=r['timeline'][t][d]['macro_Dice'];final=r['scores'][d]['macro_Dice']
                delta=group[F5]['scores'][d]['macro_Dice']-group[B0]['scores'][d]['macro_Dice']
                lines.append(f'| {s} | {f} | {d} | {early:.6f} | {final:.6f} | {early-final:+.6f} | {delta:+.6f} |')
    lines += ['', '## Interpretation and missing evidence', '',
              'The per-domain rows expose compensating effects; a positive overall Final is not a claim that every old domain improved.',
              'F5 seed161 two-order ΔFinal vs B0 = -0.000113407; seed162 = +0.013952836 (descriptive, already observed).',
              'B2 C06 development mean Final = 0.658917919. B2 seed162 has no historical target receipts; it is a new execution arm.',
              'P0_EVIDENCE.json preserves saved timelines, complete resolved_options and available stage-entry spectral summaries.',
              'Unrecorded full histories, post-training spectrum, SWD/KL decomposition, and patient-independent evidence = NA.',
              'SOURCE_REUSE.json verifies metadata/file presence only; tensor schema/hash/forward remain PENDING.',
              'No inference gate is evaluated before the complete P1 matrix. No performance-based early stop or extra seed is allowed.']
    (DOC/'P0_REPORT.md').write_text('\n'.join(lines)+'\n')


def prepare():
    # A change in shared training semantics must never be hidden by regenerating a manifest.
    diff=subprocess.check_output(['git','diff',WORKER,'--','experiments/lcrseg/five_frameworks_v1'],cwd=ROOT,text=True)
    if diff:
        raise ValueError('SHARED_TRAINING_SEMANTICS_CHANGED: stop for review')
    write(DOC/'SOURCE_REUSE.json',source_reuse(read(DOC/'REUSE_METADATA.json')))
    plan=make_plan();validate_plan(plan);write(DOC/'PLAN.json',plan)
    p0_export();write(DOC/'CODE_MANIFEST.json',manifest())
    return {'status':'STOP_AWAITING_EXTERNAL_CODE_REVIEW','new_target_stages':28,'formal_update_cap':74200,'real_updates':0}
