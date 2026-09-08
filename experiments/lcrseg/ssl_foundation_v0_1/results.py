"""Frozen full-precision selection and separate confirmation questions."""
import csv,json,math
from pathlib import Path
from .core import DOMAINS,ARMS,write_json
from .diagnostics import csv_write,METRICS

def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(Path(p).open()))
def score(table,seed,arm,domain,key='macro_fg_dice'):
    return float(next(r[key] for r in table if int(r['seed'])==seed and r['arm']==arm and r['domain']==domain and r['model']=='student' and int(r['epoch'])==100))
def q(table,seed,arm):return (score(table,seed,arm,DOMAINS[0])+score(table,seed,arm,DOMAINS[1]))/2

def screen(table,engineering=True):
    stronger=max(('SUP_G0','SUP_G1'),key=lambda a:(q(table,11,a),a=='SUP_G0'));candidates=[]
    for arm in (a for a in ARMS if not a.startswith('SUP')):
        sup='SUP_'+arm[-2:];delta=[score(table,11,arm,d)-score(table,11,sup,d) for d in DOMAINS]
        cd=[score(table,11,arm,d,k)-score(table,11,sup,d,k) for d in DOMAINS for k in ('rim_dice','cup_dice')]
        checks=dict(mean_domain_delta=(delta[0]+delta[1])/2>=.01,min_domain_delta=min(delta)>=-.005,class_guard=min(cd)>=-.02,stronger_sup=q(table,11,arm)>=q(table,11,stronger)+.005,engineering_and_diagnostics=engineering)
        candidates.append(dict(arm=arm,sup=sup,delta=delta,class_delta=cd,mean_domain_delta=(delta[0]+delta[1])/2,min_domain_delta=min(delta),Q=q(table,11,arm),checks=checks,passed=all(checks.values())))
    passed=[r for r in candidates if r['passed']]
    passed.sort(key=lambda r:(-r['min_domain_delta'],-r['Q'],not r['arm'].startswith('MT_CONF'),not r['arm'].endswith('G0'),r['arm']))
    return dict(status='ADMITTED' if passed else 'SSL_FOUNDATION_NOT_ESTABLISHED',selected=passed[0]['arm'] if passed else None,stronger_sup_P1=stronger,candidates=candidates,passed=bool(passed))

def confirmation(table,gate,engineering=True):
    a=gate['selected'];sup='SUP_'+a[-2:];deltas=[];classes=[];per=[];pas_deltas=[];pas_classes=[];strong=[]
    for seed in (12,13):
        ds=[score(table,seed,a,d)-score(table,seed,sup,d) for d in DOMAINS];deltas+=ds
        cs=[score(table,seed,a,d,k)-score(table,seed,sup,d,k) for d in DOMAINS for k in ('rim_dice','cup_dice')];classes+=cs
        per.append(dict(seed=seed,arm=a,mean_delta=sum(ds)/2,min_domain_delta=min(ds),min_class_delta=min(cs),strong_sup_delta=q(table,seed,a)-q(table,seed,gate['stronger_sup_P1'])))
        strong.append(per[-1]['strong_sup_delta'])
        pas,conf='MT_PAS_'+a[-2:],'MT_CONF_'+a[-2:]
        pas_deltas.append((score(table,seed,pas,DOMAINS[0])-score(table,seed,conf,DOMAINS[0])+score(table,seed,pas,DOMAINS[1])-score(table,seed,conf,DOMAINS[1]))/2)
        pas_classes.extend(score(table,seed,pas,d,k)-score(table,seed,conf,d,k) for d in DOMAINS for k in ('rim_dice','cup_dice'))
    checks=dict(each_new_seed_positive=all(r['mean_delta']>0 for r in per),mean_gain=sum(deltas)/4>=.01,domain_guard=min(deltas)>=-.01,class_guard=min(classes)>=-.02,stronger_sup=sum(strong)/2>=.005,engineering=engineering)
    passed=all(checks.values());paspass=all(x>0 for x in pas_deltas) and sum(pas_deltas)/2>=.005 and min(pas_classes)>=-.02
    return dict(status='SINGLE_DOMAIN_SSL_FOUNDATION_ESTABLISHED_ON_FIXED_SPLIT' if passed else 'SINGLE_DOMAIN_SSL_REPLICATION_NOT_ESTABLISHED',checks=checks,per_seed=per,mean_delta=sum(deltas)/4,PAS_status='PAS_INCREMENTAL_DICE_SIGNAL' if paspass else 'PAS_INCREMENTAL_DICE_NOT_ESTABLISHED',PAS_per_seed_delta=pas_deltas,PAS_min_class_delta=min(pas_classes))

def collect(base):
    table=[];quality=[];ledger=[];memory=[];deploy=[]
    for root in sorted(Path(base).glob('seed*/*/*')):
        if not root.is_dir():continue
        ident=dict(seed=int(root.parent.parent.name[4:]),domain=root.parent.name,arm=root.name)
        log=root/'steps.jsonl';n=sum(1 for _ in log.open()) if log.exists() else 0
        ok=(root/'receipt.json').exists() and (root/'deployment.json').exists()
        ledger.append(dict(**ident,successful_logged_updates=n,status='COMPLETE' if ok else 'INCOMPLETE'))
        if (root/'receipt.json').exists():
            rec=read(root/'receipt.json');assert rec['optimizer_updates']==n;memory.append(rec)
        if (root/'deployment.json').exists():deploy.append(dict(**ident,**read(root/'deployment.json')))
        for epoch in (20,40,60,80,100):
            er=root/f'eval{epoch}'
            if (er/'receipt.json').exists():
                assert read(er/'receipt.json')['state_preserved']
                table.extend(rows(er/'metrics.csv'));quality.extend(rows(er/'quality_aggregate.csv'))
    return table,quality,ledger,memory,deploy

def phase_complete(base,seeds,arms):
    for seed in seeds:
        for domain in DOMAINS:
            roots=[Path(base)/f'seed{seed}'/domain/a for a in arms]
            initial=[];presentations=[]
            for root in roots:
                rec=read(root/'receipt.json');dep=read(root/'deployment.json');init=read(root/'initialization.json');warm=read(root/'warmup.json')
                assert rec['optimizer_updates']==(3200 if domain==DOMAINS[0] else 2100)
                assert dep['status']=='PASS' and dep['student_hash']==rec['student_hash']
                assert rec['source']==dep['source']==init['source'];assert rec['models']==2
                assert rec['sigma_gradient'] is False
                assert warm['u_opens']==0
                if root.name.startswith('SUP'):assert rec['unlabeled_opens']==0
                for e in (20,40,60,80,100):
                    dr=read(root/f'eval{e}/receipt.json');assert dr['status']=='COMPLETE' and dr['state_preserved']
                    assert dr['val_cases']==(40 if domain==DOMAINS[0] else 25)
                    assert (root/f'eval{e}/quality_aggregate.csv').stat().st_size>0
                initial.append(init);presentations.append(rec['label_order_hash'])
            assert len({r['student_hash'] for r in initial})==1 and len(set(presentations))==1
            for gas in ('G0','G1'):
                ww=[read(r/'warmup.json') for r in roots if r.name.endswith(gas)]
                assert len({r['student_hash'] for r in ww})==1 and len({r['ema_hash'] for r in ww})==1 and len({r['label_order_hash'] for r in ww})==1
    return True

def finish(base,gate,confirmation_result=None):
    base=Path(base);out=base/'public_results';out.mkdir()
    table,quality,ledger,memory,deployment=collect(base)
    complete=not gate.get('engineering_failure') and all(r['status']=='COMPLETE' for r in ledger) and len(ledger)==(28 if gate.get('passed') else 12)
    status=dict(engineering='COMPLETE' if complete else 'INCOMPLETE_ENGINEERING',SSL=confirmation_result['status'] if confirmation_result else gate['status'],PAS=confirmation_result['PAS_status'] if confirmation_result else 'NOT_CONFIRMED_NEW_SEEDS',formal_updates=sum(r['successful_logged_updates'] for r in ledger),prior_formal_updates=99708,stopped=True,P2_admitted=gate.get('passed',False),selected=gate.get('selected'))
    if list(base.glob('seed*/*/*/diagnostic_failure.json')):status['engineering']='INCOMPLETE_DIAGNOSTICS'
    status['cumulative_formal_updates']=99708+status['formal_updates']
    csv_write(out/'RUN_LEDGER.csv',ledger)
    if table:
        final=[r for r in table if int(r['epoch'])==100]
        for name,rr in [('EPOCH_CURVES.csv',table),('SINGLE_DOMAIN_METRICS.csv',[r for r in final if r['model']=='student']),('PER_CLASS_METRICS.csv',final),('EMA_VS_STUDENT.csv',table)]:csv_write(out/name,rr)
        contrasts=[]
        for seed in sorted({int(r['seed']) for r in table}):
            arms=sorted({r['arm'] for r in final if int(r['seed'])==seed})
            pairs=[(a,'SUP_'+a[-2:]) for a in arms if not a.startswith('SUP')]
            pairs += [(f'MT_PAS_G{g}',f'MT_CONF_G{g}') for g in (0,1) if f'MT_PAS_G{g}' in arms]
            pairs += [(f'{kind}_G1',f'{kind}_G0') for kind in ('SUP','MT_CONF','MT_PAS') if f'{kind}_G1' in arms and f'{kind}_G0' in arms]
            for a,b in pairs:
                for d in DOMAINS:
                    contrasts.append(dict(seed=seed,domain=d,arm=a,reference=b,**{k+'_delta':score(table,seed,a,d,k)-score(table,seed,b,d,k) for k in METRICS}))
        csv_write(out/'COMPONENT_CONTRASTS.csv',contrasts)
    if quality:
        csv_write(out/'PSEUDO_QUALITY_BY_CLASS.csv',quality);csv_write(out/'PSEUDO_QUALITY_AGGREGATE.csv',quality)
    csv_write(out/'CONFIRMATION_BY_SEED.csv',confirmation_result['per_seed'] if confirmation_result else [dict(status='NOT_ADMITTED' if not gate.get('passed') else 'INCOMPLETE')])
    for name,obj in [('SCREEN_DECISION.json',gate),('CONFIRMATION.json',confirmation_result or {'status':'NOT_ADMITTED'}),('MEMORY_RUNTIME_AND_ACCESS.json',dict(tasks=memory,formal_updates=status['formal_updates'],qualification='Separate TEST_REPORT and real_smoke receipts; no hidden extra formal updates')),('DEPLOYMENT_REPORT.json',deployment),('STATUS.json',status)]:write_json(out/name,obj)
    (out/'PAS_COVERAGE_PRECISION.md').write_text('Mandatory quality diagnostics: raw/confidence/PAS, teacher and student separately. Patient equal means average repeats within patient first; pooled draw-pixel counts disclose all denominators. Zero support is blank UNDEFINED_NO_SUPPORT. PSEUDO_QUALITY_BY_CLASS/AGGREGATE contain the same class-level aggregate view; private per-case and per-repeat rows remain in NAS eval folders. Fresh diagnostic prototypes are never written back. Actual-training PAS library is separately identified; epoch20 is NOT_STARTED. Precision and Dice evidence are separate. Full quantitative interpretation is required at public closeout.\n')
    (out/'NEXT_STAGE_DRAFT.md').write_text('STOP. No additional tuning, seeds, backbone, frozen-layer experiment or CL run is authorized. If the fixed SSL candidate is confirmed, a separately reviewed protocol may examine new-patient evidence and explicitly defined module freezing. Otherwise only this fixed adaptation recipe is closed; this is not a claim that general SSL is ineffective.\n')
    (out/'FINAL_REPORT.md').write_text('# Single-domain SSL foundation V0.1\n\n'+json.dumps(status,indent=2)+'\n\n1. Engineering: fixed matrix and logged successful updates are in RUN_LEDGER.\n2. Final-student effects against both SUP heads are in SINGLE_DOMAIN_METRICS and COMPONENT_CONTRASTS.\n3. EMA is secondary only; EMA_VS_STUDENT includes every fixed snapshot, never best epoch.\n4. Class-specific PAS precision/coverage/recall and GAS contrasts are reported separately. No coverage-only effectiveness claim.\n5. Evidence concerns the previously exposed split and training randomness, not independent patients, CL, or SOTA. No post-hoc recipe changes.\n\nScreen: '+json.dumps(gate,indent=2)+'\n\nConfirmation: '+json.dumps(confirmation_result,indent=2)+'\n')
    write_json(base/'TERMINAL.json',status);return status
