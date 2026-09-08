"""Frozen full-precision gates. Evaluation never chooses a training checkpoint or epoch."""
import argparse,csv,json,statistics
from pathlib import Path
from experiments.lcrseg.single_teacher_scd_v0_1.report import metrics
from experiments.lcrseg.single_teacher_scd_v0_1.engine import write_json
DOC=Path('experiments/lcrseg/docs/l05_ssl_replication')
CONFIG=json.loads((DOC/'PROTOCOL.json').read_text())
def read(p):return json.loads(Path(p).read_text())
def csvrows(p):return list(csv.DictReader(Path(p).open()))
def table(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def historical():
    rows=csvrows('experiments/lcrseg/docs/single_teacher_scd_r1/STAGE_DOMAIN_MATRIX.csv')
    return [dict(optimization_seed=0,data_split_seed=0,arm=x['arm'],stage=int(x['stage']),domain=x['domain'],domain_index=int(x['domain_index']),**{k:float(x[k]) for k in ('macro_fg_dice','rim_dice','cup_dice')}) for x in rows]
def score(rows,seed,arm):
    xs=[x for x in rows if x['optimization_seed']==seed and x['arm']==arm]
    if {(x['stage'],x['domain_index']) for x in xs}!={(0,0),(1,0),(1,1),(2,0),(2,1),(2,2)}:raise ValueError('incomplete evaluation matrix')
    matrix={(x['stage'],x['domain_index']):x['macro_fg_dice'] for x in xs}
    out=metrics(matrix)
    # Python 3.12 changed builtin float sum; preserve the frozen Python 3.10 left-fold exactly.
    out['F']=((matrix[2,0]+matrix[2,1])+matrix[2,2])/3
    return out
def current(rows,s,a,t,k):return next(x[k] for x in rows if x['optimization_seed']==s and x['arm']==a and x['stage']==x['domain_index']==t)
def gate(rows,kind,engineering=True):
    entries=[]
    def add(name,value,threshold,strict=False):
        passed=value>threshold if strict else value>=threshold
        entries.append(dict(gate=name,value=value,threshold=threshold,strict=strict,passed=bool(passed)))
    seeds=[0] if kind=='P_GATE' else [1,2]
    arm='L05' if kind=='L05_REPLICATION' else 'L05_SSL';ref='S' if arm=='L05' else 'L05'
    diffs=[]
    try:
        for s in seeds:
            a,b=score(rows,s,arm),score(rows,s,ref);delta={k:a[k]-b[k] for k in a};diffs.append(dict(seed=s,**delta))
            if kind=='P_GATE':
                add('F_vs_L05',delta['F'],.01);add('H_vs_L05',delta['H'],-.01)
            else:add(f'seed{s}_F_positive',delta['F'],0.,True)
            for t in (1,2):
                for k,threshold in (('macro_fg_dice',-.01),('rim_dice',-.02),('cup_dice',-.02)):
                    add(f'seed{s}_stage{t}_{k}_vs_S',current(rows,s,arm,t,k)-current(rows,s,'S',t,k),threshold)
        if kind!='P_GATE':
            add('paired_F_mean',statistics.mean(x['F'] for x in diffs),.01)
            add('paired_H_mean',statistics.mean(x['H'] for x in diffs),0. if arm=='L05' else -.01)
    except (ValueError,StopIteration):return dict(kind=kind,status='NOT_EVALUATED_INCOMPLETE',passed=False,entries=entries)
    add('engineering',int(engineering),1)
    passed=all(x['passed'] for x in entries)
    names={'P_GATE':'ADMITTED_BY_FROZEN_GATE','L05_REPLICATION':'L05_FIXED_SPLIT_REPLICATION_SUPPORTED','SSL_REPLICATION':'SSL_ADDITION_FIXED_SPLIT_SIGNAL'}
    return dict(kind=kind,status=names[kind] if passed else 'NOT_ADMITTED_BY_FROZEN_GATE' if kind=='P_GATE' else 'NOT_ESTABLISHED',passed=passed,entries=entries,paired_differences=diffs)

def collect(base,seeds=(0,1,2)):
    base=Path(base);rows=historical();stages=[];tasks=[];exits=[];resources=[];mechanism=[];updates=0
    source=read(base/'INPUT_LINEAGE.json')['source']
    for seed in seeds:
        sd=base/f'optimization_seed{seed}'
        common_path=sd/'common/stage0';common_rows=[]
        if seed==0:common_rows=[x for x in rows if x['arm']=='S' and x['stage']==0]
        elif (common_path/'val.json').exists():common_rows=read(common_path/'val.json')['rows']
        for arm in (('L05_SSL',) if seed==0 else ('common','S','L05','L05_SSL')):
            root=sd/arm
            if not root.exists():continue
            parent_file=root/'parent_receipt.json'
            task=read(parent_file) if parent_file.exists() else dict(status='RUNNING_OR_INCOMPLETE',arm=arm,optimization_seed=seed)
            tasks.append(task)
            for f in root.glob('*_exit.json'):exits.append(dict(arm=arm,optimization_seed=seed,operation=f.name,**read(f)))
            for f in root.glob('*_resource.json'):resources.append(dict(arm=arm,optimization_seed=seed,operation=f.name,**read(f)))
            if arm!='common':rows.extend(dict(x,arm=arm,optimization_seed=seed,data_split_seed=0) for x in common_rows)
            for t in ((0,) if arm=='common' else (1,2)):
                sr=root/f'stage{t}';logs=[json.loads(x) for x in (sr/'steps.jsonl').read_text().splitlines()] if (sr/'steps.jsonl').exists() else []
                updates+=len(logs)
                assert [x['step'] for x in logs]==list(range(1,len(logs)+1))
                for x in logs:
                    if x['epoch']%5==0:mechanism.append(dict(optimization_seed=seed,arm=arm,stage=t,epoch=x['epoch'],step=x['step'],raw_u_ce=x['raw_lu'],weighted_u_ce=x['weighted_lu'],labeled_kd=x['labeled_kd'],teacher_u_forward=0,pas=x['pas'],distillation=x['distillation'],numerics=x['numerics']))
                if not (sr/'final_receipt.json').exists():continue
                rec=read(sr/'final_receipt.json');assert rec['source']==source and rec['optimization_seed']==seed and rec['data_split_seed']==0
                assert rec['counters']['optimizer_steps']==len(logs)==CONFIG['updates_per_stage'][t]
                assert rec['counters'].get('teacher_u_forward',0)==0
                if arm in ('common','S','L05'):assert rec['unlabeled_images_opened']==0
                if t and arm!='S':assert rec['teacher_initial_hash']==rec['teacher_final_hash']==rec['parent_hash']
                if t==2:assert rec['parent_hash']==read(root/'stage1/final_receipt.json')['student_hash']
                if t==1:
                    expected=next(x['student_hash'] for x in read(base/'INPUT_LINEAGE.json')['reused'] if x['arm']=='common') if seed==0 else read(common_path/'final_receipt.json')['student_hash']
                    assert rec['parent_hash']==expected
                ep=[json.loads(x) for x in (sr/'epochs.jsonl').read_text().splitlines()]
                assert max(x['max_full_models'] for x in ep)<=2
                stages.append(dict(**rec,logged_epoch_seconds=sum(x['seconds'] for x in ep),peak_memory={k:max(x[k] for x in ep) for k in ('max_allocated','max_reserved','cpu_max_rss_kib','max_full_models','student_parameters_bytes','teacher_parameters_bytes','gradient_bytes','optimizer_bytes','prototype_bytes')}))
                if not (sr/'val.json').exists():continue
                v=read(sr/'val.json');assert v['student_hash']==rec['student_hash'] and v['inference_full_models']==1
                for op in ('train','eval'):assert read(root/f'stage{t}_{op}_exit.json')['exit_code']==0
                if arm!='common':rows.extend(dict(x,arm=arm,optimization_seed=seed,data_split_seed=0) for x in v['rows'])
            if task['status']=='COMPLETE' and arm!='common':
                d=read(root/'stage2/deployment.json');assert d['status']=='PASS' and d['models']==1 and d['teacher_hidden']
                assert d['student_hash']==read(root/'stage2/final_receipt.json')['student_hash']
                assert read(root/'deploy_exit.json')['exit_code']==0
                task['deployment']=d
    return rows,dict(source=source,tasks=tasks,stages=stages,child_exits=exits,resource_waits=resources,new_formal_updates=updates,historical_formal_updates=57208,cumulative_formal_updates=57208+updates),mechanism

def seal_p_gate(base):
    base=Path(base);target=base/'P_GATE.json'
    if target.exists():raise FileExistsError('P_GATE already sealed')
    rows,audit,_=collect(base,seeds=(0,))
    complete=any(x['arm']=='L05_SSL' and x['optimization_seed']==0 and x['status']=='COMPLETE' for x in audit['tasks'])
    result=gate(rows,'P_GATE',complete);result['source']=audit['source']
    write_json(target,result);return result

def finish(base):
    base=Path(base);out=base/'public_results';out.mkdir()
    rows,audit,mechanism=collect(base);pg=read(base/'P_GATE.json')
    required={(0,'L05_SSL'),(1,'common'),(1,'S'),(1,'L05'),(2,'common'),(2,'S'),(2,'L05')}
    if pg['passed']:required|={(1,'L05_SSL'),(2,'L05_SSL')}
    done={(x['optimization_seed'],x['arm']) for x in audit['tasks'] if x['status']=='COMPLETE'}
    engineering=required<=done
    if engineering:assert audit['new_formal_updates']==(53100 if pg['passed'] else 42500)
    gates=dict(P_GATE=pg,L05_replication=gate(rows,'L05_REPLICATION',{(s,'S') for s in (1,2)}|{(s,'L05') for s in (1,2)}<=done),SSL_replication=gate(rows,'SSL_REPLICATION',engineering) if pg['passed'] else dict(status='NOT_ADMITTED_BY_FROZEN_GATE',passed=False))
    summaries=[]
    for s,a in sorted({(x['optimization_seed'],x['arm']) for x in rows}):
        try:summaries.append(dict(optimization_seed=s,arm=a,**score(rows,s,a)))
        except ValueError:pass
    factorial=[]
    by={x['arm']:x for x in summaries if x['optimization_seed']==0}
    for label,a,b in [('CE_without_U_KD','L05_SSL','L05'),('CE_with_U_KD','D','U0'),('U_KD_with_CE','D','L05_SSL')]:
        if a in by and b in by:factorial.append(dict(contrast=label,method=a,reference=b,**{k:by[a][k]-by[b][k] for k in ('F','H','N')}))
    if all(a in by for a in ('D','U0','L05_SSL','L05')):factorial.append(dict(contrast='interaction',**{k:(by['D'][k]-by['U0'][k])-(by['L05_SSL'][k]-by['L05'][k]) for k in ('F','H','N')}))
    stats=[]
    for arm in ('S','L05','L05_SSL'):
        xs=[x for x in summaries if x['arm']==arm]
        if len(xs)==3:
            for metric in ('F','H','N'):stats.append(dict(arm=arm,metric=metric,training_seeds=3,mean=statistics.mean(x[metric] for x in xs),sample_sd=statistics.stdev(x[metric] for x in xs)))
    audit.update(qualification_costs={name:read(base/name/'qualification.json') for name in ('qualification_server',) if (base/name/'qualification.json').exists()},extra_val_pas_forwards=0,pseudo_label_precision='NOT_EVALUATED',all_teacher_u_forward=0,no_historical_train_replay=True,data_split_seed=0,actual_admitted_budget=53100 if pg['passed'] else 42500)
    audit['per_task_wall_seconds']=[dict(optimization_seed=s,arm=a,wall_seconds=max(x['finished_unix'] for x in audit['child_exits'] if x['optimization_seed']==s and x['arm']==a)-min(x['started_unix'] for x in audit['child_exits'] if x['optimization_seed']==s and x['arm']==a),training_process_seconds=sum(x['finished_unix']-x['started_unix'] for x in audit['child_exits'] if x['optimization_seed']==s and x['arm']==a and '_train_' in x['operation'])) for s,a in sorted({(x['optimization_seed'],x['arm']) for x in audit['child_exits']})]
    audit['gpu_wait_seconds']=sum(x['wait_seconds'] for x in audit['resource_waits'])
    if audit['child_exits']:
        audit['parallel_wall_seconds']=max(x['finished_unix'] for x in audit['child_exits'])-min(x['started_unix'] for x in audit['child_exits'])
    table(out/'STAGE_DOMAIN_MATRIX.csv',rows)
    table(out/'PER_CLASS_METRICS.csv',[{k:x[k] for k in ('optimization_seed','data_split_seed','arm','stage','domain','rim_dice','cup_dice','macro_fg_dice')} for x in rows])
    table(out/'REPLICATION_BY_SEED.csv',summaries);table(out/'THREE_SEED_MEAN_SD.csv',stats)
    table(out/'SEED0_FACTORIAL_RESULT.csv',[dict(contrast='observed_recipe',**x) for x in summaries if x['optimization_seed']==0]+factorial)
    write_json(out/'GATE_ACCOUNTING.json',gates);write_json(out/'TRAINING_AND_MEMORY_ACCOUNTING.json',audit)
    write_json(out/'MECHANISM_DIAGNOSTICS.json',dict(unit='per-batch records every fifth epoch',rows=mechanism))
    terminal=dict(engineering='COMPLETE' if engineering else 'INCOMPLETE_ENGINEERING',gates=gates,source=audit['source'],old_states_unchanged=True,stopped=True)
    write_json(out/'STATUS.json',terminal)
    lines=['# L05_SSL factorial and fixed-split replication','',f"Engineering: {terminal['engineering']}. Fixed data_split_seed=0; optimization seeds are distinct training random streams, not patient splits.",'',f"P_GATE: {pg['status']}; L05 replication: {gates['L05_replication']['status']}; SSL addition: {gates['SSL_replication']['status']}.",'','| Optimization seed | Arm | F | H | N |','|---:|---|---:|---:|---:|']
    for x in summaries:lines.append(f"| {x['optimization_seed']} | {x['arm']} | {x['F']:.9f} | {x['H']:.9f} | {x['N']:.9f} |")
    lines+=['',f"New successful formal updates: {audit['new_formal_updates']}; historical attempts: 57208; cumulative: {audit['cumulative_formal_updates']}. Failed/qualification attempts are separate. All complete-matrix counters are reconciled with the manifest budget.",'','Every current rim/cup and full-precision gate is retained in PER_CLASS_METRICS.csv and GATE_ACCOUNTING.json. Confirmation uses only new optimization seeds1/2; THREE_SEED_MEAN_SD.csv is descriptive. Recipe contrasts in SEED0_FACTORIAL_RESULT.csv are full-training recipe differences, not pixel-level causal effects. No statistical-significance, unseen-patient, multi-split or SOTA claim is made.','', 'No teacher-U forwards or U-KD, SCD solving, EMA, third complete model, history replay, test/hidden GT, parameter search, or main merge. Optional additional val PAS precision diagnostics were not run; coverage is not precision. Old locks and results remain unchanged.','', 'All admitted tasks are terminal. Stop; no extra seed, recipe or module is started. Public publication verification is performed separately after this background run is inspected.']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    return terminal
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args();print(json.dumps(finish(a.base)))
