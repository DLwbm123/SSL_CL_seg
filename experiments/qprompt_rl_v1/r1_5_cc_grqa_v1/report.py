"""Anonymous reports and preregistered progression decisions."""
import csv,json,math
from pathlib import Path
from statistics import mean
from r1_12h.core import atomic,events


def csv_write(path,rows):
    with path.open('w') as f:
        fields=sorted({k for row in rows for k in row});w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def report_forensic(root):
    out=root/'reports';out.mkdir(exist_ok=True);routing=[];grads=[];summary=[]
    for done in sorted((root/'forensic').glob('*/DONE.json')):
        meta=json.loads(done.read_text());rows=events(done.with_name('BATCHES.jsonl'))
        assert len(rows)==100 and all(x['optimizer_calls']==0 for x in rows)
        for x in rows:
            common={k:x[k] for k in ('task','backbone','domain','seed','step','code_commit','prefix_sha','data_schedule_digest')}
            grads.append(dict(**common,**x['gradients']))
            for mode,v in x['routing'].items():routing.append(dict(**common,mode=mode,**v))
        entry=dict(task=meta['task'],batches=100,optimizer_calls=0)
        for mode in ('original','cc'):
            rs=[x['routing'][mode] for x in rows]
            entry[mode]={k:mean(v[k] for v in rs) for k in ('singleton_fraction','assignment_regret_mean','assignment_regret_p95','assignment_regret_max','clip_fraction','min_group_size','max_group_size')}
            entry[mode]['monopolization_image_fraction']=sum(max(s)>6 for v in rs for s in v['group_sizes'])/200
            entry[mode]['all_groups_within_supported_quota']=all(all(({3:3,2:4,1:12}[len(v['supported_classes'])]<=s[c]<={3:6,2:8,1:12}[len(v['supported_classes'])]) for c in v['supported_classes']) for v in rs for s in v['group_sizes'])
        entry['gradient_means']={k:mean(x['gradients'][k] for x in rows if x['gradients'].get(k) is not None) if any(x['gradients'].get(k) is not None for x in rows) else None for k in rows[0]['gradients'] if not k.endswith('_reason')}
        summary.append(entry)
    assert len(summary)==4
    csv_write(out/'FORENSIC_ROUTING.csv',routing);csv_write(out/'FORENSIC_GRADIENTS.csv',grads)
    report=dict(status='COMPLETED',cells=summary,student_forward_calls=400,reference_forward_calls=400,autograd_grad_calls=1600,optimizer_calls=0,scope='current-domain L only; fixed prefix, bank and reference; not a performance gate')
    atomic(out/'FORENSIC_REPORT.json',report)
    lines=['# Zero-update forensic','','400 paired batches; 400 student + 400 reference forwards; 1,600 autograd.grad calls; **zero optimizer calls**. No validation used.','','Per-batch regret p50/p95/max and all gradient norms/cosines are in the CSVs. Values below average batch-level diagnostics; they are not pooled percentiles.','','| Cell | Original monopoly fraction | CC monopoly fraction | Original singleton | CC singleton | CC mean regret | seg–original cosine | seg–CC cosine |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in summary:
        g=x['gradient_means'];lines.append(f"| {x['task']} | {x['original']['monopolization_image_fraction']:.4f} | {x['cc']['monopolization_image_fraction']:.4f} | {x['original']['singleton_fraction']:.4f} | {x['cc']['singleton_fraction']:.4f} | {x['cc']['assignment_regret_mean']:.6f} | {g.get('cos_seg_grqa_orig')} | {g.get('cos_seg_grqa_cc')} |")
    lines+=['','Monopolization is operationally defined here as >6 queries assigned to one class in an image. CC quota checks are exact. A zero gradient produces null cosine, never an invented zero. Full development proceeds regardless of these findings.']
    (out/'FORENSIC_REPORT.md').write_text('\n'.join(lines)+'\n')


def report_phase(root,phase):
    out=root/'reports';out.mkdir(exist_ok=True);results=[];ledger=[];resources=[];mechanism=[];audit=[]
    for p in sorted((root/phase).glob('*/tasks/*/TRAIN_DONE.json')):
        done=json.loads(p.read_text());folder=p.parent;rows=events(folder/'TASK_LEDGER.jsonl')
        counts={e:sum(x['event']==e for x in rows) for e in ('optimizer_attempt','committed','optimizer_failed')}
        ledger.append(dict(**done,**counts));res=dict(done)
        ev=folder/'EVALUATION.json'
        if ev.exists():
            value=json.loads(ev.read_text());results.append(value);res.update({k:v for k,v in value.items() if k.startswith('latency') or k=='deployment_equal'})
        res['updates_per_second']=done['committed_updates']/done['training_seconds'];resources.append(res)
        for x in events(folder/'DIAGNOSTICS.jsonl'):
            mechanism.append(dict(task=done['task'],backbone=done['backbone'],domain=done['domain'],arm=done['arm'],**x))
        audit.append(dict(task=done['task'],seed=done['seed'],recorded_global_step=done['global_step'],committed=counts['committed'],attempted=counts['optimizer_attempt'],failed=counts['optimizer_failed'],prefix_sha=done['prefix_sha'],code_commit=done['code_commit']))
    expected=20 if phase=='development' else 16
    assert len(results)==expected
    seeds=sorted({v['seed'] for v in results});backbones=sorted({v['backbone'] for v in results});domains=sorted({v['domain'] for v in results});arms=sorted({v['arm'] for v in results})
    scores={(x['seed'],x['backbone'],x['domain'],x['arm']):x for x in results}
    deltas=[];cells=[]
    contrasts=[('A1','A0'),('A2','A0'),('A3','A0'),('A4','A0'),('A2','A1'),('A3','A1'),('A4','A1')] if phase=='development' else [('A4','A0')]
    for s in seeds:
        for b in backbones:
            for d in domains:
                for a,z in contrasts:
                    row=dict(seed=s,backbone=b,domain=d,contrast=a+'-'+z,**{k:scores[s,b,d,a][k]-scores[s,b,d,z][k] for k in ('macro','rim','cup','disc_union')})
                    deltas.append(row)
                    if a=='A4' and z=='A0':cells.append(row)
    means={b:{k:mean(x[k] for x in cells if x['backbone']==b) for k in ('macro','rim','cup')} for b in backbones}
    badclass=[dict(seed=x['seed'],backbone=x['backbone'],domain=x['domain'],metric=k,delta=x[k]) for x in cells for k in ('rim','cup') if x[k]<-.01]
    gate=dict(backbone_means=means,primary_mean=mean(x['macro'] for x in cells),positive_cells=sum(x['macro']>0 for x in cells),worst_macro_cell=min(x['macro'] for x in cells),class_violations=badclass)
    gate['passes']=(all(v['macro']>=.003 for v in means.values()) and gate['positive_cells']>=3 and gate['worst_macro_cell']>=-.003 if phase=='development' else gate['primary_mean']>=.003 and all(v['macro']>0 for v in means.values()) and gate['positive_cells']>=6 and gate['worst_macro_cell']>=-.005) and not badclass and all(v[k]>=0 for v in means.values() for k in ('rim','cup'))
    factors=[]
    if phase=='development':
        for b in backbones:
            for metric in ('macro','rim','cup'):
                v={a:mean(x[metric] for x in results if x['backbone']==b and x['arm']==a) for a in arms}
                factors.append(dict(backbone=b,metric=metric,routing=((v['A3']-v['A1'])+(v['A4']-v['A2']))/2,warmup=((v['A2']-v['A1'])+(v['A4']-v['A3']))/2,interaction=v['A4']-v['A3']-v['A2']+v['A1']))
        csv_write(out/'FACTOR_EFFECTS.csv',factors)
    prefix=phase.upper();csv_write(out/(prefix+'_RESULTS.csv'),results);csv_write(out/(prefix+'_DELTAS.csv'),deltas)
    # Preserve both phases in global anonymous summaries.
    for filename,new in [('TASK_LEDGER.jsonl',ledger),('STATE_AUDIT.json',audit),('RESOURCE_REPORT.csv',resources),('MECHANISM_DIAGNOSTICS.csv',mechanism)]:
        if filename.endswith('.jsonl'):
            old=[x for x in events(out/filename) if x.get('phase')!=phase];(out/filename).write_text(''.join(json.dumps(x)+'\n' for x in old+new))
        elif filename.endswith('.json'):
            old=json.loads((out/filename).read_text()) if (out/filename).exists() else {};old[phase]=new;atomic(out/filename,old)
        else:
            old=list(csv.DictReader((out/filename).open())) if (out/filename).exists() else [];old=[x for x in old if x.get('phase')!=phase];csv_write(out/filename,old+new)
    budget=json.loads((root/'BUDGET.json').read_text())[phase]
    status=('R1_5_DEVELOPMENT_MET' if gate['passes'] else 'R1_5_DEVELOPMENT_NOT_MET') if phase=='development' else ('R1_5_CC_GRQA_CONFIRMED_FOR_R2_PROPOSAL' if gate['passes'] else 'R1_5_CONFIRMATION_NOT_MET')
    report=dict(status=status,gate=gate,cells=cells,endpoints=len(results),planned_committed=20000 if phase=='development' else 32000,actual_committed=sum(x['committed_updates'] for x in ledger),optimizer_attempted=budget['attempted'],optimizer_failed=sum(x['optimizer_failed'] for x in ledger),replay=budget['replay'],training_commits=sorted({x['training_code_commit'] for x in results}),confirmation_authorized=phase=='development' and gate['passes'],R2_authorized=False,R3_authorized=False)
    atomic(out/(prefix+'_COMPLETION.json'),report)
    lines=[f'# {prefix} completion','',f"Status: **{status}**. Endpoints: {len(results)}/{expected}. Commits: {report['actual_committed']}/{report['planned_committed']}. Attempts: {budget['attempted']}; failed: {report['optimizer_failed']}; replay: {budget['replay']}.",'','## Primary A4−A0','','| Seed | Backbone | Domain | Macro | Rim | Cup |','|---|---|---|---:|---:|---:|']
    for x in cells:lines.append(f"| {x['seed']} | {x['backbone']} | {x['domain']} | {x['macro']:+.6f} | {x['rim']:+.6f} | {x['cup']:+.6f} |")
    lines+=['','## Frozen gate','',json.dumps(gate,indent=2),'','**Class loss alerts:** '+('NONE' if not badclass else '🔴 '+json.dumps(badclass)), '', 'The full matrix completed before the progression decision. The conservative preregistered class-mean floor is zero. No quotas, weights, ramps, seeds or arms were selected after viewing results. R1 results remain unchanged.','','## Interpretation','','Factor effects for macro, rim and cup are in FACTOR_EFFECTS.csv (development). Per-cell contrasts include A1/A2/A3/A4 versus A0 and A2/A3/A4 versus A1. Mechanism diagnostics retain every 100-update record; compare early clipping and gradient cosines without inferring causal evidence beyond the registered contrasts.','','Single development seed and exposed validation are not independent patient confirmation or statistical significance. Confirmation uses new training seeds on the same exposed validation; it is not an independent dataset. No R2/R3 starts automatically.']
    (out/(prefix+'_COMPLETION.md')).write_text('\n'.join(lines)+'\n')
    (out/'PATCH_LOG.jsonl').touch(exist_ok=True)
    return gate['passes']
