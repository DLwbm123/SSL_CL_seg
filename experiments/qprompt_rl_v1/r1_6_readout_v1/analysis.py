"""Evidence-bounded aggregate tables and frozen fresh-seed decision."""
import csv,json,io,time
from pathlib import Path
from .runtime import PLAN,TASKS,diagnostic_rows,disk_bytes
from r1_12h.core import atomic,events


def table(root,name,rows):
    stream=io.StringIO();keys=sorted({k for row in rows for k in row})
    if keys:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    (root/name).write_text(stream.getvalue())


def aggregate(run,queue,budget,final=False):
    run=Path(run);root=run/'reports';root.mkdir(exist_ok=True)
    results=[];resources=[];ledger=[];diagnostics=[]
    for task,spec in TASKS.items():
        folder=run/'tasks'/task
        row=dict(task=task,seed=spec['seed'],backbone=spec['backbone'],domain=spec['domain'],arm=spec['arm'],status=queue[task]['status'])
        if (folder/'EVALUATION.json').exists():
            value=json.loads((folder/'EVALUATION.json').read_text())
            # The legacy evaluator publishes scores before the provenance-enrichment write.
            if all(k in value for k in ('optimization_seed','prefix_sha256','data_schedule_digest')):results.append(value)
        for name in ('TRAIN_DONE.json','PROGRESS.json'):
            if (folder/name).exists():
                value=json.loads((folder/name).read_text());row['global_step']=value['global_step'];row['valid_updates']=max(0,value['global_step']-spec['global_start'])
                if name=='TRAIN_DONE.json':resources.append(dict(task=task,training_seconds=value['training_seconds'],updates_per_second=row['valid_updates']/max(value['training_seconds'],1e-9),peak_allocated=value['peak_allocated'],peak_reserved=value['peak_reserved']))
                break
        ledger.append(row)
        for item in diagnostic_rows(folder/'DIAGNOSTICS.jsonl'):
            diagnostics.append(dict(task=task,step=item['meta']['step'],losses=item['losses'],readout=item['readout'],gradients=item['gradients']))
    index={(r['optimization_seed'],r['backbone'],r['domain'],r['arm']):r for r in results}
    deltas=[];factors=[]
    for seed in PLAN['optimization_seeds']:
        for backbone in PLAN['backbones']:
            for domain in PLAN['domains']:
                values={a:index.get((seed,backbone,domain,a)) for a in PLAN['arms']}
                meta=dict(seed=seed,backbone=backbone,domain=domain)
                for a,b in [('B2','B0'),('B3','B1'),('B1','B0'),('B3','B2'),('B2','B1')]:
                    for metric in ('macro','rim','cup'):
                        x=values[a];y=values[b];delta=x[metric]-y[metric] if x and y and x[metric] is not None and y[metric] is not None else None
                        deltas.append(dict(**meta,contrast=a+'-'+b,metric=metric,delta=delta))
                for metric in ('macro','rim','cup'):
                    v=[values[a][metric] if values[a] else None for a in ('B0','B1','B2','B3')]
                    a,b,c,d=v
                    factors.append(dict(**meta,metric=metric,readout_effect=((c-a)+(d-b))/2 if None not in v else None,image_effect=((b-a)+(d-c))/2 if None not in v else None,interaction=d-c-b+a if None not in v else None))
    fresh=[r for r in deltas if r['seed'] in (262,263) and r['contrast']=='B2-B0']
    macro=[r for r in fresh if r['metric']=='macro'];complete=len(results)==48 and all(r['delta'] is not None for r in fresh)
    checks={}
    if complete:
        mean=lambda rows:sum(r['delta'] for r in rows)/len(rows)
        checks=dict(mean_macro_delta=mean(macro),positive_cells=sum(r['delta']>0 for r in macro),worst_macro_delta=min(r['delta'] for r in macro),backbone_macro={b:mean([r for r in macro if r['backbone']==b]) for b in PLAN['backbones']},backbone_class={b:{m:mean([r for r in fresh if r['backbone']==b and r['metric']==m]) for m in ('rim','cup')} for b in PLAN['backbones']},worst_class_delta=min(r['delta'] for r in fresh if r['metric'] in ('rim','cup')))
        passed=checks['mean_macro_delta']>=.003 and checks['positive_cells']>=6 and checks['worst_macro_delta']>=-.005 and all(v>0 for v in checks['backbone_macro'].values()) and all(v>=0 for row in checks['backbone_class'].values() for v in row.values()) and checks['worst_class_delta']>=-.01
        decision='READOUT_BASELINE_GAIN_REPLICATED' if passed else 'READOUT_BASELINE_GAIN_NOT_ESTABLISHED'
    else:decision='INCOMPLETE_ENGINEERING_OR_BUDGET'
    summaries=[]
    for seed in PLAN['optimization_seeds']:
        for backbone in PLAN['backbones']+['ALL']:
            for contrast in ('B2-B0','B3-B1','B1-B0','B3-B2','B2-B1'):
                for metric in ('macro','rim','cup'):
                    rows=[r for r in deltas if r['seed']==seed and (backbone=='ALL' or r['backbone']==backbone) and r['contrast']==contrast and r['metric']==metric]
                    summaries.append(dict(seed=seed,backbone=backbone,contrast=contrast,metric=metric,delta_mean=sum(r['delta'] for r in rows)/len(rows) if all(r['delta'] is not None for r in rows) else None,supported_cells=sum(r['delta'] is not None for r in rows)))
    table(root,'RESULTS.csv',results);table(root,'PAIRED_DELTAS.csv',deltas);table(root,'FACTOR_EFFECTS.csv',factors);table(root,'PER_SEED_SUMMARY.csv',summaries);table(root,'RESOURCE_REPORT.csv',resources);table(root,'TASK_LEDGER.csv',ledger)
    (root/'TASK_LEDGER.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in ledger))
    atomic(root/'FRESH_SEED_REPLICATION.json',dict(decision=decision,complete=complete,checks=checks,criteria=PLAN['fresh_seed_evaluation']))
    atomic(root/'TRAINING_DIAGNOSTICS.json',diagnostics)
    if final:
        import torch
        audits=[]
        for task,spec in TASKS.items():
            path=run/'tasks'/task/('final.pt' if spec['is_final_endpoint'] else 'prefix.pt')
            if not path.exists():continue
            state=torch.load(path,map_location='cpu',weights_only=False)
            from .runner import finite_tree
            valid=state['global_step']==state['data_cursor']==state['scheduler']['last_epoch']==spec['global_end'] and finite_tree(state) and state['reference'] is None and (state['bank'] is not None)==(spec['arm'] in ('B1','B3'))
            optimizer_steps=[int(s['step']) for s in state['optimizer']['state'].values() if 'step' in s]
            audits.append(dict(task=task,passed=valid,global_step=state['global_step'],optimizer_step_min=min(optimizer_steps),optimizer_step_max=max(optimizer_steps),scheduler_step=state['scheduler']['last_epoch'],rng_present=all(k in state for k in ('python_rng','numpy_rng','cpu_rng','cuda_rng')),prefix_sha256=state['prefix_sha256'],prefix_source_commit=state['prefix_source_commit'],training_code_commit=state['training_code_commit']))
            del state
        atomic(root/'STATE_AUDIT.json',audits)
        if not all(a['passed'] for a in audits):raise RuntimeError('final checkpoint audit failed')
        forensic=[]
        for backbone in PLAN['backbones']:forensic.extend(events(run/'forensic'/backbone/'READOUT_FORENSIC.private.jsonl'))
        # Public report contains cell aggregates only; matching indices and case-level rows stay private.
        public=[]
        for backbone in PLAN['backbones']:
            for domain in PLAN['domains']:
                rows=[r for r in forensic if r['backbone']==backbone and r['domain']==domain]
                if not rows:continue
                means={k:sum(r['losses'][k] for r in rows)/len(rows) for k in rows[0]['losses']}
                blocks={}
                for block in ('all','body','readout'):
                    blocks[block]={}
                    for group in ('norms','cosines'):
                        blocks[block][group]={}
                        for key in rows[0]['gradients'][block][group]:
                            values=[r['gradients'][block][group][key] for r in rows if r['gradients'][block][group][key] is not None]
                            blocks[block][group][key]=sum(values)/len(values) if values else None
                public.append(dict(backbone=backbone,domain=domain,batches=len(rows),loss_means=means,gradient_means=blocks,structural_all_pass=all(all(r['structural'].values()) for r in rows)))
        atomic(root/'READOUT_FORENSIC.json',public)
        flat=[]
        for row in forensic:
            for sample in row['samples']:
                for cls in sample['classes']:
                    flat.append(dict(backbone=row['backbone'],domain=row['domain'],batch=row['batch'],gt_class=cls['gt_class'],supported=cls['supported'],eta=cls['eta'],semantic_error=cls.get('semantic_error')))
        summaries=[]
        for backbone in PLAN['backbones']:
            for domain in PLAN['domains']:
                for cls in (0,1,2):
                    rows=[r for r in flat if r['backbone']==backbone and r['domain']==domain and r['gt_class']==cls and r['supported']]
                    mean=lambda key:sum(r[key] for r in rows if r[key] is not None)/sum(r[key] is not None for r in rows) if any(r[key] is not None for r in rows) else None
                    summaries.append(dict(backbone=backbone,domain=domain,gt_class=cls,supported_images=len(rows),eta_mean=mean('eta'),semantic_error_mean=mean('semantic_error')))
        table(root,'READOUT_FORENSIC.csv',summaries)
        (root/'READOUT_FORENSIC.md').write_text('# Readout forensic\n\n'+f'{len(forensic)}/64 fixed zero-update batches. Original matcher assignments remain in private evidence; public JSON and CSV contain aggregate measurements only. Gradients are recorded for the full model, body and readout parameters. Prototype labels are descriptive only. No diagnostic chooses an arm or hyperparameter.\n')
        physical=[]
        for path in (run/'tasks').glob('*/TASK_LEDGER.jsonl'):physical.extend(events(path))
        atomic(root/'RESOURCE_REPORT.json',dict(budget=budget,physical_commits=sum(x['event']=='committed' for x in physical),optimizer_failures=sum(x['event']=='optimizer_failed' for x in physical),valid_updates=sum(r.get('valid_updates',0) for r in ledger),disk_bytes=disk_bytes(run),disk_cap_bytes=32*1024**3,user_quota='UNAVAILABLE',shared_capacity_is_not_quota=True))
    (root/'FINAL_INTERPRETATION.md').write_text('# R1.6 readout supervision\n\n'+f'Status: {decision if final else "RUNNING"}. Endpoints {len(results)}/48. Valid updates {sum(r.get("valid_updates",0) for r in ledger)}/64000.\n\n'+
        'B2-B0 is primary; B3-B1 is secondary. Seed 261 reuses the historical prefixes. Seeds 262/263 test optimization-seed replication on the same exposed validation images; they are not independent patient confirmation and no statistical significance is asserted. This is supervised baseline control, not evidence of a new SSL, continual-learning or reinforcement-learning contribution. Missing cells remain null. No R2/R3 is authorized.\n\n'+
        'Paired deltas, readout/image main effects and interaction are reported for macro, rim and cup without selecting an early checkpoint. B0/B1 comparisons against the old Q0/Q1 require the final closeout review; no equivalence of final scores is assumed from the loss regression alone.\n')
