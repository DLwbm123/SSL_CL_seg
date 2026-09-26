"""Post-run aggregate-only closeout. Does not launch training or read medical payloads."""
import json,os
from pathlib import Path
from .runtime import TASKS,PLAN
from .analysis import table
from r1_12h.core import atomic,events


def closeout(run,old):
    run=Path(run);old=Path(old);root=run/'reports'
    assert (run/'FINAL.json').exists(),'closeout requires execution closure'
    import csv
    results=list(csv.DictReader((root/'RESULTS.csv').open()));deltas=list(csv.DictReader((root/'PAIRED_DELTAS.csv').open()));factors=list(csv.DictReader((root/'FACTOR_EFFECTS.csv').open()))
    comparisons=[]
    for row in results:
        if int(row['optimization_seed'])!=261 or row['arm'] not in ('B0','B1'):continue
        arm={'B0':'Q0','B1':'Q1'}[row['arm']];path=old/'tasks'/f"{row['backbone']}__{row['domain']}__{arm}"/'EVALUATION.json'
        historical=json.loads(path.read_text()) if path.exists() else {}
        for metric in ('rim','cup','macro','disc_union'):
            new=float(row[metric]) if row[metric] else None;before=historical.get(metric)
            comparisons.append(dict(backbone=row['backbone'],domain=row['domain'],new_arm=row['arm'],old_arm=arm,metric=metric,new_value=new,old_value=before,delta=new-before if new is not None and before is not None else None,old_training_code_commit=historical.get('training_code_commit')))
    table(root,'HISTORICAL_REPRODUCTION.csv',comparisons)
    factor_means=[]
    for seed in PLAN['optimization_seeds']:
        for backbone in PLAN['backbones']+['ALL']:
            for metric in ('rim','cup','macro'):
                rows=[r for r in factors if int(r['seed'])==seed and (backbone=='ALL' or r['backbone']==backbone) and r['metric']==metric]
                entry=dict(seed=seed,backbone=backbone,metric=metric)
                for field in ('readout_effect','image_effect','interaction'):entry[field]=sum(float(r[field]) for r in rows)/len(rows) if rows and all(r[field] for r in rows) else None
                factor_means.append(entry)
    table(root,'FACTOR_EFFECTS_BY_BACKBONE.csv',factor_means)
    jobs=[];totals=dict(optimizer_attempts=0,optimizer_failures=0,physical_commits=0,python_exceptions=0,replayed_work=0,lost_tail_commits=0,successful_but_discarded_commits=0,final_path_unique_updates=0)
    for task,spec in TASKS.items():
        folder=run/'tasks'/task;rows=events(folder/'TASK_LEDGER.jsonl');commits=[r for r in rows if r['event']=='committed'];attempts=[r for r in rows if r['event']=='optimizer_attempt'];done=folder/'TRAIN_DONE.json';progress=done if done.exists() else folder/'PROGRESS.json';step=json.loads(progress.read_text())['global_step'] if progress.exists() else spec['global_start'];valid=max(0,step-spec['global_start'])
        row=dict(task=task,optimizer_attempts=len(attempts),optimizer_failures=sum(r['event']=='optimizer_failed' for r in rows),physical_commits=len(commits),python_exceptions=len(list(folder.glob('FAILURE_*.private.json'))),replayed_work=sum(r.get('category')=='replay' for r in attempts),lost_tail_commits=sum(r['step']>step for r in commits),successful_but_discarded_commits=max(0,len(commits)-valid),final_path_unique_updates=valid)
        for key in totals:totals[key]+=row[key]
        jobs.append(row)
    atomic(root/'TRANSACTION_AUDIT.json',dict(totals=totals,tasks=jobs,budget=json.loads((run/'BUDGET.json').read_text()),note='Successful-but-discarded is physical committed minus unique retained path. Lost tail is committed beyond the retained step; replay attempts are charged separately.'))
    forensic=[]
    for backbone in PLAN['backbones']:forensic.extend(events(run/'forensic'/backbone/'READOUT_FORENSIC.private.jsonl'))
    groups=[];routes=[]
    for backbone in PLAN['backbones']:
        for domain in PLAN['domains']:
            rows=[r for r in forensic if r['backbone']==backbone and r['domain']==domain];samples=[s for r in rows for s in r['samples']]
            for cls in (0,1,2):
                for group in ('matched','unmatched'):
                    data=[g for s in samples for c in s['classes'] if c['gt_class']==cls and c['supported'] for g in c['groups'] if g['group']==group]
                    value=dict(backbone=backbone,domain=domain,gt_class=cls,group=group,supported_images=len(data))
                    for metric in ('count','mask_dice','class_probability','no_object_probability'):
                        values=[d[metric] for d in data if d[metric] is not None];value[metric+'_mean']=sum(values)/len(values) if values else None
                    groups.append(value)
                for top in (0,1,2):
                    count=sum(a==cls and t==top for s in samples for a,t in zip(s['assignments'],s['prototype_top1']))
                    routes.append(dict(backbone=backbone,domain=domain,matched_gt_class=cls,prototype_top1_class=top,count=count))
    table(root,'MATCHED_UNMATCHED_AGGREGATES.csv',groups);table(root,'PROTOTYPE_MATCH_ROUTES.csv',routes)
    for name in ('PREFIX_AUDIT.json','SEED_AND_SCHEDULE_AUDIT.json','NATIVE_QUALIFICATION.json','DATA_BINDING.json'):
        if (run/name).exists():atomic(root/name,json.loads((run/name).read_text()))
    patch=Path(__file__).with_name('PATCH_LOG.jsonl');(root/'PATCH_LOG.jsonl').write_text(patch.read_text())
    fresh=json.loads((root/'FRESH_SEED_REPLICATION.json').read_text());qual=[]
    for backbone in PLAN['backbones']:
        p=run/'qualification'/backbone/'PASSED.json'
        if p.exists():qual.append(json.loads(p.read_text()))
    atomic(root/'NATIVE_QUALIFICATION_DETAILS.json',qual)
    text=(root/'FINAL_INTERPRETATION.md').read_text().replace('B0/B1 comparisons against the old Q0/Q1 require the final closeout review; no equivalence of final scores is assumed from the loss regression alone.','B0/B1 comparisons against old Q0/Q1 are in HISTORICAL_REPRODUCTION.csv; loss equivalence does not imply identical endpoint scores.')
    text+='\n## Evidence and limitations\n\n'+f"Physical training commits: {totals['physical_commits']}; unique retained updates: {totals['final_path_unique_updates']}; replay attempts: {totals['replayed_work']}; optimizer failures: {totals['optimizer_failures']}; Python exceptions: {totals['python_exceptions']}; successful but discarded commits: {totals['successful_but_discarded_commits']}. Qualification and smoke attempts are separate in TRANSACTION_AUDIT.json.\n\n"
    text+='All endpoints use the final step3000 student. B2 remains the sole preregistered primary candidate. Fresh-seed criteria and each threshold are preserved in FRESH_SEED_REPLICATION.json. Factor effects are reported per cell and equally averaged within each backbone. HISTORICAL_REPRODUCTION.csv compares seed261 B0/B1 with the old Q0/Q1; any numerical differences remain visible.\n\n'
    text+='The readout adds an explicit supervised target using existing L labels. Matched/unmatched masks and prototype routes are descriptive aggregates; GT-derived unmatched sets never modify training or deployment. Gradient cosines describe a fixed training batch, not generalization. The structural classifier perturbation demonstrates non-equivalence between prototype GRQA and semantic supervision, not a unique explanation of previous negative results.\n\n'
    text+='CUDA determinism remains warn_only as preregistered. Exact restoration and numerical next-update equivalence are distinct checks; PATCH_LOG records the qualification assertion repair and all consumed attempts. No bitwise training reproducibility is claimed. New seeds reuse the same exposed validation images; there is no independent patient confirmation or statistical significance claim.\n'
    (root/'FINAL_INTERPRETATION.md').write_text(text)
    print(json.dumps(dict(status=fresh['decision'],endpoints=len(results),totals=totals)))

if __name__=='__main__':closeout(os.environ['EXEC_RUN'],os.environ['EXEC_OLD_RUN'])
