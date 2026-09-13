"""All frozen contrasts are reported, without a favorable-result admission gate."""
import csv,json,time,shutil
from collections import Counter
from pathlib import Path
from . import core as c,contract as ct
from experiments.lcrseg.dpr_sign_replication_v0_1.analysis import scores,paired,dump,MEASURES

COMPARISONS=[('C_LCTX','C_FULLMIX'),('C_LCTX','C_CE'),('C_LCTX','C_CED'),('C_FULLMIX','C_CED'),('C_LCTX_LOW','C_FULLMIX_LOW')]

def table(vv,intervals):
    lines=['| Budget | Comparison | Final difference | Paired seed SD | Conditional patient 95% interval |','|---|---|---:|---:|---|']
    from experiments.lcrseg.ams_seq_transfer_v0_1.results import effect_summary
    for a,ref in COMPARISONS:
        mean,sd=effect_summary(vv[a]-vv[ref]);ci=next(r for r in intervals if r['arm']==a and r['baseline']==ref and r['order']=='both' and r['metric']=='Final')
        lines.append(f"| {'low target L' if a.endswith('_LOW') else 'standard target L'} | {a} - {ref} | {mean[0]:+.8f} | {sd[0]:.8f} | [{ci['lower_95']:+.8f}, {ci['upper_95']:+.8f}] |")
    return '\n'.join(lines)+'\n'

def finish(base):
    b=Path(base);p=ct.protocol();source=ct.verify();old=Path(ct.read(b/'private_inputs.json')['parent']);private={};srcscores={};ledger=[];resources=[];ops=Counter();kernel=Counter()
    for t in p['tasks']:
        root=b/'tasks'/t['task_id'];r=ct.read(root/'receipt.json');ev=ct.read(root/'evaluation/receipt.json')
        assert r['source']==ev['source']==source and r['task']==t and r['updates']==t['updates'] and r['student_hash']==ev['student_hash']
        assert r['U_opens']==0 and r['counts'].get('response_VJP',0)==0 and r['fixed_subset_unchanged']
        for phase in ('train','eval'):
            assert ct.read(b/'logs'/(t['task_id']+'_'+phase+'_exit.json'))['exit_code']==0
            oc=ct.read(b/'tasks'/(t['task_id']+('_operations' if phase=='train' else '_eval_operations'))/'operation_counts.json')
            assert oc['status']=='PASS';ops.update(oc['counts'])
        kernel.update(r['counts'])
        for d,v in scores(root/'evaluation/private_patient_metrics.csv',c.DOMAINS).items():private[t['order'],t['seed'],t['arm'],d]=v
        # Counts use private batch indices; only aggregate exposures leave NAS.
        exposure=Counter();valid=0;dice_groups=0
        n=r['L_patients']
        for epoch in range(1,101):
            for ii in c.orders(n,t['seed'],t['domain'],epoch,'labeled_order',t['steps_per_epoch']):exposure.update(ii)
        for line in (root/'steps.jsonl').open():
            z=json.loads(line);valid+=z['raw_valid_supervision_occurrences'];dice_groups+=z['Dice_image_denominator']
        ledger.append(dict(**t,status='COMPLETE',source=source,reused=False,actual_updates=r['updates'],L_patients=n,L_opens=r['L_opens'],U_opens=0,exposure_min=min(exposure.values()),exposure_max=max(exposure.values()),exposure_mean=sum(exposure.values())/n,raw_valid_supervision_occurrences=valid,Dice_image_groups=dice_groups))
        resources.append(dict(task_id=t['task_id'],arm=t['arm'],budget=t['budget'],seconds=r['seconds'],counts=r['counts'],memory=r['memory']))
    for order in ('O1','O2'):
        for seed in p['seeds']:
            sd=c.DOMAINS[0 if order=='O1' else 1];sid=f'{order}_s{seed}_SRC_CE';tid=f'{order}_s{seed}_F_CONV'
            srcscores[order,seed]=scores(old/'tasks'/sid/'evaluation/private_patient_metrics.csv',[sd])[sd]
            for d,v in scores(old/'tasks'/tid/'evaluation/private_patient_metrics.csv',c.DOMAINS).items():private[order,seed,'C_LCTX',d]=v
            rr=ct.read(old/'tasks'/tid/'receipt.json')
            ledger.append(dict(task_id=tid,source_task_id=sid,seed=seed,order=order,arm='C_LCTX',budget='standard',source=ct.PARENT_SOURCE,reused=True,status='REUSED_COMPLETE',new_updates=0,historical_updates=rr['updates']))
    assert ops['optimizer_steps']==ops['backward']==ops['ema_updates']==79500
    assert ops['autograd_grad']==ops['sample_train_unlabeled']==0 and ops['sample_train_labeled']==159000
    out=b/'public_results';out.mkdir()
    vv,ci=paired(out/'comparisons',private,srcscores,p['seeds'],['C_LCTX',*c.ARMS],COMPARISONS,p['analysis_seed'],'FROZEN_DEVELOPMENT_RECIPE_COMPARISON')
    dump(out/'RUN_LEDGER.csv',ledger)
    c.write_json(out/'RESOURCE_ACCOUNTING.json',dict(new_formal_updates=79500,new_source_updates=0,reused_LCTX_new_updates=0,response_VJP=0,operations=dict(ops),kernel_counts=dict(kernel),tasks=resources,qualification=ct.read(b/'qualification_ledger.json'),historical_costs_excluded=True))
    for name in ('SUBSET_MANIFEST.json','REUSE_BINDING.json','TARGET_WEIGHT_SEAL.json'):shutil.copy2(b/name,out/name)
    report=table(vv,ci)
    text='# LCTX paper closeout V1\n\n30/30 new target trainings and 30/30 final-student evaluations completed; 79,500 formal updates. Six sources and six standard-L LCTX targets reused without retraining.\n\n'+report+'\nAll contrasts are retained regardless of direction. Intervals are conditional on the trained models and reused development patients. No independent-patient, third-training-domain, official CutMix/BCP reproduction, SOTA, equivalence or novel U-increment claim follows. The FULLMIX comparison changes both donor supervision and Dice grouping.\n'
    (out/'FINAL_REPORT.md').write_text(text)
    paper=out/'paper';shutil.copytree(ct.DOC/'paper',paper)
    for filename in ('PAPER_DRAFT.md','TABLES.md'):
        path=paper/filename;content=path.read_text();content=content.replace('<!-- NEW_RESULTS_START -->\nPENDING: fixed matrix has not completed.\n<!-- NEW_RESULTS_END -->','<!-- NEW_RESULTS_START -->\n'+report+'<!-- NEW_RESULTS_END -->');path.write_text(content)
    terminal=dict(status='COMPLETE',source=source,new_trainings=30,new_evaluations=30,new_updates=79500,new_source_updates=0,reused_LCTX_new_updates=0,response_VJP=0,performance_gate=False,all_directions_reported=True,additional_experiments=False,stopped=True,finished=time.time())
    c.write_json(out/'TERMINAL.json',terminal)
    c.write_json(out/'NAS_ARCHIVE_RECEIPT.json',dict(status='ARCHIVED_ON_NAS',run_id=b.name,deployments=sum((b/'tasks'/t['task_id']/'deploy_student.pt').is_file() for t in p['tasks']),epoch_checkpoints=sum(len(list((b/'tasks'/t['task_id']).glob('checkpoint_*.pt'))) for t in p['tasks']),private_scores_retained=True,publication='GitHub verification separate',files=[dict(name=str(x.relative_to(out)),bytes=x.stat().st_size) for x in out.rglob('*') if x.is_file()],finished=time.time()))
    return terminal
