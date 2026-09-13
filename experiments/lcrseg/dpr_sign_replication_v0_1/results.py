"""New-seed primary result only; historical exploratory statistics remain separate."""
from pathlib import Path
from collections import Counter
import time
import numpy as np
from . import core as c,contract as ct
from .analysis import scores,paired,mechanisms,dump,MEASURES,effect_summary

def decision(delta,ci):
    mean,sd=effect_summary(delta);q=float(mean[0])
    return dict(scientific_state='POSITIVE_REPLICATION_ESTIMATE' if q>0 else 'NO_POSITIVE_REPLICATION_ESTIMATE',mean_delta=dict(zip(MEASURES,map(float,mean))),paired_seed_SD=dict(zip(MEASURES,map(float,sd))),seed_order_averaged_Final=delta.mean(0)[:,0].tolist(),seed_range=[float(delta.mean(0)[:,0].min()),float(delta.mean(0)[:,0].max())],order_Final_means=delta.mean(1)[:,0].tolist(),mean_positive=q>0,practical_margin_met=q>=.005,patient_CI=ci,patient_CI_crosses_zero=ci['crosses_zero'],no_secondary_conjunctive_gates=True)

def finish(base):
    b=Path(base);source=ct.verify();p=ct.protocol();seal=ct.read(b/'TARGET_WEIGHT_SEAL.json');assert seal['source']==source and len(seal['students'])==12
    private={};sources={};ledger=[];resource=[];ops=Counter();counts=Counter();sb=ct.read(b/'SOURCE_BINDING.json')
    for t in p['tasks']:
        root=b/'tasks'/t['task_id'];r=ct.read(root/'receipt.json');ev=ct.read(root/'evaluation/receipt.json')
        assert r['source']==ev['source']==source and r['task']==t and r['updates']==t['updates'] and r['student_hash']==ev['student_hash'] and ev['models']==1
        assert r['counts']['optimizer_steps']==r['counts']['backward']==r['counts']['ema_updates']==t['updates'];counts.update(r['counts'])
        for phase in ('train','eval'):
            assert ct.read(b/'logs'/(t['task_id']+'_'+phase+'_exit.json'))['exit_code']==0
            oc=ct.read(b/'tasks'/(t['task_id']+('_operations' if phase=='train' else '_eval_operations'))/'operation_counts.json');assert oc['status']=='PASS';ops.update(oc['counts'])
        if t['arm']=='SRC_CE':
            assert r['boundary']['parent_student_hash'] is None and r['U_opens']==0 and r['models']==2
            sources[t['order'],t['seed']]=scores(root/'evaluation/private_patient_metrics.csv',[t['domain']])[t['domain']]
        else:
            assert seal['students'][t['task_id']]['student_hash']==r['student_hash'] and r['memory']['full_models']==2
            assert r['boundary']['source_student_hash']==sb['sources'][t['source_task_id']]['student_hash']
            expected=80*c.COUNTS[t['domain']][1] if t['arm']==c.MAIN else 0;assert r['U_opens']==expected
            assert r['fixed_subset_unchanged'] and r['EMA_updates']==t['updates']
            for d,v in scores(root/'evaluation/private_patient_metrics.csv',c.DOMAINS).items():private[t['order'],t['seed'],t['arm'],d]=v
            n=0
            with (root/'steps.jsonl').open() as f:
                import json
                for line in f:
                    z=json.loads(line);n+=1;assert z['position']==n and all(z['checks'].values()) and z['pseudo_labels']==0
            assert n==t['updates']
        ledger.append(dict(**t,status='COMPLETE',actual_updates=r['updates'],student_hash=r['student_hash'],execution_source=source,source_student_hash=r['boundary'].get('source_student_hash'),U_opens=r['U_opens'],L_opens=r['L_opens']))
        resource.append(dict(task_id=t['task_id'],seconds=r['seconds'],counts=r['counts'],memory=r.get('memory',dict(full_models=r.get('models'),peak_reserved=r.get('peak_reserved'),peak_allocated=r.get('peak_allocated')))))
    assert ops['optimizer_steps']==ops['backward']==ops['ema_updates']==47700 and ops['autograd_grad']==counts['response_VJP']<=25440
    assert ops['sample_train_labeled']==95400 and ops['sample_train_unlabeled']==24960 and ops['sample_val']==975
    for t in [t for t in p['tasks'] if t['arm']==c.MAIN]:
        a=b/'tasks'/t['task_id'];ref=b/'tasks'/t['task_id'].replace(c.MAIN,'F_CONV');assert ct.read(a/'warmup.json')==ct.read(ref/'warmup.json')
        assert ct.read(a/'receipt.json')['label_order_hash']==ct.read(ref/'receipt.json')['label_order_hash']
    out=b/'public_results';vv,ci=paired(out,private,sources,p['seeds'],list(c.ARMS),[(c.MAIN,'F_CONV')],p['analysis_seed'],'NEW_OPTIMIZATION_SEED_REPLICATION')
    primary=decision(vv[c.MAIN]-vv['F_CONV'],next(x for x in ci if x['order']=='both' and x['metric']=='Final'))
    terminal=dict(engineering='ENGINEERING_COMPLETE',scientific_state=primary['scientific_state'],primary=primary,formal_tasks=18,formal_updates=47700,source_tasks=6,target_tasks=12,source=source,report_source=source,old_terminal_preserved=True,stopped=True,additional_experiments=False)
    dump(out/'RUN_LEDGER.csv',ledger);mechanisms(b,[t for t in p['tasks'] if t['arm']==c.MAIN],out)
    c.write_json(out/'NEW_SEED_PRIMARY_RESULTS.json',terminal);c.write_json(out/'RESOURCE_ACCOUNTING.json',dict(formal_operations=dict(ops),kernel_counts=dict(counts),tasks=resource,qualification=ct.read(b/'qualification_ledger.json'),source_updates=15900,F_CONV_updates=15900,SIGN_updates=15900,old_training_recounted=0))
    for n in ('SOURCE_BINDING.json','TARGET_WEIGHT_SEAL.json'):c.write_json(out/n,ct.read(b/n))
    c.write_json(out/'TERMINAL.json',terminal)
    (out/'FINAL_REPORT.md').write_text('# Frozen SIGN replication V0.1\n\n'+f"18/18 new trainings,18 fixed evaluations,47,700 updates. {primary['scientific_state']}. New seeds71/72/73 only.\n\nMean SIGN minus new F_CONV Final {primary['mean_delta']['Final']:+.6f}; patient95% interval [{primary['patient_CI']['lower_95']:+.6f},{primary['patient_CI']['upper_95']:+.6f}]. Seed effects {primary['seed_order_averaged_Final']}. Practical0.005 margin met: {primary['practical_margin_met']}.\n\n"+'Old seeds61/62/63 are post-hoc control discovery, separately reported under post_hoc_analysis. They do not replace the old primary or the new-seed estimate. Six new final SRC_CE students are shared pairwise, with no inherited old sources/targets. Native F_CONV has zero U accesses and VJPs. SIGN math, random-key prefix, lambda, pooling, precision and guard are unchanged frozen imports. All target weights sealed before target validation.\n\n'+'The three optimization blocks reuse development patients. Patient intervals are conditional, not independent patient confirmation. New/old and class costs, seeds, orders and tails are separately disclosed. There is no all-directions-win gate. Full pooled response nonincrease is guard-enforced, not independent retention evidence. No parameter search, further seeds or directed FINITE extension. Deployment is one ordinary student.\n')
    files=[dict(name=x.name,bytes=x.stat().st_size) for x in out.iterdir() if x.is_file()]
    c.write_json(out/'NAS_ARCHIVE_RECEIPT.json',dict(status='ARCHIVED_ON_NAS',run_id=b.name,public_files=files,deployments=18,epoch_checkpoints=sum(len(list((b/'tasks'/t['task_id']).glob('checkpoint_*.pt'))) for t in p['tasks']),finished=time.time(),publication='GitHub verification separate'))
    return terminal
