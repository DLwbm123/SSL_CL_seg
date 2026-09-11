"""Dual-source reporting on the unchanged 36-task analysis, plus actual recovery cost."""
from collections import Counter
from pathlib import Path
from experiments.lcrseg.lctx_weight_memory_v0_1 import core as c
from experiments.lcrseg.lctx_weight_memory_v0_1.results import MEASURES,ARMS
from experiments.lcrseg.ams_seq_transfer_v0_1.results import csv_read,effect_summary
from experiments.lcrseg.ssl_foundation_v0_1.diagnostics import csv_write
from . import contract as ct

def supplement(base,terminal,values,costs,scientific_ops):
    b=Path(base);out=b/'public_results';m=ct.read(b/'RECOVERY_MANIFEST.json');old=Path(m['old_run'])
    new_ops=Counter();old_ops=Counter()
    ledger=csv_read(out/'RUN_LEDGER.csv')
    for row in ledger:
        tid=row['task_id'];reused=tid in m['reused_task_ids'];root=ct.task_root(b,tid)
        oc=ct.read(root.parent/(tid+'_operations')/'operation_counts.json')['counts']
        (old_ops if reused else new_ops).update(oc)
        row.update(training_source_commit=ct.OLD_SOURCE if reused else terminal['source'],
                   reaudit_source_commit=terminal['source'],attempt_kind='REUSED_ZERO_UPDATE' if reused else 'NEW_ATTEMPT',
                   history='original success preserved' if reused else 'original failed420 preserved separately' if tid==m['rerun_task_id'] else 'original unstarted task')
    failed=ct.read(old/'tasks/O1_s62_LR_SRC_AB_operations/operation_counts.json')['counts']
    assert new_ops['optimizer_steps']==new_ops['backward']==new_ops['ema_updates']==49900
    assert old_ops['optimizer_steps']==45500 and failed['optimizer_steps']==420
    assert scientific_ops['optimizer_steps']==95400
    terminal.update(new_formal_updates=49900,reused_formal_updates=45500,old_failed_attempt_updates=420,
                    physical_formal_updates=95820,original_attempt_terminal_preserved=True,
                    training_math_changed=False,publication='NAS complete; GitHub publication verification separate')
    decisions=ct.read(out/'DECISIONS.json');decisions['terminal']=terminal
    required=[('F_FULL','O1',62,'rim_dice',-.07731558745756356),
              ('LR_RAND','O2',61,'cup_dice',-.06875783410804343),
              ('LR_RAND','O1',62,'rim_dice',-.07962406526873261)]
    counterexamples=[]
    for ref,o,seed,metric,expected in required:
        r=next(z for z in costs if z['arm']=='LR_SRC_A' and z['baseline']==ref and z['order']==o and z['seed']==seed and z['role']=='old' and z['metric']==metric)
        assert abs(r['delta']-expected)<1e-14;counterexamples.append(r)
    assert not decisions['VALUE']['checks']['class_single']
    assert not next(g for g in decisions['SOURCE_GEOMETRY'] if g['baseline']=='LR_RAND')['checks']['class_single']
    decisions['preserved_necessary_counterexamples']=counterexamples;c.write_json(out/'DECISIONS.json',decisions)
    summary=[];trajectories=[]
    for arm in ARMS:
        mean,sd=effect_summary(values[arm])
        summary.append(dict(arm=arm,**dict(zip(MEASURES,map(float,mean))),**{k+'_paired_seed_SD':float(sd[i]) for i,k in enumerate(MEASURES)}))
        for oi,o in enumerate(('O1','O2')):
            for si,seed in enumerate((61,62,63)):trajectories.append(dict(arm=arm,order=o,seed=seed,**dict(zip(MEASURES,map(float,values[arm][oi,si])))))
    for name,rows in [('RUN_LEDGER',ledger),('ARM_SUMMARY',summary),('TRAJECTORIES',trajectories),('PRESERVED_COUNTEREXAMPLES',counterexamples)]:csv_write(out/(name+'.csv'),rows)
    public_manifest={k:v for k,v in m.items() if k!='old_run'};public_manifest['old_run_id']=old.name
    c.write_json(out/'RECOVERY_MANIFEST.json',public_manifest)
    q=ct.read(b/'qualification_ledger.json')
    c.write_json(out/'RECOVERY_EXECUTION_COSTS.json',dict(new_formal_counts=dict(new_ops),reused_success_counts=dict(old_ops),old_failed_attempt_counts=failed,
                 scientific_matrix_counts=dict(scientific_ops),formal_physical_optimizer_updates=95820,qualification=q,
                 old_physical_optimizer_updates=45920,new_optimizer_updates=49900,source_training_updates=0,new_real_smoke_updates=0))
    root=b/'tasks/O1_s62_LR_SRC_AB';ev=ct.read(root/'diagnostics_420.json');ev['checkpoint']='checkpoint_420.pt'
    c.write_json(out/'FAILURE_STATE_EVIDENCE.json',dict(original_failure_updates=420,original_latest_position=399,
                 original_420_exact_state_available=False,new399=ct.read(root/'PREFIX_399_COMPARISON.json'),new420=ev,
                 original_attempt_preserved=True,interpretation='new rerun exact state, not restoration of the original unsaved 420 state'))
    c.write_json(out/'REPAIR_DIFF_AND_QUALIFICATION.json',dict(source=terminal['source'],training_function_identity=ct.read(ct.DOC/'TRAINING_IDENTITY.json'),
                 qualification=q,local=ct.read(b/'qualification_local/qualification.json'),server=ct.read(b/'qualification_server/qualification.json'),
                 reaudit=[{k:v for k,v in ct.read(b/'reaudit'/(tid+'.json')).items() if k!='rows'} for tid in m['reused_task_ids']],
                 reaudit_GT_reads=0,reaudit_forward=0,reaudit_optimizer=0,basis_recomputed=False))
    report=(out/'FINAL_REPORT.md').read_text()
    report=report.replace('# LCTX Weight Memory V0.1','# LCTX Weight Memory V0.1.1 numerical recovery',1)
    report=report.replace('36 new targets / 95400 formal updates; source recovery 0; static reference 0 updates.',
                          '36 successful targets: 18 reused (45500 updates) + 18 new (49900 updates). Scientific matrix 95400; formal physical total 95820 includes preserved failed420. Source recovery 0; six STATIC_SOURCE evaluations, zero updates.')
    lead=['','The original attempt remains INCOMPLETE_ENGINEERING / NOT_ADJUDICATED_INCOMPLETE_MATRIX. The recovered matrix has its own engineering and scientific terminal. Existing class protection violations remain necessary-condition failures, regardless of subsequent averages.','',
          '| Preserved main-minus-control old-class counterexample | Dice difference (percentage points) |','|---|---:|']
    for r in counterexamples:lead.append(f"| {r['order']} seed{r['seed']} vs {r['baseline']}, {r['metric']} | {100*r['delta']:+.6f} |")
    lead+=['','Training delta/effective/forward, Adam, EMA, initialization, cached bases, streams and scientific gates retain their frozen identities. The numerical contract was explicitly amended: the parameterized D gate remains 1e-5; actual FP32 addition uses an independent elementwise budget and a conservative FP64 engineering envelope. ROUNDOFF_LIMITED_REPRESENTATION is not strict zero leakage or old-prediction invariance.',
           'Old successes were re-audited only at available final unmerged checkpoints on the training CUDA backend, without GT, forward or optimizer calls. Old epoch20 states were not reconstructed. Original evaluation patient scores were reused; new final-student evaluations follow the original fixed protocol.',
           f"New qualification cost: {q['new_synthetic_updates']} synthetic updates, zero real-smoke updates; old 118 synthetic and 24 discarded real-smoke updates remain separate. Six STATIC_SOURCE target evaluations occurred after all 36 target weights were sealed.",'']
    pos=report.index('\n',report.index('\n')+1);report=report[:pos]+'\n'+'\n'.join(lead)+report[pos:]
    report+='\nA versus AB (SINGLE_VS_DUAL), A versus F_CONV, F_CONV versus F_FULL, and source versus random/free effects use the full paired tables; AB has a smaller effective rank-manifold dimension. ARM_SUMMARY reports three-seed SD after averaging the two orders within each seed. No additional experiment was admitted.\n'
    (out/'FINAL_REPORT.md').write_text(report)
    c.write_json(out/'TERMINAL.json',terminal)
    files=[dict(name=p.name,bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()]
    c.write_json(out/'NAS_ARCHIVE_RECEIPT.json',dict(status='REPORTS_ARCHIVED_ON_NAS',run_id=b.name,public_files=files,
                 weights='36 qualified targets retained across original and recovery NAS runs',private_artifacts='weights, RNG, patient rows and raw logs remain private on NAS',
                 verification='writer completion and one file inventory; checkpoint and source identities are separately bound by protocol-required hashes',github_verified=False))
    return terminal
