"""Read only predecessor scores and metadata. Never open old models or raw images/GT."""
import json
from pathlib import Path
from .core import DOMAINS,write_json
from .statistics import summary,q
from experiments.lcrseg.ssl_head_control_v0_1.results import rows,table_write,read
OLD_SOURCE='098c9c73dc1b7dd1b723363e45cbe34e169fe354'
def baseline(base,old):
    b=Path(base);old=Path(old);doc=Path(__file__).parents[1]/'docs/ssl_target_path_v0_1'
    final=[r for r in rows(doc/'FINAL_METRICS.csv') if r['arm'] in ('SUP','T_SCE')]
    epoch=[r for r in rows(doc/'EPOCH_METRICS.csv') if r['arm'] in ('SUP','T_SCE')]
    assert len(final)==12 and len(epoch)==120
    lineage={};late=[]
    for s in (31,32,33):
        for d in DOMAINS:
            identities=[]
            for a in ('SUP','T_SCE'):
                p=old/f'seed{s}'/d/a;r=read(p/'receipt.json');de=read(p/'deployment.json')
                assert r['source']==de['source']==OLD_SOURCE and r['status']=='TRAINING_COMPLETE' and de['status']=='PASS'
                identities.append((r['initialization']['student_hash'],r['label_order_hash']))
                # Compare the existing aggregate, not GT or an additional forward.
                actual=rows(p/'eval100/metrics.csv')
                for model in ('student','ema'):
                    expected=next(x for x in epoch if int(x['seed'])==s and x['domain']==d and x['arm']==a and x['epoch']=='100' and x['model']==model)
                    observed=next(x for x in actual if x['model']==model)
                    for field in ('macro_fg_dice','rim_dice','cup_dice'):assert expected[field]==observed[field]
                    vv=[float(x['macro_fg_dice']) for x in epoch if int(x['seed'])==s and x['domain']==d and x['arm']==a and x['model']==model and int(x['epoch']) in (60,80,100)]
                    late.append(dict(seed=s,domain=d,arm=a,model=model,epochs='60/80/100',**summary(vv),interpretation='within-trajectory late fluctuation, not independent seeds'))
            assert identities[0]==identities[1]
            lineage[f'{s}/{d}']=dict(initial_student_hash=identities[0][0],label_order_hash=identities[0][1])
    result=dict(status='PASS',old_source=OLD_SOURCE,old_run_root=str(old),old_task_bindings=lineage,
        Q={a:summary(q(final,s,a) for s in (31,32,33)) for a in ('SUP','T_SCE')},
        paired_gain=summary(q(final,s,'T_SCE')-q(final,s,'SUP') for s in (31,32,33)),
        old_GT_reads=0,old_model_loads=0,old_forward=0,old_qualification_reexecuted=False,
        old_accounting_gap='Synthetic EMA/I/O subtotals remain unavailable; not filled with zero or re-run.')
    write_json(b/'P0_BASELINE.json',result);table_write(b/'P0_EPOCH_METRICS.csv',epoch);table_write(b/'P0_LATE_VARIATION.csv',late)
    return result
