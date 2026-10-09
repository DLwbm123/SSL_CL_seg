"""Descriptive paired diagnostic, never an intervention-selection gate."""
from collections import Counter
import json
from pathlib import Path


def analyze(rows):
    table={(r['context'],r['stream'],r['mode'],r['step'],r['role']):r for r in rows}
    expected={(i,s,m,t,r) for i in range(16) for s in (3,4) for m,t in [('SOURCE',0),('ENTRY',100),('BASE',150),('BASE',300),('OFF',150),('OFF',300)] for r in ('fit','held','old')}
    assert len(table)==len(rows)==576 and set(table)==expected
    deltas=[];fields=('macro','rim','cup','ce','supervised_loss')
    for i in range(16):
        for s in (3,4):
            for mode,step,ref,rt in [('ENTRY',100,'SOURCE',0),('BASE',150,'ENTRY',100),('BASE',300,'ENTRY',100),('OFF',150,'ENTRY',100),('OFF',300,'ENTRY',100)]:
                a=table[i,s,mode,step,'fit'];r={k:a[k] for k in ('context','fold','source_step','labeled_images','condition','stream')};r.update(mode=mode,step=step,reference=ref)
                for role in ('fit','held','old'):
                    for field in fields:r[role+'_'+field]=table[i,s,mode,step,role][field]-table[i,s,ref,rt,role][field]
                r['fit_loss_down_held_loss_up']=r['fit_supervised_loss']<0 and r['held_supervised_loss']>0
                r['fit_loss_down_held_dice_down']=r['fit_supervised_loss']<0 and r['held_macro']<0
                deltas.append(r)
    summaries=[]
    groups=[('all',None)]+[(k,v) for k in ('fold','stream','source_step','labeled_images','condition','context') for v in sorted({r[k] for r in deltas})]
    for kind,value in groups:
        for mode,step in [('ENTRY',100),('BASE',150),('BASE',300),('OFF',150),('OFF',300)]:
            subset=[r for r in deltas if r['mode']==mode and r['step']==step and (kind=='all' or r[kind]==value)]
            out=dict(group='all' if kind=='all' else f'{kind}={value}',mode=mode,step=step,units=len(subset))
            for role in ('fit','held','old'):
                for field in fields:out[role+'_'+field]=sum(r[role+'_'+field] for r in subset)/len(subset)
            for field in ('fit_loss_down_held_loss_up','fit_loss_down_held_dice_down'):out[field]=sum(r[field] for r in subset)
            summaries.append(out)
    return dict(DELTAS=deltas,SUMMARY=summaries)


def selfcheck():
    rows=[dict(context=i,fold=i//8,source_step=2000,labeled_images=2,condition='brightness',stream=s,mode=m,step=t,role=r,macro=.5+(t/1000 if r=='fit' else -t/1000),rim=.5,cup=.5,ce=1.,supervised_loss=1.+(-t/1000 if r=='fit' else t/1000)) for i in range(16) for s in (3,4) for m,t in [('SOURCE',0),('ENTRY',100),('BASE',150),('BASE',300),('OFF',150),('OFF',300)] for r in ('fit','held','old')]
    v=analyze(rows);assert all(r['fit_loss_down_held_loss_up']==32 for r in v['SUMMARY'] if r['group']=='all')
    try:analyze(rows[:-1])
    except AssertionError:pass
    else:raise AssertionError('partial accepted')


if __name__=='__main__':
    import sys
    selfcheck();root=Path(sys.argv[1]);result=analyze(json.loads((root/'RESULTS.json').read_text()))
    for name,v in result.items():(root/(name+'.json')).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    print(json.dumps([r for r in result['SUMMARY'] if r['group']=='all'],indent=2))
