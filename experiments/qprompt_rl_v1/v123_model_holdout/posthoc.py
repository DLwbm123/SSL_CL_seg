"""Algebraic decomposition of published aggregates; no image/model operations."""
import json
import math
from pathlib import Path
from statistics import mean


def decompose(r):
    out={}
    for name,weighted in (('repaired','weighted_repaired'),('harmed','weighted_harmed')):
        out[name+'_10']=2*(r[name]-r[weighted]);out[name+'_21']=2*r[weighted]-r[name]
        assert all(out[name+'_'+k]>=0 and float(out[name+'_'+k]).is_integer() for k in ('10','21'))
        assert out[name+'_10']+out[name+'_21']==r[name]
    n=r['corrected_nll']-r['base_nll'];w=r['weighted_corrected_nll']-r['weighted_base_nll']
    out['nll_10']=2*(n-w);out['nll_21']=2*w-n
    for k,weight in (('10',.5),('21',1.)):
        out['accuracy_contribution_'+k]=weight*(out['repaired_'+k]-out['harmed_'+k])/r['weight']
        out['nll_contribution_'+k]=weight*out['nll_'+k]/r['weight']
    assert math.isclose(sum(out['accuracy_contribution_'+k] for k in ('10','21')),r['weighted_net_accuracy'],abs_tol=1e-13)
    assert math.isclose(sum(out['nll_contribution_'+k] for k in ('10','21')),r['weighted_nll_delta'],abs_tol=1e-13)
    return out


def selfcheck():
    # Known direction counts and NLL sums reconstruct exactly under EMA-class weights .5/1.
    r=dict(repaired=9,weighted_repaired=7,harmed=7,weighted_harmed=4.5,base_nll=10,corrected_nll=11,weighted_base_nll=8,weighted_corrected_nll=10,weight=20,weighted_net_accuracy=.125,weighted_nll_delta=.1)
    d=decompose(r)
    assert [d[k] for k in ('repaired_10','repaired_21','harmed_10','harmed_21','nll_10','nll_21')]==[4,5,5,2,-2,3]
    identity=dict(r,repaired=0,weighted_repaired=0,harmed=0,weighted_harmed=0,corrected_nll=10,weighted_corrected_nll=8,weighted_net_accuracy=0,weighted_nll_delta=0)
    assert all(v==0 for v in decompose(identity).values())


def analyze(rows):
    assert len(rows)==32 and {(r['context'],r['stream']) for r in rows}=={(i,s) for i in range(16) for s in (3,4)}
    decomposed=[dict(context=r['context'],fold=r['fold'],source_step=r['source_step'],labeled_images=r['labeled_images'],condition=r['condition'],stream=r['stream'],**decompose(r)) for r in rows]
    sums=lambda rr,k:sum(r[k] for r in rr)
    totals={k:sums(rows,k) for k in ('pixels','changed','repaired','harmed','both_wrong_changed')}
    totals.update(change_fraction=totals['changed']/totals['pixels'],original_accuracy_on_changed=totals['harmed']/totals['changed'],corrected_accuracy_on_changed=totals['repaired']/totals['changed'],net_accuracy_on_changed=(totals['repaired']-totals['harmed'])/totals['changed'])
    dirs=[];groups=[]
    for k in ('10','21'):
        keys=('accuracy_contribution_'+k,'nll_contribution_'+k)
        ctx=[{key:mean(r[key] for r in decomposed if r['context']==i) for key in keys} for i in range(16)]
        dirs.append(dict(direction=k,repaired=sums(decomposed,'repaired_'+k),harmed=sums(decomposed,'harmed_'+k),weighted_accuracy_contribution=mean(r[keys[0]] for r in decomposed),weighted_nll_contribution=mean(r[keys[1]] for r in decomposed),joint_positive_contexts=sum(r[keys[0]]>0 and r[keys[1]]<0 for r in ctx)))
        for key in ('fold','stream','source_step','labeled_images','condition'):
            for value in sorted({r[key] for r in rows}):
                rr=[r for r in decomposed if r[key]==value]
                groups.append(dict(direction=k,group=f'{key}={value}',weighted_accuracy_contribution=mean(r[keys[0]] for r in rr),weighted_nll_contribution=mean(r[keys[1]] for r in rr)))
    trueclass=dict(class0=dirs[0]['repaired'],class1=dirs[1]['repaired']-dirs[0]['harmed'],class2=-dirs[1]['harmed'])
    assert sum(trueclass.values())==totals['repaired']-totals['harmed']
    folds=[]
    for fold in (0,1):
        rr=[r for r in rows if r['fold']==fold];changed=sums(rr,'changed')
        folds.append(dict(fold=fold,selected=sums(rr,'pixels'),changed=changed,repaired=sums(rr,'repaired'),harmed=sums(rr,'harmed'),changed_fraction=changed/sums(rr,'pixels'),repair_fraction=sums(rr,'repaired')/changed,base_accuracy_mean=mean(r['base_correct']/r['pixels'] for r in rr),base_nll_mean=mean(r['base_nll']/r['pixels'] for r in rr)))
    joint=lambda r:r['weighted_net_accuracy']>0 and r['weighted_nll_delta']<0
    lookup={(r['context'],r['stream']):r for r in rows}
    return dict(status='POSTHOC_EXISTING_AGGREGATES_ONLY',model_calls=0,image_reads=0,query_calls=0,optimizer_updates=0,original_gate_unchanged=True,totals=totals,directions=dirs,direction_groups=groups,direction_rows=decomposed,true_class_unweighted_net_correct_counts=trueclass,both_wrong_direction_allocation='NOT_IDENTIFIABLE_FROM_SAVED_AGGREGATES',folds=folds,stream_joint_decision_agreement=sum(joint(lookup[i,3])==joint(lookup[i,4]) for i in range(16)),unweighted_mean_accuracy_delta=mean(r['net_accuracy'] for r in rows),unweighted_mean_nll_delta=mean(r['nll_delta'] for r in rows))


if __name__=='__main__':
    selfcheck();root=Path(__file__).parent;out=analyze(json.loads((root/'QUALITY_RESULTS.json').read_text()))
    with (root/'POSTHOC_ANALYSIS.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(totals=out['totals'],directions=out['directions'],true_class_net=out['true_class_unweighted_net_correct_counts'],folds=out['folds'],stream_agreement=out['stream_joint_decision_agreement'])))
