"""Audit existing receipts and export anonymous aggregates; no model/data access."""
import csv,json,sys
from pathlib import Path


def main(raw,out):
    raw,out=Path(raw),Path(out);out.mkdir(exist_ok=True)
    read=lambda name:json.loads((raw/name).read_text())
    lines=lambda name:[json.loads(v) for v in (raw/name).read_text().splitlines()]
    result=read('RESULTS.json');at=lines('attribution.jsonl');metrics=read('METRICS.json')
    expected=dict(optimizer=27,main=25,golden=2,candidates=14,extra_reverse=104)
    assert result['status']=='COMPLETE' and result['costs']==expected
    assert read('EXIT.json')['exit_code']==0 and not (raw/'FAILURE.json').exists()
    for file,n,key in [('physical_updates.jsonl',27,'physical'),('attempts.jsonl',27,'attempt'),('candidate_evaluations.jsonl',14,'index'),('reverse_calls.jsonl',104,'index')]:
        assert [v[key] for v in lines(file)]==list(range(1,n+1))
    assert [v['step'] for v in at]==list(range(1,26))
    assert [v['step'] for v in metrics]==list(range(26))
    ledger=lines('physical_updates.jsonl')
    assert [v['step'] for v in ledger if v['phase']=='main']==list(range(1,26))
    assert [v['step'] for v in ledger if v['phase']=='golden']==[1,25]
    checks=read('GOLDEN.json');assert all(v['complete_state_equal'] and v['loss_equal'] for v in checks)
    assert read('HISTORICAL.json')['exact_metrics_equal']
    cs=lines('candidate_evaluations.jsonl')
    assert [(v['step'],v['alpha']) for v in cs if v['kind']=='scaled']==[(k,a) for k in [1,2,5,10,25] for a in [1/16,1/4]]
    assert [(v['anchor'],v['cell']) for v in cs if v['kind']=='state_noise_cross']==[(k,c) for k in [2,25] for c in ['01','10']]
    maxclosure=0.
    for prev,nxt,a in zip(metrics,metrics[1:],at):
        for i,key in enumerate(['ce','class2_ce']):
            assert abs(nxt[key]-prev[key]-a['actual'][i])<1e-12
            assert abs(a['actual'][i]-a['linear'][i]-a['remainder'][i])<1e-12
        tr=a['transitions']['all_class2']['aggregate']
        assert [sum(tr[f'{i}_to_{j}'] for j in range(4)) for i in range(4)]==prev['class2_confusion']
        assert [sum(tr[f'{i}_to_{j}'] for i in range(4)) for j in range(4)]==nxt['class2_confusion']
        for group in a['margins']['actual']:
            for scope in a['margins']['actual'][group]:
                means={t:a['margins'][t][group][scope]['mean'] for t in a['margins']}
                res=abs(means['actual']-sum(means[t] for t in ['linear','remainder','switch']))
                maxclosure=max(maxclosure,res);assert res<1e-5 and means['switch']<=1e-7
                if scope=='aggregate':
                    for key,val in a['transitions'][group][scope].items():assert val==sum(a['transitions'][group][f'image_{i}'][key] for i in range(2))
    for cross in read('CROSS.json'):
        for term in ['Q','A','R']:
            c=cross['contrast'][term]
            assert all(abs(t-a-b-i)<1e-10 for t,a,b,i in zip(c['total'],c['state'],c['noise'],c['interaction']))
    audit=dict(status='PASS',actual_costs=expected,all_four_ledgers_contiguous=True,optimizer_attempts_equal_receipts=True,no_failure=True,exit_code=0,historical_metrics_exact=True,golden_full_state_exact=True,pure_adam_actual_delta_max_error=result['candidate_max_error'],CE_telescoping=True,all_GT2_transition_rows_match_metrics=True,per_image_transition_counts_sum_to_aggregate=True,margin_max_aggregate_closure=maxclosure,pixel_margin_closure_max=result['margin_max_closure'],switch_terms_nonpositive=True,original_ledger_unchanged=result['original_ledger_unchanged'],scope='two existing labeled source images; not generalization or continual-learning evidence')
    result={k:v for k,v in result.items() if k not in ['started','finished']}
    result['runtime_seconds']=read('RESULTS.json')['finished']-read('RESULTS.json')['started']
    for name,obj in [('RESULTS',result),('AUDIT',audit),('ATTRIBUTION',at),('METRICS',metrics),('CROSS',read('CROSS.json')),('SCALED',lines('scaled.jsonl')),('HISTORICAL',read('HISTORICAL.json')),('GOLDEN',checks),('RESTORE',read('RESTORE.json'))]:
        (out/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
    def table(name,rows):
        with (out/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    table('TRAJECTORY',[dict(step=m['step'],mean_CE=m['ce'],GT2_CE=m['class2_ce'],mean_Dice_pct=100*sum(map(sum,m['dice']))/6,**{f'GT2_pred_{i}':v for i,v in enumerate(m['class2_confusion'])}) for m in metrics])
    table('CE_ATTRIBUTION',[dict(step=a['step'],**{term+('_all' if i==0 else '_GT2'):a[term][i] for term in ['actual','linear','remainder'] for i in [0,1]}) for a in at])
    table('MARGIN_SUMS',[dict(cohort=g,scope=s,count=at[0]['margins']['actual'][g][s]['count'],**{term:sum(a['margins'][term][g][s]['mean'] for a in at) for term in ['actual','linear','remainder','switch']},**{key:sum(a['transitions'][g][s][key] for a in at) for key in ['2_to_0','2_to_1','2_to_3','0_to_2','1_to_2','3_to_2']}) for g in at[0]['margins']['actual'] for s in ['aggregate','image_0','image_1']])
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main(*sys.argv[1:])
