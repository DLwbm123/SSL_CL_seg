"""Receipt-only audit/export, without data, models or optimizer access."""
import json,csv,sys
from pathlib import Path


def main(raw,out):
    raw,out=Path(raw),Path(out);out.mkdir(exist_ok=True)
    read=lambda n:json.loads((raw/n).read_text())
    lines=lambda n:[json.loads(l) for l in (raw/n).read_text().splitlines()]
    r=read('RESULTS.json');c=r['costs']
    assert r['status']=='COMPLETE' and read('EXIT.json')['exit_code']==0 and not (raw/'FAILURE.json').exists()
    assert c['optimizer']==54 and c['main']==50 and c['golden']==4 and c['extra_reverse']==10 and c['candidates']<=143
    for f,key,n in [('physical_updates.jsonl','physical',54),('attempts.jsonl','attempt',54),('candidate_evaluations.jsonl','index',c['candidates']),('reverse_calls.jsonl','index',10)]:
        assert [v[key] for v in lines(f)]==list(range(1,n+1))
    candidates=lines('candidate_evaluations.jsonl');kinds={k:sum(v['kind']==k for v in candidates) for k in set(v['kind'] for v in candidates)}
    assert kinds['parameter_adam_cross']==8 and kinds['joint_readout']==24 and kinds['posterior_state']==3 and kinds['schedule_endpoint']==27
    assert kinds['guard_trial']<=81
    qual=read('CROSS_QUALIFICATION.json');assert qual['status']=='PASS' and qual['costs']['optimizer']==0 and qual['costs']['candidates']==8
    state_audit=read('STATE_AUDIT.json');assert state_audit['G_D_final_complete_state_exact'] and state_audit['semantic_parameters_equal_initial'] and state_audit['prior_buffers_unchanged']
    cross=read('CROSS.json');assert [(v['anchor'],v['noise']) for v in cross]==[(2,1),(2,2),(25,1),(25,25)]
    for row in cross:
        for term,v in row['contrasts'].items():assert all(abs(t-p-a-i)<1e-10 for t,p,a,i in zip(v['total'],v['parameters'],v['adam'],v['interaction']))
        assert all(v['source']==('new_candidate' if k in ['01','10'] else 'previous_state_noise_result') for k,v in row['cells'].items())
    base={};steps={};metrics={};golden={}
    ledger=lines('physical_updates.jsonl')
    for arm in ['G','D']:
        steps[arm]=lines(arm+'/steps.jsonl');metrics[arm]=lines(arm+'/metrics.jsonl');golden[arm]=read(arm+'/GOLDEN.json')
        assert [v['step'] for v in steps[arm]]==list(range(1,26)) and [v['step'] for v in metrics[arm]]==list(range(26))
        assert [v['step'] for v in ledger if v['arm']==arm and v['phase']=='main']==list(range(1,26))
        assert [v['step'] for v in ledger if v['arm']==arm and v['phase']=='golden']==[1,25]
        assert all(v['full_state_and_decision_equal'] for v in golden[arm])
        initial=metrics[arm][0]['vector'];base[arm]=initial
        for step,previous,m in zip(steps[arm],metrics[arm],metrics[arm][1:]):
            assert step['metrics']==m['metrics'] and step['vector']==m['vector']
            assert step['alpha'] in [0.,.0625,.25,1.]
            if step['alpha']==0:assert step['actual_semantic_delta_norm']==0 and m['vector']==previous['vector']
            if arm=='G':
                assert step['guard_pass'] and all(v<=b+1e-7+1e-6*abs(b) for v,b in zip(step['vector'],initial))
                assert all(not v['accepted'] for v in step['trials'][:-1])
                assert (step['alpha']==0 and not step['trials'][-1]['accepted']) or (step['alpha']>0 and step['trials'][-1]['accepted'] and step['trials'][-1]['alpha']==step['alpha'])
            for channel in ['mean','stochastic']:
                tr=m['transitions'][channel]['all_class2']['aggregate'];before=previous['metrics'][channel]['class2_confusion'];after=m['metrics'][channel]['class2_confusion']
                assert [sum(tr[f'{i}_to_{j}'] for j in range(4)) for i in range(4)]==before
                assert [sum(tr[f'{i}_to_{j}'] for i in range(4)) for j in range(4)]==after
                for cohort,scopes in m['transitions'][channel].items():
                    for key,value in scopes['aggregate'].items():assert value==scopes['image_0'][key]+scopes['image_1'][key]
    assert base['G']==base['D'] and [v['alpha'] for v in steps['G']]==[v['alpha'] for v in steps['D']]
    audit=dict(status='PASS',costs=c,candidate_categories=kinds,four_ledgers_contiguous=True,attempts_equal_optimizer_receipts=all(a['arm']==b['arm'] and a['phase']==b['phase'] and a['step']==b['step'] for a,b in zip(lines('attempts.jsonl'),ledger)),zero_update_cross_precedes_training=True,diagonal_checks=qual,complete_restore=read('RESTORE.json'),all_four_golden_comparisons_exact=True,paired_alpha_schedule_equal=True,all_G_steps_satisfy_fixed_baseline_constraint=True,zero_alpha_semantic_displacement_exactly_zero=True,GT2_transitions_match_metrics=True,per_image_transitions_sum_to_aggregate=True,original_ledgers_unchanged=r['original_ledgers_unchanged'],final_checkpoint_audit=state_audit,no_failure=True,exit_code=0)
    r=dict(r);r['runtime_seconds']=r.pop('finished')-r.pop('started')
    for name,obj in [('RESULTS',r),('AUDIT',audit),('CROSS',cross),('DENSITY',read('DENSITY.json')),('POSTERIOR',read('POSTERIOR.json')),('STEPS',steps),('METRICS',metrics),('GOLDEN',golden),('STATE_AUDIT',state_audit)]:
        (out/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
    def table(name,rows):
        with (out/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    table('TRAJECTORY',[dict(arm=a,step=v['step'],channel=ch,ce=m['ce'],GT2_ce=m['class2_ce'],macro_Dice_pct=100*sum(map(sum,m['dice']))/6,**{f'GT2_pred_{i}':n for i,n in enumerate(m['class2_confusion'])}) for a,vs in metrics.items() for v in vs for ch,m in v['metrics'].items()])
    table('ALPHAS',[dict(arm=a,step=v['step'],alpha=v['alpha'],guard_pass=v['guard_pass'],actual_semantic_delta_norm=v['actual_semantic_delta_norm'],nominal_semantic_delta_norm=v['nominal_semantic_delta_norm'],nonsemantic_delta_norm=v['nonsemantic_delta_norm'],loss_before=v['loss_before']) for a,vs in steps.items() for v in vs])
    table('CROSS_CONTRASTS',[dict(anchor=v['anchor'],noise=v['noise'],term=term,metric=metric,**{key:vv[i] for key,vv in terms.items()}) for v in cross for term,terms in v['contrasts'].items() for i,metric in enumerate(['all_CE','GT2_CE'])])
    names=[f'{ch}_image{i}_{cl}' for ch in ['mean','stochastic'] for i in range(2) for cl in ['all','C1','C2','C3']]
    failures=[]
    for alpha in [1.,.25,.0625]:
        for i,name in enumerate(names):
            ds=[next(t for t in row['trials'] if t['alpha']==alpha)['vector'][i]-base['G'][i] for row in steps['G']]
            failures.append(dict(alpha=alpha,constraint=name,violations=sum(v>1e-7+1e-6*abs(base['G'][i]) for v in ds),min_delta=min(ds),max_delta=max(ds),first_delta=ds[0],last_delta=ds[-1]))
    table('GUARD_FAILURES',failures)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main(*sys.argv[1:])
