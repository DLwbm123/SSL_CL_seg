"""One terminal, zero-model-forward audit for D/DP and read-only B/W controls."""
import collections,json,subprocess
from pathlib import Path
import torch
from .decision_budget import SEED,METHODS,CAPS,HEAD,caps_for
from .runtime import exact,finite
from .single_target_audit import rows
from .temporal_spectrum_p0 import damage
from ..main_style_ssl_v1.runtime import write

def read(p):return json.loads(Path(p).read_text())

def main(cfg):
    torch.set_num_threads(4);root=Path(cfg['root']);oldroot=Path(cfg['neighbor_root']);assert not (root/'AUDIT.json').exists()
    status=read(root/'STATUS.json');assert status['status'] in ('COMPLETE','P0_STOP');full=status['status']=='COMPLETE'
    pg=read(root/('DP_RECOVERY_LAUNCH.private.json' if (root/'DP_RECOVERY_LAUNCH.private.json').exists() else 'RESUME_LAUNCH.private.json' if (root/'RESUME_LAUNCH.private.json').exists() else 'LAUNCH.private.json'))['process_group']
    for line in subprocess.check_output(['ps','-eo','pid,pgid,stat'],text=True).splitlines()[1:]:
        f=line.split();assert int(f[1])!=pg or f[2].startswith('Z')
    for p in [*root.glob('*LAUNCH.private.json'),root/'WORKER.private.json',*root.glob('ACTIVE_*.private.json')]:
        v=read(p);s=Path('/proc')/str(v['pid'])/'stat';assert not s.exists() or s.read_text().split()[21]!=str(v['birth']) or s.read_text().split()[2]=='Z'
    if (root/'QUALIFICATION_CHECK_REPAIR.json').exists():
        repair=read(root/'QUALIFICATION_CHECK_REPAIR.json');assert repair['status']=='PASS' and repair['new_model_calls']==repair['new_optimizer_steps']==0
        assert read(root/'qualification_check_failure'/'EXIT_20271401_D_qualify.json')['exit_code']==1
    assert not list(root.glob('FAILURE*')) and all(read(p)['exit_code']==0 for p in [root/'EXIT.json',*root.glob('EXIT_*.json')])
    p0=read(root/'P0_RESULT.json');expected={};recovered=cfg.get('approved_dp_failure_recovery',False)
    if recovered:
        authorization=read(root/'DP_RECOVERY_AUTHORIZATION.private.json');recovery=read(root/'DP_RECOVERY_RESULT.json')
        assert authorization['extra_optimizer']==1 and authorization['extra_native']==10 and authorization['extra_reverse']==5
        assert recovery['status']=='PASS' and recovery['resume_step']==1242 and recovery['total_DP_optimizer_calls']==2001
        assert read(root/'dp_step_failure'/'EXIT_20271401_DP_train.json')['exit_code']==1
        assert read(root/'dp_step_failure'/'budget_counter.json')['total']==3371
        assert read(root/'FLOAT32_CAP_FIX_SYNTHETIC.json')['status']=='PASS'

    if full:
        assert p0['status']=='PASS' and read(root/'DETACH_CHECK.json')['status']=='PASS'
        expected.update({'DETACH_D:detach':32,'DETACH_DP:detach':32})
        expected.update({f'{SEED}_{a}:{phase}':n for a in METHODS for phase,n in [('engineering',32),('formal',2000)]})
    else:assert p0['status']=='STOP' and not (root/'jobs').exists()
    if full and recovered:expected[f'{SEED}_DP:formal']=2001
    count=read(root/'budget_counter.json') if (root/'budget_counter.json').exists() else dict(total=0,optimizer=0,functional=0,pending=0,by_phase={})
    assert count==dict(total=sum(expected.values()),optimizer=sum(expected.values()),functional=0,pending=0,by_phase=expected)
    events=rows(root/'budget_events.jsonl') if (root/'budget_events.jsonl').exists() else []
    assert collections.Counter(e['key'] for e in events if e['kind']=='optimizer')==expected and sum(e['kind']=='reserved' for e in events)==sum(expected.values())
    calls=read(root/'calls_counter.json');assert calls['pending'] is None and calls['caps']==caps_for(cfg)
    totals={k:collections.Counter() for k in ('reserved','executed')}
    for e in rows(root/'calls_events.jsonl'):totals[e['action']][e['kind']]+=e['samples']
    assert totals['reserved']==totals['executed'] and all(totals['executed'][k]==calls['executed'].get(k,0)<=v for k,v in caps_for(cfg).items())
    assert read(root/'REUSE_COMPATIBILITY.json')['status']=='PASS' and read(root/'DECISION_SELF_CHECK.json')['status']=='PASS'
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    original=load(cfg['source_checkpoint'])['ema'];reference=rows(cfg['original_source_rows']);keys=lambda rr:{(r['patient'],r['phase']) for r in rr}
    source_score=sum(sum(r['dice'])/len(r['dice']) for r in reference)/len(reference)*100;assert abs(source_score-80.36740663770262)<1e-9
    metrics={};values={};tail={};contrasts={};gates={};control={};composite=False
    if full:
        for arm in ('B','W','D','DP'):
            out=(oldroot if arm in ('B','W') else root)/'jobs'/f'{SEED}_{arm}'
            ini=load(out/'initial.pt');end=load(out/'final.pt');finite(end)
            assert exact(ini['model'],original) and exact(ini['ema'],original) and not ini['optimizer']['state']
            assert exact(end,load(out/'latest.pt')) and end['optimizer']['state'] and end['step']==end['physical_updates']==2000 and end['reads']==[2000,0]
            assert torch.equal(end['model'][HEAD],original[HEAD]) and torch.equal(end['ema'][HEAD],original[HEAD])
            assert all(torch.equal(v,end['model'][k]) and torch.equal(v,end['ema'][k]) for k,v in original.items() if k.endswith(('sigma.weight','grad_update')))
            assert read(out/'EXIT.json')['exit_code']==read(out/'EVALUATION_EXIT.json')['exit_code']==0
            assert len(rows(out/'physical_updates.jsonl'))==2000 and all(k in end for k in ('torch_rng','cuda_rng'))
            assert all(g['lr']==.001 and g['momentum']==.9 and g['nesterov'] for g in end['optimizer']['param_groups'])
            metrics[arm]=read(out/'RESULTS.json');values[arm]={}
            for domain,n in [('Siemens',23),('Philips',31)]:
                values[arm][domain]={mode:rows(out/(prefix+domain+'_val.jsonl')) for mode,prefix in [('student','private_'),('EMA','private_ema_')]}
                ref=reference if domain=='Siemens' else rows(oldroot/'jobs'/f'{SEED}_B/private_Philips_val.jsonl')
                for mode,rr in values[arm][domain].items():
                    assert len(rr)==len(keys(rr))==2*n and keys(rr)==keys(ref)
                    score=sum(sum(v['dice'])/len(v['dice']) for v in rr)/len(rr)
                    reported=(metrics[arm] if mode=='student' else metrics[arm]['EMA_diagnostic'])[domain]['primary_dice'];assert abs(score-reported)<1e-10
            tail[arm]={mode:damage(reference,values[arm]['Siemens'][mode]) for mode in ('student','EMA')}
            if arm in METHODS:
                assert read(root/'qualification'/f'{SEED}_{arm}'/'FIXED_HEAD_PREFLIGHT.json')['status']=='PASS'
                assert read(root/'qualification'/f'{SEED}_{arm}'/'FIXED_HEAD_TERMINAL.json')['status']=='PASS'
                assert [v['step'] for v in rows(out/'gradient_diagnostics.jsonl')]==[0,999,1999]
                steps=rows(out/'step_control.jsonl');assert len(steps)==2000 and [v['step'] for v in steps]==list(range(2000))
                assert all(v['candidate_momentum_preserved'] and v['applied']==(arm=='DP') and v['correction_ratio']<=.250001 and v['solver_kkt']<=1e-5 for v in steps)
                if arm=='DP':
                    assert all(v['actual_correction_ratio']<=(.25 if recovered and v['step']>=1242 else .25001) for v in steps)
                    if recovered:
                        assert all(v['prediction_uses_actual_float32_displacement'] for v in steps[1242:])
                        old=load(root/'dp_step_failure'/'latest_committed_1242.pt');entry=load(out/'continuation_after_step1242'/'entry.pt');assert exact(old,entry)
                        assert len(rows(out/'continuation_after_step1242'/'physical_updates.jsonl'))==758
                control[arm]=dict(steps=2000,applied_steps=sum(v['applied'] for v in steps),nonzero_correction_steps=sum(v['applied'] and v['correction_norm']>0 for v in steps),mean_correction_ratio=sum(v.get('actual_correction_ratio',0.) for v in steps)/2000,max_predicted_violation_after=max(max(v['predicted_violation_after']) for v in steps),budget_guarantee=False)
        for candidate in METHODS:
            contrasts[candidate]={other:{domain:damage(values[other][domain]['EMA'],values[candidate][domain]['EMA']) for domain in ('Siemens','Philips')} for other in ('B','W','D') if other!=candidate}
            old=tail[candidate]['EMA'];b=contrasts[candidate]['B'];m=metrics[candidate]['EMA_diagnostic']
            checks=dict(old_absolute=m['Siemens']['primary_dice']*100>=source_score,old_interval=old['paired']['patient_bootstrap95_pp'][0]>=-.30,tail_over_1=old['decline_over_pp']['1']<=2,tail_over_2=old['decline_over_pp']['2']==0,tail_p10=old['p10_pp']>=-.50,tail_vs_controls=all(old['decline_over_pp'][k]<=tail[a]['EMA']['decline_over_pp'][k] for a in ('B','W') for k in ('1','2')),new_absolute=m['Philips']['primary_dice']*100>=83,new_noninferior_B=b['Philips']['mean_pp']>=-.20 and b['Philips']['paired']['patient_bootstrap95_pp'][0]>=-.20)
            gates[candidate]=dict(checks=checks,joint_tail_new_pass=all(checks.values()))
        for a in ('W','D'):
            for d in ('Siemens','Philips'):
                v=contrasts['DP'][a][d];gates['DP']['checks'][f'noninferior_{a}_{d}']=v['mean_pp']>=-.20 and v['paired']['patient_bootstrap95_pp'][0]>=-.20
        gates['DP']['checks'].update(increment_B_old=contrasts['DP']['B']['Siemens']['mean_pp']>=.30,increment_W_old=contrasts['DP']['W']['Siemens']['mean_pp']>=.15,increment_D_old=contrasts['DP']['D']['Siemens']['mean_pp']>=.15)
        composite=all(gates['DP']['checks'].values())
        assert calls['executed']['formal_native']==(40010 if recovered else 40000) and calls['executed']['formal_reverse']==(20023 if recovered else 20018) and calls['executed']['evaluation_native']==5784
    used=int(subprocess.check_output(['du','-s','-B1',str(root)]).split()[0])+2000000;assert used<4000000000 and status['resources']['elapsed_seconds']<28800
    audit=dict(status='PASS',workers_retired=True,audit_model_forwards=0,cost=count,calls=calls,allocated_bytes_including_metadata=used,elapsed_seconds=status['resources']['elapsed_seconds'],old_B_W_reused=True,DP_recovery=read(root/'DP_RECOVERY_RESULT.json') if recovered else None,old_A_protocol_failure_not_erased=True,qualification_check_repair=read(root/'QUALIFICATION_CHECK_REPAIR.json') if (root/'QUALIFICATION_CHECK_REPAIR.json').exists() else None)
    result=dict(audit=audit,P0=p0,formal_started=full,metrics=metrics,contrasts=contrasts,old_source_damage=tail,gates=gates,control=control,composite_pass=composite,primary='final EMA .999',target_seed=SEED,independent_test=False,development_validation_reused=True,training_stability_established=False)
    write(root/'PUBLIC_SUMMARY.json',result);write(root/'AUDIT.json',audit)
    (root/'REPORT_ZH.md').write_text('# 分类别预算完整结果\n\n工程审计PASS，单种子科学'+('PASS' if composite else 'FAIL/STOP')+'。完整四配置指标、类别、患者配对区间和尾部见PUBLIC_SUMMARY.json。无独立测试/训练稳定性结论。等待用户决定，不自动后继。\n')
