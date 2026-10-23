"""Single zero-forward terminal audit, including the legitimate P0 stop endpoint."""
import collections,json,subprocess
from pathlib import Path
import torch
from .compatible_anchor import SEEDS,METHODS,CAPS,HEAD
from .runtime import exact,finite
from .single_target_audit import rows
from .temporal_spectrum_p0 import damage
from ..main_style_ssl_v1.runtime import write

def read(p):return json.loads(Path(p).read_text())

def main(cfg):
    torch.set_num_threads(4);root=Path(cfg['root']);assert not (root/'AUDIT.json').exists()
    status=read(root/'STATUS.json');assert status['status'] in ('P0_STOP','COMPLETE')
    pg=read(root/'LAUNCH.private.json')['process_group']
    for line in subprocess.check_output(['ps','-eo','pid,pgid,stat'],text=True).splitlines()[1:]:
        f=line.split();assert int(f[1])!=pg or f[2].startswith('Z')
    for p in [root/'LAUNCH.private.json',root/'WORKER.private.json',*root.glob('ACTIVE_*.private.json')]:
        v=read(p);s=Path('/proc')/str(v['pid'])/'stat';assert not s.exists() or s.read_text().split()[21]!=str(v['birth']) or s.read_text().split()[2]=='Z'
    assert all(read(p)['exit_code']==0 for p in [root/'EXIT.json',*root.glob('EXIT_*.json')])
    assert not list(root.glob('FAILURE*'))
    p0=read(root/'P0_RESULT.json');full=status['status']=='COMPLETE'
    expected={'P0_ENGINEERING:engineering':32,'PROBE:probe':256}
    if full:
        assert p0['status']=='PASS' and read(root/'DETACH_CHECK.json')['status']=='PASS'
        expected.update({'DETACH_B:detach':32,'DETACH_C:detach':32})
        expected.update({f'{s}_{a}:{phase}':n for s in SEEDS for a in METHODS for phase,n in [('engineering',32),('formal',2000)]})
    else:assert p0['status']=='STOP' and not (root/'jobs').exists()
    count=read(root/'budget_counter.json');assert count==dict(total=sum(expected.values()),optimizer=sum(expected.values()),functional=0,pending=0,by_phase=expected)
    ev=rows(root/'budget_events.jsonl');assert collections.Counter(e['key'] for e in ev if e['kind']=='optimizer')==expected
    assert sum(e['kind']=='reserved' for e in ev)==sum(expected.values())
    calls=read(root/'calls_counter.json');assert calls['pending'] is None and calls['caps']==CAPS
    totals={k:collections.Counter() for k in ('reserved','executed')}
    for e in rows(root/'calls_events.jsonl'):totals[e['action']][e['kind']]+=e['samples']
    assert totals['reserved']==totals['executed'] and all(totals['executed'][k]==calls['executed'][k]<=v for k,v in CAPS.items())
    assert read(root/'ANCHOR_SELF_CHECK.json')['status']=='PASS'
    assert read(root/'MARGIN_SELF_CHECK.json')['status']=='PASS' and read(root/'PROJECTION_SELF_CHECK.json')['status']=='PASS'
    selfcheck=read(root/'SELF_CHECK.json');assert selfcheck['status']=='PASS' and selfcheck['source_SVD']==1 and selfcheck['synthetic_SVD']<=16 and selfcheck['synthetic_reverse']<=32
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    original=load(cfg['source_checkpoint']);original=original['ema'];metrics={};gates={};contrasts={};tail={}
    source_rows=rows(cfg['original_source_rows']);source_score=sum(sum(r['dice'])/len(r['dice']) for r in source_rows)/len(source_rows)*100
    assert abs(source_score-80.36740663770262)<1e-9
    keys=lambda rr:{(r['patient'],r['phase']) for r in rr}
    assert len(source_rows)==46 and len(keys(source_rows))==46
    if full:
        for seed in SEEDS:
            rr={};mm={};tt={}
            for arm in METHODS:
                group=f'{seed}_{arm}';out=root/'jobs'/group;ini=load(out/'initial.pt');end=load(out/'final.pt');finite(end)
                assert exact(end,load(out/'latest.pt')) and exact(ini['model'],original) and exact(ini['ema'],original)
                assert not ini['optimizer']['state'] and end['optimizer']['state'] and end['step']==end['physical_updates']==2000 and end['reads']==[2000,0]
                assert len(rows(out/'physical_updates.jsonl'))==2000 and all(k in end for k in ('torch_rng','cuda_rng'))
                assert read(root/'qualification'/group/'ENGINEERING.json')['actual_calls']==32
                assert read(out/'EXIT.json')['exit_code']==read(out/'EVALUATION_EXIT.json')['exit_code']==0
                if arm!='A':assert torch.equal(end['model'][HEAD],original[HEAD]) and torch.equal(end['ema'][HEAD],original[HEAD])
                assert all(torch.equal(v,end['model'][k]) and torch.equal(v,end['ema'][k]) for k,v in original.items() if k.endswith(('sigma.weight','grad_update')))
                assert all(g['lr']==.001 for g in end['optimizer']['param_groups'])
                assert [x['step'] for x in rows(out/'gradient_diagnostics.jsonl')]==[0,999,1999]
                mm[arm]=read(out/'RESULTS.json');rr[arm]={}
                for domain,n in [('Siemens',23),('Philips',31)]:
                    rr[arm][domain]={mode:rows(out/(prefix+domain+'_val.jsonl')) for mode,prefix in [('student','private_'),('EMA','private_ema_')]}
                    reference=source_rows if domain=='Siemens' else rows(Path(cfg['comparison_root'])/'jobs/A/private_Philips_val.jsonl')
                    for mode,values in rr[arm][domain].items():
                        assert len(values)==len(keys(values))==2*n and keys(values)==keys(reference)
                        score=sum(sum(r['dice'])/len(r['dice']) for r in values)/len(values)
                        reported=mm[arm][domain]['primary_dice'] if mode=='student' else mm[arm]['EMA_diagnostic'][domain]['primary_dice']
                        assert abs(score-reported)<1e-10
                tt[arm]={mode:damage(source_rows,rr[arm]['Siemens'][mode]) for mode in ('student','EMA')}
            comp={other:{domain:damage(rr[other][domain]['EMA'],rr['C'][domain]['EMA']) for domain in ('Siemens','Philips')} for other in ('B','R','N','H')}
            old=tt['C']['EMA'];b=comp['B'];deployed=mm['C']['EMA_diagnostic']
            checks=dict(old_absolute=100*deployed['Siemens']['primary_dice']>=source_score,old_interval=old['paired']['patient_bootstrap95_pp'][0]>=-.30,new_absolute=100*deployed['Philips']['primary_dice']>=83.,new_vs_B=b['Philips']['mean_pp']>=-.20 and b['Philips']['paired']['patient_bootstrap95_pp'][0]>=-.20,tail_over_1=old['decline_over_pp']['1']<=2,tail_over_2=old['decline_over_pp']['2']==0,tail_p10=old['p10_pp']>=-.50,tail_vs_B=all(old['decline_over_pp'][k]<=tt['B']['EMA']['decline_over_pp'][k] for k in ('1','2')))
            checks.update({f'vs_{other}_old_noninferior':comp[other]['Siemens']['mean_pp']>=-.20 and comp[other]['Siemens']['paired']['patient_bootstrap95_pp'][0]>=-.20 for other in ('R','N','H')})
            checks.update({f'vs_{other}_new_noninferior':comp[other]['Philips']['mean_pp']>=-.20 and comp[other]['Philips']['paired']['patient_bootstrap95_pp'][0]>=-.20 for other in ('R','N','H')})
            gates[str(seed)]=dict(deployment=all(checks.values()),checks=checks,increment_point=b['Siemens']['mean_pp']>0)
            metrics[str(seed)]=mm;contrasts[str(seed)]=comp;tail[str(seed)]=tt
        increment_mean=sum(v['B']['Siemens']['mean_pp'] for v in contrasts.values())/3
        selectivity_mean=sum(v['R']['Siemens']['mean_pp'] for v in contrasts.values())/3
        recovery_mean=sum(v['N']['Philips']['mean_pp'] for v in contrasts.values())/3
        direction_mean=sum(v['H']['Philips']['mean_pp'] for v in contrasts.values())/3
        composite=recovery_mean>=.30 and direction_mean>=.15 and all(v['N']['Philips']['mean_pp']>0 and v['H']['Philips']['mean_pp']>0 for v in contrasts.values()) and all(v['deployment'] and v['increment_point'] for v in gates.values()) and increment_mean>=.30 and selectivity_mean>=.15 and all(v['R']['Siemens']['mean_pp']>0 for v in contrasts.values())
        assert calls['executed']['formal_native']==180000 and calls['executed']['formal_reverse']==90045 and calls['executed']['evaluation_native']==43380
    else:composite=False;increment_mean=selectivity_mean=recovery_mean=direction_mean=None
    used=int(subprocess.check_output(['du','-s','-B1',str(root)]).split()[0])+2_000_000;assert used<4_000_000_000 and status['resources']['elapsed_seconds']<28800
    audit=dict(status='PASS',cost=count,calls=calls,workers_retired=True,source_entry_exact=full,audit_model_forwards=0,allocated_bytes_including_metadata=used,elapsed_seconds=status['resources']['elapsed_seconds'])
    result=dict(audit=audit,P0=p0,formal_started=full,metrics=metrics,contrasts=contrasts,old_source_damage=tail,gates=gates,increment_mean_old_pp=increment_mean,C_minus_R_mean_old_pp=selectivity_mean,C_minus_N_mean_new_pp=recovery_mean,C_minus_H_mean_new_pp=direction_mean,composite_pass=composite,primary='final EMA .999',independent_source_replication=False,independent_test_access=False,development_validation_reused=True)
    write(root/'PUBLIC_SUMMARY.json',result)
    lines=['# CE相容参数锚完整方法结果','','工程审计 PASS；完整科学判定 '+('PASS' if composite else 'FAIL/STOP')+'。主终点为固定 EMA .999，学生结果全部保留。','','```json',json.dumps({k:v for k,v in result.items() if k not in ('metrics','contrasts','old_source_damage')},ensure_ascii=False,indent=2),'```','','完整绝对、配对和尾部见 PUBLIC_SUMMARY.json。共享源与开发患者；无独立测试结论。P0 STOP 只表示冻结门禁未过，不等于正式性能阴性。不补种子、延长探针、扫描系数或修改门槛。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');write(root/'AUDIT.json',audit)
