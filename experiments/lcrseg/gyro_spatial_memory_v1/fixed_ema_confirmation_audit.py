"""One final zero-forward audit for the frozen fixed EMA deployment confirmation."""
import collections,json,subprocess
from pathlib import Path
import numpy as np,torch
from .runtime import exact,finite
from .single_target_audit import rows
from .temporal_spectrum_p0 import damage
from .fixed_ema_confirmation import SEEDS,METHODS,CAPS,POLAR_STEPS,SOURCE_PERCENT,source_reference
from ..main_style_ssl_v1.runtime import write


def read(p):return json.loads(Path(p).read_text())


def main(cfg):
    torch.set_num_threads(4);root=Path(cfg['root']);assert not (root/'AUDIT.json').exists() and read(root/'STATUS.json')['status']=='COMPLETE'
    pg=read(root/'LAUNCH.private.json')['process_group']
    for line in subprocess.check_output(['ps','-eo','pid,pgid,stat'],text=True).splitlines()[1:]:
        f=line.split();assert int(f[1])!=pg or f[2].startswith('Z')
    for p in [root/'LAUNCH.private.json',root/'WORKER.private.json',*root.glob('ACTIVE_*.private.json')]:
        a=read(p);s=Path('/proc')/str(a['pid'])/'stat';assert not s.exists() or s.read_text().split()[21]!=str(a['birth']) or s.read_text().split()[2]=='Z'
    for p in [root/'EXIT.json',*root.glob('EXIT_*.json')]:assert read(p)['exit_code']==0
    expected={f'{s}_{m}:{p}':n for s in SEEDS for m in METHODS for p,n in [('engineering',32),('formal',2000)]}
    count=read(root/'budget_counter.json');assert count==dict(total=6096,optimizer=6096,functional=0,pending=0,by_phase=expected)
    ev=rows(root/'budget_events.jsonl');assert collections.Counter(x['key'] for x in ev if x['kind']=='optimizer')==expected and sum(x['kind']=='reserved' for x in ev)==6096
    calls=read(root/'calls_counter.json');assert calls['pending'] is None and calls['caps']==CAPS
    totals={a:collections.Counter() for a in ['reserved','executed']}
    for e in rows(root/'calls_events.jsonl'):totals[e['action']][e['kind']]+=e['samples']
    actual=dict(engineering_native=240,engineering_encoder=432,engineering_reverse=105,formal_native=12036,formal_encoder=288288,formal_reverse=6009,diagnostic_reverse=30,evaluation_native=8676)
    assert totals['reserved']==totals['executed']==calls['executed']==actual and all(actual[k]<=v for k,v in CAPS.items())
    check=read(root/'SELF_CHECK.json');assert check['numerical_SVD_calls']==10 and check['status']=='PASS' and check['EMA_formula_exact'] and check['EMA_fixed_and_integer_state_copy_exact']
    assert read(root/'SOURCE_REFERENCE_CHECK.json')==source_reference(cfg)
    pc=read(root/'polar_counter.json');assert pc==dict(executed_steps=POLAR_STEPS,executed_SVD=8*POLAR_STEPS,pending=None,cap_steps=POLAR_STEPS,cap_SVD=8*POLAR_STEPS)
    pe=rows(root/'polar_events.jsonl');assert len(pe)==POLAR_STEPS and [e['ordinal'] for e in pe]==list(range(1,POLAR_STEPS+1))
    pexpected={k:v for k,v in expected.items() if k.startswith('DETACH') or any('_'+m+':' in k for m in METHODS)}
    assert collections.Counter(e['group']+':'+e['phase'] for e in pe)==pexpected
    polar={}
    for e in pe:
        assert len(e['matrices'])==8
        for m in e['matrices']:
            assert np.isfinite([m[k] for k in ['rank','proposal_norm','balanced_norm','stable_rank','cosine','actual_norm']]).all()
            assert abs(m['actual_norm']-m['proposal_norm'])<=1e-7+.005*m['proposal_norm']
            if e['phase']=='formal':polar.setdefault(e['group'],[]).append(m)
    polar={k:dict(matrices=len(v),mean_stable_rank=float(np.mean([x['stable_rank'] for x in v])),mean_retained_rank=float(np.mean([x['rank'] for x in v])),mean_direction_cosine=float(np.mean([x['cosine'] for x in v])),max_absolute_energy_error=max(abs(x['actual_norm']-x['proposal_norm']) for x in v)) for k,v in polar.items()}
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    base=load(cfg['calibrated']);projection=load(cfg['shared_projection']);rotated=load(cfg['projection_control'])['rotated_weight'];metrics={};contrasts={};entry={};diags={}
    for seed in SEEDS:
        key=str(seed);metrics[key]={};contrasts[key]={};entry[key]={}
        for m in METHODS:
            group=f'{seed}_{m}';out=root/'jobs'/group;ini=load(out/'initial.pt');end=load(out/'final.pt');finite(end)
            assert exact(end,load(out/'latest.pt')) and end['step']==end['physical_updates']==2000 and not ini['optimizer']['state'] and end['optimizer']['state']
            assert len(rows(out/'physical_updates.jsonl'))==2000 and end['reads']==[2000,14000] and all(k in end for k in ['torch_rng','cuda_rng'])
            assert all(torch.equal(v,base['model'][k]) for k,v in ini['model'].items() if k.startswith('parent.'))
            assert all(torch.equal(ini['model']['projection.'+k],v) for k,v in projection.items()) and torch.equal(ini['model']['rotated_projection'],rotated)
            assert all(torch.equal(v,ini['model'][k]) and torch.equal(v,end['ema'][k]) for k,v in end['model'].items() if not k.startswith('parent.') or k.endswith(('sigma.weight','grad_update')))
            assert not any(k=='code' or k.startswith('generator.') or 'teacher' in k for k in end['model'])
            assert read(out/'EXIT.json')['exit_code']==read(out/'EVALUATION_EXIT.json')['exit_code']==0
            assert read(root/'qualification'/group/'ENGINEERING.json')['actual_calls']==32
            c=read(root/('SETTINGS_'+group+'_train.private.json'));assert c['polar'] is True and c['method']=='GR' and c['lambda_R']==cfg['lambda_R'] and c['target_seed']==seed
            for cp in [ini,end]:assert len(cp['optimizer']['param_groups'])==2 and all(g['lr']==.001 for g in cp['optimizer']['param_groups'])
            dd=rows(out/'gradient_diagnostics.jsonl');assert [d['step'] for d in dd]==[0,499,999,1499,1999] and all(d['displacement_includes_polar']==c['polar'] for d in dd);diags[group]=dd
            ss=rows(out/'sampling_aggregates.jsonl');assert len(ss)==2000 and all(x['fill_pool']=='L_images_only' and x['fill_labels_read']==0 for x in ss)
            metrics[key][m]=read(out/'RESULTS.json')
            for role,n in [('source_train',10),('target_train',13)]:
                rr=rows(out/('private_'+role+'.jsonl'));ref=rows(Path(cfg['old_five_root'])/'jobs/S'/('private_'+role+'.jsonl'));assert len(rr)==2*n and {(v['patient'],v['phase']) for v in rr}=={(v['patient'],v['phase']) for v in ref}
        out=root/'jobs'/f'{seed}_GRP';v=metrics[key]['GRP'];keys=lambda rr:{(r['patient'],r['phase']) for r in rr}
        domain_rows={}
        for domain,n in [('Siemens',23),('Philips',31)]:
            domain_rows[domain]={mode:rows(out/(prefix+domain+'_val.jsonl')) for mode,prefix in [('student','private_'),('EMA','private_ema_')]}
            ref=rows(Path(cfg['comparison_root'])/'jobs/A'/('private_'+domain+'_val.jsonl'))
            for mode,rr in domain_rows[domain].items():
                assert len(rr)==len(keys(rr))==2*n and keys(rr)==keys(ref)
                score=float(np.mean([np.mean(t['dice']) for t in rr]))
                reported=v[domain]['primary_dice'] if mode=='student' else v['EMA_diagnostic'][domain]['primary_dice']
                assert abs(score-reported)<1e-10
            contrasts[key][domain]=damage(domain_rows[domain]['student'],domain_rows[domain]['EMA'])
        original=rows(cfg['original_source_rows']);initial=rows(cfg['source_val_rows'])
        assert keys(original)==keys(domain_rows['Siemens']['EMA'])
        entry[key]={mode:dict(versus_original_source=damage(original,rr),versus_entry=damage(initial,rr)) for mode,rr in domain_rows['Siemens'].items()}
        original_metrics=read(cfg['original_source_results'])['val']
        for mode,rr in domain_rows['Siemens'].items():
            mm=v['Siemens'] if mode=='student' else v['EMA_diagnostic']['Siemens']
            entry[key][mode]['versus_original_source']['phase_pp']={phase:100*(mm['phases'][phase]['primary_dice']-original_metrics['phases'][phase]['primary_dice']) for phase in ['ED','ES']}
        for domain in ['Siemens','Philips']:
            contrasts[key][domain]['phase_pp']={phase:100*(v['EMA_diagnostic'][domain]['phases'][phase]['primary_dice']-v[domain]['phases'][phase]['primary_dice']) for phase in ['ED','ES']}
    gates={}
    for key,mm in metrics.items():
        v=mm['GRP'];old=entry[key]['EMA']['versus_original_source'];student=entry[key]['student']['versus_original_source'];new=contrasts[key]['Philips']
        absolute_old=100*v['EMA_diagnostic']['Siemens']['primary_dice'];absolute_new=100*v['EMA_diagnostic']['Philips']['primary_dice']
        pieces=dict(old_absolute=absolute_old>=SOURCE_PERCENT,old_interval=old['paired']['patient_bootstrap95_pp'][0]>=-.30,new_point=new['mean_pp']>=-.20,new_interval=new['paired']['patient_bootstrap95_pp'][0]>=-.20,new_absolute=absolute_new>=83.,tail_over_1=old['decline_over_pp']['1']<=2,tail_over_2=old['decline_over_pp']['2']==0,tail_p10=old['p10_pp']>=-.50,tail_no_worse_than_student=all(old['decline_over_pp'][k]<=student['decline_over_pp'][k] for k in ['1','2']))
        gates[key]=dict(composite_pass=all(pieces.values()),checks=pieces,EMA_old_percent=absolute_old,EMA_new_percent=absolute_new,old_source_delta_pp=old['mean_pp'],old_source_CI=old['paired']['patient_bootstrap95_pp'],new_student_delta_pp=new['mean_pp'],new_student_CI=new['paired']['patient_bootstrap95_pp'],source_tail_counts=old['decline_over_pp'],source_tail_p10_pp=old['p10_pp'])
    used=int(subprocess.check_output(['du','-s','-B1',str(root)]).split()[0])+2_000_000;elapsed=read(root/'STATUS.json')['resources']['elapsed_seconds'];assert used<4_000_000_000 and elapsed<28800
    audit=dict(status='PASS',cost=count,calls=calls,polar=pc,numerical_selfcheck_SVD=10,total_SVD=48778,geometry_SVD_calls=12030,total_including_geometry_SVD=60808,workers_retired=True,common_entry_empty_opt=True,frozen_projection_constants=True,finite_final_latest_opt_RNG_reads=True,patient_EDES_exact=True,source_reference=read(root/'SOURCE_REFERENCE_CHECK.json'),EMA_formula_exact=True,audit_model_forwards=0,allocated_bytes_including_metadata=used,managed_elapsed_seconds=elapsed)
    x=dict(audit=audit,metrics=metrics,EMA_minus_student=contrasts,old_patient_damage=entry,gates=gates,composite_pass=all(v['composite_pass'] for v in gates.values()),polar_summary=polar,gradient_diagnostics=diags,primary='fixed final EMA2000 decay.999',independent_source_replication=False,independent_test_access=False,development_validation_reused=True)
    write(root/'PUBLIC_SUMMARY.json',x);report(root,x);write(root/'AUDIT.json',audit)


def report(root,x):
    lines=['# 固定EMA部署：三个新目标流确认','','固定第2000步EMA(.999)主部署，最后student配对对照。共享源163/S†69与开发验证患者，不是独立源或最终测试。原源旧域80.36740663770262%，源自身报告primary EMA。','','## 绝对性能','','|流|学生旧%|学生新%|EMA旧%|EMA新%|综合|','|---|---:|---:|---:|---:|---|']
    for s,mm in x['metrics'].items():
        v=mm['GRP'];e=v['EMA_diagnostic'];lines.append(f"|{s}|{100*v['Siemens']['primary_dice']:.6f}|{100*v['Philips']['primary_dice']:.6f}|{100*e['Siemens']['primary_dice']:.6f}|{100*e['Philips']['primary_dice']:.6f}|{x['gates'][s]['composite_pass']}|")
    lines+=['','## 冻结门槛与实测','','```json',json.dumps(dict(composite_pass=x['composite_pass'],per_stream=x['gates']),ensure_ascii=False,indent=2),'```','','## 原源参照患者尾部','','|流/模型|源差均值pp [95%CI]|p10pp|下降>.5/1/2pp|改善/下降|入口差pp|','|---|---|---:|---|---|---:|']
    for s,mm in x['old_patient_damage'].items():
        for mode,v in mm.items():
            d=v['versus_original_source'];lo,hi=d['paired']['patient_bootstrap95_pp'];c=d['decline_over_pp'];lines.append(f"|{s}/{mode}|{d['mean_pp']:+.6f} [{lo:+.6f},{hi:+.6f}]|{d['p10_pp']:+.6f}|{c['0.5']}/{c['1']}/{c['2']}|{d['increased']}/{d['decreased']}|{v['versus_entry']['mean_pp']:+.6f}|")
    lines+=['','## 完整配对与类别/EDES','','```json',json.dumps(dict(EMA_minus_student=x['EMA_minus_student'],source_comparisons=x['old_patient_damage']),ensure_ascii=False,indent=2),'```','','## 成本与工程审计','','```json',json.dumps(x['audit'],ensure_ascii=False,indent=2),'```','','每流单独判断，所有流全门槛才PASS。CI为10000次患者配对重采样seed163，保留患者内ED/ES，不覆盖训练随机/共享源/重复开发和多重比较。旧-.30、新-.20pp非劣界限不是临床安全界限；均值恢复不代表逐患者恢复。无补种子、扫描alpha、事后checkpoint选择或放宽门槛。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
