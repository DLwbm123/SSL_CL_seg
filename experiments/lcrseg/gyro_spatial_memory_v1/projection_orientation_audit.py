"""One final CPU-only audit and complete paired report, without model forwards."""
import collections,json,subprocess,time
from pathlib import Path
import numpy as np,torch
from .runtime import exact,finite
from .single_target_audit import paired,rows
from .temporal_spectrum_p0 import damage
from .projection_orientation import SEEDS,METHODS,CAPS,rotated_projection
from ..main_style_ssl_v1.runtime import write


def read(p):return json.loads(Path(p).read_text())


def main(cfg):
    torch.set_num_threads(4);root=Path(cfg['root']);assert not (root/'AUDIT.json').exists() and read(root/'STATUS.json')['status']=='COMPLETE'
    launch=read(root/'LAUNCH.private.json');pg=launch['process_group']
    for line in subprocess.check_output(['ps','-eo','pid,pgid,stat'],text=True).splitlines()[1:]:
        f=line.split();assert int(f[1])!=pg or f[2].startswith('Z')
    for p in [root/'LAUNCH.private.json',root/'WORKER.private.json',*root.glob('ACTIVE_*.private.json')]:
        r=read(p);s=Path('/proc')/str(r['pid'])/'stat';assert not s.exists() or s.read_text().split()[21]!=str(r['birth']) or s.read_text().split()[2]=='Z'
    for p in [root/'EXIT.json',*root.glob('EXIT_*.json')]:assert read(p)['exit_code']==0
    expected={'DETACH_S:engineering':32,'DETACH_STOP:engineering':32,**{f'{seed}_{m}:{phase}':n for seed in SEEDS for m in METHODS for phase,n in [('engineering',32),('formal',2000)]}}
    count=read(root/'budget_counter.json');assert count==dict(total=12256,optimizer=12256,functional=0,pending=0,by_phase=expected)
    ev=rows(root/'budget_events.jsonl');assert collections.Counter(x['key'] for x in ev if x['kind']=='optimizer')==expected and sum(x['kind']=='reserved' for x in ev)==12256
    calls=read(root/'calls_counter.json');assert calls['pending'] is None and calls['caps']==CAPS
    totals={a:collections.Counter() for a in ['reserved','executed']}
    for e in rows(root/'calls_events.jsonl'):totals[e['action']][e['kind']]+=e['samples']
    assert totals['reserved']==totals['executed']==calls['executed'] and all(calls['executed'][k]<=cap for k,cap in CAPS.items())
    actual=dict(calibration_encoder=768,calibration_reverse=32,detach_native=128,detach_encoder=1536,detach_reverse=64,engineering_native=480,engineering_encoder=864,engineering_reverse=210,formal_native=24072,formal_encoder=576576,formal_reverse=12018,diagnostic_reverse=60,evaluation_native=17352)
    assert calls['executed']==actual
    stat=read(root/'PROJECTION_CONTROL.json');cal=read(root/'CALIBRATION.json');detach=read(root/'DETACH_CHECK.json')
    assert stat['status']=='PASS' and stat['optimizer_updates']==stat['medical_model_forwards']==stat['source_images_read']==0 and stat['bias_unchanged']
    assert stat['row_gram_relative_error_float32']<=1e-5 and stat['relative_input_direction_change']>=.1
    assert cal['status']=='PASS' and cal['optimizer_updates']==0 and cal['model_buffers_RNG_exact'] and cal['batches']==16 and 1e-4<=cal['lambda_R']<=10
    assert detach['status']=='PASS' and detach['updates']==64 and len(detach['steps'])==32 and all(r['model_opt_EMA_RNG_exact'] for r in detach['steps']) and read(root/'SELF_CHECK.json')['status']=='PASS'
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    base=load(cfg['calibrated']);projection=load(cfg['shared_projection']);weights=load(cfg['projection_control']);metrics={};contrasts={};diags={};entry_damage={};sampling={}
    regenerated,_=rotated_projection(projection);assert exact(regenerated,weights)
    assert weights['rotated_weight'].shape==(32,128,1,1) and torch.isfinite(weights['rotated_weight']).all()
    for seed in SEEDS:
        metrics[str(seed)]={};contrasts[str(seed)]={};entry_damage[str(seed)]={}
        for m in METHODS:
            group=f'{seed}_{m}';out=root/'jobs'/group;ini=load(out/'initial.pt');end=load(out/'final.pt');latest=load(out/'latest.pt');finite(end)
            assert end['step']==end['physical_updates']==2000 and exact(end,latest) and not ini['optimizer']['state'] and len(rows(out/'physical_updates.jsonl'))==2000
            assert end['optimizer']['state'] and all(k in end for k in ['torch_rng','cuda_rng','reads']) and end['reads']==[2000,14000]
            assert all(torch.equal(v,base['model'][k]) for k,v in ini['model'].items() if k.startswith('parent.'))
            assert all(torch.equal(ini['model']['projection.'+k],v) for k,v in projection.items())
            assert torch.equal(ini['model']['rotated_projection'],weights['rotated_weight'])
            assert not any(k=='code' or k.startswith('generator.') or 'teacher' in k for k in end['model'])
            assert all(torch.equal(v,ini['model'][k]) and torch.equal(v,end['ema'][k]) for k,v in end['model'].items() if not k.startswith('parent.') or k.endswith(('sigma.weight','grad_update')))
            assert read(root/'qualification'/group/'ENGINEERING.json')['actual_calls']==32 and read(out/'EXIT.json')['exit_code']==read(out/'EVALUATION_EXIT.json')['exit_code']==0
            c=read(root/('SETTINGS_'+group+'_train.private.json'));assert c['lambda_R']==cal['lambda_R'] and c['target_seed']==seed and c['method']==m
            dd=rows(out/'gradient_diagnostics.jsonl');assert [d['step'] for d in dd]==[0,499,999,1499,1999];diags[group]=dd
            ss=rows(out/'sampling_aggregates.jsonl');assert len(ss)==2000 and all(v['fill_pool']=='L_images_only' and v['fill_labels_read']==0 for v in ss);sampling[group]={'repeat_fraction':float(np.mean([v['repeat_fraction'] for v in ss]))}
            metrics[str(seed)][m]=read(out/'RESULTS.json')
            for role,n in [('source_train',10),('target_train',13)]:
                rr=rows(out/('private_'+role+'.jsonl'));ref=rows(Path(cfg['old_five_root'])/'jobs/S'/('private_'+role+'.jsonl'));assert len(rr)==2*n and {(r['patient'],r['phase']) for r in rr}=={(r['patient'],r['phase']) for r in ref}
        for mode,prefix in [('student','private_'),('EMA','private_ema_')]:
            contrasts[str(seed)][mode]={};entry_damage[str(seed)][mode]={}
            for domain,n in [('Siemens',23),('Philips',31)]:
                rr={m:rows(root/'jobs'/f'{seed}_{m}'/(prefix+domain+'_val.jsonl')) for m in METHODS};keys=lambda v:{(r['patient'],r['phase']) for r in v}
                ref=rows(Path(cfg['comparison_root'])/'jobs/A'/('private_'+domain+'_val.jsonl'));assert all(len(v)==2*n and keys(v)==keys(ref) for v in rr.values())
                comparisons={}
                for a,b in [('S','G'),('S','GR'),('GR','G'),('G','GR')]:
                    contrasts_pair=damage(rr[a],rr[b]);contrasts_pair['phase_pp']={ph:100*(metrics[str(seed)][b][domain]['phases'][ph]['primary_dice']-metrics[str(seed)][a][domain]['phases'][ph]['primary_dice']) if mode=='student' else 100*(metrics[str(seed)][b]['EMA_diagnostic'][domain]['phases'][ph]['primary_dice']-metrics[str(seed)][a]['EMA_diagnostic'][domain]['phases'][ph]['primary_dice']) for ph in ['ED','ES']}
                    comparisons[b+'_minus_'+a]=contrasts_pair
                contrasts[str(seed)][mode][domain]=comparisons
                entry=rows(cfg['source_val_rows']) if domain=='Siemens' else rows(Path(cfg['comparison_root'])/'preparation/private_Philips_val.jsonl')
                entry_damage[str(seed)][mode][domain]={m:damage(entry,rr[m]) for m in METHODS}
    gates={}
    for mode in ['student','EMA']:
        gates[mode]={}
        for name,threshold in [('G_minus_S',.5),('GR_minus_S',.5),('G_minus_GR',.3)]:
            old=[contrasts[str(seed)][mode]['Siemens'][name]['mean_pp'] for seed in SEEDS];new=[contrasts[str(seed)][mode]['Philips'][name]['mean_pp'] for seed in SEEDS]
            gates[mode][name]=dict(pass_gate=bool(np.mean(old)>=threshold and min(old)>0 and min(new)>=-.3),old_pp=old,new_pp=new,old_mean_pp=float(np.mean(old)),new_mean_pp=float(np.mean(new)))
        gates[mode]['absolute_target_retention']={m:[entry_damage[str(seed)][mode]['Siemens'][m]['mean_pp']>=-.5 for seed in SEEDS] for m in METHODS}
        gates[mode]['GR_point_noninferior_to_G_all_domains_streams']=all(contrasts[str(seed)][mode][domain]['GR_minus_G']['mean_pp']>=-.3 for seed in SEEDS for domain in ['Siemens','Philips'])
        gates[mode]['GR_patient_CI_noninferior_to_G_all_domains_streams']=all(contrasts[str(seed)][mode][domain]['GR_minus_G']['paired']['patient_bootstrap95_pp'][0]>=-.3 for seed in SEEDS for domain in ['Siemens','Philips'])
        gates[mode]['no_more_over_1pp_patient_damage_than_G']=[entry_damage[str(seed)][mode]['Siemens']['GR']['decline_over_pp']['1']<=entry_damage[str(seed)][mode]['Siemens']['G']['decline_over_pp']['1'] for seed in SEEDS]
    used=int(subprocess.check_output(['du','-s','-B1',str(root)]).split()[0])+2_000_000;elapsed=read(root/'STATUS.json')['resources']['elapsed_seconds'];assert used<4_000_000_000 and elapsed<28800
    audit=dict(status='PASS',cost=count,calls=calls,workers_retired=True,projection_formation_zero_model_calls=True,common_entry_empty_opt=True,frozen_P_rotation_bias_constants=True,finite_final_latest_opt_RNG_reads=True,patient_EDES_exact=True,audit_model_forwards=0,old_endpoints_unchanged=True,allocated_bytes_including_metadata=used,managed_elapsed_seconds=elapsed)
    summary=dict(audit=audit,projection_control=stat,calibration=cal,metrics=metrics,contrasts=contrasts,gates=gates,patient_damage=entry_damage,sampling=sampling,gradient_diagnostics=diags,primary='fixed final student2000',source_seeds_not_independent=True,conditional_followups_started=False)
    # Store aggregates before the PASS marker so a report error cannot masquerade as completion.
    write(root/'PUBLIC_SUMMARY.json',summary);report(root,summary);write(root/'AUDIT.json',audit)


def report(root,x):
    lines=['# 源训练投影方向依赖：两流完整结果','','固定最后student2000为主，统一EMA辅助。P与旋转P均固定32维和同奇异谱、偏置；GR经过一次入口RMS校准，非全轨迹尺度严格匹配。','','## 固定终点','','|流|组|学生旧/新%|EMA旧/新%|源/目标train学生%|','|---|---|---|---|---|']
    for seed,arms in x['metrics'].items():
        for m,v in arms.items():
            e=v['EMA_diagnostic'];lines.append(f"|{seed}|{m}|{100*v['Siemens']['primary_dice']:.6f}/{100*v['Philips']['primary_dice']:.6f}|{100*e['Siemens']['primary_dice']:.6f}/{100*e['Philips']['primary_dice']:.6f}|{100*v['source_train']['primary_dice']:.6f}/{100*v['target_train']['primary_dice']:.6f}|")
    lines+=['','## 预声明门槛','','```json',json.dumps(x['gates'],ensure_ascii=False,indent=2),'```','','G/GR−S：旧均>=+.5pp/每流旧>0/新每流>=−.3pp。G−GR方向依赖：旧均>=+.3pp/每流旧>0/新每流>=−.3pp。点估计非劣与患者CI非劣分列，均不代表临床非劣。','','## 配对增量和患者代价','','|流/读出/域|比较|均值pp [95%患者CI]|中位/p10|改善/下降|下降>.5/1/2pp|LV/MYO/RV pp|ED/ES pp|','|---|---|---|---|---|---|---|---|']
    for seed,modes in x['contrasts'].items():
        for mode,domains in modes.items():
            for domain,comparisons in domains.items():
                for name,c in comparisons.items():
                    p=c['paired'];lo,hi=p['patient_bootstrap95_pp'];cl='/'.join(f"{p['classes_pp'][k]:+.6f}" for k in ['LV','MYO','RV']);ph=c['phase_pp'];cost=c['decline_over_pp']
                    lines.append(f"|{seed}/{mode}/{domain}|{name}|{c['mean_pp']:+.6f} [{lo:+.6f},{hi:+.6f}]|{c['median_pp']:+.6f}/{c['p10_pp']:+.6f}|{c['increased']}/{c['decreased']}|{cost['0.5']}/{cost['1']}/{cost['2']}|{cl}|{ph['ED']:+.6f}/{ph['ES']:+.6f}|")
    lines+=['','## 相对入口与原源','','|流/读出/域|组|入口均值/中位/p10 pp|下降>.5/1/2pp|旧相对原源pp|','|---|---|---|---|---|']
    for seed,modes in x['patient_damage'].items():
        for mode,domains in modes.items():
            for domain,arms in domains.items():
                for m,c in arms.items():
                    delta=c['mean_pp']-.93369236648307 if domain=='Siemens' else None;cost=c['decline_over_pp'];delta='NA' if delta is None else f'{delta:+.6f}'
                    lines.append(f"|{seed}/{mode}/{domain}|{m}|{c['mean_pp']:+.6f}/{c['median_pp']:+.6f}/{c['p10_pp']:+.6f}|{cost['0.5']}/{cost['1']}/{cost['2']}|{delta}|")
    lines+=['','## 固定30点梯度与实际位移','','|组|裁剪点/5|raw范围|末点CE/aux encoder范数|末点CE-aux cosine|末点encoder/decoder SGD位移|','|---|---|---|---|---|---|']
    for group,ds in x['gradient_diagnostics'].items():
        d=ds[-1];raw=[v['raw_global_gradient_norm'] for v in ds];z=d['actual_SGD_displacement'];lines.append(f"|{group}|{sum(v['theoretical_clip_coefficient']<1 for v in ds)}/5|{min(raw):.6g}/{max(raw):.6g}|{d['CE_gradient']['encoder']:.6g}/{d['aux_gradient']['encoder']:.6g}|{d['CE_aux_cosine']['encoder']}|{z['encoder']:.6g}/{z['decoder']:.6g}|")
    lines+=['','仅固定采样点，不能外推全程；真实位移包括动量/wd。完整统计在PUBLIC_SUMMARY。','','## 工程与成本','','投影形成：'+json.dumps(x['projection_control'],ensure_ascii=False),'','一次校准：'+json.dumps(x['calibration'],ensure_ascii=False),'','最终审计：'+json.dumps(x['audit'],ensure_ascii=False),'','## 边界','','共享源163/P/入口，仅两目标流和一个固定Q；患者bootstrap10000 seed163不覆盖训练随机性、多重比较及反复开发。P的偏置与奇异谱仍源训练，不称GR无源信息。只检验当前输入方向控制，不能把方向效应等同源语义必要性或全局最优。旧终点全保留；不重抽Q、不扫描lambda。依据本轮完整结果另立有限协议继续自主研究。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
