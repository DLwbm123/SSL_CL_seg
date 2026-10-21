"""One final CPU-only audit and complete paired report, without model forwards."""
import collections,json,subprocess,time
from pathlib import Path
import numpy as np,torch
from .runtime import exact,finite
from .single_target_audit import paired,rows
from .temporal_spectrum_p0 import damage
from .task_native_spectrum import SEEDS,METHODS,CAPS
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
    count=read(root/'budget_counter.json');assert count==dict(total=20384,optimizer=20384,functional=0,pending=0,by_phase=expected)
    ev=rows(root/'budget_events.jsonl');assert collections.Counter(x['key'] for x in ev if x['kind']=='optimizer')==expected and sum(x['kind']=='reserved' for x in ev)==20384
    calls=read(root/'calls_counter.json');assert calls['pending'] is None and calls['caps']==CAPS
    totals={a:collections.Counter() for a in ['reserved','executed']}
    for e in rows(root/'calls_events.jsonl'):totals[e['action']][e['kind']]+=e['samples']
    assert totals['reserved']==totals['executed']==calls['executed'] and all(calls['executed'][k]<=cap for k,cap in CAPS.items())
    actual=dict(source_native=230,source_reverse=230,calibration_encoder=768,calibration_reverse=64,detach_native=128,detach_encoder=1536,detach_reverse=64,engineering_native=800,engineering_encoder=1440,engineering_reverse=350,formal_native=40120,formal_encoder=960960,formal_reverse=20030,diagnostic_reverse=100,evaluation_native=28920)
    assert calls['executed']==actual
    stat=read(root/'SOURCE_STATISTICS.json');cal=read(root/'CALIBRATION.json');detach=read(root/'DETACH_CHECK.json')
    assert stat['status']=='PASS' and stat['optimizer_updates']==0 and stat['slices']==230 and stat['source_train_L_only'] and stat['model_buffers_RNG_exact']
    assert stat['weight_squared_CV']>=.05 and stat['permuted_weight_squared_cosine']<=.99
    assert cal['status']=='PASS' and cal['optimizer_updates']==0 and cal['model_buffers_RNG_exact'] and cal['batches']==16 and all(1e-4<=v<=10 for v in cal['lambdas'].values())
    assert detach['status']=='PASS' and detach['updates']==64 and len(detach['steps'])==32 and all(r['model_opt_EMA_RNG_exact'] for r in detach['steps']) and read(root/'SELF_CHECK.json')['status']=='PASS'
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    base=load(cfg['calibrated']);projection=load(cfg['shared_projection']);weights=load(cfg['weights_path']);metrics={};contrasts={};diags={};entry_damage={};sampling={}
    assert weights['weight'].shape==(128,) and torch.isfinite(weights['weight']).all()
    for seed in SEEDS:
        metrics[str(seed)]={};contrasts[str(seed)]={};entry_damage[str(seed)]={}
        for m in METHODS:
            group=f'{seed}_{m}';out=root/'jobs'/group;ini=load(out/'initial.pt');end=load(out/'final.pt');latest=load(out/'latest.pt');finite(end)
            assert end['step']==end['physical_updates']==2000 and exact(end,latest) and not ini['optimizer']['state'] and len(rows(out/'physical_updates.jsonl'))==2000
            assert all(torch.equal(v,base['model'][k]) for k,v in ini['model'].items() if k.startswith('parent.'))
            assert all(torch.equal(ini['model']['projection.'+k],v) for k,v in projection.items())
            assert torch.equal(ini['model']['task_weight'],weights['weight']) and torch.equal(ini['model']['task_weight_permuted'],weights['weight'][weights['permutation']])
            assert not any(k=='code' or k.startswith('generator.') or 'teacher' in k for k in end['model'])
            assert all(torch.equal(v,ini['model'][k]) and torch.equal(v,end['ema'][k]) for k,v in end['model'].items() if not k.startswith('parent.') or k.endswith(('sigma.weight','grad_update')))
            assert read(root/'qualification'/group/'ENGINEERING.json')['actual_calls']==32 and read(out/'EXIT.json')['exit_code']==read(out/'EVALUATION_EXIT.json')['exit_code']==0
            c=read(root/('SETTINGS_'+group+'_train.private.json'));assert c['lambdas']==cal['lambdas'] and c['target_seed']==seed and c['method']==m
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
                for a,b in [('S','G'),('NI','NT'),('NP','NT'),('G','NT'),('G','NI'),('NI','NP')]:
                    contrasts_pair=damage(rr[a],rr[b]);contrasts_pair['phase_pp']={ph:100*(metrics[str(seed)][b][domain]['phases'][ph]['primary_dice']-metrics[str(seed)][a][domain]['phases'][ph]['primary_dice']) if mode=='student' else 100*(metrics[str(seed)][b]['EMA_diagnostic'][domain]['phases'][ph]['primary_dice']-metrics[str(seed)][a]['EMA_diagnostic'][domain]['phases'][ph]['primary_dice']) for ph in ['ED','ES']}
                    comparisons[b+'_minus_'+a]=contrasts_pair
                contrasts[str(seed)][mode][domain]=comparisons
                entry=rows(cfg['source_val_rows']) if domain=='Siemens' else rows(Path(cfg['comparison_root'])/'preparation/private_Philips_val.jsonl')
                entry_damage[str(seed)][mode][domain]={m:damage(entry,rr[m]) for m in METHODS}
    gates={}
    for mode in ['student','EMA']:
        gates[mode]={}
        for name,threshold in [('G_minus_S',.5),('NT_minus_NI',.3),('NT_minus_NP',.3),('NT_minus_G',.3)]:
            old=[contrasts[str(seed)][mode]['Siemens'][name]['mean_pp'] for seed in SEEDS];new=[contrasts[str(seed)][mode]['Philips'][name]['mean_pp'] for seed in SEEDS]
            gates[mode][name]=dict(pass_gate=bool(np.mean(old)>=threshold and min(old)>0 and min(new)>=-.3),old_pp=old,new_pp=new,old_mean_pp=float(np.mean(old)),new_mean_pp=float(np.mean(new)))
        gates[mode]['absolute_target_retention']=[entry_damage[str(seed)][mode]['Siemens']['NT']['mean_pp']>=-.5 for seed in SEEDS]
        gates[mode]['no_more_over_1pp_patient_damage_than_G']=[entry_damage[str(seed)][mode]['Siemens']['NT']['decline_over_pp']['1']<=entry_damage[str(seed)][mode]['Siemens']['G']['decline_over_pp']['1'] for seed in SEEDS]
    used=int(subprocess.check_output(['du','-s','-B1',str(root)]).split()[0])+2_000_000;elapsed=read(root/'STATUS.json')['resources']['elapsed_seconds'];assert used<4_000_000_000 and elapsed<36000
    audit=dict(status='PASS',cost=count,calls=calls,workers_retired=True,source_statistic_zero_update=True,common_entry_empty_opt=True,frozen_task_weights_P_constants=True,finite_final_latest_opt_RNG_reads=True,patient_EDES_exact=True,audit_model_forwards=0,old_endpoints_unchanged=True,allocated_bytes_including_metadata=used,managed_elapsed_seconds=elapsed)
    summary=dict(audit=audit,source_statistics=stat,calibration=cal,metrics=metrics,contrasts=contrasts,gates=gates,patient_damage=entry_damage,sampling=sampling,gradient_diagnostics=diags,primary='fixed final student2000',source_seeds_not_independent=True,conditional_followups_started=False)
    # Store aggregates before the PASS marker so a report error cannot masquerade as completion.
    write(root/'PUBLIC_SUMMARY.json',summary);report(root,summary);write(root/'AUDIT.json',audit)


def report(root,x):
    lines=['# 任务相关原生瓶颈谱：两目标流完整报告','','固定最后student2000为主，统一EMA辅助；旧终点不变，条件后续不启动。五组S/G/NI/NP/NT的合同见冻结协议。源敏感性使用额外源train-L统计，信息量与G不同，NP与NT共享同一权重分布。','','## 固定终点','','|流|组|student旧/新%|EMA旧/新%|源/目标train student%|','|---|---|---:|---:|---:|']
    for seed,arms in x['metrics'].items():
        for m,v in arms.items():
            e=v['EMA_diagnostic'];lines.append(f"|{seed}|{m}|{100*v['Siemens']['primary_dice']:.6f}/{100*v['Philips']['primary_dice']:.6f}|{100*e['Siemens']['primary_dice']:.6f}/{100*e['Philips']['primary_dice']:.6f}|{100*v['source_train']['primary_dice']:.6f}/{100*v['target_train']['primary_dice']:.6f}|")
    lines+=['','## 冻结门槛','',''+json.dumps(x['gates'],ensure_ascii=False,indent=2),'','NT-NI/NP/G分别要求旧均>=+.3pp、每流旧>0、新每流>=-.3；G-S要求旧均>=+.5pp。点估计通过不等于区间确认。三项新候选比较及患者代价门槛须同时满足；否则停止该配方，不扫描权重/clip/排列/维度。','','## 全部配对增量、患者幅度与器官/EDES','','|流/读出/域|比较|均值pp [95%患者区间]|中位/p10 pp|改善/下降|下降>.5/1/2pp|LV/MYO/RV pp|ED/ES pp|','|---|---|---|---|---|---|---|---|']
    for seed,modes in x['contrasts'].items():
        for mode,domains in modes.items():
            for domain,comparisons in domains.items():
                for name,c in comparisons.items():
                    p=c['paired'];lo,hi=p['patient_bootstrap95_pp'];cl='/'.join(f"{p['classes_pp'][k]:+.6f}" for k in ['LV','MYO','RV']);ph=c['phase_pp'];cost=c['decline_over_pp']
                    lines.append(f"|{seed}/{mode}/{domain}|{name}|{c['mean_pp']:+.6f} [{lo:+.6f},{hi:+.6f}]|{c['median_pp']:+.6f}/{c['p10_pp']:+.6f}|{c['increased']}/{c['decreased']}|{cost['0.5']}/{cost['1']}/{cost['2']}|{cl}|{ph['ED']:+.6f}/{ph['ES']:+.6f}|")
    lines+=['','## 相对入口和原源的代价','','|流/读出/域|组|入口均值/中位/p10 pp|下降>.5/1/2pp|旧域相对原源pp|','|---|---|---|---|---|']
    for seed,modes in x['patient_damage'].items():
        for mode,domains in modes.items():
            for domain,arms in domains.items():
                for m,c in arms.items():
                    delta=c['mean_pp']-.93369236648307 if domain=='Siemens' else None;cost=c['decline_over_pp'];delta='NA' if delta is None else f'{delta:+.6f}'
                    lines.append(f"|{seed}/{mode}/{domain}|{m}|{c['mean_pp']:+.6f}/{c['median_pp']:+.6f}/{c['p10_pp']:+.6f}|{cost['0.5']}/{cost['1']}/{cost['2']}|{delta}|")
    lines+=['','## 形成、尺度、诊断与资源','','源统计：'+json.dumps(x['source_statistics'],ensure_ascii=False),'','共同入口一次校准：'+json.dumps(x['calibration'],ensure_ascii=False),'','50固定点的梯度/裁剪/真实SGD位移全部在PUBLIC_SUMMARY，不能由这些点推断全程无裁剪。实际位移包括动量和weight decay。','','实际成本与预算：'+json.dumps(x['audit'],ensure_ascii=False),'','## 解释边界','','只支持当前冻结配置的辨别结论。NT-NP相近不能声称任务对应必要；NT-G同时改变P/维度/分母且额外形成源统计，不单独归因正则位置。该权重是梯度×激活的局部近似，不是Fisher或完整Jacobian；不保护绕过bottleneck的skip。无历史锚定不等于证明保持旧任务函数。','','原生谱维度128/n64，G为P32/n64，三原生组维度和算子严格匹配。源权重仅一次合法10患者/20EDES/230切片，未保存逐图特征，未在目标训练读取源图。完整类别缺失频率如实披露。','','共享源163/S†/P，仅两目标流；患者bootstrap10000 seed163不覆盖训练随机性、多重比较与反复开发。器官代价/绝对入口与原源保持单列，不用双域均值掩盖。EMA不替代最后学生，未构造T组合或交互。无临床阈值或独立泛化主张，未自动展开条件后续。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
