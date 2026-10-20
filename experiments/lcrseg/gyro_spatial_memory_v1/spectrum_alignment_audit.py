"""CPU-only existing-record sensitivity and one final two-stream receipt audit."""
import collections,json,subprocess
from pathlib import Path
import numpy as np,torch
from .runtime import exact,finite
from .single_target_audit import paired,rows
from ..main_style_ssl_v1.runtime import write
from ..siemens_lonly_v1.run import aggregate


def read(p):return json.loads(Path(p).read_text())
def patient_effect(a,b):
    aa={};bb={}
    for r in a:aa.setdefault(r['patient'],[]).append(np.mean(r['dice']))
    for r in b:bb.setdefault(r['patient'],[]).append(np.mean(r['dice']))
    assert aa.keys()==bb.keys() and all(len(v)==2 for v in [*aa.values(),*bb.values()])
    return np.array([100*(np.mean(bb[k])-np.mean(aa[k])) for k in sorted(aa)])


def p0(cfg):
    root=Path(cfg['root']);old=Path(cfg['old_five_root']);assert read(old/'AUDIT.json')['status']=='PASS';result={}
    for domain in ['Siemens','Philips']:
        rr={g:rows(old/'jobs'/g/('private_'+domain+'_val.jsonl')) for g in ['S','G00','G11']};d={}
        for name,reference in [('G00_minus_S','S'),('G00_minus_G11','G11')]:
            values=patient_effect(rr[reference],rr['G00']);loo=(values.sum()-values)/(len(values)-1)
            a,b=aggregate(rr[reference]),aggregate(rr['G00'])
            d[name]=dict(paired=paired(rr[reference],rr['G00']),EDES_delta_pp={phase:100*(b['phases'][phase]['primary_dice']-a['phases'][phase]['primary_dice']) for phase in ['ED','ES']},leave_one_patient_mean_min_max_pp=[float(loo.min()),float(loo.max())],leave_one_patient_all_positive=bool((loo>0).all()),no_patient_removed=True)
        result[domain]=d
    write(root/'P0_PATIENT_SENSITIVITY.json',dict(analysis=result,model_forwards=0,optimizer_updates=0,changes_to_thresholds_or_calibration=0))


def main(cfg):
    root=Path(cfg['root']);assert not (root/'AUDIT.json').exists() and read(root/'STATUS.json')['status']=='COMPLETE'
    launch=read(root/'LAUNCH.private.json');pg=launch['process_group']
    for line in subprocess.check_output(['ps','-eo','pid,pgid,stat'],text=True).splitlines()[1:]:
        f=line.split();assert int(f[1])!=pg or f[2].startswith('Z'),'managed processes not retired'
    for path in [root/'LAUNCH.private.json',root/'WORKER.private.json',*root.glob('ACTIVE_*.private.json')]:
        if not path.exists():continue
        d=read(path);stat=Path('/proc')/str(d['pid'])/'stat'
        if stat.exists():v=stat.read_text().split();assert v[21]!=str(d['birth']) or v[2]=='Z'
    for path in [root/'EXIT.json',*root.glob('EXIT_*.json')]:assert read(path)['exit_code']==0,path.name
    expected={'DETACH_S:engineering':32,'DETACH_STOP:engineering':32,**{str(seed)+'_'+method+':'+phase:n for seed in [20261021,20261022] for method in ['S','G00','Aalign'] for phase,n in [('engineering',32),('formal',2000)]}}
    count=read(root/'budget_counter.json');events=rows(root/'budget_events.jsonl')
    assert count==dict(total=12256,optimizer=12256,functional=0,pending=0,by_phase=expected)
    assert collections.Counter(v['key'] for v in events if v['kind']=='optimizer')==expected and sum(v['kind']=='reserved' for v in events)==12256
    calls=read(root/'calls_counter.json');ce=rows(root/'calls_events.jsonl');assert calls['pending'] is None
    assert all(calls['executed'][k]<=cap for k,cap in calls['caps'].items())
    executed=collections.Counter();reserved=collections.Counter()
    for v in ce:(executed if v['action']=='executed' else reserved)[v['kind']]+=v['samples']
    assert executed==reserved==calls['executed']
    expected_calls=dict(calibration_encoder=768,calibration_reverse=32,detach_native=128,detach_encoder=1536,detach_reverse=64,engineering_native=480,engineering_encoder=576,engineering_reverse=210,formal_native=24072,formal_encoder=384384,formal_reverse=12018,diagnostic_reverse=60,evaluation_native=17352)
    assert calls['executed']==expected_calls
    cal=read(root/'CALIBRATION.json');assert cal['status']=='PASS' and cal['batches']==16 and cal['shared_by_both_streams'] and cal['model_buffers_unchanged'] and 1e-4<=cal['lambda_A']<=10
    detach=read(root/'DETACH_CHECK.json');assert detach['status']=='PASS' and detach['updates']==64 and len(detach['steps'])==32 and all(v['model_opt_EMA_RNG_exact'] for v in detach['steps'])
    assert read(root/'SELF_CHECK.json')['status']=='PASS'
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    base=load(cfg['calibrated']);projection=load(cfg['shared_projection']);metrics={};contrasts={};sampling={};ema_minus_student={};gradient_diagnostics={}
    for seed in [20261021,20261022]:
        metrics[str(seed)]={};contrasts[str(seed)]={};ema_minus_student[str(seed)]={}
        for method in ['S','G00','Aalign']:
            group=str(seed)+'_'+method;out=root/'jobs'/group
            assert read(root/'qualification'/group/'ENGINEERING.json')['actual_calls']==32 and read(out/'EXIT.json')['exit_code']==read(out/'EVALUATION_EXIT.json')['exit_code']==0
            assert len(rows(out/'physical_updates.jsonl'))==2000
            ini=load(out/'initial.pt');end=load(out/'final.pt');latest=load(out/'latest.pt');finite(end)
            assert end['step']==end['physical_updates']==2000 and not ini['optimizer']['state'] and exact(end,latest)
            assert all(torch.equal(v,base['model'][k]) for k,v in ini['model'].items() if k.startswith('parent.'))
            assert all(torch.equal(ini['model']['projection.'+k],v) for k,v in projection.items())
            assert not any(k=='code' or k.startswith('generator.') for k in end['model'])
            assert all(torch.equal(v,ini['model'][k]) and torch.equal(v,end['ema'][k]) for k,v in end['model'].items() if not k.startswith('parent.') or k.endswith(('sigma.weight','grad_update')))
            diag=rows(out/'gradient_diagnostics.jsonl');gradient_diagnostics[group]=diag;assert len(diag)==5 and [v['step'] for v in diag]==[0,499,999,1499,1999]
            settings=read(root/('SETTINGS_'+group+'_train.private.json'));assert settings['lambda_A']==cal['lambda_A'] and settings['target_seed']==seed and settings['method']==method
            if method!='S':
                ss=rows(out/'sampling_aggregates.jsonl');assert len(ss)==2000 and {v['step'] for v in ss}==set(range(2000)) and all(v['fill_labels_read']==0 and v['fill_pool']=='L_images_only' for v in ss)
                sampling[group]=dict(mean_distinct_images=float(np.mean([v['distinct_images'] for v in ss])),repeat_fraction=float(np.mean([v['repeat_fraction'] for v in ss])))
            metrics[str(seed)][method]=read(out/'RESULTS.json')
            for role,n in [('source_train',10),('target_train',13)]:
                v=metrics[str(seed)][method][role];assert v['patients']==n and v['volumes']==2*n
                current=rows(out/('private_'+role+'.jsonl'));reference=rows(Path(cfg['old_five_root'])/'jobs/S'/('private_'+role+'.jsonl'))
                assert {(v['patient'],v['phase']) for v in current}=={(v['patient'],v['phase']) for v in reference} and len(current)==2*n
        for mode,prefix in [('student','private_'),('EMA','private_ema_')]:
            contrasts[str(seed)][mode]={}
            for domain,n in [('Siemens',23),('Philips',31)]:
                rr={m:rows(root/'jobs'/(str(seed)+'_'+m)/(prefix+domain+'_val.jsonl')) for m in ['S','G00','Aalign']}
                keys=lambda v:{(r['patient'],r['phase']) for r in v}
                assert all(keys(v)==keys(rr['S']) and len(v)==2*n for v in rr.values())
                contrasts[str(seed)][mode][domain]=dict(G00_minus_S=paired(rr['S'],rr['G00']),G00_minus_Aalign=paired(rr['Aalign'],rr['G00']),Aalign_minus_S=paired(rr['S'],rr['Aalign']))
        for domain in ['Siemens','Philips']:
            ema_minus_student[str(seed)][domain]={method:paired(rows(root/'jobs'/(str(seed)+'_'+method)/('private_'+domain+'_val.jsonl')),rows(root/'jobs'/(str(seed)+'_'+method)/('private_ema_'+domain+'_val.jsonl'))) for method in ['S','G00','Aalign']}
    gates={}
    for mode in ['student','EMA']:
        gates[mode]={}
        for contrast,threshold in [('G00_minus_S',.5),('G00_minus_Aalign',.3)]:
            old=[contrasts[str(seed)][mode]['Siemens'][contrast]['delta_B_minus_A_pp'] for seed in [20261021,20261022]];new=[contrasts[str(seed)][mode]['Philips'][contrast]['delta_B_minus_A_pp'] for seed in [20261021,20261022]]
            gates[mode][contrast]=dict(pass_gate=bool(np.mean(old)>=threshold and min(old)>0 and min(new)>=-.3),old_effects_pp=old,new_effects_pp=new,old_mean_pp=float(np.mean(old)))
    audit=dict(status='PASS',cost=count,calls=calls,calibration_fixed_once=True,detach_native_opt_EMA_RNG_exact=True,common_entry_empty_opt=True,frozen_P_and_buffers_native_constants=True,final_latest_finite_opt_RNG_reads=True,patients_EDES_exact=True,workers_retired=True,audit_model_forwards=0,old_endpoints_unchanged=True)
    summary=dict(audit=audit,calibration=cal,metrics=metrics,contrasts=contrasts,gates=gates,sampling=sampling,EMA_minus_student=ema_minus_student,gradient_diagnostics=gradient_diagnostics,primary='fixed last student',source_seeds_not_independent=True,conditional_followups_started=False)
    write(root/'AUDIT.json',audit);write(root/'PUBLIC_SUMMARY.json',summary)
    lines=['# 两目标流：空间谱几何与一致性对照完整结果','',f"12256 optimizer、0 functional/toy/pending；全部受管进程退休，一次必要审计PASS，0额外模型前向。共同校准λ_A={cal['lambda_A']:.10g}，两流共用。",'', '|目标流|组|最后student旧/新Dice%|EMA旧/新Dice%|','|---|---|---:|---:|']
    for seed,mm in metrics.items():
        for method,v in mm.items():lines.append(f"|{seed}|{method}|{100*v['Siemens']['primary_dice']:.6f}/{100*v['Philips']['primary_dice']:.6f}|{100*v['EMA_diagnostic']['Siemens']['primary_dice']:.6f}/{100*v['EMA_diagnostic']['Philips']['primary_dice']:.6f}|")
    lines+=['','|目标流/口径|比较|旧域pp [患者95%区间]|新域pp [患者95%区间]|','|---|---|---:|---:|']
    for seed,mm in contrasts.items():
        for mode,domains in mm.items():
            for contrast in ['G00_minus_S','G00_minus_Aalign','Aalign_minus_S']:
                a,b=domains['Siemens'][contrast],domains['Philips'][contrast]
                lines.append(f"|{seed}/{mode}|{contrast}|{a['delta_B_minus_A_pp']:+.6f} {a['patient_bootstrap95_pp']}|{b['delta_B_minus_A_pp']:+.6f} {b['patient_bootstrap95_pp']}|")
    lines+=['',f"冻结门槛结果：{json.dumps(gates,ensure_ascii=False)}。",'', '两流均保留同一源入口与P，只检验目标训练随机性，不能称两个完整独立源种子。患者bootstrap逐流计算，不把患者数当训练重复数。EMA统一辅助、不替换最后student终点。若复现门槛过而G00−A-align未过，只能称谱结构未证明必要，不能因相近性能就宣称已唯一归因纯一致性。','', '完整匿名JSON包含LV/MYO/RV、ED/ES、患者损益/中位和训练口径、两流差异、EMA对学生、固定步梯度/裁剪/实际位移另有原始匿名数值摘要可整理；旧五组P0留一患者仅CPU复用，不删患者、不改指标或门槛。条件后续未启动，旧终点保持原样。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
