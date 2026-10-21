"""One final zero-forward audit for the frozen 2x2 update-spectrum intervention."""
import collections,json,subprocess
from pathlib import Path
import numpy as np,torch
from .runtime import exact,finite
from .single_target_audit import rows
from .temporal_spectrum_p0 import damage
from .update_spectrum import SEEDS,METHODS,CAPS,POLAR_STEPS,SHAPES
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
    expected={'DETACH_S:engineering':32,'DETACH_STOP:engineering':32,**{f'{s}_{m}:{p}':n for s in SEEDS for m in METHODS for p,n in [('engineering',32),('formal',2000)]}}
    count=read(root/'budget_counter.json');assert count==dict(total=16320,optimizer=16320,functional=0,pending=0,by_phase=expected)
    ev=rows(root/'budget_events.jsonl');assert collections.Counter(x['key'] for x in ev if x['kind']=='optimizer')==expected and sum(x['kind']=='reserved' for x in ev)==16320
    calls=read(root/'calls_counter.json');assert calls['pending'] is None and calls['caps']==CAPS
    totals={a:collections.Counter() for a in ['reserved','executed']}
    for e in rows(root/'calls_events.jsonl'):totals[e['action']][e['kind']]+=e['samples']
    actual=dict(detach_native=128,detach_encoder=1536,detach_reverse=64,engineering_native=640,engineering_encoder=1152,engineering_reverse=280,formal_native=32096,formal_encoder=768768,formal_reverse=16024,diagnostic_reverse=80,evaluation_native=23136)
    assert totals['reserved']==totals['executed']==calls['executed']==actual and all(actual[k]<=v for k,v in CAPS.items())
    assert read(root/'SELF_CHECK.json')['status']=='PASS' and read(root/'DETACH_CHECK.json')['status']=='PASS'
    assert read(root/'DETACH_CHECK.json')['updates']==64 and all(v['model_opt_EMA_RNG_exact'] for v in read(root/'DETACH_CHECK.json')['steps'])
    pc=read(root/'polar_counter.json');assert pc==dict(executed_steps=POLAR_STEPS,executed_SVD=8*POLAR_STEPS,pending=None,cap_steps=POLAR_STEPS,cap_SVD=8*POLAR_STEPS)
    pe=rows(root/'polar_events.jsonl');assert len(pe)==POLAR_STEPS and [e['ordinal'] for e in pe]==list(range(1,POLAR_STEPS+1))
    pexpected={k:v for k,v in expected.items() if k.startswith('DETACH') or '_SP:' in k or '_GRP:' in k}
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
            c=read(root/('SETTINGS_'+group+'_train.private.json'));assert c['polar']==m.endswith('P') and c['method']==('GR' if m.startswith('GR') else 'S') and c['lambda_R']==cfg['lambda_R'] and c['target_seed']==seed
            dd=rows(out/'gradient_diagnostics.jsonl');assert [d['step'] for d in dd]==[0,499,999,1499,1999] and all(d['displacement_includes_polar']==c['polar'] for d in dd);diags[group]=dd
            ss=rows(out/'sampling_aggregates.jsonl');assert len(ss)==2000 and all(x['fill_pool']=='L_images_only' and x['fill_labels_read']==0 for x in ss)
            metrics[key][m]=read(out/'RESULTS.json')
            for role,n in [('source_train',10),('target_train',13)]:
                rr=rows(out/('private_'+role+'.jsonl'));ref=rows(Path(cfg['old_five_root'])/'jobs/S'/('private_'+role+'.jsonl'));assert len(rr)==2*n and {(v['patient'],v['phase']) for v in rr}=={(v['patient'],v['phase']) for v in ref}
        for mode,prefix in [('student','private_'),('EMA','private_ema_')]:
            contrasts[key][mode]={};entry[key][mode]={}
            for domain,n in [('Siemens',23),('Philips',31)]:
                rr={m:rows(root/'jobs'/f'{seed}_{m}'/(prefix+domain+'_val.jsonl')) for m in METHODS};keys=lambda v:{(x['patient'],x['phase']) for x in v}
                ref=rows(Path(cfg['comparison_root'])/'jobs/A'/('private_'+domain+'_val.jsonl'));assert all(len(v)==2*n and keys(v)==keys(ref) for v in rr.values())
                comparisons={}
                for a,b in [('S','SP'),('GR','GRP'),('S','GR'),('SP','GRP'),('S','GRP')]:
                    v=damage(rr[a],rr[b]);get=lambda m:metrics[key][m][domain] if mode=='student' else metrics[key][m]['EMA_diagnostic'][domain]
                    v['phase_pp']={ph:100*(get(b)['phases'][ph]['primary_dice']-get(a)['phases'][ph]['primary_dice']) for ph in ['ED','ES']};comparisons[b+'_minus_'+a]=v
                contrasts[key][mode][domain]=comparisons
                initial=rows(cfg['source_val_rows']) if domain=='Siemens' else rows(Path(cfg['comparison_root'])/'preparation/private_Philips_val.jsonl')
                entry[key][mode][domain]={m:damage(initial,rr[m]) for m in METHODS}
    gates={}
    for mode in ['student','EMA']:
        old=[contrasts[str(s)][mode]['Siemens']['GRP_minus_GR']['mean_pp'] for s in SEEDS];new=[contrasts[str(s)][mode]['Philips']['GRP_minus_GR']['mean_pp'] for s in SEEDS]
        gain=bool(np.mean(old)>=.3 and min(old)>0 and min(new)>=-.3);tail=[];retention=[]
        for s in SEEDS:
            a=entry[str(s)][mode]['Siemens'];tail.append(a['GRP']['decline_over_pp']['1']<=a['GR']['decline_over_pp']['1']-2 and a['GRP']['decline_over_pp']['2']<=a['GR']['decline_over_pp']['2'] and a['GRP']['p10_pp']>=a['GR']['p10_pp']-.3);retention.append(a['GRP']['mean_pp']>=-.5)
        interaction={d:[contrasts[str(s)][mode][d]['GRP_minus_GR']['mean_pp']-contrasts[str(s)][mode][d]['SP_minus_S']['mean_pp'] for s in SEEDS] for d in ['Siemens','Philips']}
        gates[mode]=dict(composite_pass=bool(gain and all(tail) and all(retention)),increment_pass=gain,patient_tail_pass=tail,absolute_retention_pass=retention,old_pp=old,new_pp=new,old_mean_pp=float(np.mean(old)),new_mean_pp=float(np.mean(new)),interaction_pp=interaction,interaction_point_gate=bool(min(interaction['Siemens'])>0 and np.mean(interaction['Siemens'])>=.2 and min(interaction['Philips'])>=-.3))
    used=int(subprocess.check_output(['du','-s','-B1',str(root)]).split()[0])+2_000_000;elapsed=read(root/'STATUS.json')['resources']['elapsed_seconds'];assert used<4_000_000_000 and elapsed<28800
    audit=dict(status='PASS',cost=count,calls=calls,polar=pc,numerical_selfcheck_SVD=10,total_SVD=65546,workers_retired=True,common_entry_empty_opt=True,frozen_projection_constants=True,finite_final_latest_opt_RNG_reads=True,patient_EDES_exact=True,audit_model_forwards=0,allocated_bytes_including_metadata=used,managed_elapsed_seconds=elapsed)
    x=dict(audit=audit,metrics=metrics,contrasts=contrasts,patient_damage=entry,gates=gates,polar_summary=polar,gradient_diagnostics=diags,primary='fixed final student2000',independent_source_replication=False)
    write(root/'PUBLIC_SUMMARY.json',x);report(root,x);write(root/'AUDIT.json',audit)


def report(root,x):
    lines=['# 等能量更新谱干预：两流完整结果','','固定student2000主终点；这是独立极分解位移干预，不是MuLoRA复现。患者bootstrap不覆盖训练随机性与反复开发。','','## 门槛','','```json',json.dumps(x['gates'],ensure_ascii=False,indent=2),'```','','## 终点','','|流|组|旧/新学生%|旧/新EMA%|','|---|---|---|---|']
    for s,mm in x['metrics'].items():
        for m,v in mm.items():lines.append(f"|{s}|{m}|{v['Siemens']['primary_dice']*100:.6f}/{v['Philips']['primary_dice']*100:.6f}|{v['EMA_diagnostic']['Siemens']['primary_dice']*100:.6f}/{v['EMA_diagnostic']['Philips']['primary_dice']*100:.6f}|")
    lines+=['','## 配对代价','','|流/模式/域|比较|均值pp [95%患者CI]|中位/p10|改善/下降|下降>.5/1/2pp|LV/MYO/RV|ED/ES|','|---|---|---|---|---|---|---|---|']
    for s,mm in x['contrasts'].items():
        for mode,dd in mm.items():
            for d,cc in dd.items():
                for name,c in cc.items():
                    p=c['paired'];cost=c['decline_over_pp'];cl='/'.join(f'{p["classes_pp"][k]:+.6f}' for k in ['LV','MYO','RV']);lo,hi=p['patient_bootstrap95_pp']
                    lines.append(f"|{s}/{mode}/{d}|{name}|{c['mean_pp']:+.6f} [{lo:+.6f},{hi:+.6f}]|{c['median_pp']:+.6f}/{c['p10_pp']:+.6f}|{c['increased']}/{c['decreased']}|{cost['0.5']}/{cost['1']}/{cost['2']}|{cl}|{c['phase_pp']['ED']:+.6f}/{c['phase_pp']['ES']:+.6f}|")
    lines+=['','## 相对入口患者损失','','|流/模式/域|组|均值/中位/p10 pp|下降>.5/1/2pp|旧相对原源pp|','|---|---|---|---|---|']
    for s,mm in x['patient_damage'].items():
        for mode,dd in mm.items():
            for d,cc in dd.items():
                for m,c in cc.items():
                    cost=c['decline_over_pp'];original=f"{c['mean_pp']-.93369236648307:+.6f}" if d=='Siemens' else 'NA'
                    lines.append(f"|{s}/{mode}/{d}|{m}|{c['mean_pp']:+.6f}/{c['median_pp']:+.6f}/{c['p10_pp']:+.6f}|{cost['0.5']}/{cost['1']}/{cost['2']}|{original}|")
    lines+=['','## 更新干预与成本','','```json',json.dumps(dict(polar=x['polar_summary'],audit=x['audit']),ensure_ascii=False,indent=2),'```','','能量只匹配同一状态下普通SGD提议；不同轨迹不保证相同能量。保留普通momentum，decoder/bias不变。数值支持均衡可能放大低能量噪声；不将平均增益解释为患者安全。交互仅为固定两流点估计。所有旧失败保留，下一问题另立冻结协议。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
