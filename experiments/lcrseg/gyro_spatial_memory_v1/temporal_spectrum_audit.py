"""One final CPU-only receipt/state/paired audit for the four-group experiment."""
import collections,json,subprocess
from pathlib import Path
import numpy as np,torch
from .runtime import exact,finite
from .single_target_audit import paired,rows
from .temporal_spectrum_p0 import damage,patient_map
from .temporal_spectrum import SEEDS,METHODS
from ..main_style_ssl_v1.runtime import write


def read(p):return json.loads(Path(p).read_text())
def interaction(rr):
    v={k:patient_map(r) for k,r in rr.items()};keys=sorted(v['S']);assert all(sorted(d)==keys for d in v.values());values=np.array([100*(v['GT'][k]-v['T'][k]-v['G'][k]+v['S'][k]) for k in keys]);rng=np.random.default_rng(163)
    return dict(mean_pp=float(values.mean()),median_pp=float(np.median(values)),patient_bootstrap95_pp=np.quantile(values[rng.integers(len(values),size=(10000,len(values)))].mean(1),[.025,.975]).tolist())

def main(cfg):
    torch.set_num_threads(4);root=Path(cfg['root']);assert not (root/'AUDIT.json').exists() and read(root/'STATUS.json')['status']=='COMPLETE'
    pg=read(root/'LAUNCH.private.json')['process_group']
    for line in subprocess.check_output(['ps','-eo','pid,pgid,stat'],text=True).splitlines()[1:]:
        f=line.split();assert int(f[1])!=pg or f[2].startswith('Z')
    for p in [root/'LAUNCH.private.json',root/'WORKER.private.json',*root.glob('ACTIVE_*.private.json')]:
        r=read(p);s=Path('/proc')/str(r['pid'])/'stat';assert not s.exists() or s.read_text().split()[21]!=str(r['birth']) or s.read_text().split()[2]=='Z'
    for p in [root/'EXIT.json',*root.glob('EXIT_*.json')]:assert read(p)['exit_code']==0
    expected={'DETACH_S:engineering':32,'DETACH_STOP:engineering':32,**{f'{seed}_{m}:{phase}':n for seed in SEEDS for m in METHODS for phase,n in [('engineering',32),('formal',2000)]}}
    count=read(root/'budget_counter.json');assert count==dict(total=16320,optimizer=16320,functional=0,pending=0,by_phase=expected)
    ev=rows(root/'budget_events.jsonl');assert collections.Counter(x['key'] for x in ev if x['kind']=='optimizer')==expected and sum(x['kind']=='reserved' for x in ev)==16320
    calls=read(root/'calls_counter.json');assert calls['pending'] is None
    totals={a:collections.Counter() for a in ['reserved','executed']}
    for e in rows(root/'calls_events.jsonl'):totals[e['action']][e['kind']]+=e['samples']
    assert totals['reserved']==totals['executed']==calls['executed']
    actual=dict(calibration_encoder=768,calibration_teacher_encoder=256,calibration_reverse=32,detach_native=128,detach_encoder=1536,detach_teacher_encoder=512,detach_reverse=64,engineering_native=640,engineering_encoder=1152,engineering_teacher_encoder=384,engineering_reverse=280,formal_native=32096,formal_encoder=768768,formal_teacher_encoder=256256,formal_reverse=16024,diagnostic_reverse=80,evaluation_native=23136)
    assert calls['executed']==actual and all(calls['executed'][k]<=cap for k,cap in calls['caps'].items())
    cal=read(root/'CALIBRATION.json');assert cal['status']=='PASS' and cal['batches']==16 and cal['model_buffers_unchanged'] and cal['teacher_unchanged'] and cal['teacher_stop_gradient'] and 1e-4<=cal['lambda_T']<=10
    detach=read(root/'DETACH_CHECK.json');assert detach['status']=='PASS' and detach['updates']==64 and len(detach['steps'])==32 and all(r['model_opt_EMA_RNG_exact'] and r['teacher_stop_gradient'] for r in detach['steps']);assert read(root/'SELF_CHECK.json')['status']=='PASS'
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    base=load(cfg['calibrated']);projection=load(cfg['shared_projection']);metrics={};contrasts={};diags={};patient_damage={};sampling={}
    for seed in SEEDS:
        metrics[str(seed)]={};contrasts[str(seed)]={};patient_damage[str(seed)]={}
        for m in METHODS:
            group=f'{seed}_{m}';out=root/'jobs'/group;ini=load(out/'initial.pt');end=load(out/'final.pt');latest=load(out/'latest.pt');finite(end)
            assert end['step']==end['physical_updates']==2000 and exact(end,latest) and not ini['optimizer']['state'] and len(rows(out/'physical_updates.jsonl'))==2000
            assert all(torch.equal(v,base['model'][k]) for k,v in ini['model'].items() if k.startswith('parent.')) and all(torch.equal(ini['model']['projection.'+k],v) for k,v in projection.items())
            assert not any(k=='code' or k.startswith('generator.') or 'temporal_teacher' in k for k in end['model'])
            assert all(torch.equal(v,ini['model'][k]) and torch.equal(v,end['ema'][k]) for k,v in end['model'].items() if not k.startswith('parent.') or k.endswith(('sigma.weight','grad_update')))
            assert read(root/'qualification'/group/'ENGINEERING.json')['actual_calls']==32 and read(out/'EXIT.json')['exit_code']==read(out/'EVALUATION_EXIT.json')['exit_code']==0
            c=read(root/('SETTINGS_'+group+'_train.private.json'));assert c['lambda_T']==cal['lambda_T'] and c['target_seed']==seed and c['method']==m
            dd=rows(out/'gradient_diagnostics.jsonl');assert [d['step'] for d in dd]==[0,499,999,1499,1999];diags[group]=dd
            ss=rows(out/'sampling_aggregates.jsonl');assert len(ss)==2000 and all(v['fill_pool']=='L_images_only' and v['fill_labels_read']==0 for v in ss);sampling[group]={'repeat_fraction':float(np.mean([v['repeat_fraction'] for v in ss]))}
            metrics[str(seed)][m]=read(out/'RESULTS.json')
            for role,n in [('source_train',10),('target_train',13)]:
                rr=rows(out/('private_'+role+'.jsonl'));ref=rows(Path(cfg['old_five_root'])/'jobs/S'/('private_'+role+'.jsonl'));assert len(rr)==2*n and {(r['patient'],r['phase']) for r in rr}=={(r['patient'],r['phase']) for r in ref}
        for mode,prefix in [('student','private_'),('EMA','private_ema_')]:
            contrasts[str(seed)][mode]={};patient_damage[str(seed)][mode]={}
            for domain,n in [('Siemens',23),('Philips',31)]:
                rr={m:rows(root/'jobs'/f'{seed}_{m}'/(prefix+domain+'_val.jsonl')) for m in METHODS};keys=lambda v:{(r['patient'],r['phase']) for r in v};assert all(len(v)==n*2 and keys(v)==keys(rr['S']) for v in rr.values())
                contrasts[str(seed)][mode][domain]={f'{b}_minus_{a}':paired(rr[a],rr[b]) for a,b in [('S','G'),('G','GT'),('T','GT'),('S','T'),('S','GT')]};contrasts[str(seed)][mode][domain]['interaction']=interaction(rr)
                entry=rows(cfg['source_val_rows']) if domain=='Siemens' else rows(Path(cfg['comparison_root'])/'preparation/private_Philips_val.jsonl')
                patient_damage[str(seed)][mode][domain]={m:damage(entry,rr[m]) for m in METHODS}
    gates={}
    for mode in ['student','EMA']:
        gates[mode]={}
        for name,threshold in [('G_minus_S',.5),('GT_minus_G',.3),('GT_minus_T',.3)]:
            old=[contrasts[str(seed)][mode]['Siemens'][name]['delta_B_minus_A_pp'] for seed in SEEDS];new=[contrasts[str(seed)][mode]['Philips'][name]['delta_B_minus_A_pp'] for seed in SEEDS];gates[mode][name]={'pass_gate':bool(np.mean(old)>=threshold and min(old)>0 and min(new)>=-.3),'old_pp':old,'new_pp':new,'old_mean_pp':float(np.mean(old))}
        gates[mode]['absolute_target_retention']=[metrics[str(seed)]['GT']['old_delta_from_entry_pp']>=-.5 if mode=='student' else metrics[str(seed)]['GT']['EMA_diagnostic']['Siemens']['primary_dice']>=.7893371427121955 for seed in SEEDS]
        gates[mode]['no_more_over_1pp_patient_damage_than_G']=[patient_damage[str(seed)][mode]['Siemens']['GT']['decline_over_pp']['1']<=patient_damage[str(seed)][mode]['Siemens']['G']['decline_over_pp']['1'] for seed in SEEDS]
    audit=dict(status='PASS',cost=count,calls=calls,workers_retired=True,calibration_once=True,teacher_stopgrad_and_unregistered=True,common_entry_empty_opt=True,frozen_P_constants=True,finite_final_latest_opt_RNG_reads=True,patient_EDES_exact=True,audit_model_forwards=0,old_endpoints_unchanged=True)
    summary=dict(audit=audit,calibration=cal,metrics=metrics,contrasts=contrasts,gates=gates,patient_damage=patient_damage,sampling=sampling,gradient_diagnostics=diags,primary='fixed final student2000',source_seeds_not_independent=True,conditional_followups_started=False)
    write(root/'AUDIT.json',audit);write(root/'PUBLIC_SUMMARY.json',summary)
    lines=['# 两流S/G/T/GT完整结果','','16320 optimizer，0 functional/toy/pending，所有受管进程退休，一次CPU审计PASS。','', '|流|组|最后学生旧/新%|EMA旧/新%|','|---|---|---:|---:|']
    for seed,arms in metrics.items():
        for m,v in arms.items():lines.append(f"|{seed}|{m}|{v['Siemens']['primary_dice']*100:.6f}/{v['Philips']['primary_dice']*100:.6f}|{v['EMA_diagnostic']['Siemens']['primary_dice']*100:.6f}/{v['EMA_diagnostic']['Philips']['primary_dice']*100:.6f}|")
    lines+=['','冻结门槛（开发点估计门槛，非区间非劣确认）：','',json.dumps(gates,ensure_ascii=False,indent=2),'','全部配对区间、类/EDES/损益/尾部、交互和实际调用见PUBLIC_SUMMARY，须完整分析后再解释。GT最好不自动等于协同，EMA不替代学生，绝对入口/原源保持分开；共享源/P/入口、两目标流及反复开发验证限制保留。条件后续不自动展开。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
