"""One saved-state-only terminal audit; no model instantiation or inference."""
import collections
from pathlib import Path
import torch
from .runtime import CAPS
from .analysis import read,rows,compare,tail,ni
from ..main_style_ssl_v1.runtime import write
from ..gyro_spatial_memory_v1.runtime import exact,finite

def main(cfg):
    torch.set_num_threads(4);r=Path(cfg['root']);assert not (r/'AUDIT.json').exists();status=read(r/'STATUS.json');assert status['status'] in ('COMPLETE','SOURCE_STOP','NUMERIC_STOP','ENGINEERING_STOP')
    for p in [*r.glob('*LAUNCH*.private.json'),r/'WORKER.private.json',*r.glob('ACTIVE_*.private.json')]:
        if not p.exists():continue
        d=read(p);q=Path('/proc')/str(d.get('pid',d.get('timeoutpg')))/'stat';assert not q.exists() or int(q.read_text().split()[21])!=d['birth'] or q.read_text().split()[2]=='Z'
    counts=read(r/'COUNTERS.private.json') if (r/'COUNTERS.private.json').exists() else dict(calls={},optimizer={},pending_optimizer=None);events=rows(r/'EVENTS.private.jsonl') if (r/'EVENTS.private.jsonl').exists() else []
    for kind,table in [('call',counts['calls']),('optimizer',counts['optimizer'])]:
        got=collections.Counter()
        for e in events:
            if e['kind']==kind:got[e['key']]+=e['n']
        assert dict(got)==table
    assert all(v<=CAPS[k] for k,v in counts['calls'].items()) and sum(counts['optimizer'].values())<=8128
    full=status['status']=='COMPLETE';source=(r/'SOURCE_GATE.json').exists();metrics={};tails={};contrasts={};checks={};load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    if source or full:
        assert counts['pending_optimizer'] is None
        reserved=collections.Counter()
        for e in events:
            if e['kind']=='reserved_optimizer':reserved[e['key']]+=e['n']
        assert dict(reserved)==counts['optimizer']
        assert counts['optimizer']['source_F_qualification']==32 and counts['optimizer']['source_F_formal']==2000
        assert counts['calls']['source_formal_samples']==8000 and counts['calls']['source_formal_reverse']==2012 and counts['calls']['source_diagnostic_samples']==192 and counts['calls']['source_evaluation_samples']==544 and counts['calls']['source_risk_calibration_samples']==460
        ini=load(r/'source/F/initial.pt');src=load(r/'source/F/final.pt');finite(src);assert src['step']==src['physical_updates']==2000 and src['reads']==[2024,0] and exact(src,load(r/'source/F/latest.pt'));assert not ini['optimizer']['state'] and src['optimizer']['state']
        assert len(rows(r/'source/F/physical_updates.jsonl'))==2000
        for k,v in ini['model'].items():
            if k.startswith('parent.'):assert exact(v,src['model'][k]) and exact(v,src['ema'][k]),k
        metrics['source']={a:read(r/'source'/a/'RESULTS.json')['metrics'] for a in ('B','F')}
        assert read(r/'qualification/source_F/ENGINEERING.json')['optimizer_calls']==32
        if not full:assert not (r/'target').exists() and not any(k.startswith('target_') for k in counts['optimizer'])
    if full:
        assert read(r/'SOURCE_GATE.json')['status']=='PASS';assert all(read(p)['exit_code']==0 for p in r.glob('EXIT*.json'));assert sum(counts['optimizer'].values())==8128
        assert counts['calls']['target_formal_samples']==24000 and counts['calls']['target_formal_reverse']==6036 and counts['calls']['target_evaluation_samples']==3612 and counts['calls']['target_risk_calibration_samples']==1270
        assert counts['calls']['dual_computations']==8128 and counts['calls']['dual_active_updates']==4064
        original=rows(cfg['original_source_rows']);rr={};metrics['target']={}
        for a in ('B','S','C'):
            assert counts['optimizer']['target_'+a+'_qualification']==32 and counts['optimizer']['target_'+a+'_formal']==2000
            out=r/'target'/a;entry=load(out/'initial.pt');last=load(out/'final.pt');finite(last);assert last['step']==last['physical_updates']==2000 and last['reads']==[2000,0] and exact(last,load(out/'latest.pt'));assert len(rows(out/'physical_updates.jsonl'))==2000
            assert not entry['optimizer']['state'] and last['optimizer']['state'];assert torch.count_nonzero(entry['model']['dual'])==0
            reference=ini['model'] if a=='B' else src['ema']
            for k,v in entry['model'].items():
                if k.startswith(('parent.','adapter.')):assert exact(v,reference[k]),k
                if k.startswith('parent.'):assert exact(v,last['model'][k]) and exact(v,last['ema'][k]),k
            assert read(r/'qualification'/('target_'+a)/'ENGINEERING.json')['optimizer_calls']==32
            settings=read(r/('SETTINGS_target_'+a+'_formal.private.json'));assert Path(settings['cache']).name=='target_'+('B' if a=='B' else 'F')+'.private.json'
            rr[a]={d:rows(out/('private_'+d+'_val.jsonl')) for d in ('Siemens','Philips')};metrics['target'][a]=read(out/'RESULTS.json')['metrics'];tails[a]=dict(own_source=compare(rows(r/'source'/('B' if a=='B' else 'F')/'private_Siemens_val.jsonl'),rr[a]['Siemens']),original_source=compare(original,rr[a]['Siemens']))
        contrasts={a:{d:compare(rr[a][d],rr['C'][d]) for d in ('Siemens','Philips')} for a in ('B','S')};own=tails['C']['own_source'];orig=tails['C']['original_source'];base=tails['B']['own_source']
        checks=dict(old_absolute=metrics['target']['C']['Siemens']['primary_dice']*100>=80.36740663770262,new_absolute=metrics['target']['C']['Philips']['primary_dice']*100>=83,own_source_CI=own['CI95_pp'][0]>=-.3,original_source_CI=orig['CI95_pp'][0]>=-.3,tail_own=tail(own),tail_original=tail(orig),tail_no_worse_B=own['over1']<=base['over1'] and own['over2']<=base['over2'] and orig['over1']<=tails['B']['original_source']['over1'] and orig['over2']<=tails['B']['original_source']['over2'],new_noninferior_B=ni(contrasts['B']['Philips']),new_noninferior_S=ni(contrasts['S']['Philips']),old_increment_B=contrasts['B']['Siemens']['mean_pp']>=.3,old_increment_S=contrasts['S']['Siemens']['mean_pp']>=.15,current_training_risk=read(r/'RISK_TARGET_C_FINAL.json')['status']=='PASS')
    audit=dict(status='PASS',audit_model_forward=0,workers_retired=True,counters=counts,resources=status['resources'],source_complete=source,target_complete=full,synthetic_calls=read(r/'SYNTHETIC_CALLS.json') if (r/'SYNTHETIC_CALLS.json').exists() else {})
    result=dict(status=status['status'],audit=audit,source_gate=read(r/'SOURCE_GATE.json') if source else None,metrics=metrics,patient_tails=tails,contrasts_C_minus=contrasts,checks=checks,composite_pass=full and all(checks.values()),primary='C final EMA .999',single_seed=20271801,development_validation_reused=True,independent_test=False)
    write(r/'AUDIT.json',audit);write(r/'PUBLIC_SUMMARY.json',result)
    import json
    (r/'NATIVE_RISK_V1_REPORT_ZH.md').write_text('# 源风险约束原生解剖等变 V1 完整结果\n\n'+('综合科学'+('PASS' if result['composite_pass'] else 'FAIL') if full else '源/工程门禁停止；没有适配性能结论')+'；一次零forward审计PASS。\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\n单种子、共享源、重复开发患者；训练代理风险不是独立患者Dice保证。不重训/重评/重审计。\n')
