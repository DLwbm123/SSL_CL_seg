"""Single terminal, saved-evidence-only audit; never instantiates a model."""
import collections,json
from pathlib import Path
import torch
from .runtime import CAPS
from .analysis import read,rows,compare,tail,ni
from ..main_style_ssl_v1.runtime import write
from ..gyro_spatial_memory_v1.runtime import exact,finite

def main(cfg):
    torch.set_num_threads(4);root=Path(cfg['root']);assert not (root/'AUDIT.json').exists();status=read(root/'STATUS.json');assert status['status'] in ('COMPLETE','SOURCE_STOP','NUMERIC_STOP','ENGINEERING_STOP')
    for p in [*root.glob('*LAUNCH*.private.json'),root/'WORKER.private.json',*root.glob('ACTIVE_*.private.json')]:
        if not p.exists():continue
        d=read(p);pid=d.get('pid',d.get('timeoutpg'));s=Path('/proc')/str(pid)/'stat'
        assert not s.exists() or int(s.read_text().split()[21])!=d['birth'] or s.read_text().split()[2]=='Z'
    count=read(root/'COUNTERS.private.json') if (root/'COUNTERS.private.json').exists() else dict(calls={},optimizer={},pending_optimizer=None)
    events=rows(root/'EVENTS.private.jsonl') if (root/'EVENTS.private.jsonl').exists() else []
    for kind,table in [('call',count['calls']),('optimizer',count['optimizer'])]:
        got=collections.Counter()
        for e in events:
            if e['kind']==kind:got[e['key']]+=e['n']
        assert dict(got)==table
    assert all(v<=CAPS[k] for k,v in count['calls'].items()) and sum(count['optimizer'].values())<=10160
    full=status['status']=='COMPLETE';source_complete=(root/'SOURCE_GATE.json').exists();metrics={};tails={};contrasts={};checks={}
    if full or source_complete:
        assert count.get('pending_optimizer') is None
        reserved=collections.Counter()
        for e in events:
            if e['kind']=='reserved_optimizer':reserved[e['key']]+=e['n']
        assert dict(reserved)==count['optimizer']
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    sources={}
    if source_complete:
        assert count['calls']['source_formal_samples']==16000 and count['calls']['source_formal_reverse']==4012 and count['calls']['source_diagnostic_samples']==384 and count['calls']['source_evaluation_samples']==1088
        metrics['source']={}
        for arm in ('B','F'):
            assert count['optimizer']['source_'+arm+'_qualification']==32 and count['optimizer']['source_'+arm+'_formal']==2000
            out=root/'source'/arm;initial=load(out/'initial.pt');last=load(out/'final.pt');finite(last);assert last['step']==last['physical_updates']==2000 and last['reads']==[2024,0] and exact(last,load(out/'latest.pt'))
            assert not initial['optimizer']['state'] and last['optimizer']['state'];assert len(rows(out/'physical_updates.jsonl'))==2000
            for k,v in initial['model'].items():
                if k.startswith('parent.native.decoder.conv_logit.') or (k.startswith('parent.') and k not in dict(last['model'])):assert exact(v,last['model'][k]) and exact(v,last['ema'][k]),k
            assert read(root/'qualification'/('source_'+arm)/'ENGINEERING.json')['optimizer_calls']==32
            sources[arm]=last;metrics['source'][arm]=read(out/'RESULTS.json')['metrics']
        if not full:assert not (root/'target').exists() and not any(k.startswith('target_') for k in count['optimizer'])
    if full:
        assert read(root/'SOURCE_GATE.json')['status']=='PASS' and all(read(p)['exit_code']==0 for p in root.glob('EXIT*.json'))
        assert sum(count['optimizer'].values())==10160 and count['calls']['target_formal_samples']==24000 and count['calls']['target_formal_reverse']==6018 and count['calls']['target_evaluation_samples']==3612
        original=rows(cfg['original_source_rows']);rr={};metrics['target']={}
        for arm in ('B','S','C'):
            assert count['optimizer']['target_'+arm+'_qualification']==32 and count['optimizer']['target_'+arm+'_formal']==2000
            out=root/'target'/arm;ini=load(out/'initial.pt');last=load(out/'final.pt');finite(last);src=sources['B' if arm=='B' else 'F']
            assert last['step']==last['physical_updates']==2000 and last['reads']==[2000,0] and not ini['optimizer']['state'] and last['optimizer']['state'];assert exact(last,load(out/'latest.pt')) and len(rows(out/'physical_updates.jsonl'))==2000
            for k,v in src['ema'].items():
                if k!='local_updates':assert exact(v,ini['model'][k]),k
            for k,v in ini['model'].items():
                if k.startswith(('parent.','anatomy.')):assert exact(v,last['model'][k]) and exact(v,last['ema'][k]),k
            assert read(root/'qualification'/('target_'+arm)/'ENGINEERING.json')['optimizer_calls']==32
            rr[arm]={d:rows(out/('private_'+d+'_val.jsonl')) for d in ('Siemens','Philips')};metrics['target'][arm]=read(out/'RESULTS.json')['metrics']
            tails[arm]=dict(own_source=compare(rows(root/'source'/('B' if arm=='B' else 'F')/'private_Siemens_val.jsonl'),rr[arm]['Siemens']),original_source=compare(original,rr[arm]['Siemens']))
        contrasts={other:{d:compare(rr[other][d],rr['C'][d]) for d in ('Siemens','Philips')} for other in ('B','S')}
        own=tails['C']['own_source'];orig=tails['C']['original_source'];base=tails['B']['own_source']
        checks=dict(old_absolute=metrics['target']['C']['Siemens']['primary_dice']*100>=80.36740663770262,new_absolute=metrics['target']['C']['Philips']['primary_dice']*100>=83,own_source_CI=own['CI95_pp'][0]>=-.3,original_source_CI=orig['CI95_pp'][0]>=-.3,tail_own=tail(own),tail_original=tail(orig),tail_no_worse_B=own['over1']<=base['over1'] and own['over2']<=base['over2'] and orig['over1']<=tails['B']['original_source']['over1'] and orig['over2']<=tails['B']['original_source']['over2'],new_noninferior_B=ni(contrasts['B']['Philips']),old_increment_B=contrasts['B']['Siemens']['mean_pp']>=.3,old_increment_S=contrasts['S']['Siemens']['mean_pp']>=.15,new_noninferior_S=ni(contrasts['S']['Philips']))
    audit=dict(status='PASS',audit_model_forward=0,workers_retired=True,counters=count,resources=status['resources'],source_complete=source_complete,target_complete=full,synthetic_calls=read(root/'SYNTHETIC_CALLS.json') if (root/'SYNTHETIC_CALLS.json').exists() else {})
    result=dict(status=status['status'],audit=audit,source_gate=read(root/'SOURCE_GATE.json') if source_complete else None,metrics=metrics,patient_tails=tails,contrasts_C_minus=contrasts,checks=checks,composite_pass=full and all(checks.values()),primary='C final EMA .999',single_seed=20271701,development_validation_reused=True,independent_test=False)
    write(root/'PUBLIC_SUMMARY.json',result);write(root/'AUDIT.json',audit)
    text=['# 外观逆校正与解剖表示 V1 完整结果','','工程审计PASS；'+('综合科学'+('PASS' if result['composite_pass'] else 'FAIL') if full else '未进入/未完成适配，不能作适配性能结论')+'。','','```json',json.dumps(result,ensure_ascii=False,indent=2),'```','','单种子、共享源及重复开发患者。训练稳定门禁不证明全局最优；外观扰动不等于真实扫描仪变化。适配不读取旧图。失败成本保留，不重训重评重审计。']
    (root/'ACQUISITION_ANATOMY_V1_REPORT_ZH.md').write_text('\n'.join(text)+'\n')
