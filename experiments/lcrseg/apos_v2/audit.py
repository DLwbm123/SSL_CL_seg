"""One zero-forward terminal audit for COMPLETE, source stop, or engineering stop."""
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
        d=read(p);s=Path('/proc')/str(d['pid'])/'stat';assert not s.exists() or int(s.read_text().split()[21])!=d['birth'] or s.read_text().split()[2]=='Z'
    count=read(root/'COUNTERS.private.json') if (root/'COUNTERS.private.json').exists() else dict(calls={},optimizer={})
    events=rows(root/'EVENTS.private.jsonl') if (root/'EVENTS.private.jsonl').exists() else []
    for kind,table in [('call',count['calls']),('optimizer',count['optimizer'])]:
        got=collections.Counter()
        for e in events:
            if e['kind']==kind:got[e['key']]+=e['n']
        assert dict(got)==table
    assert all(v<=CAPS[k] for k,v in count['calls'].items()) and sum(count['optimizer'].values())<=12128
    full=status['status']=='COMPLETE';source_complete=(root/'SOURCE_GATE.json').exists();metrics={};tails={};contrasts={};checks={}
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    if source_complete:
        assert count['optimizer']['source_CAL_qualification']==32 and count['optimizer']['source_CAL_formal']==6000
        assert count['calls']['source_formal_samples']==12000 and count['calls']['source_formal_reverse']==6000 and count['calls']['source_diagnostic_samples']==768 and count['calls']['source_evaluation_samples']==1088
        assert len(rows(root/'SOURCE_DIAGNOSTICS.jsonl'))==12
        common=load(root/'source/CAL/final.pt');initial=load(root/'source/CAL/initial.pt');finite(common)
        assert common['step']==common['physical_updates']==6000 and common['reads']==[6192,0] and exact(common,load(root/'source/CAL/latest.pt'))
        assert int(common['model']['source_error_cells'])==6000*2*1024 and int(common['ema']['local_updates'])==6000
        assert not initial['optimizer']['state'] and common['optimizer']['state']
        for k,v in initial['model'].items():
            if k.startswith(('parent.','prior_','adapter_')):assert exact(v,common['model'][k]) and exact(v,common['ema'][k]),k
        metrics['source']={a:read(root/'source'/a/'RESULTS.json')['metrics'] for a in ('N','Q','R')}
        if not full:assert not (root/'target').exists() and not any(k.startswith('target_') for k in count['optimizer'])
    if full:
        assert read(root/'SOURCE_GATE.json')['status']=='PASS' and all(read(p)['exit_code']==0 for p in root.glob('EXIT*.json'))
        assert sum(count['optimizer'].values())==12128 and count['calls']['target_formal_samples']==12000 and count['calls']['target_formal_reverse']==6000 and count['calls']['target_evaluation_samples']==3612
        original=rows(cfg['original_source_rows']);rr={};metrics['target']={}
        for arm in ('N','Q','R'):
            out=root/'target'/arm;ini=load(out/'initial.pt');last=load(out/'final.pt');finite(last)
            assert last['step']==last['physical_updates']==2000 and last['reads']==[2000,0] and not ini['optimizer']['state'] and last['optimizer']['state']
            assert exact(last,load(out/'latest.pt')) and len(rows(out/'physical_updates.jsonl'))==2000
            for k,v in common['ema'].items():
                if k!='local_updates':assert exact(v,ini['model'][k]),k
            for k,v in ini['model'].items():
                if k.startswith(('parent.','prior_','reference_mse','source_error_')):assert exact(v,last['model'][k]) and exact(v,last['ema'][k]),k
            rr[arm]={d:rows(out/('private_'+d+'_val.jsonl')) for d in ('Siemens','Philips')};metrics['target'][arm]=read(out/'RESULTS.json')['metrics']
            tails[arm]=dict(own_source=compare(rows(root/'source'/arm/'private_Siemens_val.jsonl'),rr[arm]['Siemens']),original_source=compare(original,rr[arm]['Siemens']))
        contrasts={other:{d:compare(rr[other][d],rr['R'][d]) for d in ('Siemens','Philips')} for other in ('N','Q')}
        own=tails['R']['own_source'];orig=tails['R']['original_source'];base=tails['N']['own_source']
        checks=dict(old_absolute=metrics['target']['R']['Siemens']['primary_dice']*100>=80.36740663770262,new_absolute=metrics['target']['R']['Philips']['primary_dice']*100>=83,old_source_CI=own['CI95_pp'][0]>=-.3,tail_own=tail(own),tail_original=tail(orig),tail_no_worse_N=own['over1']<=base['over1'] and own['over2']<=base['over2'],new_noninferior_N=ni(contrasts['N']['Philips']),old_increment_N=contrasts['N']['Siemens']['mean_pp']>=.3,old_increment_Q=contrasts['Q']['Siemens']['mean_pp']>=.15,noninferior_Q_both=all(ni(x) for x in contrasts['Q'].values()))
    audit=dict(status='PASS',audit_model_forward=0,workers_retired=True,counters=count,resources=status['resources'],source_complete=source_complete,target_complete=full)
    result=dict(status=status['status'],audit=audit,source_gate=read(root/'SOURCE_GATE.json') if source_complete else None,metrics=metrics,patient_tails=tails,contrasts_R_minus=contrasts,checks=checks,composite_pass=full and all(checks.values()),primary='R final EMA .999',single_seed=20271611,development_validation_reused=True,independent_test=False)
    write(root/'PUBLIC_SUMMARY.json',result);write(root/'AUDIT.json',audit)
    text=['# APOS V2 完整结果','','工程审计PASS；'+('综合科学'+('PASS' if result['composite_pass'] else 'FAIL') if full else '未进入/未完成适配，不能作目标性能结论')+'。','','```json',json.dumps(result,ensure_ascii=False,indent=2),'```','','单种子、共享源及重复开发患者；源门禁只作预注册停止，不以局部结果调参。失败成本全部保留，不重训重评重审计。']
    (root/'APOS_V2_REPORT_ZH.md').write_text('\n'.join(text)+'\n')
