"""Exactly one zero-forward terminal audit; failed and cancelled costs retained."""
import collections,json
from pathlib import Path
import torch
from .runtime import CAPS,SYNTH_CAPS
from .analysis import read,rows,compare,tail,ni
from ..gyro_spatial_memory_v1.runtime import exact,finite
from ..main_style_ssl_v1.runtime import write

def main(cfg):
    torch.set_num_threads(4);root=Path(cfg['root']);assert not (root/'AUDIT.json').exists();status=read(root/'STATUS.json');assert status['status'] in ('COMPLETE','SOURCE_STOP','P0_STOP','NUMERIC_STOP','ENGINEERING_STOP')
    for p in [*root.glob('*LAUNCH*.private.json'),root/'WORKER.private.json',*root.glob('ACTIVE_*.private.json')]:
        if not p.exists():continue
        d=read(p);s=Path('/proc')/str(d.get('pid',d.get('timeoutpg')))/'stat';assert not s.exists() or int(s.read_text().split()[21])!=d['birth'] or s.read_text().split()[2]=='Z'
    count=read(root/'COUNTERS.private.json') if (root/'COUNTERS.private.json').exists() else dict(calls={},optimizer={},pending_optimizer=None);events=rows(root/'EVENTS.private.jsonl') if (root/'EVENTS.private.jsonl').exists() else []
    for kind,table in [('call',count['calls']),('optimizer',count['optimizer'])]:
        got=collections.Counter()
        for e in events:
            if e['kind']==kind:got[e['key']]+=e['n']
        assert dict(got)==table
    assert all(v<=CAPS[k] for k,v in count['calls'].items()) and sum(count['optimizer'].values())<=6096
    if status['status']!='ENGINEERING_STOP':
        assert count.get('pending_optimizer') is None;reserved=collections.Counter()
        for e in events:
            if e['kind']=='reserved_optimizer':reserved[e['key']]+=e['n']
        assert dict(reserved)==count['optimizer']
        assert all(read(p)['exit_code']==0 for p in root.glob('EXIT*.json'))
    synth=read(root/'SYNTHETIC_COUNTERS.json') if (root/'SYNTHETIC_COUNTERS.json').exists() else {};assert all(v<=SYNTH_CAPS[k] for k,v in synth.items())
    full=status['status']=='COMPLETE';gate=read(root/'SOURCE_GATE.json') if (root/'SOURCE_GATE.json').exists() else None;metrics={};tails={};contrasts={};checks={}
    if gate:
        metrics['source']={a:read(root/'source'/a/'RESULTS.json')['metrics'] for a in ('N','Q','R')};assert count['calls']['source_evaluation_samples']==2176
    if status['status'] in ('SOURCE_STOP','NUMERIC_STOP'):assert not (root/'target').exists() and not count['optimizer']
    if full:
        assert gate['status']=='PASS' and read(root/'P0_RESULT.json')['status']=='PASS';assert sum(count['optimizer'].values())==6096 and count['calls']['target_formal_samples']==24000 and count['calls']['target_formal_reverse']==6018 and count['calls']['target_evaluation_samples']==7224
        load=lambda p:torch.load(p,map_location='cpu',weights_only=False);common=load(Path(cfg['reuse_root'])/'source/CAL/final.pt')['ema'];rr={};metrics['target']={}
        for arm in ('N','Q','R'):
            out=root/'target'/arm;ini=load(out/'initial.pt');last=load(out/'final.pt');finite(last);assert last['step']==last['physical_updates']==2000 and last['reads']==[2000,0] and exact(last,load(out/'latest.pt')) and not ini['optimizer']['state'] and last['optimizer']['state']
            assert count['optimizer']['target_'+arm+'_qualification']==32 and count['optimizer']['target_'+arm+'_formal']==2000
            for k,v in common.items():
                if k!='local_updates':assert exact(v,ini['model'][k]),k
            for k,v in ini['model'].items():
                if not k.startswith(('adapter_','local_updates')):assert exact(v,last['model'][k]) and exact(v,last['ema'][k]),k
            rr[arm]={d:rows(out/('private_'+d+'_val.jsonl')) for d in ('Siemens','Philips')};metrics['target'][arm]=read(out/'RESULTS.json')['metrics'];tails[arm]=compare(rows(cfg['original_source_rows']),rr[arm]['Siemens'])
        contrasts={other:{d:compare(rr[other][d],rr['R'][d]) for d in ('Siemens','Philips')} for other in ('N','Q')};v=tails['R'];b=tails['N']
        checks=dict(old_absolute=metrics['target']['R']['Siemens']['primary_dice']*100>=80.36740663770262,new_absolute=metrics['target']['R']['Philips']['primary_dice']*100>=83,old_source_CI=v['CI95_pp'][0]>=-.3,tail_own_and_original=tail(v),tail_no_worse_N=v['over1']<=b['over1'] and v['over2']<=b['over2'],new_noninferior_N=ni(contrasts['N']['Philips']),old_increment_N=contrasts['N']['Siemens']['mean_pp']>=.3,old_increment_Q=contrasts['Q']['Siemens']['mean_pp']>=.15,noninferior_Q_both=all(ni(x) for x in contrasts['Q'].values()))
    audit=dict(status='PASS',audit_model_forward=0,workers_retired=True,counters=count,resources=status['resources'],source_gate_complete=gate is not None,target_complete=full)
    result=dict(status=status['status'],audit=audit,source_gate=gate,P0=read(root/'P0_RESULT.json') if (root/'P0_RESULT.json').exists() else None,metrics=metrics,patient_tails_both_own_and_original=tails,contrasts_R_minus=contrasts,checks=checks,composite_pass=full and all(checks.values()),synthetic_counts=synth,primary='R final EMA .999',single_seed=20271611,development_validation_reused=True,independent_test=False)
    write(root/'PUBLIC_SUMMARY.json',result);write(root/'AUDIT.json',audit);(root/'APOS_JOINT_V1_REPORT_ZH.md').write_text('# APOS Joint V1 完整结果\n\n工程审计PASS；'+('综合科学'+('PASS' if result['composite_pass'] else 'FAIL') if full else '未完成正式适配，不能作完整目标性能结论')+'。\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\n单种子、源23/新31重复开发患者；保留原始源参照，门禁和失败不回填。不重训、不重评、不重审计。\n')
