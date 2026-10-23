"""One terminal, zero-model-forward audit and patient-paired scientific report."""
import collections,json,subprocess
from pathlib import Path
import numpy as np
import torch
from .runtime import CAPS
from ..main_style_ssl_v1.runtime import write
from ..gyro_spatial_memory_v1.runtime import exact,finite


def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(line) for line in Path(p).read_text().splitlines()]
def patient_map(rr):
    d={}
    for r in rr:
        v=d.setdefault(r['patient'],{});assert r['phase'] not in v
        v[r['phase']]=float(np.mean(r['dice']))
    assert all(set(v)=={'ED','ES'} for v in d.values())
    return {k:float(np.mean(list(v.values()))) for k,v in d.items()}
def compare(a,b):
    aa,bb=patient_map(a),patient_map(b);assert aa.keys()==bb.keys()
    v=np.array([(bb[k]-aa[k])*100 for k in sorted(aa)])
    rng=np.random.default_rng(163);sample=rng.integers(0,len(v),(10000,len(v)))
    return dict(mean_pp=float(v.mean()),CI95_pp=np.quantile(v[sample].mean(1),[.025,.975]).tolist(),p10_pp=float(np.quantile(v,.1)),decline_over_1=int((v < -1).sum()),decline_over_2=int((v < -2).sum()),patients=len(v))
def noninferior(v):return v['mean_pp']>=-.20 and v['CI95_pp'][0]>=-.20

def main(cfg):
    root=Path(cfg['root']);assert not (root/'AUDIT.json').exists()
    status=read(root/'STATUS.json');assert status['status'] in ('COMPLETE','STOPPED')
    for p in [root/'LAUNCH.private.json',root/'WORKER.private.json',*root.glob('ACTIVE_*.private.json')]:
        info=read(p);stat=Path('/proc')/str(info['pid'])/'stat'
        assert not stat.exists() or int(stat.read_text().split()[21])!=info['birth'] or stat.read_text().split()[2]=='Z'
    counters=read(root/'COUNTERS.private.json') if (root/'COUNTERS.private.json').exists() else dict(calls={},optimizer={})
    assert all(n<=CAPS[k] for k,n in counters['calls'].items())
    events=rows(root/'EVENTS.private.jsonl') if (root/'EVENTS.private.jsonl').exists() else []
    for kind,table in [('call',counters['calls']),('optimizer',counters['optimizer'])]:
        counts=collections.Counter()
        for e in events:
            if e['event']==kind:counts[e['key']]+=e['n']
        assert dict(counts)==table
    full=status['status']=='COMPLETE';metrics={};contrasts={};tail={};checks={}
    if full:
        assert all(read(p)['exit_code']==0 for p in root.glob('EXIT*.json'))
        expected={stage+'_'+arm+'_'+phase:n for stage in ('source','target') for arm in ('D','G','T') for phase,n in [('qualification',32),('formal',2000)]}
        assert counters['optimizer']==expected and sum(expected.values())==12192
        assert counters['calls']['source_formal_samples']==counters['calls']['target_formal_samples']==12000
        assert counters['calls']['source_formal_reverse']==counters['calls']['target_formal_reverse']==6000
        assert counters['calls']['evaluation_samples']==5244
        original=rows(cfg['original_source_rows']);assert len(patient_map(original))==23
        assert abs(np.mean(list(patient_map(original).values()))*100-80.36740663770262)<1e-9
        rr={};load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
        for stage in ('source','target'):
            rr[stage]={};metrics[stage]={}
            for arm in ('D','G','T'):
                out=root/stage/arm;initial=load(out/'initial.pt');end=load(out/'final.pt')
                finite(end);assert end['step']==end['physical_updates']==2000 and end['reads']==[2000,0]
                assert exact(end,load(out/'latest.pt')) and not initial['optimizer']['state'] and end['optimizer']['state']
                assert len(rows(out/'physical_updates.jsonl'))==2000
                for key,value in initial['model'].items():
                    fixed=key.startswith(('prior_','quadrature_','parent.native.decoder.conv_logit.')) or (stage=='target' and key.startswith(('parent.','detail.')))
                    if fixed:assert exact(value,end['model'][key]) and exact(value,end['ema'][key]),key
                if stage=='target':assert exact(initial['model'],load(root/'source'/arm/'final.pt')['ema'])
                assert read(root/'qualification'/(stage+'_'+arm)/'ENGINEERING.json')['actual_optimizer_calls']==32
                metrics[stage][arm]=read(out/'RESULTS.json')['metrics'];rr[stage][arm]={}
                for domain in (('Siemens',) if stage=='source' else ('Siemens','Philips')):
                    values=rows(out/('private_'+domain+'_val.jsonl'))
                    assert len(patient_map(values))==(23 if domain=='Siemens' else 31)
                    assert abs(np.mean(list(patient_map(values).values()))-metrics[stage][arm][domain]['primary_dice'])<1e-10
                    rr[stage][arm][domain]=values
        for arm in ('D','G','T'):
            tail[arm]=dict(own_source=compare(rr['source'][arm]['Siemens'],rr['target'][arm]['Siemens']),original_source=compare(original,rr['target'][arm]['Siemens']))
        contrasts={other:{d:compare(rr['target'][other][d],rr['target']['T'][d]) for d in ('Siemens','Philips')} for other in ('D','G')}
        old=tail['T']['own_source'];orig=tail['T']['original_source'];baseline=tail['D']['own_source']
        checks=dict(old_absolute=metrics['target']['T']['Siemens']['primary_dice']*100>=80.36740663770262,old_own_source_CI=old['CI95_pp'][0]>=-.30,new_absolute=metrics['target']['T']['Philips']['primary_dice']*100>=83,new_noninferior_D=noninferior(contrasts['D']['Philips']),old_increment_D=contrasts['D']['Siemens']['mean_pp']>=.30,old_increment_G=contrasts['G']['Siemens']['mean_pp']>=.15,noninferior_G_both=all(noninferior(v) for v in contrasts['G'].values()),tail_own=old['decline_over_1']<=2 and old['decline_over_2']==0 and old['p10_pp']>=-.50,tail_original=orig['decline_over_1']<=2 and orig['decline_over_2']==0 and orig['p10_pp']>=-.50,tail_no_worse_D=old['decline_over_1']<=baseline['decline_over_1'] and old['decline_over_2']<=baseline['decline_over_2'])
    resource=status['resources'];assert resource['elapsed_seconds']<28800 and resource['allocated_bytes_including_ops']<4000000000
    audit=dict(status='PASS',completed=full,workers_retired=True,audit_model_forward=0,counters=counters,resources=resource)
    result=dict(status='COMPLETE' if full else 'ENGINEERING_STOP',primary='T final EMA .999',audit=audit,metrics=metrics,contrasts_T_minus=contrasts,patient_tails=tail,checks=checks,composite_pass=full and all(checks.values()),single_seed=20271601,independent_test=False,development_patients_reused=True)
    write(root/'PUBLIC_SUMMARY.json',result)
    report=['# APOS V1 完整结果','','单种子探索性验证；工程审计PASS。科学综合：'+('PASS' if result['composite_pass'] else 'FAIL' if full else '未形成完整性能结论')+'。','','旧域患者尾部与新域非劣须同时通过；全部对照和两个源参照均保留。','', '```json',json.dumps(result,ensure_ascii=False,indent=2),'```','','共享原始源随机与反复开发23/31患者。CI不覆盖源训练随机、多重比较或独立测试；后验求积、有限变分及结构有效性均有近似，不作无遗忘或临床保证。']
    (root/'APOS_REPORT_ZH.md').write_text('\n'.join(report)+'\n');write(root/'AUDIT.json',audit)
