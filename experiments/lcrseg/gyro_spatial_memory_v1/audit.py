"""One completion audit of this round; never infer results from a live worker."""
import json,time
from collections import Counter
from pathlib import Path
import torch
from .runtime import finite,exact,guard
from ..main_style_ssl_v1.runtime import write
from ..siemens_lonly_v1.run import compare


def read(path):return json.loads(Path(path).read_text())
def checkpoint(path):return torch.load(path,map_location='cpu',weights_only=False)
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main(cfg):
    root=Path(cfg['root'])
    if (root/'AUDIT.json').exists():raise FileExistsError('completion audit already recorded; do not repeat')
    assert read(root/'STATUS.json')['status']=='COMPLETE'
    for path in root.glob('ACTIVE_*.json'):
        active=read(path);proc=Path('/proc')/str(active['pid'])/'stat'
        assert not proc.exists() or proc.read_text().split()[21]!=str(active['birth']),'worker has not retired'
    counter=read(root/'budget_counter.json');assert counter['pending']==0 and counter['total']<=11783
    events=rows(root/'budget_events.jsonl');executed=Counter((r['stage']+':'+r['phase']) for r in events if r['kind'] in ('optimizer','functional'))
    assert dict(executed)==counter['by_phase']
    assert counter['total']==3+sum(executed.values())
    assert counter['optimizer']==sum(r['kind']=='optimizer' for r in events)
    assert counter['functional']==3+sum(r['kind']=='functional' for r in events)
    codec_initial=checkpoint(root/'codec/initial.pt')['model'];audit={}
    for stage,folder,steps in [('codec',root/'codec',500),('C0',root/'C0',256),('C1',root/'C1',256)]:
        assert read(folder/'EXIT.json')['exit_code']==0
        assert read(root/'qualification'/stage/'ENGINEERING.json')['status']=='PASS'
        initial=checkpoint(folder/'initial.pt');final=checkpoint(folder/'final.pt');latest=checkpoint(folder/'latest.pt')
        finite(final);assert exact(final,latest) and final['step']==steps and not initial['optimizer']['state']
        assert all(torch.equal(v,initial['model'][k]) for k,v in final['model'].items() if k.startswith('parent.') or k=='frames')
        if stage!='codec':
            assert all(torch.equal(v,initial['model'][k]) for k,v in final['model'].items() if k.startswith(('projection.','generator.')))
            memory=checkpoint(folder/'memory.pt');assert not memory['historical_images_retained']
            assert read(folder/'MEMORY.json')['bytes']<=8*1024**2
        assert executed[stage+':formal']==(500 if stage=='codec' else 512)
        audit[stage]=dict(steps=steps,native_and_frame_exact=True,final_latest_exact=True,finite=True)
    metrics={};arms=[a for a in ('A','B','C','D','E') if (root/'jobs'/a/'RESULTS.json').exists()]
    assert arms[:3]==['A','B','C']
    patient_sets={}
    for arm in arms:
        out=root/'jobs'/arm;assert read(out/'EXIT.json')['exit_code']==read(out/'EVALUATION_EXIT.json')['exit_code']==0
        initial=checkpoint(out/'initial.pt');final=checkpoint(out/'final.pt');latest=checkpoint(out/'latest.pt')
        finite(final);assert final['step']==final['physical_updates']==2000 and exact(final,latest) and not initial['optimizer']['state']
        assert executed[arm+':formal']==2000
        assert all(torch.equal(v,codec_initial[k]) for k,v in initial['model'].items() if k.startswith('parent.'))
        assert all(torch.equal(v,initial['model'][k]) and torch.equal(v,final['ema'][k]) for k,v in final['model'].items() if not k.startswith('parent.'))
        assert len(rows(out/'physical_updates.jsonl'))==2000
        assert read(root/'qualification'/arm/'ENGINEERING.json')['actual_calls']==32
        for domain,num in [('Siemens',23),('Philips',31)]:
            rr=rows(out/('private_'+domain+'_val.jsonl'));keys={(r['patient'],r['phase']) for r in rr}
            assert len(keys)==2*num and len({k[0] for k in keys})==num
            if domain in patient_sets:assert keys==patient_sets[domain]
            patient_sets[domain]=keys
        result=read(out/'RESULTS.json');assert result['source_train']['patients']==10 and result['target_train']['patients']==13
        metrics[arm]={k:result[k] for k in ('Siemens','Philips','source_train','target_train','old_delta_from_entry_pp','target_delta_from_entry_pp','paired_old_entry','paired_target_entry')}
        audit[arm]=dict(formal_calls=2000,common_native_entry=True,empty_optimizer=True,frozen_memory=True,final_latest_exact=True,finite=True,patients_EDES_exact=True)
    paired={}
    for a,b in [('A','B'),('B','C'),('A','C'),('D','C'),('E','C')]:
        if a in arms and b in arms:
            paired[b+'-'+a]={d:compare(rows(root/'jobs'/a/('private_'+d+'_val.jsonl')),rows(root/'jobs'/b/('private_'+d+'_val.jsonl'))) for d in ('Siemens','Philips')}
    decision=read(root/'ABC_DECISION.json')
    assert decision['run_DE']==('D' in arms and 'E' in arms)
    result=dict(status='PASS',cost=counter,stages=audit,workers_retired=True,raw_replay=False,storage=guard(cfg),completed_arms=arms,DE_gate=decision)
    write(root/'AUDIT.json',result)
    write(root/'PUBLIC_SUMMARY.json',dict(audit=result,metrics=metrics,paired=paired,primary='fixed last student',exploratory=True,independent_test_used=False))
    report=['# GYRO空间凝缩闭环：固定终点实验报告','', '所有结果为探索性单seed；无历史原图回放。凝缩几何采用预声明PIL直通近似，任务效用保留二阶超梯度。','', '|组|Siemens验证%|Philips验证%|源10L训练%|目标13L训练%|','|---|---:|---:|---:|---:|']
    for arm,m in metrics.items():report.append('|'+arm+'|'+'|'.join(f"{100*m[k]['primary_dice']:.6f}" for k in ('Siemens','Philips','source_train','target_train'))+'|')
    report+=['','共同入口：Siemens79.4337143%、Philips73.2043174%。旧源80.3674%至本轮入口的约−.9337pp接入损失另列，不混入本轮遗忘。','', 'A/B/C主判定：'+json.dumps(decision,ensure_ascii=False),'','未运行D/E时不能归因成对对应或集合闭环。没有C1/R0及noNewU单元，不能宣称完整形成×读取交互或U语义收益。','', '各器官、ED/ES、绝对变化、患者改善/下降与中位差及配对结果见完整匿名聚合JSON；不以双域均值掩盖单域代价。','', '## 完整匿名聚合','```json',json.dumps(dict(metrics=metrics,paired=paired,audit=result),ensure_ascii=False,indent=2),'```']
    (root/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
