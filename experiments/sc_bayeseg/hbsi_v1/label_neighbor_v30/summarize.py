"""Sealed N plus matched historical A/B; anonymous offline summaries only."""
import argparse,json
from pathlib import Path
from statistics import mean
from ..trainability_v21.summarize import patients,METRICS


def analyze(x):
    assert x['RESULTS.json']['preflight_reverse_calls']==1 and x['RESULTS.json']['training_audit_reverse_calls']==6
    old=x['controls_v25']
    for snapshot,arms,steps in ((old,'ABC',1200),(x,'N',400)):
        assert snapshot['STATUS.json']['stage']=='COMPLETE' and snapshot['EXIT.json']['exit_code']==0
        assert snapshot['RESULTS.json']['new_formal_updates']==steps
        assert snapshot['RESULTS.json']['counter']['outer_backward_calls']==steps
        assert len(snapshot['RESULTS.json']['cells'])==2*len(arms)
        assert all(c['complete'] and c['patients']==4 for c in snapshot['RESULTS.json']['cells'].values())
        assert len(snapshot['ledger']['training_sessions'])==len(arms)
        assert all(s['completed_steps']==2000 and s['status']=='COMPLETE' for s in snapshot['ledger']['training_sessions'].values())
        assert all(a['status']=='PASS' for a in snapshot['ledger']['attempts'])
    assert x['MATCHING.json']['status']=='PASS' and x['PREFLIGHT.json']['status']=='PASS'
    for field in ('common_payload','prior_hash','teacher_payload'):
        assert len({snapshot[f'BRANCH_{arm}.json'][field] for snapshot,arms in ((old,'AB'),(x,'N')) for arm in arms})==1
    assert sum(r['extra_reverse_calls'] for arm in 'N' for r in x['audit'][arm])==6
    assert all([r['step'] for r in x['audit'][arm]]==[1600,1799,1999] for arm in 'N')
    pp={arm:{d:patients(rows) for d,rows in snapshot['patients'][arm].items()} for snapshot,arms in ((old,'AB'),(x,'N')) for arm in arms}
    means={arm:{d:{k:(sum(v[k] for v in rows.values()) if k=='empty_predictions' else mean(v[k] for v in rows.values())) for k in METRICS} for d,rows in ds.items()} for arm,ds in pp.items()}
    effects={}
    for arm,control in (('N','A'),('N','B')):
        ds={}
        for d in ('RUNMC','BMC'):
            assert pp[arm][d].keys()==pp[control][d].keys()
            metrics={k:means[arm][d][k]-means[control][d][k] for k in METRICS};agg={}
            for k in METRICS:
                v=[pp[arm][d][i][k]-pp[control][d][i][k] for i in pp[arm][d]]
                agg[k]=dict(n=4,mean=mean(v),minimum=min(v),maximum=max(v),positive=sum(t>1e-12 for t in v),negative=sum(t< -1e-12 for t in v))
            ds[d]=dict(metrics=metrics,patient_aggregates=agg)
        checks=dict(source_gain_ge_1pp=ds['RUNMC']['metrics']['Dice']*100>=1,transfer_drop_le_0_5pp=ds['BMC']['metrics']['Dice']*100>=-.5,no_new_empty=all(means[arm][d]['empty_predictions']<=means[control][d]['empty_predictions'] for d in ds),HD95_worsening_le_10pct=all(means[arm][d]['HD95_penalized_mm']<=1.1*means[control][d]['HD95_penalized_mm'] for d in ds))
        effects[arm+'_minus_'+control]=dict(domains=ds,gate=dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks))
    return dict(status='COMPLETE',seed=20261023,training_seeds=1,patients_per_domain=4,inference_repeats=3,means=means,effects=effects,training=x['training'],audit=x['audit'],matching=x['MATCHING.json'],reused_controls=['v25_A2000','v25_B2000'])


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');args=p.parse_args();x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    assert x['TARGET_SUMMARY.json']['source_unchanged'] and not x['TARGET_SUMMARY.json']['U_labels_read']
    a['target_summary']=x['TARGET_SUMMARY.json']
    v23=json.loads((out.parent/'semantic_alignment_v23/ANALYSIS.json').read_text())
    a['relative_to_v23_L1600_pp']={arm:{d:100*(a['means'][arm][d]['Dice']-v23['means']['L'][d]['Dice']) for d in ('RUNMC','BMC')} for arm in ('A','B','N')}
    cost=dict(status='COMPLETE',original_start=x['LAUNCH.json']['started'],exit_time=x['EXIT.json']['time'],hard_deadline=x['LAUNCH.json']['deadline'],elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=400,target_backbone_forwards=351,teacher_new_semantic_forwards=0,preflight_reverse_calls=1,training_telemetry_reverse_calls=6,extra_optimizer_updates=0,old_models_rescored=0,ledger={k:x['ledger'][k] for k in ('limits','seconds_charged','updates_reserved','bytes')})
    lines=['# v30 真实标签锚定局部近邻传递','', 'N=冻结L标签特征库向U传递的top8软类别，学生完全复用v25B。A/B复用匹配的v25固定2000终点，未重训/重评分；仅N从共同v23L1600续400步。','', '| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 |','|---|---|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {100*m['Dice']:.5f} | {m['NLL']:.7f} | {m['Brier']:.7f} | {m['HD95_penalized_mm']:.5f} | {m['empty_predictions']:.3f} |")
    lines+=['','| 对比 | 源Dice pp | 迁移Dice pp | 原门槛 |','|---|---:|---:|---|']
    for name,v in a['effects'].items():lines.append(f"| {name} | {100*v['domains']['RUNMC']['metrics']['Dice']:+.5f} | {100*v['domains']['BMC']['metrics']['Dice']:+.5f} | {v['gate']['status']} |")
    lines+=['','N−A判断这套U标签传递训练的净收益，N−B判断相对固定分割均值教师的条件改进。两者分开应用原门槛，N−B不能代替N−A。','', '| 对比 | 域 | 患者Dice范围 pp | 正/负 | NLL变化 | Brier变化 | HD95变化 |','|---|---|---:|---:|---:|---:|---:|']
    for name,v in a['effects'].items():
        for d,z in v['domains'].items():
            m=z['metrics'];r=z['patient_aggregates']['Dice'];lines.append(f"| {name} | {d} | [{100*r['minimum']:+.5f},{100*r['maximum']:+.5f}] | {r['positive']}/{r['negative']} | {m['NLL']:+.7f} | {m['Brier']:+.7f} | {m['HD95_penalized_mm']:+.5f} |")
    lines+=['','逐项判定：','']
    for name,v in a['effects'].items():lines.append('- '+name+': '+'; '.join(k+'='+('PASS' if b else 'FAILED') for k,b in v['gate']['checks'].items()))
    lines+=['','| 臂 | 相对v23L1600源 pp | 相对v23L1600迁移 pp |','|---|---:|---:|']
    for arm,v in a['relative_to_v23_L1600_pp'].items():lines.append(f"| {arm} | {v['RUNMC']:+.5f} | {v['BMC']:+.5f} |")
    lines+=['','额外400步不能全归U；绝对背景与严格同预算对照分开。近邻软票不等于U GT或校准概率。N−B改变信息来源、目标分布/熵和建库成本，不声称隔离单一纯梯度原因。L标签额外用于一次固定库构造，没有新增标签患者，但标签使用量不与A/B逐项相同。单seed、每域4开发患者、3推断不是3训练seed；BMC未适应，不是CL结果。失败不扩seed/k/网格/特征层/坐标/权重/增强网格。', '', f"新增400正式更新、1真实预检反向、6训练内遥测反向、0额外Adam；一次351骨干前向建库、0新增语义后验预测；全根{cost['root_bytes']}字节，历时{cost['elapsed_seconds']:.2f}秒。旧teacher与A/B来源成本保留；新目标缓存成本计入本run，未重评分旧格。",'', '[冻结协议](PREREG_ZH.md)；[全指标/遥测](ANALYSIS.json)；[执行摘要](EXECUTION_SUMMARY.json)。']
    for n,v in [('ANALYSIS.json',a),('RESULTS.json',x['RESULTS.json']),('EXECUTION_SUMMARY.json',cost)]:(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v['gate'] for k,v in a['effects'].items()}))

if __name__=='__main__':main()
