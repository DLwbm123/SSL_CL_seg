"""Sealed H/R plus matched historical A/B, without any new inference."""
import argparse,json
from pathlib import Path
from statistics import mean
from ..trainability_v21.summarize import patients,METRICS


def analyze(x):
    old=x['controls_v25']
    for snapshot,arms,steps in ((old,'ABC',1200),(x,'HR',800)):
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
        assert len({snapshot[f'BRANCH_{arm}.json'][field] for snapshot,arms in ((old,'AB'),(x,'HR')) for arm in arms})==1
    assert sum(r['extra_reverse_calls'] for arm in 'HR' for r in x['audit'][arm])==12
    assert all([r['step'] for r in x['audit'][arm]]==[1600,1799,1999] for arm in 'HR')
    pp={arm:{d:patients(rows) for d,rows in snapshot['patients'][arm].items()} for snapshot,arms in ((old,'AB'),(x,'HR')) for arm in arms}
    means={arm:{d:{k:(sum(v[k] for v in rows.values()) if k=='empty_predictions' else mean(v[k] for v in rows.values())) for k in METRICS} for d,rows in ds.items()} for arm,ds in pp.items()}
    effects={}
    for arm,control in (('H','A'),('R','A'),('H','B'),('R','H'),('R','B')):
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
    v23=json.loads((out.parent/'semantic_alignment_v23/ANALYSIS.json').read_text())
    a['relative_to_v23_L1600_pp']={arm:{d:100*(a['means'][arm][d]['Dice']-v23['means']['L'][d]['Dice']) for d in ('RUNMC','BMC')} for arm in ('A','B','H','R')}
    cost=dict(status='COMPLETE',original_start=x['LAUNCH.json']['started'],exit_time=x['EXIT.json']['time'],hard_deadline=x['LAUNCH.json']['deadline'],elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=800,teacher_new_forwards=0,preflight_reverse_calls=2,training_telemetry_reverse_calls=12,extra_optimizer_updates=0,old_models_rescored=0,ledger={k:x['ledger'][k] for k in ('limits','seconds_charged','updates_reserved','bytes')})
    lines=['# v26 固定教师类别与区域风险','', 'H=固定硬类别CE；R=同硬类别CE+前景Dice；A/B复用匹配的v25固定2000终点，未重训/重评分。两个新臂共同v23 L1600入口各400步。','', '| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 |','|---|---|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {100*m['Dice']:.5f} | {m['NLL']:.7f} | {m['Brier']:.7f} | {m['HD95_penalized_mm']:.5f} | {m['empty_predictions']:.3f} |")
    lines+=['','| 对比 | 源Dice pp | 迁移Dice pp | 原门槛 |','|---|---:|---:|---|']
    for name,v in a['effects'].items():lines.append(f"| {name} | {100*v['domains']['RUNMC']['metrics']['Dice']:+.5f} | {100*v['domains']['BMC']['metrics']['Dice']:+.5f} | {v['gate']['status']} |")
    lines+=['','所有对比分开应用原门槛，H−A/R−A判断U探索信号；H−B/R−H判断条件机制增量，不互相替代。','', '| 对比 | 域 | 患者Dice范围 pp | 正/负 | NLL变化 | Brier变化 | HD95变化 |','|---|---|---:|---:|---:|---:|---:|']
    for name,v in a['effects'].items():
        for d,z in v['domains'].items():
            m=z['metrics'];r=z['patient_aggregates']['Dice'];lines.append(f"| {name} | {d} | [{100*r['minimum']:+.5f},{100*r['maximum']:+.5f}] | {r['positive']}/{r['negative']} | {m['NLL']:+.7f} | {m['Brier']:+.7f} | {m['HD95_penalized_mm']:+.5f} |")
    lines+=['','逐项判定：','']
    for name,v in a['effects'].items():lines.append('- '+name+': '+'; '.join(k+'='+('PASS' if b else 'FAILED') for k,b in v['gate']['checks'].items()))
    lines+=['','| 臂 | 相对v23L1600源 pp | 相对v23L1600迁移 pp |','|---|---:|---:|']
    for arm,v in a['relative_to_v23_L1600_pp'].items():lines.append(f"| {arm} | {v['RUNMC']:+.5f} | {v['BMC']:+.5f} |")
    lines+=['','额外400步不能全归U；绝对背景与严格同预算对照分开。硬类别不等于GT。H改变目标熵/大小，R增加区域项也改变总量，不声称隔离纯梯度方向。单seed、每域4开发患者、3推断不是3训练seed；BMC未适应，不是CL结果。失败不扩seed/温度/权重/阈值/增强网格。', '', f"新增800正式更新、2真实预检反向、12训练内遥测反向、0额外Adam；全根{cost['root_bytes']}字节，历时{cost['elapsed_seconds']:.2f}秒。旧teacher与A/B来源成本保留，未新生成教师或重评分旧格。",'', '[冻结协议](PREREG_ZH.md)；[全指标/遥测](ANALYSIS.json)；[执行摘要](EXECUTION_SUMMARY.json)。']
    lines += ["", '## 实测机制结论\n\nH−A源−0.36382pp、迁移−0.56376pp，源收益门和迁移保护门均失败；硬化教师既没有分割增益，还使NLL/Brier明显变差。R−H源+0.23017pp、迁移+0.66598pp，BMC3/4患者改善，但源只有2/4，且R−A源仍−0.13364pp，原收益门失败。区域项有局部补偿，不能当作整套U可行。R−B源仍负，所有五个对比原门槛均FAILED；空预测与HD95保护均PASS。\n\nH/R相对封存L1600的迁移分别−1.55606/−0.89008pp。固定终点不能换回1600，额外400步的源提升不能全算U。R的Dice局部恢复伴随NLL更差，不能当作概率质量或可靠不确定性改善。\n\n起点H/R的q均值U/基础梯度范数约3.52/4.06，余弦−0.948/−0.965；末点R的q均值比例1.63、余弦+0.743，却仍无终点源收益。三个训练点不能证明教师错误或冲突是唯一根因，但明确不支持通过增大U权重解决问题。本轮已检验固定教师下软KL、硬CE、硬CE+Dice的有限目标变化；不再沿同一缓存标签换权重/温度/阈值。\n\n下一独立假设将改变U信息进入模型的方式：保留L图像一半区域的原始像素和真实标签，只替换未监督的另一半上下文，比较训练L供体与U供体。与v24不同，监督区域像素不变，不用相位保留推断标签仍有效；与v25/v26不同，不把U伪标签作为监督。接缝和不自然上下文仍有风险，必须以匹配L供体对照和原门槛验证；不把这个设计本身称为成功。该用途仍是当前q/phi/条件域后验分割模型的主线改进，不能证明原U生成评分或CL机制有效。\n']
    for n,v in [('ANALYSIS.json',a),('RESULTS.json',x['RESULTS.json']),('EXECUTION_SUMMARY.json',cost)]:(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v['gate'] for k,v in a['effects'].items()}))

if __name__=='__main__':main()
