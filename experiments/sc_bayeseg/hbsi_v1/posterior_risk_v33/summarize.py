"""Only sealed metrics; no model calls or patient identities in public artifacts."""
import argparse,json
from pathlib import Path
from statistics import mean
from ..trainability_v21.summarize import patients,METRICS


def analyze(x):
    old=x['controls_v25'];neighbor=x['controls_v30']
    for s,arms,updates in ((old,'ABC',1200),(neighbor,'N',400),(x,'J',400)):
        assert s['STATUS.json']['stage']=='COMPLETE' and s['EXIT.json']['exit_code']==0
        assert s['RESULTS.json']['new_formal_updates']==updates
        assert s['RESULTS.json']['counter']['outer_backward_calls']==updates
        assert len(s['RESULTS.json']['cells'])==2*len(arms) and all(c['complete'] and c['patients']==4 for c in s['RESULTS.json']['cells'].values())
        assert len(s['ledger']['training_sessions'])==len(arms)
        assert all(v['completed_steps']==2000 and v['status']=='COMPLETE' for v in s['ledger']['training_sessions'].values())
        assert all(v['status']=='PASS' for v in s['ledger']['attempts'])
    assert x['RESULTS.json']['preflight_reverse_calls']==1 and x['RESULTS.json']['training_audit_reverse_calls']==6
    assert x['PREFLIGHT.json']['same_student_draw_mean'] and x['MATCHING.json']['status']=='PASS'
    for field in ('common_payload','prior_hash','teacher_payload'):
        assert len({s[f'BRANCH_{arm}.json'][field] for s,arm in ((old,'A'),(neighbor,'N'),(x,'J'))})==1
    assert [row['step'] for row in x['audit']['J']]==[1600,1799,1999]
    assert sum(row['extra_reverse_calls'] for row in x['audit']['J'])==6
    pp={arm:{d:patients(rows) for d,rows in s['patients'][arm].items()} for s,arm in ((old,'A'),(neighbor,'N'),(x,'J'))}
    means={arm:{d:{k:(sum(v[k] for v in rows.values()) if k=='empty_predictions' else mean(v[k] for v in rows.values())) for k in METRICS} for d,rows in ds.items()} for arm,ds in pp.items()}
    effects={}
    for control in ('A','N'):
        ds={}
        for d in ('RUNMC','BMC'):
            assert pp['J'][d].keys()==pp[control][d].keys();agg={}
            for k in METRICS:
                v=[pp['J'][d][i][k]-pp[control][d][i][k] for i in pp['J'][d]]
                agg[k]=dict(n=4,mean=mean(v),minimum=min(v),maximum=max(v),positive=sum(t>1e-12 for t in v),negative=sum(t< -1e-12 for t in v))
            ds[d]=dict(metrics={k:means['J'][d][k]-means[control][d][k] for k in METRICS},patient_aggregates=agg)
        checks=dict(source_gain_ge_1pp=ds['RUNMC']['metrics']['Dice']*100>=1,transfer_drop_le_0_5pp=ds['BMC']['metrics']['Dice']*100>=-.5,no_new_empty=all(means['J'][d]['empty_predictions']<=means[control][d]['empty_predictions'] for d in ds),HD95_worsening_le_10pct=all(means['J'][d]['HD95_penalized_mm']<=1.1*means[control][d]['HD95_penalized_mm'] for d in ds))
        effects['J_minus_'+control]=dict(domains=ds,gate=dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks))
    return dict(status='COMPLETE',seed=20261023,training_seeds=1,patients_per_domain=4,inference_repeats=3,means=means,effects=effects,training=x['training'],audit=x['audit'],matching=x['MATCHING.json'],reused_controls=['v25_A2000','v30_N2000'])


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');args=p.parse_args();x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    old=json.loads((out.parent/'semantic_alignment_v23/ANALYSIS.json').read_text())
    a['relative_to_v23_L1600_pp']={arm:{d:100*(a['means'][arm][d]['Dice']-old['means']['L'][d]['Dice']) for d in ('RUNMC','BMC')} for arm in ('A','N','J')}
    cost=dict(status='COMPLETE',original_start=x['LAUNCH.json']['started'],exit_time=x['EXIT.json']['time'],hard_deadline=x['LAUNCH.json']['deadline'],elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=400,training_model_reverse_calls=400,preflight_reverse_calls=1,fixture_reverse_calls=0,additional_audit_reverse_calls=6,extra_optimizer_updates=0,new_target_forwards=0,old_models_rescored=0,ledger={k:x['ledger'][k] for k in ('limits','seconds_charged','updates_reserved','bytes')})
    lines=['# v33 固定语义目标的后验条件风险','', 'J仅改变U风险在后验平均内外的计算位置；A/N分别只读复用v25/v30固定2000终点，共同v23L1600入口，J新增400更新。','', '| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 |','|---|---|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {100*m['Dice']:.5f} | {m['NLL']:.7f} | {m['Brier']:.7f} | {m['HD95_penalized_mm']:.5f} | {m['empty_predictions']:.3f} |")
    lines+=['','| 对比 | 源Dice pp | 迁移Dice pp | 原门槛 |','|---|---:|---:|---|']
    for name,v in a['effects'].items():lines.append(f"| {name} | {100*v['domains']['RUNMC']['metrics']['Dice']:+.5f} | {100*v['domains']['BMC']['metrics']['Dice']:+.5f} | {v['gate']['status']} |")
    lines+=['','J−A整套净收益、J−N条件增量分开判断，不相互替代。','', '| 对比 | 域 | 患者Dice范围 pp | 正/负 | NLL变化 | Brier变化 | HD95变化 |','|---|---|---:|---:|---:|---:|---:|']
    for name,v in a['effects'].items():
        for d,z in v['domains'].items():
            m=z['metrics'];r=z['patient_aggregates']['Dice'];lines.append(f"| {name} | {d} | [{100*r['minimum']:+.5f},{100*r['maximum']:+.5f}] | {r['positive']}/{r['negative']} | {m['NLL']:+.7f} | {m['Brier']:+.7f} | {m['HD95_penalized_mm']:+.5f} |")
    lines+=['','逐项判定：','']
    for name,v in a['effects'].items():lines.append('- '+name+': '+'; '.join(k+'='+('PASS' if b else 'FAILED') for k,b in v['gate']['checks'].items()))
    lines+=['','| 臂 | 相对v23L1600源 pp | 迁移 pp |','|---|---:|---:|']
    for arm,v in a['relative_to_v23_L1600_pp'].items():lines.append(f"| {arm} | {v['RUNMC']:+.5f} | {v['BMC']:+.5f} |")
    m=a['training']['J']['full_means'];last=a['training']['J']['last100_means']
    lines+=['',f"全400步expected U风险{m['U_risk']:.7f}，同样本marginal风险{m['marginal_U_risk']:.7f}，Jensen gap{m['Jensen_gap']:.7f}；末100步gap{last['Jensen_gap']:.7f}。这是变化批次/模型的轨迹，不是固定探针或U标签正确率。数值下限触发比例均值{m['probability_floor_fraction']:.7f}。",'', '| 时点 | q均值U/基础 | q均值余弦 | q尺度U/基础 | q尺度余弦 |','|---|---:|---:|---:|---:|']
    for r in a['audit']['J']:
        q=r['groups']['q_mean'];v=r['groups']['q_scale'];lines.append(f"| {r['step']} | {q['U_over_base']:.6f} | {q['cosine']:.6f} | {v['U_over_base']:.6f} | {v['cosine']:.6f} |")
    lines+=['', '基础含全部原L目标，实际Adam位移不是U反事实。2×2为嵌套样本，经验方差不等可靠不确定性；期望条件风险可能压低有用分歧，不能仅以gap下降宣称成功。', '',f"400正式Adam/模型反向，1真实预检反向0Adam，6额外遥测autograd0额外Adam；无新目标前向/缓存。历时{cost['elapsed_seconds']:.2f}秒，全根{cost['root_bytes']}字节。v30旧建库成本保留，旧A/N未重新评分。", '', '单seed、每域4开发患者、3推断重复，不证明独立泛化、精确后验、可靠不确定性或CL。失败不扩seed/样本数/风险混合权重/温度/阈值/增强。', '', '[冻结协议](PREREG_ZH.md)；[全部指标/轨迹](ANALYSIS.json)；[成本](EXECUTION_SUMMARY.json)。']
    lines += ['', '## 实测结论与机制边界', '',
        'J−A源域+0.53584pp、迁移+0.15390pp，J−N仅+0.04208pp/−0.00041pp；两项均只失败于原源域≥1pp门槛。保护门通过、空预测为0，不能替代主门。J−N两域均2正2负患者，因此这次积分位置调整没有建立有意义的条件分割收益。配置关闭，不追加样本数、混合权重或seed搜索。', '',
        '相对N，J源域NLL和HD95均值变差，Brier仅微降；迁移三项均值都变差。相对A的概率误差改善大部分已经出现在N，不能归功于J的新风险。J迁移Dice仍比共同v23L1600入口低0.83840pp，不能将额外400步的源域增益全部归U或称双域净提升。', '',
        '全400步Jensen gap范围0.000462689至0.037241589，最小采样概率0.000153356，数值下限触发比例每步均为0；本次轨迹的非负gap符合正概率条件下的Jensen关系。它证明新条件风险确实区别于原均值风险，未证明分歧被正确压缩，也未证明U目标正确。没有对N做相同样本的后续反事实重评分，不以两个版本的不同训练轨迹判定后验方差因果变化。', '',
        '三点q均值U/基础范数约0.656、0.925、0.511，余弦约−0.017、−0.764、−0.844；q尺度余弦后两点转正。不同组和时点方向不同，不能以“始终梯度太小”或“全参数冲突”解释结果。现有证据支持停止同一固定目标上的进一步风险变体；后继需引入不同的信息约束，不能把小标量改善视为机制已经可行。']
    for n,v in [('ANALYSIS.json',a),('RESULTS.json',x['RESULTS.json']),('EXECUTION_SUMMARY.json',cost)]:(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v['gate'] for k,v in a['effects'].items()}))

if __name__=='__main__':main()
