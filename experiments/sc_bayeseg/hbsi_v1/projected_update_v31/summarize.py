"""Only sealed metrics; no model calls or patient identities in public artifacts."""
import argparse,json
from pathlib import Path
from statistics import mean
from ..trainability_v21.summarize import patients,METRICS


def analyze(x):
    old=x['controls_v25'];neighbor=x['controls_v30']
    for s,arms,updates in ((old,'ABC',1200),(neighbor,'N',400),(x,'P',400)):
        assert s['STATUS.json']['stage']=='COMPLETE' and s['EXIT.json']['exit_code']==0
        assert s['RESULTS.json']['new_formal_updates']==updates
        assert s['RESULTS.json']['counter']['outer_backward_calls']==updates
        assert len(s['RESULTS.json']['cells'])==2*len(arms) and all(c['complete'] and c['patients']==4 for c in s['RESULTS.json']['cells'].values())
        assert len(s['ledger']['training_sessions'])==len(arms)
        assert all(v['completed_steps']==2000 and v['status']=='COMPLETE' for v in s['ledger']['training_sessions'].values())
        assert all(v['status']=='PASS' for v in s['ledger']['attempts'])
    r=x['RESULTS.json'];assert r['training_model_autograd_calls']==800 and r['training_surrogate_backward_calls']==400
    assert r['preflight_model_autograd_calls']==2 and r['preflight_surrogate_backward_calls']==1
    assert r['counter']['projection_model_reverse_calls']==802
    assert x['PREFLIGHT.json']['exact_projected_gradient'] and x['MATCHING.json']['status']=='PASS'
    for field in ('common_payload','prior_hash','teacher_payload'):
        assert len({s[f'BRANCH_{arm}.json'][field] for s,arm in ((old,'A'),(neighbor,'N'),(x,'P'))})==1
    assert [row['step'] for row in x['audit']['P']]==[1600,1799,1999]
    assert sum(row['extra_reverse_calls'] for row in x['audit']['P'])==0
    pp={arm:{d:patients(rows) for d,rows in s['patients'][arm].items()} for s,arm in ((old,'A'),(neighbor,'N'),(x,'P'))}
    means={arm:{d:{k:(sum(v[k] for v in rows.values()) if k=='empty_predictions' else mean(v[k] for v in rows.values())) for k in METRICS} for d,rows in ds.items()} for arm,ds in pp.items()}
    effects={}
    for control in ('A','N'):
        ds={}
        for d in ('RUNMC','BMC'):
            assert pp['P'][d].keys()==pp[control][d].keys();agg={}
            for k in METRICS:
                v=[pp['P'][d][i][k]-pp[control][d][i][k] for i in pp['P'][d]]
                agg[k]=dict(n=4,mean=mean(v),minimum=min(v),maximum=max(v),positive=sum(t>1e-12 for t in v),negative=sum(t< -1e-12 for t in v))
            ds[d]=dict(metrics={k:means['P'][d][k]-means[control][d][k] for k in METRICS},patient_aggregates=agg)
        checks=dict(source_gain_ge_1pp=ds['RUNMC']['metrics']['Dice']*100>=1,transfer_drop_le_0_5pp=ds['BMC']['metrics']['Dice']*100>=-.5,no_new_empty=all(means['P'][d]['empty_predictions']<=means[control][d]['empty_predictions'] for d in ds),HD95_worsening_le_10pct=all(means['P'][d]['HD95_penalized_mm']<=1.1*means[control][d]['HD95_penalized_mm'] for d in ds))
        effects['P_minus_'+control]=dict(domains=ds,gate=dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks))
    return dict(status='COMPLETE',seed=20261023,training_seeds=1,patients_per_domain=4,inference_repeats=3,means=means,effects=effects,training=x['training'],audit=x['audit'],matching=x['MATCHING.json'],reused_controls=['v25_A2000','v30_N2000'])


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');args=p.parse_args();x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    old=json.loads((out.parent/'semantic_alignment_v23/ANALYSIS.json').read_text())
    a['relative_to_v23_L1600_pp']={arm:{d:100*(a['means'][arm][d]['Dice']-old['means']['L'][d]['Dice']) for d in ('RUNMC','BMC')} for arm in ('A','N','P')}
    cost=dict(status='COMPLETE',original_start=x['LAUNCH.json']['started'],exit_time=x['EXIT.json']['time'],hard_deadline=x['LAUNCH.json']['deadline'],elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=400,training_model_autograd_calls=800,training_surrogate_backward_calls=400,preflight_model_autograd_calls=2,preflight_surrogate_backward_calls=1,fixture_reverse_calls=0,additional_audit_reverse_calls=0,extra_optimizer_updates=0,new_target_forwards=0,old_models_rescored=0,ledger={k:x['ledger'][k] for k in ('limits','seconds_charged','updates_reserved','bytes')})
    lines=['# v31 固定语义证据的非对称梯度投影','', 'P仅改变全参数U梯度合并规则；A/N分别只读复用v25/v30固定2000终点，共同v23L1600入口，P新增400更新。','', '| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 |','|---|---|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {100*m['Dice']:.5f} | {m['NLL']:.7f} | {m['Brier']:.7f} | {m['HD95_penalized_mm']:.5f} | {m['empty_predictions']:.3f} |")
    lines+=['','| 对比 | 源Dice pp | 迁移Dice pp | 原门槛 |','|---|---:|---:|---|']
    for name,v in a['effects'].items():lines.append(f"| {name} | {100*v['domains']['RUNMC']['metrics']['Dice']:+.5f} | {100*v['domains']['BMC']['metrics']['Dice']:+.5f} | {v['gate']['status']} |")
    lines+=['','P−A整套净收益、P−N条件增量分开判断，不相互替代。','', '| 对比 | 域 | 患者Dice范围 pp | 正/负 | NLL变化 | Brier变化 | HD95变化 |','|---|---|---:|---:|---:|---:|---:|']
    for name,v in a['effects'].items():
        for d,z in v['domains'].items():
            m=z['metrics'];r=z['patient_aggregates']['Dice'];lines.append(f"| {name} | {d} | [{100*r['minimum']:+.5f},{100*r['maximum']:+.5f}] | {r['positive']}/{r['negative']} | {m['NLL']:+.7f} | {m['Brier']:+.7f} | {m['HD95_penalized_mm']:+.5f} |")
    lines+=['','逐项判定：','']
    for name,v in a['effects'].items():lines.append('- '+name+': '+'; '.join(k+'='+('PASS' if b else 'FAILED') for k,b in v['gate']['checks'].items()))
    lines+=['','| 臂 | 相对v23L1600源 pp | 迁移 pp |','|---|---:|---:|']
    for arm,v in a['relative_to_v23_L1600_pp'].items():lines.append(f"| {arm} | {v['RUNMC']:+.5f} | {v['BMC']:+.5f} |")
    m=a['training']['P']['full_means']
    lines+=['',f"400步投影触发比例{m['projection_applied']:.6f}；基础·U内积均值{m['base_dot_U']:.6f}，基础·投影U内积均值{m['base_dot_projected_U']:.6f}。这些随模型和训练批次变化，不是固定探针或U语义质量。",'', '| 训练点 | 实际Adam位移范数 | 基础梯度·实际位移 | U梯度·实际位移 |','|---|---:|---:|---:|']
    for row in a['audit']['P']:lines.append(f"| {row['step']} | {row['combined_Adam_displacement_norm']:.7f} | {row['base_gradient_dot_actual_displacement']:.7f} | {row['U_gradient_dot_actual_displacement']:.7f} |")
    lines+=['','实际位移是合并Adam步，不是U反事实；基础梯度包含L全部任务/image/anchor/KL。原梯度投影性质不能保证带动量/自适应缩放后的有限步风险改善。投影改变范数和方向，不是原标量风险或ELBO梯度、精确后验。近邻缓存仍可能错误且不是独立教师；L额外建库成本保留。', '',f"本run 400正式Adam更新、800模型autograd和400 surrogate backward；预检2模型autograd+1 surrogate backward、0Adam；fixture/额外遥测反向均0，无新目标预测。历时{cost['elapsed_seconds']:.2f}秒，全根{cost['root_bytes']}字节，旧A/N未重新评分。",'', '单训练seed、每域4开发患者、3推断重复；不宣称稳定泛化、可靠不确定性或CL成功。失败不搜索seed/分组/阈值/权重/优化器/增强。', '', '[冻结协议](PREREG_ZH.md)；[全部匿名指标/轨迹](ANALYSIS.json)；[执行摘要](EXECUTION_SUMMARY.json)。']
    lines+=['', '## 实测解释与收口', '',
        'P−A源/迁移分别+0.48636/+0.08340pp，但P−N为−0.00740/−0.07092pp。两项均只在源≥1pp门失败，其他保护通过。投影没有把原N的亚门槛收益提升为继续信号，本配置关闭。',
        'P−A源4/4患者改善；相对N源2正2负，迁移3正1负但单个负差幅度较大，平均仍下降。P相对N的NLL、Brier、HD95两域均值全部恶化；不能仅以患者正方向数量判定优于N。相对v23L1600迁移仍下降0.90891pp，不是双域绝对净收益。',
        '400步中154步触发投影，说明合并规则实际介入。1600点基础梯度与实际Adam位移内积为正，中点U梯度与位移内积为正；原始欧氏投影约束不自动传递给带历史动量和逐坐标缩放的实际位移。这里仅有三个时点的一阶局部量，不能据此断言真实有限步损失上升，更不能反推冲突是唯一根因。',
        '结果排除了“在该固定缓存、共同入口与400步预算下，当前全参数欧氏投影足以建立≥1pp源增益”这一具体候选。它没有排除所有冲突处理，也没有验证近邻目标的语义正确性。不会改为分组投影、调优化器或权重继续搜索。后继若推进，应改变可证伪的语义信息机制并另立协议，而不是据此改门槛。']
    for n,v in [('ANALYSIS.json',a),('RESULTS.json',x['RESULTS.json']),('EXECUTION_SUMMARY.json',cost)]:(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v['gate'] for k,v in a['effects'].items()}))

if __name__=='__main__':main()
