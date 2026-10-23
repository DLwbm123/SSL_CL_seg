"""Offline anonymous report; unchanged paired patient metrics and original gates."""
import argparse,json
from pathlib import Path
from statistics import mean
from ..trainability_v21.summarize import patients,METRICS

def analyze(x):
    assert x['STATUS.json']['stage']=='COMPLETE' and x['EXIT.json']['exit_code']==0
    assert x['RESULTS.json']['new_formal_updates']==800
    assert x['BRANCH_L.json']['common_payload']==x['BRANCH_LU.json']['common_payload']
    assert x['BRANCH_L.json']['prior_hash']==x['BRANCH_LU.json']['prior_hash']
    l=x['ledger'];assert len(l['training_sessions'])==2 and l['updates_reserved']==800
    failed=[v for v in l['attempts'] if v['status']!='PASS']
    assert not failed
    assert all(v['completed_steps']==2000 and v['status']=='COMPLETE' for v in l['training_sessions'].values())
    p={arm:{d:patients(v) for d,v in ds.items()} for arm,ds in x['patients'].items()}
    means={arm:{d:{k:mean(v[k] for v in rows.values()) if k!='empty_predictions' else sum(v[k] for v in rows.values()) for k in METRICS} for d,rows in ds.items()} for arm,ds in p.items()}
    effects={}
    for d in ('RUNMC','BMC'):
        assert p['L'][d].keys()==p['LU'][d].keys()
        changes={k:means['LU'][d][k]-means['L'][d][k] for k in METRICS}
        patient_effects={}
        for k in METRICS:
            v=[p['LU'][d][identity][k]-p['L'][d][identity][k] for identity in p['L'][d]]
            patient_effects[k]=dict(n=4,mean=mean(v),minimum=min(v),maximum=max(v),positive=sum(t>1e-12 for t in v),negative=sum(t< -1e-12 for t in v))
        effects[d]=dict(metrics=changes,patient_aggregates=patient_effects)
    checks=dict(source_U_gain_ge_1pp=effects['RUNMC']['metrics']['Dice']*100>=1,transfer_drop_le_0_5pp=effects['BMC']['metrics']['Dice']*100>=-.5,no_new_empty=all(means['LU'][d]['empty_predictions']<=means['L'][d]['empty_predictions'] for d in effects),HD95_worsening_le_10pct=all(means['LU'][d]['HD95_penalized_mm']<=1.1*means['L'][d]['HD95_penalized_mm'] for d in effects))
    return dict(status='COMPLETE',seed=20261023,training_seeds=1,patients_per_domain=4,inference_repeats=3,means=means,U_effect=effects,gate=dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks),training_summaries=x['training'])



def main():
    parser=argparse.ArgumentParser();parser.add_argument('snapshot');args=parser.parse_args()
    x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    r=x['RESULTS.json']
    assert len(r['cells'])==4 and all(v['complete'] and v['patients']==4 for v in r['cells'].values())
    assert r['counter']['outer_backward_calls']==800 and r['preflight_real_reverse_calls']==2 and r['auxiliary_path_reverse_calls']==1 and r['preflight_fixture_reverse_calls']==1
    for tag,folder,arm in [('v23L1600','semantic_alignment_v23','L'),('v25A2000','posterior_teacher_v25','A')]:
        old=json.loads((out.parent/folder/'ANALYSIS.json').read_text())
        a[tag+'_historical_delta_pp']={new:{d:100*(a['means'][new][d]['Dice']-old['means'][arm][d]['Dice']) for d in ('RUNMC','BMC')} for new in ('L','LU')}
    cost=dict(status='COMPLETE',original_started=x['LAUNCH.json']['started'],original_deadline=x['LAUNCH.json']['deadline'],wall_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=800,preflight_model_reverse_calls=3,preflight_fixture_reverse_calls=1,preflight_optimizer_updates=0,source_updates=0,common_payload_and_prior=True,ledger={k:x['ledger'][k] for k in ('limits','seconds_charged','updates_reserved','bytes')})
    lines=['# v29：空间通道去冗余 L/U 池对照结果','', '共同v23L1600入口，各400步到固定2000；两臂均有相同辅助目标，只辅助图像池不同。', '', '| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 |','|---|---|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {100*m['Dice']:.5f} | {m['NLL']:.7f} | {m['Brier']:.7f} | {m['HD95_penalized_mm']:.5f} | {m['empty_predictions']:.3f} |")
    lines+=['','| LU−L | Dice pp | 患者范围 pp | 正/负患者 | NLL变化 | Brier变化 |','|---|---:|---:|---:|---:|---:|']
    for d,v in a['U_effect'].items():
        m=v['metrics'];q=v['patient_aggregates']['Dice'];lines.append(f"| {d} | {100*m['Dice']:+.5f} | [{100*q['minimum']:+.5f}, {100*q['maximum']:+.5f}] | {q['positive']}/{q['negative']} | {m['NLL']:+.7f} | {m['Brier']:+.7f} |")
    lines+=['',f"原门槛总判定：**{a['gate']['status']}**。",'', '| 原门槛 | 判定 |','|---|---|']
    for k,v in a['gate']['checks'].items():lines.append(f"| {k} | {'PASS' if v else 'FAILED'} |")
    for tag in ('v23L1600','v25A2000'):
        lines+=['',f'| 臂 | 相对{tag}源 pp | 迁移 pp |','|---|---:|---:|']
        for arm,v in a[tag+'_historical_delta_pp'].items():lines.append(f"| {arm} | {v['RUNMC']:+.5f} | {v['BMC']:+.5f} |")
    lines+=['','历史v25A少两次骨干前向，是无辅助目标背景，不是新两臂等计算的主对照。', '', '| 臂 | 末100辅助风险 | 对角相关 | 非对角相关平方 | 空间标准差 |','|---|---:|---:|---:|---:|']
    for arm,t in a['training_summaries'].items():
        m=t['last100_means'];lines.append(f"| {arm} | {m['auxiliary_risk']:.6f} | {m['correlation_diagonal']:.6f} | {m['correlation_offdiag_square']:.6f} | {m['feature_spatial_sd']:.6f} |")
    lines+=['','相关性与空间标准差来自重复训练图像位置，不是U分割质量、独立样本或可靠后验。取消空间负对不保证语义正确，通道去相关不等统计独立；显式坐标置零仍可能有隐式位置捷径。辅助只直接更新phi骨干，其他模块仍由原L目标学习。', '',f"800正式更新，3模型预检反向+1小fixture反向、0预检Adam。总历时{cost['wall_seconds']:.2f}秒，全根{cost['root_bytes']}字节。只读封存指标，无新评分。",'', '单训练seed、每域4患者、3推断不是3训练seed。共享开发val反复使用，不能宣称稳定泛化/精确后验/CL成功；原门槛与负结果保留，失败不追加seed/epsilon/lambda/权重/空间网格/增强搜索。', '', '[冻结协议](PREREG_ZH.md)；[全部匿名指标](ANALYSIS.json)；[执行摘要](EXECUTION_SUMMARY.json)。']
    old=json.loads((out.parent/'spatial_correspondence_v28/ANALYSIS.json').read_text())
    a['historical_v28_same_arm_delta_pp']={arm:{d:100*(a['means'][arm][d]['Dice']-old['means'][arm][d]['Dice']) for d in ('RUNMC','BMC')} for arm in ('L','LU')}
    a['historical_v28_U_effect_difference_pp']={d:100*(a['U_effect'][d]['metrics']['Dice']-old['U_effect'][d]['metrics']['Dice']) for d in ('RUNMC','BMC')}
    lines+=['','| 臂 | 相对封存v28同臂源 pp | 迁移 pp |','|---|---:|---:|']
    for arm,v in a['historical_v28_same_arm_delta_pp'].items():lines.append(f"| {arm} | {v['RUNMC']:+.5f} | {v['BMC']:+.5f} |")
    lines+=['','v28比较只读封存指标。目标、归一化与风险量级同时改变，不能单独归因于去掉假负对；同首seed跨版本背景不替代新LU−L原门槛。']
    lines += ['', '## 实测机制结论\n\nLU−L源−0.03353pp（1正3负）、迁移+0.01317pp（3正1负），原源≥1pp门失败，其他保护门通过，空预测均0。源Brier四患者均恶化；迁移NLL和Brier四患者均恶化。小迁移Dice差异不构成概率质量或分割可行性信号。本配置关闭，不增加seed或搜索epsilon/lambda/总权重/位置/视图。\n\n末100步对角相关约0.981、辅助风险约0.028，说明训练中的跨视图相关目标有较低数值；非对角相关平方仍约0.356，且低辅助风险不保证语义可分。相关与标准差摘要来自不断变化的模型和训练样本，不能当固定探针或独立U证据。相对v28，源U增量减少0.35065pp、迁移减少0.33899pp；取消空间负对没有带来更好的U增量。目标、归一化、风险尺度同时改变，这既不证明空间负对有益，也不能排除其他参数化的去冗余方法。\n\nL/LU均低于封存v25A2000，源分别−0.03522/−0.06875pp、迁移−0.21578/−0.20260pp；额外骨干计算未建立净改进。v28的正方向结果仍是未过门槛的探索性证据，v29不把它升级为已验证可行。\n\n目前固定自预测教师、上下文迁移和无标签表征辅助都未建立原门槛要求的收益。下一独立假设改为从训练L真实标签向U传递局部语义：冻结同一入口识别骨干，在训练L的固定空间网格建带标签特征库，用U特征的局部近邻投票生成固定软目标。学生保留v25B的目标、视图、随机流和400步终点，复用严格匹配的A/B封存对照。它检验标签传递能否优于自身分割均值教师，不通过调损失权重寻找成功。相似外观不必同类、坐标捷径、稀疏网格漏小结构及错误传递都仍是未验证风险；近邻投票不是校准概率或独立教师证据。须另立协议、预算和实际资源准入后执行。\n']
    for name,value in [('ANALYSIS.json',a),('RESULTS.json',r),('EXECUTION_SUMMARY.json',cost)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(a['gate']))

if __name__=='__main__':main()
