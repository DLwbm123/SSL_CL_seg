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
    lines=['# v28：空间对应表征 L/U 池对照结果','', '共同v23L1600入口，各400步到固定2000；两臂均有相同辅助目标，只辅助图像池不同。', '', '| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 |','|---|---|---:|---:|---:|---:|---:|']
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
    lines+=['','历史v25A少两次骨干前向，是无辅助目标背景，不是新两臂等计算的主对照。', '', '| 臂 | 末100对应风险 | 正对余弦 | 负对余弦 | 对应检索率 |','|---|---:|---:|---:|---:|']
    for arm,t in a['training_summaries'].items():
        m=t['last100_means'];lines.append(f"| {arm} | {m['correspondence']:.6f} | {m['correspondence_positive_cos']:.6f} | {m['correspondence_negative_cos']:.6f} | {m['correspondence_accuracy']:.6f} |")
    lines+=['','对应检索率是重复训练图像/位置上的任务统计，不是U分割正确率、独立样本或泛化。显式坐标置零不排除隐式位置捷径；不同位置的相同组织可能是假负对。辅助只直接更新phi骨干，整个方法仍由原L目标训练其他模块。', '',f"800正式更新，3模型预检反向+1小fixture反向、0预检Adam。总历时{cost['wall_seconds']:.2f}秒，全根{cost['root_bytes']}字节。只读封存指标，无新评分。",'', '单训练seed、每域4患者、3推断不是3训练seed。共享开发val反复使用，不能宣称稳定泛化/精确后验/CL成功；原门槛与负结果保留，失败不追加seed/温度/权重/空间网格/增强搜索。', '', '[冻结协议](PREREG_ZH.md)；[全部匿名指标](ANALYSIS.json)；[执行摘要](EXECUTION_SUMMARY.json)。']
    lines += ['', '## 实测结论与后继依据', '',
        '这是当前几轮中一个方向较一致、但量级不足的U条件增量：源域与迁移域均4/4患者Dice正向，源域NLL/Brier/HD95也均4/4改善；迁移域Brier4/4改善、NLL3/4改善，HD95为2名略升、1名下降、1名不变。不能只按患者正负数认定统计显著或稳定成功。', '',
        '原源域至少1pp的门槛仍失败；源+0.31712pp与迁移+0.35216pp不构成通过。相对无辅助的匹配更新数v25A，LU源提高0.73787pp但迁移下降0.40560pp；相对1600入口迁移仍下降1.39790pp。因此这是辅助表征路径的局部研究线索，尚非双域净收益或可行性验证。', '',
        '两臂对应风险降低与LU较高的末段检索率仅为训练内摘要。不同图像批次、位置和演化模型不能充当固定探针或U分割真值；本实验也没有测出错误负对比例、位置捷径贡献或梯度冲突因果。', '',
        '后继不扩跑本配置或调温度。直接检验一个不同目标：保留同一共享骨干、同一图像池对照和视图，但取消不同空间位置必须互斥的负对，改为对应视图的通道交叉相关逼近单位阵。该目标以通道去冗余替代空间排斥，避免显式把不同位置的同组织当负对；这只是机制假设，不把v28不足归因为已证明的假负对问题。新独立协议/预算仍采用共同入口、固定终点和原门槛。']
    for name,value in [('ANALYSIS.json',a),('RESULTS.json',r),('EXECUTION_SUMMARY.json',cost)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(a['gate']))

if __name__=='__main__':main()
