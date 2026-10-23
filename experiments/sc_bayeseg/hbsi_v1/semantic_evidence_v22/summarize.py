"""Offline fixed-endpoint report from sealed metrics; no model calls."""
import argparse
import json
from pathlib import Path
from statistics import mean
from ..trainability_v21.summarize import patients, METRICS


def analyze(x):
    assert x['STATUS.json']['stage']=='COMPLETE' and x['EXIT.json']['exit_code']==0
    assert x['RESULTS.json']['new_formal_updates']==2400
    assert x['BRANCH_L.json']['common_payload']==x['BRANCH_LU.json']['common_payload']
    assert x['BRANCH_L.json']['prior_hash']==x['BRANCH_LU.json']['prior_hash']
    l=x['ledger'];assert len(l['training_sessions'])==2 and l['updates_reserved']==2400
    assert all(v['status']=='PASS' for v in l['attempts'])
    assert all(v['completed_steps']==1600 and v['status']=='COMPLETE' for v in l['training_sessions'].values())
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


def report(a,cost):
    lines=['# v22：语义锚定＋SAR边缘图像证据 L／LU 固定终点结果','', '两臂已完成，固定绝对1600步，各新增1200步，共2400次正式更新。共同RUNMC L400入口、原q/phi/域状态/先验/Adam/RNG，同L曝光和独立L/U局部随机流；仅LU启用U图像证据。BMC没有训练，是未适应迁移。','', '| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测（3推断均值） |','|---|---|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {m['Dice']*100:.4f} | {m['NLL']:.6f} | {m['Brier']:.6f} | {m['HD95_penalized_mm']:.4f} | {m['empty_predictions']:.3f} |")
    lines+=['','## U 的独立增量与预注册门槛','', '| 域 | LU−L Dice pp | NLL变化 | Brier变化 | HD95变化 mm | 患者Dice差值范围 pp | 正/负患者 |','|---|---:|---:|---:|---:|---:|---:|']
    for d,v in a['U_effect'].items():
        m=v['metrics'];t=v['patient_aggregates']['Dice'];lines.append(f"| {d} | {m['Dice']*100:+.4f} | {m['NLL']:+.6f} | {m['Brier']:+.6f} | {m['HD95_penalized_mm']:+.4f} | [{t['minimum']*100:+.4f}, {t['maximum']*100:+.4f}] | {t['positive']}/{t['negative']} |")
    lines+=['', '冻结的首seed继续研究信号：源RUNMC的LU−L≥1pp、BMC迁移不下降>.5pp、两域无新增空预测且HD95相对新L不恶化>10%。这些是本次新协议的探索条件，不改变v20/v21已有门槛与负结论。','', '| 门槛 | 判定 |','|---|---|']
    for k,v in a['gate']['checks'].items():lines.append(f"| {k} | {'PASS' if v else 'FAILED'} |")
    lines+=['',f"**总判定：{a['gate']['status']}。**"]
    if a['gate']['status']=='FAILED':lines+=['本配置未达到预定继续研究信号，不自动追加seed、权重或增强网格。正的小幅增量也不能改称达到≥1pp门槛；负结果完整保留。']
    else:lines+=['仅获得独立确认/消融研究资格；这是一个训练seed的开发结果，不能宣称稳定泛化或Bayesian机制已被独立验证。']
    lines+=['','## 机制记录与解释边界','', '观测由Student-t改为高斯；SAR空间协方差经FFT、低秩偏置经Woodbury/Cholesky解析边缘化，S/b不再作为局部识别变量。训练L增加GT相对两种8px平移候选的生成评分锚定。此为组合改造后的广义贝叶斯风险，不是原ELBO的等价改写，也没有分离锚定和边缘化的独立贡献。','', '| 臂 | 最后100步 task | anchor | GT−错位评分/HW | L图像边缘ELBO/HW | U图像边缘ELBO/HW |','|---|---:|---:|---:|---:|---:|']
    for arm,t in a['training_summaries'].items():
        m=t['last100_means'];lines.append(f"| {arm} | {m['task']:.6f} | {m['anchor']:.6f} | {m['structure_margin']:+.6f} | {m['image_L']:+.6f} | {m['image_U']:+.6f} |")
    lines+=['','本轮NLL虽在两域下降，但Dice和Brier没有同步改善，不能把密度或对数评分改善当成分割收益。LU在RUNMC的4名患者中1正3负，BMC为0正4负；只覆盖首seed，不能据此认定所有未来U机制均无效。', '', '最后100步L/LU锚定损失分别约0.692985/0.693128，接近log(2)=0.693147；相应GT相对固定错位评分差/HW约0.000327/0.000039。训练评分间隔很小，没有证明已建立强的结构辨别能力；这些是不同训练批次的轨迹摘要，不能把首步与末100步当作固定探针对照或独立泛化检验。', '', 'L臂U项为0表示未启用该目标。结构评分来自已用于锚定的训练L/固定错位规则，只能说明训练轨迹，不能证明真实U语义质量、独立结构辨别或精确边缘后验。两个臂都可通过数据接口访问相同U图像，L臂没有U损失；通用引擎输入图像计数包含该无目标输入。','', '每域同4名val患者，先对每名3推断求平均再配对；不是12独立患者，也不是3训练seed。开发val反复使用，没有test或U隐藏GT访问。全部辅助指标患者差值汇总在ANALYSIS.json。','', '## 执行与交付','', f"COMPLETE/EXIT0，总历时{cost['wall_elapsed_seconds']:.2f}秒，ledger {cost['ledger']['bytes']}字节，2400正式新增更新；2次真实批次预检反向/0预检Adam，另有小型密度/梯度fixture。原400步来源成本单独保留，没有重训或重评分v19/v20/v21。两训练阶段和全部评估PASS，无重试、时钟重置或门槛修改。",'', '[冻结协议](PREREG_ZH.md)；[匿名结果](ANALYSIS.json)；[执行摘要](EXECUTION_SUMMARY.json)；[预检](PREFLIGHT.json)。不发布身份、原始影像、概率数组、权重、私有路径或原始日志。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');args=p.parse_args();x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    cost=dict(status='COMPLETE',exit=x['EXIT.json'],original_launch=x['LAUNCH.json'],wall_elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],ledger={k:x['ledger'][k] for k in ('limits','seconds_charged','updates_reserved','bytes')},formal_updates=2400,real_batch_preflight_reverse_calls=2,preflight_optimizer_updates=0,density_autograd_fixture_extra=True,common_payload_and_prior_equal=True,attempts=[{k:v for k,v in row.items() if k in ('name','status','seconds_actual','updates_reserved')} for row in x['ledger']['attempts']])
    for name,obj in [('ANALYSIS.json',a),('EXECUTION_SUMMARY.json',cost),('RESULTS.json',x['RESULTS.json'])]:(out/name).write_text(json.dumps(obj,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text(report(a,cost));print(json.dumps(dict(gate=a['gate'],means=a['means'],effects={d:v['metrics']['Dice']*100 for d,v in a['U_effect'].items()})))

if __name__=='__main__':main()
