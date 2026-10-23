"""Offline aggregate reporting from existing scores; no model or data calls."""
import argparse
import datetime
import json
from pathlib import Path
import statistics

METRICS = ('Dice', 'NLL', 'Brier', 'HD95_penalized_mm', 'empty_predictions')
SEEDS = (20261023, 20261024, 20261025)
DOMAINS = ('RUNMC', 'BMC')


def aggregate(values):
    return dict(values=values, n_seeds=len(values), mean=statistics.mean(values),
                sample_sd=statistics.stdev(values) if len(values)>1 else None,
                positive=sum(v>0 for v in values), negative=sum(v<0 for v in values),
                minimum=min(values), maximum=max(values))


def summarize(snapshot):
    out={'checked_epoch':snapshot['checked_epoch'], 'experiments':{}}
    for version, exp in snapshot['experiments'].items():
        cells={}
        for key, cell in exp['cells'].items():
            assert cell['complete'] and cell['patients']==4 and len(cell['repeats'])==3
            cells[key]={m:statistics.mean(r[m] for r in cell['repeats']) for m in METRICS}
        result=dict(status=exp['status'],exit=exp.get('exit'),resources=exp['resources'],sessions=exp['sessions'],cells=cells,contrasts={})
        if version=='v19':
            contrasts={
                'source_U':{'LU_A':1,'L_A':-1},
                'history_L':{'L_M1_B':1,'L_M0_B':-1},
                'history_LU':{'LU_M1_B':1,'LU_M0_B':-1},
                'U_at_M0_cumulative':{'LU_M0_B':1,'L_M0_B':-1},
                'U_at_M1_cumulative':{'LU_M1_B':1,'L_M1_B':-1},
                'U_history_interaction':{'LU_M1_B':1,'LU_M0_B':-1,'L_M1_B':-1,'L_M0_B':1},
                'L_M0_target_minus_source':{'L_M0_B':1,'L_A':-1},
                'L_M1_target_minus_source':{'L_M1_B':1,'L_A':-1},
                'LU_M0_target_minus_source':{'LU_M0_B':1,'LU_A':-1},
                'LU_M1_target_minus_source':{'LU_M1_B':1,'LU_A':-1},
            }
        else:
            contrasts={'B-A':{'B':1,'A':-1},'C-A':{'C':1,'A':-1},'D-C':{'D':1,'C':-1},'D-B':{'D':1,'B':-1},'D-A':{'D':1,'A':-1},'interaction':{'D':1,'B':-1,'C':-1,'A':1}}
        for name, weights in contrasts.items():
            result['contrasts'][name]={}
            for domain in DOMAINS:
                available=[seed for seed in SEEDS if all(f's{seed}_{arm}_{domain}' in cells for arm in weights)]
                if not available:
                    continue
                metrics={m:aggregate([sum(w*cells[f's{seed}_{arm}_{domain}'][m] for arm,w in weights.items()) for seed in available]) for m in METRICS}
                # Four patients stay the denominator after averaging seeds; don't treat 3*4 as independent.
                per_patient={}
                private=exp.get('patient_metrics',{})
                if all(f's{seed}_{arm}_{domain}' in private for seed in available for arm in weights):
                    for metric in ('Dice','NLL','Brier','HD95_penalized_mm'):
                        diffs=[statistics.mean(sum(w*private[f's{seed}_{arm}_{domain}'][i][metric] for arm,w in weights.items()) for seed in available) for i in range(4)]
                        per_patient[metric]=dict(n_patients=4,mean=statistics.mean(diffs),minimum=min(diffs),maximum=max(diffs),positive=sum(v>0 for v in diffs),negative=sum(v<0 for v in diffs))
                result['contrasts'][name][domain]=dict(seeds=available,metrics=metrics,patient_aggregate=per_patient)
        out['experiments'][version]=result
    exp=out['experiments']['v20_P2']
    exp['preregistered_source_gate']={}
    for arm in 'BCD':
        c=exp['contrasts'][arm+'-A']['RUNMC'];m=c['metrics']['Dice']
        n=m['n_seeds'];remaining=3-n
        checks={
            'mean_gain_at_least_1pp':m['mean']>=.01 if n==3 else None,
            'at_least_2_positive_seeds':False if m['positive']+remaining<2 else (True if m['positive']>=2 else None),
            'worst_seed_at_least_minus_0_5pp':False if m['minimum']<-.005 else (True if n==3 else None),
        }
        ratios=[exp['cells'][f's{seed}_{arm}_RUNMC']['HD95_penalized_mm'] for seed in c['seeds']]
        base=[exp['cells'][f's{seed}_A_RUNMC']['HD95_penalized_mm'] for seed in c['seeds']]
        ratio=statistics.mean(ratios)/statistics.mean(base)-1
        checks['mean_HD95_worsening_at_most_10pct']=ratio<=.10 if n==3 else None
        new_empty=any(exp['cells'][f's{seed}_{arm}_RUNMC']['empty_predictions']>exp['cells'][f's{seed}_A_RUNMC']['empty_predictions'] for seed in c['seeds'])
        checks['no_new_empty_predictions']=False if new_empty else (True if n==3 else None)
        failed=[key for key,value in checks.items() if value is False]
        verdict=('FAILED' if n==3 else 'FAILED_DETERMINED_PENDING_FINAL_METRICS') if failed else ('PASS' if n==3 else 'PENDING')
        exp['preregistered_source_gate'][arm]=dict(verdict=verdict,complete_seed_metrics=n,checks=checks,failed_criteria=failed,observed_HD95_relative_change=ratio)
    return out


def report(d):
    date=datetime.datetime.fromtimestamp(d['checked_epoch'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    v19=d['experiments']['v19'];v20=d['experiments']['v20_P2']
    lines=['# v19 三种子完整结果与 v20 P2 阶段汇总','',f'截点：{date}。仅汇总已有封存预测及指标，0新增模型前向/优化/重新评分。','',
           '## 完成状态与核心判断','',
           '- v19：18个训练阶段全部完成、19200更新，36个评估格完整、EXIT0。',
           f"- v20 P2：{len(v20['cells'])}/24格已完整评估，当前{v20['status']['stage']}，seed={v20['status'].get('seed')}、arm={v20['status'].get('arm')}；运行进度见匿名JSON。缺失格保留NA，不能将门槛已确定失败写成运行已经完成。",
           '- 语义一致性：1600步固定终点，已完成两种子的C−A源域均负，800步P1的小幅正收益没有保持。不能选800步替换1600步。',
           '- U图像ELBO：B−A前两个种子源域与迁移均负，完整已出种子逐项见下表；D−C跨种子混合。B/C/D均已无法满足至少2个训练种子源域改善，门槛失败不依赖尚未完成种子的结果。',
           '- 历史先验：v19 L组旧域三种子方向为正但幅度约百分之一个百分点；LU旧域三种子均为负，新增历史先验没有具有实际量级的稳定收益。','',
           '## 数据、配对与解释边界','',
           '训练种子20261023/24/25，冻结split0。源RUNMC为4L患者/80切片，LU另用14U患者；v19目标BMC为4L/14U。每域4验证患者、每格3推断重复，先患者宏平均再平均推断，再汇总训练种子；推断重复不是训练种子，反复开发验证不是独立泛化。test与U隐藏标签未访问。',
           'v19 L/LU源模型分别训练后各分M0/M1。M比较在同源内配对，U比较包含源与目标两阶段累计使用U，不能解释成孤立目标U效应；后验采样流不逐步完全配对。M0是初始全局先验，仍继承源模型与域记忆，不是完全无历史。',
           'v20四臂同源入口、相同Adam/后验/原始先验、独立按步global/L/U随机流。首种子P1 400→800后P2续800→1600；另两种子从各自L400续到1600。A无U学习目标（仍具备U图像访问），B图像U，C语义一致性U，D二者。BMC是未适应迁移，不能与v19目标适应后成绩混为一类。','',
           '## v19 完整三种子终点 Dice（%）','',
           '| 阶段/臂 | 域 | seed23 | seed24 | seed25 | 三seed均值 |','|---|---|---:|---:|---:|---:|']
    for arm in ['L_A','LU_A','L_M0_B','L_M1_B','LU_M0_B','LU_M1_B']:
        for domain in DOMAINS:
            vals=[v19['cells'][f's{s}_{arm}_{domain}']['Dice']*100 for s in SEEDS]
            lines.append(f'| {arm} | {domain} | '+ ' | '.join(f'{v:.4f}' for v in vals)+f' | {statistics.mean(vals):.4f} |')
    def contrast_table(exp,names):
        lines.extend(['','| 对比 Dice（pp） | 域 | 各已完成seed差值 | 均值 ± seed SD | 正/负seed |','|---|---|---|---:|---:|'])
        for name in names:
            for domain in DOMAINS:
                c=exp['contrasts'][name].get(domain)
                if c is None:continue
                m=c['metrics']['Dice'];s=m['sample_sd']
                seq='/'.join(f'{v*100:+.4f}' for v in m['values'])
                lines.append(f"| {name} | {domain} | {seq} | {m['mean']*100:+.4f} ± {s*100:.4f} | {m['positive']}/{m['negative']}（n={m['n_seeds']}） |")
    contrast_table(v19,['source_U','history_L','history_LU','U_at_M0_cumulative','U_at_M1_cumulative','U_history_interaction'])
    lines.extend(['','v19源域LU−L跨seed有正有负；适应后LU相对L的新域均值小幅改善，旧域方向混合。M1−M0极小且依赖L/LU。不能用均值为正替代实际量级、每seed和每患者的稳定性证据。','',
                  '### v19 目标适应相对自身源模型（Dice pp）','',
                  'RUNMC差值为负表示旧域遗忘；BMC差值为负表示目标适应后低于未适应迁移。不是将两个域之间直接相减。'])
    contrast_table(v19,['L_M0_target_minus_source','L_M1_target_minus_source','LU_M0_target_minus_source','LU_M1_target_minus_source'])
    lines.extend(['','## v20 P2 固定1600步，明确缺失格','',
                  '| 臂 | 域 | seed23 | seed24 | seed25 |','|---|---|---:|---:|---:|'])
    for arm in 'ABCD':
        for domain in DOMAINS:
            vals=[f"{100*v20['cells'][f's{s}_{arm}_{domain}']['Dice']:.4f}" if f's{s}_{arm}_{domain}' in v20['cells'] else 'NA（未完整评估）' for s in SEEDS]
            lines.append(f'| {arm} | {domain} | '+' | '.join(vals)+' |')
    lines.extend(['','每项机制差值仅使用该项具备完整配对的种子。B−A已3种子齐全；含C/D的对比当前仍只有2种子；不把多出的A/B单独混入缺失对照。'])
    contrast_table(v20,['B-A','C-A','D-C','D-B','D-A','interaction'])
    lines.extend(['','## 预注册源域门槛逐项判定','',
                  '所有候选均相对A，预注册要求均值≥1pp、至少2seed正、最差≥−.5pp、无新增空预测、平均HD95恶化≤10%。这些条件是AND关系。前两个seed已负意味着第三seed即使为正也最多1/3正，故可以确定失败；未知的最终统计仍保留待完成。','',
                  '| 候选 | 配对seed数 | 均值≥1pp | ≥2seed正 | 最差≥−.5pp | 无新空预测 | HD95≤10%恶化 | 判定 |','|---|---:|---|---|---|---|---|---|'])
    def flag(value):
        return 'PASS' if value is True else ('FAIL' if value is False else '待完整统计')
    for arm,g in v20['preregistered_source_gate'].items():
        c=g['checks'];keys=['mean_gain_at_least_1pp','at_least_2_positive_seeds','worst_seed_at_least_minus_0_5pp','no_new_empty_predictions','mean_HD95_worsening_at_most_10pct']
        lines.append(f"| {arm}−A | {g['complete_seed_metrics']} | "+' | '.join(flag(c[key]) for key in keys)+f" | {g['verdict']} |")
    lines.extend(['','这不是根据两seed估计三seed均值，也不是改变阈值提前淘汰；至少一个已被观察结果不可逆违反的必要条件足以否定联合门槛。既有队列仍按用户冻结指令完成第三seed，不停跑、不调参。D−C的额外贡献独立报告，不能以候选整体失败宣称图像ELBO在所有设置无效。'])
    lines.extend(['','## 辅助指标和患者异质性','',
                  '下表是各配对差值的训练种子均值。NLL、Brier、HD95越低越好；Dice越高越好。患者范围先在同一患者上平均可用种子，再统计4患者范围，未把12次观测当独立患者。','',
                  '| 版本/对比 | 域 | n seeds | ΔNLL | ΔBrier | ΔHD95 mm | 患者Dice差值范围 pp | 正/负患者 |','|---|---|---:|---:|---:|---:|---|---|'])
    for version,exp,names in [('v19',v19,['source_U','history_L','history_LU','U_at_M0_cumulative','U_at_M1_cumulative']),('v20_P2',v20,['B-A','C-A','D-C','D-B','D-A'])]:
        for name in names:
            for domain in DOMAINS:
                c=exp['contrasts'][name][domain];m=c['metrics'];p=c['patient_aggregate']['Dice']
                lines.append(f"| {version}/{name} | {domain} | {len(c['seeds'])} | {m['NLL']['mean']:+.7f} | {m['Brier']['mean']:+.7f} | {m['HD95_penalized_mm']['mean']:+.5f} | [{p['minimum']*100:+.4f}, {p['maximum']*100:+.4f}] | {p['positive']}/{p['negative']} |")
    empty=[(version,k) for version,exp in d['experiments'].items() for k,c in exp['cells'].items() if c['empty_predictions']>0]
    lines.extend(['',f'已完成格中有空预测的格数：{len(empty)}。所有完整终点辅助指标详见[匿名汇总JSON](AGGREGATE_RESULTS.json)，保留负结果。','',
                  '## 科学结论与下一允许动作','',
                  '1. 保留CE＋Dice。其与v18完整首seed源格的改善已有前期报告；v18已取消格不补跑，不声称跨目标三seed配对。',
                  '2. v19已完成收口；不对微小历史先验差异宣称可靠抗遗忘。共享识别器phi没有直接历史KL保护，只能作为待检验机制，不能从现有表格归因。',
                  '3. v20继续原P2队列直至第三seed完整，不改权重/增强、不选择早期checkpoint、不增seed至成功。完整三seed后按预注册源域均值≥1pp、≥2seed正、最差≥−.5pp、无新增空预测及平均HD95恶化≤10%作正式筛选。B/C/D至少2seed正这一必要条件已确定失败；未齐全的最终均值/辅助门槛仍标待完成，不冒充完整三seed统计。',
                  '4. 当前两个seed的固定终点并不支持语义一致性扩展带来分割增益。本配置门槛已失败，完成既有队列后不继续扩样，保留负结果；不得通过反复调参或替换终点抹去失败。',
                  '5. 当前未启动共同源目标U×M或其他新模型。P2完成后由唯一runmc整理最终三seed报告并push，原时钟/空间/隐私边界继续。','',
                  '## 完成与资源证据','',
                  f"v19资源ledger bytes={v19['resources']['bytes']}，秒账单={v19['resources']['seconds_charged']:.2f}；EXIT0。运行中v20_P2 ledger bytes={v20['resources']['bytes']}，秒账单={v20['resources']['seconds_charged']:.2f}，其中运行阶段含预留，不能当作实际已耗时。",'P0/P1/P2原始起点和绝对截止不变；不重启/重置预算，不重新评分。初始启动错误记录继续保留。源码/协议/匿名指标可公开；患者身份、原始图像/预测数组、权重、原始日志、私有配置路径均不发布。','',
                  '本报告为v19最终、v20 P2阶段报告；不会将尚未完成的三seed交付称为最终完成。'])
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('snapshot');parser.add_argument('--out',required=True)
    args=parser.parse_args();output=Path(args.out);output.mkdir(exist_ok=True,parents=True)
    d=summarize(json.loads(Path(args.snapshot).read_text()))
    (output/'AGGREGATE_RESULTS.json').write_text(json.dumps(d,indent=2)+'\n')
    (output/'THREE_SEED_REPORT_ZH.md').write_text(report(d))
    for version,exp in d['experiments'].items():
        print(version)
        for name, domains in exp['contrasts'].items():
            print(name,{domain:round(100*c['metrics']['Dice']['mean'],5) for domain,c in domains.items()})


if __name__=='__main__':main()
