"""Only sealed fixed endpoints; B-A and C-B have distinct, unchanged gates."""
import argparse,json
from pathlib import Path
from statistics import mean
from ..trainability_v21.summarize import patients,METRICS


def analyze(x):
    assert x['STATUS.json']['stage']=='COMPLETE' and x['EXIT.json']['exit_code']==0
    assert x['RESULTS.json']['new_formal_updates']==1200 and x['RESULTS.json']['counter']['outer_backward_calls']==1200
    assert len(x['RESULTS.json']['cells'])==6 and all(v['complete'] and v['patients']==4 for v in x['RESULTS.json']['cells'].values())
    assert len({x[f'BRANCH_{a}.json']['common_payload'] for a in 'ABC'})==1
    assert len({x[f'BRANCH_{a}.json']['prior_hash'] for a in 'ABC'})==1
    ledger=x['ledger'];assert len(ledger['training_sessions'])==3 and ledger['updates_reserved']==1200
    assert all(v['status']=='PASS' for v in ledger['attempts'])
    assert all(v['status']=='COMPLETE' and v['completed_steps']==2000 for v in ledger['training_sessions'].values())
    assert all([r['step'] for r in x['audit'][a]]==[1600,1799,1999] for a in 'ABC')
    assert sum(r['extra_reverse_calls'] for a in 'ABC' for r in x['audit'][a])==15
    pp={a:{d:patients(rows) for d,rows in ds.items()} for a,ds in x['patients'].items()}
    means={a:{d:{k:mean(v[k] for v in rows.values()) if k!='empty_predictions' else sum(v[k] for v in rows.values()) for k in METRICS} for d,rows in ds.items()} for a,ds in pp.items()}
    effects={}
    for name,arm,control in [('B_minus_A','B','A'),('C_minus_B','C','B'),('C_minus_A','C','A')]:
        ds={}
        for d in ('RUNMC','BMC'):
            assert pp[arm][d].keys()==pp[control][d].keys()
            m={k:means[arm][d][k]-means[control][d][k] for k in METRICS};paired={}
            for k in METRICS:
                v=[pp[arm][d][i][k]-pp[control][d][i][k] for i in pp[arm][d]]
                paired[k]=dict(n=4,mean=mean(v),minimum=min(v),maximum=max(v),positive=sum(t>1e-12 for t in v),negative=sum(t< -1e-12 for t in v))
            ds[d]=dict(metrics=m,patient_aggregates=paired)
        checks=dict(source_gain_ge_1pp=ds['RUNMC']['metrics']['Dice']*100>=1,transfer_drop_le_0_5pp=ds['BMC']['metrics']['Dice']*100>=-.5,no_new_empty=all(means[arm][d]['empty_predictions']<=means[control][d]['empty_predictions'] for d in ds),HD95_worsening_le_10pct=all(means[arm][d]['HD95_penalized_mm']<=1.1*means[control][d]['HD95_penalized_mm'] for d in ds))
        effects[name]=dict(domains=ds,gate=dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks))
    return dict(status='COMPLETE',seed=20261023,training_seeds=1,patients_per_domain=4,inference_repeats=3,means=means,effects=effects,training=x['training'],audit=x['audit'],teacher=x['TEACHER_SUMMARY.json'])


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');args=p.parse_args();x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    previous=json.loads((out.parent/'semantic_alignment_v23/ANALYSIS.json').read_text())
    a['relative_to_sealed_v23_L1600_pp']={arm:{d:100*(a['means'][arm][d]['Dice']-previous['means']['L'][d]['Dice']) for d in ('RUNMC','BMC')} for arm in 'ABC'}
    cost=dict(status='COMPLETE',original_start=x['LAUNCH.json']['started'],hard_deadline=x['LAUNCH.json']['deadline'],exit_time=x['EXIT.json']['time'],elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=1200,teacher_prediction_forwards=8672,preflight_reverse_calls=3,training_telemetry_reverse_calls=15,extra_optimizer_updates=0,ledger={k:x['ledger'][k] for k in ('limits','seconds_charged','updates_reserved','bytes')})
    lines=['# v25 固定后验教师：A/B/C 固定2000终点','', '三臂共同v23 L1600活动状态，各继续400步；固定教师不更新。A无U风险，B均值教师，C仅比B多类内归一化后验分歧权重。', '', '| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 |','|---|---|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {100*m['Dice']:.5f} | {m['NLL']:.7f} | {m['Brier']:.7f} | {m['HD95_penalized_mm']:.5f} | {m['empty_predictions']:.3f} |")
    lines+=['','| 对比 | 源Dice增量 pp | 迁移Dice增量 pp | 原门槛 |','|---|---:|---:|---|']
    for name,v in a['effects'].items():lines.append(f"| {name} | {100*v['domains']['RUNMC']['metrics']['Dice']:+.5f} | {100*v['domains']['BMC']['metrics']['Dice']:+.5f} | {v['gate']['status']} |")
    lines+=['','三个对比分别应用原≥1pp源增益、迁移降≤0.5pp、无新增空预测、HD95恶化≤10%门槛。C−A不能替代C−B；小正差不改称通过。', '', '| 对比 | 域 | 患者Dice范围 pp | 正/负 | NLL变化 | Brier变化 | HD95变化 |','|---|---|---:|---:|---:|---:|---:|']
    for name,v in a['effects'].items():
        for d,ds in v['domains'].items():
            m=ds['metrics'];r=ds['patient_aggregates']['Dice'];lines.append(f"| {name} | {d} | [{100*r['minimum']:+.5f},{100*r['maximum']:+.5f}] | {r['positive']}/{r['negative']} | {m['NLL']:+.7f} | {m['Brier']:+.7f} | {m['HD95_penalized_mm']:+.5f} |")
    lines+=['','逐项判定：','']
    for name,v in a['effects'].items():lines.append(f"- {name}: "+'; '.join(f"{k}={'PASS' if b else 'FAILED'}" for k,b in v['gate']['checks'].items()))
    lines+=['','| 臂 | 相对封存L1600源Dice pp | 相对封存L1600迁移Dice pp |','|---|---:|---:|']
    for arm,v in a['relative_to_sealed_v23_L1600_pp'].items():lines.append(f"| {arm} | {v['RUNMC']:+.5f} | {v['BMC']:+.5f} |")
    lines+=['','相对1600只是绝对背景，包含额外训练400步，不能全算U。若只因对照退化得到相对提升，必须与净改进区分。','', '## 教师与训练解释','', f"一次缓存{a['teacher']['slices']}张U切片，8全局×4局部；tau={a['teacher']['tau']:.8g}。教师无更新且无U标签访问。",'', '| 教师组 | 像素占比 | 平均权重 | 有效像素数 |','|---|---:|---:|---:|']
    for name,v in a['teacher']['groups'].items():lines.append(f"| {name} | {v['fraction']:.6f} | {v['weight_mean']:.6f} | {v['effective_pixels']:.2f} |")
    lines+=['','权重不等于已校准正确率，所有权重为正，没有可靠前景硬筛选成功的结论。8×4嵌套样本不是32个独立全局样本；全局抽样跨图像复用。', '', '预定三步的分组梯度与合并Adam位移完整记录于ANALYSIS.json。基础梯度包含L任务/图像/anchor/KL，合并位移不是U反事实更新；只用于解释，不追改门槛或选checkpoint。', '', 'B−A检验固定教师带来的U增量；C−B检验本规则的分歧加权增量，不能据此证明整套Bayesian方法优于确定性模型。单seed、每域4名反复使用的开发val，3推断不是3训练seed。BMC未适应，不是CL成功。', '', f"正式1200更新，教师8672预测前向，3预检和15训练遥测额外反向，无额外Adam；全根{cost['root_bytes']}字节，历时{cost['elapsed_seconds']:.2f}秒。来源L1600成本保留，没有重评分历史模型。",'', '失败不扩seed/权重/tau/阈值/增强求成功；通过仅是独立确认候选。[冻结协议](PREREG_ZH.md)；[全指标及遥测](ANALYSIS.json)；[执行成本](EXECUTION_SUMMARY.json)。']
    lines += ["", '## 实测结论与下一机制问题\n\n固定教师B−A源−0.00423pp、BMC−0.11756pp，近零或略负；两域各2名患者改善、2名下降。B的NLL和Brier均值降低，说明概率拟合指标与固定决策分割收益并不等价。C−B源−0.15673pp、BMC−0.18167pp，NLL和Brier也变差；当前后验分歧类内加权没有额外收益。三个对比均只因源≥1pp门失败，其余保护门通过；不放宽门槛，不扩样调tau。\n\nA自身从1600到2000源+1.00156pp、BMC−0.99230pp。B/C的源绝对提升没有超过同算力A，不能把额外训练提升记作U收益；BMC绝对下降也不能忽略。\n\n预定训练遥测排除了“这三处U梯度总是太小或未接通”的说法：起点B/C的q均值U/基础梯度范数均约1.48，余弦约−0.905/−0.960；phi_cR比例约1.63/1.57，余弦约−0.958/−0.978。中点部分方向转正，末点q均值比例降至0.087/0.023，方向混合。基础项包括任务、图像、anchor和KL；三个批次不证明总体冲突根因。合并Adam位移非U单独贡献。C最后100步U风险0.00857低于B的0.02737，是不同加权目标和训练轨迹，不能据小loss证明更可靠。\n\n当前可检验的新问题是：软教师分布匹配能改变概率误差，却没有直接建立区域决策收益。下一独立实验将复用同一教师和匹配A/B端点，分别检验固定教师硬类别监督相对软KL，以及硬类别监督加入前景Dice的区域目标；权重沿既有CE=1、Dice=1，不搜索温度/权重/阈值。该假设未被本次结果证实，硬标签可能固化教师错误；必须由新固定终点与原门槛否证，不能宣称已找到根因。保留固定教师、数据、视图、学生抽样及原L目标，避免把改教师和改风险混在一起。\n']
    for n,v in [('ANALYSIS.json',a),('RESULTS.json',x['RESULTS.json']),('EXECUTION_SUMMARY.json',cost)]:(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v['gate'] for k,v in a['effects'].items()}))

if __name__=='__main__':main()
