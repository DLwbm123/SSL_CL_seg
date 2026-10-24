"""Aggregate sealed anonymous reports only. No model, data loader, or scoring."""
import json
from pathlib import Path

base=Path(__file__).resolve().parent.parent
runs=[];comparisons=[];endpoints=[]
for p in sorted(base.glob('*_v*/ANALYSIS.json'),key=lambda p:int(p.parent.name.rsplit('_v',1)[1])):
    version=int(p.parent.name.rsplit('_v',1)[1])
    if not 24<=version<=40:continue
    a=json.loads(p.read_text());c=json.loads((p.parent/'EXECUTION_SUMMARY.json').read_text())
    arms={'25':['A','B','C'],'26':['H','R'],'30':['N'],'31':['P'],'32':['I','E'],'33':['J']}.get(str(version),['L','LU'])
    runs.append(dict(version=version,folder=p.parent.name,arms=arms,status=c['status'],formal_updates=c['formal_updates'],wall_seconds=c.get('wall_seconds',c.get('elapsed_seconds')),root_bytes=c['root_bytes']))
    for arm in arms:endpoints.append(dict(version=version,arm=arm,means=a['means'][arm]))
    specs={'LU_minus_L':dict(gate=a['gate'])} if 'gate'in a else a['effects']
    for name,e in specs.items():
        new,old=name.split('_minus_');d={domain:100*(a['means'][new][domain]['Dice']-a['means'][old][domain]['Dice']) for domain in ('RUNMC','BMC')}
        comparisons.append(dict(version=version,comparison=name,dice_delta_pp=d,gate=e['gate'],source_report=f'../{p.parent.name}/REPORT_ZH.md'))
entry=json.loads((base/'semantic_alignment_v23/ANALYSIS.json').read_text())['means']['L']
late=[dict(version=e['version'],arm=e['arm'],BMC_delta_pp=100*(e['means']['BMC']['Dice']-entry['BMC']['Dice'])) for e in endpoints if e['version']>=25]
failure=json.loads((base/'semantic_cycle_v41/EXECUTION_SUMMARY.json').read_text())
x=dict(as_of='2026-10-24 09:10 Asia/Shanghai',scope='v24-v40 full-run sealed results plus v41 implementation failure; not the final 12-hour-window total',runs=runs,comparisons=comparisons,new_endpoints=endpoints,late_entry_transfer_deltas=late,v41_failure=failure,totals=dict(completed_experiment_roots=len(runs),new_training_endpoints=len(endpoints),registered_comparisons=len(comparisons),gate_pass=sum(v['gate']['status']=='PASS' for v in comparisons),formal_updates=sum(v['formal_updates'] for v in runs),sum_run_wall_seconds=sum(v['wall_seconds'] for v in runs),sum_sealed_root_bytes=sum(v['root_bytes'] for v in runs),late_entry_endpoints=len(late),late_entry_transfer_below_entry=sum(v['BMC_delta_pp']<0 for v in late)),new_model_calls=0,new_training_updates=0,new_label_reads=0)
assert len(runs)==17 and len(comparisons)==30 and len(endpoints)==32 and len(late)==30
assert x['totals']['gate_pass']==0 and x['totals']['late_entry_transfer_below_entry']==30
out=Path(__file__).parent;(out/'INTERIM_0910.json').write_text(json.dumps(x,indent=2)+'\n')
lines=['# 09:10阶段路线汇总：已封存结果与下一步证据缺口','','这是北京时间2026-10-24 09:10阶段汇总，不是12:13最终窗口报告。只读取公开封存匿名聚合，不重新推断、训练或读取标签。', '', '## 主要发现', '', 'v24–v40共17个已完成实验根、32个新增训练终点、30项预注册比较，科学门槛通过0项。全部比较的源域≥1pp项失败；v26 H−A还失败于迁移保护。比较共享入口、患者和对照，不能把30项视为30次独立重复或据此进行显著性推断。', '', 'v25–v40从同一v23 L1600入口产生的30个新2000终点，BMC Dice全部低于该入口44.07538%。这是已观察到的继续训练迁移损失模式，不证明唯一机制根因。原监督A额外400步本身源+1.00156pp、迁移−.99230pp；因此相对A的U改善仍可能只补回一部分继续训练损失。不能借此修改固定终点、挑旧checkpoint、放宽门槛或推导所有U机制无效。', '', '| 版本 | 冻结比较 | 源Dice差 pp | 迁移Dice差 pp | 总判定 |', '|---|---|---:|---:|---|']
for r in comparisons:lines.append(f"| [v{r['version']}]({r['source_report']}) | {r['comparison']} | {r['dice_delta_pp']['RUNMC']:+.5f} | {r['dice_delta_pp']['BMC']:+.5f} | {r['gate']['status']} |")
lines+=['', '## 机制层面的解释', '', '固定教师均值、分歧加权和硬化没有建立U净收益。R−H的迁移补偿不能替代R−A；概率误差改善也不能替代分割门槛。v30引入L真实类别的近邻信息后相对A有所改善，但不足1pp且仅单seed；它支持继续关注语义信息来源，不证明近邻目标更正确。', '', 'v31投影相对N、v32图条件相对N、v33风险积分位置相对N都没有建立有意义的增量，不能将这几项拼接为已成功路线。它们分别改变了合并向量、目标结构或分歧惩罚，三点梯度与残差也没有隔离唯一训练根因。', '', 'v28源和迁移患者方向一致，但源增量仍只有+.31712pp；换成v29通道去相关没有保留该增量。不同目标与量级同时变化，无法独立认定假负对是原因。v34–v40的遮挡证据、亲和、平滑、强度观测、面积分布、形状子空间和特征条件密度均未通过；实现可微、解析积分正确、风险降低或参考构造成功均不等于U语义改善。', '', '已完成证据仍缺少“当前L真类别通过在线可微关联进入U表征”的有效终点评估。v41原计划检验这一点，但其Bool argmax预检错误发生在模型前向前，因此这个科学问题仍未得到检验。此缺口不授权违背v41不自动retry，也不支持仅修复后换名。当前没有可报告为已验证的后继方案；本次不登记或启动v42，不为保持队列而任意更换辅助损失。持续主线授权与唯一监测保持，后继设计仍须有独立机制依据，12:13按时累计汇总。', '', '## 成本与完整性', '', f"v24–v40完整运行合计{x['totals']['formal_updates']:,}次正式更新；各根封存占用合计{x['totals']['sum_sealed_root_bytes']:,}字节；各次完整运行历时之和{x['totals']['sum_run_wall_seconds']:.2f}秒。此为全run成本相加，包含窗口开始前已发生的v24/v25成本，且存在并行重叠，不能当12小时窗口增量、GPU活跃时长或墙钟时间。", '', 'v41另有3.545482秒、1,009,632字节、1次小fixture autograd及小张量前向，0模型前向/反向/Adam/评分；无科学门槛判定。v27初次失败及修复已包含原根成本，失败模型前向次数未知，不能写成0。v35终点有效但三点审计文件缺失，不能用未保存遥测解释梯度机制。', '', '正式更新不等总计算：教师/近邻缓存、图求解、标签拟合、方向autograd、surrogate backward与额外审计分别见各报告及执行摘要。本汇总没有新增这些计算，也不以更新数对不同机制作等计算排名。', '', '仅1训练seed、每域4val患者，3推断不是3训练seed。多次开发使用同一val和同一入口，不是独立泛化、可靠不确定性、精确后验或CL证据。完整匿名数值与来源：[INTERIM_0910.json](INTERIM_0910.json)。']
(out/'INTERIM_0910_ZH.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(x['totals']))
