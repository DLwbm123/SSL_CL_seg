"""Format sealed audit aggregates only; no patient, image, model or evaluation access."""
import json
from pathlib import Path
r=Path(__file__).parent;s=json.loads((r/'PUBLIC_SUMMARY.json').read_text());m=s['metrics'];tails=s['patient_tails'];contrasts=s['contrasts_T_minus']
fmt=lambda x:f'{x:.6f}'
ci=lambda v:'['+', '.join(fmt(x) for x in v)+']'
lines=['# APOS V1 单种子完整结果','','**结论：完整候选 T 未能同时达到旧域患者尾部与新域非劣/绝对性能要求，综合 FAIL。** 六份源/适配正式训练及其固定终点评估全部完成；唯一零模型前向终态审计 PASS。不能把工程 PASS 当作科学成功。','','固定种子20271601；主部署为阶段末 EMA .999。D=确定性距离场观测；G=Gaussian 结构后验；T=Student-t 局部残差结构后验（冻结主候选）。各配置源2000步、适配2000步，源与适配各32资格调用。适配只当前Philips 13名训练患者L；固定主体表征/结构先验/细节头，不访问旧训练图、U或测试。原始FixMatch源43600EMA为共同暖启动，不是独立源重复。','','## 1. 源阶段与适配后绝对 Dice','','单位%；旧域23患者、新域31患者，每患者ED/ES、三前景平均。原始源旧域80.367407%。','','| 配置 | 源阶段旧域 | 适配后旧域 | 适配后新域 | 旧域相对本次源差(pp) | 旧域相对原始源差(pp) |','|---|---:|---:|---:|---:|---:|']
for a in ('D','G','T'):
 lines.append(f"| {a} | {m['source'][a]['Siemens']['primary_dice']*100:.6f} | {m['target'][a]['Siemens']['primary_dice']*100:.6f} | {m['target'][a]['Philips']['primary_dice']*100:.6f} | {tails[a]['own_source']['mean_pp']:+.6f} | {tails[a]['original_source']['mean_pp']:+.6f} |")
lines+=['','新域绝对门槛83%：三个配置均未通过。旧域绝对门槛80.367407%：三个配置也均未通过。源阶段D/G/T已经分别比原始源低2.321956/3.889082/2.506029pp；源阶段未评新域，不能补造源新域结果。','','## 2. 旧域患者尾部：两种源参照全部报告','','下降严格按 >1pp、>2pp 计数；p10为逐患者差值的线性10%分位。配对CI为ED/ES共同重采样，10000次、seed163。','','| 配置 | 源参照 | 均值差(pp) | 患者95% CI(pp) | 下降>1pp人数 | 下降>2pp人数 | p10(pp) |','|---|---|---:|---|---:|---:|---:|']
for a in ('D','G','T'):
 for key,title in [('own_source','本次同配置源EMA'),('original_source','原始FixMatch源')]:
  v=tails[a][key];lines.append(f"| {a} | {title} | {v['mean_pp']:+.6f} | {ci(v['CI95_pp'])} | {v['decline_over_1']} | {v['decline_over_2']} | {v['p10_pp']:.6f} |")
lines+=['','冻结尾部门槛：>1pp人数≤2，>2pp人数=0，p10≥−.50pp。**三个配置在两种参照下全部未通过**。T相对自己源均值+0.461014pp不能掩盖9人/7人的尾部退化，其配对CI下界−0.610723pp也低于−.30。T与D相对各自源的两种退化人数相同，只满足“不更差”，不等于安全。','','## 3. 完整候选 T 的配对对照','','| 对比 | 域 | 均值差(pp) | 患者95% CI(pp) | 冻结含义 |','|---|---|---:|---|---|']
for other in ('D','G'):
 for domain in ('Siemens','Philips'):
  v=contrasts[other][domain];meaning=('旧增量≥.30' if other=='D' else '旧增量≥.15且非劣') if domain=='Siemens' else '均值及CI下界≥−.20'
  lines.append(f"| T−{other} | {'旧域' if domain=='Siemens' else '新域'} | {v['mean_pp']:+.6f} | {ci(v['CI95_pp'])} | {meaning} |")
lines+=['','T对D新域+0.674800pp，CI[+0.327471,+1.067478]，相对非劣通过；但T新域仅74.567821%，未达83%，不能将相对较弱对照的通过写成完整新域达标。T对G新域较高，但旧域−0.954086pp且CI全负，未同时保留两域。T对D旧域增量+0.275532pp低于.30，CI跨0。','','## 4. 主候选全部冻结门禁','','| 门禁 | 判定 |','|---|---|']
names={'old_absolute':'旧域绝对≥原始源','old_own_source_CI':'相对本次源旧域CI下界≥−.30','new_absolute':'新域绝对≥83%','new_noninferior_D':'新域相对D均值/CI非劣','old_increment_D':'T−D旧域增量≥.30','old_increment_G':'T−G旧域增量≥.15','noninferior_G_both':'相对G两域同时非劣','tail_own':'相对本次源患者尾部','tail_original':'相对原始源患者尾部','tail_no_worse_D':'本次源参照退化人数不劣D'}
for k,v in s['checks'].items():lines.append(f"| {names[k]} | {'PASS' if v else 'FAIL'} |")
lines+=['','**综合FAIL。主候选保持T，不换成旧域均值更高的G，不调整门槛或新增种子。**','','## 5. 类别与ED/ES完整聚合','','单位%。完整类/相位数据同时保存在PUBLIC_SUMMARY.json。','','| 阶段/配置/域 | LV | MYO | RV | ED均值 | ES均值 |','|---|---:|---:|---:|---:|---:|']
for stage in ('source','target'):
 for a in ('D','G','T'):
  for domain,v in m[stage][a].items():
   values=[100*v['classes'][c] for c in ('LV','MYO','RV')]+[100*v['phases'][phase]['primary_dice'] for phase in ('ED','ES')]
   lines.append('| '+ '/'.join([stage,a,domain])+' | '+' | '.join(fmt(x) for x in values)+' |')
lines+=['','## 6. 科学解释与限制','','本轮没有证实“稳健结构后验能同时改善旧患者尾部和新域适配”。共同确定性观测/距离场解码配置D在源阶段也低于原始源，说明性能损失已在源重构与训练阶段出现，不能把全部失败归咎于持续适配或某一种先验。T比G的新域更高但旧域更低，表现为实际权衡，未形成同时达标。当前证据不足以唯一归因于32网格、源2000步、后验收缩、两点求积或冻结表征中的任何单项；不据此启动逐项扫描。也不能外推所有解剖先验/概率分割都无效。','','23旧/31新患者为重复开发验证，单种子、共享原始源；患者bootstrap CI不覆盖训练随机、多重比较、独立源或独立测试。学到的精度不是已识别的纯医学aleatoric noise；重尾不能区分病理和预测误差；未建立临床无损或无遗忘保证。','','## 7. 成本、修复和交付边界','','真实优化12192；原生encoder/pre-head decoder样本前向29678（含失败2样本），医疗反向12204；低秩条件Gaussian component-sample求解395680；评估5244样本已包含在29678内，不重复相加。源先验拟合230标签切片另列，0模型前向/优化。合成反向调用15（14完整成功+1失败尝试）另列，不混入医疗反向。原始运行总历时4335.199222秒（72.253320分钟），完成时占用252777600B（含保守ops余量），低于8h和4GB；全轮原钟未重置。','','首次源D资格二维CUDA NLL确定性兼容故障发生于2样本前向后，0反向/优化。数学等价gather-mean修复后，从干净initial完整state精确恢复，未重做资格优化或源先验；原EXIT1/异常状态/日志保留在私有incident，异常checkpoint未续跑。另一次独立合成检查遗漏CuBLAS环境的失败尝试同样保存。唯一终态审计PASS、0模型前向，全训练/评估EXIT0、全部worker/controller退休。','','本报告只格式化已封存审计聚合；没有重新训练、评估、审计或读取新图像。公开源码、协议、匿名聚合和审计；患者行/图像/标签/权重/先验/私有配置日志不公开。完整交付后删除唯一监测并等待用户，不自动启动下一轮。']
(r/'APOS_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
