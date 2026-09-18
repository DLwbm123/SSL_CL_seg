# NATIVE_KEY_ALIGNMENT_V0_1：D1 实验结果

16/16 阶段完整合规结束。**C3 与基线基本持平，未达到预设 D1 性能门槛；本轮不支持有实际幅度的改进主张。** 所有正负结果保留，不自动继续 D2/D3。

实际运行提交：`145b3c1b3ab301031ea379c9d67452e57b28ae1d`。独立结果分支：`codex/native-key-alignment-d1-results`。运行代码checkout保持原批准提交且干净；本结果提交只增加本目录。

## 固定实验与汇总口径

四臂 × 开发seed163/164 × 两顺序，共16个第二目标阶段。O1从已学RIM的共同B2前缀训练Drishti2100步；O2从已学Drishti的共同B2前缀训练RIM3200步。新增source与第一目标训练为0。父方法为NATIVE_LR_SRC_A_3DOMAIN_V1/B2_C06，科学配置未改变。

先在seed/order内配对，再在seed内平均两个顺序，最后平均两个seed。下表为Dice×100；差值单位为百分点。Final是最终三域平均，Old是前两个域平均，Incoming是新目标域；Forget越低越好。

|方法|Final↑|Old↑|Incoming↑|Forget↓|
|---|---:|---:|---:|---:|
|C0 匹配 no-op 基线|67.9011|66.2410|71.2212|10.7007|
|C1 完整144维 patch|67.8896|66.2247|71.2195|10.7170|
|C2 随机8维辅助键|67.8948|66.2316|71.2213|10.7101|
|C3 原生入口 V 的8维键|67.9032|66.2428|71.2240|10.6989|

## 主要配对结果

|比较|ΔFinal|ΔOld|ΔIncoming|ΔForget|
|---|---:|---:|---:|---:|
|C3−C0|+0.002122|+0.001782|+0.002800|-0.001782|
|C3−C2|+0.008379|+0.011218|+0.002702|-0.011218|
|C3−C1|+0.013615|+0.018157|+0.004533|-0.018157|

C3−C0的Final仅+0.002122个百分点（原始Dice+0.000021218），低于预设+0.5个百分点（原始Dice+0.005）门槛。四个seed/order配对中2正2负；两seed内顺序平均后分别为+0.003740与+0.000503个百分点，均很小。

C3−C2的Final四个seed/order均为正，两seed均为正，方向一致性检查通过；但均值仅+0.008379个百分点。C3−C1平均+0.013615个百分点，四个配对中3正1负。平均超过两个对照这一布尔检查通过，不等于效果显著或具备实用幅度。

冻结门槛：D1_performance=false；key_vs_random_direction_consistent=true；exceeds_both_controls=true。其他性能条件（各seed Final>0、O2 Old不下降、各顺序Incoming下降不超过0.5个百分点）通过，但主均值提升未达标。recommendation=STOP_NO_AUTOMATIC_FOLLOWUP。

## 诊断：辅助项确实生效，但影响很小

C1/C2/C3各有8480个活跃U更新，全部具有有效类别支持；不存在因为无支持而全程关闭辅助项的问题。C0的valid_steps=0是定义上的no-op，不能当成数据缺失。

在每臂4阶段×4个固定诊断点（16点）上，C3辅助梯度范数/监督梯度范数的中位数为0.1682%，辅助/KL为0.7210%。这些是既定诊断向量的范数比，非optimizer实际合并更新的因果分解，也不能推出调大权重就会改善。

每域rim/cup、第一目标刚学完及最终分数均完整保存在DOMAIN_METRICS.csv。每个seed/order的正负配对保存在PAIRED_COMPARISONS.csv，固定梯度与支持统计保存在DIAGNOSTICS.json。

## 完成与成本

- 正式阶段16/16；实际模型验收16/16；四份历史共同前缀全部VERIFIED，当前完整性证明有效。
- CUDA12/12用例通过，24次物理调用，含4次预设after_optimizer失败注入。
- Smoke四臂各8次，共32次当前L-only更新；未读取U/val/test，全部状态丢弃。
- 正式科学/物理调用42400；真实总量42432。新增source=0、第一目标训练=0。
- 旧D0 CPU30、R1补充CPU34，总CPU64，未再运行。各账本口径不混用。
- 正式worker session合计11772.53秒（约3.27小时，包含评价）；连同资格、前缀与目标完整性验收约11802.85秒（约3.28小时）。这是累计worker session墙钟，不是独占GPU计费时长。
- 正式/验收/资格未发生非预设失败，未额外重试。CUDA的4次预设失败属于已通过资格的一部分；历史CPU R1两次预设失败留在原CPU成本证据中。
- 正式额外clean-U前向33920次（每臂8480，C0也匹配曝光）；诊断额外VJP176次（C0为32，其余各48）。这些是操作计数子集，不能再与总操作数相加。
- 全部会话最大CUDA allocated约1.965GiB；完整操作/显存/会话成本见COST_AND_COMPLETION.json。C3四阶段worker session2961.69秒，C0为2858.51秒，约多3.6%；顺序共享机器测量不作为严格吞吐基准。

## 结论边界与状态

本轮是已观察seed163/164上的共同前缀、第二目标域切换诊断；不是完整新方法轨迹、独立患者验证、原始KI复现或SOTA比较。共同前缀使ΔForget=−ΔOld，不能当成两份独立证据。微小数值排序不构成统计显著性证据。

状态：COMPLETE_D1_AWAITING_SCIENTIFIC_REVIEW。实验已停止，D2/D3未启动；没有追加seed、调参或监测。

公开文件仅含聚合结果、诊断、成本、环境、资格与必要完整性元数据。模型权重、图像/标签、逐样本/患者记录、私有路径与凭据均未发布。

## 文件

- [原始机器生成报告](FINAL_REPORT.md)、[全部聚合数据](PUBLIC_RESULTS.json)
- [Final表16行](FINAL_METRICS.csv)、[域表48行](DOMAIN_METRICS.csv)、[配对表21行](PAIRED_COMPARISONS.csv)
- [诊断](DIAGNOSTICS.json)、[成本](COST_AND_COMPLETION.json)
- [完成复核](COMPLETION_CHECK.json)、[公开完整性证明](PUBLIC_COMPLETION_PROOFS.json)
- [CUDA资格](CUDA_QUALIFICATION.json)、[Smoke](SMOKE.json)、[环境](RUNTIME_ENVIRONMENT.json)
