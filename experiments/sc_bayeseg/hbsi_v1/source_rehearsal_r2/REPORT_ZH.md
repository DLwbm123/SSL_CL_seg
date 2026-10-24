# R2：源标签重放下的目标生成反馈

**冻结判定：REHEARSAL_FEEDBACK_FAILED。** 两臂继承同一 C1 Cut 源1600状态，目标阶段各800次更新；每步均取一张RUNMC源L图和一张BMC目标L图，各占监督风险一半，目标图额外保留完整BayeSeg形状/外观生成目标。唯一臂间干预是目标生成梯度是否进入U-Net；主预测均为标准双logit U。每臂均有800源＋800目标标签曝光，不能与R1无重放臂当成等暴露性能比较。

| 臂 | RUNMC开发Dice% | BMC开发Dice% | RUNMC完整L训练Dice% | BMC完整L训练Dice% |
|---|---:|---:|---:|---:|
| ReplayCut | 63.923 | 56.182 | 83.858 | 75.731 |
| ReplayJoint | 71.442 | 68.027 | 89.892 | 85.711 |

ReplayJoint−ReplayCut 的旧/新域开发均值变化为 +7.519/+11.845pp；相对封存源U旧域均值 +8.181pp。每个Dice是四患者等权三维结果。

只读R1无重放终点的RUNMC/BMC开发Dice为RemainCut 69.334/62.107%，SwitchJoint 67.091/65.184%。R2的ReplayCut相对R1 RemainCut为 -5.411/-5.925pp，ReplayJoint相对R1 SwitchJoint为 +4.351/+2.843pp。重放同时改变源标签曝光与每步任务混合，这些跨实验差异不构成单独生成反馈的因果估计。

| 臂 | BMC precision% | BMC recall% | BMC惩罚HD95 mm | BMC NLL | BMC Brier |
|---|---:|---:|---:|---:|---:|
| ReplayCut | 75.260 | 47.697 | 21.292 | 0.08687 | 0.02661 |
| ReplayJoint | 70.028 | 71.711 | 27.377 | 0.06123 | 0.02202 |

BMC惩罚HD95均值从 21.292 增至 27.377mm，比例 1.286，超过冻结1.10上限；Dice显著提高不能覆盖该边界损失。
匿名BMC四人的惩罚HD95 Cut→Joint 为 18.382→17.203, 19.711→20.926, 9.000→22.952, 38.075→48.427mm；第3、4人明显上升。
相同BMC完整L训练患者的惩罚HD95则为 6.265→5.415mm；训练边界改善与开发边界恶化并存，支持将风险定位为跨患者边界泛化问题的线索，但不能由此确定唯一机制。

| 类别 | 预注册门 | 判定 |
|---|---|---|
| 旧域 | RUNMC_mean_vs_Cut_ge_minus_1pp | PASS |
| 旧域 | RUNMC_each_vs_Cut_ge_minus_5pp | PASS |
| 旧域 | RUNMC_HD95_vs_Cut_le_1_10 | PASS |
| 旧域 | RUNMC_mean_vs_source_ge_minus_0_5pp | PASS |
| 旧域 | RUNMC_each_vs_source_ge_minus_5pp | PASS |
| 旧域 | no_RUNMC_patient_empty | PASS |
| 目标 | BMC_mean_vs_Cut_ge_plus_2pp | PASS |
| 目标 | BMC_at_least_3_of_4_nonnegative_vs_Cut | PASS |
| 目标 | BMC_each_vs_Cut_ge_minus_5pp | PASS |
| 目标 | BMC_HD95_vs_Cut_le_1_10 | FAILED |
| 目标 | BMC_train_mean_vs_Cut_ge_minus_5pp | PASS |
| 目标 | no_BMC_patient_empty | PASS |

匿名逐患者Dice，最后一列为ReplayJoint−ReplayCut百分点：

| 域 | 索引 | 源U | ReplayCut | ReplayJoint | 差值pp |
|---|---:|---:|---:|---:|---:|
| RUNMC | 1 | 68.629 | 62.717 | 68.628 | +5.910 |
| RUNMC | 2 | 71.836 | 54.154 | 67.716 | +13.562 |
| RUNMC | 3 | 66.608 | 60.393 | 68.454 | +8.061 |
| RUNMC | 4 | 45.970 | 78.430 | 80.971 | +2.541 |
| BMC | 1 | — | 22.402 | 52.995 | +30.593 |
| BMC | 2 | — | 64.409 | 75.300 | +10.890 |
| BMC | 3 | — | 71.463 | 77.178 | +5.715 |
| BMC | 4 | — | 66.452 | 66.635 | +0.183 |

旧域相对ReplayCut的最小配对差 +2.541pp、下降 0/4 患者；新域最小配对差 +0.183pp、获益 4/4。旧域绝对最差Dice Cut/Joint=54.154/67.716%，四人插值p10=56.025/67.937%；新域绝对最差=22.402/52.995%。四名配对患者的4⁴有放回重采样均值95%经验描述区间：旧域 [+3.921,+11.649]pp，新域 [+2.860,+24.373]pp；这些并非独立患者总体推断，也不进入门槛。

最后100批训练读数（生成bound每像素，不等于分割）：

| 臂 | 源标签风险 | 目标标签风险 | 目标生成bound/像素 |
|---|---:|---:|---:|
| ReplayCut | 0.12512 | 0.10891 | -2.59337 |
| ReplayJoint | 0.12211 | 0.11253 | -2.52967 |

旧域安全门 **PASS**，BMC积极门 **FAILED**。任一门失败则固定策略关闭，不调重放比例、生成系数、步数、阈值、种子或检查点。C1/R1/S1原冻结判定不变。四名源/目标开发患者已反复用于路线选择，本轮至多提供探索性资源信号；未取得独立患者、多种子、无重放或长序列持续学习证据。隐藏U标签与test未访问。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[成本](EXECUTION_SUMMARY.json)。
