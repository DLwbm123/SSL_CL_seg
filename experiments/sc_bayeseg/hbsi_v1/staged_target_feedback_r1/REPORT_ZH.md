# R1：目标阶段才接通 BayeSeg 生成反馈

**冻结判定：STAGED_FEEDBACK_FAILED。** 两目标臂从同一 C1 Cut 源第1600步状态出发，源 U 与 v45 U1600 逐张量相同；BMC L4患者134片各800更新，同数据、随机流及完整形状/外观图像生成层级。RemainCut 的目标 U 与 v45 U_B800 逐张量相同；SwitchJoint 只改变生成目标能否反馈 U，主要预测仍是双 logit U-Net。C1 Joint 源硬停及其未运行目标链不变。

| 目标臂 | 旧域 RUNMC Dice% | 新域 BMC Dice% | BMC完整L训练Dice% | 旧域惩罚HD95 mm | 新域惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
| RemainCut | 69.334 | 62.107 | 85.252 | 46.633 | 36.439 |
| SwitchJoint | 67.091 | 65.184 | 87.243 | 45.034 | 31.932 |

SwitchJoint−RemainCut 旧/新域为 -2.243/+3.077 pp；相对共同源 U 的旧域变化 +3.830 pp。每个 Dice 是四患者等权三维结果。

| 类别 | 预注册门 | 判定 |
|---|---|---|
| 旧域 | RUNMC_mean_vs_Cut_ge_minus_1pp | FAILED |
| 旧域 | RUNMC_each_vs_Cut_ge_minus_5pp | FAILED |
| 旧域 | RUNMC_HD95_vs_Cut_le_1_10 | PASS |
| 旧域 | RUNMC_mean_vs_source_ge_minus_0_5pp | PASS |
| 旧域 | RUNMC_each_vs_source_ge_minus_5pp | PASS |
| 旧域 | no_RUNMC_patient_empty | PASS |
| 目标 | BMC_mean_vs_Cut_ge_plus_2pp | PASS |
| 目标 | BMC_at_least_3_of_4_nonnegative_vs_Cut | FAILED |
| 目标 | BMC_each_vs_Cut_ge_minus_5pp | PASS |
| 目标 | BMC_train_mean_vs_Cut_ge_minus_5pp | PASS |
| 目标 | BMC_HD95_vs_Cut_le_1_10 | PASS |
| 目标 | no_BMC_patient_empty | PASS |

匿名逐患者 Dice（RUNMC 列依次为源 U、RemainCut、SwitchJoint；BMC 列依次为 RemainCut、SwitchJoint；末列为 SwitchJoint−RemainCut）：

| 域 | 索引 | 源 U | RemainCut | SwitchJoint | 差值pp |
|---|---:|---:|---:|---:|---:|
| RUNMC | 1 | 68.629 | 68.390 | 68.958 | +0.567 |
| RUNMC | 2 | 71.836 | 80.852 | 70.327 | -10.525 |
| RUNMC | 3 | 66.608 | 82.765 | 77.886 | -4.879 |
| RUNMC | 4 | 45.970 | 45.330 | 51.193 | +5.863 |
| BMC | 1 | — | 47.889 | 43.915 | -3.975 |
| BMC | 2 | — | 63.678 | 71.806 | +8.129 |
| BMC | 3 | — | 64.254 | 74.081 | +9.827 |
| BMC | 4 | — | 72.606 | 70.934 | -1.672 |

旧域 SwitchJoint 对 RemainCut：2/4患者下降，最大配对跌幅 -10.525 pp；两臂绝对最差患者 Dice 分别为 45.330%/51.193%，四人线性插值 p10 分别为 52.248%/56.522%。新域 SwitchJoint 虽有均值增益，只有 2/4 患者获益；两臂新域最差 Dice 为 47.889%/43.915%。对四名患者配对差做全部4⁴个有放回重采样所得的经验均值95%描述区间：RUNMC [-7.752,+3.215] pp，BMC [-2.823,+8.978] pp。这些区间没有新独立患者，不能当作总体非劣或疗效置信区间，也不进入预注册门。

最后100批训练读数（生成 bound 为每像素值；它与分割终点不是同一指标）：

| 臂 | 标签风险 | 生成bound/像素 | qS均值logsd | qZ均值logsd |
|---|---:|---:|---:|---:|
| RemainCut | 0.11176 | -2.50668 | -2.50225 | -1.06445 |
| SwitchJoint | 0.11221 | -2.47320 | -2.50202 | -1.06439 |

旧域安全门 **FAILED**，BMC积极门 **FAILED**。本实验只识别当前封存源生成状态下“目标阶段接通反馈”的整条条件效应；两臂训练参数许可不同，不能归因单独形状/外观因子。相同的是目标800更新和标签曝光，生成反馈多了计算与参数更新；C1 Cut 历史源1600更新不计作本轮新成本但必须计入方法总成本。四名开发患者已多次用于路线选择，任何增益仅为探索性，不证明独立泛化、长期CL、统计非劣或绝对旧域安全。隐藏U标签与test未访问。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行成本](EXECUTION_SUMMARY.json)。
