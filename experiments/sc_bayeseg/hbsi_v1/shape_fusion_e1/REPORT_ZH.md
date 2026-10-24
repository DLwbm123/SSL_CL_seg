# E1：保留形状/外观生成层级的形状后验融合

**冻结判定：NO_COLD_BMC_SIGNAL。** 本轮只给封存C1 Cut的标准U-Net/完整生成层级增加小型残差头；原模型全部冻结。Raw与Shape各有1600次新增头更新，同数据、同初始化与容量。U为只读旧绝对参考，未获得同等新增更新；所有Dice为各域4名患者等权三维指标（%）。

| 臂 | RUNMC开发Dice | 冷BMC Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
| U | 63.261 | 30.997 | 82.632 | 27.910 | 43.788 |
| Raw | 63.126 | 29.310 | 85.452 | 25.187 | 37.138 |
| Shape | 63.123 | 29.331 | 85.461 | 25.151 | 36.902 |

形状臂相对原图臂RUNMC开发Dice -0.003pp、冷BMC +0.022pp；相对旧U分别-0.137/-1.666pp。源安全门：

| 门 | 结果 |
|---|---|
| RUNMC_mean_vs_Raw_ge_minus_1pp | PASS |
| RUNMC_mean_vs_U_ge_minus_1pp | PASS |
| RUNMC_each_vs_Raw_ge_minus_5pp | PASS |
| RUNMC_each_vs_U_ge_minus_5pp | PASS |
| RUNMC_train_mean_vs_Raw_ge_minus_5pp | PASS |
| RUNMC_train_mean_vs_U_ge_minus_5pp | PASS |
| RUNMC_HD95_vs_Raw_le_1_10 | PASS |
| RUNMC_HD95_vs_U_le_1_10 | PASS |
| no_collapse_source | PASS |

冷BMC积极信号门（需源安全门全过才有后继意义）：

| 门 | 结果 |
|---|---|
| BMC_mean_vs_both_ge_plus_2pp | FAILED |
| BMC_3_of_4_positive_vs_both | FAILED |
| BMC_each_vs_both_ge_minus_5pp | FAILED |
| BMC_HD95_vs_both_le_1_10 | PASS |
| no_collapse_BMC | PASS |

RUNMC匿名患者：

| 索引 | U Dice | Raw Dice | Shape Dice | Shape−Raw pp | Shape−U pp |
|---|---:|---:|---:|---:|---:|
| 1 | 68.629 | 68.324 | 68.494 | +0.170 | -0.135 |
| 2 | 71.836 | 70.662 | 70.607 | -0.056 | -1.229 |
| 3 | 66.608 | 64.180 | 64.250 | +0.070 | -2.358 |
| 4 | 45.970 | 49.339 | 49.143 | -0.196 | +3.172 |

BMC匿名患者：

| 索引 | U Dice | Raw Dice | Shape Dice | Shape−Raw pp | Shape−U pp |
|---|---:|---:|---:|---:|---:|
| 1 | 12.997 | 9.390 | 9.173 | -0.217 | -3.824 |
| 2 | 27.264 | 21.661 | 21.624 | -0.037 | -5.640 |
| 3 | 52.472 | 54.540 | 54.662 | +0.122 | +2.190 |
| 4 | 31.256 | 31.649 | 31.867 | +0.218 | +0.610 |

[匿名分析](ANALYSIS.json)保留逐患者TP/FP/FN、惩罚HD95、NLL/Brier和完整L训练指标；[执行摘要](EXECUTION_SUMMARY.json)保留实际时间、字节和全部阶段。q(S|I)、q(A|I)、图像重建及形状—分割生成关系完整来自C1 Cut，但本轮主要预测仅使用形状后验**均值**，不证明采样或不确定性有益，也不证明潜变量已解耦。两臂使用C1历史预训练成本，不能把本轮新增头更新当作总成本。重复使用的4名开发患者只支持资源门，不是独立患者证据；隐藏U标签和test未访问。C1/C2原硬停及负结果不被本轮修改。[冻结协议](PREREG_ZH.md)

终点后另做了一次**不用于选择模型**的[零更新训练探针](POSTHOC_TRAIN_PROBE.json)：固定源L数据流前4批，将新增头的第三输入通道置零。形状头首层该通道权重范数1.623（双logit通道范数2.200），并非权重全零；但在四批上置零后平均前景概率仅改变约0.00053–0.00251。这个小探针说明形状特征在当前输出上影响有限，不能证明形状后验本身没有信息，也不能据此重调E1头或推断全部患者情况。
