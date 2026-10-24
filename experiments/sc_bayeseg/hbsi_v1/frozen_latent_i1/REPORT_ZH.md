# I1：冻结生成层级后的直接监督潜变量头

**冻结判定：SOURCE_SAFETY_FAILED。** 两臂从H1共同零更新状态出发，完整BayeSeg形状/外观生成层级与标准U-Net全程逐张量冻结，仅同容量头各做1600次RUNMC L标签更新。H1 Joint-Shape和旧U只读参照；Dice为每域4患者等权三维结果（%）。

| 臂 | RUNMC开发Dice | BMC冷迁移Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
| U | 63.261 | 30.997 | 82.632 | 27.910 | 43.788 |
| H1_Shape | 63.718 | 31.072 | 83.438 | 26.524 | 40.986 |
| I_Appearance | 62.128 | 29.906 | 83.293 | 26.288 | 40.728 |
| I_Shape | 62.170 | 29.893 | 83.334 | 26.269 | 40.849 |

I-Shape−I-Appearance RUNMC/BMC为+0.043/-0.013pp；I-Shape−H1 Joint-Shape为-1.548/-1.179pp；I-Shape−旧U为-1.091/-1.104pp。I1和H1同入口与任务数据流，但更新参数集合不同，历史对比只能描述整条训练策略效应。

源安全门：

| 门 | 判定 |
|---|---|
| RUNMC_mean_vs_I_Appearance_ge_minus_1pp | PASS |
| RUNMC_each_vs_I_Appearance_ge_minus_5pp | PASS |
| RUNMC_train_mean_vs_I_Appearance_ge_minus_5pp | PASS |
| RUNMC_mean_vs_H1_Shape_ge_minus_1pp | FAILED |
| RUNMC_each_vs_H1_Shape_ge_minus_5pp | PASS |
| RUNMC_train_mean_vs_H1_Shape_ge_minus_5pp | PASS |
| RUNMC_mean_vs_U_ge_minus_1pp | FAILED |
| RUNMC_each_vs_U_ge_minus_5pp | PASS |
| RUNMC_train_mean_vs_U_ge_minus_5pp | PASS |
| RUNMC_HD95_vs_all_le_1_10 | PASS |
| no_source_collapse | PASS |
| full_generator_finite_and_exactly_frozen | PASS |

冷BMC积极门（须源门全过才有后继意义）：

| 门 | 判定 |
|---|---|
| BMC_mean_vs_all_ge_plus_2pp | FAILED |
| BMC_3_of_4_nonnegative_vs_I_Appearance_and_U | FAILED |
| BMC_each_vs_I_Appearance_and_U_ge_minus_5pp | PASS |
| BMC_HD95_vs_I_Appearance_and_U_le_1_10 | PASS |
| no_BMC_collapse | PASS |

匿名逐患者：

| 域 | 索引 | U Dice | H1 Joint-Shape Dice | I-Appearance Dice | I-Shape Dice | I-Shape−I-Appearance pp | I-Shape−U pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| RUNMC | 1 | 68.629 | 69.154 | 67.795 | 67.776 | -0.019 | -0.853 |
| RUNMC | 2 | 71.836 | 72.030 | 70.046 | 70.109 | +0.062 | -1.728 |
| RUNMC | 3 | 66.608 | 67.581 | 64.346 | 64.366 | +0.019 | -2.242 |
| RUNMC | 4 | 45.970 | 46.106 | 46.323 | 46.431 | +0.108 | +0.461 |
| BMC | 1 | 12.997 | 12.799 | 11.212 | 11.252 | +0.040 | -1.745 |
| BMC | 2 | 27.264 | 26.571 | 24.812 | 24.779 | -0.033 | -2.485 |
| BMC | 3 | 52.472 | 52.415 | 52.539 | 52.550 | +0.011 | +0.078 |
| BMC | 4 | 31.256 | 32.503 | 31.062 | 30.991 | -0.070 | -0.265 |

完整生成层级的终态逐张量与H1入口完全相同。最后100个源L训练批的生成项损失I-Appearance为2.488、I-Shape为2.488；样本图像重建MSE分别0.01682/0.01682。冻结生成读数仅描述同一图像流，不构成潜变量解耦证明。BMC无训练，U隐藏标签/test未访问。H1/C/F/G旧硬停不变；重复使用的4名开发患者只能支持资源门。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。
