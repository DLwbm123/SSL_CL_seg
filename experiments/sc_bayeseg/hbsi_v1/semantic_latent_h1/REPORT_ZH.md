# H1：直接监督形状与外观潜变量

**冻结判定：NO_SHAPE_SPECIFIC_BMC_SIGNAL。** 两臂从同一C1 Cut完整形状/外观生成层级出发，各新增1600次RUNMC标签加完整生成目标更新；U-Net始终冻结。新增头仅读一个后验均值，主预测固定叠加到U，另有同权直接分割监督。Dice是每域4患者等权三维结果（%）；旧U只读、累计成本不同。

| 臂 | RUNMC开发Dice | BMC冷迁移Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
| U | 63.261 | 30.997 | 82.632 | 27.910 | 43.788 |
| Appearance | 63.520 | 31.005 | 83.312 | 26.649 | 41.533 |
| Shape | 63.718 | 31.072 | 83.438 | 26.524 | 40.986 |

Shape−Appearance RUNMC/BMC为+0.198/+0.067pp；Shape−U为+0.457/+0.075pp。形状与外观后验原容量不同，这些是完整配置差值，不单独证明解耦。

源安全门：

| 门 | 判定 |
|---|---|
| RUNMC_mean_vs_Appearance_ge_minus_1pp | PASS |
| RUNMC_mean_vs_U_ge_minus_1pp | PASS |
| RUNMC_each_vs_Appearance_ge_minus_5pp | PASS |
| RUNMC_each_vs_U_ge_minus_5pp | PASS |
| RUNMC_train_mean_vs_Appearance_ge_minus_5pp | PASS |
| RUNMC_train_mean_vs_U_ge_minus_5pp | PASS |
| RUNMC_HD95_vs_both_le_1_10 | PASS |
| no_source_collapse | PASS |
| full_generator_finite | PASS |

冷BMC形状特异积极门（须源门全过才有后继意义）：

| 门 | 判定 |
|---|---|
| BMC_mean_vs_both_ge_plus_2pp | FAILED |
| BMC_3_of_4_nonnegative_vs_both | FAILED |
| BMC_each_vs_both_ge_minus_5pp | PASS |
| BMC_HD95_vs_both_le_1_10 | PASS |
| no_BMC_collapse | PASS |

匿名逐患者：

| 域 | 索引 | U Dice | Appearance Dice | Shape Dice | Shape−Appearance pp | Shape−U pp |
|---|---:|---:|---:|---:|---:|---:|
| RUNMC | 1 | 68.629 | 69.123 | 69.154 | +0.031 | +0.525 |
| RUNMC | 2 | 71.836 | 71.883 | 72.030 | +0.147 | +0.194 |
| RUNMC | 3 | 66.608 | 67.211 | 67.581 | +0.371 | +0.974 |
| RUNMC | 4 | 45.970 | 45.863 | 46.106 | +0.244 | +0.136 |
| BMC | 1 | 12.997 | 12.767 | 12.799 | +0.032 | -0.198 |
| BMC | 2 | 27.264 | 26.638 | 26.571 | -0.067 | -0.692 |
| BMC | 3 | 52.472 | 52.450 | 52.415 | -0.035 | -0.057 |
| BMC | 4 | 31.256 | 32.165 | 32.503 | +0.338 | +1.246 |

预注册当时状态所选后验梯度审计（标签范数 / 实际0.01系数后生成范数；余弦见匿名分析）：

| 更新前步号 | Appearance | Shape |
|---|---:|---:|
| 1 | 0.00000 / 0.01837 | 0.00000 / 0.02863 |
| 400 | 1.20110 / 1.32007 | 0.70448 / 0.71456 |
| 800 | 0.62973 / 0.18218 | 0.37370 / 0.72515 |
| 1200 | 0.75931 / 1.06779 | 0.69313 / 0.16855 |
| 1600 | 1.65974 / 0.65565 | 0.48354 / 1.40831 |

最后100个源L训练批的生成项损失Appearance为4.326、Shape为4.018；重建MSE分别0.09288/0.13379。该固定配置未访问BMC训练标签或隐藏U标签/test。旧C/F/G负判定不变；同一4名开发患者多轮使用，即使积极门通过也不能替代独立患者、多种子和长期CL证据。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。

终点后另做[零更新固定批探针](POSTHOC_FIXED_PROBE.json)，只描述生成拟合、不参与模型选择：同一RUNMC L训练图像和固定噪声下，初态/Appearance/Shape的生成项损失为2.516/4.158/5.005，重建MSE为0.01690/0.10682/0.18108；同一BMC U无标签图像为2.496/4.118/4.722及0.01640/0.10234/0.14632。两个固定批均显示终态生成拟合恶化，且Shape更明显；仅两张图像，不能推断全域幅度或把恶化单独归因某一个损失项。
