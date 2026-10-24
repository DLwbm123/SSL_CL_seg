# G1：BMC无标签生成适应

**冻结判定：SOURCE_SAFETY_FAILED。** 两臂从相同F1 Anchor完整形状/外观层级出发，每臂仅用BMC U无标签图像做800次生成目标更新；U-Net与分割头冻结。A_only只更新外观和其他生成参数，S_plus_A额外更新形状后验。Dice为每域4名患者等权三维结果（%），旧U/F1 Anchor只读，累计成本不同。

| 臂 | RUNMC开发Dice | BMC冷迁移Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
| U | 63.261 | 30.997 | 82.632 | 27.910 | 43.788 |
| Anchor | 63.645 | 29.535 | 85.647 | 25.221 | 37.020 |
| A_only | 63.645 | 29.535 | 85.647 | 25.221 | 37.020 |
| S_plus_A | 62.466 | 28.529 | 85.439 | 24.877 | 36.556 |

S_plus_A−A_only RUNMC/BMC分别-1.179/-1.006pp；BMC相对旧U为-2.468pp。A_only的逐患者预测与封存F1 Anchor相同：True。

源安全门：

| 门 | 判定 |
|---|---|
| RUNMC_mean_vs_A_only_ge_minus_1pp | FAILED |
| RUNMC_each_vs_A_only_ge_minus_5pp | PASS |
| RUNMC_train_mean_vs_A_only_ge_minus_5pp | PASS |
| RUNMC_HD95_vs_A_only_le_1_10 | PASS |
| no_source_collapse | PASS |
| full_generator_finite | PASS |

冷BMC积极门（须源门全过才有后继意义）：

| 门 | 判定 |
|---|---|
| BMC_mean_vs_all_ge_plus_2pp | FAILED |
| BMC_3_of_4_nonnegative_vs_A_and_U | FAILED |
| BMC_each_vs_A_and_U_ge_minus_5pp | FAILED |
| BMC_HD95_vs_A_and_U_le_1_10 | PASS |
| no_BMC_collapse | PASS |

匿名逐患者：

| 域 | 索引 | U Dice | A_only Dice | S_plus_A Dice | S_plus_A−A_only pp | S_plus_A−U pp |
|---|---:|---:|---:|---:|---:|---:|
| RUNMC | 1 | 68.629 | 68.919 | 67.724 | -1.195 | -0.905 |
| RUNMC | 2 | 71.836 | 71.355 | 69.727 | -1.628 | -2.109 |
| RUNMC | 3 | 66.608 | 64.902 | 63.095 | -1.807 | -3.513 |
| RUNMC | 4 | 45.970 | 49.403 | 49.316 | -0.086 | +3.346 |
| BMC | 1 | 12.997 | 9.522 | 8.592 | -0.930 | -4.405 |
| BMC | 2 | 27.264 | 21.881 | 20.321 | -1.559 | -6.942 |
| BMC | 3 | 52.472 | 54.886 | 54.421 | -0.465 | +1.949 |
| BMC | 4 | 31.256 | 31.850 | 30.783 | -1.067 | -0.473 |

BMC U训练的生成项最后100批：A_only 2.348、S_plus_A 2.286；重建MSE分别0.01798/0.01723。生成拟合读数不等于分割改善，也不能凭本轮证明形状/外观潜变量已解耦。F1原硬停不变；同一4名开发患者多轮使用，任何正信号都需要独立患者和多种子确认。BMC U隐藏标签/test未访问。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。
