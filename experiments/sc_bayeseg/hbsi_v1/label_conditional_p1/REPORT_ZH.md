# P1：标签条件形状先验与完整图像生成

**冻结判定：MECHANISM_FAILED。** 两臂同H1共同零更新初态、同RUNMC L4患者80片、各1600次更新；只改变训练时形状先验输入`sampled Z`或`observed Y`。`q(S|I)`、独立`q(A|I)`、`p(I|S,A)`全图像生成、qZ空间/Gamma先验均保留，U冻结；预测只输入影像，绝不使用测试标签。Dice为4患者等权三维结果（%）；BMC开发患者多次用于路线选择，属于探索性资源门。

| 臂 | RUNMC开发Dice | BMC开发Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
| U | 63.261 | 30.997 | 82.632 | 27.910 | 43.788 |
| H1_Shape | 63.718 | 31.072 | 83.438 | 26.524 | 40.986 |
| ProbZ | 63.746 | 30.912 | 83.499 | 26.469 | 40.880 |
| LabelY | 63.808 | 30.921 | 83.511 | 26.440 | 41.132 |

LabelY−ProbZ RUNMC/BMC为+0.062/+0.009pp；LabelY−旧U为+0.547/-0.076pp。U/H1 Shape仅不同累积成本的历史参考。

源L患者形状后验与标签对齐、无采样均值重建：

| 匿名索引 | ProbZ绝对Pearson | LabelY绝对Pearson | ProbZ重建MSE | LabelY重建MSE |
|---|---:|---:|---:|---:|
| 1 | 0.4220 | 0.4182 | 0.07106 | 0.05774 |
| 2 | 0.3726 | 0.3751 | 0.05916 | 0.05137 |
| 3 | 0.5130 | 0.5191 | 0.06889 | 0.05621 |
| 4 | 0.5789 | 0.5747 | 0.08685 | 0.06606 |

机制门：

| 门 | 判定 |
|---|---|
| LabelY_abs_Pearson_mean_vs_ProbZ_ge_plus_0_05 | FAILED |
| LabelY_abs_Pearson_at_least_3_of_4_nonnegative | FAILED |
| LabelY_mean_reconstruction_MSE_le_2x_ProbZ | PASS |
| training_and_scales_finite | PASS |

源安全门：

| 门 | 判定 |
|---|---|
| LabelY_RUNMC_mean_vs_ProbZ_ge_minus_1pp | PASS |
| LabelY_RUNMC_each_vs_ProbZ_ge_minus_5pp | PASS |
| LabelY_RUNMC_train_mean_vs_ProbZ_ge_minus_5pp | PASS |
| LabelY_RUNMC_mean_vs_U_ge_minus_1pp | PASS |
| LabelY_RUNMC_each_vs_U_ge_minus_5pp | PASS |
| LabelY_RUNMC_train_mean_vs_U_ge_minus_5pp | PASS |
| LabelY_RUNMC_HD95_vs_both_le_1_10 | PASS |
| LabelY_no_source_collapse | PASS |
| ProbZ_vs_U_mean_ge_minus_1pp | PASS |
| ProbZ_vs_U_each_ge_minus_5pp | PASS |
| ProbZ_no_source_collapse | PASS |

BMC积极门（须机制/源安全全过才允许后继）：

| 门 | 判定 |
|---|---|
| LabelY_BMC_mean_vs_max_ProbZ_U_H1_ge_plus_2pp | FAILED |
| LabelY_BMC_3_of_4_nonnegative_vs_ProbZ_and_U | FAILED |
| LabelY_BMC_each_vs_ProbZ_and_U_ge_minus_5pp | PASS |
| LabelY_BMC_HD95_vs_ProbZ_and_U_le_1_10 | PASS |
| LabelY_no_BMC_collapse | PASS |

匿名开发逐患者Dice：

| 域 | 索引 | U | ProbZ | LabelY |
|---|---:|---:|---:|---:|
| RUNMC | 1 | 68.629 | 69.116 | 69.276 |
| RUNMC | 2 | 71.836 | 72.365 | 72.366 |
| RUNMC | 3 | 66.608 | 67.481 | 67.462 |
| RUNMC | 4 | 45.970 | 46.022 | 46.126 |
| BMC | 1 | 12.997 | 12.670 | 12.970 |
| BMC | 2 | 27.264 | 26.282 | 26.205 |
| BMC | 3 | 52.472 | 52.394 | 52.385 |
| BMC | 4 | 31.256 | 32.301 | 32.125 |

预注册更新前qS梯度范数（标签 / 实际0.01系数后生成；余弦见匿名分析）：

| 步号 | ProbZ | LabelY |
|---|---:|---:|
| 1 | 0.00000 / 0.02445 | 0.00000 / 0.02447 |
| 400 | 0.97438 / 0.35731 | 1.01557 / 0.52278 |
| 800 | 4.82232 / 0.59981 | 2.77360 / 1.38242 |
| 1200 | 8.36110 / 0.40620 | 3.82023 / 1.01083 |
| 1600 | 0.47380 / 0.74876 | 0.78223 / 0.37721 |

最后100批ProbZ/LabelY变分损失=4.199/4.062，样本重建MSE=0.14360/0.12226。源L患者绝对Pearson均值仅从0.47163到0.47178，LabelY虽改善均值重建，却未产生预定形状语义收益。

第一次零更新预检因误把设计中应不同的变分值列入相等断言而退出，0正式更新且未读开发结果；只修该断言后在新根沿原时钟完成。失败根5,818,051字节，修复根202,956,982字节，合计208,775,033字节；[故障记录](PREFLIGHT_INCIDENT_ZH.md)与[成本](EXECUTION_SUMMARY.json)保留。源L相关性不是解耦证明，BMC4名患者不是独立验证。隐藏U标签和test未访问。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)。旧路线硬停不变。
