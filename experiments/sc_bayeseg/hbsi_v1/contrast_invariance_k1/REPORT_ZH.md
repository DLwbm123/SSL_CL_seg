# K1：形状后验灰度不变性

**冻结判定：NO_BMC_GAIN。** 两臂同入口、同RUNMC L原图与单调灰度变换图、同完整BayeSeg形状/外观/图像生成训练，每臂1600次更新；唯一干预是ShapeInv多了系数1.0的形状后验均值两视图一致项。U-Net冻结，主预测仅用原图。Dice为每域4患者等权三维结果（%）。BMC U强度统计参与变换设计，BMC开发结果是**目标无标签分布知情的探索**，不能称严格冷迁移。

| 臂 | RUNMC开发Dice | BMC开发Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
| U | 63.261 | 30.997 | 82.632 | 27.910 | 43.788 |
| H1_Shape | 63.718 | 31.072 | 83.438 | 26.524 | 40.986 |
| AugOnly | 62.821 | 29.987 | 83.409 | 26.405 | 40.682 |
| ShapeInv | 62.875 | 30.091 | 83.356 | 26.438 | 40.709 |

ShapeInv−AugOnly RUNMC/BMC为+0.054/+0.104pp；ShapeInv−旧U为-0.386/-0.907pp。H1单视图Shape只读，累计训练成本不同。

机制门（最后100个训练批）：

| 门 | 判定 |
|---|---|
| last100_invariant_mse_le_0_8_AugOnly | PASS |
| last100_shape_spatial_sd_ge_0_5_AugOnly | PASS |

AugOnly/ShapeInv不变性MSE=0.061888/0.016724；形状特征空间SD=0.21219/0.17289。

源安全门：

| 门 | 判定 |
|---|---|
| RUNMC_mean_vs_AugOnly_ge_minus_1pp | PASS |
| RUNMC_each_vs_AugOnly_ge_minus_5pp | PASS |
| RUNMC_train_mean_vs_AugOnly_ge_minus_5pp | PASS |
| RUNMC_mean_vs_U_ge_minus_1pp | PASS |
| RUNMC_each_vs_U_ge_minus_5pp | PASS |
| RUNMC_train_mean_vs_U_ge_minus_5pp | PASS |
| RUNMC_HD95_vs_both_le_1_10 | PASS |
| no_source_collapse | PASS |
| full_generator_finite | PASS |

BMC积极门（须机制和源门全过才有后继意义）：

| 门 | 判定 |
|---|---|
| BMC_mean_vs_all_ge_plus_2pp | FAILED |
| BMC_3_of_4_nonnegative_vs_AugOnly_and_U | FAILED |
| BMC_each_vs_AugOnly_and_U_ge_minus_5pp | PASS |
| BMC_HD95_vs_AugOnly_and_U_le_1_10 | PASS |
| no_BMC_collapse | PASS |

匿名逐患者：

| 域 | 索引 | U Dice | AugOnly Dice | ShapeInv Dice | ShapeInv−AugOnly pp | ShapeInv−U pp |
|---|---:|---:|---:|---:|---:|---:|
| RUNMC | 1 | 68.629 | 68.177 | 68.188 | +0.010 | -0.441 |
| RUNMC | 2 | 71.836 | 71.161 | 70.917 | -0.244 | -0.920 |
| RUNMC | 3 | 66.608 | 65.495 | 65.898 | +0.402 | -0.710 |
| RUNMC | 4 | 45.970 | 46.451 | 46.498 | +0.047 | +0.528 |
| BMC | 1 | 12.997 | 11.317 | 11.364 | +0.047 | -1.633 |
| BMC | 2 | 27.264 | 25.110 | 25.240 | +0.130 | -2.024 |
| BMC | 3 | 52.472 | 52.504 | 52.502 | -0.003 | +0.030 |
| BMC | 4 | 31.256 | 31.016 | 31.256 | +0.241 | -0.000 |

预注册当时状态的形状后验梯度范数审计（标签 / 实际0.01系数后生成 / 未乘系数的一致项；余弦见匿名分析）：

| 更新前步号 | AugOnly | ShapeInv |
|---|---:|---:|
| 1 | 0.00000 / 0.02044 / 0.36009 | 0.00000 / 0.02044 / 0.36009 |
| 400 | 0.24024 / 0.63993 / 0.03473 | 0.25847 / 0.43144 / 0.01555 |
| 800 | 0.80632 / 0.54151 / 0.35969 | 0.74946 / 0.20093 / 0.03739 |
| 1200 | 1.12834 / 0.33403 / 0.37573 | 1.35855 / 0.07121 / 0.18326 |
| 1600 | 0.43980 / 0.67105 / 0.48680 | 0.73390 / 0.29442 / 0.15319 |

最后100批生成项损失AugOnly/ShapeInv=3.750/3.482，样本重建MSE=0.11393/0.07271。这仍不能单独证明潜变量解耦或临床有效；BMC开发患者反复用于路线判断，隐藏U标签/test未访问。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。旧C/F/G/H/I/J判定不变。
