# N1：限制外观后验空间带宽

**冻结判定：SWAP_MECHANISM_FAILED。** 从L1共同零更新初态出发，两臂同RUNMC L4患者80片、各1600次更新，完整保留形状/外观图像生成及空间Gamma先验，U冻结，生成梯度到qS。Smooth仅将qA均值和尺度固定池化到4×4后上采样，Full原样128×128。Dice为每域4患者等权三维结果（%）；BMC开发患者被多次用于路线选择，仅为探索性资源门。

| 臂 | RUNMC开发Dice | BMC开发Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
| U | 63.261 | 30.997 | 82.632 | 27.910 | 43.788 |
| H1_Shape | 63.718 | 31.072 | 83.438 | 26.524 | 40.986 |
| L1_Cut | 63.541 | 30.488 | 83.749 | 26.519 | 40.638 |
| Full | 62.688 | 29.565 | 84.146 | 25.829 | 39.631 |
| Smooth | 62.704 | 29.612 | 84.103 | 25.853 | 39.675 |

Smooth−Full RUNMC/BMC为+0.016/+0.047pp；Smooth−U为-0.557/-1.385pp。U/H1/L1仅历史参考，累积训练成本不匹配。

源L患者固定跨患者外观交换审计，冻结U读数：

| 臂 | 原图Dice | 自身重建Dice | 外观交换Dice | 重建MSE | 交换图越界像素% |
|---|---:|---:|---:|---:|---:|
| Full | 82.632 | 82.711 | 61.872 | 0.00043 | 0.020 |
| Smooth | 82.632 | 0.008 | 0.051 | 0.15030 | 0.243 |

交换机制门：

| 门 | 判定 |
|---|---|
| Smooth_self_reconstruction_mse_le_0_02 | FAILED |
| Smooth_swap_outside_fraction_le_0_01 | PASS |
| Smooth_swap_Dice_vs_Full_ge_plus_10pp | FAILED |
| Smooth_swap_Dice_vs_own_original_ge_minus_5pp | FAILED |
| Smooth_swap_each_vs_original_ge_minus_10pp | FAILED |
| Smooth_swap_no_patient_empty | PASS |

源安全门：

| 门 | 判定 |
|---|---|
| Smooth_RUNMC_mean_vs_Full_ge_minus_1pp | PASS |
| Smooth_RUNMC_each_vs_Full_ge_minus_5pp | PASS |
| Smooth_RUNMC_train_mean_vs_Full_ge_minus_5pp | PASS |
| Smooth_RUNMC_mean_vs_U_ge_minus_1pp | PASS |
| Smooth_RUNMC_each_vs_U_ge_minus_5pp | PASS |
| Smooth_RUNMC_train_mean_vs_U_ge_minus_5pp | PASS |
| Smooth_RUNMC_HD95_vs_both_le_1_10 | PASS |
| Smooth_no_source_collapse | PASS |
| Full_vs_U_mean_ge_minus_1pp | PASS |
| Full_vs_U_each_ge_minus_5pp | PASS |
| Full_no_source_collapse | PASS |
| finite_training_and_scales | PASS |

BMC积极门（须交换机制与源安全全过才允许后继）：

| 门 | 判定 |
|---|---|
| Smooth_BMC_mean_vs_max_Full_U_H1_L1_ge_plus_2pp | FAILED |
| Smooth_BMC_3_of_4_nonnegative_vs_Full_and_U | FAILED |
| Smooth_BMC_each_vs_Full_and_U_ge_minus_5pp | PASS |
| Smooth_BMC_HD95_vs_Full_and_U_le_1_10 | PASS |
| Smooth_no_BMC_collapse | PASS |

匿名开发患者Dice：

| 域 | 索引 | U | Full | Smooth |
|---|---:|---:|---:|---:|
| RUNMC | 1 | 68.629 | 67.991 | 68.022 |
| RUNMC | 2 | 71.836 | 70.691 | 70.733 |
| RUNMC | 3 | 66.608 | 64.700 | 64.761 |
| RUNMC | 4 | 45.970 | 47.372 | 47.300 |
| BMC | 1 | 12.997 | 11.019 | 11.066 |
| BMC | 2 | 27.264 | 23.753 | 23.902 |
| BMC | 3 | 52.472 | 52.732 | 52.696 |
| BMC | 4 | 31.256 | 30.757 | 30.786 |

匿名源L交换患者Dice：

| 臂 | 索引 | 原图 | 自身重建 | 外观交换 |
|---|---:|---:|---:|---:|
| Full | 1 | 90.520 | 90.255 | 60.891 |
| Full | 2 | 72.654 | 73.060 | 61.977 |
| Full | 3 | 84.110 | 83.822 | 59.013 |
| Full | 4 | 83.242 | 83.708 | 65.606 |
| Smooth | 1 | 90.520 | 0.031 | 0.000 |
| Smooth | 2 | 72.654 | 0.000 | 0.000 |
| Smooth | 3 | 84.110 | 0.000 | 0.000 |
| Smooth | 4 | 83.242 | 0.000 | 0.203 |

预注册更新前qS梯度范数（标签 / 实际0.01系数后生成；余弦见匿名分析）：

| 步号 | Full | Smooth |
|---|---:|---:|
| 1 | 0.18413 / 0.10602 | 0.18413 / 0.10601 |
| 400 | 1.04269 / 0.01690 | 1.08093 / 0.02058 |
| 800 | 0.13115 / 0.13370 | 0.13263 / 0.13420 |
| 1200 | 0.63133 / 0.16796 | 0.62085 / 0.16246 |
| 1600 | 0.25944 / 0.01893 | 0.24160 / 0.01778 |

最后100批Full/Smooth生成损失=2.285/4.311，重建MSE=0.00755/0.18668。Smooth原图分割接近Full，但自身生成图的MSE=0.15030，冻结U在该图上的Dice仅0.008%；这一固定4×4瓶颈未让其余生成层级补偿丢失的图像结构。冻结U在合成图上的Dice只是可识别性审计，不证明合成标签真实有效或潜变量已因果解耦。未访问隐藏U标签/test。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。旧路线硬停不变。
