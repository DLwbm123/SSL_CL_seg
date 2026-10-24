# L1：标签锚定的 BayeSeg 形状场

**冻结判定：NO_BMC_GAIN。** 两臂同一RUNMC L4患者80片，各1600更新；qS本身是U初值锚定的分割logit场，qA独立，完整图像生成与层级先验保留。Cut截断生成梯度到qS，Joint允许反馈。Dice为4患者等权三维结果（%）；BMC4开发患者已反复用于路线选择，以下只是探索性资源门。

| 臂 | RUNMC开发Dice | BMC开发Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
| U | 63.261 | 30.997 | 82.632 | 27.910 | 43.788 |
| H1_Shape | 63.718 | 31.072 | 83.438 | 26.524 | 40.986 |
| Cut | 63.541 | 30.488 | 83.749 | 26.519 | 40.638 |
| Joint | 63.042 | 30.033 | 83.868 | 26.227 | 40.812 |

Joint−Cut RUNMC/BMC为-0.499/-0.455pp；Joint−旧U为-0.219/-0.964pp。U与H1 Shape为不同累积训练成本的历史参考，不是同轮匹配对照。

源安全门：

| 门 | 判定 |
|---|---|
| Joint_RUNMC_mean_vs_Cut_ge_minus_1pp | PASS |
| Joint_RUNMC_each_vs_Cut_ge_minus_5pp | PASS |
| Joint_RUNMC_train_mean_vs_Cut_ge_minus_5pp | PASS |
| Joint_RUNMC_mean_vs_U_ge_minus_1pp | PASS |
| Joint_RUNMC_each_vs_U_ge_minus_5pp | PASS |
| Joint_RUNMC_train_mean_vs_U_ge_minus_5pp | PASS |
| Joint_RUNMC_HD95_vs_both_le_1_10 | PASS |
| Joint_no_source_collapse | PASS |
| Cut_vs_U_mean_ge_minus_1pp | PASS |
| Cut_vs_U_each_ge_minus_5pp | PASS |
| Cut_no_source_collapse | PASS |
| finite_training_and_scale | PASS |

BMC积极门（源门全过才允许后继）：

| 门 | 判定 |
|---|---|
| Joint_BMC_mean_vs_max_Cut_U_H1_ge_plus_2pp | FAILED |
| Joint_BMC_3_of_4_nonnegative_vs_Cut_and_U | FAILED |
| Joint_BMC_each_vs_Cut_and_U_ge_minus_5pp | PASS |
| Joint_BMC_HD95_vs_Cut_and_U_le_1_10 | PASS |
| Joint_no_BMC_collapse | PASS |

匿名逐患者：

| 域 | 索引 | U Dice | Cut Dice | Joint Dice | Joint−Cut pp | Joint−U pp |
|---|---:|---:|---:|---:|---:|---:|
| RUNMC | 1 | 68.629 | 68.934 | 68.481 | -0.453 | -0.148 |
| RUNMC | 2 | 71.836 | 71.720 | 71.175 | -0.546 | -0.662 |
| RUNMC | 3 | 66.608 | 66.543 | 65.443 | -1.100 | -1.165 |
| RUNMC | 4 | 45.970 | 46.966 | 47.071 | +0.104 | +1.100 |
| BMC | 1 | 12.997 | 12.137 | 11.627 | -0.510 | -1.370 |
| BMC | 2 | 27.264 | 25.777 | 24.760 | -1.017 | -2.504 |
| BMC | 3 | 52.472 | 52.432 | 52.697 | +0.265 | +0.225 |
| BMC | 4 | 31.256 | 31.607 | 31.048 | -0.559 | -0.209 |

预注册第1/400/800/1200/1600步更新前qS梯度范数（标签 / 实际0.01系数后生成；余弦见匿名分析）：

| 步号 | Cut | Joint |
|---|---:|---:|
| 1 | 0.24252 / 0.00000 | 0.24252 / 0.09536 |
| 400 | 0.12591 / 0.00000 | 0.38021 / 0.02055 |
| 800 | 0.12399 / 0.00000 | 0.09909 / 0.08832 |
| 1200 | 0.57767 / 0.00000 | 0.48590 / 0.08818 |
| 1600 | 0.16558 / 0.00000 | 0.31138 / 0.01867 |

最后100批Cut/Joint生成损失=5.354/2.330，重建MSE=0.00755/0.00755，qS平均log标准差=-4.000/-1.708，形状均值空间SD=3.315/1.931。Joint生成目标下降与形状尺度扩张同时发生，不能从这些观测确定哪一项造成BMC下降；本轮继承的`p(S|Z)`此前拟合的是灰度形状场，语义logit场尺度不匹配也限制解释。这不证明形状/外观解耦，也不构成独立患者证据；隐藏U标签和test未访问。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。旧路线判定不变。
