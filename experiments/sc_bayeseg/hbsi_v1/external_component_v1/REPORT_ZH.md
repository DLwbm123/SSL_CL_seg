# V1：封存R2与固定三维空间规则的额外前列腺站点评价

**冻结判定：EXTERNAL_SPATIAL_FAILED。** BIDMC/HK/I2CVB/UCL全部既有val患者2/2/3/2名均纳入，未用于此前HBSI两站点模型选择；两只封存R2第800步模型0更新。每臂原U双logit概率与固定三维6邻接最大预测前景组件变换各评一次；后者不使用标签，但此规则曾根据BMC开发诊断提出，因此本次即使阳性也不是未受选择影响的确认性试验。R2原 `REHEARSAL_FEEDBACK_FAILED` 不变。

| 站点 | 患者 | Cut原Dice% | Joint原Dice% | Cut组件Dice% | Joint组件Dice% | 组件Joint−Cut pp | Cut组件HD95 mm | Joint组件HD95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BIDMC | 2 | 40.762 | 58.192 | 46.688 | 63.913 | +17.225 | 18.772 | 18.171 |
| HK | 2 | 69.400 | 71.123 | 72.117 | 75.591 | +3.474 | 7.860 | 10.848 |
| I2CVB | 3 | 31.969 | 10.392 | 29.132 | 0.000 | -29.132 | 35.926 | 76.862 |
| UCL | 2 | 28.117 | 44.739 | 35.998 | 42.677 | +6.679 | 29.061 | 25.117 |

四站点等权：原预测Cut/Joint Dice=42.562/46.112%；固定组件Cut/Joint Dice=45.984/45.545%（Joint−Cut -0.438pp）。Joint原→组件HD95=48.038→32.749mm；组件Cut/Joint HD95=22.905/32.749mm。原概率四站点等权NLL Cut/Joint=0.08741/0.08144，Brier=0.02950/0.02718；硬组件输出不拿来比较软概率质量。

I2CVB是明确失败站点：ReplayJoint原预测三患者Dice已仅 10.392%，组件规则又将三人的全部原TP分别移除 7893, 99, 4038 个，保留的最大预测组件与真值均无重叠，三人Dice变为0%。这不是整患者空预测，因此 `no_new_empty_patient` 仍显示PASS；它不能覆盖零重叠风险。不能据其他三站点的正值忽略该负例。

| 类别 | 冻结门 | 判定 |
|---|---|---|
| 空间安全 | site_equal_Dice_vs_original_ge_minus_1pp | PASS |
| 空间安全 | each_patient_Dice_vs_original_ge_minus_5pp | FAILED |
| 空间安全 | site_equal_HD95_vs_original_le_1_10 | PASS |
| 空间安全 | no_new_empty_patient | PASS |
| 配对积极 | site_equal_Dice_Joint_minus_Cut_ge_plus_2pp | FAILED |
| 配对积极 | at_least_3_of_4_sites_nonnegative | PASS |
| 配对积极 | each_patient_Joint_minus_Cut_ge_minus_5pp | FAILED |
| 配对积极 | site_equal_HD95_Joint_le_Cut_1_10 | FAILED |
| 配对积极 | no_component_patient_empty | PASS |

匿名逐患者Dice（组件末列为Joint−Cut）：

| 站点 | 索引 | Cut原% | Joint原% | Cut组件% | Joint组件% | 差值pp |
|---|---:|---:|---:|---:|---:|---:|
| BIDMC | 1 | 30.377 | 57.983 | 40.385 | 62.825 | +22.440 |
| BIDMC | 2 | 51.148 | 58.401 | 52.991 | 65.001 | +12.010 |
| HK | 1 | 61.643 | 61.600 | 65.658 | 70.428 | +4.770 |
| HK | 2 | 77.156 | 80.646 | 78.576 | 80.755 | +2.179 |
| I2CVB | 1 | 45.316 | 22.162 | 52.467 | 0.000 | -52.467 |
| I2CVB | 2 | 13.745 | 0.179 | 0.000 | 0.000 | +0.000 |
| I2CVB | 3 | 36.847 | 8.836 | 34.930 | 0.000 | -34.930 |
| UCL | 1 | 7.384 | 20.766 | 0.000 | 0.000 | +0.000 |
| UCL | 2 | 48.850 | 68.712 | 71.996 | 85.354 | +13.358 |

组件Joint相对Cut的9患者最差配对差 -52.467pp、获益 7/9；Joint组件绝对最差Dice 0.000%、9人插值p10 0.000%，Cut对应 0.000%/0.000%。Joint组件相对自身原预测最差患者差 -22.162pp。站点效果范围 -29.132至+17.225pp；患者与站点并非可互换的独立重复。

空间安全门 **FAILED**，外部配对积极门 **FAILED**。本固定候选未达到预设外部联合门，不按站点结果删病例、调阈值或改连通定义。单训练seed、先前BMC开发驱动候选、9名患者和四站点异质性限制结论；不读取任何test或U隐藏标签。[协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[成本](EXECUTION_SUMMARY.json)。
