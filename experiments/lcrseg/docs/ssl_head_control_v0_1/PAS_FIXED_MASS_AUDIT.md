# PAS 等数量审计

P0 的4个旧最终模型对已全部完成：65例×2臂×5次预测×2模型=1300次 sample-model 前向，0更新。实际原型来自各旧模型 epoch96 刷新，未重拟合原型。旧缓存缺少所需概率/特征字段，因此按授权重做固定前向；没有替换epoch或seed。

下表为教师 argmax 主口径的前景 precision。4次随机排列先平均，再平均同一患者的4次训练式预测，最后按患者平均；clean 单列。K为每患者/预测draw的平均几何接受数。LR为同一confidence集合C内、按教师正确性计算的汇总筛选比，不是假设条件独立后的保证。

| 域 | 旧臂 | 预测模式 | 类别 | PAS | 同数量 confidence top-k | 同数量随机 | 平均K | 有效患者数 | pooled LR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RIM_ONE_r3 | MT_PAS_G0 | posterior_mean_clean | rim | 0.741240 | 0.742238 | 0.740974 | 10756.450000 | 40 | 1.015401 |
| RIM_ONE_r3 | MT_PAS_G0 | posterior_mean_clean | cup | 0.812080 | 0.812242 | 0.812101 | 3089.800000 | 40 | 0.999862 |
| RIM_ONE_r3 | MT_PAS_G0 | training_like | rim | 0.744108 | 0.745673 | 0.743771 | 10538.743750 | 40 | 1.023622 |
| RIM_ONE_r3 | MT_PAS_G0 | training_like | cup | 0.822357 | 0.822571 | 0.822424 | 2935.418750 | 40 | 0.999430 |
| RIM_ONE_r3 | MT_PAS_G1 | posterior_mean_clean | rim | 0.753577 | 0.753584 | 0.753580 | 9923.000000 | 40 | 1.000043 |
| RIM_ONE_r3 | MT_PAS_G1 | posterior_mean_clean | cup | 0.779283 | 0.779283 | 0.779283 | 3399.075000 | 40 | 1.000000 |
| RIM_ONE_r3 | MT_PAS_G1 | training_like | rim | 0.751583 | 0.755891 | 0.757781 | 6753.375000 | 40 | 1.001896 |
| RIM_ONE_r3 | MT_PAS_G1 | training_like | cup | 0.774874 | 0.774814 | 0.774824 | 2049.006250 | 40 | 0.996381 |
| Drishti_GS | MT_PAS_G1 | posterior_mean_clean | rim | 0.786494 | 0.810665 | 0.775363 | 8024.960000 | 25 | 1.045951 |
| Drishti_GS | MT_PAS_G1 | posterior_mean_clean | cup | 0.720174 | 0.720075 | 0.720030 | 12087.640000 | 25 | 1.000586 |
| Drishti_GS | MT_PAS_G1 | training_like | rim | 0.794709 | 0.807888 | 0.770301 | 7395.850000 | 25 | 1.130949 |
| Drishti_GS | MT_PAS_G1 | training_like | cup | 0.724013 | 0.723718 | 0.723483 | 11850.790000 | 25 | 1.001899 |
| Drishti_GS | MT_PAS_G0 | posterior_mean_clean | rim | 0.710964 | 0.711127 | 0.710980 | 10640.960000 | 25 | 0.999543 |
| Drishti_GS | MT_PAS_G0 | posterior_mean_clean | cup | 0.708665 | 0.708665 | 0.708665 | 11984.400000 | 25 | 1.000000 |
| Drishti_GS | MT_PAS_G0 | training_like | rim | 0.715555 | 0.715837 | 0.715617 | 10296.580000 | 25 | 0.999090 |
| Drishti_GS | MT_PAS_G0 | training_like | cup | 0.711898 | 0.711898 | 0.711898 | 11763.030000 | 25 | 1.000000 |

全432个公开聚合行覆盖旧P0和新LIN_MT_PAS固定快照，包括background/rim/cup、teacher/student两种标签来源和两种预测模式。每组有效计分数量相等比例均为1。GT255没有参与K或排序；当前观察到计分支持相等不改变这一边界。新旧总计1950组预测掩码密封记录已核验为先密封后读GT。GT、患者标识及逐病例像素记录不公开。

旧G0以及多数cup对照几乎相同。Drishti G1 rim 的PAS相对随机有正差：clean约+0.0111、训练式约+0.0244；但均低于相同K的confidence top-k，分别约−0.0242和−0.0132。RIM G1 rim训练式PAS还低于随机对照约0.0062。因此不能把减少数量或改变类别构成后的precision变化直接解释为PAS提供了稳定、额外的正确性信息。

这是描述性结果，没有统计显著性、逐病例安全性或部署风险保证；P0未用于候选选择。PAS在少数cup行相对top-k有很小正差，同样完整保留，不能扩大为统一优越性结论。

完整 PAS_FIXED_MASS_AUDIT.csv 还保留预测覆盖率、图像覆盖率、accepted-correct recall、正确/错误计数、接受分歧比例、各指标有效患者数、LR无穷/undefined计数，以及student标签来源的对照。空分母不填1；LR不加伪计数。汇总像素计数是重复draw-pixels，不作为额外患者。新LIN_MT_PAS在epoch40/60/80/100的同数量审计同表提供，epoch20没有实际训练原型库。
