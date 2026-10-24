# B1 图像条件后验与同结构目标对照

**H-IS 新基座联合继续门：FAILED（4/8通过）。** 两新增臂各完成源RUNMC1600＋目标BMC800更新。旧U-Std和H-Std终点只读复用；四格均为4名开发患者等权三维Dice（%）。

| 臂 | 源RUNMC | 源BMC | 目标RUNMC | 目标BMC |
|---|---:|---:|---:|---:|
| U_Std | 63.261 | 30.997 | 69.334 | 62.107 |
| H_Std | 58.565 | 37.275 | 64.249 | 60.407 |
| H_IS | 56.573 | 34.446 | 61.398 | 64.416 |
| H_IS_NoVar | 61.198 | 42.503 | 50.096 | 40.960 |

| 冻结继续门 | 判定 |
|---|---|
| source_RUNMC_within_1pp_of_U | FAILED |
| target_BMC_within_2pp_of_U | PASS |
| target_RUNMC_within_2pp_of_U | FAILED |
| old_mean_change_ge_minus_0_5pp | PASS |
| old_each_change_ge_minus_5pp | FAILED |
| BMC_at_least_3_of_4_within_5pp_of_U | PASS |
| BMC_each_within_10pp_of_U | FAILED |
| no_empty_foreground_collapse | PASS |

| 位置 | H-IS−H-Std pp | H-IS−NoVar pp | H-IS−U-Std pp |
|---|---:|---:|---:|
| source_RUNMC | -1.992 | -4.625 | -6.688 |
| source_BMC | -2.829 | -8.057 | +3.448 |
| target_RUNMC | -2.851 | +11.302 | -7.936 |
| target_BMC | +4.009 | +23.455 | +2.309 |

目标BMC匿名患者差：

| 索引 | H-IS−U-Std pp | H-IS−H-Std pp | H-IS−NoVar pp |
|---|---:|---:|---:|
| 1 | +21.287 | +6.427 | +13.777 |
| 2 | +1.652 | +7.554 | +16.377 |
| 3 | -11.809 | -7.303 | +19.284 |
| 4 | -1.895 | +9.358 | +44.384 |

## 判读

H-IS的目标BMC均值64.416%，比U-Std高2.309pp、比H-Std高4.009pp；但源RUNMC比U-Std低6.688pp，适应后目标RUNMC低7.936pp。目标BMC 3/4名患者达到相对U-Std的−5pp界限，剩下一名低11.809pp；旧域源→目标虽均值提高4.825pp，一名患者下降15.690pp。因此八项联合门只通过4项，不能以BMC均值胜出称新基座合格。

同结构NoVar在源RUNMC比H-IS高4.625pp，但目标BMC比H-IS低23.455pp，目标RUNMC低11.302pp；目标BMC全训练患者Dice仅37.657%，空真值切片每片FP 3260.91，对应H-IS的318.13。这是本固定训练种子中去掉层次目标后明显过分割的证据，说明不能简单把变分项视为纯性能负担；两臂优化轨迹不同，单种子与反复使用的开发患者不支持普遍因果或论文级贡献结论。H-IS本身的源域与旧域绝对能力、最差患者仍是当前阻塞。

H-IS目标BMC全训练患者Dice 80.350%；空真值切片每片FP 318.13，非空真值切片每片FN 286.95。[匿名完整分析](ANALYSIS.json)含患者TP/FP/FN、HD95及四格NLL/Brier；[执行摘要](EXECUTION_SUMMARY.json)含时间、字节与每阶段状态。

初始权重及函数与H-Std匹配，新图像通道初始为零；NoVar拥有相同网络、初始概率与采样，只去掉变分目标。任何性能差都仅为本种子/当前四名开发患者的诊断，不足以证明层次目标在U或CL中的作用。固定患者反复用于开发；MC4不是独立训练种子，隐藏U标签和test未访问。[冻结协议](PREREG_ZH.md)定义全部门槛。

H-IS未过门，本批关闭；不追加风险权重、旁路、阈值、长度或中途模型搜索。
