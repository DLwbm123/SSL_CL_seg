# C1 判别均值路径 × 层次生成反馈

**冻结结论：SOURCE_FAILED。** Cut/Joint的主要预测为标准双logit softmax，MC4是独立次要读出；所有患者指标为4名开发患者等权三维结果，百分数。

| 臂 | 源RUNMC | 源BMC | 目标RUNMC | 目标BMC |
|---|---:|---:|---:|---:|
| U_Std | 63.261 | 30.997 | 未执行 | 未执行 |
| Cut | 63.261 | 30.997 | 未执行 | 未执行 |
| Joint | 62.113 | 27.361 | 未执行 | 未执行 |

## 源域硬停门

| 门 | 判定 |
|---|---|
| Cut_exact_v45 | PASS |
| Joint_minus_Cut_mean_ge_minus_1pp | FAILED |
| Joint_each_minus_Cut_ge_minus_5pp | PASS |
| Joint_absolute_vs_U_mean_ge_minus_1pp | FAILED |
| Joint_each_absolute_vs_U_ge_minus_5pp | PASS |
| Joint_HD95_within_10pct_of_Cut | PASS |
| no_empty_collapse | PASS |

Cut与封存v45 U1600主分割参数最大绝对差 0；生产零更新梯度准入PASS。[C0只读诊断](C0_DIAGNOSTIC_ZH.md)没有用于选择配置。

源RUNMC Joint−Cut -1.148pp，低于预定−1pp界限 0.148pp；Cut与封存U-Std逐项相同，故表中相对Cut和相对旧U的两条均值失败是同一个事实，不能计作两个独立失败。四名患者差值均≥−5pp，最差-4.656pp；未适应BMC Joint−Cut -3.636pp。源完整训练患者Dice Cut82.632%、Joint75.991%，差距已出现在训练患者，不能只归咎于开发集泛化。固定MC4的主次读出接近，没有替代失败的均值门。

源门未过，协议规定的两条目标链均未启动。

源RUNMC匿名患者差：

| 索引 | Joint−Cut Dice pp | Joint−旧U Dice pp |
|---|---:|---:|
| 1 | -2.338 | -2.338 |
| 2 | +3.700 | +3.700 |
| 3 | -1.299 | -1.299 |
| 4 | -4.656 | -4.656 |

次要MC4后验预测Dice（不得替代上述主要门）：

| 臂 | 位置 | MC4 Dice % |
|---|---|---:|
| Cut | source_RUNMC | 63.274 |
| Cut | source_BMC | 31.006 |
| Joint | source_RUNMC | 62.108 |
| Joint | source_BMC | 27.361 |

[匿名分析](ANALYSIS.json)含逐患者TP/FP/FN/HD95、precision/recall、NLL/Brier及全训练患者读出；[执行摘要](EXECUTION_SUMMARY.json)列原时钟、正式更新、字节和全部阶段。Joint−Cut只识别整个生成反馈策略在当前训练种子与重复使用的四名开发患者上的效果，不能拆成单个潜变量的贡献；通过非劣门也不自动证明层次机制有益。隐藏U标签和test未访问，私有预测与模型未发布。[协议](PREREG_ZH.md)定义停止与后继边界。

补充的[源终点梯度审计](GRADIENT_AUDIT.json)在固定最后一批上0更新执行：Cut层次项到U-Net梯度仍为零，Joint非零；Joint监督梯度范数1.390、未乘系数的变分梯度范数0.848、余弦−0.035。单批接近正交不证明全程无冲突。训练日志逐步保留task/变分项/尺度，但运行时未保留预注册中间步数的梯度审计快照，故无法补做1/400/800/1200的原状态梯度分析；此解释性诊断缺口不改变预先计算的源门与停止动作，也不以重跑填补。
