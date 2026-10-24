# C2：BMC 小样本跨中心复核

**冻结判定：NO_ROBUST_POSITIVE_SIGNAL。** BMC 4 名标注患者、134 张切片从同一初值各训练1600步；以下 Dice 为各域4名开发患者等权三维结果，单位%。Cut 主干是本轮从零训练的标准 U-Net。RUNMC 列为未适应迁移，MC4 为次要读出。

| 臂 | BMC 域内 | RUNMC 冷迁移 | BMC 全训练患者 | BMC MC4 |
|---|---:|---:|---:|---:|
| Cut | 64.815 | 16.983 | 81.172 | 64.810 |
| Joint | 60.450 | 34.748 | 77.182 | 60.455 |

Joint−Cut BMC均值 **-4.364pp**；RUNMC冷迁移均值 **+17.765pp**。域内每名患者：

| 索引 | Cut Dice % | Joint Dice % | Joint−Cut pp |
|---|---:|---:|---:|
| 1 | 57.025 | 55.321 | -1.704 |
| 2 | 72.576 | 70.025 | -2.551 |
| 3 | 62.748 | 52.927 | -9.821 |
| 4 | 66.911 | 63.529 | -3.382 |

| 预定积极信号门 | 判定 |
|---|---|
| BMC_mean_gain_ge_1pp | FAILED |
| BMC_at_least_3_of_4_nonnegative | FAILED |
| BMC_each_decline_le_5pp | FAILED |
| BMC_HD95_within_10pct_of_Cut | FAILED |
| no_full_patient_collapse | PASS |

RUNMC冷迁移的均值优势由患者差异共同构成，特别是第4名患者；它不是BMC域内主门的替代结果。RUNMC逐患者：

| 索引 | Cut Dice % | Joint Dice % | Joint−Cut pp |
|---|---:|---:|---:|
| 1 | 15.253 | 12.167 | -3.086 |
| 2 | 33.060 | 44.462 | +11.402 |
| 3 | 14.134 | 25.501 | +11.367 |
| 4 | 5.482 | 56.860 | +51.378 |

RUNMC惩罚HD95均值Cut 55.977mm、Joint 76.469mm；Brier分别为0.0649和0.0672。冷迁移Dice提高伴随这些指标变差，不能称为全面跨域改善。

完整匿名患者计数、NLL/Brier、惩罚HD95及次要MC4见[分析](ANALYSIS.json)，资源收据见[执行摘要](EXECUTION_SUMMARY.json)。C1原RUNMC源门失败和未做目标训练的结论保持不变；本轮只检验另一中心的单域从零训练，BMC开发患者已经被历史路线多次观察，不是外部独立验证。BMC和RUNMC的训练切片数也不同，不能把域内Dice差异单独归因于中心。单种子4名开发患者及层次反馈整项对比，不足以识别某个潜变量的因果贡献或声称普遍优势。隐藏U标签和test未访问；原图、标注、预测和模型均留NAS。[冻结协议](PREREG_ZH.md)
