# M0：源域形状/外观交换零更新审计

**冻结判定：CUT_ADMISSION_FAILED。** 用L1 Cut和Joint封存终点，对RUNMC四名L训练患者80片做确定性自身重建与跨患者外观交换；伙伴是循环下一名患者，按轴向相对位置配片。没有优化器更新，也未读取BMC数据。用冻结标准U预测原图、重建图及交换图，Dice为四患者等权三维结果（%）。

| L1模型 | 原图U Dice | 自身重建U Dice | 外观交换U Dice | 重建MSE | 交换图越界像素% |
|---|---:|---:|---:|---:|---:|
| Cut | 82.632 | 82.801 | 61.986 | 0.00038 | 0.025 |
| Joint | 82.632 | 82.627 | 62.028 | 0.00039 | 0.030 |

预注册合成准入门（仅Cut全通过才考虑另立合成训练协议）：

| 模型 | 门 | 判定 |
|---|---|---|
| Cut | self_reconstruction_mse_le_0_02 | PASS |
| Cut | swap_outside_fraction_le_0_01 | PASS |
| Cut | self_mean_Dice_drop_le_5pp | PASS |
| Cut | self_each_Dice_drop_le_10pp | PASS |
| Cut | swap_mean_Dice_drop_le_5pp | FAILED |
| Cut | swap_each_Dice_drop_le_10pp | FAILED |
| Cut | swap_no_patient_empty | PASS |
| Joint | self_reconstruction_mse_le_0_02 | PASS |
| Joint | swap_outside_fraction_le_0_01 | PASS |
| Joint | self_mean_Dice_drop_le_5pp | PASS |
| Joint | self_each_Dice_drop_le_10pp | PASS |
| Joint | swap_mean_Dice_drop_le_5pp | FAILED |
| Joint | swap_each_Dice_drop_le_10pp | FAILED |
| Joint | swap_no_patient_empty | PASS |

匿名逐患者Dice：

| L1模型 | 索引 | 原图 | 自身重建 | 外观交换 |
|---|---:|---:|---:|---:|
| Cut | 1 | 90.520 | 90.409 | 60.962 |
| Cut | 2 | 72.654 | 73.221 | 61.984 |
| Cut | 3 | 84.110 | 83.715 | 59.153 |
| Cut | 4 | 83.242 | 83.858 | 65.846 |
| Joint | 1 | 90.520 | 90.321 | 61.166 |
| Joint | 2 | 72.654 | 72.899 | 61.954 |
| Joint | 3 | 84.110 | 83.503 | 59.140 |
| Joint | 4 | 83.242 | 83.786 | 65.854 |

Cut自身重建相对原图U为+0.169pp，交换后为-20.646pp。重建误差和越界比例都很低，但交换后4名患者的Dice均大幅下降；这一组合与外观场携带空间结构或模型不能组合跨患者潜变量均相容，不能仅凭此审计确定原因。交换后的分割读数仅检验冻结U是否仍能识别源标签；它不是合成图真实标签正确性的直接证明，更不说明外观潜变量已与形状解耦。Cut失败，本轮关闭直接外观交换合成路线，不对伙伴、裁剪、权重或checkpoint作结果后调整。身份、图像、标签、合成图和原始预测留NAS。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。
