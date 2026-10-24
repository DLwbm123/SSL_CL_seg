# Q0：真值形状下的类条件外观审计

**冻结判定：ORACLE_ADMISSION_FAILED。** RUNMC 四名 L 训练患者 80 片，0 次正式更新；只用源训练标签固定解剖形状，不读取 BMC 或 test。`C` 为真值掩膜加本患者两类全局灰度均值；`D` 在原图空间残差上替换成下一名患者的两类均值。原图、C、D 均由相同冻结 U-Net 评分，Dice 为四患者等权三维结果（%）。

| 原图 U Dice | C 的 U Dice | D 的 U Dice | C 重建 MSE | D 越界像素% |
|---:|---:|---:|---:|---:|
| 82.632 | 16.200 | 78.007 | 0.188365 | 4.5202 |

| 预注册门 | 判定 |
|---|---|
| class_only_mse_le_0_02 | FAILED |
| class_only_mean_Dice_drop_le_5pp | FAILED |
| class_only_each_Dice_drop_le_10pp | FAILED |
| class_only_no_patient_empty | FAILED |
| delta_outside_fraction_le_0_01 | FAILED |
| delta_mean_Dice_drop_le_5pp | PASS |
| delta_each_Dice_drop_le_10pp | FAILED |
| delta_no_patient_empty | PASS |

| 匿名源训练患者 | 原图 Dice | C Dice | D Dice |
|---:|---:|---:|---:|
| 1 | 90.520 | 55.166 | 87.887 |
| 2 | 72.654 | 9.495 | 73.383 |
| 3 | 84.110 | 0.000 | 83.805 |
| 4 | 83.242 | 0.141 | 66.952 |

仅类均值发射门 **FAILED**，保留原图残差的类均值交换门 **FAILED**。C 相对原图 Dice -66.431 pp；D 为 -4.625 pp。C 的大幅重建与分割损失说明两类全局灰度均值不足以表示这组 MRI；D 虽在均值上接近 5 pp 限制，最差个体仍下降 16.290 pp，且越界像素超过预设线。两门须同时通过才考虑另立对应生成模型训练协议；本轮不训练，也不调整平滑、混合比例、伙伴或阈值。D 保留的是**原患者**空间残差，因此即使 D 通过，也不证明完整外观潜变量能跨患者交换；真值标签提供了学得 qS 所没有的理想解剖条件。源训练患者分数不是独立验证或目标域证据。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。
