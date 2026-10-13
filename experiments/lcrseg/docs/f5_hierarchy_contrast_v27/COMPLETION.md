# V27 完成报告

状态：COMPLETE_DEVELOPMENT_ONLY；相对 F5 / B2 双 gate FAIL。V27 在阶段分析覆盖的四轨迹方法中平均 Final 最高，但尚不满足冻结开发门槛。用户已要求暂停自动新增层级变体，B1 暂不补跑。

两 seeds × 两 orders 共四 jobs、八 target stages 全部完成，正式 optimizer updates=21,200；四 jobs 和 controller/process exit 均为 0，COMPLETION_AUDIT=PASS，八个 target 均先封存后 readout。无重新训练或新增评估。执行源代码为 `0c35ab0c30632d77541209ec4290ab81a8e90d8f`。

| seed/order | Final | Old | Incoming | Forget↓ |
|---|---:|---:|---:|---:|
| 163/O1 | 70.1618 | 69.1901 | 72.1053 | 6.7905 |
| 163/O2 | 66.8018 | 65.3375 | 69.7303 | 12.0919 |
| 164/O1 | 69.2910 | 69.6070 | 68.6590 | 6.5918 |
| 164/O2 | 69.5082 | 69.2373 | 70.0500 | 9.0571 |
| 平均 | 68.9407 | 68.3430 | 70.1362 | 8.6328 |

| 参照 | ΔFinal | ΔOld | ΔIncoming | ΔForget |
|---|---:|---:|---:|---:|
| F5 | +3.5298 | +5.1741 | +0.2411 | -5.0428 |
| B2 | +1.0396 | +2.1019 | -1.0851 | -2.0678 |
| V13 | +1.1149 | +2.0158 | -0.6867 | -1.7903 |
| V21 | +0.6612 | +1.4287 | -0.8738 | -1.3768 |
| V23 | +0.3293 | +0.8119 | -0.6361 | -0.6578 |
| V26 | +1.5756 | +2.9719 | -1.2170 | -2.9921 |

全部为百分数/百分点。F5 gate 的失败项为 O1 Incoming −0.5376 pp；B2 gate 的失败项包括 seed 164 Final −0.6560 pp、O1/O2 Incoming −0.7255/−1.4446 pp。B2 24 个类别单元均未突破 −5 pp 底线，最差为 164/O1/RIM rim −3.8307 pp。完整正负单元与比较判定保留于 [RESULTS](completion/RESULTS.json)，不以均值替代 gate。

阶段分解表明主要收益来自保持，且 disc_union 与 rim/cup 存在明显脱钩，详见 [阶段分析](../stage_retention_analysis_20261013/ANALYSIS.md)。两 seed 开发实验没有独立患者确认。

## 激活与实际成本

hierarchy/coarse/parts 各 21,200 calls；alignment calls=0、aligned edges=0，与 contrast-only 合约一致。原 U 路 A/B/R 各 16,960 非零调用，entry-reference VJP=16,960，投影调用=9,009。Student/teacher full forwards=89,040/50,888，anchor full forwards=16,960，额外 clean-U forwards=16,960；这些是已有计数器的不同维度，不应全部相加当作独立图像数。Readout=1,190 sample/feature/native-readout calls。完整 counters 见 [ACTIVATION_COSTS](completion/ACTIVATION_COSTS.json)。

Controller wall time=3,647.6954 s；四个 job 用时约 1,758.42–1,783.77 s（存在并发，不相加冒充 wall time）。各 job 峰值 CUDA allocated=2,242,036,224 bytes，reserved=2,336,227,328 bytes。没有单独测量 GPU residence、图像去重曝光量或本轮 checkpoint 字节数；没有据此提出速度或内存优势。

本轮 qualification 实际为 32 CPU / 10 generated-CUDA / 2 real-L optimizer updates，独立于正式 21,200。系列 qualification 累计为 1,248/327/66，含 B1 为 1,280/337/68。正式方法累计 594,273，B1 另 5,300，合计 599,573；历史 rollback 673 已计入，5 个历史 NA 不补造。完成整理新增 optimizer/model calls=0。

## 交付与停止边界

公开源代码、预注册、qualification、启动记录及本目录完成聚合证据；私有权重、样本行、图像、标签、原始运行日志和运行配置留在 NAS。原始 RESULTS 的 automatic_followup 是产生结果时的历史字段，不构成当前授权；最新用户指令优先：停止自动新增变体、不补 B1、等待分析后决定。
