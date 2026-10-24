# 09:10阶段路线汇总：已封存结果与下一步证据缺口

这是北京时间2026-10-24 09:10阶段汇总，不是12:13最终窗口报告。只读取公开封存匿名聚合，不重新推断、训练或读取标签。

## 主要发现

v24–v40共17个已完成实验根、32个新增训练终点、30项预注册比较，科学门槛通过0项。全部比较的源域≥1pp项失败；v26 H−A还失败于迁移保护。比较共享入口、患者和对照，不能把30项视为30次独立重复或据此进行显著性推断。

v25–v40从同一v23 L1600入口产生的30个新2000终点，BMC Dice全部低于该入口44.07538%。这是已观察到的继续训练迁移损失模式，不证明唯一机制根因。原监督A额外400步本身源+1.00156pp、迁移−.99230pp；因此相对A的U改善仍可能只补回一部分继续训练损失。不能借此修改固定终点、挑旧checkpoint、放宽门槛或推导所有U机制无效。

| 版本 | 冻结比较 | 源Dice差 pp | 迁移Dice差 pp | 总判定 |
|---|---|---:|---:|---|
| [v24](../appearance_transport_v24/REPORT_ZH.md) | LU_minus_L | -0.45911 | +0.20733 | FAILED |
| [v25](../posterior_teacher_v25/REPORT_ZH.md) | B_minus_A | -0.00423 | -0.11756 | FAILED |
| [v25](../posterior_teacher_v25/REPORT_ZH.md) | C_minus_B | -0.15673 | -0.18167 | FAILED |
| [v25](../posterior_teacher_v25/REPORT_ZH.md) | C_minus_A | -0.16095 | -0.29924 | FAILED |
| [v26](../teacher_region_v26/REPORT_ZH.md) | H_minus_A | -0.36382 | -0.56376 | FAILED |
| [v26](../teacher_region_v26/REPORT_ZH.md) | R_minus_A | -0.13364 | +0.10223 | FAILED |
| [v26](../teacher_region_v26/REPORT_ZH.md) | H_minus_B | -0.35959 | -0.44619 | FAILED |
| [v26](../teacher_region_v26/REPORT_ZH.md) | R_minus_H | +0.23017 | +0.66598 | FAILED |
| [v26](../teacher_region_v26/REPORT_ZH.md) | R_minus_B | -0.12942 | +0.21979 | FAILED |
| [v27](../observed_region_v27/REPORT_ZH.md) | LU_minus_L | -0.03402 | -0.00805 | FAILED |
| [v28](../spatial_correspondence_v28/REPORT_ZH.md) | LU_minus_L | +0.31712 | +0.35216 | FAILED |
| [v29](../redundancy_reduction_v29/REPORT_ZH.md) | LU_minus_L | -0.03353 | +0.01317 | FAILED |
| [v30](../label_neighbor_v30/REPORT_ZH.md) | N_minus_A | +0.49376 | +0.15431 | FAILED |
| [v30](../label_neighbor_v30/REPORT_ZH.md) | N_minus_B | +0.49799 | +0.27188 | FAILED |
| [v31](../projected_update_v31/REPORT_ZH.md) | P_minus_A | +0.48636 | +0.08340 | FAILED |
| [v31](../projected_update_v31/REPORT_ZH.md) | P_minus_N | -0.00740 | -0.07092 | FAILED |
| [v32](../image_graph_v32/REPORT_ZH.md) | E_minus_A | +0.42511 | +0.15488 | FAILED |
| [v32](../image_graph_v32/REPORT_ZH.md) | E_minus_N | -0.06865 | +0.00057 | FAILED |
| [v32](../image_graph_v32/REPORT_ZH.md) | E_minus_I | +0.15866 | +0.03691 | FAILED |
| [v32](../image_graph_v32/REPORT_ZH.md) | I_minus_A | +0.26644 | +0.11797 | FAILED |
| [v32](../image_graph_v32/REPORT_ZH.md) | I_minus_N | -0.22732 | -0.03634 | FAILED |
| [v33](../posterior_risk_v33/REPORT_ZH.md) | J_minus_A | +0.53584 | +0.15390 | FAILED |
| [v33](../posterior_risk_v33/REPORT_ZH.md) | J_minus_N | +0.04208 | -0.00041 | FAILED |
| [v34](../masked_evidence_v34/REPORT_ZH.md) | LU_minus_L | +0.00000 | +0.00000 | FAILED |
| [v35](../posterior_affinity_v35/REPORT_ZH.md) | LU_minus_L | +0.06289 | +0.08883 | FAILED |
| [v36](../posterior_smoothness_v36/REPORT_ZH.md) | LU_minus_L | -0.00794 | +0.03806 | FAILED |
| [v37](../class_observation_v37/REPORT_ZH.md) | LU_minus_L | +0.00069 | +0.00097 | FAILED |
| [v38](../semantic_mass_v38/REPORT_ZH.md) | LU_minus_L | +0.04983 | -0.07502 | FAILED |
| [v39](../spatial_subspace_v39/REPORT_ZH.md) | LU_minus_L | -0.02931 | -0.02918 | FAILED |
| [v40](../conditional_feature_v40/REPORT_ZH.md) | LU_minus_L | -0.12685 | +0.05841 | FAILED |

## 机制层面的解释

固定教师均值、分歧加权和硬化没有建立U净收益。R−H的迁移补偿不能替代R−A；概率误差改善也不能替代分割门槛。v30引入L真实类别的近邻信息后相对A有所改善，但不足1pp且仅单seed；它支持继续关注语义信息来源，不证明近邻目标更正确。

v31投影相对N、v32图条件相对N、v33风险积分位置相对N都没有建立有意义的增量，不能将这几项拼接为已成功路线。它们分别改变了合并向量、目标结构或分歧惩罚，三点梯度与残差也没有隔离唯一训练根因。

v28源和迁移患者方向一致，但源增量仍只有+.31712pp；换成v29通道去相关没有保留该增量。不同目标与量级同时变化，无法独立认定假负对是原因。v34–v40的遮挡证据、亲和、平滑、强度观测、面积分布、形状子空间和特征条件密度均未通过；实现可微、解析积分正确、风险降低或参考构造成功均不等于U语义改善。

已完成证据仍缺少“当前L真类别通过在线可微关联进入U表征”的有效终点评估。v41原计划检验这一点，但其Bool argmax预检错误发生在模型前向前，因此这个科学问题仍未得到检验。此缺口不授权违背v41不自动retry，也不支持仅修复后换名。当前没有可报告为已验证的后继方案；本次不登记或启动v42，不为保持队列而任意更换辅助损失。持续主线授权与唯一监测保持，后继设计仍须有独立机制依据，12:13按时累计汇总。

## 成本与完整性

v24–v40完整运行合计14,400次正式更新；各根封存占用合计5,927,066,964字节；各次完整运行历时之和15349.10秒。此为全run成本相加，包含窗口开始前已发生的v24/v25成本，且存在并行重叠，不能当12小时窗口增量、GPU活跃时长或墙钟时间。

v41另有3.545482秒、1,009,632字节、1次小fixture autograd及小张量前向，0模型前向/反向/Adam/评分；无科学门槛判定。v27初次失败及修复已包含原根成本，失败模型前向次数未知，不能写成0。v35终点有效但三点审计文件缺失，不能用未保存遥测解释梯度机制。

正式更新不等总计算：教师/近邻缓存、图求解、标签拟合、方向autograd、surrogate backward与额外审计分别见各报告及执行摘要。本汇总没有新增这些计算，也不以更新数对不同机制作等计算排名。

仅1训练seed、每域4val患者，3推断不是3训练seed。多次开发使用同一val和同一入口，不是独立泛化、可靠不确定性、精确后验或CL证据。完整匿名数值与来源：[INTERIM_0910.json](INTERIM_0910.json)。
