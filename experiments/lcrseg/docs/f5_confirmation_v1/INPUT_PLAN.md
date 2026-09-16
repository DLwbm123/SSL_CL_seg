# F5_CONFIRMATION_V1：下一轮实验计划

本计划是拟议方案；本交付未启动训练、修改仓库或恢复监测。

## 目标

保持 history-free、已训练参数作为记忆和现有输入侧约束；冻结 F5_C02，验证新 seed 增益、简单半监督基线比较以及第二顺序的旧域保持。不再开展五框架搜索。

## P0：已有结果审计

只读已有规范化汇总/receipt，导出阶段×域×类别指标和时间线。开发seed161、已观察seed162、新目标复验seed163/164分开标记。现有 source 先审计元数据，真实执行预检再加载核验；不为补表重训或打开患者图像。优先定位O2中REFUGE与Drishti_GS各自的变化。

## P1：唯一正式矩阵

| 方法 | 固定配置 | 新seed | 顺序 | 新目标阶段 | 正式更新 |
|---|---|---|---|---:|---:|
| B0_PARENT_LCTX | P1 | 163、164 | O1、O2 | 8 | 21200 |
| F5 | F5_C02 | 163、164 | O1、O2 | 8 | 21200 |
| B2_PARENT_PAS_KL | C06 | 162、163、164 | O1、O2 | 12 | 31800 |
| 合计 | 不重新选择 | — | — | 28 | 74200 |

O1：REFUGE→RIM_ONE_r3→Drishti_GS。
O2：REFUGE→Drishti_GS→RIM_ONE_r3。
每条目标轨迹3200+2100=5300更新。复用SOURCE_S162/163/164；source新训练0。B0/F5 seed162结果只导入。

真实smoke上限24更新，与正式74200分开；真实数据物理optimizer总上限74224。合成CUDA资格测试上限60，CPU合成测试为冻结的有限集合，均单独记账；74224不是包含所有合成测试的全口径总调用数。

## 配置核对

三方法A/B初始学习率0.0005，父Adam/调度/数据划分/几何/随机流保持原执行语义。
B2_C06：lambda_U=1.0，PAS_confidence=0.6，PAS_cosine=0.7。
F5_C02：lambda_U=0.25，lambda_SWD=0.2，rank_ratio=0.25，PAS_confidence=0.7，PAS_cosine=0.5；R初始学习率0.001。
F5无标注目标= ramp×0.25×(KL+0.2×SWD)，满ramp后SWD总体系数0.05。
完整配置必须从封存收据与实际代码交叉绑定；不一致时停止而非用本简表覆盖。

## 固定分析

Final为最终三个已见域等权macro Dice，macro=(rim+cup)/2；Old为前两域平均；Incoming为最终进入域。Forget保持既有source/第一目标域各自早期分数至最终分数的平均下降。

主要结果是新seed163/164中每seed先平均两个顺序后的配对差；另列逐顺序数据。162–164全部结果仅作描述性汇总；161为开发结果。不得当作独立患者泛化。

推进门槛仅作预算决策，不代表统计显著性：
G1：新seed两个F5-B0平均差均>0，跨新seed平均>=0.005。
G2：每个顺序跨新seed平均Delta Old>=-0.005、Delta Forget<=0.005。
G3：新seed双顺序平均F5-B2的Delta Final>0，同时解释Old/Incoming/Forget取舍。
全通过只建议进入P2审阅；否则停止分析具体未支持项。任何情形都不自动追加实验。

## P2：后续条件性计划，当前不执行

若P1支持继续，优先考虑F5_L_ONLY、F5_NO_SWD、F5_RANDOM_Q：各2个seed×2顺序，合计24目标阶段、63600更新。分别回答结构容量、SWD独立贡献和参数导出子空间贡献。详细归约、权限和随机基控制应在下一份协议中固定，不能把本段当作执行授权。

梯度权限联合更新与合并前景分布匹配留作更后面的机制实验；本轮不实现/启动。

## 使用顺序

先转发02_PROMPT_PREPARE.md，完成准备与代码提交后停在外部审阅点。只有真实审阅与本轮代码/计划绑定通过后，才转发03_PROMPT_RUN.md。不要把两段一次性交给执行代理并要求自动连续通过。

## 原始证据定位

仓库：DLwbm123/SSL_CL_seg。
结果锚点：96ba0e29f601589a069bcbfdf50b2f4aafc1d883。
训练实现：cc21871a43e3cd105cddaf831f5c3ea2fef59b9e。
原报告目录：experiments/lcrseg/docs/five_frameworks_v1/native_execution/。
候选目录：experiments/lcrseg/docs/five_frameworks_v1/delivery/manifests/。
