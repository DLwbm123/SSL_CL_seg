# 04｜设计修订、来源与创新边界

## 1. 请求基础与证据层级

本包基于用户上传 `SSL_CL_seg_loss_review_2023-2026.md`、当前对话中已讨论的五框架，以及下列固定commit中的代码。上传综述的原文保留在sources，不将其中建议冒充已验证结果。

本轮实际核对了固定commit `8a6e93a8bc39554c878cffe0dedf7fbfefe7d16d` 的：
- `experiments/lcrseg/lctx_weight_memory_v0_1/core.py`：确认已有BA低秩/输入右投影与dense merge实现；**不是本轮原KI绑定证据**。
- `experiments/lcrseg/cist_v0_1/core.py`：确认现有通道读出代理、实际classifier前特征路径、3×3读出及native resize。

该commit是本包可核验的参考快照，不宣称为当前GitHub最新分支。不把旧代码性能/冻结集合自动代入新父实现。

## 2. 从概念草图到V1的显式修订

### D1：不再把任意层B直接与classifier前Q相乘

早期F1的ΔW=(B+QR)A需要B输出与Q在同一线性坐标，当前代码提示该条件不应假定。V1改为classifier前feature-sidecar `F_prev(I+QRQᵀ)h`，Q从`C F_prev`对应通道代理得到。

这是**有意的工程/科学定义修订**，不是声称前式已实现。它保留父方法，但新增一个常数大小d²学习参数F；此变更必须在CodeReview确认。增加dense-G L-only与joint-U基线，防止将容量变化误当golden-space收益。

### D2：严格区分两种子空间

父隔离空间作用于参数更新，读出敏感空间作用于指定feature。后者不等于低遗忘空间、不保证旧预测不变。V1不使用完整row-space恒等梯度投影来宣传创新；真正差异来自截断+前向参数化+监督路径分配。

### D3：F3承认是custom update，不伪装普通loss

逐位置各向异性D不能靠一个R参数全局hook实现。V1在低维残差a上使用identity forward/custom backward，并明确其不一定对应某个标量目标的梯度。它是UnCL-inspired而非原UnCL重现，取消硬PAS参与集合是另外一个必须匹配的因素。

### D4：F4有限内循环不是CGPM等价复制

采用原结构损失的有限步、受Q和幅度限制的teacher target修正。概率正确、有限与梯度隔离可验；严格凸性、最优解、解剖正确性都不能由3次更新保证。原始PAS掩码不随修正目标变尖锐而扩张。

### D5：F5不强迫在混合feature接缝上做分布匹配

KL仍用互补UL，SWD另加一次clean/strong-appearance U student特征前向。额外计算公开，消融保持曝光。此做法与原草图中的“任意混合特征来源恢复”不同，目的是避免将卷积感受野混合当严格纯来源。

### D6：三域而非原来的两域评价

为评价持续学习，计划用REFUGE共同源+两种后续顺序，Final改为三域等权。旧LCTX两域分数只作动机，不直接填新基线；源合法性与源训练成本必须实际绑定。

### D7：history-free定义与新增参数如实记录

当前原型、当前梯度校准/协方差、当前不确定性不跨阶段保留给训练器。保留父参数与新增累计F。训练checkpoint可存全部状态以恢复，旧checkpoint可在隔离evaluator/NAS档案保留，但不允许训练器历史回读。模型参数本身有历史知识；history-free不等于无历史信息。

## 3. 文献注册（Codex需在代码审阅前补齐author commit、license、精确API）

下面链接是检索/固定依赖入口，不是声称所有作者代码已在本环境跑通。F1–F5均为本研究改造，不得把原论文性能归为本组合性能。

### JML

Wang et al., NeurIPS 2023, *Jaccard Metric Losses: Optimizing the Jaccard Index with Soft Labels*。
论文：https://proceedings.neurips.cc/paper_files/paper/2023/hash/ee208bfc04b1bf6125a6a34baa1c28d3-Abstract-Conference.html
作者代码：https://github.com/zifuwanggg/JDTLosses
用途：F1的软标签区域度量。只采用明确JML1；masked image/class reduction是本研究适配，不把整个JDT/Dice/Tversky后续系列混称同一论文。

### CWMI

Lu, ICML 2025, *Complex Wavelet Mutual Information Loss: A Multi-Scale Loss Function for Semantic Segmentation*。
论文：https://proceedings.mlr.press/v267/lu25c.html
作者代码：https://github.com/lurenhaothu/CWMI
用途：F2的复可转向金字塔结构互信息。官方目录本轮可访问；完整后端/参数/许可尚需Codex固定。只提取结构项，避免重复CE；不以FFT误差替代。

### DyCON / UnCL

Assefa et al., CVPR 2025, *DyCON: Dynamic Uncertainty-aware Consistency and Contrastive Learning for Semi-supervised Medical Image Segmentation*。
论文：https://openaccess.thecvf.com/content/CVPR2025/html/Assefa_DyCON_Dynamic_Uncertainty-aware_Consistency_and_Contrastive_Learning_for_Semi-supervised_Medical_CVPR_2025_paper.html
作者代码：https://github.com/CVML-KU/DyCON
用途：F3借鉴连续不确定性使用，而非完整FeCL/projector或原UnCL MSE+正熵目标。停止梯度的spectral update是本研究新设计。

### D-Convexity

Chen and Yan, CVPR 2026论文记录，*D-Convexity: A Unified Differentiable Convex Shape Prior via Quasi-Concavity for Data-driven Image Segmentation*。
全文：https://arxiv.org/html/2605.19210v1
作者代码：https://github.com/ShengzheC/D-Convexity
用途：F4的准凹性结构项。只约束disc union与cup；内层Q修正是改造，不称原CGPM或严格凸投影。本轮可读取论文HTML；正式loss后端未在参考包中实现。

### SWSEG

Lu et al., CVPR 2025, *Improving Semi-Supervised Semantic Segmentation with Sliced-Wasserstein Feature Alignment and Uniformity*。
论文：https://openaccess.thecvf.com/content/CVPR2025/html/Lu_Improving_Semi-Supervised_Semantic_Segmentation_with_Sliced-Wasserstein_Feature_Alignment_and_Uniformity_CVPR_2025_paper.html
本轮网页直接访问返回403；论文信息与建议来自用户上传综述，Codex应在正式接入时读取可访问的论文/作者版本。
用途：F5借鉴sliced-Wasserstein，类别条件Q空间对齐与L/U采样是本研究设计。没有复现原论文的全部uniformity/高效近似目标。

### GOLD

*The Golden Subspace: Where Efficiency Meets Generalization in Continual Test-Time Adaptation*。
作者代码：https://github.com/AIGNLAI/GOLD
用途：判别子空间适配思想。V1不复制历史源原型/AGOP跨域库，不以“golden”名义主张全网或遗忘最优。卷积通道代理不等于完整卷积patch算子的精确行空间。

### JASCL / PAS

作者代码：https://github.com/prinshul/JASCL
用途：类别原型兼容性校验思想。这里当前域teacher-only PAS不是完整JASCL及其历史prototype replay。逐项读源码确认可微学生概率路径，不使用argmax标签间MSE当可微分割损失。

## 4. 来源映射要求

每个后端在 `review/SOURCE_MAP.md` 写：原论文目标／本次使用的部分／删掉或改变的部分／公式→函数→测试位置。论文与代码若不一致，记录差异并选择明确本轮定义，不能混合几种版本却声称完整复现。

本包是候选工程计划，不是查新完整报告。公开题名、会议与原论文实验不得转化为“本方法已具备期刊创新性/SOTA”的结论。正式稿件前还需对选中方案做近邻方法比较及合理贡献消融。
