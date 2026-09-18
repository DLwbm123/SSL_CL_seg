# 证据来源与解释边界

检索/读取日期：2026-09-18。

## 已实际读取的项目原始资料

[S1] DOSE缩减结果：
https://github.com/DLwbm123/SSL_CL_seg/blob/12e8bee31df6723a333f702c8a011e19053b628d/results/nka_dose_v0_1_s163_o1/FINAL_REPORT.md
支持：4/4新节点、8400正式更新、用户取消12节点、性能差值、计量有效、原门槛NOT_ASSESSED_REDUCED_SCOPE。未重新读取服务器真实权重或患者数据。

[S2] B2历史逐轨迹指标：
https://github.com/DLwbm123/SSL_CL_seg/blob/3034b2199aa7d6e54ea67499f391a0dd4ea3d21e/results/f5_confirmation_v1_p1/FINAL_METRICS.csv
支持：seed163/O1与O2独立配对的baseline值。新研究只导入相关两行及其原收据，而不是两个seed总均值。

[S3] 实际父方法：
https://github.com/DLwbm123/SSL_CL_seg/blob/d557bd245eb204ec3ca377a5d868a03dc31436a5/experiments/lcrseg/five_frameworks_v1/native_parent.py
支持：A-only参数化、允许参数组、主头valid卷积的几何、teacher dense EMA与stage-exit合并。

[S4] 原训练器：
https://github.com/DLwbm123/SSL_CL_seg/blob/d557bd245eb204ec3ca377a5d868a03dc31436a5/experiments/lcrseg/five_frameworks_v1/train_stage.py
支持：原监督/U分开求导合并、参数组只包含父A/B和可选R、新aux必须显式接入、warmup与EMA。

[S5] 原互补视图：
https://github.com/DLwbm123/SSL_CL_seg/blob/d557bd245eb204ec3ca377a5d868a03dc31436a5/experiments/lcrseg/five_frameworks_v1/recipes.py
支持：两互补视图的anchor来源收集、无arm的stateless RNG、U去重；新aux损失须保持正确source mask。

[S6] 原CE+Dice定义：
https://github.com/DLwbm123/SSL_CL_seg/blob/d557bd245eb204ec3ca377a5d868a03dc31436a5/experiments/lcrseg/ssl_anchored_mix_v0_1/core.py
支持：主监督不能简写成仅CE；新深监督沿用明确的ignore、类/图像归约与精度。

## 已核验的主要相关论文/期刊标准

[R1] ECOCSeg, NeurIPS 2025, Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective.
https://proceedings.neurips.cc/paper_files/paper/2025/hash/9f6c0fb150fc9de81d78d85b082f26f1-Abstract-Conference.html
正式摘要核验：ECOC类属性表示与bit-level伪标签去噪。AGMS的disc父类集合监督是任务改造，不是原ECOCSeg复现，也不继承纠错码保证。

[R2] LoMix, NeurIPS 2025, Learnable Weighted Multi-Scale Logits Mixing for Medical Image Segmentation.
https://proceedings.nips.cc/paper_files/paper/2025/hash/5f0e54e34ba4c30fbc6f229986dbffb4-Abstract-Conference.html
正式摘要核验：多尺度预测混合和训练期监督。AGMS没有复用全部融合操作/NAS式loss权重；普通深监督和uniform融合是必要对照。本文不复述原文效果幅度来预测AGMS收益。

[R3] SCNP, CVPR 2026, Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels.
https://openaccess.thecvf.com/content/CVPR2026/html/Valverde_Towards_High-Quality_Image_Segmentation_Improving_Topology_Accuracy_by_Penalizing_Neighbor_CVPR_2026_paper.html
官方条目核验：借局部最差邻居预测惩罚改善结构。仅作为P2方向；当前P1不实现，不宣称严格全局拓扑或解剖健康形状保证。

[R4] IEEE TMI, Key Criteria for Publication.
https://ieeetmi.org/key-criteria-for-publication/
官方标准包括新科学/方法、技术质量、适配性与非增量影响。本文设定的0.3/0.1点等开发门槛不是TMI规则，也不能预测录用。

## 哪些内容是新提案，而不是已有事实

AGMS的两个aux头、父类接收规则、Brier风险EMA、风险融合权重、0.25/0.5系数、0.9阈值、.01分歧门槛、六臂矩阵及开发门槛，均为本次尚未验证的设计选择。它们不是来自原论文的已验证医学持续学习组合，也不是被证明的最优参数。

没有在本次执行训练、GPU资格、真实数据审计或代码审阅批准。文件中的PROPOSED_PLAN是供Codex实现并提交审阅的提案；只有对应新代码通过后才可按单独授权执行。
