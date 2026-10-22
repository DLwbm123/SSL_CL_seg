# 实际核验的依据与借鉴边界

核验时间：2026-10-22。以下“全文方法章节”表示成功访问论文全文并阅读相关方法/公式；不声称逐页精读全部实验及附录。不复制作者代码。

| 文献与实际版本 | 阅读范围与用途 | 不移植/限制 |
|---|---|---|
| R1 Mishra et al., [Robust Amortized Bayesian Inference with Self-Consistency Losses on Unlabeled Data](https://arxiv.org/html/2501.13483v4), v4, 2025-05-15；另核验 [ICLR 2026 会议记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0fd3d8093b3ba73d19b393a1326fdba7-Abstract-Conference.html) | 全文 §2.1–2.3，Eq.4、Propositions 1–3：固定模型的 log joint − log posterior 方差作为推断诊断 | 参数标签不是分割 mask；支持集覆盖与已知 likelihood 是性质的条件；不把诊断加入生成训练 |
| R2 Melo, Abate, Gal, [Temporal-Difference Variational Continual Learning](https://proceedings.neurips.cc/paper_files/paper/2025/file/d147f24cac1b6cd88753ca830e462bdc-Paper-Conference.pdf), NeurIPS 2025 正式 PDF | 全文 §3–4，Eq.2–8；顺序近似误差会累积，多步目标涉及历史 likelihood | 不实施 TD-VCL，不删除历史 likelihood 后保留名字；不复制其 likelihood tempering |
| R3 Sloman, Martinelli, Kaski, [Proxy-informed Bayesian transfer learning with unknown sources](https://arxiv.org/html/2411.03263v3), v3, 2025-06-13；[UAI 2025 会议记录](https://proceedings.mlr.press/v286/sloman25a.html) | 全文 §1–2，Assumption 1.2、Eq.1–3：共享与任务特异因素，以及错误先验造成负迁移 | 不使用 proxy、relevance weighting 或其源未知设定；本文只是建模启发，HBSI 不是 PROMPT 复现 |
| R4 Mishra, Kucharský, Bürkner, [Unsupervised Continual Learning for Amortized Bayesian Inference](https://arxiv.org/html/2602.22884v1), v1, 2026-02-26 预印本 | 全文 §2，Eq.1–4、§2.3 方法划分：识别网络自身可遗忘 | 未独立确认正式发表版本；不采用 episodic replay/EWC，不声称 self-consistency+CL 是新贡献 |
| R5 Kim et al., [Semi-Amortized Variational Autoencoders](https://proceedings.mlr.press/v80/kim18e/kim18e.pdf), ICML 2018 正式 PDF | 全文 §2.2–3、Algorithm 1：摊销初始化加局部梯度更新，最终目标对初始化/生成参数取总导数 | 使用 autograd 精确有限展开，不复制其有限差分 Hessian 实现、momentum、clipping 或额外损失；有限 K 不是真后验 |
| R6 Nguyen et al., [Variational Continual Learning](https://arxiv.org/pdf/1710.10628), v3, 2018-05-20, ICLR 2018 | 全文 §2–3，Eq.1 与 Eq.4：前一近似 posterior 作为新目标 prior，mean-field 权重 VI | 不采用 coreset；顺序近似与优化误差可累积，点估计 checkpoint 不是 posterior |
| R7 Kingma et al., [Semi-supervised Learning with Deep Generative Models](https://arxiv.org/pdf/1406.5298), v2, 2014-10-31, NeurIPS 2014 | 全文 §2–3，M1/M2、Eq.1–5：有无标签是同一生成模型的观测模式 | HBSI 的连续结构同时生成图像和标签，不等同于 M2 的先离散 y 再生成 x；不移植额外分类目标 |

Pro 共享分析已阅读：从文献筛选、层级变量、条件记忆、KL 抵消，到半摊销与 evidence-weighted 预测；其报告的解析数值没有直接作为本次测试结果。T07 独立计算。

许可记录：R1/R4 的 arXiv 页面标注 CC BY 4.0，R3 页面标注 arXiv 非独占分发许可；仅记录链接、概括方法，不随代码分发论文全文。未下载或改写任何外部方法源代码。新增 HBSI 代码为本次工作区实现，未擅自添加项目整体开源许可证；仓库根未发现统一 LICENSE。未来真实 evaluator 按接口调用仓库既有数据/指标实现，未修改其许可。依赖 PyTorch/NumPy/h5py/SciPy 沿用各自发行许可。
