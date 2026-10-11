# 2026 跨领域机制检索：F5 / V15 后续候选

检索截止：2026-10-11。对象：ICLR、ICML、CVPR、ECCV、NeurIPS 2026 的相关论文，以及近期 arXiv；不是这些会议全部论文的系统综述。优先原始论文、官方会议/论文集、作者来源。会议身份与阅读深度分列，预印本不自动视为录用论文。没有复现论文，也没有新增优化、患者探针或正式实验。

结论：优先检验**实际优化器更新是否破坏已有保护方向**；第二候选是**空间结构感知的入口知识保护**；第三是**教师可靠性与稳健蒸馏**。参数重要性和更新集中度作为备选。这里给出的是候选与判别条件，不是新的冻结实验协议，也不指定 V16 必须采用其中任何一个模块。正在运行的 V15 保持冻结；全部封存、统一读出并交付后，再依据完整结果选择一个主改动。

## 1. 已有项目证据与不能越过的推断

- F5 与 B2 共享 NativeLRParent 的硬输入投影。原 F5 的 U 只进入末端 R；历史 B2 的 U 进入可训练 A/B，而且没有 F5 的额外 R。B2 的 PAS 和 U 系数也不同，不能把差距归因于单一因素。
- V9 联合 A/B/R KL 反馈明显改善；V10 的入口教师拒绝进一步改善。V11 恢复入口目标有均值收益，但仍有严重负类别，不能把入口教师当成正确旧标签。
- V13 的类别均衡参考修复了 V12 的部分 seed 163 cup 退化。V14 额外监督半空间在 16,960 次调用中新增双活动为 0，完整匿名结果数值等于 V13：Final 67.8258%，相对历史 B2 −0.0753 pp。164/O1 RIM rim 仍为 58.3813%，相对 B2 −5.5651 pp；order 2 Incoming 相对 B2 −1.3553 pp。它不支持“再加总监督约束即可解决”的解释。[已交付 V14 报告](https://github.com/DLwbm123/SSL_CL_seg/blob/9364356094b75afaea8667b0efd3507b62d71a65/experiments/lcrseg/docs/f5_dual_reference_v14/COMPLETION.md)
- V15 检验各类别参考方向之间的抵消；当前没有在本次文献工作中读取其部分患者结果。旧域指标下降不能直接等同于边界错位、教师错误、梯度冲突或 Adam 引发的损害，这些需要分开验证。
- V8 的 Cayley 正交几何真实生效而 Old 未改善，因此不优先再做同类正交模块；V6 提高 U 系数也没有解决问题。

### 当前实现的准确边界

`five_frameworks_v1/train_stage.py` 的顺序是 split_gradients → transform_gradients → Adam.step → parent.apply_constraints → EMA/原型提交。V15 的约束作用于 `p.grad`。

核实 `five_frameworks_v1/native_parent.py` 后，**native apply_constraints 仅检查有限值和投影残差，不再次改写参数**。硬输入投影在有效权重参数化中。不能把它说成 Adam 后又执行一次参数投影。仍需区分原始梯度、Adam 参数位移，以及低秩乘积形成的有效权重变化；参数空间一阶条件不等于有限步函数保持。

## 2. 优先级最高：把保护从原始梯度推进到实际更新

### GUA — Gradient–Update Mismatch: Rethinking Conflict-Free Training of Physics-Informed Neural Networks

- 来源：arXiv，2026-09-01；本次未确认会议录用。已读正文方法与优化器状态附录。[论文](https://arxiv.org/html/2609.01558v1)
- 机制：研究经优化器变换后重新出现的方向冲突；将优化器提议投影到约束锥，并可调整内部动量状态。完整方法还含状态对齐强度，不能描述为只有一个无系数的投影。
- 本项目推断：V15 已有至多三个参考梯度和小规模活动集求解器，可先做有限诊断，确认“梯度合格、Adam 位移不合格”是否实际发生。若存在，再将**更新层面的最小修正**作为一个独立候选。仅迁移方向修正应明确为部分借鉴，不能称完整 GUA 复现。
- 限制：论文主要证据来自 PINN，不能外推医学旧域收益；投影仍依赖当前参考，无法修复错误入口目标或当前 U 缺少旧域支持的问题。完整状态对齐增加实现和恢复风险。

有限诊断建议（尚未执行、须在新协议中计费）：用相同状态和当前获准批次，定义下降位移 `u = theta_before - theta_after`，分别记录投影梯度与各 `h_c` 的 dot、Adam 提议位移与各 `h_c` 的 dot、有效低秩权重变化及参考函数变化。覆盖 A/B/R，不用 R 的方向代替 A/B。已有 before 参数副本和参考梯度可减少重复计算；参考函数复核若需要前向，必须单列预算。

若发现冲突极少或修正后函数损害仍在，不把该方向强行推进到 21,200 步；转向参考质量/覆盖或结构机制。若正式采用，先验证 disabled 对照、Adam state/step/scheduler 恢复、教师冻结、部署、失败调用台账。保持主损失、系数和数据边界不变，不与结构蒸馏同时叠加。

### ICML 2026 — Reducing Per-Sample Harm in Stochastic Optimization

已核对官方海报摘要。它把批内梯度平均和历史动量导致的逐样本损害作为优化问题，并以最后线性层的统计近似降低成本。[官方来源](https://icml.cc/virtual/2026/poster/62866)

借鉴价值是检查**均值保护是否掩盖局部损害**。原论文的最后线性层近似不能直接移植：本项目分类头冻结，而且可训练 A/B 分散在主干。逐样本不自动等于逐像素/逐区域；先做有限区域诊断，不添加全量逐像素梯度。

## 3. 第二候选：保护空间结构，而不只保护类别概率

### CVPR 2026 — SARD: Structure-Aware Representation Distillation for Tiny-Dense Object Segmentation

已核对 CVF 官方摘要及检索到的消融表；全文 PDF 获取不稳定，尚未完成逐式审阅。它用结构重要性图强调边界、几何复杂度和局部特征变化，并进行表示蒸馏。[官方论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Structure-Aware_Representation_Distillation_for_Tiny-Dense_Object_Segmentation_CVPR_2026_paper.html)

本项目推断：类别总量平衡不能指定 rim/cup 的空间关系。候选可以只替换入口参考的构造：复用当前 U 的入口/学生特征，保护局部关系或结构重要区域，保持主 KL、教师数量及推理模型不变。它与 V1 的边界/内部 CDF 分布对齐不同，不能仅换名称重做 V1。

先核对已完成轮次的许可范围内证据是否支持局部结构损伤；单凭 rim Dice 下降不足以认定边界原因。入口模型可能给错结构，局部特征存储、归约和额外 VJP 也要计费。原文是教师到学生压缩，不是当前同容量入口教师的持续学习证明。

## 4. 第三候选：提高参考知识的可靠性，而非只调 PAS 阈值

### ECCV 2026 — REDistill: Robust Estimator Distillation for Balancing Robustness and Efficiency

ECCV 官方目录及 Springer 论文集确认；已读 arXiv 正文。[论文](https://arxiv.org/html/2602.04677v1)，[ECCV 2026 论文集 Part XIX](https://link.springer.com/book/10.1007/978-3-032-37550-6)

机制是借助稳健估计的 power-divergence 改变错误监督的影响，而非统一缩放 KL。可考虑单独替换参考 KL 的几何，先固定公式、不扫描参数、不同时改主 KL。其证据不直接覆盖本项目的分割和旧域；稳健性也可能削弱正确的新域信号。必须通过受控错教师生成案例确认期望影响，再谈正式试验。

### NeurIPS 2026 — CaRE-KD / Distilling What Matters

已核对官方海报，阅读 arXiv 方法。[官方来源](https://neurips.cc/virtual/2026/poster/149862)，[正文](https://arxiv.org/html/2609.36734v1)

它按教师/学生置信切换蒸馏散度，并用认识不确定性拒绝不可靠监督。对本项目的启发是“入口模型很自信”不等于参考正确；但 LLM token 蒸馏不同于分割像素、同容量 EMA 和入口教师。完整不确定性估计可能需要多次前向，不能视为免费。若迁移，应单独选择可靠性模块，并用获准当前 L 校验，不用封存测试挑阈值。

### ICLR 2026 — Rethinking Model Calibration through Spectral Entropy Regularization in Medical Image Segmentation

已核对官方论文集摘要，未完成全文逐式审阅。论文从频域校准置信图，结合谱熵和功率谱平滑，覆盖边界与区域结构的不确定性。[官方来源](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c6e954799a0218f6d341ad5cbfb58999-Abstract-Conference.html)

它更接近医学分割，但先应测量当前教师的校准缺陷；不能凭跨域落差就断言过度自信，也不能把新的正则项偷偷加入现有冻结轮次。它与 CaRE-KD/REDistill 是不同候选，不自动组合。

## 5. 备选：有效权重的重要性与更新集中度

### ICLR 2026 — Revisiting Weight Regularization for Low-Rank Continual Learning / EWC-LoRA

已读官方全文。它在低秩参数化下估计全维有效权重的重要性，并约束共享低秩更新，而非只分别限制 A/B 的范数。[官方论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/47197750b407ef48daa1438e0c0af242-Abstract-Conference.html)

本项目可以研究阶段结束时保存不含患者的参数重要性摘要，以减轻仅靠当前 U 保护旧知识的覆盖问题。代价是额外梯度、存储及陈旧重要性风险。必须在新轮阶段开始前冻结采集规则和预算，不能回读历史阶段私有训练数据补算、也不能假定现有 checkpoint 已含 Fisher。当前已有硬输入投影，二者功能需区分。

### NeurIPS 2026 — Concentrated Gradients Amplify Forgetting / DiGPro

已核对官方摘要，未取得全文，不列为可直接实施方案。它从近期梯度历史提取主方向，减弱更新在少数方向的集中；不需要历史任务原始数据。[官方来源](https://neurips.cc/virtual/2026/poster/154350)

这与 V15 按类别保护方向不同。先测 A/B/R 更新的集中度是否与损害伴随，不能只见低秩就推断集中有害；短梯度缓存、分解、参数选择以及恢复规则均有新成本。

### ICML 2026 — SABER: Continual Learning with Representation Conflict Management

已核对 PMLR 官方摘要，未完成全文审阅。它研究子空间对齐、正交分量分解和按能量重组。[官方来源](https://proceedings.mlr.press/v306/luo26ag.html)

有助于考虑在哪个表示位置保留/吸收变化，但项目已包含 Q 子空间与硬输入投影；在逐式确认它提供额外机制前，不优先再加表示分解模块。

## 6. 近期 arXiv：可借鉴的发现与降优先级依据

| 论文 / 首次提交 | 阅读情况与可迁移发现 | 本项目判断 |
|---|---|---|
| [Muon Can Outperform Dedicated Continual Learning Methods](https://arxiv.org/html/2609.24678v1)，2026-09-21 | 阅读摘要及主要机制：优化器可改变矩阵更新的谱分布；组合结果并非全部正向 | 优先诊断实际更新；直接更换所有 A/B 优化器会引入较大归因混杂。Muon 的矩阵更新与 V8 的特征旋转不是同一个机制 |
| [Does Gradient Conflict Predict the Understanding–Generation Trade-off?](https://arxiv.org/abs/2609.38465)，2026-09-29 | 摘要和受控实验结论：方向冲突指标并不可靠预测该受控多模态任务的取舍 | 防止把“冲突减少”当成科学成功；其负发现也不能外推为本项目所有投影无效 |
| [Divergence controls entropy in distillation](https://arxiv.org/html/2610.03529v1)，2026-10-02 | 阅读理论解释和实验：蒸馏散度及采样会影响学生熵，不能把 KL 权重当作唯一机制 | 支持检查置信分布和参考几何；推导假设不自动适用于当前有限容量分割模型 |
| [PaLoRA: Paced Low-Rank Adaptation for Continual Learning](https://arxiv.org/html/2610.04226v1)，2026-10-03 | 阅读方法：历史更新压缩、投影及有效秩相关 pacing；arXiv 作者注明 NeurIPS 2026，本文按所读预印本引用 | 包含多项变化及步幅控制；不作为“下一轮只做一个方法模块”的首选，也不借名重做系数搜索 |

检索覆盖截至当天可获取的相关条目，不宣称已经纳入 arXiv 最后一天全部新增论文。额外检索到的 2026-10 条目中，元数据与 HTML 标题不一致的来源未用于机制结论。

## 7. 选择规则和停止条件

1. 先等待 V15 全部封存、统一读出和完整公开交付。若已通过原双门槛，按既定停止条件执行，无需为了新论文继续跑。
2. 若未达标，首选**有限实际更新诊断**。只有测到足够明确且可修复的梯度—更新失配，才注册更新层面的单模块试验；若未测到，转向结构或参考可靠性，避免继续堆叠没有激活的约束。
3. 结构候选需要证据区分边界/区域关系损伤与整体类别偏置。可靠性候选需要区分入口与 EMA 的正确/错误置信，且只使用新协议明确允许的数据；不能从教师互相同意推导正确。
4. 仍固定两种子、两顺序和必要匹配对照；逐 seed/order/domain/class 公开所有负细胞。主要关注 164/O1 RIM rim、order 2 Incoming 和 163/O2 cup，但不能用这三个局部指标替代完整矩阵和原门槛。
5. 下一轮只选一项主改动。不得将 GUA、SARD、CaRE-KD、EWC 或 DiGPro 一次性叠加，也不得因为论文较新就默认优于既有方法。

原成功规则保持：同时相对匹配 F5 和历史 B2，各 seed 平均顺序后的 Final 增益均正、总均值至少 +0.005；各 order Old/Incoming 至少 −0.005；每 seed/order/domain/rim 或 cup 至少 −0.05。开发患者的重复使用不构成独立泛化、SOTA 或临床有效性证据。

本次仅文献和源码分析：新增正式优化 0、资格优化 0、患者前向/VJP 0。未修改 V15、数据、权重、配置、预算或训练代码。所有适配建议都是待检验假设。
