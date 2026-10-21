# BayeSeg＋G 方法设计的全文阅读与近邻边界

2026-10-21。配套 [完整 Bayesian 方法、PGM 与匹配实验方案](BAYESEG_G_BAYESIAN_METHOD_AND_MATCHED_PLAN_ZH.md)。阅读不是代码复现、结果复核或定理独立认证；本文不给新方法虚构性能。

## 1. 阅读范围、版本和来源

直接方法文献集为：BayeSeg 原论文1篇；Pro所列关键全文8篇；与本问题直接相关的UAI/AISTATS5篇，共14篇。Pro的8篇此前已完成正文和附录阅读，本次复用阅读记录并重新核对机制/假设；新增5篇阅读正文、证明和实验附录，QAVI/BSVI另下载了独立补充PDF，没有只停在摘要。

|论文|本地版本与全文范围|正式来源|
|---|---|---|
|BayeSeg|arXiv2303.01710v1，15页，正文/实验/统计模型/讨论|[原论文](https://arxiv.org/abs/2303.01710)|
|Amortized Variational Inference: When and Why?|UAI2024，16页，含附录证明与附加实验|[PMLR](https://proceedings.mlr.press/v244/margossian24a.html)|
|Amortized Structural Variational Inference|AISTATS2026，30页，含理论与实验附录|[PMLR](https://proceedings.mlr.press/v300/fan26b.html)|
|A Decoder Suffices for Query-Adaptive Variational Inference|UAI2023，正文12页＋独立补充11页|[正文与补充](https://proceedings.mlr.press/v216/agarwal23a.html)|
|Training VAEs with Buffered Stochastic VI|AISTATS2019，正文10页＋独立补充7页|[正文与补充](https://proceedings.mlr.press/v89/shu19a.html)|
|Closed-Form CAVI for Student-t Process Regression with Student-t Likelihood|AISTATS2026，36页，附录A–L随主PDF|[PMLR](https://proceedings.mlr.press/v300/onoue26a.html)|
|TaskRel: Generate to Teach|所提供匿名投稿全文28页|[OpenReview](https://openreview.net/forum?id=11fDd05sMx)|
|Variable Elimination Networks / VEN|所提供匿名投稿全文24页|[OpenReview](https://openreview.net/forum?id=DiVwmT4kNy)|
|Amortized Factor Inference Networks / AFIN|arXiv2605.26419v2，35页|[arXiv](https://arxiv.org/abs/2605.26419)|
|Mitigating Representation Gaps with Auxiliary Supervision|arXiv2609.39525v1，24页|[arXiv](https://arxiv.org/abs/2609.39525)|
|Hierarchical Symbol Coupling / HiSC-EBM|所提供匿名投稿全文22页|[OpenReview](https://openreview.net/forum?id=NzzGCMwHGU)|
|Prior-to-Precision / PoPNet|所提供匿名投稿全文26页|[OpenReview](https://openreview.net/forum?id=g5t2rHHk4R)|
|AC-Diff|所提供匿名投稿全文20页|[OpenReview](https://openreview.net/forum?id=vkJ81EVoxD)|
|When Amortized Posteriors Become Priors|所提供匿名投稿全文30页|[OpenReview](https://openreview.net/forum?id=4bvVvNBWt5)|

匿名ICLR27近邻按提供的投稿版本讨论，不称已录用成果；正文与arXiv版本可能不同，页数对应本次实际PDF。图表数据按论文阅读，不将图像重绘、作者代码未复现的现象叫独立验证。PDF和用户提供的Pro材料保留私有，不上传仓库。

前次重定位报告还列过ACE、函数空间VI和TreeCopula等外围候选。此次没有把它们纳入核心方法或声称全部全文读完；ACE/函数空间PDF已取得但仅作后续扩展资料，不作为本文新颖性或理论保证依据。直接阅读集有明确边界，不等于“读完两个会议所有相关工作”。

两份用户提供的Pro分享分析已通过浏览器核对可见全文。较早分析关注MC、RV及均值/随机路径，较新分析关注匹配双视图控制与明确停止条件。另复用用户提供的两轮论文机制分析。Pro是建议，不是论文结果或新训练授权；其中建议的固定G输出一致性四臂已经执行并失败，不作为新方法重新启动。

## 2. 原 BayeSeg：继承什么，不能冒称什么

原论文§3建立图像分解、空间形状/标签和层级先验，并用神经网络近似局部后验；§3.3/实验训练式还区分监督与变分部分。§4–5的对照说明统计建模的作用，但使用不同数据、训练时程和评价策略，不能把其Dice搬到当前Siemens20%设置。

§5的先验消融同时影响形状和外观解释；§6也承认形状不完全域不变。t-SNE聚集不证明分割正确或因果解耦。原论文随机局部变量不等于所有生成网络权重都采用Bayesian后验。本方案的G/F、R32和权重q是扩展，保留形状—外观—图像联合路径但不冒称官方复现。

## 3. UAI/AISTATS：决定本方案的五个结论

### 3.1 UAI2024 AVI：摊销间隙与变分间隙分开

正文的模型条件和附录构造说明，识别函数能否达到因子化VI解取决于生成图及可用输入，而不是只看网络宽度。时间序列中需要的上下文与分层独立样本模型不同；不存在对所有模型通用的局部输入保证。[原文及附录](https://proceedings.mlr.press/v244/margossian24a.html)

本项目即使把H训练到独立族最优，仍可能缺少c/R依赖；也不能由该论文直接推断必须增加空间邻域。实验因此分别控制G学习和后验族，不以更低训练损失宣称真实后验更准确。

### 3.2 AISTATS2026 ASVI：分布有依赖，不是网络看到邻域

正文§2–3的条件分解与附录理论/仿真是关键。邻域输入的mean-field网络仍然因子化；真实条件后验需要在采样密度中体现依赖。论文时序链的结论有模型、近似和正则条件，不能移植成二维医学场保证。附录中的flow结果也显示更灵活或更高ELBO未必带来更好潜变量恢复。[正文和附录](https://proceedings.mlr.press/v300/fan26b.html)

采用：明确写q(R|c,I,Theta)，密度和KL可计算。不同：依赖取自G的共同特征注入子空间，没有复制时序链或其定理。

### 3.3 UAI2023 QAVI：生成器可支撑条件推断，但局部优化不免费

正文和补充§1推导缺失观测条件下的局部ELBO；对缺失特征边缘化依赖观测模型条件。补充§2.4显示mixture组件重初始化和混合梯度估计并非无代价；§3中beta可调，部分结果不是普通ELBO。缺失像素NRMSE采用大量候选的评价，不能照搬为部署时可用的GT选优。[正文与补充](https://proceedings.mlr.press/v216/agarwal23a.html)

采用生成因子参与条件后验的思想；不加入测试时每病例300/1000步优化、flow或mixture，不把图像补全收益直接外推为分割收益。

### 3.4 AISTATS2019 BSVI：重要性轨迹下界有条件，紧界不等于好分割

正文§3–5和补充A证明一般相关提议使用条件密度作分母；不能随意将一批优化轨迹按独立样本计算IWAE。最优buffer权重的比较不能改写成任意均匀buffer总更好；后验KL比较需要额外独立条件/双采样。补充B–D保留训练和测试差异及计算设置。[正文与补充](https://proceedings.mlr.press/v89/shu19a.html)

启示是生成器学习和推断近似会相互影响。当前ESS退化历史下，不添加buffer、重采样或局部迭代；更紧训练界不能替代患者Dice和RV失败审计。

### 3.5 AISTATS2026 Student-t CAVI：闭式代理投影不是任意非线性模型精确CAVI

正文§4和附录B/E保留函数与全局精度的Gaussian–Gamma依赖：先优化因子化代理，再最小化KL(proxy||structured)作投影。这里的投影有闭式解，但不能据此说直接优化原结构族有精确闭式CAVI或每步必然提高原ELBO。§5/附录C–D给稀疏与自然参数扩展；§6及附录G–L显示完整批次与小批次结果有差异，PNLL不总最佳。[全文](https://proceedings.mlr.press/v300/onoue26a.html)

对本项目：尺度混合与结构依赖须写清；非线性G/F没有该回归共轭结构，不直接套用其更新式。新方案仍MC重参数化训练，不假称已推导无采样医学分割CAVI。

## 4. Pro所列八篇：机制、采用边界与明确区别

|近邻|全文方法/附录核对重点|本方案的区别与不采用项|
|---|---|---|
|[TaskRel](https://openreview.net/forum?id=11fDd05sMx)|同噪声/同目标下正负类条件比较；正条件拟合不能被排斥负条件替代；双分支与调度计算需要公平计费|不加类对比hinge、LoRA或flow生成；保留正标签Cat似然与原CE/Dice，不把“错条件分数更低”叫后验正确|
|[VEN](https://openreview.net/forum?id=DiVwmT4kNy)|SPD局部Gaussian势与Schur/Cholesky/Woodbury消元；fill-in/缓存和非Gaussian混合限制|G/F是非线性，整体后验不Gaussian；只使用固定上采样算子的最小二乘投影，不宣称Gaussian全图精确推断|
|[AFIN](https://arxiv.org/abs/2605.26419)|typed模型因子作输入、模拟训练forward-KL、节点对表示与Gaussian后验；模型维数/代价边界|不训练跨先验/跨模型通用推断器；仅使当前G的Theta通过A(c)进入特定解剖后验，不输入全权重token或任意因子列表|
|[Auxiliary Supervision](https://arxiv.org/abs/2609.39525)|summary直接监督/独立点估计头与保留自由容量；点估计弱时无改善，高权重可伤联合密度|不另加点教师或summary监督；原标签似然只计一次，F图像目标由联合模型给出；U_F专门控制辅助目标本身|
|[HiSC-EBM](https://openreview.net/forum?id=NzzGCMwHGU)|层级符号耦合、共享生成/识别潜变量、符号边缘化；EBM分区与prior采样成本，mean读出不是积分|保留共享P的图像—标签联合解释；不用EBM、符号层搜索或MCMC；密集Cat标签像素条件分解，不直接复用其标量符号公式|
|[PoPNet](https://openreview.net/forum?id=g5t2rHHk4R)|VFM跨层affinity、近似NCut与门控融合、ROI图像生成；预测ROI错误可漏薄类，额外基础模型与算力混杂|无VFM谱先验、ROI选择或生成修补；G是归一化概率模型中的解剖因子，F为共同图像因子，不能把重构好当分割证据|
|[AC-Diff](https://openreview.net/forum?id=vkJ81EVoxD)|对齐多对比和跨患者swap；解剖旁路随机对比生成；下游微调及数据条件|当前无配对多对比授权，不加diffusion或跨患者swap；不从“外观分离”推导当前S具有因果不变性|
|[Recursive Posterior](https://openreview.net/forum?id=4bvVvNBWt5)|单次准确与分批组合分离、自诱导prior、固定数据split；组成一致不等于准确，训练匹配有失败|首版不做CL/后验递推，不把同一训练数据训练的posterior当新prior重复使用；后续持续学习必须另有组合/保持资格|

## 5. Pro建议与本次选择的关系

较早Pro建议先收口MC、RV、风险和温度，再以D0/Dmu检验局部分布的额外价值。这些已经完成；均值模型复现主要风险收益和局部分歧可靠性失败是约束，不是可删掉的旧结论。

较新Pro建议匹配U/G双视图、正监督不改、同Theta比较及停止条件。该轮已完成，G自身增益约0.0735pp且新增真实RV损失，没有晋级。它没有证明G永远不能学，也没有支持原固定接口继续扫JS。Pro进一步提出“后验扰动是否超过普通噪声”；新方案用NP_S对应这一问题。

现在用户明确要求基于BayeSeg＋G的完整Bayesian设计，因此研究对象变为可学习G及其条件后验，而非把旧K1重新运行。F和随机局部变量的恢复都有匹配控制；如果结构族/权重后验无增量，应保留确定性或独立族G，撤回不成立的Bayesian主张。

## 6. 从文献到可证伪新意

本次提出的具体候选是：在G的既有加性全局/局部特征注入处，构造参数只增17的图像门控、Theta条件化三角后验；把该后验放入BayeSeg形状/外观联合模型，与G共同学习，给出可计算logq与KL。这不是“ASVI＋VEN＋AFIN＋HiSC＋PoPNet”的拼接。

已有结构化VI、条件Gaussian后验和生成模型联合学习不能据为原创。新颖性只可能来自该特定生成几何与解剖推断的结合及其可验证作用，仍需更广检索和匹配实验支持；本报告不宣称首创、理论泛化保证或已提高Dice。设计给出10个有明确作用的对照、完整PGM/密度/监督目标、训练与推断预算、门槛和负结果处理，训练尚未启动。
