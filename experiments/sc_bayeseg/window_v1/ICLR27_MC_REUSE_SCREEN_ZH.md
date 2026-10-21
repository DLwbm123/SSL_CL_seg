# ICLR2027投稿：MC梯度模块筛选

2026-10-21。用户补充允许参考ICLR27可复用模块。读取2026-10-18公开投稿快照43,013条，以control variate、variance reduction、pathwise/reparameterization gradient、QMC、Rao–Blackwell、Hermite检索得到100条候选，人工按当前连续Gaussian后验/可微G筛选。不是录用名单，也不是全量全文阅读。OpenReview论坛/PDF当前403；不把摘要筛选说成全文复现。

## 优先借鉴：梯度空间控制变量

[Variance-Optimal Control Variates for Learning with Black-box Feedback](https://arxiv.org/html/2610.04766v1)，目录ID DSbWea7F6G。已读公开正文§3–4.4及实验适用范围，查看作者仓库README/baselines.py（未执行、未移植源码）。

文章指出共享参数下逐项最优的控制变量未必使总梯度方差最小；通过中心化残差及可实现梯度子空间投影构造修正。它采用score-function估计，并包含固定方差Gaussian黑盒实例；当前连续可微pathwise后验同时学习尺度/相关项，不能直接套用其最优性定理。可复用的是梯度级目标、明确零均值校正和历史样本拟合流程；尚非已复现的模块。

## 次级候选，当前仅摘要证据

- [SLICE](https://openreview.net/forum?id=dn6zk6OOOE)：过去批次拟合、当前批次固定的控制变量；可借鉴避免同样本拟合校正导致偏差，原场景为semi-value而非VI。
- [ISCV](https://openreview.net/forum?id=GyQIgFqovN)：按残差二阶矩分配采样，原场景masked diffusion；当前无时间维，不直接添加模块。
- [Foundation Models for Variance Reduction](https://openreview.net/forum?id=ndMwLPvInU)：识别与估计分离；预训练成本不适合当前小预算，不引入基础模型。

## 与现有负结果对应的下一独立假设

二次解析控制变量不能解释非线性高阶梯度噪声。下一实验采用Gaussian零均值Hermite基，直接拟合局部输出梯度后映射到共享参数，必须使用独立校准与评估随机样本；总draw仍128，明确64校准+64估计，与128纯IID及Sobol对照。有限四次多项式只作可证伪单元基准，若近零残差不能冒称一般G已解决；随后加入固定非多项式G及学习G耦合检验。基的选择、预算、停止条件在运行前另立协议，不根据评估集挑基/系数/seed，不将Hermite旧思想包装成ICLR27新模块。正式实验条件授权已收到，但完整准入当前未满足。
