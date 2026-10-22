# 独立后验参照：推导与证据范围

本轮固定生成模型与域变量，仅求 p(c,R,S,b|x,Theta,a)。没有标签似然、phi、优化器、全局KL或CL更新。它不是完整HBSI成绩，也不能证明真实医学效果。原生成密度和原失败实例完全保留。

## Student等价增广
每像素omega~Gamma(nu/2,nu/2)，Gamma为shape-rate；x|S,b,omega~N(exp(gamma)S+beta+Bb,exp(2xi)/omega)。积分omega恰好返回原Student-t密度，含全部常数。omega|其余为Gamma((nu+1)/2,(nu+residual²exp(-2xi))/2)。这不改变噪声模型。

## 联合与collapsed更新
令u=[c,R]，h=[S,b]，m(u)=[F(Pi(u)),0]，V=blockdiag(sigma²(A'A)^-1,tau²I)，H=[exp(gamma)I,B]，D=diag(exp(2xi)/omega)，C=HVH'+D。积分h后的条件图像密度为N(x-beta;Hm(u),C)。每次omega固定时对C作一次Cholesky，椭圆切片更新u使用完整Gaussian logpdf及标准高斯先验。

然后联合条件均值为m+VH'C^-1(x-beta-Hm)，协方差为V-VH'C^-1HV。实现用条件模拟而非显式反复求逆：先采h0~N(m,V)、e~N(0,D)，返回h=h0+VH'C^-1(x-beta-Hh0-e)。样本均值与协方差恰好为上述条件分布；因此保留S/b的不确定性及其相关性。

核顺序必须为u|omega,x（积分h）→h|u,omega,x→omega|h,x。第一步丢弃旧h依赖，第二步恢复与新u匹配的h后才更新omega，从而保持联合目标不变。不能在collapsed u更新后拿旧h直接更新omega。每次椭圆方向独立标准高斯，slice阈值及完整2pi初始角区间按原算法构造；最多1000次收缩，耗尽直接失败。

与Pro提出的逐条件参考相比，本实现进一步在结构更新中解析积分S/b，以减少结构与高维场之间的条件依赖。该选择在首次数值运行前冻结，没有根据诊断结果调整。

## 多链诊断
所有保留步同时记录256维前景概率、34维c/R、271维S/b、256维omega及原始完整log_joint。概率和参数分别报告诊断，不把两类ESS与旧重要性权重ESS混称。

每条链分成前后两段，池化平均秩后使用Blom变换( rank-.375 )/(N+.25)，进行rank-normalized split Rhat；折叠到池化中位数的绝对偏差重复同一计算，取两者最大。跨链ESS由均值间/链内方差与FFT自协方差、Geyer初始正序列及单调对和计算；bulk用rank变换，tail取5%/95%指标两者较小。ESS估计保守上限为样本总数，常数坐标判不合格，避免静止链假通过。MCSE使用每链50步非重叠batch均值；链均值差另列，不能用小MCSE掩盖链不一致。

1.01/100等预注册门槛是本次预算中的可信度筛选，不是严格收敛保证。8幅全通过才允许对信息性做正式阶段判断；任何图不合格时全部标签评分只能用于诊断，不以好看的平均Dice升级。

## 自检与局限
参考检查涵盖Student解析积分、独立dense Gaussian条件公式及密度恒等式、12000条件模拟样本矩、椭圆切片对一维解析高斯目标、IID正对照/链偏移及静止负对照。全部独立于神经参照的标签成绩。回归T01–T18/R01–R09另行全量运行。

当前实现限16×16灰度二分类诊断；dense协方差不是可直接扩展到医学原分辨率的预测器。不会估计边际证据logZ，MCMC的目标密度有归一化项不等于已求出模型证据。真实latent Pi只作为oracle诊断，既不参与链初始化也不进入推断。

## 阅读来源
- Murray, Adams, MacKay. Elliptical slice sampling. AISTATS2010, PMLR9:541–548. 阅读全文中的目标、椭圆参数化及算法。https://proceedings.mlr.press/v9/murray10a/murray10a.pdf
- Vehtari et al. Rank-normalization, folding, and localization: an improved R-hat for assessing convergence of MCMC. arXiv1903.08008v5 (2021-06-22). 阅读Rhat、rank/folding及ESS定义。https://arxiv.org/html/1903.08008v5
- Pro分享仅为研究设计输入：https://chatgpt.com/share/6ada1d0e-9fe4-83ea-b0dc-82e0dbfb8a66

源码为独立公式实现，未复制第三方代码或引入新依赖。论文给出算法，不为本轮有限计算提供成功保证。
