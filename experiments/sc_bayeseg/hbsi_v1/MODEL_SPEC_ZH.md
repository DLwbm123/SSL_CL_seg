# HBSI-Seg v1 数学与实现规格

本规格先于合成运行冻结。独立实现，不改变 B11。新代码只依赖 PyTorch、标准库及真实接口所需的既有数据/指标模块；不复用 B11 step_loss。

## 模型及单位

一个观测是固定预处理后的二维全网格图像。volume 的切片在给定同一 Theta、a 后条件独立；切片不是独立患者。类别 K=2/3/4，图像通道 C=1/3。

Theta 包含 G、F 的全部权重与偏置。每个权重的独立先验标准差为 fan_in^(-1/2)，偏置标准差 0.1。全局 q 是对角高斯，sd=softplus(raw)+1e-6。首次 q 的均值为数据无关随机初始化，sd 为先验尺度的 0.1；优化初始化不改变先验。激活使用 tanh，生成器无 BatchNorm 或其他可训练旁路。

c~N(0,I)，R~N(0,I)，Pi=softmax([0,G(c,R)])；F(Pi) 为 S 的均值。G 由 c 的线性广播、R 插值及两层卷积组成；F 是两层卷积。固定 sigma_S=0.2、alpha=0.9：A=I-alpha W，W 是环面四邻域平均。S 密度为

`-CHW/2 log(2pi) - CHW log(sigma_S) + C logdet(A) - ||A(S-F(Pi))||²/(2 sigma_S²)`。

矩形网格特征值为 `1-alpha*(cos(2pi u/H)+cos(2pi v/W))/2`。b~N(0,0.1²I)，B 为二维正交归一 DCT 的低频 4×4 块、按 u 后 v 顺序、去掉 DC。a=(gamma,beta,xi)~N(0,I)，各通道独立；x~StudentT(4, exp(gamma) S+beta+Bb, exp(xi))。scale 不是标准差；a 的先验定义在无约束坐标，无额外 Jacobian。

标签只计入观测位置：叶类别使用 Pi，父类别使用其叶集合概率之和。每位置一个观测事件；未观测标签完全边缘化，贡献零。U 不读取隐藏标签。图像项逐像素逐通道完整求和。

## 条件记忆与目标推导

`Q_t=q_t(Theta) ∏d r_d(a_d|Theta)`，`r_d=N(m_d+C_d PTheta,diag(s_d²))`。P 使用固定 seed 的特征哈希投影，先用固定先验 sd 白化，再按桶内计数平方根归一；flatten 顺序、形状、先验尺度、桶/符号均入指纹。当前 q 的统计量从不改变坐标。

阶段目标未归一化密度是 `q_prev ∏old r_old p(a_t) ∏i p(x_i,z_i,y_obs|a_t,Theta)`。变分密度是 `q_t ∏old r_old r_t ∏i q_i`。取 log 比值时每个旧条件项逐点抵消，取期望得到

`KL(Q_t || normalized target)=KL(q_t||q_prev)+E_q KL(r_t||p(a_t))-Σi E[log joint_i-log q_i]+log Z_t`。

Z_t 不依赖待优化变分参数。此目标基于上阶段近似分布，不是历史精确 posterior 的一般保证。q_prev 深拷贝、阶段内固定；finalize 只封存一次，不在 epoch 刷新。

令 H=C_d P，则域边缘均值 `m+H mu`，协方差 `diag(s²)+H diag(sigma²) Hᵀ`。对单位域先验，期望 KL 为 `0.5 Σ(s²+(m+Hmu)²+diag(H diag(sigma²)Hᵀ)-1-2log s)`。旧 m/C/s 固定不意味着旧域边缘不变。

每个全局 draw 共享一个域 a 给所有当前 batch 图像。L/U 分别按 N/b 缩放求和，MC 求平均，global/domain KL 各一次。允许整个目标同乘固定常数，记录未缩放值；无单项权重、Dice、EMA、EWC 或自一致性训练项。

## 局部近似与梯度

小型无状态 U 形图像编码器输出特征。上下文包含图像特征、PTheta、a。顺序采样 `q(c|ctx) q(R|c,ctx) q(S,b|c,R,ctx)`，每个条件因子为对角高斯。每次采样重新计算条件参数。rho 是各因子 mean/raw-scale 上的局部偏移，初始为零；网络输出仍依赖当前父 latent。

`rho_next=rho+eta ∇rho ELBO` 使用函数式更新、训练 create_graph=True；K=0 即原始识别分布。训练 L 可用标签校正，初始化器与所有预测只用图像。proposal 拟合与最终估计使用独立 generator；预测不改变任何持久化参数、缓冲、优化器或外部 RNG。

## 联合预测与资格

对每域抽 Theta_s~q、a_s~r，局部 z_sl~q_rho。完整权重 `logw=log p(x,z|Theta,a)-log q_rho`；q/r 已抵消。volume 每个联合局部 draw 的 logw 为全部切片 logw 求和，所有切片共享 Theta/a。各域加 `log pi_d-log(S_d L_d)`，统一 logsumexp 后加权 Pi。默认 pi_d 均匀，无真实域 ID。不能替换为模型均匀平均。

分别报告固定 Theta/a 的 local ESS、按 W_s=mean_l(w_sl) 得到的 global ESS、域 posterior、最大权重、重复 MC 稳定性。非有限值报错，不 clip/drop。自归一化 IS 有有限样本偏差。合成运行前冻结初步资格门槛：所有 local ESS/L>=0.25、global ESS/S>=0.25、三个独立重复最大概率差<=0.1、log evidence 极差<=1 nat。未满足为 INFERENCE_UNQUALIFIED，不能进入真实比较。

## 理论边界

对固定旧域核 K，联合分布 q_t(Theta)K 与 q_prev(Theta)K 的 KL 等于 KL(q_t||q_prev)，边缘化由数据处理不等式给出 `KL(q_t K||q_prev K)<=KL(q_t||q_prev)`。该结论不保证真实数据条件预测、Dice、HD95、局部优化准确性、生成模型可识别性或无遗忘。对角 q、低秩条件依赖、低频外观与环面边界均为近似假设。

## 运行预登记

seed=20261022；toy 16×16、c_dim=2、R=2×4×4、width=4、projection_dim=4；Adam lr=3e-4，局部 eta=1e-4，训练 K=3，global/local MC=2/2；两域各40外层步，共80步，不因失败扩大。float64 数值测试，float64 toy。推断 S=8/L=4，三重复；K=0 与 K=3 都保留。数值测试每项<=300秒，累计自动计算<=3600秒、所有失败计入、总更新<=1000、产物<=2GiB。真实正式训练本轮不运行。

合成优化可行性门槛（首次神经运行前固定）：最终阶段 K=3 的两个域各自 mean foreground Dice>=0.65，预测前景比例在[0.02,0.98]。未通过报告 OPTIMIZATION_FAIL；即使通过，也必须另外通过上述推断资格，且不代表医学收益。

规格澄清：对同一组对齐的联合样本，逐切片归一化权重的乘积再归一化在单域内与 log 权重求和等价；错误在于丢失 evidence 常数后再混合域，或独立采样/平均切片预测。T13 分别验证这些情况，而不把上述代数等价情况伪造为负对照。
