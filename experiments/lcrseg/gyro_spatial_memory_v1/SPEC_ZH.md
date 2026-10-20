# GYRO 空间凝缩闭环：完整公式草案 v0.1

日期：2026-10-20。仅为用户要求的方法与实验设计；没有实现或启动训练，没有恢复自动监测，不修改任何已完成轮次。本文是独立新候选，不是 LCR-Seg V0.1 的修改。

## 0. 本版明确选择什么

先研究 Siemens→Philips 两阶段、同一器官标签空间的持续分割。源阶段数据可访问时形成记忆；Philips 阶段训练不得访问 Siemens 原始训练图像／mask、历史 image bank、逐原图特征或旧模型在线监督。

最小记忆采用 **32 个可学习的 8×8×32 空间码＋共享图像／mask 生成器＋固定空间投影器＋固定 frame**。不是逐图特征缓存；不是给每个像素独立拟合高斯；不宣称无穷新样本采样。本版是有限潜在码的任务效用凝缩，生成变体仅来自增强。

图像与 mask 由同一个空间码、同一个生成躯干联合产生。进入 Philips 后，码、投影器、生成器和 frame 全部固定，只有原生分割网络更新。先用这个版本检验共同几何接口，**不同时解决无限阶段共享生成器更新**。

## 1. 设定、数据与算子

第 t 阶段合法训练数据为

\[
\mathcal D_t^L=\{(x,y,v)\},\qquad \mathcal D_t^U=\{(x,v)\}.
\tag{1}
\]

x∈R^{C_in×H×W}，y 为四类像素标签（背景、LV、MYO、RV），v 是既有 loader 的有效像素掩码。H、W、输入通道与归一化沿用原生数据合同，启动前由 manifest／真实张量确认，不能猜测或重采样整个真实数据集。U 接口不返回隐藏 GT。

原生模型为

\[
f_\theta(x)=\operatorname{softmax}\bigl(S_\theta(E_\theta(x),\operatorname{skip}_\theta(x))\bigr).
\tag{2}
\]

S 是原 segmentation decoder/head；它与下面的图像生成器 G 是两个不同映射。原 native 允许训练参数全部保留可训练，原 sigma.weight／grad_update 常量仍固定。

定义空间编码

\[
\Psi_{\theta,p}(x)=\mathcal N\!\left(P_p\left[\operatorname{AAP}_{8\times8}(E^b_\theta(x))\right]\right)
\in\mathbb R^{8\times8\times32},
\quad \mathcal N(a)=a/\max(\|a\|_2,10^{-6}).
\tag{3}
\]

E^b 是最后一级 encoder 特征，AAP 为 adaptive average pooling；P 是一个可训练 1×1 线性卷积，输出 D=32，不使用 BN。归一化沿每个空间位置的通道轴执行。P 在源记忆准备后固定；固定 P 不冻结 E。

这替换了原 GYRO 的全局 GAP 投影接口，是明确的 dense 变体。8×8 是为小规模验证选的空间瓶颈，不声称足以重现薄心肌边界。

## 2. 空间记忆与联合生成

### 2.1 记忆的参数化

可训练参数为 V_j∈R^{8×8×32}，实际使用

\[
Z_j=\mathcal N(V_j),\quad j=1,\ldots,M,\quad M=32,
\qquad q_{\mathcal M}=\frac1M\sum_{j=1}^M\delta_{Z_j}.
\tag{4}
\]

每个 Z_j 是完整空间场，所有位置与通道共同优化；不把 64 个空间位置打散独立采样。整个记忆代表源训练任务，**没有“原图 i→保留码 i”的索引**。

### 2.2 图像与 mask 如何生成

\[
F_j=G^{\rm trunk}_\omega(Z_j),\quad
\tilde x_j=G^x_\omega(F_j),\quad
\tilde m_j=\operatorname{softmax}(G^m_\omega(F_j)/T),\quad T=1.
\tag{5}
\]

G trunk：在 8×8 上用 3×3 Conv 将 32 通道变为 128；随后五个 bilinear×2＋3×3 Conv＋GroupNorm(4 groups)＋SiLU block，输出通道依次 128、96、64、32、16。最后按原生 H、W 插值，同一个 F 接两个 3×3 head：图像 head 输出 C_in 个实数通道，mask head 输出 4 个 logits。图像 head 不使用 tanh，保留既有归一化输入的取值范围。

不允许从真实图像向生成器提供 skip connection；target 阶段 G 只接 Z。原生分割网络的 skip connection 则正常保留。

mask 是生成器的联合输出，不是每次 Philips 更新时调用旧模型重新打标签。两者共享空间码和生成躯干，但**共享本身不保证图像与 mask 正确对应**，必须通过下述源监督与任务效用检验。

## 3. 源阶段 A：准备共同可解码坐标

源原生入口固定为同一个 S† 参数 θ₀；全程只读 θ₀，包括 buffers。仅更新 p、ω。对当前源 L、U 的图像，取 Z(x)=Ψ_{θ₀,p}(x)，生成重建图与 mask：

\[
(\hat x,\hat m)=G_\omega(\Psi_{\theta_0,p}(x)).
\tag{6}
\]

用源当前数据训练

\[
\mathcal L_{\rm codec}=
\mathbb E_{\mathcal D_0^{L}\cup\mathcal D_0^{U}}\operatorname{MAE}_{v}(\hat x,x)
+\mathbb E_{\mathcal D_0^{L}}
\left[\operatorname{CE}_{v}(\hat m,y)+\operatorname{SDice}_{v}(\hat m,y)\right].
\tag{7}
\]

所有项按有效像素平均；前景 soft Dice 对 LV/MYO/RV 等权，平滑常数 10⁻⁶。三项系数均为 1。源 U 只参与图像重建，绝不读取 mask。

这是源阶段普通可解码性准备，所有实验组共用相同的最终 p₀、ω₀；不能把这部分的重建或监督本身宣传为 GYRO 新机制。500 固定更新，不在源／目标验证集选最佳 codec。

冻结 p₀、ω₀ 后，对合法源训练图像做一次无优化器的顺序编码，流式估计每位置每通道的 μ、σ，不保存逐图码。初始化

\[
V_j^{(0)}=\mu+0.1\,\sigma\odot\epsilon_j,\quad\epsilon_j\sim\mathcal N(0,I),
\tag{8}
\]

随后按式(4)归一化。μ、σ 只用于初始化，凝缩完成后释放。本式不表示最终记忆假设空间位置独立或服从高斯。

## 4. 一次共同的 GYRO 空间算子

源、目标各有独立随机定向、初始化后固定的 simplex frame E_t∈R^{K×D}，K=10、D=32，β=1/3。frame 不命名器官。native S† 原有全局 projector/frame 不用于这个新空间，且不被改写。

### 4.1 空间采样与增强

每图 8×8 网格分为四个 4×4 象限，每个象限均匀取一个位置，R=4。来源阶段 t、图／码索引和位置均有明确标识，但真实患者索引不进入持久记忆。

最小版本不新增随机裁剪、仿射或空间增强：弱视图是原生预处理后的图像；强视图复用同一确定性 strong_view 两套种子（step、step+2000）。强变换只改变光度／Cutout，不移动坐标。当前图与合成图、全部对照使用相同规则。CE 仅在弱视图上计算，因此不使用被 Cutout 遮挡图像的伪监督。若原数据 loader 默认做空间增强，须显式关闭该层并验证；不能悄悄错配 latent 网格与图像。

### 4.2 普通图像视图与旧码视图

u_{i0},u_{i1},u_{i2} 是同位置三个单位化空间表示。普通串联 GYRO 的质心为

\[
c_i^{\rm img}=\tfrac12u_{i0}+\tfrac14u_{i1}+\tfrac14u_{i2}.
\tag{9}
\]

对于来自码 Z_j 的合成图，闭环读取使用

\[
c_{jr}^{\rm mem}=\tfrac12\operatorname{sg}(Z_{jr})
+\tfrac14\Psi_{\theta,p_0}(a_1(\tilde x_j))_r
+\tfrac14\Psi_{\theta,p_0}(a_2(\tilde x_j))_r.
\tag{10}
\]

不再次单位化 c。Z 是同一坐标中的历史固定伪视图，不是把任意生成 latent 当 GYRO feature。当前真实 L、U 始终使用式(9)；闭环组仅替换旧合成行的第一视图。所有组仍计算 u₀ 供 CE／诊断，使三视图 encoder 工作量一致。

### 4.3 选锚与谱目标

\[
k_i=\arg\max_k\langle\operatorname{sg}(c_i),e_{s_i,k}\rangle,
\quad a_i=e_{s_i,k_i},\quad
y_i=(1-\beta)c_i+\beta a_i,
\tag{11}
\]

\[
\kappa(Y)=\frac{\|Y\|_*}{\sqrt{n r_*}},\quad r_*=\min(n,D),
\qquad \mathcal L_{\rm geo}=1-\kappa(Y).
\tag{12}
\]

n 是矩阵行数而非图像数。所有视图单位化、质心权重和为 1 时，0≤κ≤1。使用归一化防止矩阵行数改变直接放大 loss；不声称等梯度范数或与旧 global GYRO 数值可比。

选锚无梯度，anchor 永不训练；最近参照每次重算，**没有固定样本→anchor 身份**。source code 的固定视图合同与原 GYRO 的固定 frame 伪视图是不同对象。

### 4.4 这个共同操作究竟保留什么

对任意式(9)/(10)的加权视图组，记权重 w_v，总和为 1；有

\[
v_i=\sum_vw_v\|u_{iv}-c_i\|^2=1-\|c_i\|^2,
\quad 1-\|y_i\|^2=(1-\beta)v_i+\beta(1-\beta)\|c_i-a_i\|^2.
\tag{13}
\]

因此

\[
1-\kappa(Y)^2=(1-\beta)\bar v
+\beta(1-\beta)\overline{\|c-a\|^2}
+\frac{r_*\|Y\|_F^2-\|Y\|_*^2}{n r_*}.
\tag{14}
\]

式(10)让“码→生成输入→再编码”的分歧进入 GYRO 的同一个视图项；参照与谱项仍相互耦合。该分解是代数性质，不是旧域功能保证，也不要求最大化时每一项单独下降。

## 5. 源阶段 B：空间记忆如何真正凝缩

冻结 θ₀、p₀、ω₀，只优化 32 个 V_j。没有保存真实源图以供未来回放；本阶段源数据仍是当前合法数据。

### 5.1 训练效用是凝缩的主约束

每个外层迭代重新从 θ₀ 构造一个临时 learner，空 SGD 状态。抽 8 个码，按式(5)得到图像／软 mask，进行一个可微的 native 更新：

\[
\mathcal L_{\rm syn}(\theta,V)=\frac18\sum_{j\in B_M}
\operatorname{CE}_{\rm soft}(f_\theta(\tilde x_j),\tilde m_j),
\quad
\theta'(V)=\theta_0-\eta\nabla_\theta\mathcal L_{\rm syn}(\theta_0,V),\ \eta=10^{-3}.
\tag{15}
\]

这里只作一阶 SGD 更新、不用 momentum／wd／clip，便于精确定义和验证超梯度。它与后续正式 Nesterov SGD 不等价，这一近似须披露。保留完整二阶梯度，通过生成图像与软 mask 两条路径回传 V，不做 first-order 截断。

从源 **train-L** 取两个真实 query 图，用原生 CE：

\[
\mathcal L_{\rm utility}(V)=\mathbb E_{(x,y,v)\sim\mathcal D_0^L}
\operatorname{CE}_{v}(f_{\theta'(V)}(x),y).
\tag{16}
\]

query 是优化用训练数据，不是源验证集／测试集，也不是独立泛化证据。没有专家轨迹 matching，没有让码复制某张真实图像。所有凝缩分支的 query 与码采样顺序严格相同。

### 5.2 mask 的源阶段语义约束

防止码优化把联合 mask head 推入任意软标签区域，增加共同的源教师一致性：

\[
\mathcal L_{\rm mask}(V)=\mathbb E_j\operatorname{KL}
\bigl(\operatorname{sg}[f_{\theta_0}(\tilde x_j)]\ \|\ \tilde m_j\bigr).
\tag{17}
\]

KL 按所有生成像素平均。教师输出 stop-gradient；梯度只沿 m(V) 回传。源 θ₀ 在这里可访问，仅用于记忆形成，不在 Philips 训练期间继续调用。此项不能消除源教师错误；式(16)的真实训练 mask 反馈不可省略。

### 5.3 两种有相同参数量的凝缩方式

\[
J_{C0}=\mathcal L_{\rm utility}+0.1\mathcal L_{\rm mask},
\qquad
J_{C1}=J_{C0}+0.1\mathcal L_{\rm geo}^{\rm mem}(\theta_0,p_0,\omega_0,V;E_0).
\tag{18}
\]

C1 的最后一项使用式(10)–(12)，输入为当前抽取的 8 个码×4 位置。参考视图 sg(Z) 不接收该项的直接梯度，**但经 Z→G_x→Ψ 的生成／再编码路径接收梯度**。codec、投影器、源 native 的权重均固定，input Jacobian 保留。每次迭代刷新 sg(Z)，所以这是显式 stop-gradient 的更新合同，不把它伪称为无 stop-gradient 标量函数的普通全导数。

两个分支各作 256 次 outer Adam 更新，lr=0.01、betas=(0.9,0.999)、eps=10⁻⁸、wd=0，outer V 梯度全局 clip1。每次有且仅有式(15)的一次临时 native 更新。每次开始与结束按式(4)使用归一化码；原始 V 不逐步硬投影、不改变 Adam 状态。

为什么 C1 不退化为“随意最大核范数”：式(16)要求合成图／mask 真正使临时分割 learner 改善合法真实训练 query；生成器固定且已经通过真实源数据准备。两者仍不构成生成真实性或统计泛化的定理，失败应如实报告。

## 6. 目标阶段：完整原生分割与闭环读取

每步当前 Philips L=2、U=14，历史合成=8。几何共 24 图×4 位置=96 行，三种图像视图共 72 个 encoder 输入，D=32；不按显存修改这些值。

\[
\mathcal L_{\rm current}=\frac12\sum_{(x,y,v)\in B_L}
\operatorname{CE}_{v}(f_\theta(x),y),
\quad
\mathcal L_{\rm replay}=\frac18\sum_{j\in B_M}
\operatorname{CE}_{\rm soft}(f_\theta(\tilde x_j),\operatorname{sg}\tilde m_j),
\tag{19}
\]

\[
\mathcal L_{\rm target}=\mathcal L_{\rm current}
+\mathcal L_{\rm replay}+0.1\mathcal L_{\rm geo}.
\tag{20}
\]

R0（串联）所有行使用式(9)；R1（闭环）仅历史合成行用式(10)。来源阶段标记使当前图使用 E₁、历史合成图使用 E₀；frame 不是语义类别。当前 U 只进入几何，无隐藏 mask。

target 从相同 θ₀、空 native 动量开始；nativeLR=0.001、SGD Nesterov 0.9、矩阵 wd=0.0005／vector wd=0、全局 clip1，固定 2000 更新。p₀、G、Z、frame requires_grad=False，EMA 只诊断；输入梯度穿过固定 P 回到 native encoder。Replay CE 更新完整 native decoder/head，弥补只保护投影层的不足，但它属于普通合成监督，不能归为 GYRO 独有效果。

**这是与旧无标签 image reservoir 不同的新协议：合成软 mask 提供了额外历史语义监督。** 因此只能在同样软 mask、数量及数据访问的对照之间归因，不能与旧轮成绩直接拼成机制证据。

## 7. 保存什么、丢弃什么

|对象|Philips 训练期间保留？|原因|
|---|---|---|
|32 个最终 Z_j|是|联合优化的空间凝缩对象|
|固定 p₀、ω₀|是|码和 native 表示的共同坐标／图像-mask 解码|
|固定 E₀、E₁、采样 seed／小型合同|是|几何读取与复现|
|源原始图像／mask、历史 reservoir|否|target 训练配置与 loader 均不得引用|
|源逐图 Z(x)、患者→码映射、初始化 μ／σ|否|不是逐原图压缩缓存|
|单独源教师、临时 learner、形成期优化器|否|不属于 target 读取所需对象|
|每一步生成图像与 mask|不持久保存|按需生成，用后释放；不能积累生成历史副本|

不删除既有受保护的原始数据或旧实验归档。“不保留历史原图”指新方法记忆及后续训练访问，而非擅自销毁数据集。独立评估进程可按原评估协议读取源验证／训练评估集；不得将其连接到训练梯度或配置。

仅 Z 的 float32 大小为 32×8×8×32×4=262,144 bytes。新增 p、G、frame、码和小型解码必要 buffer 的**总保留上限为 8 MiB**，以真实参数／buffer 的字节计，不用 zip 大小。不含各组本来相同的当前 native 模型，但另列其大小；本方案 target 无额外旧 native 副本。达到上限不能自动放宽；若真实模块超过上限则设计准入失败，不在实验中临时改结构。

source 只保存准备阶段 initial/latest/final、两个凝缩分支 initial/latest/final 和必要失败现场；target 同理，不存逐步权重。磁盘上为审计留下的旧 checkpoint 不等于运行算法可在线调用，需分开报告。

## 8. 新增机制与已有内容的边界

|内容|归属与可声称程度|
|---|---|
|编码—生成—再编码闭环、紧凑 latent memory|i-CTRL／CTRL 等已有，不能声称首次|
|有限潜在码／decoder 的训练效用凝缩|既有 DD 范式，本方案不把它当新颖性|
|空间码＋联合图像/mask 生成|为密集任务所需适配，单独不是充分创新|
|固定几何 frame、多视图与核范数|原 GYRO 核心，不归为我们新提出|
|**同一个 Z 既是 G 输入、又是凝缩与持续读取中的固定伪视图**|本候选要检验的共同接口；C1 形成可读取码，R1 保留其被读取的关系|
|**码身份／空间位置的一一回路关系进入同一个 GYRO 质心**|与只比较集合协方差不同；必须通过错配对照验证|
|源准备／生成器固定／超参选择|最小版本的工程和范围决策，不包装成新原理|

一个说明区别的代数例子：旧码为 e₁、e₂，再编码正确或交换两者，集合协方差相同，所以集合级协方差闭环不区分。若 c_i=(z_i+u_i)/2、β=0，正确读取的质心矩阵核范数为 2；交换后两行均为 (e₁+e₂)/2，核范数为 1。说明成对视图关系可以识别某些集合级闭环遗漏的错误。

这不是含 β、有限网络和任务 CE 的完整系统保证；其他谱／参照变化可能补偿视图失配。相关思想也可能与既有 paired consistency 重叠，论文新颖性仍需继续核查，不能仅凭此例宣称。

## 9. 为什么不只用重建、不只用谱

codec 保真不等于 condensed training utility，因此有式(15)–(16)；谱几何不等于 organ label，因此有源 mask 与共同 replay CE；集合统计一致不等于码—图像对应，因此检验式(10)；单域合成有效不等于与当前学习相容，因此评估同一持续 native learner。

这些角色相互连接，但本方法**并非没有辅助项或完全摆脱 matching／consistency 家族**。可研究的是具体的共同码与成对谱回路是否提供超出直接串联的作用，不是给常见对齐换名字。

## 10. 原文依据

- [i-CTRL](https://arxiv.org/abs/2202.05411)：类增量的结构化统计记忆与闭环约束。
- [GYRO](https://openreview.net/forum?id=aYH3QSudNZ)：固定 task frame、未重归一化多视图质心与核范数分解。
- [CTRL](https://arxiv.org/abs/2111.06636)：单一谱率空间中的闭环反馈与可解码性问题。
- [D3](https://arxiv.org/abs/2403.00999)：分布／latent 凝缩、总字节与下游计算的评价。

以上具体网络、超参、方程组合与实验门槛是本次待检验设计，不是引用论文已经验证的算法。
