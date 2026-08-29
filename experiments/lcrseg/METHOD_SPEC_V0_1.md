# LCR-Seg 方法规范 V0.1

**文件名：** `METHOD_SPEC_V0_1.md`  
**方法工作名：** LCR-Seg  
**英文全称：** Learnability–Compatibility Routing for Semi-Supervised Cross-Site Continual Segmentation  
**版本：** 0.1  
**状态：** V0–V3 实现冻结候选  
**适用任务：** fixed-label, cross-site, semi-supervised continual semantic segmentation  
**最后更新：** 2026-08-18

---

## 0. 本规范的作用与边界

本文件固定 LCR-Seg 第一版可实现、可测试、可消融的数学定义。除非同步更新本文件、实现合同和验收测试，代码不得自行改变下列定义。

### 0.1 V0.1 的正式范围

V0.1 **只包含**：

1. 一个当前可训练分割模型；
2. 一个上一站点冻结模型；
3. 单语义锚点关系场，`K = 1`；
4. Current Learnability \(L_i\)；
5. Historical Compatibility \(C_i\)；
6. continuous assimilation–consolidation routing；
7. 可靠性加权的当前语义锚点更新。

### 0.2 V0.1 明确不包含

以下内容不是 V0.1 的组成部分，不得在首次实现时加入：

- EMA teacher 或第三个网络；
- multi-agent，`K > 1`；
- Relation-Induced Consolidation，RIC；
- EWC、MAS、SI 或梯度子空间；
- channel chunking、hard channel split；
- 全层 feature distillation；
- triplet/contrastive auxiliary loss；
- content–style VAE 或图像重建；
- diffusion/generative replay；
- SAM/foundation-model teacher；
- 历史患者图像、切片或 patch 回放；
- 站点特定 adapter、expert 或测试时 site ID。

`K > 1` 与 RIC 只能在 V0–V3 通过验收并出现明确实验需求后进入后续版本。

---

## 1. 研究问题与核心假设

### 1.1 任务设定

第 \(t\) 个站点到来时，可访问：

\[
\mathcal D_t = \mathcal D_t^l \cup \mathcal D_t^u,
\]

其中：

- \(\mathcal D_t^l = \{(x_n,y_n)\}\)：少量有标注病例；
- \(\mathcal D_t^u = \{x_n\}\)：大量无标注病例；
- 历史站点原始影像和标签不可访问；
- 所有站点共享同一标签空间 \(\mathcal C=\{0,\ldots,C-1\}\)；
- 测试时不提供站点 ID；
- 训练无标注 loader 不得读取隐藏标签。

训练阶段仅保留：

\[
M_t:\text{当前可训练模型},
\qquad
M_{t-1}:\text{上一站点冻结模型}.
\]

当前模型由上一阶段模型初始化：

\[
\theta_t \leftarrow \theta_{t-1}.
\]

第一站点不存在历史模型。

### 1.2 两类不同错误

当前站点的无标注区域存在两种相互独立的错误来源。

#### Assimilation error

当前伪标签错误，却被用于新站点学习：

\[
\widetilde y_i \neq y_i.
\]

#### Consolidation error

历史模型在新站点上因域偏移而预测错误，却被当作历史保持目标：

\[
\widehat y_i^{old} \neq y_i.
\]

现有 SSL 主要处理前者，传统 CL 往往默认历史 teacher 有效。LCR-Seg 的核心假设是：

\[
\boxed{
\text{当前伪标签可靠性与历史知识有效性必须分别估计。}
}
\]

### 1.3 统一对象

LCR-Seg 不是将一个 SSL loss 与一个 CL regularizer 并列相加，而是建立统一的：

\[
\boxed{
\text{Dense Temporal Semantic Relation Field}
}
\]

即像素或体素特征与类别语义锚点之间的稠密关系分布。该关系场同时用于：

1. 验证和恢复当前伪标签；
2. 估计当前可学习性；
3. 判断历史模型在当前区域是否仍然有效；
4. 选择性执行历史关系巩固；
5. 更新下一阶段使用的紧凑语义记忆。

核心闭环为：

\[
\text{historical relations}
\rightarrow
\text{calibrate current SSL}
\rightarrow
\text{reliable current semantics}
\rightarrow
\text{update future memory}.
\]

---

## 2. 设计依据与非复制性说明

本方法吸收的是若干工作的构造原则，而不是直接拼接其原始模块。

- **DC²T**：SSL 与 CL 应围绕共同表示协同，而非使用彼此独立的现成组件。LCR-Seg 将共同对象从 content representation 改为 dense semantic relation field。
- **USP**：伪标签中间关系可继续服务于无标注知识蒸馏。LCR-Seg 将图像级 class mean 扩展为像素级关系场，并增加历史兼容性。
- **LDKA**：低间隔像素具有较高更新敏感性和较小稳定半径。LCR-Seg 只吸收低间隔诊断与渐进参与思想，不复制 SKD/MAD。
- **STAR/LAG/IDEC**：历史保持应具有空间和语义选择性，不应无差别约束全部特征。LCR-Seg 使用 compatibility-conditioned relation consolidation，而非复制 prototype replay、LRP 或 triplet loss。
- **SMG-Learning**：多个训练目标可能产生梯度冲突。V0.1 只记录并分析梯度余弦，不额外加入梯度对齐模块。

---

## 3. 模型结构

### 3.1 分割模型

每个模型由共享编码器—解码器、分割头和稠密投影头构成：

\[
M(x) = \left(H(F(x)), P(F_r(x))\right),
\]

其中：

- \(F\)：分割主干；
- \(H\)：输出 full-resolution logits；
- \(F_r\)：decoder 的 \(1/4\) 分辨率特征；
- \(P\)：轻量 projection head；
- \(z=P(F_r(x))\)：L2 归一化后的 dense relation feature。

模型输出：

\[
s \in \mathbb R^{B\times C\times H\times W},
\]

\[
z \in \mathbb R^{B\times D\times h\times w},
\qquad
h=H/4,\ w=W/4.
\]

默认：

\[
D=128.
\]

MRI 与眼底都先实现 2D 网络。3D 是后续 backbone robustness，不改变本规范的语义定义。

### 3.2 两模型状态

站点 \(t>1\)：

- `current_model`：训练模式，可反向传播；
- `old_model`：冻结、`eval()`、无梯度；
- 二者结构完全一致，均包含 projection head。

站点开始：

\[
M_t \leftarrow M_{t-1}.
\]

站点结束：

\[
M_t \rightarrow M_{t-1}^{next}.
\]

---

## 4. 语义锚点与关系场

### 4.1 什么是 semantic anchor

V0.1 中，每个类别维护一个语义锚点：

\[
A^v=\{a_c^v\}_{c=0}^{C-1},
\qquad
a_c^v\in\mathbb R^D,
\qquad
\|a_c^v\|_2=1,
\]

其中 \(v\in\{cur,old\}\)。

语义锚点是类别在 projection feature space 中的代表性方向，不是：

- 额外网络；
- 可训练参数；
- reinforcement-learning agent；
- 原始患者数据。

V0.1 正文统一使用 **semantic anchor**，不单独使用容易引起歧义的 “agent”。

### 4.2 当前与历史锚点

必须维护两个独立 bank：

\[
A^{cur},\qquad A^{old}.
\]

- `old_anchor_bank`：随 `old_model` 冻结；
- `current_anchor_bank`：站点开始时从 old bank 初始化，训练中更新；
- 第一站点无 old bank；
- 不得只维护一个同时承担当前与历史功能的可变 bank。

### 4.3 锚点初始化

#### 第一站点

1. 先进行 `anchor_bootstrap_steps` 的 supervised/baseline SSL warm-up；
2. 用当前模型提取有标注像素的 relation feature；
3. 按类别计算 masked mean；
4. L2 归一化，初始化当前锚点。

#### 后续站点

\[
A^{cur}\leftarrow A^{old}.
\]

然后由当前站点的标注语义和可靠无标注语义逐步更新。

### 4.4 背景锚点

背景类别仍参与关系分布，但采用以下限制：

- 标注背景只从远离前景边界的区域采样；
- 无标注背景只使用高可学习性且 segmentation/relation 一致的像素；
- 每 batch 每类使用相同上限的像素数，避免背景数量主导；
- V0.1 不允许为背景单独引入额外损失。

若单背景锚点在三数据集上明显不足，作为 V4 multi-agent 的实验依据。

### 4.5 关系分数与关系概率

给定归一化特征 \(z_i^v\) 和锚点 \(a_c^v\)：

\[
r_{i,c}^v
=
\frac{\cos(z_i^v,a_c^v)}{T_r},
\]

\[
q_{i,c}^v
=
\frac{\exp(r_{i,c}^v)}
{\sum_{k=0}^{C-1}\exp(r_{i,k}^v)}.
\]

其中：

- \(T_r>0\)：relation temperature；
- \(q_i^v\in\Delta^{C-1}\)：pixel-to-anchor relation distribution。

关系预测：

\[
\widehat y_i^{rel,v}
=
\arg\max_c q_{i,c}^v.
\]

### 4.6 监督关系损失

为了让 projection head 形成可用的类别关系空间，有标注数据必须训练 relation classifier：

\[
\mathcal L_{anchor}^{sup}
=
\frac{1}{|\Omega_l|}
\sum_{i\in\Omega_l}
\operatorname{CE}(q_i^{cur},y_i^\downarrow),
\]

其中 \(y^\downarrow\) 使用 nearest-neighbor 下采样到 \(h\times w\)。

监督总损失定义为：

\[
\mathcal L_{sup}
=
\mathcal L_{CE}^{seg}
+
\lambda_{dice}\mathcal L_{Dice}^{seg}
+
\lambda_{anchor}\mathcal L_{anchor}^{sup}.
\]

`L_anchor_sup` 属于监督损失内部，不单独形成第四条方法主线。

---

## 5. Weak-to-Strong 视图规则

对每个无标注样本构造：

\[
x^w,\qquad x^s.
\]

V0.1 强制：

1. weak 与 strong 使用**同一几何变换**；
2. strong 只额外加入强度、颜色、噪声、模糊、bias field 或 cutout；
3. 若使用 cutout，必须返回 `strong_valid_mask`；
4. weak/strong relation map 必须空间对齐；
5. 不允许 V0.1 使用未记录形变的独立随机旋转或 elastic transform。

当前 weak prediction 全部 `stop-gradient`，用于伪标签、\(L_i\) 和 \(C_i\) 的计算。

---

## 6. Current Learnability \(L_i\)

### 6.1 含义

\(L_i\in[0,1]\) 表示：

> 当前模型是否已经对无标注位置 \(i\) 形成足够稳定、语义一致且适合参与优化的判断。

它不是可训练网络输出，而是由当前模型的 detached 预测与关系场计算。

### 6.2 当前 segmentation prediction

对 weak view：

\[
p_i^w=\operatorname{softmax}(s_i^w).
\]

定义 segmentation top-1 与 top-2：

\[
c_{1,i}^{seg}=\arg\max_c p_{i,c}^w,
\]

\[
p_{1,i}^{seg},p_{2,i}^{seg}
=
\operatorname{Top2}(p_i^w).
\]

logit margin：

\[
m_i=s_{i,c_{1,i}^{seg}}^w-s_{i,c_{2,i}^{seg}}^w.
\]

### 6.3 稳定性代理与渐进参与

使用概率近似 margin-direction sensitivity：

\[
\kappa_i
=
p_{1,i}^{seg}
+
p_{2,i}^{seg}
-
\left(p_{1,i}^{seg}-p_{2,i}^{seg}\right)^2.
\]

定义 robust progress index：

\[
R_i
=
\frac{|m_i|}
{\kappa_i+\epsilon}.
\]

按当前预测类别，在 mini-batch 内计算 percentile rank：

\[
u_i
=
F_{c_{1,i}^{seg}}(R_i)
\in[0,1].
\]

若某类别有效像素少于 `min_rank_pixels`，退化为 batch-global rank。

训练进度：

\[
\rho=\frac{\text{current step within site}}
{\text{total steps within site}}
\in[0,1].
\]

课程阈值：

\[
\tau_p(\rho)
=
(1-\rho)\tau_{p,start}
+
\rho\tau_{p,end},
\]

默认：

\[
\tau_{p,start}=0.8,\qquad \tau_{p,end}=0.2.
\]

渐进权重：

\[
w_i^{prog}
=
\sigma
\left(
\frac{u_i-\tau_p(\rho)}
{T_p}
\right).
\]

这实现训练早期强调高稳定区域，后期逐渐接纳低间隔区域。

### 6.4 relation evidence

当前 relation top-1 与 top-2：

\[
c_{1,i}^{rel}=\arg\max_c q_{i,c}^{cur},
\]

\[
q_{1,i}^{cur},q_{2,i}^{cur}
=
\operatorname{Top2}(q_i^{cur}).
\]

relation margin：

\[
\Delta_i^{cur}
=
q_{1,i}^{cur}-q_{2,i}^{cur}.
\]

relation gate：

\[
w_i^{rel}
=
\sigma
\left(
\frac{\Delta_i^{cur}-\delta_{rel}}
{T_{rel}}
\right).
\]

### 6.5 空间一致性

先得到候选语义标签 \(\widetilde y_i\)，然后在 \(3\times3\) relation grid 邻域计算：

\[
a_i^{sp}
=
\frac{
\sum_{j\in\mathcal N(i)}
v_j\,
\mathbb I[\widetilde y_j=\widetilde y_i]
}{
\sum_{j\in\mathcal N(i)}v_j+\epsilon
},
\]

其中 \(v_j\) 是候选标签有效性。

为了不系统性丢弃边界：

\[
w_i^{sp}
=
s_{floor}
+
(1-s_{floor})a_i^{sp},
\]

默认：

\[
s_{floor}=0.25.
\]

### 6.6 空间化 divide-and-conquer 伪标签

#### 分支 A：classifier-easy

若：

\[
\max_c p_{i,c}^w \ge \tau_{cls},
\]

且：

\[
c_{1,i}^{seg}=c_{1,i}^{rel},
\]

则：

\[
\widetilde y_i=c_{1,i}^{seg},
\qquad
b_i^{src}=\texttt{classifier}.
\]

#### 分支 B：anchor-recoverable

若 classifier-easy 不成立，但：

\[
q_{1,i}^{cur}\ge\tau_{anchor},
\]

\[
\Delta_i^{cur}\ge\delta_{anchor},
\]

且 preliminary spatial agreement 不低于 \(\tau_{sp}\)，则：

\[
\widetilde y_i=c_{1,i}^{rel},
\qquad
b_i^{src}=\texttt{anchor}.
\]

#### 分支 C：deferred

其余位置：

\[
\widetilde y_i=\varnothing,
\qquad
v_i=0.
\]

候选标签仅用于 relation-resolution 的 unlabeled supervision。上采样至 segmentation logits 时使用 nearest-neighbor。

### 6.7 source confidence

若来自 classifier：

\[
w_i^{src}
=
\sigma
\left(
\frac{\max p_i^w-\tau_{cls}}
{T_{cls}}
\right).
\]

若来自 anchor：

\[
w_i^{src}
=
\sigma
\left(
\frac{q_{1,i}^{cur}-\tau_{anchor}}
{T_{anchor}}
\right).
\]

### 6.8 最终可学习性

\[
\boxed{
L_i
=
v_i\,
w_i^{prog}\,
w_i^{rel}\,
w_i^{sp}\,
w_i^{src}
}
\]

实现时：

\[
L_i\leftarrow\operatorname{stopgrad}(L_i).
\]

任何情况下不得允许网络通过主动降低 \(L_i\) 来减小训练损失。

---

## 7. Historical Compatibility \(C_i\)

### 7.1 含义

\(C_i\in[0,1]\) 表示：

> 上一阶段模型在当前站点位置 \(i\) 的历史语义关系是否仍然明确、空间一致且可作为保持目标。

历史模型不被默认视为始终正确。

### 7.2 当前与历史关系

在 weak view 上分别得到：

\[
q_i^{cur,w},
\qquad
q_i^{old,w}.
\]

二者在不同模型各自的 feature/anchor space 中计算，但都投影到同一类别概率单纯形，因此可以比较。

### 7.3 历史模型自身置信

历史 relation margin：

\[
\Delta_i^{old}
=
q_{1,i}^{old}-q_{2,i}^{old}.
\]

历史清晰度：

\[
w_i^{old}
=
\sigma
\left(
\frac{\Delta_i^{old}-\delta_{old}}
{T_{old}}
\right).
\]

### 7.4 时序关系一致性

Jensen–Shannon divergence：

\[
d_i^{JS}
=
\operatorname{JS}
\left(
q_i^{cur,w},
q_i^{old,w}
\right).
\]

关系一致性：

\[
w_i^{JS}
=
\exp
\left(
-\frac{d_i^{JS}}{T_{JS}}
\right).
\]

主类别一致性：

\[
w_i^{agree}
=
\mathbb I
\left[
\arg\max q_i^{cur,w}
=
\arg\max q_i^{old,w}
\right].
\]

### 7.5 历史空间一致性

基于历史 relation prediction 计算与 6.5 相同的邻域一致性：

\[
w_i^{old,sp}
=
s_{floor}
+
(1-s_{floor})a_i^{old,sp}.
\]

### 7.6 最终历史兼容性

\[
\boxed{
C_i
=
w_i^{old}
w_i^{JS}
w_i^{agree}
w_i^{old,sp}
}
\]

实现时：

\[
C_i\leftarrow\operatorname{stopgrad}(C_i).
\]

第一站点：

\[
C_i=0.
\]

---

## 8. Learnability–Compatibility Routing

### 8.1 解释性四象限

四象限只用于分析，不用于训练中的硬路由。

| \(L_i\) | \(C_i\) | 状态 | 含义 |
|---|---|---|---|
| 高 | 高 | stable consolidation | 当前可靠且历史可解释 |
| 高 | 低 | domain assimilation | 当前可靠但历史失效 |
| 低 | 高 | historical recovery | 当前未学稳但历史关系清晰 |
| 低 | 低 | deferred | 两者均不可靠 |

### 8.2 训练采用连续权重

#### 当前域吸收

对 strong view segmentation logits：

\[
\mathcal L_{assim}
=
\frac{
\sum_i
L_i\,
\operatorname{CE}
\left(
p_i^{cur,s},
\widetilde y_i
\right)
}{
\sum_iL_i+\epsilon
}.
\]

要求：

- 只对 \(v_i=1\) 的候选标签计算；
- strong cutout 区域由 `strong_valid_mask` 排除；
- unlabeled loss 不使用 Dice，以避免噪声区域全局耦合。

#### 历史关系巩固

当前 strong relation distribution：

\[
q_i^{cur,s}.
\]

历史关系 teacher：

\[
\operatorname{stopgrad}(q_i^{old,w}).
\]

关系巩固：

\[
\mathcal L_{rel}
=
T_d^2
\frac{
\sum_i
C_i\,
\operatorname{KL}
\left(
\operatorname{stopgrad}(q_i^{old,w})
\|
q_i^{cur,s}
\right)
}{
\sum_iC_i+\epsilon
}.
\]

该损失保持的是类别关系，而不是绝对 feature coordinate。

### 8.3 总目标

\[
\boxed{
\mathcal L
=
\mathcal L_{sup}
+
\lambda_a(\rho)\mathcal L_{assim}
+
\lambda_c(\rho)\mathcal L_{rel}
}
\]

其中：

- 第一站点 \(\mathcal L_{rel}=0\)；
- \(\lambda_a\) 在 anchor bootstrap 后线性 ramp-up；
- \(\lambda_c\) 在增量站点开始后线性 ramp-up；
- 所有方法使用相同 segmentation backbone、训练步数和 supervised loss。

---

## 9. 当前锚点更新与长期语义记忆

### 9.1 锚点不是 optimizer 参数

锚点必须注册为 buffer，通过 no-grad 统计更新。

### 9.2 标注像素权重

有标注像素：

\[
w_i^{lab}=1.
\]

标签下采样至 relation resolution 后参与每类 masked mean。

### 9.3 无标注记忆权重

定义：

\[
w_i^{mem}
=
L_i
\left[
\eta+(1-\eta)C_i
\right],
\qquad
0<\eta<1.
\]

默认：

\[
\eta=0.25.
\]

含义：

- 高 \(L_i\)、高 \(C_i\)：高权重巩固共享语义；
- 高 \(L_i\)、低 \(C_i\)：以较低权重写入新域可靠模式；
- 低 \(L_i\)：不写入长期记忆。

### 9.4 class-balanced batch center

对类别 \(c\)，从 labeled 和可靠 unlabeled 中分别采样最多 `max_anchor_pixels_per_class` 个位置，得到加权中心：

\[
\bar z_c
=
\frac{
\sum_i w_{i,c}z_i
}{
\sum_iw_{i,c}+\epsilon
}.
\]

若当前 batch 类别 \(c\) 无有效像素，则不更新该类锚点。

### 9.5 EMA 更新

\[
a_c^{cur}
\leftarrow
\operatorname{Norm}
\left[
\mu a_c^{cur}
+
(1-\mu)\bar z_c
\right].
\]

默认：

\[
\mu=0.99.
\]

同时更新：

- `anchor_count[c]`；
- labeled/unlabeled support；
- source-site support；
- update step。

### 9.6 阶段结束

保存：

- 当前模型及 projection head；
- 当前锚点；
- 每类 support counts；
- 配置和版本。

随后：

\[
A^{cur}\rightarrow A^{old,next}.
\]

V0.1 不保存任何患者图像或 feature map。

---

## 10. 第一站点与增量站点行为

### 10.1 第一站点

1. 无 old model；
2. 无 compatibility；
3. 无 relation consolidation；
4. 先执行 bootstrap；
5. anchor 有效后启用 learnability-based pseudo-labeling；
6. 总损失为：

\[
\mathcal L
=
\mathcal L_{sup}
+
\lambda_a\mathcal L_{assim}.
\]

### 10.2 后续站点

1. 从上一阶段 checkpoint 初始化 current model；
2. 冻结 old model；
3. 冻结 old anchors；
4. current anchors 从 old anchors 复制；
5. 同时计算 \(L_i\) 和 \(C_i\)；
6. 执行 assimilation 与 relation consolidation；
7. 更新 current anchors；
8. 站点结束后冻结新的模型与 anchors。

---

## 11. 推理

V0.1 正式推理只使用：

\[
\widehat y
=
\arg\max_c H(F(x))_c.
\]

即只使用当前分割模型的 segmentation head。

默认不使用：

- historical model；
- anchor-based test-time replacement；
- site classifier；
- test-time adaptation；
- ensemble。

relation head 和 anchor bank仅用于训练与分析。若未来测试时使用 relation fusion，必须作为独立扩展和消融，不得静默改变主结果。

---

## 12. 分析输出

每个训练 step 或固定间隔记录：

- supervised、assimilation、relation loss；
- pseudo-label source counts；
- pseudo-label coverage；
- \(L_i\) 均值、分位数、直方图；
- \(C_i\) 均值、分位数、直方图；
-四象限像素比例；
- relation JS；
- anchor norm 与 drift；
- anchor support counts；
- \(\cos(g_{assim},g_{rel})\) 的抽样统计。

利用独立 diagnostics manifest 的 hidden GT，离线计算：

\[
P(\widetilde y_i=y_i\mid L_i\text{ bin}),
\]

\[
P(\widehat y_i^{old}=y_i\mid C_i\text{ bin}).
\]

hidden GT 不得进入训练进程。

---

## 13. 默认超参数

以下是 V0.1 的统一起点，不代表最终论文最优值。

```yaml
method:
  name: lcrseg_v0_1
  relation_dim: 128
  relation_temperature: 0.1
  distill_temperature: 0.5

  anchor:
    k: 1
    momentum: 0.99
    max_pixels_per_class: 2048
    bootstrap_steps: 500
    min_support_pixels: 64
    background_boundary_exclusion: 3
    memory_eta: 0.25

  pseudo_label:
    tau_cls: 0.95
    tau_anchor: 0.80
    delta_anchor: 0.15
    tau_spatial: 0.60
    temperature_cls: 0.05
    temperature_anchor: 0.05

  learnability:
    rank_start: 0.80
    rank_end: 0.20
    rank_temperature: 0.10
    relation_margin_center: 0.10
    relation_margin_temperature: 0.05
    spatial_floor: 0.25
    min_rank_pixels: 128

  compatibility:
    old_margin_center: 0.10
    old_margin_temperature: 0.05
    js_temperature: 0.20
    spatial_floor: 0.25

  loss:
    lambda_dice: 1.0
    lambda_anchor_sup: 0.1
    lambda_assim: 1.0
    lambda_relation: 1.0
    assim_ramp_steps: 1000
    relation_ramp_steps: 1000
```

调参范围必须在 development protocol 中预先规定，不能按测试站点单独调节。

---

## 14. 关键消融

### 14.1 方法递进

| ID | Relation field | \(L_i\) | Relation KD | \(C_i\) routing | Anchor update |
|---|---:|---:|---:|---:|---:|
| A0 |  |  |  |  |  |
| A1 | ✓ |  |  |  | labeled only |
| A2 | ✓ | ✓ |  |  | labeled + \(L_i\) |
| A3 | ✓ | ✓ | uniform |  | labeled + \(L_i\) |
| A4 | ✓ | ✓ | ✓ | ✓ | labeled + \(L_i,C_i\) |

### 14.2 必须对照

1. confidence-only vs \(L_i\)；
2. fixed threshold vs progressive participation；
3. segmentation confidence vs relation evidence；
4. uniform relation KD vs compatibility-conditioned KD；
5. logit KD vs relation KD；
6. feature L2 vs relation KL；
7. no anchor update vs reliability-weighted anchor update；
8. continuous weights vs hard four-quadrant routing；
9. sequential SSL vs LCR-Seg；
10. LCR-Seg V0.1 vs DC²T。

---

## 15. 延后模块的进入条件

### 15.1 Multi-agent V4

仅当以下现象成立时进入：

- M&Ms 或 fundus 中同类 relation feature 明显多峰；
- `K=1` 的 relation accuracy 或 calibration 成为瓶颈；
- 背景锚点过度分散；
- `K=2/4` 在至少两个数据集稳定改善，而非单 seed 偶然收益。

### 15.2 RIC V5

仅当以下现象成立时进入：

- prostate 长序列早期站点仍显著遗忘；
- 当前站点无法覆盖历史外观时 relation KD 失效；
- V0.1 已通过 \(L_i,C_i\) calibration；
- 参数保护重要性必须从可靠 relation field 派生，不得直接复制普通 EWC。

---

## 16. 方法贡献的推荐表述

V0.1 的贡献应围绕一个主线组织，而不是逐项枚举来源组件。

1. **双重可靠性诊断。** 区分当前伪标签是否可学习与历史知识在新域上是否仍然有效。
2. **统一时序关系场。** 使用同一 pixel-to-anchor relation field 同时支持伪标签恢复、历史兼容性判断和关系巩固。
3. **连续吸收—巩固路由。** 当前可靠但历史不兼容的区域释放可塑性；历史兼容区域进行选择性关系保持。
4. **无回放语义闭环。** 可靠无标注语义更新下一阶段 anchor memory，使 CL 校准 SSL，SSL 决定 CL 保存什么。

一句话总结：

> LCR-Seg learns currently reliable semantics while consolidating only historically valid relations, using one shared dense semantic relation field to couple semi-supervised learning and continual knowledge preservation.

---

## 17. 版本冻结条件

当以下条件同时满足时，V0.1 可以冻结并进入三数据集正式实验：

- `METHOD_ACCEPTANCE_TESTS_V0_1.md` 中所有 hard engineering gates 通过；
- \(L_i\) 与伪标签正确率具有明确正相关；
- \(C_i\) 与历史模型正确率具有明确正相关；
- compatibility routing 相比 uniform KD 改善稳定性—可塑性权衡；
- checkpoint 可完整恢复模型和 anchor 状态；
- hidden label leakage 测试通过；
- 第一站点、增量站点及空类别情况均无 NaN/Inf；
- 代码实现与本文件公式逐项对应。
