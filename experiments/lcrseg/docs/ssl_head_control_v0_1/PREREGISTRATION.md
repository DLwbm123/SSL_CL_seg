# SSL_CL：分类头对照与 PAS 等数量归因实验 V0.1

> 本文件是可交给 Codex 的有限执行计划。目标依次为：有效性、机制可解释性、再考虑创新性与 SOTA。它不是旧 `SSL_FOUNDATION_NOT_ESTABLISHED` 的重判，不是完整 JASCL/U-Net 论文复现，也不预设普通分类头一定更好。

## 0. 本轮决定与授权范围

下一轮只做两件相互独立的事：

1. **只读诊断**：在已有最终 PAS 模型上，比较 PAS 与“接受相同数量、相同预测类别构成”的 confidence 排序和随机筛选，检验原型筛选是否提供额外的正确性信息。这一诊断不训练模型，也不产生新的可部署方法。
2. **实际训练**：同一 UNet 主体、同一监督目标、同一 Mean Teacher/PAS 规则，只将分类头替换为同卷积几何的非归一化线性头，训练 `LIN_SUP`、`LIN_MT_CONF`、`LIN_MT_PAS`。用已经完成的 G0/G1 结果作固定参照。达到预定条件后自动完成两个新的优化种子，不在每个阶段结束时等待确认。

不增加多原型、可靠性回归、KD、冲突删除、SCD投影、历史模型路由、类别特殊规则或新损失。不得把0.010的旧准入要求改成0.005以启动旧研究未获准的seed12/13。

| 项目 | 固定值 |
|---|---|
| 仓库 | `DLwbm123/SSL_CL_seg` |
| 新分支起点 | `405337e71d22aef011a041664fbc1b4b72b60408`，这是已公开报告的完整提交，不使用不确定的最新HEAD |
| 新分支 | `codex/ssl-classifier-head-control-v0-1` |
| 新代码 | `experiments/lcrseg/ssl_head_control_v0_1/` |
| 新测试 | `experiments/lcrseg/tests/ssl_head_control_v0_1/` |
| 新报告 | `experiments/lcrseg/docs/ssl_head_control_v0_1/` |
| 原基础实现 | `d9542bd6f7b4acfe351bdaa8c8107bb875d40717` 下 `ssl_foundation_v0_1` |
| 数据划分 | `data_split_seed=0`；不改变病例、患者角色或标注预算 |
| 筛查优化种子 | 11；与旧实验作配对开发比较，不称为新数据 |
| 条件复核优化种子 | 21、22；不是新患者划分，不读取其结果挑选种子 |
| 单域任务 | RIM-ONE-r3、Drishti-GS分别从随机初始化独立训练，绝不串联 |
| 更新预算 | 首轮15,900；条件复核最多42,400；本轮正式更新上限58,300 |
| 模型数 | 每任务最多学生＋当前EMA两份完整模型；最终仅部署学生 |

本次计划被用户转交并要求执行时，即授权下面完整的P0–P2条件流程，不授权任何额外调参、测试集评价、持续学习阶段或合并main。

---

## 1. 已有事实、来源中的机制与本轮假设

### 1.1 已有事实 [R1–R3]

- 上轮12个任务已完成，31,800次更新，终态 `SSL_FOUNDATION_NOT_ESTABLISHED`。
- `MT_PAS_G1` 的两域平均学生Dice为0.7123316875900498；相对同GAS的SUP平均增益为0.00696713556298878，未达0.010。
- 同GAS的PAS相比CONF有正面组件差值；但同网络上PAS过滤后的前景precision并非均匀提高。
- G0只关闭GAS随机采样，**仍有特征归一化、权重归一化、固定logit倍率10以及原有卷积/插值几何**。它不是普通线性头。
- 上轮只验证了这一特定头和配方的SSL效果，没有比较同主体上的原始线性卷积输出。
- 旧报告的正式尝试累计为131,508；它不是所有早期项目实验的完整算力总计。

### 1.2 JASCL原文及源码 [R4–R5]

附件第5页公式(3)–(5)定义了逐图像类别原型、confidence和cosine共同验证，以及学生/教师有效集交集上的概率MSE。Appendix B第28–30页将双准则精度优势与判别性等假设联系起来；不能视为任意新数据上原型筛选必然优越的无条件保证。

固定官方分类头源码的实际输出，在GAS关闭时相当于：

\[
z_{\mathrm{norm}}=10\,\operatorname{Conv}\bigl(\operatorname{normalize}(F,\mathrm{dim}=1),\operatorname{normalize}(W,\mathrm{dim}=1)\bigr).
\]

这里是对源码的描述，不将三维卷积核的整个运算误称为一个标准余弦点积。

### 1.3 本轮仅有的训练假设

**同一特征主干上的分类头参数化，可能影响监督优化、概率分布和半监督目标的可用性。**

这不等于已经证明归一化头错误、温度10过大、背景梯度占优或伪标签校准是唯一根因。把输出倍率、特征归一化、权重归一化一起移除，是一个明确的“头参数化替换”因素；即使有效，也不能把效果单独归因于温度。

关于概率校准的文献 [R6] 只提供“置信度数值与真实正确率应分开测量”的外部背景；它不证明本项目改头后一定获益。本轮不拟合温度、不做验证集校准。

---

## 2. P0：不训练的PAS等数量诊断

### 2.1 固定输入与成本

只使用旧 `MT_PAS_G0` 和 `MT_PAS_G1` 在两个单域、优化种子11、epoch100的最终学生、EMA和**实际训练原型库**。实际原型最后刷新于epoch96，不能偷偷换成新的原型库。

优先复用包含所需像素级字段的既有缓存；若旧缓存只有汇总而没有字段，明确允许在限定val上重做下述诊断前向。不得伪称已存在未验证的缓存。

每例：1次clean双模型预测＋4次原训练式噪声/分类头抽样双模型预测。两个域共65例、两个PAS臂，新增诊断最多：

\[
65\times2\times2\times5=1,300
\]

次sample-model前向。按batch1执行便于计数；缓存复用可减少实际调用。无optimizer更新，无新的原型拟合，无训练图像/GT补采样。病例数由manifest核验；1300是这一固定人群上的上限，不是可自由消费的推理额度。

每次只加载一个旧学生/EMA对。若文件缺失，不选择别的epoch/种子代替；记录该诊断缺失。缺失本诊断不自动否定新训练结果，但限制PAS机制解释；新训练的核心输入或配对参照无法核验时才阻塞对应正式任务。

### 2.2 掩码必须在不接触GT的API中形成

设学生/教师概率为 \(p_s,p_t\)，预测类别为 \(y_s,y_t\)，当前几何支持为G。复用原严格阈值：

\[
C=G\cap\{\max p_s>0.7\}\cap\{\max p_t>0.7\}.
\]

\[
M=C\cap\{s_s>0.7\}\cap\{s_t>0.7\}\cap\text{prototype support}.
\]

不额外要求 \(y_s=y_t\)。原协议没有这个条件；本轮不能偷偷加入。

对每图、每次draw、每个**教师预测类别**c，计算

\[
K_c=|M\cap\{y_t=c\}|.
\]

形成三个掩码：

- `PAS_NATIVE`：原M。
- `CONF_TOPK_MATCHED`：在 \(C\cap\{y_t=c\}\) 中按 \(\min(\max p_s,\max p_t)\) 递减取相同的 \(K_c\) 个像素。
- `RANDOM_MATCHED`：在同一集合中用固定、独立RNG取相同 \(K_c\) 个像素，固定4次排列；先平均这4次，再平均预测draw，最后按患者聚合。

精确同分用展平像素索引升序打破；随机种子由单独的`mask_audit`命名空间确定，不进入训练随机流。若K=0，三者均为空；若K等于候选数，三者均取全。

GT不进入掩码、K、排序、随机选择或原型更新。先形成掩码及摘要，再由独立evaluator读val GT。

**几何等数量不等于含ignore GT后的计分数量必定相同。**GT=255只用于计分排除，不用于重新分配K。逐行报告有效计分数量是否仍相等；不相等的行不能冒充严格有效支持等数量见证。不得为此访问train-U隐藏GT。

### 2.3 固定报告内容

按域、类别、模型标签来源、clean/training-like分别报告：precision、预测覆盖率、图像覆盖率、accepted-correct recall、接受正确/错误计数、有效患者数及undefined分母。

teacher argmax正确性为主；student argmax诊断另列。教师和学生预测不一致的接受比例另列。

额外计算条件筛选比：

\[
\mathrm{LR}_{sim\mid C}
=\frac{P(M\mid \text{teacher正确},C)}{P(M\mid\text{teacher错误},C)}.
\]

这是在同一C条件下的描述性统计，不套用条件独立假设。零分母分别记`undefined`或有明确定义的无穷比，不加伪计数制造有限值。不能由某个全局比值推出逐病例可靠或部署风险保证。

**P0不用于选择头、候选、阈值、epoch或训练权重，不作为P1的效果准入门槛。**它的用途是解释“原型信息”与“减少监督像素/改变构成”是否能够区分。若PAS不优于等数量confidence，这一结果应直接保留，不再加相似度阈值搜索。

---

## 3. P1：只新增三个训练臂

| 新臂 | 分类头 | 监督目标 | 无标注目标 |
|---|---|---|---|
| LIN_SUP | 原始线性3×3卷积，同几何 | CE | 无；不打开U图像 |
| LIN_MT_CONF | 同上 | CE | 当前EMA＋confidence交集概率MSE |
| LIN_MT_PAS | 同上 | CE | 当前EMA＋原PAS交集概率MSE |

旧六臂的最终数值只读复用；本轮不重训旧seed11，不使用旧训练完成的网络初始化新臂。

解释时至少报告以下两个不同的差值：

\[
\Delta_{SSL}^{LIN}(a)=Q_{LIN\_MT_a}-Q_{LIN\_SUP},
\]

\[
\Delta_{head}^{SUP}=Q_{LIN\_SUP}-Q_{SUP\_G0}.
\]

并描述：

\[
I_a=(Q_{LIN\_MT_a}-Q_{LIN\_SUP})-(Q_{MT_a\_G0}-Q_{SUP\_G0}).
\]

这里I是同一初始化与预算下完整训练配方的交互差；不是唯一机制的因果证明。G1作为更强监督/既有PAS参照，而不是将开启/关闭GAS混入主head替换对比。

### 3.1 线性头精确定义

固定为：

\[
z_{LIN}=\operatorname{Conv2D}(F,W;\;kernel=3,\;padding=0,\;stride=1,\;bias=None).
\]

保留旧wrapper将低分辨率logits用`bilinear, align_corners=True`插值到原输入的流程。**不加padding、不换1×1、不添加bias、不改输出温度、不加入dropout。**

- 移除输出计算中的feature normalization、weight normalization和固定×10。
- 关闭GAS采样，不更新GAS敏感度；不做无用途的监督GAS autograd。
- 初始化W使用同domain/seed下原构造器生成的`mu.weight`，主干初始化完全相同；不要重新随机初始化头来引入第二因素。
- 为保持权重字段和初始化可比，可保留原`sigma`与`grad_update`惰性字段，必须报告其字节数、冻结/无梯度情况。`grad_update`排除Adam；未进入forward的`sigma`必须无梯度、无实际更新。
- 原官方构造器可能固定分类头种子1024。必须如实保留并报告，不能声称不同优化种子中每个张量初始化都不同。
- 新代码明确持久化`head_mode=raw_linear3_same_geometry`。单靠state_dict字段相同不足以说明模型定义相同。
- 本配置不是完整原版U-Net，其原型、归一化主干和输出几何仍沿用项目；只称为同几何线性头对照。

必须新建命名空间，复用旧纯函数，不修改旧核心文件或松开旧训练入口的lock来启动新任务。新runner显式接受新臂和种子，不能通过改写历史终态绕过权限。

### 3.2 固定数据与初始化

| 单域 | train_labeled | train_U | val | 每epoch更新 | 100epoch更新 |
|---|---:|---:|---:|---:|---:|
| RIM-ONE-r3 | 16 | 63 | 40 | 32 | 3200 |
| Drishti-GS | 10 | 41 | 25 | 21 | 2100 |

每个域从随机初始化开始；不加载REFUGE、前一域、旧专家、Foundation epoch20/100或预训练大模型。

三个新臂在同域/同种子具有完全相同的初始主干和mu权重；新臂初始化应与旧G0同域/同种子的相应参数hash吻合。原始头与归一化头初始logits不需要相同。

warm-up结束，三个LIN臂的student、EMA、optimizer和已消费的标注顺序应吻合；不同头之间warm-up不要求吻合。

数据读取继续使用已有canonical binding `DATA/h5/v1/relative`。train-U accessor不暴露label path、label hash或label tensor。SUP可读取U数量以匹配步数，但不打开U图像。

### 3.3 固定训练配置

完全继承Foundation V0.1在新单域任务中已使用的配置，除head计算外不再改变：

- UNet通道16/32/64/128、GroupNorm、原decoder；全部实际可学习权重训练。
- 输入RGB 3×384×384；标注0/1/2，ignore=255；不重新映射。
- Adam：lr=0.001，betas=(0.9,0.999)，eps=1e-8，weight_decay=4e-5。
- 每任务100epoch；batch_l=batch_u=2；每epoch步数取两loader长度最大值向上取整，短loader循环，不丢尾batch。
- LR=`0.001*(1-k/K)^0.9`，k从0开始；不使用val scheduler。
- 前20epoch仅监督，SUP和warm-up不打开U图像。
- `lambda_cons(e)=0.5*min(1,max(0,(e-20)/20))`，e从1开始。
- 配对几何增强沿用原flip/rot90；U学生加原std=0.02输入噪声并clip，教师输入保持同几何的干净图像。
- 三个LIN臂所有前向都不采样分类头；SUP_G1复核参照仍原样使用GAS，见P2。
- 当前EMA decay=0.99；浮点有效权重EMA，离散buffer复制，GAS字段遵循各臂既有规则。EMA永不进optimizer。
- 当前学生原型逐图类别中心归一化后求均值，沿用原`prototypes()`；不增加教师独立原型或历史库。
- LIN_MT_PAS在epoch21、26、…、96刷新当前标注原型；LIN_MT_CONF训练不使用原型。
- confidence、similarity均严格`>0.7`；mask为两模型有效集交集，不增加类别一致条件。
- SSL用`sum_c (p_s-p_t.detach())^2`，按接受像素数归一；不额外除C。
- 空mask返回与学生图相连的零值，不降阈值、不强制top-k。
- 监督仍为原有效像素CE；不加入Dice loss、类别加权、特征损失、强增强/UniMatch新分支。
- 最终指标固定为epoch100学生。EMA和中间epoch仅诊断，不允许择优切换。

同一数值阈值在不同头下未必意味着同一概率校准/接受率，这正是要测量的配方交互，不靠临时调整阈值“修平”。

P1必须完成全部六个单域任务，不能看到一个候选先失败就中断另一域。正式更新为：

\[
3\times(3200+2100)=15,900.
\]

---

## 4. 必须测量的诊断：用证据判断，不预设根因

每个新任务固定在epoch20/40/60/80/100做与旧实验相同的独立evaluator诊断；训练过程和随机流不得因diagnostic调用改变。

### 4.1 分割与质量

- 学生和EMA的每患者rim、cup Dice、宏平均；背景另列，不并入主前景指标。
- 所有病例都保留；ignore与全ignore按冻结评分器处理，同时报告有效支持。
- 伪标签precision、预测覆盖率、图像覆盖率和accepted-correct recall，分别按class、teacher/student、clean/training-like报告。
- raw、confidence、PAS分开；实际训练原型与fresh诊断原型分开，不能混在主质量表。实际不存在原型的SUP/CONF只能标明fresh为诊断，不称为实际训练mask。
- 重做新LIN_MT_PAS的等数量筛选对照；仍然仅诊断，不把TOPK或RANDOM纳入训练。
- precision/recall分母为零时保存undefined和患者计数，不用1填补，不删除该病例的分割结果。

### 4.2 概率尺度与损失信息

需要以下固定字段，而不是只看coverage：

1. logit range、margin及pmax的固定分位点、`pmax>0.99`比例。
2. 真值可评价支持上的NLL和多类Brier；它们来自val evaluator，不回传训练。
3. 按预测类别分组的可靠性表：固定10个等宽confidence bin、每bin计数/平均confidence/实际正确率。ECE仅辅助，保留完整表。
4. 错误前景预测上的平均confidence及有效计数；不能将全背景平均值代替前景。
5. 实际训练每类接受数、概率MSE和weighted MSE。按teacher预测类分组，不能只记录总平均loss。
6. 学生—教师同错、单方正确、双方正确/错误的evaluator分层，用来区分一致性与正确性；不进入训练mask。

可由已保存概率解析计算概率MSE对logits的局部梯度：

\[
g_z=2\,[\mathrm{diag}(p_s)-p_sp_s^T](p_s-p_t).
\]

报告应用真实mask、分母和lambda后的每类局部梯度范数；**这不是全网络参数梯度**，不能由此直接宣称encoder梯度消失。禁止新增真实网络backward来凑诊断；先用合成autograd验证解析式。

校准诊断的用途是检验解释，而不是在val上拟合温度。不能因为某个head的ECE下降，就直接宣布其Dice或SSL贡献提高。

---

## 5. P1筛查与自动分流：旧门槛不改，结果不混写

### 5.1 记号

\(D_{a,d}\) 为本臂最终学生在域d上的患者等权rim/cup宏平均Dice，\(Q_a=(D_{a,RIM}+D_{a,Drishti})/2\)。

\(\Delta_d(a)=D_{a,d}-D_{LIN\_SUP,d}\)。类别差值同理。

旧参照的精确值从冻结CSV解析，本文小数只供核对：

- `SUP_G0 Q≈0.696040805`
- `SUP_G1 Q≈0.705364552`
- `MT_PAS_G1 Q≈0.712331688`

不能直接用以上截断值作gate。

### 5.2 SSL候选筛查

两个预先声明的候选是`LIN_MT_CONF`和`LIN_MT_PAS`。它们是基础配方对照，不预设PAS一定是主方法。对两者执行完全相同的门槛：

| 条件 | 门槛 |
|---|---|
| 对匹配的LIN_SUP的平均SSL增益 | `Q_a - Q_LIN_SUP >= 0.010` |
| 最差域 | 每域`Delta_d >= -0.005` |
| 类别保护 | 两域各rim/cup差值均`>= -0.020` |
| 不只战胜变弱的监督对照 | `Q_a >= max(Q_LIN_SUP,Q_old_SUP_G0,Q_old_SUP_G1)+0.005` |
| 完整性 | 两域最终模型、部署和必需诊断完整 |

保持上轮的1个百分点同配方收益要求及更强监督对照裕量。不利用EMA输出挽救学生gate。

多个候选通过时选至多一个确认主候选：先最大化`min_domain_delta`，再最大化Q，精确同分选较简单的CONF，再按ID。禁止按P0的precision临时重排。

若有SSL候选通过，进入P2-SSL；另一个SSL臂继续作为固定组件对照，不允许在新种子结果出来后替换主候选。

### 5.3 仅监督参考改善的独立分流

若没有SSL候选通过，但`LIN_SUP`满足以下规则，进入P2-SUP：

- `Q_LIN_SUP - max(Q_old_SUP_G0,Q_old_SUP_G1) >= 0.010`；
- 每域相对旧SUP_G1的macro下降不超过0.005；
- 每个前景类别相对旧SUP_G1下降不超过0.020；
- 工程完整和单学生部署合格。

这只允许确认“监督参考实现更值得保留”，**绝不称为SSL成功**。它不是用监督臂替换失败SSL候选。

若两个分流条件均不满足，完整发布P0/P1，结束本轮；不增训练epoch、换阈值、换种子或切换EMA输出。保留有方向但幅度不足的结果。

---

## 6. P2：一次授权完成新的训练随机性复核

复核数据仍是split_seed0，同一患者角色，优化种子21和22。两个种子都必须完成，不能第一种子不好就换或停、第一种子好就提前宣布成功。

### P2-SSL

对每个新种子、两个独立域，固定训练：

- LIN_SUP
- LIN_MT_CONF
- LIN_MT_PAS
- SUP_G1：直接复用Foundation的原始G1监督算法，不改配置，用来防止基线强度比较只存在于seed11。

更新数：`4*2*(3200+2100)=42,400`。

只对P1选中的主候选作确认判断，其他配方如实列出，不能事后换赢家。

**SSL确认条件（实用续投规则，不是显著性检验）：**

1. 两个新种子对各自LIN_SUP的Q差值均严格为正；两种子平均差值`>=0.010`。
2. 各域跨新种子的平均差值`>=0`；每seed/domain差值`>=-0.005`。
3. 每seed/domain的rim/cup相对匹配LIN_SUP差值`>=-0.020`。
4. 新种子主候选平均Q `>= max(平均Q_LIN_SUP,平均Q_SUP_G1)+0.005`。
5. 无缺失轨迹、无测试GT泄露、部署一模型、数值/数据语义一致。

另报告PAS−CONF的每域/每种子差值；只有这些差值支持才能说PAS有额外组件价值。若主候选为CONF且确认成功，就保留简单CONF，不能为了新颖性硬保留PAS。

### P2-SUP

仅在P1没有SSL候选通过、但监督head条件通过时启动。每个新种子、两个域只训练`LIN_SUP`与`SUP_G1`。

更新数：`2*2*(3200+2100)=21,200`。

确认条件：两个新种子Q差值均>0、平均差值>=0.010；每seed/domain宏平均下降不超过0.005，每seed/domain前景类别下降不超过0.020。通过也仅称`SUPERVISED_HEAD_REFERENCE_SIGNAL_REPLICATED`，SSL状态仍未成立。

### 证据边界

- 不混入开发seed11抵消两个新种子的失败。
- 三seed总体均值/样本SD可描述性列出；确认判断只用21/22。
- 新优化种子不是新患者、独立数据或SOTA证据，固定head初始化部分仍相同的事实必须披露。
- 现有val被多轮研究者开发使用，即使算法内没有泄露，也不能当成全新独立确认。
- 本轮不自动进入冻结层/CL实验。确认后只提交固定下一阶段设计，不额外添加前驱教师。

---

## 7. 预算、资源与执行顺序

| 路径 | P1 | P2 | 本轮正式合计 | 加旧131,508后的累计 |
|---|---:|---:|---:|---:|
| 没有配方准入 | 15,900 | 0 | 15,900 | 147,408 |
| 仅监督参考复核 | 15,900 | 21,200 | 37,100 | 168,608 |
| SSL完整复核 | 15,900 | 42,400 | 58,300 | 189,808 |

两条P2分支互斥，不把两个预算同时执行。失败尝试、诊断、qualification算力另计；表内是全部成功任务的计划更新数，不等于实际所有尝试成本。

P0前向至多1300次sample-model调用、0更新。新训练inline诊断单独记录。工程真实smoke最多12次optimizer更新，与正式初始化隔离，只检查功能和数据边界，不能用于选择配置；随机初始化下SSL掩码可能为空，应如实记录，用合成样本检验非零SSL梯度通路，不能为smoke降低阈值。

每任务最多学生和当前EMA两份完整模型；SUP也维持EMA仅作匹配诊断。模型权重、梯度、优化器、激活与临时预测张量分别计数，不能把两份权重内存等同于总显存。最终部署必须在EMA/原型不可用时只读学生运行。

资源默认GPU4/5，每GPU一任务，先检查可用显存及现有负载；已明确授权的6/7可按当次资源授权使用，不由报告中的旧说明擅自扩大。不得终止别人的进程，不为填满GPU启动额外实验。无硬编码“多少分钟完成”的承诺，按本次实际吞吐报告ETA和等待原因。

执行次序：发布新协议和来源绑定 → 新代码合成/CPU-CUDA资格 → 固定来源冻结 → P0与P1按固定内容执行 → 编译完整P1 → 自动执行已准入的P2 → 最终部署、报告与归档。P0观察结果不改变P1/P2内容。

---

## 8. 必要回归与工程容错

不重建整个旧审计体系。复用已通过的loader、metric、RNG、EMA、resume基础测试，只增加能检出本轮真实风险的检查：

1. `raw_linear3_same_geometry`与`F.conv2d(x,mu,padding=0,bias=None)`逐值一致；与原G0不同；无normalize、×10或stochastic调用。
2. 相同seed/domain，三个LIN臂的初始主干/mu哈希相同；与原构造器相应字段一致；warm-up阶段新三臂精确匹配。
3. 删除sigma/grad_update的学习贡献不改变mu初始化与RNG外部状态；惰性参数计数与无梯度检查准确。
4. 原G0/G1参照计算语义不受新head类影响；禁止修改旧源码实现“全局开关”。
5. confidence/PAS严格阈值、实际原型刷新、联合mask、空mask和ignore计分与旧定义一致。
6. 等数量掩码在每图/draw/teacher预测类满足K；同分、K=0、K=N、多标签预测不一致均有测试。
7. mask API不接收GT；GT改变不能改变mask/count/rank。计分支持变化仅影响metric，并正确标出有效数量不匹配。
8. 精度/coverage/recall分母分别测试，不能把undefined视为完美预测。随机draw和null排列不能变成独立患者。
9. 解析概率MSE logit梯度与合成autograd一致；不把局部梯度叫参数梯度。
10. 独立canonical-only HDF5布局测试、U无标签读取测试、warm-up零U读取。
11. 同源中断恢复保持optimizer、EMA、head_mode、RNG、epoch和数据位置；旧完整轨迹不可覆盖。
12. 部署checkpoint含head_mode，单学生加载时能复现最终预测；没有teacher/prototype文件也能运行。

资格失败先修代码，不启动正式任务。正式任务发生非有限loss/错误数据/断言失败时保留checkpoint与原始日志，不降阈值、不忽略batch、不改loss。仅机械性I/O/资源失败可同源恢复；改变数值算法、head数学定义或数据语义必须新协议，不在本轮边看val边改。科学数值不达标不是工程重试理由。

---

## 9. 交付文件与终态

尽量由同一冻结结果编译器从原始receipt/CSV生成，不手填PASS。最低交付：

- `PREREGISTRATION.md/json`、`SOURCE_AND_INPUT_LINEAGE.json`：固定来源、所有约束和历史保护。
- `HEAD_CONTRACT.json`：准确forward、初始化、惰性字段、参数/输出几何、部署head_mode。
- `PAS_FIXED_MASS_AUDIT.csv/md`：实际数量、P/C/I/R、条件筛选比、undefined与噪声/标签来源。
- `SINGLE_DOMAIN_METRICS.csv`、`PER_CLASS_METRICS.csv`、`HEAD_SSL_CONTRASTS.csv`。
- `PROBABILITY_DIAGNOSTICS.csv`、`PSEUDO_QUALITY_BY_CLASS.csv`、`GRADIENT_LOGIT_DIAGNOSTICS.csv`。
- `SCREEN_DECISION.json`：所有候选完整判定，不只保存赢家。
- `REPLICATION_BY_SEED.csv`、`CONFIRMATION.json`：未准入标NOT_ADMITTED，不补数。
- `TRAINING_AND_MEMORY_ACCOUNTING.json`：正式更新、诊断、失败/qualification尝试、进程exit、教师/学生状态、实际开图次数。
- `DEPLOYMENT_REPORT.json`、`TEST_REPORT.json`、`FINAL_REPORT.md`、`PUBLICATION_VERIFICATION.json`。

最终必须分开列出：

1. `ENGINEERING`：COMPLETE / INCOMPLETE。
2. `PAS_MATCHED_COUNT_EVIDENCE`：描述性结果，不能取代分割效果。
3. `HEAD_SUPERVISED_EFFECT`：是否改善监督参考及是否复核。
4. `SSL_INCREMENT`：是否达到同head监督对照增益、是否新种子复核。
5. `PAS_INCREMENT`：相对CONF是否有组件价值。
6. `CL_AND_SOTA`：本轮NOT_EVALUATED。

候选没有通过就如实止损。head仅提高监督分数，不叫SSL成功；PAS仅提高precision，不叫分割成功；筛查通过但新种子失败，不混入seed11挽救。

本轮结束后不再自动搜索温度、confidence、similarity、增强或loss权重。若两个基础臂仍不能建立可复核的SSL增量，关闭此固定配方的头参数化排查；后续只能另立实质不同问题，例如参考实现完整复现或新数据支持，而不是继续同一val上的小网格。

---

## 10. 来源索引（Codex必须实际核对）

[R1] 已完成Foundation报告、终态、指标与成本。
[R2] 同模型PAS质量诊断及实际/新原型区分。
[R3] Foundation核心训练/原型/mask实现与协议；本轮多数规则继承于此。
[R4] 用户附件：Pandey等《Continual Segmentation under Joint Nonstationarity》，arXiv:2605.20538v1，PDF SHA256 `dc5cdfc70cdda70e6175ce055632a0cf00431bcdbf88fd2801f6f5cf37742b93`。第5页公式(3)–(5)，第28–30页Appendix B。
[R5] 固定官方分类头实现。
[R6] Guo等，On Calibration of Modern Neural Networks，ICML 2017。这里只作为置信度诊断背景，不作为本实验成功预测。

```text
R1 https://github.com/DLwbm123/SSL_CL_seg/blob/405337e71d22aef011a041664fbc1b4b72b60408/experiments/lcrseg/docs/ssl_foundation_v0_1/FINAL_REPORT.md
R2 https://github.com/DLwbm123/SSL_CL_seg/blob/405337e71d22aef011a041664fbc1b4b72b60408/experiments/lcrseg/docs/ssl_foundation_v0_1/PAS_COVERAGE_PRECISION.md
R3 https://github.com/DLwbm123/SSL_CL_seg/blob/d9542bd6f7b4acfe351bdaa8c8107bb875d40717/experiments/lcrseg/ssl_foundation_v0_1/core.py
R3 https://github.com/DLwbm123/SSL_CL_seg/blob/405337e71d22aef011a041664fbc1b4b72b60408/experiments/lcrseg/docs/ssl_foundation_v0_1/PROTOCOL.json
R4 https://arxiv.org/abs/2605.20538v1
R5 https://github.com/prinshul/JASCL/blob/3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53/Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT/models/deeplabv3/deeplab.py
R6 https://proceedings.mlr.press/v70/guo17a.html
```

## 11. 直接启动语句

```text
请完整阅读并执行 SSL_CL_Classifier_Head_Control_V0_1_Codex_Plan.md。

起点：405337e71d22aef011a041664fbc1b4b72b60408
新分支：codex/ssl-classifier-head-control-v0-1

本次是独立的新问题，不重判旧SSL_FOUNDATION_NOT_ESTABLISHED。
先做限定的最终模型PAS等数量诊断，再完成同几何线性头的
LIN_SUP、LIN_MT_CONF、LIN_MT_PAS，两域独立随机初始化训练。
不修改原型/阈值/目标/优化器/数据，不把G0误当普通线性头。
首轮optimization_seed11，正式15900次更新。

若按附件SSL门槛通过，自动完成optimization_seed21/22的
LIN_SUP/LIN_MT_CONF/LIN_MT_PAS/SUP_G1，新增42400次更新；
若仅监督head门槛通过，则只确认LIN_SUP/SUP_G1，新增21200步。
两条复核分支互斥；无需再次等待确认，不按结果换配方或种子。

不得只返回计划或合成测试结果。资格通过后实际执行完整已准入任务。
每任务最多学生+当前EMA，最终单学生。禁止历史教师、回放、KD、
SCD/风险头、隐藏GT、test、main合并或额外超参数搜索。
分别报告监督head、SSL新增贡献、PAS组件与复核结论。
完整发布、核验公开访问并归档NAS后停止。
```
