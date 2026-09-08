# SSL_CL：真实监督锚定与区域混合——稳定性实验 V0.1

**执行代号：`SSL_ANCHORED_MIX_STABILITY_V0_1`**  
**研究优先级：有效性 → 创新性 → SOTA；不为追求复杂度保留无效模块。**  
**定位：受控的基础配方开发与训练随机性复核。不是完整 BCP/JASCL 复现，不是独立患者验证，不是临床安全评价。**

---

## 0. 给 Codex 的执行摘要

完整阅读本文件，完成已准入的真实训练、评价和发布，不仅返回计划。用户将本文件交付执行时，授权范围是本文件定义的 P0、P1 和条件 P2；不含未来 CL 或外部测试。

```yaml
repository: DLwbm123/SSL_CL_seg
base_commit: 61b13a171a992999f1908126f9a4313c64bac647
new_branch: codex/ssl-anchored-mix-stability-v0-1
source_namespace: experiments/lcrseg/ssl_anchored_mix_v0_1/
test_namespace: experiments/lcrseg/tests/ssl_anchored_mix_v0_1/
doc_namespace: experiments/lcrseg/docs/ssl_anchored_mix_v0_1/
data_split_seed: 0
domains: [RIM_ONE_r3, Drishti_GS]
discovery_optimization_seeds: [31, 32, 33]
conditional_replication_optimization_seeds: [41, 42]
new_discovery_arms: [SUP_CED, MT_CED, MIX_CTX, MIX_CED]
primary_scientific_candidate: MIX_CED
predeclared_simpler_alternative: MT_CED
final_prediction: epoch100_student
max_full_models_per_task: 2
P1_formal_updates: 63600
P2_if_MT_CED_selected: 31800
P2_if_MIX_CED_selected: 42400
maximum_new_formal_updates: 106000
historical_formal_updates: 248108
```

这轮不是把旧门槛调低后重判 T_SCE。旧 `TARGET_PATH_SSL_NOT_ESTABLISHED`、所有前驱终态、协议、锁和文件保持不变。

具体改变：

1. 增加**仅用于真实标签**的前景 soft Dice 项，与原监督 CE 相加。
2. 检验 BCP 启发的**标注／无标注互补区域混合**：不再只在同一张 U 图的轻噪声副本上要求师生相似。
3. 设置完全相同的混合输入、但关闭 U 伪目标损失的 `MIX_CTX` 对照，避免把数据混合或真实标签使用方式的收益冒充伪标签收益。
4. 探索准入允许轻微个别种子退化，采用三种子配对平均及分层风险记录；效果确认不因门槛改变而自动成立。
5. 不增加网络、历史教师库、PAS、GAS、投影、风险回归、对比学习、SAM、生成器或新的类别专用规则。

P1 的三个种子全部执行，不按第一个种子结果决定是否继续。P2 只按第 10 节的确定性规则准入，准入后两个种子全部完成，不再等待额外确认。

---

## 1. 决策依据与证据边界

### 1.1 已有报告明确支持的事实

依据 [R1] 的完整真实矩阵：

| 配方 | 三种子两域平均最终学生 Dice | 相对 SUP |
|---|---:|---:|
| SUP | 0.689790 | 0 |
| J_MSE | 0.687621 | -0.002170 |
| T_MSE | 0.686731 | -0.003059 |
| J_SCE | 0.688166 | -0.001624 |
| T_SCE | 0.688352 | -0.001439 |

表格是展示精度；执行分析须读取完整 CSV。T_SCE 的 seed31/32/33 配对差约为 -0.003420、+0.019430、-0.020327，配对差 sample SD 约 0.019952。所有四个 SSL 配方的三种子平均均低于监督参照。

这不是只有“方差较大”的问题，也有“净增益接近零且偏负”的问题。把平均收益门槛从 0.010 降到 0.005，不能使这些结果成为正面证据。

报告中的最终 T_SCE、training-like 诊断：

| 域 | T 相对 J 新增教师对／学生错 | T 相对 J 新增教师错／学生对 |
|---|---:|---:|
| RIM | 约 103.765 像素／患者 | 约 60.852 像素／患者 |
| Drishti | 约 100.200 像素／患者 | 约 145.507 像素／患者 |

这是公开患者平均计数的相减，不是实际梯度效用，也不是隐藏 train-U 的错误率。它支持“扩大监督通路同时扩大错误传播”的解释，不证明唯一根因。

### 1.2 必须撤销的推断

不得假设：

- 高置信度等于正确；增加纠错梯度就必然提高 Dice。
- 同类共同错误等于零梯度。只有分布相同才是精确一致性固定点。
- 背景像素多就证明背景参数梯度支配。
- 三种子全部正增益是小效应方法的必要数学性质。
- 新模型平均更好就证明方差更小；低方差但没有收益也不是有效方法。
- 使用同一开发 split 的新优化种子等于独立患者验证。

### 1.3 本轮的新假设——不是已有结果

H1：真实标签上的 CE + 前景 Dice，比仅 CE 更直接地约束 rim/cup 的区域重叠，可能改善基础学生及教师质量。它不保证降低训练方差。

H2：让标注与无标注区域进入同一次学生前向，并用各自正确的监督来源训练，可能比独立处理两类输入更有效。空间混合也可能产生不自然的图像边界，因而可能失败。

H3：若 `MIX_CED` 不优于同输入的 `MIX_CTX`，不能声称伪标签监督产生了增量，即便二者均优于 SUP。

这两项干预改变了**真实监督目标与输入语义组合**，不是继续搜索已关闭路线的阈值或求根参数。

---

## 2. 文献依据与适配声明

- JASCL 的 PAS 位于 Mean Teacher 框架中；其置信度／原型有效集合与 MSE 定义见附件第 5 页式 (3)–(5)。本轮不将 JASCL 的理论条件当作本数据上的收益保证，也不使用其历史原型回放或 GAS。[R3]
- BCP 使用标注／无标注双向 copy-paste，将真实标签与教师伪标签分别用于其来源区域；其分割目标包含 Dice 与交叉熵。这为本轮混合提供已有方法依据。[R4]
- 医学分割复合损失的比较研究支持把 CE 与区域重叠类损失作为基础对照，而不是先增加新网络。[R5]
- 少量运行的点估计应与不确定性一起报告；反复依据同一开发集修改选择规则会引入选择偏差。[R6–R7]

**与 BCP 的明确差异：**本轮保留项目已有 2D RGB UNet、GroupNorm、线性 3×3 头、100-epoch 预算和 Adam；采用相同 L/U 配对的互补两视图，以控制真实标签曝光量；保留教师软目标与 0.7 掩码；不做最大连通域伪标签后处理、不做 copy-paste 专用预训练、不在伪标签上加 Dice。它只能称为 **BCP 启发的受控适配**，不是 BCP 完整复现，更不能继承 BCP 论文中的数值收益。

若有效，其首先是一个可用基础配方。没有自动的新颖性或 SOTA 主张。

---

## 3. 固定数据、模型与优化设置

### 3.1 数据

两个域独立训练，不构成序列；不能把前一个域训练出的模型作为另一个域初始化。

| 域 | train_labeled | train_unlabeled | val | 每 epoch 更新 | 100 epoch 更新 |
|---|---:|---:|---:|---:|---:|
| RIM_ONE_r3 | 16 | 63 | 40 | 32 | 3,200 |
| Drishti_GS | 10 | 41 | 25 | 21 | 2,100 |

保持现有 split0、患者身份及角色；读取冻结 manifest 做实际核验。历史行数是预期，不可代替本轮输入校验。

- 图像保持 3×384×384，uint8 / 255；标签为 0=background、1=rim、2=cup，ignore=255。
- 数据资源继续经既有 `safe_asset(DATA, relative)` 绑定 `DATA/h5/v1/relative`。
- 仅当前域 train_labeled 标签可进入训练；U loader 不提供 GT 路径或标签。
- val 仅隔离评价；test、未来域标签、旧 formal_03 和历史专家预测缓存不读。
- 历史任务的完整评分和初始化哈希可只读复用，不加载其最终权重来初始化新任务。

### 3.2 模型

沿用 `ssl_target_path_v0_1` 的线性头模式 `raw_linear3_same_geometry`：3×3、无 bias、无 padding，保持原 wrapper 插值；全有效网络参数训练。

不换骨干、宽度、分辨率、归一化、输出温度和训练范围；不重新解释为完整标准 U-Net 或 JASCL 实现。

每任务最多学生和当前 EMA 教师两个完整模型。SUP 对照也维护 EMA 用于对称诊断，但 EMA 不影响其梯度。惰性 sigma／grad_update 不进入优化器且仍计入状态字节。最终部署只运行学生。

### 3.3 固定训练配置

```yaml
epochs: 100
labeled_batch_size: 2
unlabeled_batch_size: 2
drop_last: false
optimizer: Adam
learning_rate: 0.001
betas: [0.9, 0.999]
optimizer_eps: 1.0e-8
weight_decay: 4.0e-5
lr_schedule: lr0 * (1 - zero_based_step / total_steps)^0.9
EMA_decay: 0.99
warmup_epochs: 20
lambda_u: 0.5 * min(1, max(0, (one_based_epoch - 20)/20))
confidence_threshold: 0.7
confidence_operator: strict_greater
temperature: 1
input_noise_std_on_U_student_view: 0.02
supervised_dice_weight: 1.0
supervised_dice_smoothing: 1.0e-5
mix_patch_side_fraction: 2/3
```

原几何增强、数据顺序、U 噪声与 RNG 函数保持不变；新增 mixing 使用独立 keyed stream，不消耗已有流。arm 名不能改变共享初始化／数据流。P1 的旧 SUP/T_SCE 和新臂在相同 seed/domain 的初始化及标签呈现必须可追溯匹配。

不改变标签比例，不加入强颜色增强、CutMix 参数搜索、额外 dropout、SAM、对比项、原型或知识蒸馏。

---

## 4. 固定矩阵与因果解释边界

### 4.1 P1 新增四臂

| 臂 | 输入组织 | 真实标注损失 | U 目标损失 | 作用 |
|---|---|---|---|---|
| SUP_CED | 原始标注图像 | CE + 前景 Dice | 无 | 新的匹配监督基线 |
| MT_CED | L、U 分开前向 | CE + 前景 Dice | 原 T_SCE | 仅改变真实监督损失 |
| MIX_CTX | L/U 互补混合 | 来源于 L 的 CE + 前景 Dice | 0 | 控制混合上下文与标签使用方式 |
| MIX_CED | 与 MIX_CTX 完全相同 | 与 MIX_CTX 相同 | 来源于 U 的 T_SCE | 唯一主候选 |

旧 `SUP` 和 `T_SCE`（seeds31–33）的结果作为固定配对历史参照；不重训。

### 4.2 必须回答的五个比较

1. `SUP_CED − old SUP`：真实标签复合损失的配方效果。
2. `MT_CED − SUP_CED`：在新的监督基础上，原 U 目标的净增量。
3. `MIX_CTX − SUP_CED`：混合上下文和监督输入改变的效果。
4. `MIX_CED − MIX_CTX`：固定混合输入后，U 伪目标的增量。
5. `MIX_CED − MT_CED`：混合配方相对分开前向的效果。

这些是完整训练轨迹之间的配方效应，不能称为单像素因果证明。

`MIX_CTX` 读取真实 U 图像，属于**使用无标注上下文的控制臂**，不是纯监督臂。它的收益不能在论文中被隐藏；也不能把 `MIX_CED − old SUP` 的全部提升归因于伪标签或 mixing。

---

## 5. 真实标注损失：定义必须统一

原 CE 仍在所有有效 GT 像素上平均，包括背景，保持旧权重为 1，不用 batch 内频率重新加权。

对每张图像 b，前景类别 c∈{1,2}，在合法标注支持 V_b 上：

\[
D^{soft}_{b,c}
=\frac{2\sum_{i\in V_b}p_{b,i,c}\,\mathbf1[y_{b,i}=c]+\epsilon}
{\sum_{i\in V_b}p_{b,i,c}+\sum_{i\in V_b}\mathbf1[y_{b,i}=c]+\epsilon},
\quad \epsilon=10^{-5}.
\]

\[
L_{FGDice}=\frac{1}{B}\sum_b\left[1-\frac{D^{soft}_{b,1}+D^{soft}_{b,2}}2\right],
\qquad L_L=L_{CE}+L_{FGDice}.
\]

规定：

- Dice 按图像、按两个前景类别等权，不把整个 batch 当一张大图，不把 rim/cup 合并成 union foreground。
- GT=255 同时排除预测支持、真值支持和交集；不把它计作背景。
- 某类 GT 为空仍按公式计入，不能跳过该类假阳性。
- 整张 L 图没有有效 GT 时，其 CE/Dice 项均返回与学生图相连的零并显式计数；不挑替代病例、不私改主分母。不把空支持的评价约定当作训练质量证据。
- 计算至少 FP32，空间和建议 FP64 或经过误差测试的稳定 FP32；不能 detach 学生概率。
- **Dice 只用于真实标签。**U 伪目标不加 Dice，以免同时改变伪目标损失形式。

增加 Dice 既改变梯度方向，也改变监督项尺度；本轮没有把二者完全分开。不宣称 Dice 天然无偏、必然校准或必然降低方差。

---

## 6. 区域混合：保证标签来源和曝光量明确

### 6.1 输入与掩码

每个 L 图 x_L 与一个同域 U 图 x_U 配对。几何增强已经分别作用于各自图像及相应标签；禁止依据 GT、教师分割或 val 表现挑配对／区域。

在 384×384 上，从 top/left∈{0,…,128} 均匀抽样一个 256×256 方块，M在方块内为1，其余为0。使用独立 `mix_rectangle` RNG，所有类别同一规则。

令 x'_U = clamp(x_U + N(0,0.02²),0,1)，教师输入仍是 x_U。构造互补两视图：

\[
x_A=M\odot x_L+(1-M)\odot x'_U,
\quad
x_B=(1-M)\odot x_L+M\odot x'_U.
\]

学生分别得到 p_A、p_B；两次前向使用同一个学生，并累加到一次 optimizer 更新。不得为了双视图再创建完整网络。

M只是空间方块，不是器官 mask；不声称合成图像保持完整解剖结构。不要加羽化边缘、器官中心定位、类别特例或 seam-ignore。

### 6.2 按来源收集预测用于损失

\[
p_L=M\odot p_A+(1-M)\odot p_B,
\quad
p_U=(1-M)\odot p_A+M\odot p_B.
\]

对 L，使用原 y_L 和 V_L 在 p_L 上计算第5节损失。这样每个原L像素在一对混合视图中恰好监督一次；不把额外干净 L 损失再叠加一次。

对 U，令 q=stopgrad(T_EMA(x_U))、m=geometry_U ∧ [max(q)>0.7]，使用原来的软目标 CE：

\[
L_U=\frac{\sum_{i}m_i[-\sum_c q_{i,c}\log p_{U,i,c}]}{\max(1,\sum_i m_i)}.
\]

- 不对学生 confidence 再筛选。
- q、m 和 M 全部停止梯度；GT 绝不参与 U mask。
- softmax／log_softmax 用同一次真实前向。对 logp 用 `torch.where` 按来源选择 `log_softmax(z_A)`／`log_softmax(z_B)`，不要先低精度 softmax 再 `log` 导致 -inf。
- 未被 mask 接受的 U 像素没有目标损失，但仍提供图像上下文。
- 互补视图上的来源收集是训练计算，不是测试拼接策略；部署只对原图做一次学生前向。

正式目标：

\[
L_{MIX\_CED}=L_L(p_L,y_L)+\lambda_u(e)L_U(p_U,q,m).
\]

`MIX_CTX` 的输入、M、噪声和 L 损失完全相同，U 损失为精确零。可以不运行无用的 teacher-U 前向，但必须如实记录计算差异；不得用其真值或教师目标添加其它约束。

### 6.3 odd batch 处理

不得 drop_last。L batch 通常为2，U最后可能为1：

- 按固定索引映射为两张L分配这一张U，不额外抽样替换。
- 每张原L仍然恰好计算一次来源监督。
- 同一U被用于多个混合上下文时，先对这些上下文的**损失贡献**按重复次数倒数加权，再按唯一U来源的有效／接受支持归一化。
- 不平均 logits 后再算损失；不把重复U当作两个独立样本；教师同一U只需前向一次。
- 用合成编号图验证每个原始像素的来源、次数、mask与梯度。

### 6.4 warm-up

epoch1–20，四个新臂全部只训练相同的 L_CE+Dice，不读U、不混合。每个seed/domain的四臂warm-up权重和Adam状态应一致。

epoch21开始，MT_CED使用原分开输入的T_SCE；两个mixing臂开始互补混合；U权重继续按第3节ramp。不重置Adam、学习率或EMA。

---

## 7. 资源匹配与不能声称的内容

- 同seed/domain四臂使用相同的L呈现顺序及更新次数；互补混合保持每个L来源像素一次计分。
- MT_CED通常每步学生L两张＋U两张；mixing通常四张混合学生输入，教师U两张。最后不完整U batch可能导致前向数不同，必须计数。
- SUP_CED少做U计算。不能宣称所有臂FLOPs完全相等。
- MIX_CTX和MIX_CED的学生输入数及标签来源最为直接匹配；teacher-U前向与损失开销需分别记录。
- 不复制CPU完整shadow模型逃避两模型约束；checkpoint序列化临时张量、Adam、梯度、activations分开记录，不把“两个模型”写成“只有两倍参数显存”。
- 新实验是单域SSL；当前EMA不是t−1教师。没有当前EMA之外的历史模型。未来CL怎样满足同一预算不在本轮自动实现。

---

## 8. 执行阶段与预算

### P0：一次只读基线整理与必要资格

只读 [R1] 的已有汇总、必要的私有逐病例评分和原任务哈希，不重做旧训练／推理：

1. 核对旧SUP/T_SCE seeds31–33的最终学生、EMA和epoch20/40/60/80/100表。
2. 分开报告：配方Q的seed SD、配对增益SD、两域差异和同一轨迹晚期波动。五个epoch不是五个独立模型运行，不挑最好epoch。
3. 对新公式做必要数值与隔离测试；对旧CE、输出头、初始化和RNG做原版本兼容性检查。
4. 建立公共协议、源码清单、输入哈希和任务ledger，再启动新真实训练。

这不是新的大规模oracle或反事实拟合；不为补齐旧合成EMA/I/O计数重跑历史资格。旧计数缺项继续披露，不填0。本轮从实际操作边界记录新计数即可。

### P1：全部四臂×两域×三个固定发现种子

- optimization seeds31、32、33是配对开发区组，已经研究者暴露，不能叫新独立确认。
- 每臂每seed 3,200+2,100=5,300更新。
- 24任务，共 **63,600更新**。
- 不复用旧训练权重或拼接warm-up；复用的只有旧评分、元数据、已冻结的基础实现。
- 任一种子结果好坏均不改变剩余P1队列。

### P2：自动执行一次固定配方复核

按第10节选中的配方：

**若选 MT_CED：** 在新 seeds41、42的两个域各跑 `SUP_CE`、`SUP_CED`、`MT_CED`；12任务，共31,800更新。

**若选 MIX_CED：** 同样两个新seed/两个域各跑 `SUP_CE`、`SUP_CED`、`MIX_CTX`、`MIX_CED`；16任务，共42,400更新。

`SUP_CE`为旧纯监督线性配方，仅把新优化种子接入初始化和所有随机流，不更改算法。P2需要它来避免仅战胜一个变弱的CE+Dice对照。

若P1无候选准入，不启动P2，不追加权重网格、backbone或新种子。

| 路径 | 本轮更新 | 历史正式更新累计 |
|---|---:|---:|
| P1后停止 | 63,600 | 311,708 |
| P1＋MT复核 | 95,400 | 343,508 |
| P1＋MIX复核（上限） | 106,000 | 354,108 |

资格及smoke单列，不能混入正式预算。最多1,000次合成更新；真实labeled-only smoke最多每域4步、合计8步且不用于初始化。未用额度不补跑。该上限不是要求全部用完。

GPU只使用本项目最新明确获准的设备；现有记录为4/5/6，7若无新的明确许可不得推断可用。核验可用显存后排队，每GPU最多一项正式任务，不停止其它用户进程。记录实际吞吐，不承诺固定分钟数。

---

## 9. 如何调整 gate：探索、收益与风险分开

### 9.1 原门槛不动

所有旧终态、判断和文件不改。新门槛仅对尚未训练的新配方生效。尤其不能把旧T_SCE平均负收益通过更改阈值改写成成功。

### 9.2 为什么不再要求“每个seed都正”

收益接近随机波动尺度时，要求所有seed、domain、class同时满足严格边界，是很强的交集事件；它不是均值有效的必要条件。但不要求全正，也不意味着忽略方差和类别损害。

本轮把**探索准入**的实用平均门槛从0.010改成0.005，并要求至少2/3个发现seed为正。小退化被报告和聚合，不再一票否决；严重退化仍阻止自动升级。

这些是研究资源分配规则，不是统计显著性水平或临床安全阈值。

### 9.3 必须同时报告的量

对候选A、同seed监督基线B：

\[
Q_{s,A}=\tfrac12(D_{s,RIM,A}+D_{s,Drishti,A}),\quad
\Delta_s=Q_{s,A}-Q_{s,B}.
\]

报告平均、median、sample SD、range、正负seed数；同时列出domain和rim/cup差值。不得只报告Q均值。

输出分项状态：

- `MEAN_VALUE`
- `PAIRED_SEED_VARIATION`
- `DOMAIN_CLASS_COST`
- `PSEUDO_TARGET_INCREMENT`
- `REPLICATION_STATUS`

低SD但负均值，不算有效。均值正但区间宽／个别严重退化，归为“有均值信号、稳定性待确认”，而不是包装成已解决。

---

## 10. P1 确定性准入与配方选择

对候选 `MT_CED`、`MIX_CED`，都先相对 `SUP_CED` 计算配对差。只有以下条件全部满足，才有资格进入复核：

1. 三seed平均 ΔQ ≥ **0.005**。
2. 三seed中至少 **2个** ΔQ > 0；不要求3/3。
3. 候选三seed平均Q，比旧SUP_CE和新SUP_CED的三seed均值中较强者至少高 **0.005**。避免弱基线获胜。
4. 每个域的跨seed平均macro差 ≥ **−0.005**。
5. 每个domain/class的跨seed平均差 ≥ **−0.020**。
6. 任何单seed/domain/class差都不得 < **−0.050**。这仅是阻止自动晋级的重大退化保护线，不是可接受的部署损害。
7. 必需任务、评分、部署和输入隔离完整。

所有单seed类别下降超过0.020的行都加 `AMBER_CLASS_COST` 标记并进入复核报告，不能因未超过0.050而不披露。

`MIX_CED`还需：三seed平均 `Q_MIX_CED−Q_MIX_CTX ≥ 0.003`。否则最多是上下文增强信号，不认定伪目标收益。该差是最小组件实用幅度，不是p值。

固定选择顺序：

```python
if eligible_MT:
    if eligible_MIX and mean_Q_MIX - mean_Q_MT >= 0.003:
        selected = 'MIX_CED'
    else:
        selected = 'MT_CED'  # 效果接近时选择更简单配方
elif eligible_MIX:
    selected = 'MIX_CED'
else:
    selected = None
```

主科学问题仍是MIX；若转入MT复核，报告必须写明MIX没有足够增量，是**预先允许的简单方案分流**，不能把MT改名为主创新。不得新增第三个候选或改排序。

若只有SUP_CED明显改善，记为 `SUPERVISED_FOUNDATION_SIGNAL_ONLY`，本轮不自动开另一条监督种子搜索。

---

## 11. P2 如何解释：不把更宽松准入当最终成功

P2不重新选择配方。所有差值使用新seed41/42的真实匹配对照，分别报告相对SUP_CE和SUP_CED。

新seed上的**实用均值收益信号**要求：

- 相对两种监督基线分别计算的新seed平均 ΔQ 均 ≥ **0.010**；
- 相对两种基线的每domain平均 Δmacro ≥ −0.005、每domain/class平均 Δ ≥ −0.020；
- 无任何单seed/domain/class Δ < −0.050，且披露全部amber行；
- 若选MIX，相对MIX_CTX的新seed平均增量 ≥0.003。

这一步没有把正式收益幅度下调成0.005。P1的0.005是继续研究的成本门槛，不是新“成功分数”。

同时独立判断稳定性：

- 两个新seed相对匹配SUP_CED均正，且不出现平均类别退化超界：可记为 `DIRECTION_REPRODUCED_ON_TWO_NEW_SEEDS`。
- 均值通过但一个新seed为负：记为 `MEAN_GAIN_ONLY_STABILITY_UNRESOLVED`，不能称稳定解决。
- 两新seed均值未通过：`VALUE_NOT_REPRODUCED`，不是隐藏失败seed后把5个seed混合救回。

即使两个新seed均正，样本仍小；最多称固定患者划分上的开发收益复现。需要更强的稳定性或独立泛化主张，另行设计新的患者／标签划分和足够运行数，本轮不自动执行。

发现3seed、复核2seed和全部5seed汇总分开。全部5seed属于有选择过程的描述性结果，不能代替新seed复核。

---

## 12. 不确定性与患者级诊断

### 12.1 预先规定的区间报告

在新实验中允许并要求2,000次配对cluster bootstrap，固定分析seed=2026090807，分开给出：

1. **患者重采样**：固定训练seed，在每个域内重采样患者；所有配方、所有训练seed使用相同患者权重。解释为固定模型在当前队列上的患者不确定性。
2. **训练seed重采样**：整块重采样优化seed，保留其两个域及所有配方配对关系。解释为极少训练运行下的粗略敏感性，不宣称精确覆盖。
3. 可另外给出联合seed×患者重采样，但不能把它当作独立外部验证。

域只有两个，固定等权，不把域随机重抽或按病例数改变权重。四个诊断draw不能变成独立病例，像素更不能变成独立训练重复。

P1以效应幅度准入，不要求3seed的95%区间下界>0。区间包含0时明确“不确定”，不通过调整区间方法救结果。新P2只有2seed时尤其要说明bootstrap局限，不据此宣称显著性。

### 12.2 只保留能解释问题的诊断

固定epoch20、40、60、80、100，不选best；最终以epoch100学生为主，EMA为对称辅助。

必需表：

- train_labeled上的CE、GT-Dice分项及其实际权重；U软CE、目标熵、去熵KL分项。
- 每类真实标注曝光量、U接受率、每次混合的来源计数；区分记录张次和唯一患者。
- 隔离val上的rim/cup precision、accepted-correct recall、错误前景confidence。
- 普通原图val上的学生/EMA Dice、NLL/Brier；主结果不评价拼接后的人工图。
- late-epoch波动与训练seed间SD分开；不能挑某epoch报告最大收益。
- 错误/正确教师目标的诊断仅用于解释，不决定新掩码或损失权重。

不再默认生成数百万行四种候选局部梯度。局部算术验证在合成测试完成；如要记录真实参数梯度，必须在冻结配置中限定少量epoch/batch、计入成本，不将它作为本轮必须扩展项。核心交付是训练效果，而不是诊断规模。

---

## 13. 必要测试与失败处理

必须覆盖以下类别，不以测试条数作为科学证据：

- Dice值、梯度、GT=255、空类、全ignore、batch内支持差异；rim/cup宏平均不能误成union Dice。
- complementary mixing每个L/U像素的来源和次数；M及1−M测试；odd U batch的反复用权重。
- q和mask无梯度、学生两视图都有预期梯度；MIX_CTX的U目标梯度精确为零。
- 稳定soft-CE用log_softmax，零目标概率／空接受集合返回有限且正确的值。
- 新四臂共同warm-up一致；旧SUP/T_SCE初始化与seed流兼容；mix stream不污染已有RNG。
- canonical路径的独立目录夹具，不能让错误路径和夹具一起通过。
- teacher冻结于梯度但按EMA更新；无第三完整模型；checkpoint恢复包含模型、EMA、Adam、RNG、step、数据顺序和mix流。
- 最终无EMA文件时仍可独立加载并部署学生。

源码freeze后不因val表现改配置。真实工程异常如NaN、数据错误可暂停并保留来源；不能自动换seed、跳病例或改loss。最多一次明确记录的工程修复；若改变数学目标／训练超参，则不再属于本轮，停止并交付，而不是在原名字下继续。

计数缺项不得填0，不为了旧合成计数缺项重跑历史任务；本轮实际计数应从边界持久化。训练成功、工程完整、统计证据和科学门槛四者分开。

---

## 14. 交付文件（保持精简）

必要公共交付：

```text
PROTOCOL.md / PROTOCOL.json
SOURCE_AND_INPUT_LINEAGE.json
TEST_AND_RUNTIME_REPORT.json
RUN_LEDGER.csv
FINAL_METRICS.csv
PAIRED_EFFECTS.csv
SEED_DOMAIN_CLASS_COST.csv
UNCERTAINTY_SUMMARY.csv
EPOCH_METRICS.csv
SUPERVISION_SOURCE_ACCOUNTING.csv
MECHANISM_SUMMARY.md
P1_SELECTION.json
P2_REPLICATION.json
MEMORY_AND_DEPLOYMENT.json
FINAL_REPORT.md
PUBLICATION_VERIFICATION.json
```

大日志、原图、GT、患者ID、逐病例预测和模型权重留NAS；公开必要聚合证据。不公开凭据。不合并main、不篡改旧锁或结果。

最后给用户的摘要应包括：

1. 哪些任务真实完成及实际更新数；
2. CE+Dice是否改善监督基础；
3. 无标注伪目标是否超越匹配SUP及CTX控制；
4. 均值、配对SD、每seed和最差类别代价；
5. P2是否准入、执行了哪些固定配方；
6. 哪些结论是观察、哪些仍是不确定；
7. 精确源码／报告SHA、公开链接、NAS归档和停止状态。

不要只给一个PASS/FAIL，也不要因存在方差就写“没有任何收益”。

---

## 15. 明确止损与后续边界

- P1无候选满足新探索规则：停止本轮，不继续同一val的权重/阈值网格。
- P2均值或类别代价未复核：停止所选配方扩展，不增加新seed救结果。
- 只有GT-Dice监督基础提高：保留这个事实，不包装成SSL成功。
- MIX_CTX与MIX_CED相当：保留上下文增强解释，关闭“伪标签带来增量”的主张。
- MT_CED与MIX_CED接近：按预定规则保留简单MT，不为创新性强留混合。
- 若获得开发收益：冻结配方，只准备下一步独立患者／标签预算确认草案。**本轮仍不进入CL。**

不能因为这轮没有达到预期，就用“底层任务不可能”概括；也不能每次失败都自动重启一个名称不同、实质相同的新模块。

---

## 16. 参考来源

以下来源用于事实与方法动机；本文件的损失细节、互补来源计分、预算及gate是前瞻实验设计，不是引用文献已证明的方案。

[R1] 项目最新完整报告与配套CSV：
https://github.com/DLwbm123/SSL_CL_seg/blob/61b13a171a992999f1908126f9a4313c64bac647/experiments/lcrseg/docs/ssl_target_path_v0_1/FINAL_REPORT.md

[R2] 本次前驱执行源码：
https://github.com/DLwbm123/SSL_CL_seg/tree/098c9c73dc1b7dd1b723363e45cbe34e169fe354/experiments/lcrseg/ssl_target_path_v0_1

[R3] Pandey et al., Continual Segmentation under Joint Nonstationarity, 用户附件，arXiv:2605.20538v1，第5页PAS、第45页配置。理论假设和实验设置不能跨任务无条件套用。
https://arxiv.org/abs/2605.20538

[R4] Bai et al., Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation, CVPR 2023，第3节。
https://arxiv.org/abs/2305.00673
https://openaccess.thecvf.com/content/CVPR2023/html/Bai_Bidirectional_Copy-Paste_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html

[R5] Ma et al., Loss Odyssey in Medical Image Segmentation, Medical Image Analysis 71 (2021), 102035。作者实现与研究结论：
https://github.com/JunMa11/SegLossOdyssey
https://doi.org/10.1016/j.media.2021.102035

[R6] Cawley & Talbot, On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation, JMLR 2010。
https://jmlr.org/papers/v11/cawley10a.html

[R7] Agarwal et al., Deep Reinforcement Learning at the Edge of the Statistical Precipice, NeurIPS 2021。只借鉴有限运行下的不确定性报告原则，不声称其RL实证能证明分割结论。
https://arxiv.org/abs/2108.13264

---

## 17. 可直接使用的启动语句

```text
请完整阅读附件 SSL_CL_Anchored_Mix_Stability_V0_1_Codex_Plan.md。

在独立新分支执行真实监督锚定与区域混合实验。
旧T_SCE、L05、SCD及全部历史终态不改。

P1：split0，优化seed31/32/33，两个域分别训练，
SUP_CED、MT_CED、MIX_CTX、MIX_CED全部完成，共63600步。
Dice只加在真实标签；MIX采用来源明确的互补区域，
每个原始L像素计分一次；不得多监督一次后冒充SSL收益。

按新协议区分探索准入与效果复核：
探索可接受平均+0.005及2/3正种子，但不能救回旧负均值；
风险、方差、类别代价都完整保留。
若准入，按固定简单优先规则选择配方并自动执行seed41/42，
本轮正式更新上限106000；不看中间结果追加模块或网格。

每任务最多学生+当前EMA，最终单学生部署。
不读test或隐藏GT，不引入历史教师，不进入CL，不合并main。
完整发布分项效果、复核、成本、源码与报告后停止。
```
