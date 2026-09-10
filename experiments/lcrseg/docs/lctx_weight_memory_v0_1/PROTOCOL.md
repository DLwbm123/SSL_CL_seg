# LCTX × Weight Memory V0.1：参数记忆保持的受控实验

## 0. 决策、定位与执行范围

执行代号：`LCTX_WEIGHT_MEMORY_V0_1`。

本计划取代“继续扩大 AMS 伪标签实验”作为下一轮工作。固定已经有开发价值的 LCTX，检验参数更新容量和源权重子空间保护能否减少绝对旧域遗忘，同时保持新域学习。

这是一项**新实现、新命名空间、受控适配实验**，不是恢复旧 KI，不继承其身份或历史 SOTA。旧 KI 的两个缺失终态、单域 AMS 的 VALUE_REPRODUCED、序列 AMS 的失败及所有原报告保持不变。不得再搜索或索要旧 KI 启动命令、run_id 或配置。

本轮不是半监督增益确认：正式训练只读当前域 L 图像与标签，U 图像读取为零。LCTX 的 donor 也来自当前 L。若获得保持收益，只能称 LCTX 上的参数保护证据，不能称 SSL 成功。

用户将本文件交付执行时，范围为必要资格、下述固定六臂矩阵、隔离阶段末评价、聚合与归档；GPU 4/5/6/7 的共享授权沿用。不自动启动后续新种子、第三域、独立患者、AMS 接入或超参网格。

## 1. 冻结起点：已经真实存在的父实验

- 仓库：`DLwbm123/SSL_CL_seg`
- 结果/集成提交：`671d0e57c8a2b56b5162933be4973ea0f538f58c`
- 已完成序列实验执行源码：`f174cf2a68212cf04ad19b18d8203a6455c94aee`
- 父文档目录：`experiments/lcrseg/docs/ams_seq_transfer_v0_1/`
- 新分支建议：`codex/lctx-weight-memory-v0-1`
- 新源码目录：`experiments/lcrseg/lctx_weight_memory_v0_1/`
- 新文档目录：`experiments/lcrseg/docs/lctx_weight_memory_v0_1/`

读取父实验 FINAL_REPORT.md、IMPLEMENTATION_NOTES.md、SOURCE_INITIALIZATION_LEDGER.csv、SOURCE_FREEZE/QUALIFICATION/运行清单中实际存在的绑定文件。文件名称以目录清单为准，不凭猜测制造依赖。

复用其六个 `SRC_CE` 最终学生：seed61/62/63 × RIM/Drishti。以完整 source receipt 和 student hash 验证，仅加载学生参数，不加载源 EMA、Adam、RNG。不按源分数挑选入口，不从上一轮最终 target 模型初始化。

若某份本轮已知 SRC_CE 权重确实丢失或损坏，允许从这份已知父源码、配置和同一 seed 重新生成该源模型，另起恢复任务 ID 并记录；无需寻找任何旧 KI。最多补齐六份，最多新增15900次源训练更新。任何重新生成的源都必须供该区组所有六臂共同使用，不能混用不同源状态，也不能把旧目标分数直接当作这份新源的匹配结果。数据/源码损坏等真实工程问题仍可停止，不能跳过来源与数据隔离。

## 2. 证据及本轮假设

父实验平均结果：

| 配方 | Final | Incoming | Old | Forget |
|---|---:|---:|---:|---:|
| CE | 0.549103 | 0.697653 | 0.400553 | 0.284266 |
| LCTX | 0.600145 | 0.718077 | 0.482212 | 0.202606 |
| UCTX | 0.587655 | 0.725457 | 0.449854 | 0.234965 |
| AMS | 0.588958 | 0.719420 | 0.458495 | 0.226324 |

LCTX通过相对监督对照的保护线，不等于无遗忘。平均源分数约0.684818，LCTX后旧域仍损失约0.202606。

从公开 FINAL_SITE_CLASS_METRICS.csv 计算的 LCTX 绝对旧类遗忘（每类先三种子平均）：

| 旧域/类别 | 源分数 | LCTX旧域分数 | 遗忘百分点 |
|---|---:|---:|---:|
| RIM/rim | 0.726043 | 0.522492 | 20.36 |
| RIM/cup | 0.666189 | 0.541649 | 12.45 |
| Drishti/rim | 0.626883 | 0.425400 | 20.15 |
| Drishti/cup | 0.720161 | 0.439308 | 28.09 |

本轮假设：

H1：减少可更新参数或采用低秩残差，可能比全参数LCTX更好地保持旧域，但可能牺牲新域。

H2：源权重的主右奇异子空间可能含有值得保护的映射方向。基于源权重选择方向，可能优于同维随机方向。这是待检验假设，不能把权重能量当作旧数据激活、Fisher重要性或类别记忆的已证明替代。

H3：只限制输入侧更新可能比双侧投影保留更多可塑性。不能预定其必胜；双侧臂是比较对象，不是恢复旧BA实现。

H4：任何保持收益必须超过强LCTX与容量匹配对照；仅旧域更好而新域明显恶化，不构成本轮采纳。

相关工作不是本项目证据：O-LoRA、InfLoRA和MiLoRA已有相关低秩/子空间先例；OPLoRA（AAAI 2026）直接提出源权重SVD的双侧投影。因此本轮不得声称“SVD+投影”本身是新的，也不得自称完整复现这些论文。这里固定的是适合当前小型U-Net、来源与容量可解释的适配。

## 3. 数据、顺序与预算

沿用父 split0 和原有患者角色：RIM L16/U63/val40，Drishti L10/U41/val25。U的数量可从既有清单读取以校验继承的更新预算；不创建U图像访问器、不加载U图像或GT。

顺序：
- O1：RIM_ONE_r3 → Drishti_GS
- O2：Drishti_GS → RIM_ONE_r3

开发种子：61、62、63。它们已研究者暴露，本轮是开发比较，不是新的患者或优化种子确认。

每个目标任务100 epochs：Drishti每epoch21步、共2100步；RIM每epoch32步、共3200步。L batch2、drop_last=false及原顺序、增强、标签计分全部沿用。虽然本轮不用U图像，也不得因L只有10/16人而缩短epoch。

六臂 × 两顺序 × 三种子 =36个新增目标训练任务：

| 部分 | 任务 | 正式更新 |
|---|---:|---:|
| O1目标Drishti：6×3×2100 |18|37800|
| O2目标RIM：6×3×3200 |18|57600|
| 常规合计 |36|95400|
| 条件源权重恢复上限 |至多6|至多15900|
| 含源恢复的正式更新绝对上限 |至多42|111300|

正常情况下源模型复用，新增源训练为0。资格/smoke成本另列，旧95400和106000均不重计。建议新增合成optimizer更新不超过120，真实目标L smoke不超过24次且全部丢弃；现有纯数学/访问测试应复用，不为增加测试条数扩展合成训练。若工程资格确需超过上限，先在冻结记录中说明测试需求，不能混入正式更新或改数学候选。

## 4. 六臂：所有臂只使用同一个LCTX目标

| 臂 | 参数更新策略 | 用途 |
|---|---|---|
| F_FULL | 原T_LCTX全有效参数微调 | 强监督混合参照与回归验证 |
| F_CONV | 仅下述14个主干Conv2d权重全量更新，其他参数冻结 | 控制冻结读出/归一化/上采样造成的收益 |
| LR_FREE | 同14层，冻结源权重，普通rank8 LoRA残差 | 控制低秩容量及参数化 |
| LR_RAND | 同上，输入侧同维随机子空间保护 | 控制约束维数而非源信息 |
| LR_SRC_A | 同上，源权重右奇异子空间的输入侧保护 | 唯一预指定主候选 |
| LR_SRC_AB | 同上，源权重左右奇异子空间的双侧保护 | OPLoRA启发的比较臂；不是原KI的BA版本 |

不得执行到一半删除不佳臂、调rank/k/LR、调整阈值或把获胜对照重新指定为原主候选。

### 4.1 可更新层的确切定义

父模型主干为16/32/64/128通道的U-Net、GroupNorm，三个转置卷积上采样块，线性3×3分类头。固定以下14个Conv2d权重：

```
enc1.block.0.weight
enc1.block.3.weight
enc2.block.0.weight
enc2.block.3.weight
enc3.block.0.weight
enc3.block.3.weight
bottleneck.block.0.weight
bottleneck.block.3.weight
decoder.dec3.merge.block.0.weight
decoder.dec3.merge.block.3.weight
decoder.dec2.merge.block.0.weight
decoder.dec2.merge.block.3.weight
decoder.dec1.merge.block.0.weight
decoder.dec1.merge.block.3.weight
```

F_CONV和四个LR臂都冻结：线性头`decoder.conv_logit.mu.weight`，GroupNorm affine，全部ConvTranspose2d权重及bias，以及原本惰性的sigma/grad_update。

F_CONV与LR臂使用相同可改层，区别仅密集更新与低秩更新。F_FULL与F_CONV的差是多类冻结操作的组合，不得解释成“单独冻结分类头”的因果效应。四个LR臂的其他层、初始化源模型、rank、优化设置和数据流完全一致。

如果载入的模型与这份已核验架构不匹配，报告具体模型/配置不一致，不去找旧KI、不擅自扩大模块集合。

### 4.2 LCTX及优化设置

所有臂沿用T_LCTX：目标epoch1–20为干净L的CE+前景Dice；epoch21开始使用L/L互补混合。将当前L batch两位不同患者互换为donor，donor只提供图像/geometry，不额外使用其GT。每个锚定L像素恰好计分一次，不增加干净L损失。

继承原384×384 RGB、原几何增强、2/3边长方块、donor学生噪声0.02、真实标签Dice定义和ignore处理。

每个目标任务重置Adam和本域LR进度。默认沿用父Adam：lr0=0.001，betas=(0.9,0.999)，eps=1e-8，weight_decay=4e-5，poly power0.9。执行前核对父有效配置，任何真实差异必须事前披露并统一冻结，不能事后调参。LR臂Adam只包含当前A/B；F_CONV只包含14个权重；F_FULL保持父集合。不同参数化的weight decay不等价于相同函数正则，报告这一点。

所有新分支名字/臂名字不能污染原标签顺序、mask或增强key。适配器与随机子空间使用独立keyed stream。保留父T_LCTX的数据随机流标识。

本轮四个LR臂均训练A和B；A-only是限制有效输入侧方向，不是冻结B。适配器约束从目标第1步生效，所以不同更新策略的epoch20状态不应强制相同。仅F_FULL应还原父T_LCTX语义与轨迹。

## 5. 参数保护的数学定义

对任一允许层，将源卷积核按原存储顺序展平：

W0 ∈ R^(m×d)，d=Cin×kh×kw。

固定rank r=8，scaling alpha/r=1（alpha=r），无LoRA dropout。源W0全程冻结。

对非零源权重做FP64 SVD：W0=UΣV^T；选择

k=min(floor(m/2), d-r)。

本架构下m≥16，d≥27，此定义保证最小层剩余输入维度和输出维度均容纳r。k由形状决定，不由val或谱能量阈值搜索决定。记录所选奇异能量比例作为诊断，不用于更改k。全零或数值秩异常应作为源/数值异常明确报告，不能默默赋予“重要方向”含义。

令Vk为前k个右奇异向量，Uk为对应左奇异向量。它们仅由任务入口W0得到，不用历史图像、特征、梯度或标签。

### LR_FREE

W = W0 + BA。

### LR_RAND

独立固定seed的高斯矩阵经FP64 thin QR得到Q∈R^(d×k)，Q^TQ=I。

W = W0 + B[A − (AQ)Q^T]。

### LR_SRC_A（主候选）

W = W0 + B[A − (AVk)Vk^T]。

这通过有效参数化保证ΔW Vk=0，不是给正交loss设置一个软权重。B不加左侧保护。

对固定输入patch h∈span(Vk)，有(W−W0)h=0。但网络上游输入会改变，旧患者patch不一定在该空间，因此**不能从这个局部等式推出旧域预测不变或零遗忘**。

### LR_SRC_AB

W = W0 + [B − Uk(Uk^T B)] [A − (AVk)Vk^T]。

这是双侧受限比较。它与输入侧臂的有效自由度不同，报告这一差异，不宣称所有LR臂的有效容量相等。LR_RAND与LR_SRC_A才是同维约束比较。

### 初始化与实现细节

所有LR臂B0=0，使初始有效W与W0一致。共享初始高斯G∈R^(r×d)，元素标准差1/sqrt(d)。

LR_FREE取A0=G。保护臂先计算G在相应输入补空间的投影Gp，然后令原始A0=Gp×(||G||F/||Gp||F)。LR_SRC_AB与LR_SRC_A使用相同输入侧初始化。这样有效A初始Frobenius范数匹配，避免单纯因投影缩小初始化尺度。投影norm数值为零时作为数值资格问题处理，不能依据真实表现反复换随机种子。

每次forward都用有效投影参数构造卷积核，保持原stride/padding/dilation/groups；不得只投影梯度而假定Adam更新后仍满足约束。不要构造d×d稠密P矩阵，使用低秩乘法。不得detach有效A/B或输入特征，冻结分类头只冻结其参数，不切断对主干的梯度。

SVD/QR在固定环境与独立CPU FP64临时张量中完成，恢复所有随机流；保留源、基底、shape和初始化hash。完整U/V无需跨层常驻，缓存的仅必要thin基底。基底从源参数再计算，不是旧患者统计。

## 6. 模型状态、当前EMA与部署

任务入口只继承源学生。目标Adam、EMA和数据进度重新建立；不载入源EMA/Adam/RNG。

为保持现有工程边界，可以维护一份当前EMA，但它在本轮所有LCTX臂中不参与目标或选择，最终不部署它。

LR学生由冻结W0+小A/B组成；EMA采用原始稠密架构，按每步学生**有效合成权重**做0.99 EMA，而不是把EMA(A)EMA(B)冒充EMA(BA)。每次只有学生和这份当前EMA两个完整模型对象，不另建源teacher或完整CPU shadow。F_FULL继承父EMA语义。

必须分别报告：冻结基参数、可训练参数、EMA、thin基底、Adam、梯度、临时SVD workspace、activations和实测CUDA峰值。两模型对象不等于所有显存仅两倍参数。

目标结束将有效ΔW合并回普通卷积，删除A/B及基底；输出与原网络形状兼容的单学生权重。部署时仅一份普通U-Net，不需要域ID、任务路由、历史适配器库或源样本。逐层合并并验证数值，不在训练中常驻第三个合并模型。

后续多任务可从合并权重重新构造新阶段，但本轮两域结果不证明长期恒定性能或完全保住更早域。不得用“不增长部署参数”推出“训练无额外状态”。

## 7. 必要资格与执行顺序

P0只进行确切父六源验证、模型层集合/shape验证、参数化与访问资格，不做新的谱/阈值搜索。

必要检查：
1. F_FULL的loss、梯度、更新、随机流和中断恢复与父T_LCTX一致；结构以外的旧数学核不改。基线轨迹差异属于工程回归，不能靠降低分数门槛放过。
2. 所有LR臂初始输出等价，B0=0；B可在首步获得梯度，后续A可学习；冻结参数字节保持不变。
3. LR_SRC_A和LR_RAND在多次Adam更新后保持对应右侧残差接近0；LR_SRC_AB同时检查左右残差。使用包含非零梯度的数值夹具，不能仅测试B=0。
4. 归一化残差使用||ΔW Vk||F/(||ΔW||F+1e-12)等，非零update的FP32目标容差1e-5；超过时调查数值原因。残差为零不是有效性证据。
5. 训练和合并后的student输出在固定合成输入上max_abs≤1e-5，epoch0可按相同有效卷积路径做更严格比较；不因val结果调整容差。
6. frozen head仍传递feature梯度；新loader绝不打开U或源训练图像/GT；LCTX donor无额外GT计分。
7. optimizer、源hash、A/B、基底、EMA、RNG与step的恢复和身份校验通过，失败任务原痕迹保留，不用qualification=True绕过旧正式检查。

资格通过后执行完整六臂×六区组。不要以首seed旧域或current分数准入剩余队列。真实工程错误暂停并记录；数学定义/超参修改意味着新版本，不能在同一臂下继续。

每GPU最多一个本轮任务。仅在任务准入时检查实时显存，沿用项目共享内存准入；无独占要求，不杀其他进程。实际完成在当前执行会话/运行器中验证，不把“已launch”当“完成”。

## 8. 评价：绝对遗忘、增量和类别代价

固定目标epoch100学生。训练退出后，独立单学生evaluator读取源域与目标域val。没有中间GT选epoch；val统计不能回传给训练器。

对于A→B：SA为相同源学生在A上的分数，Old=R_BA，Incoming=R_BB，Final=(Old+Incoming)/2，Forget=SA−Old（不截零）。

必报：各seed/order/domain/role/class的完整表；三种子先做顺序平均的SD；旧域各类绝对遗忘；最差旧域类别均值；相对各对照的配对差。ΔForget=−ΔOld，不当作两份独立支持。

新增一个**零更新STATIC_SOURCE参考**：不训练，在全部目标任务权重封存后，由隔离evaluator将六份源学生在对应目标val各评价一次；其Old复用匹配源评分。不得训练前查看目标GT或根据STATIC分数改配置。此参考用于识别“几乎不学习所以不遗忘”，不是候选，不增加optimizer更新。新增评价前向/GT读取如实计数。

旧AMS/UCTX只做历史补充叙述，不重跑，不当作本轮主要门槛基线。

患者bootstrap2000次，固定分析seed2026091011。各物理域患者权重在方法、seed、order、源/目标阶段共享；域和order等权。三种训练seed仍按完整配对轨迹处理，两顺序不是六个独立seed。患者区间是固定训练/开发条件下的不确定性，不是未暴露患者确认。无逐病例公开输出。

## 9. 两套分项判定，不把“有效”与“源几何有效”混为一谈

所有阈值均为本轮预定工程实用门槛，非临床界限、非统计显著性保证。原AMS gate不改。

### 9.1 VALUE：主候选LR_SRC_A相对F_FULL

需同时满足：
- 平均Final≥+0.005；平均Old≥+0.020；平均Incoming≥−0.005。
- 两个顺序平均Final均>0、Old均>0；三个顺序平均seed至少2/3的Final增益为正。
- 每个旧域/类别的三seed平均差≥−0.005。
- 每个当前域/类别的三seed平均差≥−0.010。
- 任一seed/order/role/class差≥−0.050。

主候选还须报告与F_CONV、LR_FREE、LR_RAND、LR_SRC_AB的同口径表现。满足对全量LCTX的价值门槛，只说明候选是可用的参数保护方案，不能证明选源子空间优于普通低秩或更简单冻结。

### 9.2 SOURCE_GEOMETRY：输入侧源信息是否有额外作用

预指定LR_SRC_A−LR_FREE和LR_SRC_A−LR_RAND两个比较，各需：
- 平均Final≥+0.003，平均Old>0，平均Incoming≥−0.005；
- 两顺序平均Final均不为负；三seed顺序平均差至少2/3为正；
- 相同单次类别−0.050保护线。

同时报告患者区间，不把mean门槛等同确定效应。即使均值达到但区间跨零，也只称方向/幅度信号，不称独立患者效应已确定。

### 9.3 单侧与双侧

LR_SRC_A−LR_SRC_AB必须完整报告新域/旧域/Final/类别/有效自由度。没有预设A侧必胜门槛。若AB更优，就报告单侧优越性未支持；不能因为过去讨论偏好A-only而删除该结果。

### 9.4 终态

- 工程完成但VALUE不满足：`RETENTION_VALUE_NOT_ESTABLISHED`，列出到底是current损失、old增益不足还是类别代价。
- VALUE满足、SOURCE_GEOMETRY不满足：`RETENTION_VALUE_WITH_GEOMETRY_UNCERTAIN`。可能收益主要来自冻结或低秩，不宣布新参数记忆机制。
- VALUE和SOURCE_GEOMETRY均满足：`WEIGHT_SUBSPACE_VALUE_OBSERVED`，仍是开发结果，不用REPRODUCED命名。
- 更简单或双侧控制更好：独立记录控制优势；不将其重命名为预指定主候选成功。
- 不得因为某个随机控制与主候选“不显著”就称两者等效。

无论结果如何，本固定矩阵完成后停止；不自动试rank4/16、改k、改学习率、解冻head、追加伪标签或切新患者救结果。

## 10. 后续路线（仅草案，不自动执行）

若获得参数保护开发价值：先用新优化seed做匹配复核，保留最强简单控制；然后在至少三个域或更长序列验证长期保留。新优化seed不替代独立患者。

只有在保护后的LCTX参照成立后，才在同一个冻结参数策略上并排比较LCTX/UCTX/AMS，并改变预注册标签预算检查U价值。不能仅因项目名有SSL就把未建立的伪目标强行放回最终配方。

若几何优势未建立但普通低秩/冻结有效：保留其为强控制，不继续把SVD包装成创新。

若所有受限臂的Incoming明显下降：判断当前固定容量/冻结组合不合适；本轮不能据此否定所有参数记忆方案。后续需要新协议区分rank容量与冻结模块，而不是重开旧KI溯源。

独立患者评价保留到最终方法冻结之后。当前split已反复开发，应如实披露。

## 11. 必要交付

保持交付精简，合并可合并的日志，不以文件/测试数量当科学证据：
- PROTOCOL.md/json、TASK_MATRIX.csv、BUDGET.json；
- SOURCE_BINDING_AND_LAYER_MANIFEST.json（源与14层、rank/k、基底hash、参数字节）；
- QUALIFICATION_AND_EXECUTION.json（真实更新/前向/内存/恢复/退出）；
- FINAL_SITE_CLASS_METRICS.csv、PAIRED_EFFECTS.csv、PATIENT_INTERVALS.csv；
- PARAMETER_DIAGNOSTICS.csv（只在epoch0/20/100记录有效update相对norm、受保护方向泄漏、谱能量；不额外backward）；
- DECISIONS.json、FINAL_REPORT.md、PUBLICATION_VERIFICATION.json。

公开源码、配置、汇总指标与核验；源/目标权重、患者数据、逐病例输出与私有运行路径留NAS。不合并main，不改变旧文件、旧锁、旧分数，不公开凭据。

## 12. 参考来源

项目结果：
https://github.com/DLwbm123/SSL_CL_seg/blob/671d0e57c8a2b56b5162933be4973ea0f538f58c/experiments/lcrseg/docs/ams_seq_transfer_v0_1/FINAL_REPORT.md

项目逐类结果：
https://github.com/DLwbm123/SSL_CL_seg/blob/671d0e57c8a2b56b5162933be4973ea0f538f58c/experiments/lcrseg/docs/ams_seq_transfer_v0_1/FINAL_SITE_CLASS_METRICS.csv

已核验模型定义：
https://github.com/DLwbm123/SSL_CL_seg/blob/671d0e57c8a2b56b5162933be4973ea0f538f58c/experiments/lcrseg/di_dmpa_jascl/modeling.py

O-LoRA（EMNLP Findings 2023）：
https://aclanthology.org/2023.findings-emnlp.715/

InfLoRA（CVPR 2024）：
https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html

MiLoRA（NAACL 2025）：
https://aclanthology.org/2025.naacl-long.248/

OPLoRA（AAAI 2026，2026-03-14发表）：
https://ojs.aaai.org/index.php/AAAI/article/view/40703

这些论文提供已存在的方法背景，不证明本实验的权重方向就是旧患者功能记忆，也不证明本实验能通过任何效果门槛。

## 13. 给Codex的启动摘要

```
请完整阅读本文件，执行LCTX_WEIGHT_MEMORY_V0_1。
不要再找旧KI，不需要旧run_id。使用已经完成的ams_seq_transfer_v0_1
六份SRC_CE和精确receipt/hash；缺失时仅按其已知配方重建所需源并计数。

固定split0、seed61/62/63、O1 RIM→Drishti、O2 Drishti→RIM。
所有臂只使用父T_LCTX损失和图像来源，不读U。
六臂F_FULL/F_CONV/LR_FREE/LR_RAND/LR_SRC_A/LR_SRC_AB全部完成。
唯一主候选LR_SRC_A；r8、k=min(floor(m/2),d-r)、alpha/r1固定。
仅14个已列Conv2d可使用残差，LR臂其余参数全部冻结。
源W0保持不变，训练A/B，每次forward通过显式参数化满足子空间约束。
随机投影是同维控制，双侧是已有方法启发的比较，不叫旧KI或新原创机制。

正常36个目标任务、95400正式更新，源恢复额外上限15900；成本分别计数。
源旧EMA/Adam/RNG不载入；当前EMA不参与训练目标，按有效稠密权重更新。
最终合并回原始单U-Net，不保留适配器库或任务路由。
GPU4/5/6/7共享授权沿用，不杀其他人的进程。

先完成必要资格，通过后直接执行完整矩阵；不增加新父版本审计门槛。
不因早期分数删臂/换seed/调rank/k/LR；不把目标冻结换成假保持收益。
统一阶段末评价Final/Incoming/Old/绝对Forget、逐域类别、STATIC_SOURCE。
分别判定实用保持与源几何增量；相对零起点/CE好不够，必须对比强LCTX。
全部完成、公开汇总并归档NAS后停止，不自动执行后续新种子或SSL实验。
```
