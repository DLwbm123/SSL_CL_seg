# AMS 独立序列迁移与来源归因实验 V0.1

**执行代号：`AMS_SEQ_TRANSFER_V0_1`**  
**定位：独立的新实验；不是原 KI 的恢复、移植或复现。**  
**研究问题：已验证的混合配方，在从共同历史参数状态出发的两域序列中，能否改善新域学习；这种改善是否以旧域退化为代价？**

## 0. 本轮决定与执行范围

结束旧 KI 父版本溯源。本协议不再要求旧 KI 的源码、配置、run_id、启动命令、成绩或 D0 checkpoint，也不再将这些字段作为任何阶段的输入。

旧 `BLOCKED_KI_PARENT_UNVERIFIED`、`PARENT_NOT_LOCATED_IN_ACCESSIBLE_SCOPE` 以及旧单域 Anchored Mix 的 `VALUE_REPRODUCED` 均保留，不改写成 PASS。旧 106,000 次更新不重跑、不计入本轮。

本轮跳过的是“对旧 KI 工程实体的依赖”，不是跳过数据隔离、数值正确性和来源计分。允许并明确声明使用现有全参数网络建立 **sequential supervised / sequential SSL 参考实验**；不得将它命名为 KI、A-only、O-LoRA 或具有已验证记忆保护的算法。

在当前域训练时，历史信息仅由继承的学生参数承载；不读旧域训练样本、不保留历史特征/原型/伪标签库、不引入独立历史教师。当前 EMA 在域边界重置为当前学生。这里满足“不访问历史训练数据”的实验边界，但没有实现 key isolation，也不宣称参数隔离或零遗忘。

用户交付本文件要求执行时，执行范围是本文件定义的 P0 工程资格、P1 新源模型训练、P2 完整目标域矩阵、评分和归档。不仅返回计划。沿用 GPU 4/5/6/7 的共享使用授权，不重复索要 GPU 权限；不停止其他人的任务，不将历史显存快照当作当前资源情况。

正式训练总量固定为 **36 个训练任务、95,400 次成功 optimizer updates**。资格/smoke 更新单独计数。没有条件扩展到更多 seed、数据集、标签预算、第三域、原 KI、新 KI 或新损失。完成本轮矩阵后停止。

## 1. 已有证据与本轮假设

### 1.1 已有证据

原单域实验的 P2：MIX_CED 相对 SUP_CE 的均值差为 +0.036492，相对 SUP_CED 为 +0.046829；相对 MIX_CTX 为 +0.006228。相对 MIX_CTX 的患者 bootstrap 区间 [-0.001576,+0.014222] 包含零。Drishti cup 相对 SUP_CE 的 P2 平均差约 -0.010916。[R1]

因此，完整配方有开发价值，但不能将其全部收益归于 U 伪目标，不能声称普遍降低方差、独立患者确认或改善持续学习。

恢复任务检查运行元数据和历史源码后，未建立原 KI 的源码/有效配置/运行身份链；这是可访问范围内的溯源终点，不是方法不存在或无效的证据。[R2]

### 1.2 本轮的新假设

H1：给定相同的源域最终学生参数，MIX_CED 可提高第二域训练后的 Final / Incoming 指标。

H2：这种收益不必然减少遗忘。必须测量 Old / Forget，不能从新域涨分推导记忆保护。

H3：标注图像混合、额外 U 图像上下文和 U 伪目标，可能提供不同的增量。用五臂分开比较。

本轮全部假设都允许失败。不会因第一个种子、顺序或源模型分数不好而删减矩阵。

## 2. 源码、分支与实验身份

```yaml
repository: DLwbm123/SSL_CL_seg
proposed_base_commit: 77af53864fa116030c832d69e7a404c79de2456e
new_branch: codex/ams-seq-transfer-v0-1
experiment_id: AMS_SEQ_TRANSFER_V0_1
source_namespace: experiments/lcrseg/ams_seq_transfer_v0_1/
test_namespace: experiments/lcrseg/tests/ams_seq_transfer_v0_1/
doc_namespace: experiments/lcrseg/docs/ams_seq_transfer_v0_1/
math_reference_commit: 800cabe5e5dd69612738aaba45ad35121d63c0f2
prior_mix_report_commit: 1321dd84f67f2de9bd5ceaf513703bda250cc850
old_ki_required: false
old_ki_recovery_enabled: false
new_ki_implementation_authorized: false
training_mode: full_parameter_sequential_control
```

保留已有工作树和未提交修改；必要时使用独立 worktree，不 reset/clean 用户工作目录。旧源码和旧报告只读。新代码完成资格后冻结独立 source commit；所有任务绑定这一提交和完整配置。

优先复用以下数学/数据基础：

- `ssl_anchored_mix_v0_1/core.py`：真实标签 CE+Dice、互补混合、来源收集、soft target、odd-U 权重。[R3]
- `ssl_foundation_v0_1/core.py`：模型构造、DomainData、orders、batch、keyed RNG、EMA。[R4]
- `ssl_head_control_v0_1/core.py`：`raw_linear3_same_geometry` 与 `from_state`。[R5]
- `ssl_anchored_mix_v0_1/engine.py`：作为训练循环参考，不原样当作新的正式 runner。[R6]

**必须处理的现有工程限制：**旧 engine 硬编码旧 seed 集、旧 reservation 结构，以及某些历史初始化/标签顺序哈希检查；它的新任务路径还会调用 `build()` 从头初始化。新命名空间要显式支持本轮任务清单和 `init_student` 的跨域载入。不得用 `qualification=True` 绕过旧正式 gate 后偷偷执行正式训练，不得修改旧 gate 以冒充原协议运行。

新实验中 rank、A/B 形状、KI_REF、KI parent、原 KI D0 均为 **NOT_APPLICABLE**，不是待用户补齐的字段。

## 3. 数据与独立性边界

沿用已验证的 split0 与当前允许的数据路径绑定。只读核验既有 manifest/split 哈希和角色。不要重新搜索原 KI 数据定义，也不要重新划分患者。

| 域 | train L | train U | val | 每 epoch 更新 | 100 epoch 更新 |
|---|---:|---:|---:|---:|---:|
| RIM_ONE_r3 | 16 | 63 | 40 | 32 | 3,200 |
| Drishti_GS | 10 | 41 | 25 | 21 | 2,100 |

现有 step 定义为 `max(ceil(n_L/2), ceil(n_U/2))`。SUP/LCTX 可以只读 U 行数元数据以匹配预算，但不读取 U 图像或标签。实际人数/角色/hash 与冻结数据不一致时，报告真实数据契约问题，不擅自换数据或改人数凑预算。[R4,R6]

优化 seeds：**61、62、63**。这是预定的三个新任务区组，不是新增患者队列。不要将“这个整数以前在别的任务出现过”本身作为停止理由；新任务采用独立实验 namespace、完整配置和来源身份，禁止复用旧训练状态。若确有同一新实验身份已完成的任务，核验并原样续用，不重复训练；若目录只是同名但身份不同，使用新 run_id，不能覆盖。

图像、预处理、增强、384×384 输入、0=background/1=rim/2=cup、ignore=255、患者级宏平均定义保持不变。

两个顺序是两套不同训练轨迹：

- O1：RIM_ONE_r3 → Drishti_GS；
- O2：Drishti_GS → RIM_ONE_r3。

每条轨迹的源阶段只读取该源域训练数据。另一个顺序会训练另一个源模型，但其参数、EMA、优化器、预测、诊断或其他统计不得进入本顺序。

目标阶段的训练进程只获得目标域 train L/U 数据接口。旧域验证数据只由隔离 evaluator 在目标阶段结束后读取。未来域评价 GT、test/hidden GT、formal_03 等既有禁读范围继续禁止。旧域训练数据、旧域预测缓存不得进入目标训练。

既有角色/身份元数据可用于核对训练/评价患者分离；不扫描原始 GT 来挑样本，不使用失败病例修改规则。本轮仍是重复使用开发划分的迁移实验，不是独立患者确认。

## 4. 模型与五个目标域配方

沿用原 raw-linear 网络、宽度、GroupNorm、输出头和全部有效可训练参数；惰性 sigma/grad_update 按原规则冻结/处理，不引入新的 head、LoRA、adapter、rank 或正交项。[R3-R5]

### 4.1 共同源阶段

每个 seed、每个源域独立从头训练一个 **SRC_CE**，100 epochs，纯真实标签 CE。源阶段维护当前 EMA 以复用基础代码，但不使用 U 图像、伪目标或混合。最终使用 epoch100 学生，不以验证分数选择模型。

三个 seed × 两个源域，共 6 个源任务。之所以统一使用 CE，是为了定义一个清晰共同入口；这不宣称 CE 在所有设定上最优。

所有源任务都完成后，进入目标矩阵。源阶段分数仅报告，不作为目标矩阵的科学筛选门槛；非有限数值、数据错误等工程异常除外。

### 4.2 目标阶段

每个 source checkpoint 分叉五个目标域训练任务，100 epochs：

| 新臂名 | 目标数据组织 | L 损失 | U 目标 | 旧数学对应 |
|---|---|---|---|---|
| T_CE | 原图 L | CE | 无 | SUP_CE |
| T_CED | 原图 L | CE+前景 Dice | 无 | SUP_CED |
| T_LCTX | L/L 互补混合，donor 只供图像上下文 | 锚定 L 来源 CE+Dice | 无 | 新的来源对照 |
| T_UCTX | L/U 互补混合 | L 来源 CE+Dice | 0 | MIX_CTX |
| T_AMS | 与 T_UCTX 相同的混合规则 | L 来源 CE+Dice | 来源于 U 的 soft-CE | MIX_CED |

T_AMS 是唯一事先指定的主候选。T_UCTX/T_LCTX 是不可删除的来源对照，不是查看结果后替换主候选的工具。

### 4.3 T_LCTX 的精确定义

在本轮固定数据下，L batch 为两位不同患者的图像。将同一 L batch 反序作为 donor：`donor.image = L.image.flip(0)`、`donor.geometry = L.geometry.flip(0)`。只传图像和 geometry，不传 donor label；无需额外读取标签或 U 图像。用既有 batch indices/manifest 验证配对为不同患者，不能仅靠图像数推断患者不同。

图像已经接受既有几何增强；在 donor 路径上使用与 U 路径相同的 0.02 keyed noise。互补方块、M/1−M、学生两视图和来源收集与 MIX_CTX 相同，只在锚定 L 来源上计算 CE+Dice。不要对 donor 区域额外使用可得的 GT，不增加干净 L 损失。

固定的 L16/L10 加 batch2 与原无放回分批应支持上述配对。若实际 batch/患者分组与此不符，先报告数据契约不符，不能静默自混合、读新患者或改变原 L 呈现。

该对照可复用已验证 MIX_CTX 数学核，但日志必须准确标记 donor 来源为 L。不得沿用旧函数字段，把 L donor 记录成实际 U 图像访问。输入上下文的重复出现与 GT 计分次数分别记录。

每个锚定 L 像素在互补视图对中仍只计分一次。一个 L 图像可同时作为另一条输入的 donor，但这不额外产生该 donor 区域的真值损失。

T_LCTX 不是常规“两个来源区域都监督”的全监督 CutMix。因此它提供来源匹配的归因控制，不自动代表最强监督增强基线。它与 T_UCTX 的差包含上下文患者池、重复频率等改变，不能声称只隔离某一种解剖语义因素。

## 5. 跨域状态转移与训练配置

### 5.1 域边界

源域最终学生为 θ_A。目标任务必须执行：

1. 加载本轮对应 seed/source 的 **学生**状态到目标学生；初始化哈希等于源学生最终哈希。
2. 从该学生新建当前 EMA，EMA 初始化哈希等于学生。
3. 新建 Adam，优化器状态为空；目标阶段学习率计数从零开始。
4. 不携带源 Adam、源 EMA、源 RNG 轨迹、源数据 loader、源特征、源预测或源原型。
5. 目标域数据/RNG 按 `(seed,target_domain,local_epoch,local_step,stream)` 的冻结规则运行；arm 名不改变共享流。run_id/路径用于身份和文件隔离，不悄悄改变配对随机流。

这是预注册的域边界策略，不能按顺序或配方改变。旧参数被继承，但没有额外的旧参数独立副本作教师或正则目标。

**common source 不等于 source ensemble。** 不混合两个源模型或三个种子的权重，不利用另一个域的独立源模型初始化目标任务。

### 5.2 固定配置

```yaml
epochs_per_stage: 100
labeled_batch_size: 2
unlabeled_batch_size: 2
drop_last: false
optimizer: Adam
learning_rate: 0.001
betas: [0.9, 0.999]
eps: 1.0e-8
weight_decay: 4.0e-5
lr: lr0 * (1 - zero_based_local_step / stage_total_steps)^0.9
EMA_decay: 0.99
boundary_optimizer: reset
boundary_EMA: copy_current_student_then_restart
mix_start_epoch: 21
pseudo_weight: 0.5 * min(1, max(0, (local_epoch - 20)/20))
confidence_threshold: 0.7
confidence_comparison: strict_greater
temperature: 1.0
student_donor_noise_std: 0.02
GT_dice_weight: 1.0
GT_dice_smoothing: 1.0e-5
mix_patch_side_fraction: 2/3
student_deployment: original_image_single_forward
```

T_CE 全程 CE。其余四个目标臂 epoch1–20 均为相同的干净 L CE+Dice，不读 U 图像、不混合；这四臂的 epoch20 学生/EMA/Adam/标签顺序哈希应一致。T_CE 的目标 warm-up 不要求与这四臂一致。

epoch21 起 T_LCTX/T_UCTX/T_AMS 开始混合，只有 T_AMS 有伪目标损失。T_UCTX/T_AMS 使用同一目标 U 顺序、同一掩码和几何/噪声规则；模型权重在训练中自然分叉，不能要求 logits 恒同。

沿用原 soft-CE、唯一 U 接受支持归一化、odd-U 多上下文损失重复权重。Dice 只作用于真值 L。q/mask/M 停止梯度。不要对 U 加 Dice、student confidence 双端筛选、seam-ignore、cup 特例或新正则。

U 目标梯度为零不表示 U 上下文对共享网络梯度没有影响；不要测试错误的“U 图像导数必须为零”性质。

每任务最多两份完整模型：student+current EMA。临时序列化张量/Adam/activations 另计，不能用 CPU shadow 绕过限制。目标完成后释放训练模型再做独立部署评价，最终不需要 EMA 文件。

## 6. 正式矩阵与精确预算

| 部分 | 任务数 | 每任务正式更新 | 小计 |
|---|---:|---:|---:|
| RIM 源模型，SRC_CE，seeds61/62/63 | 3 | 3,200 | 9,600 |
| Drishti 源模型，SRC_CE，seeds61/62/63 | 3 | 2,100 | 6,300 |
| O1：RIM→Drishti，5臂×3seed | 15 | 2,100 | 31,500 |
| O2：Drishti→RIM，5臂×3seed | 15 | 3,200 | 48,000 |
| **合计** | **36** | — | **95,400** |

源模型共享只减少相同前缀的重复计算，不能把同一源模型虚记成五次独立训练。

共 30 条分叉后的两域轨迹，只有 6 个唯一源训练任务、30 个唯一目标训练任务。三个优化 seed、两个固定域顺序，不是六个独立数据集。两域患者和源前缀之间的依赖在统计分析中保留。

正式矩阵在源训练前冻结；源文件尚未产生时，任务 DAG 可以绑定预定 source_task_id，而不是要求旧 D0 文件已存在。后续以本轮源任务的 complete receipt 和内容哈希解析。不得因“找不到旧源 checkpoint”再次进入溯源。

资格/smoke 的尝试与成功更新、前向、读数据量单列，不能伪记为正式更新。正式中断若可精确恢复，则恢复相同任务；不能悄悄从头重训并只报告成功的一次成本。

## 7. P0 工程资格：只保留真实必要的前置条件

本阶段目的不是重新做一轮历史审计。只核验本轮可执行性：

**数据与模型：**复用当前验证过的路径解析和 manifest；模型/头/可训练参数与 frozen math reference 对应。确认两域的实际 step 预算。

**数学一致性：**对同一组合成 batch、同一新初始化，T_CE/T_CED/T_UCTX/T_AMS 分别与旧数学核比较损失和参数更新；不需要旧训练权重。测试 T_LCTX 的来源映射、ignore255、empty support、每个 L 像素计分次数、无 U label、无 teacher 目标。

**边界和恢复：**验证源学生→目标学生哈希一致，EMA 正确重置、Adam 真正重置；同一目标 epoch 内/边界上的 resume 与不中断执行一致。旧域训练路径在目标训练器中不可用。测试存储和部署不需要第三完整模型。

**正式入口：**新 seed/arm/sequence/parent_source_task 的预约与身份检查生效；不用原 engine 的 qualification 标志绕过正式 gate。

**GPU：**在实际发起任务前核验当前剩余显存；共享 GPU4/5/6/7，每卡至多一个本实验训练进程。使用实际 smoke 峰值判定可装入，不改输入分辨率或 batch 来迁就资源，也不占用其他未授权卡。可在合适的授权卡上调度，不要求四卡全部空闲。

允许源码修复并重新运行失败的必要测试。修复必须保留原数学/数据/预算；不能依据任何正式验证分数改变方法。对真实运行的 NaN、角色泄漏或无法精确续跑的损坏，保留失败证据并停止受影响的正式执行，不伪造完成。

P0 通过并冻结新代码后，直接完成 P1/P2 固定矩阵，不额外请求旧 KI 线索、不设置外部父版本审查门槛、不以首个源模型的低 Dice 拒绝其余任务。

## 8. 评价时点、指标与配对估计

源阶段只在 epoch100 评价自己的 val。目标阶段只在 epoch100 评价两个已见域。固定学生，不挑 epoch、EMA 或 best checkpoint。

本轮不需要原每20epoch的大诊断矩阵；关闭中间 GT 评价仅是新评价协议，不改变训练流。训练过程中可以保存损失/来源计数和恢复 checkpoint。评价进程不得向训练器返回用于调参或停止的分数。

对 A→B、seed s、目标臂 r，记录：

- S_A(s)：共同源学生在 A 的最终分数；
- R_BA(r,s)：目标训练完成后，在旧域 A 的分数；
- R_BB(r,s)：目标训练完成后，在当前域 B 的分数。

分数仍为每域内患者等权的 rim/cup 宏平均，域之间等权。

\[
\mathrm{Final}_{r,s}=\tfrac12(R_{BA}(r,s)+R_{BB}(r,s)),\quad
\mathrm{Incoming}_{r,s}=R_{BB}(r,s),\quad
\mathrm{Old}_{r,s}=R_{BA}(r,s),\quad
\mathrm{Forget}_{r,s}=S_A(s)-R_{BA}(r,s).
\]

Forget 允许为负，不截断；负值表示该评价上的后向改善。由于共同源初始化：

\[
\Delta\mathrm{Forget}=-\Delta\mathrm{Old},\qquad
\Delta\mathrm{Final}=\tfrac12(\Delta\mathrm{Incoming}+\Delta\mathrm{Old}).
\]

Old 与 Forget 的配对差不是两条独立统计证据；上述恒等式也应作为报告算术测试。

### 8.1 事先指定的比较

1. T_AMS−T_CE：整包相对 CE；
2. T_AMS−T_CED：整包相对 CE+Dice；
3. T_LCTX−T_CED：L 来源上下文混合对照；
4. T_UCTX−T_LCTX：额外 U 上下文患者池的配方差；
5. T_AMS−T_UCTX：U 伪目标增量。

全部比较都报告 Final/Incoming/Old/Forget，两个顺序、三个 seed、每域每类、old/current 两种角色。不能只公布有利的域或最终总分。

NLL/Brier 可以作为已有评价器支持的辅助概率指标，不把它们直接解释成减少遗忘或临床获益。不增加真实训练参数梯度网格、隐藏 U GT 诊断或新的 oracle。

### 8.2 不确定性

分别报告三个 seed 的配对差和 sample SD，以及两个固定顺序的结果。先将两个顺序等权平均得到每 seed 效应，再汇总；不要将六个 seed×order 单元当作六个独立优化 seed。

患者配对 bootstrap：2,000 次，分析 seed=2026090901。在每个物理域内部按患者重采样，同一患者权重用于所有配方、训练 seed、顺序以及源/末阶段评分。固定域不重采样。报告这是对当前已训练模型和开发患者队列的条件区间。

训练 seed bootstrap 如报告，应整块重采样同一 seed 的两个顺序与所有方法，只作为少量 seed 的敏感性，不宣称精确覆盖。无需为了这个实验引入复杂混合模型或追加独立假设检验。

不跨患者拆像素、切片、增强 draw；不把重新训练 seed 等同于独立患者；不把 bootstrap 跨零认作等效或无效。

## 9. 预注册的描述性决策规则

下列值为本轮工程/实用决策阈值，不是统计显著性、确证性非劣效或临床安全界限。不能在看到本轮结果后修改。

对 T_AMS 相对 T_CE 和 T_CED **分别**判定：

- 均值 Final 增量至少 +0.005；
- 均值 Incoming 增量不低于 0；
- 两个固定顺序的平均 Final 增量均 >0；三个 seed 的顺序平均增量至少 2/3 >0；
- 均值 Old 增量不低于 -0.005（Forget 增加不超过0.005是同一约束，不重复计证据）；
- 每个 order×old/current角色×类别的三 seed 平均差不低于 -0.020；任何单 seed、单 order、单域、单类别差不低于 -0.050。

另外，Drishti cup 在任一角色下三 seed 平均差 < -0.010 标 amber，即使总体 guard 通过也不隐藏。完整列出其他类别代价。

伪目标组件：T_AMS−T_UCTX 的平均 Final 差 ≥+0.003 记为 `PRACTICAL_INCREMENT_MET`，其区间、Incoming/Old 代价独立报告；不把这一幅度视为统计独立贡献已确定。未达到伪目标幅度不自动否定整包。

使用结构化分项终态，而非为获得 PASS 更换问题：

| 观测 | 本轮含义 |
|---|---|
| 完整配方达到以上实用规则 | `SEQ_TRANSFER_VALUE_OBSERVED`，只是开发序列迁移证据 |
| Incoming 有改善而旧域 guard 未通过 | `PLASTICITY_RETENTION_TRADEOFF`，下一步需要显式记忆保护而非更多伪标签规则 |
| 混合对照有价值，伪目标增量不确定 | `CONTEXT_VALUE_WITH_PSEUDO_UNCERTAINTY`，主候选与对照结论分别保留 |
| 整包相对监督基线不具实用收益 | `SEQ_TRANSFER_VALUE_NOT_ESTABLISHED`，不否定旧单域结果 |

这些状态可作为分项并存；ENGINEERING_COMPLETE 与 scientific result 分开。若 T_LCTX/T_UCTX 更好，报告其相同 guard 的结果，但不事后称它们为预定的 T_AMS 主候选。

没有“第一 seed 不够好就不跑第二 seed”的科学 gate，也没有“未达到预期就换源模型、换种子、加 epochs”的自动救结果。

## 10. 对下一阶段的影响（本轮不执行）

若混合迁移收益与旧域 guard 同时成立，下一阶段可以在显式新定义的参数隔离参考上研究其增量，并进行公平基线比较；新参考的源码和配置从新提交开始，不再以找回旧 KI 为前提。它必须诚实标为新实现，不继承旧 SOTA 身份。

若只有新域收益而出现更大遗忘，优先补显式参数记忆保护，保留本轮强 sequential baseline；不把这一结果当作增加更复杂 SSL gate 的理由。

若主要收益来自 LCTX，则将其作为强监督增强对照；若 UCTX 超过 LCTX 且 AMS 增量小，则把当前正面证据定位为 U 上下文利用，不夸大伪标签机制。

独立患者、其他标注预算和更长序列在最终配方与比较问题稳定后另行预注册，不用本轮开发结果提前宣称泛化。

本轮本身不是完成后的 TMI 论文，也不是对原 KI 创新性/有效性的否定或确认。已有 BCP 说明 L/U 混合不是新概念；本轮的新增知识来自受控的序列迁移和来源比较。[R7]

## 11. 交付、账本与停止

必要公共交付：

```text
PROTOCOL.md / PROTOCOL.json
SOURCE_AND_INPUT_LINEAGE.json
TASK_MATRIX.csv
BUDGET.json
QUALIFICATION_REPORT.json
SOURCE_INITIALIZATION_LEDGER.csv
DOMAIN_BOUNDARY_TEST.json
RUN_LEDGER.csv
FINAL_SITE_CLASS_METRICS.csv
PAIRED_EFFECTS.csv
UNCERTAINTY_SUMMARY.csv
SUPERVISION_SOURCE_ACCOUNTING.csv
MEMORY_AND_DEPLOYMENT.json
FINAL_REPORT.md
PUBLICATION_VERIFICATION.json
```

文件可以合并以减少重复，只要字段完整可追溯。报告必须从完整 CSV 和实际 receipt 生成，不手抄数值。公开代码与聚合证据；图像、GT、患者 ID、逐患者预测、权重与原始路径记录保留 NAS，不推送私人数据。

完成本轮 36 任务/95,400 正式更新后停止。既有 GPU 共享授权不等于允许新的研究矩阵或占用他人进程。无主分支自动合并、无追加新方法。

最终用户摘要先说真实完成任务与更新，再说 Final/Incoming/Old 的配对效应及来源分解，最后说哪个问题需要下一阶段。不得仅返回审计文件或拿旧106000步冒充本轮进度；也不能把本轮新实验称为“旧 KI 已恢复”。

## 12. 给 Codex 的直接启动文本

```text
请完整阅读 AMS_Sequential_Transfer_V0_1_Codex_Plan.md，
执行独立新实验 AMS_SEQ_TRANSFER_V0_1。

旧 KI 溯源正式结束，不再搜索或索要旧源码、run_id、配置或旧成绩。
旧 BLOCKED/PARENT_NOT_LOCATED 和旧单域 VALUE_REPRODUCED 均原样保留。
本轮不是 KI，也不是新建 O-LoRA；它是使用已验证全参数网络的
来源匹配序列迁移实验，不能冒充原方法复现。

在新 namespace 中复用冻结 AMS 数学核和数据处理，
增加真正的 source-student → target-student 跨域初始化、
当前 EMA/Adam 边界重置以及 T_LCTX 对照。
不要用旧 engine 的 qualification=True 绕过正式门禁。

split0，优化 seed61/62/63。
先重新训练 RIM 和 Drishti 各3个 SRC_CE 源模型，
共6任务15900步，不用旧 checkpoint，不按源分数筛选。

然后每个源模型分叉 T_CE、T_CED、T_LCTX、T_UCTX、T_AMS：
RIM→Drishti：15任务31500步；
Drishti→RIM：15任务48000步。
本轮合计36任务、95400正式更新，资格成本单列。

T_LCTX 用当前 L batch 反序图像作 donor，donor 不增加 GT 损失；
T_UCTX 使用目标 U 图像而无伪目标；T_AMS 使用冻结 MIX_CED。
只有当前目标域训练数据能进入第二阶段。
最多 student+current EMA，最终仅部署学生。

只保留必要工程资格，全部通过后直接完成限定矩阵。
不要再次要求旧 KI 父版本，不根据早期分数换 seed/删臂/改规则。
GPU4/5/6/7 共享授权沿用，实际调度前检查显存，不停止其他任务。

阶段末评价 Final、Incoming、Old、Forget，报告全部域/类别代价，
分开比较 CE、CED、LCTX、UCTX、AMS；不把宏平均提高等同于减少遗忘。
完整发布新实验的源码、账本、配对结果、成本、归档证据后停止。
```

## 来源与事实边界

[R1] `DLwbm123/SSL_CL_seg@1321dd84f67f2de9bd5ceaf513703bda250cc850:experiments/lcrseg/docs/ssl_anchored_mix_v0_1/FINAL_REPORT.md`。

[R2] `DLwbm123/SSL_CL_seg@77af53864fa116030c832d69e7a404c79de2456e:experiments/lcrseg/docs/ki_parent_recovery_v0_2/RECOVERY_REPORT.md`。

[R3] `DLwbm123/SSL_CL_seg@800cabe5e5dd69612738aaba45ad35121d63c0f2:experiments/lcrseg/ssl_anchored_mix_v0_1/core.py`，Git blob `b39d7d3e3ceb0104a9418dac62d8ad6b61633c82`。

[R4] 同一数学参考提交：`experiments/lcrseg/ssl_foundation_v0_1/core.py`，Git blob `d9da2a489d165b71bb8be0dc74951932db8f6748`。

[R5] 同一数学参考提交：`experiments/lcrseg/ssl_head_control_v0_1/core.py`，Git blob `35297fac520bd5dcd059ae222726574ea2413af6`。

[R6] 同一数学参考提交：`experiments/lcrseg/ssl_anchored_mix_v0_1/engine.py`，Git blob `0c789c55d48957cd3be27fd29c1b097cc1b48a65`。

[R7] Bai et al., Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation, CVPR 2023, pp.11514–11524, DOI 10.1109/CVPR52729.2023.01108。

本文件的新矩阵、边界重置、种子、预算和决策阈值为前瞻实验设计，不是来源文献或既有结果已经证明的结论。
