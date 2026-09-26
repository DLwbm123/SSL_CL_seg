# Codex 执行 Prompt：R1_6_READOUT_SUPERVISION_V1

你现在执行 SSL_CL 项目中的一个新的、有界监督实验：`R1_6_READOUT_SUPERVISION_V1`。

本 prompt 经用户交给你并要求执行，即授权下述准备、资格测试、有限诊断和完整三种子矩阵。它不是旧 R1.5 confirmation 的放行，不授权 R2/R3，也不授权改变 history-free / 以已训练参数作为长期记忆的研究设计。

## 0. 目标与证据边界

要回答的唯一核心问题是：在保持 query 架构、匹配损失、数据和训练长度不变时，显式监督最终 semantic 聚合概率，能否带来可重复的分割收益？同时用因子对照检验 Limg 是否必要。

本轮不再训练 GRQA、CC、quota 或 GRQA warm-up，也不加入新的 reward、teacher、蒸馏、RL 控制器、梯度投影或 memory 模块。

不要把普通 NLL/soft-Dice 监督包装成新 RL 算法或已经成立的论文创新。本轮是决定 query 支线是否值得继续的监督基线校准；成功也不能写成半监督或持续学习成功。

必须保留且只读核验的来源：
- R1 报告提交：`b8d9e90afd6a6eafaa4b13fd0e96ca9518684a2b`。
- R1 实际训练提交：`41dfa243f15b184d728c19422a9f23ddcb346de9`。
- R1.5 最终报告提交 / 本轮代码起点：`28872f8dfced13c143fb2e9446d735b5f683effd`。
- R1.5 最终训练提交：`2205de41b64a7eb263e72c2d4254d222ff813d4a`。
- R1.5 状态：`R1_5_DEVELOPMENT_NOT_MET`，保持不变。

先读取该提交中的：
`r1_5_cc_grqa_v1/reports/FINAL_INTERPRETATION.md`、`DEVELOPMENT_DELTAS.csv`、`PATCH_LOG.jsonl`、`STATE_AUDIT.json`，以及 `qprompt/models.py`、`qprompt/losses.py`、`r1_12h/core.py`、`r1_12h/augmentation.py` 和实际 seed/schedule/checkpoint 实现。

已知事实不是新假设的证明：CC 达到容量约束却未稳定超过 A0；warm-up 未降低所记录的早期 clipping；R1.5 有 1,564 次重放与 15 次日志异常，但对照重现差异最大约 0.000232。不得据此宣称新损失一定有效。

## 1. 新分支、目录和权限

仓库：`DLwbm123/SSL_CL_seg`。
从精确提交 `28872f8dfced13c143fb2e9446d735b5f683effd` 创建新分支：
`codex/qprompt-r1-6-readout-v1`。
若已存在且不属于本次执行，用唯一后缀，不覆盖、不 force-push。

新增源码目录：
`experiments/qprompt_rl_v1/r1_6_readout_v1/`。

目标研究根仍是用户已有的 `${PROJECT_ROOT}`；用已配置连接验证 owner、realpath 和实际挂载后，在其中建立本阶段独立运行目录。复用既有 SSH 配置与私有回执，不猜 SSH alias，不再次迁移数据，不改旧 NAS、系统驱动或共享环境。

保留原 R0、R0 review-fix、R1、R1.5 的代码、账本、结果和阶段授权。不要修改原目录的结果或 authorization，也不要把旧失败 gate 改成 true。

新增 `EXECUTION_AUTHORIZATION.json`、`METHOD_SPEC.md`、`TASK_PLAN.json`、`SOURCE_BINDINGS.json`。本阶段自己的授权读取器独立于旧 R1.5 confirmation gate。

正式运行前冻结提交 E0，包含实现、测试、配置与完整任务 DAG。每项运行记录实际训练提交，不用报告提交代替训练提交。不要声称尚未存在的新执行器已通过外部审阅。

## 2. 固定实验矩阵：四臂 × 两骨干 × 两域 × 三种子

骨干：`UNET_QUERY_128`、`DINOV2_VITS14_QUERY`。
域：`RIM_ONE_r3`、`Drishti_GS`。
优化种子：261、262、263；split seed 始终为原来的 0。

所有 query 模型的 global steps 0–1999 都只用原 `supervised_query_loss` 训练，形成共享 prefix。四臂在相同 prefix 上独立分叉，各做 global steps 2000–2999，正好 1,000 次成功更新。

四臂如下，不允许增删：

| Arm | 1,000-step suffix 目标 |
|---|---|
| B0 | Lset：当前 Q0 对照 |
| B1 | Lset + 10 Limg：当前 Q1 对照 |
| B2 | Lset + Lreadout：预先指定的主要候选 |
| B3 | Lset + 10 Limg + Lreadout：Limg 交互对照 |

Lset 严格保持现有 query class/BCE/Dice matching 及其全部权重。B0/B2 不创建、更新或使用 prototype bank；B1/B3 各自使用原 Limg 和阶段内 EMA bank。全部四臂都不使用 EMA reference 网络。

本轮不重跑 D0；不把旧 Q0−D0 解释为纯架构效应，因为原训练目标也不同。

## 3. 唯一新增数学：最终 semantic 读出的监督

从同一次可微模型 forward 取得现有的 `output['semantic']`，记为 p，形状 [B,3,384,384]。

它必须仍由现有函数计算：
- qclass = softmax(class_logits, dim=-1)，保留四类 softmax 中的前三类质量；
- masks = sigmoid(mask_logits)；
- scores[b,c,h,w] = sum_k qclass[b,k,c] * masks[b,k,h,w]；
- 按现有 F1/F2 修复后的 `semantic_probabilities` 进行 FP32 聚合、归一化与低质量回退。

不得新增 head，不得改变推理公式，不得先 argmax，不得 detach p，不得把 p 当成 logits 再送入 cross_entropy。

新损失全部在禁用 autocast 的 FP32 区间中计算。

对图像 b，令 V_b 为 label != 255 的像素集合，令 y_bc 为类别 c 的二值 GT。

1. `L_nll_b = mean_{v in V_b}[-log(max(p[b,y_v,v], 1e-8))]`。
2. 对 c=1,2（rim、cup）：
   `L_dice_bc = 1 - (2*sum_{V_b}(p_bc*y_bc) + 1) / (sum_{V_b}p_bc + sum_{V_b}y_bc + 1)`。
3. `L_readout_b = L_nll_b + (L_dice_b1 + L_dice_b2)/2`。
4. `Lreadout` 为所有有至少一个有效像素的图像 b 的等权平均。

固定外部系数为 1；内部 NLL:DICE 系数为 1:1。平滑项是上式中的 1，log 下限是 1e-8，不搜索这些值。从 suffix 第一次更新即启用，不额外 ramp。

边界定义：
- 全 ignore 图像退出 NLL、Dice 和 batch 归一化分母。
- 整批全 ignore：返回连接 class/mask 计算图的零损失、零梯度。
- 合法全背景图像仍监督背景 NLL；rim/cup Dice 保留，并惩罚前景假阳性。
- 某个前景 GT 缺失时，仍按上式计算该类 soft Dice，不凭预测结果筛除它。
- ignore 像素不参与任何分子、分母或梯度；label 255 必须在 gather 前安全处理。
- 维度错误、非法标签、NaN/Inf 必须报错；不能用 nan_to_num 继续。
- 保持原 semantic 质量 <1e-12 时的均匀回退，记录回退比例与 log-floor 比例。
- 训练 soft Dice 的平滑/缺类规则不是评价 Dice 的定义；评价逻辑不得随之改动。

本阶段得到的是附加输出监督的效果，不是“无需额外监督”的算法；虽然没有新增人工标注，原 L 的 GT 被用于新增目标，必须如实表述。

## 4. 原型、初始化和前缀复用

seed261 复用原 R1 的四个 2,000-step QUERY_PREFIX，不重训、不替换为终点或 best checkpoint。

逐个核验：
- 私有路径通过既有运行记录发现；
- 文件 SHA256 与可信前缀绑定一致；
- 原始训练 code_commit 为 `41dfa243f15b184d728c19422a9f23ddcb346de9`；
- student、optimizer、scheduler 可读且有限；
- global_step=data_cursor=scheduler.last_epoch=2000；
- data schedule、模型初始化、骨干与域身份相符；
- 保存原文件来源，不通过重写 code_commit 冒充新产物。

新 checkpoint 单列 `prefix_source_commit`、`prefix_sha256` 和 `training_code_commit`。

每臂独立恢复 student、optimizer、scheduler、所有已用 RNG 与 cursor。绝不能 B0 完成后接着训练 B2。

B1/B3 在恢复 prefix 后，按旧协议用当前域完整 L、无随机增强的可重复 pass 初始化各自 bank；此 pass 不推进训练随机流，统计 forward 成本。bank 的 FP32 状态及 EMA=0.9 不变。B0/B2 不初始化 bank。

seed262/263 各自产生新的 query 初始化、数据/增强 schedule，并训练本 seed 下的 2,000-step prefix；不得使用 seed261 prefix 假装新种子训练。

每个 seed/backbone/domain 内四臂初始化与输入逐步配对。DINO 官方 backbone 初始权重跨种子相同是预期；query/readout 等随机初始化及输入随机流应随优化种子变化。

查找不到已注册 prefix 时，只隔离对应 seed261 cell；先完成其余可执行任务。未经预算登记，不得秘密多训练替代 prefix，也不得拿别的终点替代。最终标记矩阵不完整。

## 5. 数据、增强、优化器与精度不变

只用现有 M1：RIM 的16张 L/40张 val；Drishti 的10张 L/25张 val。以真实冻结 manifest 验证数量。

冻结元数据：
- manifest SHA256：`0622f54f42f05d6ef87f9dc89ee9435cf8da03c6c30cd970db6ea167e00dd8a3`
- split SHA256：`f250d97aea1f36f21899f5dd40bb6c9a819e7755aee458c8ee27506496b46a88`

训练只读当前域 train_labeled；evaluator 单独读取本域 val。禁止 U、U labels、REFUGE payload、test、MRI、外部数据、旧域训练样本和旧模型蒸馏。

只复用旧 R1 已实现的增强数学和顺序，不根据描述重新手写一个近似 ColorJitter。核验 `r1_12h/augmentation.py`、schedule seed 构造和旧 schedule。

固定：batch2，384×384，K12，bg/rim/cup=0/1/2，no-object=3，ignore=255。

UNet：AdamW lr1e-3、wd1e-4。
DINO：backbone lr1e-5，query/head/upsample/readout lr1e-4，wd1e-2。
其余 AdamW 参数、分组及可训练集合与旧 R1 一致。
poly power0.9、完整 horizon3000，不重启 suffix scheduler。

BF16 autocast、FP32 master weights；现有 matching、semantic、新 Lreadout 使用 FP32。
沿用旧 R1 TF32 禁用和已记录的 deterministic 配置，不声称 CUDA bitwise 可重复。

seed261 的 schedule 必须与旧文件逐值一致，不能只看种子相同。seed262/263 的 helper 不允许仍硬编码261。给 schedule seed、初始化 seed、split seed 分别命名并测试。

## 6. 先做有限实现测试，不再为日志 bug 重放一批训练

CPU 数学测试默认零 optimizer；至少覆盖：
1. NLL/Dice 手算值、普通有效输入、正常归一化。
2. 全 ignore、混合 ignore、有效全背景、缺 rim/cup、非法标签。
3. FP32/FP16/BF16 输入与 autocast 下输出/梯度有限。
4. 新 loss 对 class_logits 和 mask_logits 都可反传；不能在 semantic 处 detach。
5. 关闭 readout 后，B0 对旧 Q0、B1 对旧 Q1 的同状态损失/梯度等价。
6. 原 F1/F2 回归不退化。复用测试代码，但使用本阶段新账本，不重置旧账本。
7. 同一 seed 四臂数据顺序相同；不同 seed 确实改变随机部分；split 不变。
8. bank=None 的 B0/B2、bank 存在的 B1/B3 都能保存/恢复和部署。
9. qualifier、train、evaluate、aggregate 使用同一个配置 schema。

必须在真实 tensor 之外单独做日志状态机测试，直接触发 local step 1/99/100/101/249/250/251/1000 和 global prefix 边界。无需做100次 optimizer 才测试日志。

诊断记录使用嵌套命名空间：
`meta`、`losses`、`readout`、`gradients`、`runtime`。
禁止用多个 `**dict` 向同一 flat dict 注入可能重名的 `Lseg`、`categorical_kl` 等键。测试实际 writer、序列化、resume、聚合器，而非仅测试一个 mock 字典。

原生合成资格总 optimizer attempts 上限64，CPU/CUDA分开记账但共享这64上限。覆盖两骨干、四臂、一次 forward/backward/update、完整状态恢复、诊断不改变更新，以及故障账本；不要求消耗完64。

真实 L smoke 上限16次 attempts：四个 backbone/domain cells 各4步，覆盖四臂。smoke 状态全部丢弃，不进入正式 prefix/suffix。不用 smoke Dice 调参。

## 7. 小规模零更新诊断：64个固定批次

只对seed261的四个prefix，各取16个旧 suffix batch：
`global_index = 2000 + floor(j*1000/16), j=0..15`，复用原输入与增强。

共64个batch，不再重复400个batch。一个batch尽量复用同一次student forward；参数、RNG、bank和optimizer不更新，optimizer calls=0。bank初始化的forward另计。

诊断内容：
- 原匹配选中的query索引及类别；每图匹配数、未匹配数。
- 按GT类别、匹配/未匹配分层的 query mask Dice、分类概率、no-object概率。
- 旧 top-1 prototype class 与匹配类别的关系，仅作为描述，不用于新训练。
- 最终semantic误差和未匹配query的贡献份额：
  令 `a_ic(v)=p_i(c)*sigmoid(m_i(v))`，在当前GT决定的未匹配集合 U 上计算
  `eta(v)=sum_{i in U,c<3}a_ic(v) / sum_{all i,c<3}a_ic(v)`。
  分母为0则null，按图像/GT类报告均值。未匹配由GT决定，只能用于分析，不能用该集合改变训练或部署聚合。
- Lset、Lreadout、10Limg 的梯度范数和余弦，按全参数及body/readout参数块报告。
- 零范数余弦为null，不伪造0；这些同训练batch的梯度不证明泛化。

匹配索引若由辅助函数导出，必须与原matcher逐项等价；不得为分析改变Hungarian目标或实际监督。

额外做一个零更新结构测试：固定 queries/prototypes/mask logits，仅改变classifier输出，验证GRQA可以不变而semantic/readout loss变化。它证明二者不等价，不证明这是R1.5失败的唯一原因。

诊断只决定是否存在实现/数据完整性问题，不按相关性、cosine正负、eta大小选择arm、调权重或停止后续科学矩阵。

## 8. 预算与完整队列

seed261：4 cells ×4 suffix arms ×1000 =16,000新正式更新。
seed262：4 cells ×(2000 shared prefix +4×1000 suffix)=24,000。
seed263：同上24,000。

合计：64,000有效最终路径更新、48个最终学生、56个新增训练任务（8新prefix+48suffix）。seed261复用的四个历史prefix不能计入本轮新增训练。

三个种子和四臂在开始前全部注册，不能因seed261好/坏选择性运行262/263。旧R1.5 confirmation仍关闭；这是新阶段的预注册多种子实验。

新增独立预算：
- 正式有效更新cap64,000；
- recovery/replay/因工程修复废弃后重做的训练attempts合计cap6,400；
- 合成optimizer attempts cap64；
- 真实L smoke optimizer attempts cap16；
- 零更新diagnostic正式batch64，工程性重做须独立记录，禁止拿重做样本作新的结果筛选。
- 全部实际optimizer calls理论上限70,480，不包含虚拟optimizer mock；mock也单列。

每模型必须仍是global step3000；预算余额不能用于延长训练或额外种子。

12小时wall-clock上限从E0冻结、连接/数据绑定完成后第一次本阶段CUDA资格或真实模型诊断开始计时。达到完整队列完成或12小时上限即收尾；不为跑满时间增加训练。若超时只得部分结果，标记incomplete，不当作科学失败或完整复验。

## 9. 调度和自主修复

检查实际可合法使用的GPU，最大并行4个workers；不抢占、终止未知进程，不假设全部A100空闲。不改系统级环境。

优先让每个cell完成B0与B2配对，再B1与B3；种子与cell按固定轮转调度。不得依据val分数改优先级。

同一错误fingerprint在同一代码版本出现两次时，暂停使用该故障路径的workers，先修复再恢复；不要像R1.5那样让相同日志异常在各worker重复消耗预算。不受影响的任务可继续。

工程修复Class E：路径、writer、日志schema、进程恢复等，不改数学路径。回归和状态等价通过后，可以明确绑定旧checkpoint与新代码续跑。

科学路径修复Class S：只允许把新实现修回本规格；不允许设计新方法。若涉及loss、数据、seed、optimizer、precision或模型前向，记录影响范围，从最早合法前缀边界重跑；不得拼接不可比轨迹。重做成本计入6,400。

不得为了出正结果修改lambda、Dice平滑、log floor、step数、augmentation、K、split、metric或precision。若必须改变这些科研定义，隔离受影响任务，输出修订提案，不擅自用新配方继续。

OOM先定位张量保留、图未释放、共享GPU占用；不单独降低某臂batch/resolution。NaN保存输入标识/状态并做有限诊断，不做静默clip或nan_to_num。数据hash/role/hidden-GT问题隔离相关任务，不能关闭检查。

预算、数据安全和无法证明一致的checkpoint仍是边界；这些不是可忽略的performance gate。

本阶段新目录磁盘使用上限32GiB；实际checkpoint大小生成后滚动估算。查询quota不可用时明确记录，不能把整文件系统df空闲当成用户配额。保留前缀、终点、latest和previous有效恢复点；禁止删除旧实验或用户数据。

## 10. 事务、checkpoint和故障分类

每次真实optimizer调用前写入不可变attempt事件；成功后才推进scheduler、bank、cursor/global_step与commit事件。

分别统计optimizer failure、Python/日志exception、replayed work、lost-tail work、successful-but-discarded work和final-path unique updates。单纯optimizer失败为0不等于工程故障为0。

新阶段保存间隔：每100成功更新；prefix/suffix边界和终点强制保存。先原子保存完整状态，再写非关键周期诊断，避免日志错误抹掉一个已完成的检查点。

checkpoint至少含student、optimizer、scheduler、实际使用的bank、全部Python/NumPy/Torch及独立generator状态、CUDA RNG、next data step、code/config/schedule/prefix SHA、precision与可训练参数清单。

临时文件在同一文件系统写入、flush/fsync、原子rename；不得在共享NFS上仅依赖含糊的全局flock保证安全。优先单一协调器序列化全局预算与任务所有权，每worker自己的append-only日志；验证租约/锁确实生效。

重启从checkpoint的完整状态恢复；物理attempt账本不能回滚。所有丢失后重放的步数收费，但最终路径不重复累计。

## 11. 日志与评价

每100成功更新记录loss分项、LR、grad norm、readout质量回退比例、GT概率下限命中比例、资源与计数。B1/B3记录bank支持情况。

梯度分项诊断每250步即可：Lset、Lreadout、10Limg；在同状态且不改变RNG/参数/bank的上下文中运行，不累计到正式.grad。诊断disabled与enabled的下一次更新要有等价测试。

训练不读取val；独立evaluator只评价预定global step3000学生，不用EMA、不选best、不早停、不按val调参。

保持原`image_dice`及`equal_domain_mean`的rim/cup、缺类、ignore与域等权逻辑。完整报告rim、cup、macro、disc_union和support；不能用disc_union取代主指标。

每个seed×backbone×domain计算：
- B2−B0：预先指定主要对比；
- B3−B1：同一readout改变在有Limg时的对比；
- B1−B0与B3−B2：Limg影响；
- B2−B1：辅助，不能拿它替换主要对比。

因子效应：
`E_readout=((B2-B0)+(B3-B1))/2`
`E_img=((B1-B0)+(B3-B2))/2`
`Interaction=B3-B2-B1+B0`。
按seed/cell/骨干及rim/cup/macro完整报告。

B2是唯一预注册主要候选。不得看结果后把B3改称主要候选；B3单独成功只允许报告预注册次要证据。

## 12. 结果判读一次完成，不用中间gate筛选种子

seed261为继续开发与历史桥接；seed262/263为预先固定的新增优化种子复验。同一已暴露split仍不是独立患者验证。

对新种子262/263的B2−B0，使用八个seed×backbone×domain cells：
1. 八cell等权macro均值>=+0.003；
2. 每个backbone在两域两seed上的macro均值>0；
3. 至少6/8 cells严格>0；
4. 最差cell>=−0.005；
5. 每个backbone各自的rim、cup等权平均delta均>=0；
6. 任一seed/domain/backbone的单类delta不得<−0.01。

这些是新阶段的研发投入阈值，不是统计显著性声明。无论261结果如何都完成既定三种子矩阵。

输出完整三种子分布、两个新种子的分布和历史桥接差异。不把同一验证图像在多个seed下的评价当作新增独立患者。若做bootstrap，用同一组按域重采样的图像索引同时作用于所有配对臂/seed，说明是给定这些模型与验证图像的条件不确定性，而非新患者或充分训练随机性推断。

满足全部标准且矩阵/状态合法：`READOUT_BASELINE_GAIN_REPLICATED`。
完整运行但不满足：`READOUT_BASELINE_GAIN_NOT_ESTABLISHED`。
未完整运行：`INCOMPLETE_ENGINEERING_OR_BUDGET`，不得做上述完整判定。

即使满足，也不启动R2/R3，不把本轮基线收益算作history-free memory、SSL或RL贡献。下一阶段是否开展当前域U/参数记忆交互由另一个明确计划决定。

## 13. 交付与最终行动

至少输出：
- `METHOD_SPEC.md`、`EXECUTION_AUTHORIZATION.json`、`TASK_PLAN.json`
- `SOURCE_BINDINGS.json`、`PREFIX_AUDIT.json`、`SEED_AND_SCHEDULE_AUDIT.json`
- `NATIVE_QUALIFICATION.json`、`LOGGING_BOUNDARY_TESTS.json`
- `READOUT_FORENSIC.md/json`、`READOUT_FORENSIC.csv`
- `RESULTS.csv`、`PAIRED_DELTAS.csv`、`FACTOR_EFFECTS.csv`
- `PER_SEED_SUMMARY.csv`、`FRESH_SEED_REPLICATION.json`
- `TASK_LEDGER.jsonl`、`PATCH_LOG.jsonl`、`STATE_AUDIT.json`
- `RESOURCE_REPORT.csv`、`FINAL_INTERPRETATION.md`

endpoint关联训练commit、checkpoint SHA、prefix SHA、schedule SHA与config SHA。最终报告区分新prefix训练、复用prefix、正式有效更新、物理调用、重放、smoke和合成调用。

GitHub只推送代码、预注册配置、去标识汇总与来源摘要。不得上传医学payload、case级数据/图像、checkpoint、密钥、私有host/path或原始日志。不给外部模型服务发送这些私有内容。 本prompt中已给出的实际运行根也不原样提交公开仓库；公共计划中用 `${PROJECT_ROOT}` 占位，真实值只在私有运行配置中保存。

读取执行环境已有模型/工具配置，使用当前用户选定的Codex模型；不要凭空猜测Astra的API model ID，也不要为了自主修复另行购买API或启用未授权服务。自主修复发生在本次真实可用的会话/执行环境内；若无法维持agent诊断，会如实说明，不能把shell重试器称为模型持续推理。

从实现、测试与E0冻结开始，资格通过后直接执行这个新阶段已注册的矩阵，不停在“准备完成等待另一次外审”。期间修复只在上面的权限与预算内进行。完整结束、预算耗尽或12小时到达后保存、评价可用终点、停止本研究workers/维护任务并报告；不要留下无截止时间的自动任务，不自动扩大研究范围。
