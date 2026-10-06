你正在 SSL_CL_seg 项目中执行下一轮实验。请完整实施以下 V7.1 计划：核验来源、实现隔离实验入口、完成资格检查、运行规定端点、统一评估、生成报告并发布匿名结果。

不要重新设计持续学习方法，不要调参寻找正结果，不要扩大本计划范围。正常的代码实现和测试不需要逐项询问；遇到下文规定的阻断条件，应保存证据并停止。

# 一、任务名称与唯一目标

实验名称：
V7.1 — Frozen-policy deployment diagnosis

本轮只回答两个问题：

1. V7 中“随机采样训练、贪心选择部署”的差异，是否对最终表现造成了可测量的损伤？
2. 改为随机部署后，冻结的 GRPO 策略是否真正优于采用同一采样实现的均匀策略，而不只是恢复到随机选择水平？

本轮不是新方法性能验证，也不是 GRPO 超参数优化。

明确禁止：
- 重新训练、微调或更新任何 GRPO 控制器；
- 切换 PPO、DAPO 或其他 RL 算法；
- 调整奖励、温度、熵系数、选择比例或窗口长度；
- 实现全池加权、更新锚定或新的 RETRIEVE 变体；
- 恢复 V6 的 10% / 5% 标注实验；
- 自动扩源种子、标注划分、数据集或连续域顺序；
- 修改 history-free / 参数作为 memory 的核心设计；
- 将 A-only 约束改成 BA 双正交；
- 使用旧域数据参与训练、策略决策、奖励计算或超参数选择。

# 二、基准代码与证据来源

仓库：
https://github.com/DLwbm123/SSL_CL_seg

公开基准提交：
03224a49cef09a3325abb7d32ffd8762e678e880

必须阅读并核验：
experiments/qprompt_rl_v1/v7_unlabeled_coreset/EXPERIMENT_PLAN.md
experiments/qprompt_rl_v1/v7_unlabeled_coreset/FROZEN_SCOPE.json
experiments/qprompt_rl_v1/v7_unlabeled_coreset/controller.py
experiments/qprompt_rl_v1/v7_unlabeled_coreset/worker.py
experiments/qprompt_rl_v1/v7_unlabeled_coreset/protocol.py

以及：
experiments/qprompt_rl_v1/v7_unlabeled_coreset/results/pilot_20261005/

重点读取其中：
RESULT_REPORT.md
RESULTS.csv
COMPLETION_AUDIT.json
GROUP_DIAGNOSTICS.json
COSTS.json
DECISION.json
PUBLICATION_RECEIPT.json（若存在）

同时核验所引用的原生学生、数据、损失、优化器和状态恢复实现。

注意：
- 公开结果提交不等于原始训练代码提交。
- V7 报告记录的训练提交为：
  37ac930652dc2da6585342a437d014ae3b4b995d
- 对 checkpoint 必须核验实际训练来源，不能要求它错误地匹配后续报告发布提交。
- 通过原有运行记录定位 NAS 上的 campaign、源模型和四份冻结策略，不得猜测路径。
- 当前运行底座是原实验的 NATIVE_LR_SRC_A 参考实现，不得把它直接写成原始 KI 的精确复现。

建立独立分支和独立实验目录，例如：
codex/v7_1-frozen-policy-deployment
experiments/qprompt_rl_v1/v7_1_frozen_deployment/

不得覆盖 V7 代码、checkpoint、结果或原始判定。

# 三、严格冻结的实验条件

所有新增端点必须保持：

源模型：
- 同一个 REFUGE 源 checkpoint；
- 源训练种子 168；
- 源训练 8,000 次更新；
- 源模型哈希与 V7 一致。

两个独立适应任务：
- REFUGE → RIM_ONE_r3；
- REFUGE → Drishti_GS。

不得把这两项写成顺序三域持续学习。

数据与角色：
- 继续使用原始 20% 标注池；
- 角色划分种子沿用 V7 的 7101；
- RIM：16 个标注样本，其中 12 个学生训练、4 个奖励保留；63 张 U；
- Drishti：10 个标注样本，其中 8 个学生训练、2 个奖励保留；41 张 U；
- 以原始患者角色 manifest 核验，不仅核对数量；
- 本轮所有新增端点都不得将奖励保留样本重新加入学生训练；
- 不读取 U 的隐藏标签；
- 不访问 sealed test。

训练：
- RIM：3,200 次学生更新；
- Drishti：2,100 次学生更新；
- 每 100 步重新选择子集；
- RIM 每窗口选择 16 张 U；
- Drishti 每窗口选择 11 张 U；
- lambda_U = 0.5；
- 原生三分类 consistency KL；
- 原置信度门槛 0.7；
- 原生 Adam、学习率计划、增强、EMA、batch、损失归一化和 A-only 投影全部保持不变；
- 保留原始 steps_per_epoch；
- 入选集合仍按原实现排序后进入原生 loader；
- 不改变学生 RNG 或数据增强规则；
- 不改变原生最终学生的部署方式。

四份控制器：
- RIM × 601；
- RIM × 602；
- Drishti × 601；
- Drishti × 602。

必须使用 V7 已训练完成的 FROZEN_POLICY。
不存在、损坏或来源无法核验时，停止，不得自动重训。

# 四、先生成来源与范围审计

在性能训练之前，输出并冻结：

RUN_MANIFEST.json
PROVENANCE_AUDIT.json
FROZEN_SCOPE.json
EXPERIMENT_PLAN.md

至少记录：
- 代码基准提交及本轮执行提交；
- 源模型哈希；
- 四份控制器哈希；
- 数据与角色 manifest 哈希；
- 原生训练配置；
- RNG 规则；
- 八个端点的完整任务表；
- 物理更新预算；
- 评估顺序；
- 判定门槛；
- 环境和硬件信息。

来源或角色不一致时，不得静默替换成“相近配置”。

原始 ALL_U、RANDOM_100、GRPO_GREEDY，以及其他 V7 非 RL 端点，只能在来源、数据角色、学生训练配置和评估定义可核验一致时作为历史对照复用。

若关键对照无法核验：
- 输出 BLOCKED_BASELINE_PROVENANCE；
- 列出缺失证据；
- 不得自行增加补跑端点突破预算。

# 五、阶段 A：零优化器更新的机制审计

这一阶段允许必要的只读前向和反向诊断，但：
- 学生 optimizer.step = 0；
- 控制器 optimizer.step = 0；
- 不实施虚拟参数更新；
- 不新增学生训练轨迹。

只读诊断必须恢复模型模式、buffer、EMA 状态、RNG、provider 状态和曝光计数，不得影响后续训练。

物理成本计数器不能随模型状态恢复而回滚。

## A1. 策略分布与贪心行为

使用确实存在、可准确恢复的 V7 entry / endpoint / latest 状态。
不要假定中间窗口 checkpoint 全部存在。

在同一个学生状态上比较策略分布与均匀分布，记录：
- 每个选择位置的合法候选数；
- 条件熵 H；
- 均匀熵 log(N-j)；
- 相对均匀分布的 KL：log(N-j)-H；
- 有限 logits 的标准差；
- 最大动作概率；
- 贪心选择时第一名与第二名的 logit 间隔；
- 随机采样集合间的 Jaccard；
- 可获取状态之间的贪心集合稳定性。

注意策略是 set-conditioned：
- 后续 logits 依赖已选集合；
- 不能把整个过程误写成一次静态 top-K 排序；
- logit 间隔按每个选择位置记录。

不能仅凭训练熵较高就认定熵正则过强。
不能仅凭贪心重复就认定训练分布坍缩。

若 group_latest 保存了所需状态，可仅用已保存特征、动作和奖励，分离计算策略项与熵项的梯度范数及夹角。
只反向，不更新；缺少必要状态则明确标记 unavailable。

## A2. 实际曝光诊断

优先读取原始私有选择记录和实际曝光记录，计算：
- 累计覆盖率；
- 相邻窗口 Jaccard；
- 每张图像累计曝光次数；
- 曝光次数的 min / median / max；
- 曝光变异系数；
- 每个窗口及全程的有效样本数：

  N_eff = (sum_i c_i)^2 / sum_i c_i^2

- 长期未被使用的图像比例；
- 连续窗口重复暴露情况。

c_i 必须对应实际训练曝光，不是特征提取访问次数。

没有实际曝光记录时：
- 可以通过原 loader 的索引规则做零训练索引重放；
- 必须标注 reconstructed，不得冒充原始实测；
- 无法重建时如实报告缺失。

不得把“累计覆盖全部 U”解释为“窗口内曝光均衡”。

## A3. 奖励分解与解释边界

优先分析现有奖励记录；仅在已有状态足够时做只读重算。

分别报告：
- 标注质量项；
- 源预测 KL 项；
- 两项的尺度和变化；
- 可获得时，背景 / rim / cup 的贡献及有效像素占比。

不得：
- 重新选择源 anchor；
- 改置信度阈值或 KL 分母；
- 调整 0.1 的源 KL 系数；
- 用新诊断结果修改控制器；
- 将目标 U 上的源预测 KL 宣称为旧域保持保证。

若只保存了总奖励而没有对应前后状态，明确说明无法完整分解。
不允许为补齐诊断自动重跑 V7 训练。

阶段 A 输出：
MECHANISM_AUDIT.md
POLICY_DISTRIBUTION_DIAGNOSTICS.json
EXPOSURE_DIAGNOSTICS.json
REWARD_COMPONENT_DIAGNOSTICS.json

病例 ID、图像索引映射和私有路径不进入公开报告。

# 六、阶段 B：实现两个新增部署组

只新增以下方法。

## B1. GRPO_SAMPLE

- 加载对应域和控制器种子的 V7 冻结策略；
- 每 100 步使用当前学生状态提取原有特征；
- 按原策略逐位置无放回随机采样 K 个图像；
- temperature 固定为 1；
- greedy=False；
- 不增加 epsilon-greedy、重采样、温度退火或覆盖补丁；
- 不访问奖励标签来决定动作；
- 整个端点期间策略权重保持完全不变。

注意：
这仅改变动作选择规则，不能宣称已经消除所有训练—部署分布差异。
最终冻结策略从源状态部署，与训练中逐步变化的行为策略仍有区别。

## B2. UNIFORM_MATCHED

- 使用与 GRPO_SAMPLE 完全相同的逐位置无放回采样函数；
- 每个选择位置，未选候选的 logits 恒为零；
- 已选候选正确 mask；
- K、窗口长度、排序和原生训练路径完全一致；
- 不能用另一套 numpy permutation 实现代替；
- 不引入模型学习或额外评分。

这组是主要的“是否学到有效选择”对照。

不要为了表面计算量一致，强行给均匀策略增加无用的特征提取。
真实选择开销分别记录。

## B3. RNG 配对与隔离

采样使用私有 RNG，不能消耗学生 RNG。

建议固定种子键：
stable("V7.1/deploy", domain, controller_seed, block)

要求：
- stable 使用项目已有稳定哈希机制，不用 Python 内置 hash；
- 同一 domain / controller_seed / block 的两个新增方法使用同一私有随机流规则；
- 种子键中不包含方法名；
- 选择规则唯一差异是 logits；
- 学生轨迹分叉后，各方法从自己的实际状态提取特征和记录曝光，不强行共享学生状态。

必须测试：
当 GRPO_SAMPLE 的输出 logits 人为置零时，它与 UNIFORM_MATCHED 在同一输入和 RNG 下逐位置得到完全相同的集合。

# 七、资格检查与严格预算

## C1. 不产生优化器更新的测试

至少包括：
- 选择数量正确、索引合法、无重复；
- logits mask 正确；
- 全零 logits 的精确采样一致性；
- 私有 RNG 不改变学生 RNG；
- 保存恢复后的采样结果一致；
- 特征提取不改变训练状态；
- 冻结策略不存在梯度或参数变化；
- ALL_U 的 batch 路径仍与原生实现一致；
- U 标签不可访问；
- 日志、计数和结果序列化完整。

## C2. 原生训练恢复检查

只允许以下 32 次学生资格更新：

2 个域 × 2 个新增方法 ×
[连续运行 4 步 + 从相同 entry 运行 2 步、保存恢复后再运行 2 步]
= 32 次。

可使用控制器种子 601 执行原生恢复检查；
其余冻结策略必须完成零更新加载与采样检查。

对比连续 4 步和 2+2 恢复路径的：
- 学生参数和 buffer；
- EMA；
- Adam；
- scheduler / scaler；
- step / cursor；
- RNG；
- 入选集合；
- 曝光；
- provider 状态。

依照原项目精确恢复标准判定。
不一致时停止，不得先启动完整训练再解释。

这些资格轨迹不得作为正式端点的热启动。
正式端点必须重新从共同源 entry 开始。

## C3. 八个新增端点

RIM_ONE_r3：
- GRPO_SAMPLE / 601 / 3200 steps
- GRPO_SAMPLE / 602 / 3200 steps
- UNIFORM_MATCHED / 601 / 3200 steps
- UNIFORM_MATCHED / 602 / 3200 steps

Drishti_GS：
- GRPO_SAMPLE / 601 / 2100 steps
- GRPO_SAMPLE / 602 / 2100 steps
- UNIFORM_MATCHED / 601 / 2100 steps
- UNIFORM_MATCHED / 602 / 2100 steps

新增端点学生更新：
4 × 3200 + 4 × 2100 = 21,200

本轮总预算：
- 只读审计：0 次优化器更新；
- 原生资格检查：32 次学生更新；
- 正式端点：21,200 次学生更新；
- 合计最多：21,232 次学生更新；
- 控制器更新：0；
- 额外训练重放：0。

任何采用真实学生的测试更新均计入该预算。
不得把恢复、失败尝试或验证性训练排除在物理账本之外。

资格检查失败或正式作业失败：
- 保存已有日志和实际消耗；
- 暂停依赖作业；
- 不自动重试；
- 不通过覆盖目录抹除失败成本；
- 不静默扩充预算。

# 八、执行与状态管理

从原实验记录确认可用计算环境。
沿用原获准 GPU 范围；若记录确认为 5/6/7，则只使用空闲且可合法使用的这些 GPU。
不得停止、抢占或干扰其他作业。

数据、原始 checkpoint、缓存和私有运行结果留在 NAS。
不得将图像、患者信息或模型权重上传 GitHub。

每个正式端点：
- 从同一个源状态初始化；
- 使用各自独立输出目录；
- 按固定训练长度运行；
- 不 early stop；
- 不根据奖励、验证集或旧域表现选择 checkpoint；
- 最终学生使用预登记的训练末端 checkpoint；
- 每窗口保存选择摘要与必要恢复状态；
- 策略哈希在开始和结束时一致。

账本必须独立于可恢复的训练状态，恢复不能回滚物理成本。

持续维护：
STATUS.json
PHYSICAL_LEDGER.jsonl
JOB_MANIFEST.json
COSTS.json

记录实际开始、结束时间和明确时区；同时保存 UTC。
不得根据文件名推断实际时间。
不得用“后台已启动”或“进程存在”代替实验完成。

# 九、统一评估与数据隔离

八个端点全部训练结束并冻结后，才启动本轮最终评估。

训练与机制诊断阶段：
- 不新增读取旧域原始图像或标签；
- 不读取历史开发验证集来选择方法或 checkpoint；
- 可以读取已发布的 V7 汇总指标；
- 奖励角色标签仅可用于上述预登记的只读奖励诊断，不能用于新端点动作或训练。

最终评估：
- 沿用 V7 的历史开发验证集及同一评估程序；
- 不访问 sealed test；
- 同时评估对应新域和旧 REFUGE；
- 核验共同源模型的旧域分数；
- 不将每次目标域适应误写成顺序三域训练。

至少报告：
- 新域 macro Dice；
- rim Dice；
- cup Dice；
- disc union Dice；
- 旧域 REFUGE Dice；
- 相对源模型的绝对遗忘；
- 相对 ALL_U 的旧域差值；
- 累计 U 覆盖及曝光统计；
- 训练、特征提取、选择、诊断和评估成本。

绝对遗忘定义：
D_old(source) - D_old(endpoint)

必须区分：
“相对 ALL_U 没有进一步退化”
与
“相对源模型没有遗忘”。

# 十、预登记比较与判定

所有门槛使用 Dice 百分点 pp。
内部 Dice 使用 0–1 时必须显式换算：
0.005 = 0.5 pp。

每个 domain × controller_seed 构成一个配对，共四对。
这些不是四个独立源模型重复。

## D1. 部署方式比较

对每对报告：

Delta_deploy_new =
D_new(GRPO_SAMPLE) - D_new(V7_GRPO_GREEDY)

Delta_deploy_old =
D_old(GRPO_SAMPLE) - D_old(V7_GRPO_GREEDY)

报告四个原始差值、分域均值和整体等权均值。
不能只报告平均正向结果。

随机部署优于贪心，只能支持部署修复解释，
不能单独支持“策略学习有效”。

## D2. 主要学习效用比较

对每对报告：

Delta_learn_new =
D_new(GRPO_SAMPLE) - D_new(UNIFORM_MATCHED)

Delta_learn_old =
D_old(GRPO_SAMPLE) - D_old(UNIFORM_MATCHED)

预登记“值得保留的学习筛查信号”：
- 新域四对平均提升至少 +0.5 pp；
- 至少 3/4 对新域差值为正；
- 每个控制器种子跨两个域的新域平均差值为正；
- 相对 UNIFORM_MATCHED，旧域平均差值不低于 -0.5 pp；
- 任一配对的旧域差值不低于 -1.0 pp。

这是后续研究筛查门槛，不是统计显著性结论。

未达到门槛时，应写“未建立有实际意义的学习优势”，
不能据此证明两种方法完全等价。

## D3. 实际方法价值与 V7 原标准

除上述匹配均匀对照外，继续报告对 V7 原非 RL 对照的比较。

每对的非 RL 参考分数取以下可比对照中的最高新域分数：
- NO_U；
- ALL_U；
- CONFIDENCE；
- KCENTER；
- RETRIEVE_FO；
- 对应种子的 V7 RANDOM；
- 本轮对应的 UNIFORM_MATCHED。

该参考仅用于冻结后的结果判定，不能用于训练选择。

不得降低 V7 原来的筛查要求：
- 相对最佳可比非 RL 对照，新域平均至少 +0.5 pp；
- 相对原匹配 RANDOM，新域平均至少 +0.5 pp；
- 相对 ALL_U，旧域平均差值不低于 -0.5 pp；
- 任一端点相对 ALL_U 的旧域差值不低于 -1.0 pp；
- 任一端点相对最佳可比非 RL 的新域差值不低于 -0.25 pp；
- 每个控制器种子跨两域的平均新域优势为正。

V7.1 即使出现正向结果，也不得追溯性修改 V7 的原判定。

## D4. 最终解释

优先按以下证据结构解释：

1. 随机部署改善，但没有通过匹配均匀对照的学习门槛：
   DEPLOYMENT_RECOVERY_ONLY
   或在部署改善也不明确时：
   NO_POSITIVE_EVIDENCE

2. 通过匹配均匀对照的学习门槛，但未通过 ALL_U 旧域保持门槛：
   LEARNED_SIGNAL_RETENTION_FAIL

3. 通过匹配均匀对照和 ALL_U 旧域保持，但未达到最佳非 RL 的方法价值门槛：
   LEARNED_SIGNAL_NOT_COMPETITIVE

4. 全部门槛通过：
   SCREENING_CANDIDATE

同时输出每项 gate 的布尔值和原始指标。
单一状态名称不能隐藏混合结果或部分改善。

无论哪个状态，都只结束本轮，不自动开启下一阶段。

# 十一、统计与成本解释

必须写明：
- 一个固定源模型；
- 两个域；
- 两个控制器种子；
- 两个控制器种子不等于独立源训练重复；
- 同一历史开发患者集不能支持独立患者泛化确认；
- 四个配对端点不能当成四个独立临床样本做显著性推断。

不通过选择某一个域、某一个控制器种子或某一个通道，替代全体预登记结果。

成本分开报告：

A. 本轮新增成本：
资格、部署、诊断、评估的实际消耗。

B. 方法完整成本：
明确 GRPO 策略来自 V7 训练；必要时列出可归属的既有策略学习成本。
不能因为本轮复用策略，就宣称 GRPO 不需要策略学习开销。

C. 部署成本：
分别比较 GRPO_SAMPLE、UNIFORM_MATCHED 与可比历史方法。
硬件占用或环境差异影响墙钟可比性时必须注明。

不能把 25% 子集、相同学生步数解释为已经实现训练加速。

# 十二、最终交付

至少生成：

EXPERIMENT_PLAN.md
FROZEN_SCOPE.json
PROVENANCE_AUDIT.json
RUN_MANIFEST.json
QUALIFICATION.json
MECHANISM_AUDIT.md
RESULTS.csv
PAIRED_COMPARISONS.csv
POLICY_DISTRIBUTION_DIAGNOSTICS.json
EXPOSURE_DIAGNOSTICS.json
REWARD_COMPONENT_DIAGNOSTICS.json
COSTS.json
DECISION.json
COMPLETION_AUDIT.json
FINAL_INTERPRETATION.md
PUBLICATION_RECEIPT.json

公开报告必须包含：
- 八个新增端点的全部结果；
- 被复用的 V7 对照及其来源；
- 四对 GRPO_SAMPLE vs UNIFORM_MATCHED 的逐对差值；
- 四对 GRPO_SAMPLE vs GRPO_GREEDY 的逐对差值；
- 新域与旧域两条结果；
- 实际曝光与分布诊断；
- 所有缺失诊断和解释限制；
- 新增与完整方法成本；
- 每项预登记门槛是否通过；
- 停止状态。

公开前检查：
- 没有图像、患者 ID、逐病例敏感结果；
- 没有模型权重和私有 checkpoint；
- 没有访问令牌或私有机器路径；
- 原 V7 结果未被覆盖。

将本轮代码、冻结协议、测试、匿名汇总和报告提交并推送独立 GitHub 分支。
核验远端 SHA；公开仓库匿名访问成功后才能声称完成发布。
推送或匿名访问失败时，分别报告已完成和未完成项，不得伪称交付成功。

最终回复按以下结构给出：
1. 实验状态及实际完成数量；
2. 四对主要比较及总体结论；
3. 是否支持部署不匹配解释；
4. 是否建立超越均匀采样的学习收益；
5. 是否通过旧域保持和最佳非 RL 门槛；
6. 学生与控制器实际更新次数；
7. 成本与局限；
8. 分支、提交 SHA、报告地址；
9. 明确说明没有启动任何下一轮实验。

现在先核验 V7 运行来源并冻结本轮计划。
所有资格检查通过后，在上述预算内执行到报告完成。
不要增加任何未列明的实验分支。