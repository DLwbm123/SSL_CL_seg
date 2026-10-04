# V5_GROUP_RL frozen preregistration

Baseline: `90eacbacca00d3c03e1296d5c7e6ac22a047df3b`. The full authorized specification follows without changing its thresholds. All added choices below are preregistered implementation decisions, not validated optima.

## Concrete implementation choices

- P0 native budget exactly 690 updates: unchanged V4 qualification (390) plus two domains, two exact replays of the mixed 0/.125/.5 three-block sequence (300). No performance experiment starts until every qualification passes.
- CPU synthetic qualification: PPO_MATCHED, GRPO_STD and GRPO_FS; fixed 256 groups per task, G=4, four decisions per episode, 16 actor calls/group. Immediate reward for action 1; delayed reward at step 4 for first action 1. An additional fixed 256-group delayed task per domain/method starts from the softened real OFFLINE_25 copy and fixed first-panel state, with target action one past its original argmax. No real rewards used. These calls are separately counted, never charged as student updates.
- Initialization panel: every V4 h25 development panel state, ordered seed 168/169/170 then increasing step; no filtering. Per-domain V4 actor/scaler reused. Exact 64 bisection iterations, positive last-layer scaling, no score input.
- Development entry for each group is the native source-to-stage1 state. Environment seed equals segmentation seed only. Independent action streams use controller seed, domain, group, trajectory; method name is omitted to match random draws where distributions coincide. Minibatch/shuffle streams are separate. Calibration stream fixed at seed 400.
- Six U=0 reference entries per domain are computed as budgeted, even when cyclic source seeds repeat. A reference is shared across algorithms and controller seeds only after comparison of complete entry state, fixed options, Provider identity, source receipt, frozen manifest/split, implementation commit and group. P1 references are separate.
- GRPO group advantage is terminal relative online quality; PPO/Bandit use raw online block increments divided by the frozen per-domain scale. Bandit advantages are not additionally standardized; PPO advantages are normalized across all four trajectories.
- Full group decision mean, equal length trajectories, equal minibatches. Separate actor/critic Adam and separate norm clips; warm-up critic moments do not seed training optimizer (training starts a fresh lr=.0003 Adam for all methods). GRPO has no instantiated critic.
- All pilot training jobs must finish and a global endpoint lock must exist before any pilot val process starts. The same barrier applies to confirmation. Val evaluator is a separate process; its aggregate outputs are used only by the preregistered decision code.
- Source seeds 183–187 are frozen as not participating in this round of development. Bounded source receipt search found no matching source directories but encountered an inaccessible historical directory, so project-wide novelty is not established. No replacement based on results.
- Only physical GPUs 5/6/7; at least 6 GiB free before each job. No deadline, automatic retry, resampling, background monitor or external scheduler automation. A finite dependency queue runs autonomously through authorized gates. Engineering failure stops dependent launches and preserves evidence.
- Storage planning upper reserve 80 GiB: one entry and latest complete student fork per active job, fixed final policy per learner plus small group optimizer states, source and endpoint models, private trajectories/diagnostics. Completed rollout student checkpoints are not all retained. Existing historical artifacts untouched; no copying dataset/model caches. This is a conservative allocation, not measured output size. Require NAS mount/write/read probe and 80 GiB free before launch.
- Actual accelerator active cost is measured with per-process CUDA event intervals around student updates/features/reward evaluation, synchronized at accounting boundaries; sum of operation intervals is not kernel-only occupancy or energy. Coordinator resource samples and job wall time are separately retained. Evaluator wall/peak and label access are separately recorded.
- New snapshot wrapper adds options, physical update count, teacher modes, Provider immutable identity and dataset checked sets to the V4 full state. Native algorithm unchanged. Diagnostic read-only wrappers restore these bookkeeping fields as well.
- U0 cache/source identity hashing is required protocol provenance, not a full NAS file audit. Private role maps, state panels, models, trajectories and case scores never enter Git.

## Pre-performance engineering correction

Reward-shuffle permutations use their own `RandomState(seed ^ 0x5A17)`, leaving the shared minibatch stream unchanged. The original qualification worker completes once (690 real updates); its immutable receipt is adopted into the final campaign without rerunning or resetting those updates. One additional synthetic four-trajectory shuffle update (16 actor calls, zero real student updates) checks this correction before performance launch. Both source commits and the dependency hold are retained. No reward/performance result determined this correction.

## Authorized specification

请在 SSL_CL_seg 项目中设计、实现并执行下一轮实验：

V5_GROUP_RL：
冻结现有分割底座，验证同起点、多轨迹、无 critic 的组相对策略学习，
能否超过原始监督 ORIGINAL，并建立相对匹配 PPO/Bandit 的增益证据。

不是泛泛讨论方案，也不是只把 PPO 的 loss 替换成 GRPO。
先核验现有源码、NAS 产物和数据角色，再形成冻结计划、实现与资格测试。
通过工程资格后，执行下面已经限定范围的实验。

所有新增超参数和判据均属于本轮预注册设计，不是已证实的最优配置。
不得根据本轮性能结果偷偷修改它们。


一、仓库、工作范围和不可变约束

仓库：
DLwbm123/SSL_CL_seg

基线提交：
90eacba
开始时解析并记录完整 commit SHA，不能只记录可移动分支。

首先阅读：
experiments/qprompt_rl_v1/v4_rl_block_control/EXPERIMENT_PLAN.md
experiments/qprompt_rl_v1/v4_rl_block_control/controller.py
experiments/qprompt_rl_v1/v4_rl_block_control/worker.py
experiments/qprompt_rl_v1/v4_rl_block_control/protocol.py
experiments/qprompt_rl_v1/v4_rl_block_control/reports/FINAL_INTERPRETATION.md

以及上述文件实际引用的 engine、数据角色、原生优化器、投影、
checkpoint、evaluator 和 NAS 存储约束。

建议新分支：
codex/v5-grpo-group-control

建议新增目录：
experiments/qprompt_rl_v1/v5_grpo_group_control/

如果名称已存在，先检查内容，不能覆盖既有实验。

必须保持：
1. history-free 设计，不新增旧域样本、特征、梯度、原型或 replay buffer。
2. 已训练参数作为跨任务 memory 的基本思路。
3. V4 的学生结构、可训练参数集合、参数保护/正交投影和原生优化器。
4. 当前监督损失、U 损失、teacher EMA、伪标签阈值和增强。
5. 原始训练标签集合与患者划分。
6. 每域分别从 REFUGE 源模型进入 stage 1 的实验设置。

不要根据印象把当前实现改成 A-only 或 BA 双正交。
先核对现有实现与冻结基线；本轮不比较或修改这些机制。

若原始 KI 的历史身份仍未核实，报告只能称“V4 底座上的增量实验”，
不能声称已经复现原始 KI 或原论文 SOTA。

当前任务内的临时分叉 checkpoint 和策略训练轨迹允许用于实验，
但不得把它们变成部署时或下一域所依赖的旧数据 memory。
明确区分研究归档与方法运行时的持久化需求。


二、数据角色、动作与终点

开发源种子沿用：
168、169、170。

确认阶段使用五个未参与本轮策略开发的新分割源种子。
默认候选为 183–187；必须先核对项目种子登记和历史回执。
若已使用，按预先声明的规则选择未使用种子，并在读取本轮性能前冻结。
不得根据结果替换种子。
若不能核实全项目使用历史，诚实标注“未参与本轮开发”，
不能直接宣称“项目历史全新”。

独立控制器训练种子：
401、402、403。

控制器种子用于动作采样和控制器优化随机性；
不能同时改变匹配方法之间的学生环境随机数。

开发沿用 V4 的 fit/online/audit：
fit：学生梯度和状态特征；
online：控制器奖励；
audit：仅描述性诊断，不选择方法、回合、阈值或超参数。

本轮不重划患者，不扩大 online 患者集合。
这保留了奖励覆盖有限的局限，但避免与算法改动混杂。

val 可在完整 pilot 冻结后用于研发晋级判断。
必须明确：val 已被历史研发使用，pilot 晋级也会使用它；
后续新种子确认不是独立患者泛化验证。
test 全程封存。
U 隐藏标签不得读取。
旧域标签不得进入学生训练、控制器奖励、特征、归一化或策略更新。

动作固定为：
a=0 → lambda_U=0
a=1 → lambda_U=0.125
a=2 → lambda_U=0.5

每个动作持续 25 个真实学生更新。

完整日程：
RIM_ONE_r3：3200 步，即 128 次决策；
Drishti_GS：2100 步，即 84 次决策。

不得用短窗口收益、1200 步快照或开发轨迹终点代替正式终点评估。


三、P0：基线审计与工程资格

在性能实验前完成：

A. 基线等价性
确认 lambda_U=0 与 V4 ORIGINAL 原生路径一致。
核验学生、EMA teacher、Adam、scheduler、参数保护、数据流和 RNG。
核验状态特征提取不会改变上述状态。

B. 分叉恢复
完整分叉状态必须包含：
学生参数及 buffers、teacher、optimizer、scheduler、AMP scaler（若有）、
各类 RNG、数据采样器/Provider 状态、训练步数和必要计数器。

同一状态恢复两次、执行同一动作序列，应得到一致结果。
不能只恢复模型权重而漏掉 Adam 动量或 teacher。

C. 概率与梯度
检查 behavior log-prob 与真实采样分布一致；
old log-prob、奖励、优势和学生状态特征均正确 detach；
控制器反向不得进入学生网络。

D. 数学测试
覆盖：
组奖励平移不变性；
全相同奖励时组优势为零；
组内标准差使用 population std；
终止边界与 GAE；
正负优势下的 PPO clip 方向；
打乱奖励保持组内多重集；
控制器及 optimizer 的精确恢复；
固定 U=0 参考不进入 on-policy 样本集合。

E. 合成可学习性
构造一个即时收益任务和一个具有延迟收益的多步任务。
分别检查 PPO、GRPO_STD、GRPO_FS 的实现能学到预设信号。
另外从真实离线初始化的副本出发，检查预处理后能否改变错误动作偏好。
合成测试不能使用真实 audit/val 或 U 隐藏标签。

真实学生工程资格预算上限 1500 次更新；
合成控制器调用单独计账。
资格测试的具体数量在启动前写入计划。

工程失败停止依赖阶段。
不能用“结果不符合预期”当作代码错误，自动调参后重跑。


四、共同初始化与 P1 校准

本轮所有可学习方法使用相同的 V4 OFFLINE_25 actor 和固定状态归一化。

为检验饱和初始化问题，统一执行一次不使用奖励的初始化软化：

在预先固定的开发状态面板上，计算 actor softmax 的平均熵。
若低于 0.8*log(3)，仅将 actor 最后一层 weight 和 bias 同乘正数 alpha。
用固定二分程序寻找 alpha∈(0,1]，使平均熵达到目标。
若原熵已达标，alpha=1。

记录面板来源、alpha、软化前后熵和 argmax 一致率。
该操作不能使用 online/audit/val 分数选 alpha。
所有新学习器使用同一个软化后初始化；不对某一种算法单独处理。

本轮新学习器直接从 softmax 策略采样，
不再叠加 V4 的 10% 均匀混合分布。
采样、log-prob、重要性比和熵必须使用同一个分布。

这属于 V5 公共训练配方改动。
不能把 V5 与 V4 的差异全部归因于 GRPO。

P1 校准：
每个开发源种子、每个域，从相同完整入口状态运行：
4 条共同初始化策略采样的完整轨迹；
1 条完整 U=0 参考轨迹。

共 3 源种子 × 2 域 × 5 条轨迹。
策略不更新。
分支顺序不能影响结果，动作 RNG 与环境 RNG 分离。

设每组第 i 条轨迹的未缩放相对收益为：
R_i = Q_online(theta_T_i) - Q_online(theta_T_U0)。

每域冻结一个奖励尺度：
S_R = max(
    1e-4,
    sqrt(mean_over_calibration_groups_and_i((R_i - group_mean_R)^2))
)。

只能用校准阶段 online 反馈确定 S_R，
不能在后续训练中动态重估或按算法分别重估。

Q 沿用 V4 的反馈质量定义，不是 Dice 或 Dice 百分点。
不得把奖励数值当作终点分割改善。

PPO 和 Bandit 的 critic 可以使用这些共同校准轨迹进行
固定 64 次 critic-only warm-up，actor 不更新：
PPO critic 拟合 Monte Carlo return-to-go；
Bandit critic 拟合当前块奖励。
warm-up 学习率固定 0.001，单独记账，不按 audit 选轮数。

同时复查 V4 动作面板：
将原先相对 FINE_05 的收益重新表达为相对同状态 U=0 的收益。
报告 online 选择能否在 audit 上复现。
这部分仅诊断，不据此筛掉不利状态或改变主候选。


五、组轨迹采样与算法定义

一个 group 的正确含义：
同一个完整学生入口状态，
在同一冻结 behavior policy 下独立采样 G=4 条完整控制轨迹。

四条轨迹必须全部完成后，才能更新控制器。
不能一边生成组内轨迹一边改变 behavior policy。
不能把不同入口、不同源种子或不同训练进度的轨迹混成一个组。

组内采用共同环境随机数以减少比较噪声，
但每条轨迹的动作采样流独立。
不得强制四条动作序列不同，不得拒绝重复轨迹后补采样。
不得为了制造严格共同随机数，破坏 lambda=0 的原生训练路径。
如只能保证同入口而不能保证全过程同 batch/增强，必须报告这一限制。

每个开发 group 另有同入口完整 U=0 参考。
它只用于解释相对收益，不属于策略采样的 G 条轨迹。
相同完整入口、数据流和实现 hash 下，可跨算法、控制器种子共享参考。
共享必须有明确缓存键和 provenance，不能因为同一个源种子就随意复用。

重要：
共同减去 U=0 参考不会改变组内中心化优势。
它让“是否超越监督”更清晰，不会自动解决奖励失配。

必做学习器：

1. PPO_MATCHED
完整回合，gamma=1，GAE lambda=0.95，终点 bootstrap=0。
块奖励：
r_t = (Q_after - Q_before) / S_R。
检查回合奖励望远镜恒等式。
优势在当前四轨迹批次内统一标准化。

2. BANDIT_MATCHED
与 PPO 相同采样和更新预算。
advantage = 当前块奖励 - V(s_t)。
critic 只拟合当前块奖励，不传播未来回报。

3. GRPO_STD
不使用 critic。
A_i = (R_i - mean(R)) / (population_std(R) + 1e-8)。
同一轨迹的所有决策使用该轨迹的组相对终局优势。

4. GRPO_FS —— 本轮预先指定主候选
不使用 critic。
A_i = (R_i - mean(R)) / S_R。
不再进行逐组或逐 minibatch 的标准差归一化。
这是固定尺度组相对变体，不直接命名为完整 Dr. GRPO。

5. GRPO_FS_SHUFFLE
与 GRPO_FS 完全相同，但在每个 group 内均匀随机排列
完整轨迹与终局奖励的对应。
允许排列包含不动点，不强制 derangement。
每组只抽一次排列，在该组所有优化 epoch 内保持固定。
保留组内奖励多重集，但破坏真实轨迹与回报的对应。

共同 actor 目标：
rho_it = pi_theta(a_it|s_it) / pi_behavior(a_it|s_it)

L_actor =
- mean_it[min(rho_it*A_it, clip(rho_it,0.8,1.2)*A_it)]
- 0.01*mean_it[entropy(pi_theta(.|s_it))]

每域轨迹长度固定，明确 loss 分母。
reference-policy KL 系数固定为 0，旧/新策略 KL 仍必须记录。
说明这是本任务的匹配控制实现，不是照搬原论文全部配置。

全相同组奖励不补采样、不删除；
奖励策略梯度为零，但保留预设熵项并如实记账。


六、P2：有限开发矩阵与 pilot

第一轮只训练控制器种子 401。

每域、每种学习器：
6 个 group；
开发源种子按 168、169、170 循环两次；
每组 4 条完整轨迹；
共 24 条完整学生轨迹。

每组做 4 个优化 epoch；
每个 epoch 将整个组的决策随机分成 4 个等大小 minibatch；
共 16 次 actor 更新/group，即 96 次 actor 更新/学习器/域。

actor Adam lr=0.0003，gradient norm clip=1。
PPO/Bandit 的 critic 使用独立 optimizer 和独立梯度裁剪，
训练期 lr=0.0003，value loss coefficient=0.5。
不得由 critic 的大梯度通过共享梯度裁剪压缩 actor 更新。

记录 PPO/Bandit 的额外 critic 计算；
不宣称控制器计算完全相同。
核心公平口径是相同学生 rollout 数、学生更新数和奖励访问机会。

所有学习器固定使用第 6 个 group 更新后的策略。
不按最佳 online/audit 回合挑 checkpoint。
完成整个开发矩阵后再做 pilot，不能边看 val 边改训练。

pilot 使用三个开发分割种子、两个域，全部运行完整原生日程。
当前域训练使用与正式部署一致的完整 L/U，控制器冻结。
每个 seed/domain 运行以下九组：

ORIGINAL
FINE_05
OFFLINE_25
PPO_MATCHED
BANDIT_MATCHED
GRPO_STD
GRPO_FS
GRPO_FS_SHUFFLE
PHASE_SHUFFLE_FS

PHASE_SHUFFLE_FS：
取同一单元 GRPO_FS 的实际动作序列，
在早、中、晚三个按 block 划分的阶段内独立打乱。
保持每阶段动作计数，不声称保持梯度范数或实际效用。
它不是 group 内奖励打乱的替代。

共 54 个 pilot 终点。
全部模型/策略冻结并完成训练后，才由独立 evaluator 读取 val。
部署动作统一 argmax，不根据 pilot 改成随机采样部署。


七、pilot 晋级规则

完成全部必做 pilot 单元后，只按预注册主候选 GRPO_FS 判断。

以下条件同时满足，才进入扩展控制器复现与五种子确认：

A. 相对 ORIGINAL：
新域均值至少 +0.10 个百分点；
旧域均值不下降；
至少 2/3 分割种子新域提高且旧域不下降；
两个域分别的新域和旧域均值均不下降。

B. 相对 PPO_MATCHED、BANDIT_MATCHED、GRPO_FS_SHUFFLE：
新域平均差分别严格为正，旧域平均差分别不为负。

C. 全部工程、数据隔离和冻结检查通过。

这些是研发筛选阈值，不是统计显著性或临床意义阈值。

不把“clip_fraction 必须非零”或“动作必须改变超过某比例”设为晋级条件。
策略改变不是性能目标。

如果不通过：
完成诊断、成本审计和负结果报告后停止。
不能自动扩大预算、更换奖励、重划患者或换主候选。
即使 GRPO_STD 更好，也只能报告为次要发现，
不能事后将它替换成已经预注册的主候选。

pilot 晋级明确使用了研发 val；
audit 仍只描述，不参与晋级。


八、P3：独立控制器复现与正式确认

只有 pilot 晋级，才启动此阶段。

新增控制器种子 402、403；
对五种必做学习器完整重复相同开发训练。
不能从种子 401 的最终策略继续微调。
共同初始化和校准尺度保持冻结。

三个控制器种子均保留；
不选最好一个，不集成三个策略后当作单策略结果。

然后准备五个预先冻结的新分割源种子，
每个按原生流程训练 REFUGE 8000 步并保存来源回执。

五个分割种子、两个域，每个单元运行：

三个不依赖控制器训练种子的基线：
ORIGINAL、FINE_05、OFFLINE_25。

以及三个控制器种子各自的：
PPO_MATCHED、BANDIT_MATCHED、GRPO_STD、
GRPO_FS、GRPO_FS_SHUFFLE、PHASE_SHUFFLE_FS。

共：
5 × 2 × (3 + 3×6) = 210 个确认终点。

基线只需实际训练一次，不能重复计为三个独立基线样本。
同一 seed/domain 的所有方法从相同完整入口开始。
确认阶段控制器更新数必须为 0。
确认阶段不能用 online/audit 奖励改变动作或参数。

统计方式：
先计算同分割种子、同域、同控制器种子的配对差；
主汇总对三个控制器种子等权平均，再对两域等权平均；
得到五个分割种子层面的配对差，报告均值和样本 SD。

同时报告三个控制器种子各自的结果，
全部 seed/domain/controller 单元，以及最差单元。
不能把 30 个交叉单元当成 30 个独立患者或独立分割种子。

通用联合条件：
旧域均值不下降；
至少 4/5 分割种子新域提高且旧域不下降；
两个域分别的新旧指标均不下降。

最终分开判断：

practical_candidate：
GRPO_FS 相对 ORIGINAL 新域至少 +0.2 个百分点；
相对 FINE_05 新域至少 +0.3 个百分点；
相对 OFFLINE_25 新域严格提高；
上述比较均满足通用联合条件。

grpo_advantage_supported：
相对 PPO_MATCHED 新域至少 +0.2 个百分点，
并满足通用联合条件。

multistep_control_supported：
相对 BANDIT_MATCHED、GRPO_FS_SHUFFLE，
新域分别至少 +0.2 个百分点；
相对 PHASE_SHUFFLE_FS 新域严格提高；
上述比较均满足通用联合条件。

此外，至少 2/3 个控制器种子单独汇总后，
相对 ORIGINAL 达到新域 +0.2 个百分点且旧域不下降，
才允许称具有控制器训练复现支持。

不满足哪项就报告哪项不满足。
实际性能提升不自动等于 GRPO 优势或多步信用分配优势。

所有 Dice 增益阈值均为“百分点”：
+0.2 个百分点在 0–1 Dice 尺度上是 +0.002。
反馈奖励 Q 与这些阈值不是同一单位。


九、DAPO 组件：仅一个条件性补充实验

首轮不移植 Dynamic Sampling、token-length weighting、
overlong reward shaping，也不搭建大型 LLM RL 框架。

预先实现 clip_low / clip_high 参数化和 active-clipping 诊断。

只有当必做 GRPO_FS 开发训练中：
至少一个域、至少两个 group 出现
“正优势样本中实际受到上界裁剪的比例 ≥5%”，
才允许运行一个补充方法：

GRPO_FS_CLIPHI：
clip_low=0.2，clip_high=0.28；
其余与 GRPO_FS 完全相同；
从共同初始化重新训练；
控制器种子仅 401；
两域均使用 6 group × 4 条完整轨迹；
在三个 pilot 分割种子上评估。

不能从 GRPO_FS 最终 checkpoint 接着训练来冒充同预算比较。

这个补充方法不参与主候选替换，不自动扩展到正式确认。
若触发条件不满足，记录 NOT_TRIGGERED，不为凑算法名称强行运行。

它只能称“DAPO-inspired Clip-Higher 消融”，不能称完整 DAPO。
不得按奖励大小或组方差筛掉不利组，也不得无限补采样。


十、必须保存的诊断与成本

固定状态面板上：
初始化前后及每个 group 后的 actor 概率；
与初始化/behavior 的 categorical KL；
argmax 一致率、top-2 margin、每动作概率及熵；
状态越界与裁剪比例。

每次控制器更新：
actor loss、critic loss、entropy 分开记录；
actor/critic 原始梯度范数和实际裁剪系数；
概率比范围；
原始越界比例；
真正进入 clipped 分支的正/负优势比例；
PPO/Bandit critic explained variance 和回报预测误差。
回报方差为零时 EV 标记未定义，不伪造为零或一。

每个 group：
四条动作序列及唯一序列数；
未缩放终局奖励、相对 U=0 收益、组均值和 std；
优势尺度；
online/audit 排名一致性与迁移诊断。
audit 信息不得回流训练或选择。

公平性报告：
新增学生更新；
复用历史源训练；
actor/critic optimizer 调用；
状态提取、VJP、虚拟 Adam；
奖励标签访问与 evaluator 访问；
U=0 参考与分叉计算；
实际 GPU 活跃成本、墙钟和峰值显存。
不能把“部署步骤相同”写成“总计算成本相同”。

按当前固定矩阵，预期新增学生更新为：

P1 校准：
3×5×(3200+2100) = 79,500。

开发共享 U=0 参考：
6×(3200+2100) = 31,800。

单控制器种子的五学习器开发：
5×6×4×(3200+2100) = 636,000。

九组 pilot：
3×9×(3200+2100) = 143,100。

完成 pilot 的合计：
890,400，另加 P0 实际资格更新。

晋级后新增两个控制器种子：
1,272,000。

五个确认源模型：
40,000。

210 个确认终点：
556,500。

主流程全部晋级完成：
2,758,900，另加 P0 实际资格更新。

条件性 Clip-Higher：
额外最多 143,100。

这些数字不包含特征提取、评估、控制器更新或历史复用训练。
这是有明确上限的分阶段研究，不应宣传为低成本小试。

启动前由代码自动展开矩阵并复算预算，
核对参考共享的前提与缓存身份。
若无法实现计划中的有效共享，先在真实训练前报告预算差异，
不能在运行中静默突破额度。

不设墙钟截止时间、不按剩余时间缩短训练。
阶段一旦启动，负性能不停止该阶段剩余必做对照。
工程错误、数据泄漏、数值故障或用户停止除外；
这些情况保留现场，不隐式重跑或重置计数。


十一、资源、执行和交付

仅使用已授权服务器 jiangsuiyang 的物理 GPU 5、6、7。
实际连接和调度前检查权限、GPU 映射、显存与其他作业。
不停止、抢占或修改他人的进程。

通过项目已有 NAS 存储入口执行。
读取真实回执解析存储位置，不猜测私有路径。
源码快照、数据缓存、checkpoint、日志与临时产物均按项目 NAS 规则存放；
挂载或读写探针失败即停止，不静默回退 home。
输出存储预算与检查点保留策略。

不新建定时监测自动化。
若通过已有授权作业系统提交，明确区分“已提交”“运行中”“已完成”。
未能实际连接、提交或执行时，报告具体阻塞；
不得把写好代码或生成配置称为实验已经启动。

至少交付：
EXPERIMENT_PLAN.md
BASELINE_AUDIT.md
IMPLEMENTATION_DIFF.md
QUALIFICATION_REPORT.md
FROZEN_CONFIG.json
SEED_AND_DATA_ROLE_LOCK.json（公开版移除私有角色信息）
BUDGET_PLAN.json
运行状态与不可变配置/源码回执

完成相应阶段后交付：
PILOT_RESULTS.csv
POLICY_DIAGNOSTICS.json
REWARD_DIAGNOSTICS.json
PROMOTION_DECISION.json
CONFIRMATION_RESULTS.csv（仅实际进入确认时）
SUMMARY.csv
DECISION.json
ALL_COSTS.json
COMPLETION_AUDIT.json
FINAL_INTERPRETATION.md

公开内容限源码、冻结计划和允许公开的匿名汇总。
患者身份、病例级分数、原始影像、标签、特征、
分叉状态、checkpoint、凭据和私有日志不得进入 Git。

按项目规则提交并推送新分支，
核对远端 SHA 和公开文件可访问性；
不 force-push，不覆盖历史负结果，不夹带其他工作区改动。

最终报告必须回答：
1. 策略是否真的改变了概率和有价值的决策？
2. 改变是否对应更好的 online 回报，能否迁移到 audit/val？
3. 是否真正超过 ORIGINAL，而不仅比强 U 少退化？
4. GRPO_FS 是否超过同预算 PPO？
5. 多步控制是否超过 Bandit、奖励打乱和时序打乱？
6. 收益能否跨控制器种子、分割种子和两个域复现？
7. 额外收益付出了多少真实计算和标签访问成本？

不能把代理奖励改善写成 Dice 改善，
不能把动作不同写成方法有效，
不能把优化种子确认写成独立患者泛化，
不能把本轮失败推广成所有 RL 无效。

现在先完成源码/NAS 核验、完整计划与预算展开，
再完成实现和工程资格；
只有资格通过后，按上述有限矩阵和晋级规则执行。