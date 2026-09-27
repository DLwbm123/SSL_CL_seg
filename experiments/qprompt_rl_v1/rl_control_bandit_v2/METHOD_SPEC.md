# RL_CONTROL_BANDIT_V2：学习动作控制器实验计划

状态：PROPOSAL_NOT_EXECUTED。本文不证明新执行器已实现、已审阅或已启动。
冻结代码参考：DLwbm123/SSL_CL_seg @ e0b2f673e3f8816e5223a258af75439bbd9bcc5b。

## 1. 目标与范围

下一轮检验真正的训练决策学习：给定当前学生、当前域标注数据和无标注图像，由可训练策略采样“跳过 U / 学粗粒度 / 学完整三类”，以临时学生更新后的真实标注质量变化作为奖励。

先进行 R3a 奖励测量，再进行 R3b 完整策略实验。它们是单域 contextual-bandit pilot，不宣称长期规划或持续遗忘改善。原始 KI / A 侧约束 / history-free 参数记忆保持为之后正式持续学习接入的底座，不以 native LR reference 或新 full-rank 参数化替代。

本轮不训练 GRQA、CC、Limg 或 readout 辅助损失；readout 概率可以用于反馈评分，但评分不反传到学生。旧 R1/R1.5/R1.6 阴性结论原样保留。不启动旧 R2 矩阵、旧 R3 launcher 或自动 KI 正式实验。

旧 R3 提案曾采用组归一化优势、反馈后另采实际动作。V2 明确修订为：独立 L-only 锚点优势；先密封实际动作，反馈只更新未来策略；三个预注册优化种子；九个对照臂；独立 1200-step 新阶段学习率进度。这些是新提案，不是“恢复已经验证的算法”。

## 2. 两项预注册问题

H1：在相同训练状态与 batch 下，三个动作产生的当前训练反馈收益是否存在超过数值噪声的差异，并随可观测状态变化？
H2：上下文策略能否把这些差异转化为最终学生分割增量，且收益不能完全由固定粗粒度监督、非 RL 收益预测、非上下文自适应或额外标注更新解释？

不把“某次 reward > 0”“loss 降了”“策略参数变化”“gate 放宽后通过”视为 H2 成立。

## 3. 数据、prefix 和新的训练阶段

骨干：UNET_QUERY_128、DINOV2_VITS14_QUERY。
域：RIM_ONE_r3、Drishti_GS。
优化种子：261、262、263；split seed 仍为 0。261 用于 R3a 的 reward-scale 校准，R3b 三种子均预先进入队列；262/263 的 R3b 结果是主要优化种子复验结果，不是独立患者确认。

复用 12 个已归档的纯监督 global_step=2000 QUERY_PREFIX（每 seed/backbone/domain 一个）。执行时从真实来源索引解析路径、验证每个文件 SHA、原训练 commit、schema、optimizer 和 global_step。不得猜私有路径；不得以 B1/B2/B3 或 Q0/Q3 final endpoint 替代。seed261 来源可追溯到原 R1，262/263 到 R1.6 新 prefix。未通过绑定的组合不训练，其它合法组合可继续。

canonical 元数据：
- manifest SHA256: 0622f54f42f05d6ef87f9dc89ee9435cf8da03c6c30cd970db6ea167e00dd8a3
- split SHA256: f250d97aea1f36f21899f5dd40bb6c9a819e7755aee458c8ee27506496b46a88
- RIM：L16、U63、val40；Drishti：L10、U41、val25。数量执行时按冻结清单核验。

本轮单独建立 R3_U_IMAGE_ONLY 迁移/访问授权，只补 104 个 U image payload；不自动触发包含 REFUGE 的旧 M2 清单。U accessor 不保存/打开 label 路径或 label key。不搬 test、REFUGE payload、MRI 或旧实验整包。val 只由独立 endpoint evaluator 读取。训练、控制器和故障诊断不以 val 分数调参。

### 3.1 训练内 fit/feedback 分离

将每域现有 L 按冻结、可核验的组 ID 划成 5 个近似等量 fold。patient ID 可靠时按 patient；否则按 image，明确不声称患者独立性。至少 5 个可用组；不满足的域报告具体不可构造原因，不伪造组。

decision d=0..59：
- F_online = fold[d mod 5]；
- F_audit = fold[(d+1) mod 5]；
- L_fit = 其余 3 fold。

当前 20 个 ordinary updates 的所有拟合 batch 均从 L_fit 取；每步 L batch=2。R3a/R3b 的所有臂在同 seed/domain 下共用 fold 轮转和 L-fit schedule。Drishti 无组聚合时约为 6 fit / 2 online / 2 audit；RIM 根据组划分实际报告。

F_online 是训练奖励数据；F_audit 是仅用于机制审计的训练数据，其结果不回送控制器，也不触发人工/自动调参。二者不是正式 val。旧 prefix 已训练过全部 L，因此只声称本次候选更新内不重叠，绝不声称 feedback 从未被学生见过。轮转后某图像可以进入未来 L_fit。

为 R3 新阶段生成新的 1200-step fold-constrained schedule，包含 L/U IDs、geometry/photometric seeds、policy draw streams 和额外 L 更新计划。它不等于旧 R1 suffix schedule；九臂必须使用同一预注册新 schedule。

## 4. 学生目标与三个动作

学生 supervised objective 保持现有 supervised_query_loss：L_set，不添加 NLL/Dice/readout/GRQA/Limg。

每个 ordinary step 取一张当前域 U 图像（U batch=1）。首轮决策单位是该图像的整个有效区域，不再增加区域排序或 patch 采样规则。L 使用原冻结的 geometry + strong photometric augmentation；U weak 和 strong 共享 geometry，仅 strong 加同一冻结 photometric 变换。teacher 为当前学生初始化的 EMA，momentum=.99；它不是旧域教师。

记 q 为 teacher weak-view semantic 概率，p 为 student strong-view semantic 概率，V 为无 GT 的有效几何 mask。
- a0 SKIP：U loss=0。
- a1 COARSE：q_c=(q_bg,q_rim+q_cup)，p_c 同理；M1=V & [max(q_c)>.7]；U loss=Σ_v M1(v) KL(q_c(v)||p_c(v)) / max(Σ_v V(v),1)。
- a2 FINE：M2=V & [max(q_bg,q_rim,q_cup)>.7]；U loss=Σ_v M2(v) KL(q(v)||p(v)) / max(Σ_v V(v),1)。

L_student=L_set+.5 L_U(a)。概率计算和 KL 使用 FP32、安全对数及 stop-gradient teacher。不得把 coarse supervision 实现成单独监督 rim/cup 或强造 cup 伪标签。无 admitted pixel 时 U loss 为 graph-connected zero。

coarse/fine admission mask 可以不同，这是动作定义的一部分。记录覆盖率和实际梯度，不声称两个动作梯度尺度匹配。固定 COARSE 对照专门排除“只是一直减弱监督”解释。

已有 2000-step L-only prefix 作为预热；本阶段不再增加 U warm-up 或 ramp。所有比较采用相同 .5 系数和 .7 阈值，不看结果搜索。

## 5. 状态与策略

上下文策略输入 16 个 detached scalars，均在实际动作和 probe reward 产生之前计算：
1 local progress t/1200；2 current_lr/entry_lr；3–5 teacher argmax bg/rim/cup fractions；6–8 teacher mean bg/rim/cup probabilities；9 normalized fine entropy；10 normalized coarse entropy；11 q_disc-weighted conditional rim/cup entropy；12 fine coverage；13 coarse coverage；14 normalized teacher–student weak-view JS；15 transformed current L-fit L_set, L/(1+L)；16 teacher–student argmax disagreement fraction。

conditional entropy 的 q_disc 总量≤1e-8时定义为1并记录缺支持标志；所有 mask 与 mean 只用 valid geometry。状态不含域 ID、seed、case ID、U GT、旧域数据、validation metric 或本次候选 reward。弱视图状态前向使用 eval/no_grad 并恢复原 mode/RNG，不更改学生训练状态。

策略 MLP 16→32→3，tanh，输出层初始化为零。实际动作分布：πφ=.9 softmax(logits)+.1/3。所有概率比值和 KL 都使用这个实际混合分布，不能用 softmax 分布冒充分母。

策略 optimizer=Adam，lr=3e-4，betas=(.9,.999)，eps=1e-8，wd=0；每个有效反馈 decision 最多4次更新。输入尺度如上固定，不在线用未来/val数据拟合 normalizer。

## 6. 真实奖励

对反馈图像定义 Q(θ,F)=-mean_image[CB-NLL + foreground soft-Dice loss]。
CB-NLL：对图像内有 GT 支持的 semantic 类等权，再对各类 GT 区域像素取 -log(p_c) 均值；log floor=1e-8。
soft-Dice：rim/cup 两类等权，smooth=1，ignore255排除；缺前景类保留正常 false-positive 惩罚。全 ignore 图像不进入平均；若整个 F 无有效标注，标记无有效反馈，不伪造 reward。

Q 用 clean F、学生最终 semantic 概率、FP32 eval/inference_mode 计算。它仅产生 detached 标量，不通过 feedback loss 或临时 optimizer 反向传播学生或策略。这里复用语义质量指标不等于恢复 R1.6 的辅助训练 loss。

候选实际执行 h=1 AdamW 更新：θ'_a=Update(S_t,L_fit,U,a)，包含同一个 student/optimizer/scheduler 起点及相同 batch/augmentation/teacher targets。
锚点：θ'_0=Update(S_t,L_fit,U,a0)。
原始任务 reward：r_t(a)=Q(θ'_a,F_online)-Q(θ'_0,F_online)。

采用 anchor-relative advantage A_g=stopgrad(r_g/s_bd)，不做组内减均值或逐组 std 归一化。独立 L-only 锚点与所采动作无关；负值明确代表不如本次 L-only 更新。所有 candidate 都负时仍提供负 policy-gradient credit，不丢掉这个训练信号；不将“最不坏”自动称为正奖励。

s_bd 在 R3a 按每 backbone/domain 的全部预定 COARSE/FINE raw rewards 计算 RMS，并取 max(1e-4,RMS)，随后冻结供三种子使用。它是预注册的训练内量纲校准，不选动作、学习率或阈值。保留所有 raw reward 和数值 null-noise 结果；不放大接近零的奖励到单位优势，不静默 reward clipping。

F_audit 对完全相同的候选学生重复评分，只记日志，不能进入 A、策略更新或主动作选择。

## 7. R3a：固定320次临时学生更新

seed261，2骨干×2域×16预定状态/batch组合=64个场景。模型都由对应2000-step prefix恢复；变化来自预先冻结的 L/U/augmentation/fold。

每场景独立从同一 S 做 a0/a1/a2 三次候选更新，再重复 a0/a2 各一次作数值噪声对照：5 optimizer calls。
总320次真实数据临时学生更新；持久学生更新0；策略更新0。不是零训练成本。

每场景报告：a1/a2对a0的在线/审计收益、top-action一致性、类支持、null差异、两种 KL admission coverage、U-grad是否非零。聚合 action spread、不同反馈折排序稳定性、最佳动作分布及可学性不确定性。重复同动作不是独立样本，不拿它增加统计样本数。

R3a 不以正奖励比例或某个梯度门槛取消 R3b。s_bd 按上述公式冻结。错误状态恢复、非法数据访问或无法构造有效反馈属于工程/数据问题，修复或隔离受影响组合，不能绕过。

## 8. R3b：实际闭环，不是候选择优

每条轨迹 1200 ordinary student updates；t=0,20,...,1180为60个反馈 decision；策略每一步都对新 U 状态给动作，而不是将一个状态的动作固定20步。

在反馈 decision：
1. 取得真实 S_t、z_t、same L_fit/U/teacher target；冻结行为策略π_beh。
2. 先从π_beh独立采样并密封本步 actual action a_exec。规则/固定臂按其定义密封。
3. RL/NC_RL/REG 从各自当前行为分布另采G=4个动作，有放回。独立执行4次h=1 candidate与1次L-only anchor；所有clone完整恢复，候选不提交。
4. 用 F_online 算4个raw rewards，F_audit仅审计。
5. 用冻结z/actions/rewards更新控制器最多4次。此次更新仅影响未来动作，不改变已经密封的a_exec。
6. 丢弃全部候选，从真实未被修改的S_t重新执行a_exec，提交一次普通学生更新，然后更新学生EMA/scheduler/cursor。
7. 余下19步策略继续按新状态采样，不额外取reward、不在线优化控制器。

所有拟合路径保持同样的L_fit schedule。G=4中重复动作仍分别计费并保留重复性诊断；不为凑满3动作而拒绝采样；不缓存来冒充执行次数。不按reward选择candidate checkpoint，不把候选权重promote到真实学生，不用argmax reward替代策略学习。

## 9. Policy objective：anchored PPO-style contextual bandit

ρ_g=πφ(a_g|z)/π_beh(a_g|z)。
L_policy=-mean_g min(ρ_g A_g,clip(ρ_g,.9,1.1)A_g)+.001 KL(πφ(.|z)||π_ref(.|z))。

exact categorical KL over3 actions；π_ref是单独冻结的EMA policy，四次policy更新期间固定，decision完成后EMA=.99更新一次。π_beh是采样时快照，不能被EMA替换。student EMA teacher、policy behavior snapshot、policy KL reference为三个不同对象。

只通过动作log-prob/ratio/KL更新φ；feedback reward、z、candidate students均detached。这是PPO形式的bandit提案，不是QPrompt/GRQA公式复现，也不是学习长期return。G=4样本可用时照常处理负收益；全部A精确零时不伪造梯度，允许少于预算的optimizer调用，不补做。

## 10. 九臂对照

SUP：每步a0，无U读取、不probe。
FIX_COARSE：每步a1。
FIX_FINE：每步a2；唯一预注册主要固定比较对象。
RANDOM：每步均匀随机a0/a1/a2，无feedback学习。
RULE：normalized mean coarse entropy>0.5选a0；否则q_disc-weighted conditional foreground entropy>0.5选a1；否则a2。固定规则，不从val拟合阈值。
REG：16→32→3 MLP，拟合所采动作的normalized reward；loss=mean squared error over sampled actions，4 optimizer calls/decision，Adam与RL相同；π=.9 softmax(predicted_values)+.1/3，初始化均匀，无policy gradient和KL；同样先密封actual action。其反馈次数/候选更新数匹配RL，但不同轨迹和动作导致反馈内容不完全相同。
NC_RL：仅3个可训练常数logits，不看z；其它采样、奖励、PPO、参考、预算同RL。检验是否只需学习一个非上下文/随训练变化的全局动作偏好。
RL：完整16维contextual policy，唯一主要方法候选。
FIX_FINE_EXTRA_L：普通步骤同FIX_FINE；每次decision额外5次L-only真实提交更新，每次batch2从F_online固定顺序取（不足batch2按注册有放回规则）；没有candidate probes。每轨迹额外300更新，总1500 suffix学生更新。这是student-optimizer-call-matched额外标注计算对照，不是严格GPU-time/FLOPs或反馈前向等价。

RL/REG/NC_RL各有1200 ordinary+300disposable=1500 student calls；EXTRA_L有1500 committed student calls。所有额外前向、标签访问、控制器开销、墙钟和GPU占用另外报告。

## 11. 三种子完整矩阵与预算

R3b：3seeds×2backbones×2domains×9arms=108个最终学生，108条新训练轨迹；已有12prefix不重训、不再计为新增成本。
- ordinary student commits：108×1200=129,600。
- adaptive probes：3adaptive arms×12cells×60×5=10,800。
- EXTRA_L commits：12×60×5=3,600。
- R3b学生physical calls计划：144,000；最终路径保留更新133,200，另10,800是临时probe。
- R3a临时calls：320。
- 合计计划真实数据学生calls：144,320。
- controller calls最多：3adaptive arms×12cells×60×4=8,640，其中RL和NC_RL为policy5,760，REG回归2,880。

单列上限：学生恢复/重放14,432、控制器恢复/重放864；合成学生资格64、合成controller资格128、真实L-only smoke16。所有fail/discard/replay按真实次数记账，不能作为新的优化长度、候选配置或额外种子使用。

从首次CUDA资格开始最多12小时，重启不重置deadline。完整队列完成可提前结束；deadline到达保存可恢复状态、报告未完成cell，不谎报完整。seed261即使负向也不取消262/263注册轨迹；R3a数值正常但信号弱时不按研究门槛提前退出已注册R3b。

## 12. 新阶段scheduler：避免旧3000-step终止点

每臂独立恢复2000-step prefix的模型与AdamW moments，保留原scheduler作为来源证据，但为R3重新定义suffix局部t=0..1199。

每参数组读取prefix恢复后的真实entry_lr，采用：lr_g(t)=entry_lr_g*(1-t/1200)^.9。

不能直接让旧horizon=3000的LambdaLR继续到global3200，导致末200步零学习率；也不能误将initial_lr恢复为原始大lr。测试t=0、999、1000、1199。

ordinary scheduler只在ordinary成功提交后前进；probe scheduler clone永不修改真实状态。EXTRA_L的5个额外更新保持该decision的lr，不额外推进ordinary clock。单独记录ordinary_global_step、suffix_optimizer_commits和cumulative_updates：一般臂3200累计学生更新，EXTRA_L3500，不把二者混报。

学生优化器超参数与对应prefix保持同一骨干原R1配置；BF16模型拟合、FP32 optimizer/master parameters与KL；feedback评分统一禁用autocast用FP32，并用重复null诊断检查噪声。teacher=.99；不加新的梯度投影、orthogonality loss或超参数搜索。

## 13. 工程资格与恢复

每次候选从完整S恢复：student、optimizer、local scheduler、teacher、all RNG/explicit generators、fold/cursor、action seeds、模式、精度、计数。policy/reference/optimizer独立。候选结束验证真实状态未改变。actual action密封记录在耗时probe之前，重启不能重采。

资格覆盖：a0与anchor数值等价；coarse合并正确；r符号；actual-action不受本次reward影响；负reward的策略梯度符号；实际混合π分母；policy-reference不alias行为策略；REG不调用policy-gradient；candidate discard；feedback no_grad；hidden-GT拒绝；零U mask；临时fold disjoint；label exposure统计；同seed九臂输入一致。

日志使用嵌套字段。预演20步decision、100步checkpoint、文件发布完成标记、并发读取半写文件、deadline/队列恢复等边界。同一错误在同一代码版本两次出现，暂停受影响路径、修复和回归，不在所有worker上反复重放。

每100 ordinary commits和完整decision边界保存atomic recovery状态；controller draw/probe/update/actual commit各有持久事务ID。崩溃后须整体恢复到完整边界，不能只恢复θ而保留已前进φ。保留physical计费；未完成事务不能伪装成功。

工程patch可自治；改变model/loss/reward/action/scheduler/data/precision的修复要版本化、标出影响范围并从合法起点重跑，不能看val后称为engineering-only。只使用合法空闲GPU，不杀其它用户进程；私有case/路径、医疗数据、checkpoints不公开。

## 14. 评价与分级结论

主结果是所有学生的预定新阶段终点；student-only inference，控制器、teacher、reward evaluator、candidate clones不进入部署。指标保持每图rim/cup macro，再等域/种子聚合，报告pp和raw Dice避免混淆。不选best epoch/EMA或改变val。

预注册primary：fresh seeds262/263的RL-FIX_FINE，共8cell。seed261独立报告，不并入主要复验。
研发目标：平均Δmacro≥.003，每骨干平均>0，8cell至少6个>0，任一macro≥-.005、单域单类≥-.01。它是投入标准，不是显著性标准。

另报告而不换primary：RL-SUP、RL-FIX_COARSE、RL-RANDOM、RL-RULE、RL-REG、RL-NC_RL、RL-FIX_FINE_EXTRA_L。
- RL胜FIX_FINE但不胜SUP/COARSE：可能只是回避有害U，不宣称SSL正增益。
- RL约等于REG：有监督反馈适应的价值，未证明RL特有优势。
- RL约等于NC_RL：未证明context有价值。
- RL胜固定但不胜EXTRA_L：不主张计算效率或反馈效率优势。
- reward上升而val无提升：只能说优化了训练反馈，不能说分割泛化已改善。
- 小差值/跨seed反向：如实标为证据不足，不改阈值凑成功。

报告raw reward、online/audit一致性、动作分布/概率/entropy、clip/KL、baseline退化组、teacher coverage、student和controller成本。重复action和共享图像不是独立样本；只有两个fresh优化seed，不把8cell当8个独立患者实验。不作无依据p值/置信度断言；可发布逐图私有配对数据和描述性区间，但不得据此声称独立外部确认。

## 15. 原始KI接入（R3c，不包含在当前训练预算）

本轮同时整理0训练KI绑定规范：原始KI的真实repo/commit、A侧约束、rank、可训练参数集合、阶段source/target checkpoints及hash。不能用标为“not original KI recovery”的native reference代替。

将来接入时只允许策略控制当前U supervision action，不允许RL改变KI投影/rank/source/teacher数据访问规则。probe与真实step都通过同一个原KI更新路径。old-domain val只用于独立终点评价，绝不成为reward。可以跨域保留已学习的θ和φ参数；当前域raw L/U、feedback/probe、trajectory buffer及reward-normalization统计不得跨域累积。phase-local训练状态按原KI契约重置，不重新定义其记忆机制。

单域pilot不是CL效益证据。R3c须另行固定真实KI协议与有限预算；当前不自动启动，不让prototype loss提分成为RL实验的前置门槛。

## 16. 交付

METHOD_SPEC、ACTION_CONTRACT、REWARD_CONTRACT、STATE_SCHEMA、FOLD_AND_DATA_BINDING（私有细节分开）、PREFIX_BINDINGS、R3A_REWARD_AUDIT、POLICY_LEARNING_CURVES、RESULTS、PAIRED_DELTAS、COST_AND_FEEDBACK_COUNTS、TRANSACTION_AUDIT、TASK_LEDGER、PATCH_LOG、FINAL_INTERPRETATION。

每个run/endpoint绑定真实training commit、配置hash、prefix SHA、新schedule SHA与controller版本；report commit与training commit分开。禁止将本文状态改成外审通过的回执。

## 来源与性质

[S1] SSL_CL_seg @ e0b2f673e3f8816e5223a258af75439bbd9bcc5b，experiments/qprompt_rl_v1/RL_CONTROL_DESIGN.md：原R3为接口提案。
[S2] 同提交 EXPERIMENT_PLAN.md §8：三动作、L-fit/feedback、probe、behavior/reference区分的原设计；V2的修订已在§1列明。
[S3] 同提交 r1_6_readout_v1/reports/FINAL_INTERPRETATION.md 与 SEED_AND_SCHEDULE_AUDIT.json：先前结果、种子与来源边界。
[S4] 同提交 EXPERIMENT_PLAN.md §2：冻结split和L/U访问规则。
[S5] Schulman et al., Proximal Policy Optimization Algorithms, arXiv:1707.06347：采样动作与clipped surrogate的算法参考；不证明本任务收益。
[S6] Ren et al., Learning to Reweight Examples for Robust Deep Learning, arXiv:1803.09050：标注反馈可以用于非RL适应，故需要非RL对照；REG不是该论文的原样复现。

除来源中明确说明的历史事实外，本文动作细化、奖励、采样顺序、网络、超参数、预算和标准均为本次新研究提案，不是已验证的效果。
