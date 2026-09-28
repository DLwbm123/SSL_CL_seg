# RL_CONTROL_FINE_INTERVENTION_V3A
## 有限决策价值审计与全信息策略可学习性实验

**状态：PROPOSED。本文是新实验计划，不是外审通过回执、训练完成回执或远端启动命令。**

冻结仓库：`DLwbm123/SSL_CL_seg`。冻结起点：`6eb22a65cb81f50f523f4f3a3ff2e926d051ba68`。
建议独立目录：`experiments/qprompt_rl_v1/rl_control_fine_intervention_v3a/`。
建议独立分支：`codex/rl-control-fine-intervention-v3a`；有冲突时新建后缀分支，不覆盖或 force-push。

## 1. 研究决策

继续 RL，不恢复 GRQA/CC/Limg/readout 辅助损失搜索，也不立即再训练 108 个终点。

本轮完整执行三个部分：

1. P0：重算现有 V2 训练反馈日志，不进行新的医学模型前向或优化。
2. P1：构造 64 个完整三动作短分支场景；在同一分支第 1 和第 5 步评分；960 次主探测更新，加 80 次重复分支噪声测量，总计 1,040 次临时真实数据学生更新。
3. P2：在 P1 的完整动作收益表上，做按 U 图像分组的交叉拟合；真正优化小型策略网络，但不更新或部署医学学生。

P0/P1/P2 不按中途科研分数取消剩余注册单元。完成就是本轮正常结束，不自动扩大成正式学生矩阵。实际数据安全、完整恢复和预算是不可绕过的执行条件。

**本轮新增持久医学学生更新为 0，新增医学学生终点为 0。1,040 次临时更新仍是实际训练成本。**

核心问题依次为：

- 相对于 fine，其他动作在测得的场景中是否有收益空间？
- online 反馈选出的动作，换一份 audit 反馈后是否仍有益？
- 当前 16 维可观测状态能否预测这种收益？
- 如果使用完整动作反馈，策略优化能否学到有用的条件化选择？

## 2. 现有证据与限定

V2 fresh seeds 的 RL−FIX_FINE 为 −0.001546709 Dice（−0.154671 pp），8/8 cell 为负。RL 的实际行为接近均匀混合；固定 fine 相对 SUP 有正的平均增量。

R3a 中 RIM 的 online/audit 最佳动作一致率较高，Drishti 较低。它们都是固定纯监督前缀上的当前训练数据诊断，不是独立患者验证，也不覆盖完整训练状态分布。

这些结果不支持继续原样放大 V2，也不证明所有 RL 或所有动作空间无效。本轮不承诺正的最终 Dice。

## 3. 冻结内容与禁止范围

继承 V2：两骨干、RIM/Drishti、seed261 的四个纯监督 2000-step prefix、规范化数据、三动作数学、L batch=2、U batch=1、图像级决策、16 维状态、原 Lset、lambda_U=0.5、confidence>0.7、teacher EMA=0.99、原 augmentation 和反馈 quality。

训练型前向继续 BF16，KL 和反馈评分继续 FP32；master/optimizer 状态按旧协议。不得悄悄把 FP32 反馈改成 BF16。

不修改 K、学生架构、segmentation loss、输入尺寸、optimizer 超参数、lambda_U、置信阈值和数据 split。不使用 validation/test/U GT 产生动作奖励。

不启动 R2/R3c；原始 KI 仍未绑定时不使用 native reference 冒充。原 KI 的 history-free 参数记忆及 A 侧约束不在本轮修改范围。

## 4. P0：现有日志再分析

### 4.1 来源

复用 V2 的有效 R3A.private.json、有效 R3b decisions/*.private.json、动作记录和来源审计。不得读 invalidated/P1 为有效样本，不把同一事务重放重复计数。

先核对 task、seed、backbone、domain、t、code commit、prefix/schedule digest、成功事务与作废标记。按数值 t 排序，不能按文件名字典顺序定义 early/late。

所有 3 种子和 3 个自适应方法均纳入描述性分析，分方法/种子报告，不能混成独立同分布样本。

### 4.2 从 skip 基准切换到 fine 对比的分析口径

V2 原始记录 r(a)=J(a)−J(skip)。若同一状态记录了 fine，则 d(a)=r(a)−r(fine)。

fine 未被采到时，对 fine 的差值必须保持 missing。不同状态的 fine、另一方法的 fine、另一 seed 的 fine 不能补齐。重复动作先核对同状态差异，按动作求记录均值用于配对表，另保留最大数值差和次数。

对完整三动作场景报告：

- V_oracle = mean(max_a d_audit(a))。这是有限场景、带测量噪声的事后乐观值，不是泛化或长期策略性能的理论上界。
- a_online 按 online 差值选择；V_transfer = mean(d_audit(a_online))。
- online 选择非 fine 的比例、其中 audit 为正/负/不确定的比例。
- fine/coarse、fine/skip 的成对排序一致性，条件于相应动作确实被记录。
- 非 skip 动作相对 skip 的符号一致率单列；不允许 skip 对自身的零奖励人为提高一致率。

部分动作面板只报告 observed-action contrast；其完整样本子集可能受行为策略影响，不外推为全部状态的无偏结果。不在这个小审计中加入新 IPS/DR 估计器。

P0 医学前向=0，学生 optimizer=0，控制器 optimizer=0。

## 5. P1：64 个完整动作面板

### 5.1 场景清单

4 cell = 2 backbone × 2 domain，固定 seed261。每 cell 16 场景 j=0..15。

严格继承 V2 R3a 选取：origin_t(j)=20*floor(60*j/16)。每个场景使用该起点及后 4 个已注册 schedule item。

16 场景的学生根状态均为该 cell 的同一 2000-step prefix；变化的是拟合图像、U、增强和反馈折。不得将 schedule origin_t 当成该学生已训练到的时间。真实 local scheduler 从 0 开始，h=1..5 使用 local t=0..4。

旧 prefix 来源、SHA、optimizer moments/step、无 bank/reference 状态须核验。缺失时隔离受影响 cell；不能用旧最终高分模型替换，不隐式重训 prefix。

在任何新 reward 计算前生成并封存 CONTEXT_MANIFEST.private.json，绑定全部 L/U indices、fold、seeds、5-step schedule 与 prefix digest。

### 5.2 三条分支的精确定义

| 分支标记 a | 第1步 | 第2–5步 |
|---|---|---|
| SKIP | 只 L | FINE |
| COARSE | L + coarse KL | FINE |
| FINE | L + fine KL | FINE |

**第一步是唯一干预，后四步都 follow FIX_FINE。** 不把 coarse/skip 重复五步；否则 horizon 改变同时改变监督剂量，无法解释一次动作的延迟效果。

同一场景三分支使用完全相同的 5-step L/U/augmentation/fold 序列，各分支从根状态独立恢复。L_fit 在五步内不改变，online 与 audit 在五步内也固定；起点是20-step块边界，因此可使用旧 schedule 连续5步，但仍逐项断言折不变。

### 5.3 分支内部更新必须像真实训练

分支每次成功学生更新后，在**分支内部**推进 optimizer、local scheduler、teacher EMA、cursor、step；teacher 下一步预测来自该分支自己的 teacher。

不能沿用 V2 `temporary=True` 导致不更新 teacher 的单步实现直接跑五步。不得把 fine 分支下一步 teacher 目标借给另两条分支。

候选五步可改变自己的状态，不能改变根状态或真实工作状态。结束即丢弃。teacher/optimizer 必须 deep copy，尤其 CPU Adam step tensor；保存和恢复不能共享可变 storage。

### 5.4 配对反馈与评分时点

在 h=1、h=5 对同一候选学生评分。online、audit 使用固定 clean 当前 L，FP32 eval/no_grad；评分不修改 RNG、mode、buffers、optimizer 或 teacher。

沿用 V2 quality：负的逐图、存在类别等权 NLL 加 rim/cup mean soft Dice；log floor 1e-8，Dice smooth=1，缺前景仍保留 Dice，all-ignore 图像排除；无有效反馈不填0。

定义：

    d_h^online(a)=J(theta_h^a,F_online)−J(theta_h^fine,F_online)
    d_h^audit(a) =J(theta_h^a,F_audit) −J(theta_h^fine,F_audit)

fine 对比自身为定义上的0，不当作数值重复实验。

h=1 是第一步干预后；h=5 是第一步干预+后四步共享默认策略后的结果。它仍是有限短视反事实，不是长期回报保证。

**h=5 是本轮预注册主要 horizon；h=1 是配对解释项。** 不按谁好看换主 horizon，不扫描 h=2/10/20。

### 5.5 成本及 null

主面板：4×16×3×5=960 次学生 optimizer 调用。

每 cell 的 j=0、8，额外重跑 SKIP 和 FINE 分支各一次：4×2×2×5=80 次调用。

总计1,040次有效临时学生调用，全部计费。h=1/h=5共用同一条分支，不重复训练第1步。

对每 cell/h，e=max(重复分支 quality 的绝对差)，同时取 online/audit、skip/fine 的最大值。

    delta_numeric=max(1e-6,5*e)

它只是预设数值死区，不是置信区间、安全保证或统计显著性阈值。raw 与死区版本同时报告。P2 训练不裁剪、置零或删除原始 reward。

决策 a_online：只有 online 最佳非 fine 动作超过 fine 大于 delta_numeric 才偏离；否则选 fine。数值精确并列时优先 fine，其次 coarse，再 skip。报告原始 top1、死区 top1 和不确定比例。

## 6. P1 必交付指标

按 cell/h 报告：

1. d(coarse)、d(skip) 的均值、分位数与全部方向计数。
2. V_oracle（事后、噪声乐观）与 V_transfer（online选、audit评）。
3. 每次偏离的 audit 收益、错误干预率，分 skip/coarse。
4. 同一 (scene,action) 的 h1/h5 符号变化、排序变化；不得把 fine 的恒定0放入相关系数。
5. 重复分支噪声、微小margin比例、缺反馈/无支持情况。
6. teacher coverage、U KL、实际U梯度；coverage不等于准确率。
7. 五步分支中 teacher 与学生状态差、scheduler、Adam step、恢复后root fingerprint。

同一图像的多次增强、同一前缀和共享反馈图像不是独立患者样本。不据64场景做独立样本p值结论。

## 7. P2：完整反馈上的策略可学习性

目的：先排除抽样漏动作/重复动作的测量低效，检查状态能否承载决策。不是把候选最优权重直接部署，也不是另一套学生辅助损失。

### 7.1 外层分组交叉拟合

每 cell 分开训练，不能从另一域借reward或样本。用 U 图像的原始 identity/digest 将16场景分组，同一U的所有场景放同一折。

稳定hash排序不同U组，round-robin分4折；完全在读取reward之前固定。少于4个可验证U组时，该cell交叉拟合不具资格，报告具体缺失，不退回逐行随机分折。

对 h=1、5 用相同折。每折模型只用其他折的 (z,d_online)，只在留出折用 z 产生策略概率，随后由独立分析器用 d_audit 评分。

不读取 held-out online reward 来选动作/调参，不读取任何 audit reward 来选模型、epoch、scale、网络或学习率。不能根据训练reward挑checkpoint；固定终点。

这是U-context留出，并非患者或L-feedback身份独立验证。L反馈图像早已用于旧prefix，且会在不同角色轮换。只有一个学生prefix状态/cell，所得结论不覆盖后期状态漂移。

### 7.2 固定默认分布与网络

    pi0=[0.05,0.05,0.90]  # skip, coarse, fine
    pi_phi(z)=0.9*softmax(log([1,1,52])+f_phi(z))+0.1/3

f_phi 为16→32→3、Tanh，与V2同宽度；输出层初始化为0，故精确初始化到pi0（浮点容差单测）。保留V2状态，不加域ID、caseID、reward、val等输入。

fine默认是一个科研初始化选择，不是经过证明的safe bandit。探索概率下界和KL不保证最终Dice不下降。

### 7.3 奖励尺度

按cell读取V2有效R3a已冻结scale，绑定SHA；h1/h5共用同一scale。

    normalized_d = d_online / old_scale

不再按新全数据或audit估计scale，不因h5较大单独调归一化，不做reward clipping/group standardization。尺度敏感性作为限制记录，不扫描温度。

### 7.4 FI_POLICY：全信息期望收益优化

因为三动作的同状态收益全部测得，直接优化：

    L(phi)= -mean_z sum_a pi_phi(a|z)*stopgrad(normalized_d_z(a))
            +0.001*mean_z KL(pi_phi(.|z)||pi0)

只通过策略概率反传；不穿过学生更新、teacher或反馈评分。没有把argmax动作当标签做模仿。

这叫**全信息条件下的上下文策略优化/可学习性诊断**，不称标准采样PPO。枚举动作后不应直接等权平均未校正的 sampled-logprob 目标；上式与V2 importance-ratio目标不同。

每个fold固定 full-batch Adam，lr=0.003，betas=(0.9,0.999)，eps=1e-8，weight_decay=0，256次更新；无early stop、无lr sweep。学习率与训练方式是新计划固定选择，不伪称沿用V2。

### 7.5 对照

- FINE：pi=(0,0,1)，价值相对fine恒为0。
- STATIC_PRIOR：始终pi0，不学习。
- NC_OPT：用训练折online平均normalized reward，精确解带同样KL和概率下界1/30的常数策略。避免再次用欠训练的三个logits当“不需要context”的证据。
- LIN_VALUE：训练折的16维状态拟合动作差值，线性ridge；目标fine固定0。输入标准化仅用训练折，std下限1e-6；目标不中心化掉fine基准。最小化 mean squared error +0.01*||W||²，截距不罚。对每个held-out状态，以预测差值解与NC_OPT相同的三动作KL/下界优化。它是非RL预测控制，不宣称与神经策略算力相同。
- FI_POLICY：预注册主要学习策略。
- FI_POLICY_SHUFFLE：与FI_POLICY相同初始化、batch、更新数；只在训练折内按预定置换打乱整个3动作reward向量与状态的对应，保持reward向量内部关系/边际分布。固定一套置换，不挑最有利置换。它是单个负对照，不是完整置换检验。

NC_OPT 的精确解可枚举被固定在1/30下界的动作子集，对其余动作按 pi0*exp(d/beta) 归一化，取可行最优；采用稳定logsumexp。小规模独立数值解验证可行性与最优值。此数学解不是GPU训练。

### 7.6 P2预算

2 horizons ×4cells ×4folds ×2神经方法=64个小网络fit。

64×256=16,384次控制器optimizer调用上限，均CPU，独立计费。

另有NC_OPT 32个分析解；LIN_VALUE 32个多输出ridge fit及逐状态3动作解析决策。解析解数量单列，不能说全部方法“零训练”。

少于上限仅可因已记录的缺失/工程条件，不能把预算转成更多epochs或随机种子。

### 7.7 评分

在每个held-out场景计算概率加权价值，而不是仅汇总argmax accuracy：

    V(pi)=mean_s sum_a pi(a|z_s)*d_audit_s(a)

分别报告对FINE、STATIC_PRIOR、NC_OPT、LIN_VALUE、SHUFFLE的差值。

先合并每cell所有OOF场景，避免不等大小fold被错误等权；再四cell等权、两骨干等权。报告干预概率、平均每次干预的audit价值、负干预概率质量和可识别上下文数。

预注册主要结果为h5、FI_POLICY相对FINE的OOF audit价值；其余是解释对比，不能事后改主候选。

这不是最终macro Dice终点，更不能把quality差值乘100写成Dice百分点。

## 8. 合成资格

在医学反馈读取前，测试已知答案的：永远fine最好、状态切换决定最优动作、全零/无信息reward。

使用与P2完全相同的策略、损失、初始化和更新预算；固定4个合成种子，不挑赢家。至多4,096次合成控制器optimizer调用。报告初始/末期概率和真实合成期望收益，不用“loss下降”替代行为改变。

测试失败时修数学/实现错误可版本化；不能根据真实audit结果调网络或学习率。纯学习能力不足必须如实标注，P0/P1等独立任务继续完成，不无限扩大合成搜索。

模型/恢复合成资格最多64次学生optimizer调用；真实L smoke额外额度为0。P1正式探测、null和恢复分别计账，不能冒用旧R0/R1额度。

## 9. 总预算

- 真实数据学生主探测：960。
- 真实数据学生null探测：80。
- 有效临时学生总数：1,040；新增持久学生更新：0。
- 新医学学生终点：0。
- 实测reward小网络optimizer：16,384上限。
- 合成学生optimizer：64上限。
- 合成控制器optimizer：4,096上限。
- 学生失败/恢复/重放：104上限。
- 控制器失败/恢复/重放：1,640上限。
- 额外真实L smoke：0。

最大物理学生调用=1,208；最大物理控制器调用=22,120。不得挪用未使用预算。

首次资格执行后至多12h，重启不重置。完整计划结束可提前收尾，不用余额启动新训练。

## 10. 执行与恢复

实现后先冻结E0。每个artifact记录code/config/data/prefix来源。旧结果只读。

分支开始/结束检查root fingerprint及tensor storage隔离；student/optimizer/teacher/scheduler/RNG/cursor/controller状态全部覆盖。真实worker不保留候选权重。

checkpoint在每个完整分支事务后保存必要回执；不无限保存候选模型。失败重跑整个5-step分支，原物理调用保留计费。P2按fold-fit边界恢复，不泄漏audit来决定重跑。

同一异常同一代码版本发生两次，暂停受影响路径，最小回归+commit后继续；其余独立路径可运行。不 kill 未知用户进程、不改全局环境。

工程自治不授权改变学生/动作/reward/horizon/学习器定义。科学路径修补需新commit并使受影响panel和fit作废，不能拼接旧reward表与新实现。

公开仅汇总和匿名统计。私有图像、逐case状态、反馈样本和checkpoint不push。用户要求执行时须单独确认本轮运行及push权限，不能继承旧阶段授权。

## 11. 完成后的决策表（不是中途停止gate）

所有计划单元结束后，按h5主要结果解释：

- oracle乐观值接近数值死区：当前动作/入口状态没有清楚可利用余地，不能承诺加大学习率能解决。
- oracle有余地，transfer不正：online反馈无法稳定选出对另一反馈集合有用的动作，优先改奖励测量设计，而非放大policy训练。
- transfer正，LIN_VALUE/NC_OPT能学而FI_POLICY不能：学习器/优化需要处理，不归咎于缺信息。
- FI_POLICY优于prior，但仍负于fine：只是少损失，不算新增益。
- FI_POLICY正于fine，但不优于NC_OPT/SHUFFLE：未建立上下文价值。
- h5 FI_POLICY在至少3/4cell高于该cell数值死区、两骨干平均均正，且总体同时优于STATIC_PRIOR/NC_OPT/SHUFFLE：具备提出小规模闭环pilot的依据；不是独立确认或CL收益。

LIN_VALUE结果始终报告；若其更好，不宣称RL专属优势。不把训练reward学习性当成最终Dice成功。

门槛不足不触发自动调参或另一个候选。所有失败、null和缺失保持原样。

## 12. 下一段V3B仅作后续提案

只有完成本轮解释后，再单独固定一个小型真实学生pilot，而不是本轮偷偷启动。

建议至少FINE、STATIC_PRIOR、NC_POLICY、VALUE_REG、CONTEXT_POLICY五臂；同一fine偏置初始化、同一反馈预算。执行动作必须在读取该次反馈之前密封，新反馈只影响后续决策。不得把oracle候选直接部署。

如果采用h5 action-value，必须明确它是一次干预后的fine rollout，而非已执行策略的无偏长期Q值。验证时仍报告实际学生终点和成本。

这一未来pilot的seed、学生更新数和自动扩展权限不在本轮预算中。原KI绑定未完成时不进入R3c，不替换history-free底座。

## 13. 交付

P0_REANALYSIS.md / PAIR_COVERAGE.csv / FINE_RELATIVE_LOG_SUMMARY.csv
CONTEXT_MANIFEST.private.json / COUNTERFACTUAL_VALUES.private.csv
HORIZON_TRANSFER.csv / NULL_AND_RESTORE_AUDIT.json
OOF_SPLITS.private.json / OOF_POLICY_VALUES.csv / POLICY_CONTROL_SUMMARY.csv
SYNTHETIC_CONTROLLABILITY.json / FINAL_INTERPRETATION.md
PHYSICAL_LEDGER.jsonl / PATCH_LOG.jsonl / PROVENANCE.json / RESOURCE_REPORT.json

FINAL_INTERPRETATION必须分别回答空间、反馈迁移、状态可预测性、策略优化、未来闭环证据范围。不能只写一个PASS/FAIL。

## 14. 来源

- V2报告：`experiments/qprompt_rl_v1/rl_control_bandit_v2/reports/FINAL_INTERPRETATION.md` @冻结提交。
- V2 data.py、method.py、runner.py、METHOD_SPEC.md、REWARD_CONTRACT.md @冻结提交。
- V2 R3A_REWARD_AUDIT.json、ACTION_DISTRIBUTIONS.csv @冻结提交。
- Schulman et al., Proximal Policy Optimization Algorithms, arXiv:1707.06347。多轮代理目标更新不等于新增环境反馈；本轮FI_POLICY不冒称PPO复现。
- Kazerouni et al., Conservative Contextual Linear Bandits, arXiv:1611.06426。这里只借鉴基线相对决策的问题意识；不继承其线性bandit安全保证。

以上来源用于定义和动机；本轮h5、fine prior、交叉拟合、学习器超参数与预算都是新研究提案。
