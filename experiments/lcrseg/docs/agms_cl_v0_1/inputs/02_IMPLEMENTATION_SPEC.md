# AGMS_CL_V0_1：实现规格

本文件定义新研究的核心V0，不是已发表算法的原样复现，不是实验批准。科学矩阵以01和05为准；代码准备允许合成测试，生产执行需要新审阅和用户授权。

## A. 兼容性边界

1. 父方法仍为 `NATIVE_LR_SRC_A_3DOMAIN_V1`，不是自动恢复的原始KI。保留14层、r=8、阶段入口V、A-only右投影，所有允许的原生A/B继续接受原B2监督与KL。
2. 历史B2/NKA/F5/DOSE的Python、结果、计划、账本和批准全部保持原字节。新研究只新增 `experiments/lcrseg/agms_cl_v0_1/` 和对应docs；必要公开输入只复制到新docs/inputs并固定内容hash和原提交来源。
3. 不添加NKA/SWD、累计F、额外历史teacher、历史特征库、跨阶段风险统计、逐域decoder或路由表。
4. 当前训练新增的辅助头、其EMA与优化状态、风险R/计数全部在阶段退出时丢弃；只有主网络合并权重进入下一阶段。部署永远使用原最终分类头，不报告aux融合的推理结果为单学生结果。
5. 现有 `StageTrainer` 构造器只显式收集父A/B和可能的R参数，新头不会自动进入optimizer/EMA；必须有独立且受审的模型/更新适配。不可仅给model挂模块而漏掉optimizer组，也不可更改旧类权限或monkeypatch其类型以绕过检查。

## B. 输入、标签与输出坐标

标签0=background，1=rim，2=cup，255=ignore；父类disc={1,2}。
`G`为合法U geometry，不是U标签。U accessor应根本不加载标签文件。

原主读出为3×3 valid logits再插值回输出图；该路径原样保留，不能在新分支处理中悄悄改主头padding/crop。
辅助头在 `decoder.dec3` 和 `decoder.dec2` 最终输出上读取特征。384输入时预期为64×96×96和32×192×192；合成32输入也必须按实际尺度工作。
每个辅助头为带bias的1×1 Conv2d，输出3类别；logits以bilinear、align_corners=True独立映射到当前输出网格，再做log_softmax。不要再错误套用最终主头的valid-crop坐标变换。以脉冲、棋盘、边界和ignore洞构造单元测试，并记录明确的几何约定。

额外参数：3*(64+1)+3*(32+1)=294。

### 复用前向，不额外改变数据曝光

- 学生监督：原warmup为单个L前向；active时是原LCTX两个互补混合前向。捕获这些同一次前向的中间特征，辅助头分别输出，再用同一个原mask在输出网格收集anchor来源。不得拿混合图整幅结果与原未混合标签直接算aux损失。
- 原主logits、mask生成、donor噪声及随机数调用顺序不变。不要重新调用`complementary`得到另一个mask。
- teacher-L：复用原当前原型估计中的clean-L特征；增加必要的辅助readout，以及原主头的只读L概率（如原路径尚未计算）。
- teacher-U：复用原clean-U前向的多尺度特征；不增加teacher全网前向。
- 学生U父类监督：使用原UL互补来源收集后的主logp，不新增clean-U全网前向。
- 目标：核心方法新增全骨干前向为0，新增的是readout及辅助反向。实际实现不能做到时，在审阅前明确报告计算差异，不能继续声称零新增全网前向。

## C. 原B2细粒度分支完全保留

主teacher产生q0和原PAS集合`mF`。沿用原KL、lambda_U=1、PAS=.6/.7、原几何和原prototype语义。

细粒度目标和mask的规则在所有臂都不变。不同训练轨迹上q0/mF自然可以不同；“规则相同”不代表跨臂实际mask永远相同。禁止回放其他臂mask。

细分支先按原路径确定mF，再将其补集供父类分支使用；新分支不会删除原B2已接受的fine监督，也不会把同一个像素作为新的fine和coarse样本重复计算。

## D. H：父类集合监督

对每个像素，`d_s=q_s[1]+q_s[2]`。A1/A3的mask为：
`mH = G & ~mF & (d0 >= 0.9)`。

A4/A5额外使用多尺度证据（下一节）。本版只回收可信disc父类；不重新监督原PAS拒绝的背景，不另设cup伪边界loss。

H的稳定实现：
`parent_nll = -logsumexp(logp_U[:,1:3], dim=1)`。

每图`LH_b=sum(mH_b*parent_nll_b)/max(1,sum(G_b))`，然后对拥有合法geometry的图像等权平均。无合法图像或无mH返回与主student图连接的零，保留0支持记录。

这选择了明确的父类集合目标，而不是给rim/cup硬选一个标签。它不提供“rim/cup条件概率绝对不变”的数学保证，不能这样描述。

粗门槛不能先经过细类别PAS，否则(qbg,qrim,qcup)=(.04,.48,.48)这种目标会在新增分支前就被丢弃。必须有合成测试验证该例能进入A1的父类分支，而不会伪造成rim或cup真值。

## E. M：普通辅助深监督

A2/A3/A4/A5拥有相同两个辅助头。每个头使用与主分支相同的 `supervised_parts` CE+Dice定义；两头均值形成LDS，lambda_DS=.25。主CE+Dice始终系数1，不能被归一化权重缩小。

头初始LR=.001、weight_decay=4e-5，其余Adam/betas/eps和polynomial调度保持原定义。与A/B属于同一个optimizer，一次step更新所有受监督参数，不增第二optimizer。

新头只接L梯度，U只更新原B2允许的A/B。teacher auxiliary heads由student对应头复制并按原.99 EMA更新，teacher不接受梯度。检测辅助头是否真的被optimizer更新以及主A/B收到aux L梯度。

两头初始化用新命名空间和局部generator/fork_rng；不得消耗或改变原父初始化、数据、LCTX、UL随机流。A2-A5同seed/order/stage的辅助头初始值一致，arm不参与其初始化seed。

## F. 多尺度融合与风险状态

三尺度包括原主teacher(q0)和两个EMA辅助头(q1,q2)，均在同一输出网格。

A4：alpha=(1/3,1/3,1/3)。

A5：每个尺度一个当前阶段风险标量R_s，初值.25。在已有teacher-L前向上，用当前L的disc/bg标签计算Brier误差。每图分别求出现类别(bg/disc)内均方误差，再对出现类别平均，最后对有效图像平均。这样不让大量背景完全支配风险。无合法标签时不更新。

`R_pending=.9*R+.1*r_batch`。
当前步融合使用上一已提交的R，`alpha_s=.1+.7*softmax(-R/.1)_s`。
三权重和1、每项至少.1。R、alpha、teacher概率、所有mask均detach。

风险是训练内的选择状态，不是独立校准误差估计；没有共形覆盖或真实不确定性保证。不得使用val/test或当前U真值更新R；不得为了权重更好看在线修改temperature。

共同公式：
`dbar=sum_s alpha_s*d_s`，`v=sum_s alpha_s*(d_s-dbar)^2`。
A4/A5的mask：
`mH=G & ~mF & (d0>=.9) & (dbar>=.9) & (v<=.01)`。

A3计算/记录与A4/A5相同的辅助预测与risk诊断，但H仍用单主头mask；A4可同样更新R作观察而不使用它。由此A3-A5不新增可训练参数，A5-A4只改变固定融合权重为当前L风险权重。

P1不引入像素级学习门控、不做自由loss权重下降。细粒度通道始终是原B2，这是一版保守的父类信息回收机制；不得声称已完成所有医学粒度自适应问题。

## G. 总目标和更新流程

`L_total=L_main_sup+L_parent_constraint + enabled_M*.25*LDS + ramp*(L_KL_B2+enabled_H*.5*LH)`。
B2主监督包含原CE+Dice；不能只保留CE。enabled开关必须由canonical arm绑定，不是运行时自由flag。

每次：
1. 验证研究/节点/完整参数组/数据scope；读取原当前L。
2. 原前向产生主/aux监督；active时运行原teacher L/U与PAS、原学生UL。
3. 从上一R得到mask，生成DS/H及待提交R。
4. 固定诊断点额外求四项VJP，retain_graph，仅记录；原实际grad合并顺序不变。
5. 一个Adam step；原约束检查和scheduler；校验teacher/risk pending状态，再原子提交EMA、prototypes、R、游标与诊断。
6. 失败尝试记账；无成功提交不得写为成功诊断或封存。不可通过恢复重放导致预算超限。

不同实现若原引擎的原子边界无法复用，需要显式的小型adapter并列出逐操作等价，不允许在共享模块里改全局行为。

## H. 诊断应测什么

四固定点/阶段。40点全保留，不用主loss最后一次覆盖前面结果。
- 每图fine/coarse/ignore比例；已选coarse的概率、分歧、risk与alpha；缺类按NA/计数报告。
- 当前L“影子选择器”使用同一teacher/PAS/风险规则，仅在掩码产生后用L真值评价，不用标签修改已生成mask；明确这是训练L诊断，不能称独立泛化精度。
- 主监督/KL/DS/H的原始、加权loss，all_AB与辅助可达上游参数/heads的梯度范数、夹角；缺失项not_applicable。
- 辅助头与主输出在父类和细类上的L误差；uniform与risk加权的选择差异；空mask/全拒绝的频率。
- 原teacher/U/clean-L/aux-head前向、额外VJP、显存、时间；语义子计数不能重复加总。

不增加Adam双候选计量；已有剂量实验已提供“确实改变更新”的证据，新研究重点是监督信息的正确性与覆盖。

## I. 资格测试清单

纯数学/元数据：三类→disc映射、ignore与geometry、fine/coarse互斥、.04/.48/.48例子、全空graph-zero、logsumexp数值稳定、梯度不触及teacher/R、alpha归一与下界、缺类Brier不制造伪精度、风险滞后一拍、不同坐标mask、同一LCTX源收集、越界canonical重算hash拒绝、历史进口原身份。

原生生成模型：
- A0关闭等价原B2，逐参数/EMA/优化状态/数据读取/随机流一致；不足时阻止历史复用。
- 两辅助头在optimizer/EMA中；主头仍冻结；无额外sidecar/R；A/B硬投影保持；辅助grad真的影响上游。
- A2-A5初始主/aux一致，开启风险不改变基础RNG。
- 最低构造所有A0-A5，在active阶段验证mask分支，不仅跑warmup。
- A1和A5真实新trainer的连续/恢复等价；保存的R、头、EMA、optimizer、support、diagnostics均恢复；跨arm/前缀/温度/层级/头schema拒绝。
- 所有新aux权重为0时原B2等价；禁止恢复时随机重新初始化teacher头或risk。
- 部署丢弃全部训练期状态后，主输出等于丢弃前主输出；不能用融合输出替代。
- 实际模型hash/schema/finite验收、失败记账、尾段不重放、报告覆盖及单seed门槛缺项正确处理。

CPU新套件每次预声明<=32 calls/总96/最多3 attempts，设计可使用：六臂warmup+active两步=12；A1/A5各一warmup+两连续+两恢复=10；原B2与A0对照各两步=4；两失败调用=2，总28。实现若需要不同调用分配，必须先在EXECUTION_PLAN冻结并保持当前总和单次上限，不可测试后改账。

CUDA计划36，smoke24，P0最多32图像，P1正式26500。生产前缀、患者、CUDA均不得在代码准备阶段触碰。

## J. 状态与交付

checkpoint须绑定study/arm/seed/order/stage/完整loss配置/层级树/坐标变换/aux schema/前缀/数据/风险规则/命名空间。训练resume与stage-exit不是同一操作：resume保留当前全部状态，stage-exit丢弃辅助与风险再以合并主权重开始下一阶段。

推理验收：仅主网络；真正加载student文件检查身份、step、tensor schema/finite/hash；metadata_sealed不等同模型verified。

权限：新study新审批，外部审阅JSON绑定最终完整commit、代码树、科学/执行计划、前缀、导入及环境。用户确认单独存放，均在干净checkout外。不得生成APPROVED占位符，不复用旧DOSE/NKA批准。

新的CPU次数与旧134分别统计；旧报告不更新不清空。真实执行/发布/科学结论分别记状态。未来P1R/P2/P3不生成可执行任务，不恢复监测。
