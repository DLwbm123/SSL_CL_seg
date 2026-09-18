# NKA_DOSE_V0_1 — 有限辅助强度实验计划

状态：PROPOSED_CODE_PREPARATION_ONLY。本文不是外部审阅批准，也不是生产启动确认。
旧 NKA-D1 的负结果、停止决定、代码和审批保持封存；本轮是新的开发研究，不是旧 D1/D2 的自动续跑。

## 1. 目的与边界

只回答两个问题：
1. 将现有辅助系数从 0.05 提高到 0.5 或 2.0，能否在不改变 B2 学习路径的前提下带来实际收益？
2. 同一系数下，原生 V 键 C3 是否优于同维度随机键 C2？

旧结果 C3−C0 约 +0.002122 个百分点；固定诊断点上的辅助/监督梯度范数中位数约 0.1682%。这支持做一次有限强度检验，不证明增大系数会提高 Dice。
0.5、2.0 分别是旧系数的 10、40 倍；仅在同一模型、batch 和损失状态下，辅助梯度随系数线性缩放。新轨迹、Adam 增量和 Dice 不按此比例预测。

不增加模型模块、层、秩、采样量或投影方向；不加入动态加权、梯度归一化/手术、额外教师、历史样本、历史原型/特征队列。不把调系数或只读优化器计量宣称为新方法创新。

## 2. 代码与证据锚点

- 实际 NKA-D1 训练代码：145b3c1b3ab301031ea379c9d67452e57b28ae1d。
- NKA-D1 结果：0da596f4050a3701f41ff74248ca9d4ba21222e2，results/native_key_alignment_v0_1_d1/。
- 四个 B2 第一目标前缀的公开来源：3034b2199aa7d6e54ea67499f391a0dd4ea3d21e。
- 复用原 PREFIX_BINDINGS.json、FROZEN_OPTIONS.json、alignment.py、原数据/优化器/EMA数学逻辑；不改写这些封存文件。
- 新分支建议 codex/nka-dose-v0-1；新增 experiments/lcrseg/nka_dose_v0_1/ 及对应 docs 目录。

D0 准备必须读取适用 AGENTS/已有合同和上述实际文件；凭空猜测的 hash、审批、患者路径不可用于绑定。

## 3. 唯一新增正式矩阵

| 辅助坐标 | lambda_align | seed | 顺序 | 新阶段 | 更新 |
|---|---:|---|---|---:|---:|
| C2：固定随机 8 维键 | 0.5 | 163、164 | O1、O2 | 4 | 10,600 |
| C3：原生入口 V 的 8 维键 | 0.5 | 163、164 | O1、O2 | 4 | 10,600 |
| C2：固定随机 8 维键 | 2.0 | 163、164 | O1、O2 | 4 | 10,600 |
| C3：原生入口 V 的 8 维键 | 2.0 | 163、164 | O1、O2 | 4 | 10,600 |
| 总计 | | | | 16 | 42,400 |

O1：从已学 RIM_ONE_r3 的 B2 第一目标学生出发，只训练 Drishti_GS，2,100 步。
O2：从已学 Drishti_GS 的 B2 第一目标学生出发，只训练 RIM_ONE_r3，3,200 步。
因此恰有 8 个 Drishti 阶段和 8 个 RIM 阶段。新增 source/第一目标训练均为 0。

每个新节点都从相应 B2 第一阶段封存学生独立初始化，不从旧 C2/C3 最终学生、高低剂量终点或其他臂的 warmup 初始化。共享的只是规定前缀；A/B、当前 EMA、optimizer、原型、warmup 和数据游标按原规则独立重置。

新节点建议格式 DOSE__C3__L0p5__S163__O1__STAGE2；身份新增 dose_id 和 lambda_align，不改变随机流。

## 4. 历史复用：12 条记录，不重训

导入旧研究：C0 × 4 配对，以及 C2/C3、lambda_align=0.05 × 各 4 配对，共 12 条第二目标最终轨迹。
每条标记 historical_import，保留实际 source commit/node/identity；不能改成 new_execution。
与 16 条新轨迹合并，主最终表 28 行。旧 C1 0.05 可作为背景链接，不混入主剂量矩阵；本轮不运行高剂量 C1，不能声称胜过同剂量完整 patch 对齐。

复用条件：相同前缀、数据 manifest/split、基础 B2 语义、共同随机流和数值环境；生成模型上的旧/新 0.05 以及新计量开/关等价测试通过。聚合指标相同不是实际权重逐位相同的证明。
从公开的 RUNTIME_ENVIRONMENT.json 绑定旧软件、CUDA/cuDNN、驱动、设备类型、确定性/TF32/thread 等指纹；不要假定当前服务器仍相同。CUDA 阶段实际比对失败则 BASELINE_REUSE_BLOCKED，不默许换环境、不暗中增跑 C0。
D0 不加载真实前缀；真实文件验收留给新审阅和用户启动确认之后。历史 C0/低剂量无需加载模型补造 Adam 诊断，缺失字段为 NOT_MEASURED_HISTORICAL。

## 5. 固定科学配置

L = L_sup + L_parent_constraint + ramp * [1.0 * KL + lambda_align * L_align]。
L_align 为已实现的当前 batch 类别条件 SWD（包括原 D/8 缩放；这里 C2/C3 的 D=8，系数为 1）。不得把 lambda_align 乘进 lambda_U 再乘一次，不能混淆遗留且不参与此路径的 lambda_SWD 字段。

父方法：NATIVE_LR_SRC_A_3DOMAIN_V1 / B2_C06；原 A-only 硬输入投影不变。
A/B 初始学习率均为 0.0005；Adam betas=(0.9,0.999)、eps=1e-8、weight_decay=4e-5；基础 lr=0.001、parent_lr_multiplier=0.5、lr_B_over_A=1。
保持原多项式调度、warmup_fraction=0.2、U_ramp_fraction=0.2、dense EMA、FP32/无autocast、batch=2、384×384 和读出几何。
KL 权重 1；PAS confidence=0.6、cosine=0.7。U 权限沿用 B2，不能变成 R-only。
辅助位置 decoder.dec1.merge.block.3 的输入；144 维 patch → 8 维；L2 normalize eps=1e-6；每类至多64、至少8样本；32投影；rim/cup按有效类别等权；geometry/ignore patch侵蚀与PAS中心规则不变。
C2只替换辅助基，父保护 V 不变；C3引用原入口 V，无新SVD。

新 study/run/dose_id 不得进入旧训练、增强、像素抽样或随机基的随机种子。保留 NATIVE_KEY_ALIGNMENT_V0_1 的辅助随机 namespace，是为保持比较，不等于复用旧授权。
同 seed/order 的 C2 在所有剂量下共享同一固定随机基，C3共享同一入口V；在相同状态测试随机流相同。训练分叉后 teacher/PAS/类别集合自然可以不同，不能强行回放另一臂的mask或承诺全程实际采样位置完全相同。

## 6. 核心新诊断：同状态的 Adam 一步对照

### 6.1 时间点

按 successful update 编号，执行更新前测量：
- Drishti：525、1050、1575、2100。
- RIM：800、1600、2400、3200。
共16阶段×4=64个正式诊断点。不要在 scheduler 已推进到0后推算末步；不用最终checkpoint代替早期状态。

### 6.2 梯度的求法和原训练保持

同一已有计算图上分别求：
- gS = grad(L_sup + L_parent_constraint)；
- gK = grad(已有的 kl * opt['lambda_U'] * ramp 标量)；
- gA = grad(已有的 ramp * lambda_align * L_align 标量)；
- gU = grad(原训练实际返回的完整 unlabeled 标量)。
最后一个直接引用 K+A 的原标量对象，不另用等价括号重建；上面的目标公式只是数学表述，不许可改变浮点乘加顺序。

第四次 VJP 用于避免把分项梯度相加时的浮点差误认为优化器影响。
按原 split_gradients 的顺序及 None 语义得到 g0=merge(gS,gK) 和 g_lambda=merge(gS,gU)。不可用 Adam(gS)+Adam(gK)+Adam(gA) 代替。
这些额外求导只读、retain_graph；原 StageTrainer 仍用原来的 L/U 两次求导产生实际梯度并只执行一次 optimizer.step。不得用诊断算出的分项梯度替换实际训练梯度或改变求和顺序。
正式每点4次额外 VJP，共256次。计量不增加模型前向；正式原有 extra clean-U 前向仍为33920次。若实现不能满足该路径/计数，代码审阅前说明，不训练后重定义。

### 6.3 Adam 候选步

从同一个更新前的 (theta, exp_avg, exp_avg_sq, per-parameter step, param groups, lr) 独立计算 u0 与 u_lambda，分别使用 g0 和 g_lambda。
必须匹配实际安装的 PyTorch 2.2.1 Adam 分支，包括实际 foreach/fused/capturable/amsgrad 等标志，或显式拒绝本协议不支持的配置。不能使用当前 main 版本的新参数语义替代旧版本。
weight_decay 是该版本 Adam 的耦合 L2 项，不是 AdamW；grad=None 的参数不做该参数的step/矩更新，零梯度与None不同；未初始化状态、不同参数step、零/极小lr、bias correction与eps位置均须覆盖。

正式运行中仅允许无副作用的独立数值公式，detach/clone，不可调用 torch.optim.Adam.step 或其in-place functional更新接口来执行第二个候选步。公式也不更新正式参数、m/v、scheduler、EMA或训练cursor。
在生成参数的CPU/CUDA资格测试中可用真实Adam更新可丢弃副本作oracle，所有oracle optimizer调用记入资格预算。
64点×2=128次完整候选计算，另记只读算子/时间，不称为128次训练更新。

### 6.4 输出量

rho_theta = ||u_lambda - u0|| / ||u0||。
W_eff,l(theta)=W0,l+B_l[A_l-(A_l V_l)V_l^T]，按原参数化及dtype运算计算三个状态，不省略 DeltaB*DeltaA 交叉项。
rho_W = [sum_l ||W_eff,l(theta+u_lambda)-W_eff,l(theta+u0)||_F^2]^0.5 /
        [sum_l ||W_eff,l(theta+u0)-W_eff,l(theta)||_F^2]^0.5。

保留分子、分母、绝对增量、实际学习率，按全部A/B、固定辅助可达上游层、每层、A/B组报告。可达集合以架构/导数路径冻结，不用每次“非零梯度”临时挑选。
分母为零或落入预先声明的数值解析阈值时，写null与原因，不以epsilon硬凑极大比例。可另在FP64汇总范数，但不改变FP32训练或把FP64候选误称为实际FP32更新。
在资格测试及正式诊断点，将预测的 theta+u_lambda 与随后唯一真实更新后的 theta 对照，记录误差；容差从生成资格中事前固定，不根据真实结果放宽。用完整候选参数误差并同时报增量误差，不能只用参数相对误差掩盖微小更新的不准。

保留每类支持/覆盖、原始和加权损失、梯度范数/夹角、有效patch数量；不在线调整lambda或学习率。
这是在高剂量训练状态上的局部一步比较，不是完整的“无辅助训练轨迹”，也不是相对历史C0同一状态的因果结论。

## 7. 有限资格预算

当前转发准备prompt仅授权代码、元数据和新账本下的CPU生成测试。其余均是待审执行定义。

| 类别 | 本轮限额 | 说明 |
|---|---:|---|
| 新 CPU 生成资格 | 累计96次、最多3次调用，每次计划≤32次 | 每次预声明case/call数；失败和oracle调用全部计数；不清空旧64次账本 |
| 合成 CUDA | 40次 | 新四cells各6次=24，含4次预设失败；两个坐标低剂量旧/新计量开/关对照各6次=12；生成参数Adam oracle4次 |
| 真实 L-only smoke | 32次 | 新四cells各8步，绑定seed163/O1前缀与正式warmup；不读U/val/test；全部状态丢弃 |
| 正式 | 42,400次 | 科学更新与物理调用上限相同 |
| 真实合计 | 42,432次 | formal+smoke；不包括合成CPU/CUDA |
| 新source/第一目标训练 | 0 | 不补训、不借预算 |

CUDA 40次详细结构：
A. (C2/C3)×(0.5/2.0)，每cell warmup1、continuation2、restore continuation2、after_optimizer失败1，共24；合成活跃分支须有非零辅助支持和数值有限。
B. C2/C3各比较旧0.05两步、新0.05且计量开启两步、新0.05且计量关闭两步，各6，共12；比较模型/EMA/optimizer/scheduler/原型/读取计数/RNG，计量专属counter允许不同。0.05只在生成资格中可执行，不能进入本轮正式nodes。
C. 4次可丢弃小参数oracle optimizer调用，每次可含多个param groups/None/zero等场景；验证纯函数公式和实际已安装Adam路径。有效权重映射和拒绝测试不需增加optimizer调用。

CPU测试可以分组执行，但所有尝试与每次计划必须先持久化。历史原生NKA CPU64次、CUDA24次、smoke32次、formal42400次是另一研究的成本，不能混写为本轮执行或抵扣预算。
非预设CUDA/真实故障停止并保留证据，不自动重试；用例必要性超过计划时必须在审阅前调整定义，不事后静默扩容。

## 8. 冻结统计、门槛与停止

每个固定lambda分别比较 C3-C0、C2-C0、C3-C2；先 seed/order 内配对，两个顺序在seed内等权，再平均seed。
报告0.05为已观察的低剂量历史信息，0.5/2.0为新执行；C0只导入一次，不将重复作参照的C0计为多次重复。
每个lambda有三个contrast，每个contrast 4个seed/order +2个seed均值+1个总体均值，共7行。三剂量总计63条主配对摘要。
所有结果均为开发信息，不能声称两seed×两order是四个独立seed或独立患者验证。

性能推进门槛 G_perf(lambda)：
1. C3-C0跨seed均值 Final >=0.005 raw Dice（+0.5百分点）；
2. 两个seed各自双顺序平均 DeltaFinal 都>0；
3. O2跨seed平均 DeltaOld>=0；
4. O1和O2各自平均 DeltaIncoming>=-0.005。

新增的参数键差异开发门槛 G_key(lambda)：
- 同lambda下C3-C2总体 DeltaFinal>=0.001 raw Dice（+0.1百分点）；
- 两个seed各自双顺序平均 C3-C2 DeltaFinal 均>0。
这是本轮新冻结的投入决策阈值，目的是不再用极小正号充当充分证据，不是统计显著性或一般公认界值，不改写旧D1。

只有同一个全局lambda同时满足G_perf和G_key才建议完整轨迹的另行审阅；两者都合格则预先选择较小lambda=0.5，不按seed/order选系数，不按看到的结果另改择优规则。
只有G_perf通过而G_key不通过：最多支持一般对齐候选，不支持参数键独有贡献；不自动推进NKA。
两个新lambda都未通过性能门槛：关闭本剂量研究及当前NKA-SWD候选，不继续加大lambda/加层/改秩/降PAS搜索。
所有16节点结束后统一判定，无性能剪枝。无论结果如何，均不自动进入完整轨迹、D2/D3、新seed、其他数据集或追加实验。
共同前缀下 DeltaForget=-DeltaOld，两者不是独立保持证据。

## 9. 权限、恢复和报告

D0必须同时实现有限生产路径，但审批和用户启动确认缺失时拒绝执行，不能把“接入代码未完成”伪报就绪，也不能自生成批准。
新研究的canonical plan/剂量/node/prefix/import/environment/execution/caps/完整代码指纹全量绑定。旧审批不能授权NKA_DOSE。
现有KeyAlignmentTrainer正系数固定0.05且含exact-type检查。不得只改旧常量或传字段冒充高剂量；新增窄适配与独立能力/恢复身份，复用旧数学核和公共单次更新机制，低剂量等价必须核验。

checkpoint包括新study/arm/dose/lambda、完整options、prefix、V/辅助basis、所有优化状态、RNG/游标、支持/固定诊断和新增只读计量版本。恢复必须是新DoseTrainer，并恢复原C2随机基；拒绝错剂量、错prefix、错旧study的恢复。不从另一剂量warmup/终点初始化。
读训练数据只限当前L和当前U图像/geometry；旧域val仅阶段封存后的隔离评价，不读取test/future/U真值。
前缀和最终模型实际验收沿用已审完整性机制。物理调用包括失败尝试；丢失tail导致超预算或成本未闭合即停止，不覆盖封存、不删账本。

准备交付至少：REVIEW_REQUEST、PLAN、EXECUTION_PLAN、PREFIX_BINDINGS、IMPORT_BINDINGS、BASELINE_EQUIVALENCE、ADAM_ORACLE_TESTS、TEST_REPORT及累计COST、CODE_MANIFEST、RUNBOOK。
准备后状态 STOP_AWAITING_EXTERNAL_CODE_REVIEW，真实数据optimizer=0。

生产完成交付：
- FINAL_REPORT.md；
- FINAL_METRICS.csv：16新+12历史=28行；
- DOMAIN_METRICS.csv：每条最终轨迹3域，共84行，第一目标早期值另列；
- PAIRED_COMPARISONS.csv：63条主配对摘要；
- GRADIENT_DIAGNOSTICS.json；ADAM_COUNTERFACTUAL.json：64个新诊断点与scope/lr/分母有效性/预测误差；
- COST_AND_COMPLETION.json、PUBLIC_RESULTS.json、完整性/完成复核。

历史Adam字段标NOT_MEASURED_HISTORICAL；只读公式成本、额外VJP与训练optimizer成本分列，不重复相加。
全部执行、模型、资格、统计与文件schema验收后：COMPLETE_DOSE_AWAITING_SCIENTIFIC_REVIEW。
公开推送和可读性核验是独立状态；不发布权重、患者/逐样本记录、private_val、私有路径和凭据。

## 10. 资料

项目证据在以上固定commit的README、FINAL_METRICS、DIAGNOSTICS、RUNTIME_ENVIRONMENT及trainer/state/authority/protocol中。
Adam实现依据：PyTorch官方仓库 tag v2.2.1，torch/optim/adam.py；以服务器实际安装版本文件与执行参数最终绑定，不以最新文档代替旧环境。
本交付仅为计划和prompt。没有创建审阅批准、用户启动确认、训练节点或新实验结果。
