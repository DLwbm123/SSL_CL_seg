# 单前驱教师 R1：D 臂有效性归因＋一次 E 数值收尾

## 0. 任务、起点与边界

请实际执行本协议，而非仅返回新的计划。目标是优先确认已出现效果的简单方案，以及无标注数据的真实贡献；E 的求解器修复是独立、受限的收尾，不得阻塞其他实验。

仓库：DLwbm123/SSL_CL_seg。
起点：bd29494153aa0175b5b52d360f87a44255293d9f。
新分支：codex/single-teacher-r1-effectiveness-ablation。
新文件限定于：
- experiments/lcrseg/single_teacher_scd_r1/
- experiments/lcrseg/tests/single_teacher_scd_r1/
- experiments/lcrseg/docs/single_teacher_scd_r1/

旧单教师 V0.1、历史路由研究、原授权锁及所有结果保持不变。不得更新旧报告为PASS。新结果通过来源引用与旧完整臂比较，不覆盖原实验。

已知证据起点：
- V0.1报告：3109e5a9d9dec2d75a4135b2f5c5b1293c99aa43。
- 公共stage0及S/A/B/C/D源：057ce07fa7bd02f8319e0c2bdd9d4adbceff23d8。
- 最后一次失败的E源：270e23985c8d78d0508fe6c4d43150f6ebffcc92。
- V0.1终态：INCOMPLETE_TRAINING_MATRIX。

本次仅seed0、相同Fundus三域、相同train/val角色。
允许：三个新增D归因臂的真实训练；有条件的一次E重新训练；独立evaluator对已见域val评分；数值修复及必要诊断。
禁止：seed1/2、test、旧formal_03、外部数据、旧专家库、历史训练图像/标签/特征回放、第三完整模型、EMA、改阈值/增强/网络/学习率以救结果、合并main。

本协议不是对旧完整矩阵科学门槛的重新定义。旧B-S半监督支持条件已经具有不满足的描述性数值；修好E不能改变那个事实。本轮分开输出：工程完成、分割巩固效果、无标注贡献、E增量价值，不能把它们合并成误导性的总PASS。

## 1. 先确认已有结果，不重跑完整臂

只读核验并复用原common、S/A/B/C/D的配置、最后一步学生、stage-domain矩阵、单学生部署记录及来源hash。不要重新训练公共stage0或S/A/B/C/D。

核验公开结果的精确值与私有收据一致，缺失不得用四舍五入表格代替：
- S: F=0.585957181..., H=0.525668818..., N=0.704849150...
- B: F=0.503730021..., H=0.417375901..., N=0.692086118...
- C: F=0.684184175..., H=0.747438542..., N=0.630016095...
- D: F=0.718089435..., H=0.742590645..., N=0.698910789...

特别保留D的stage2 cup=0.6351047008700549及S=0.7457152979081902；不能只用F或N掩盖类别代价。

E旧的1+1507次成功更新、最新持久checkpoint=1504步、诊断重放及失败源均单独保留，不当作新E的训练进度。

输入若真正缺失，应报告具体缺失项；不能重新选择公共初始化、历史teacher或数据split。公共stage0只作stage1初始化，stage2只能用本新臂的stage1最终学生。

## 2. 主实验：三个固定的D归因臂

记Lsup为原当前监督CE，Lu为原学生弱到强PAS伪标签CE，Kl/Ku为原D的标注/无标注分支冲突删除KD。

旧D（只读参照）：Lsup + lambda_u(epoch)*Lu + 0.5*Kl + 0.5*Ku。

新增：

| ID | 精确定义 | U图像 | 用途 |
|---|---|---|---|
| U0 | Lsup + 0*Lu + 0.5*Kl + 0.5*Ku | 读取 | 只移除伪标签CE的梯度贡献 |
| L05 | Lsup + 0.5*Kl | 不读取 | 保留相同标注KD系数，去掉全部U分支 |
| L10 | Lsup + 1.0*Kl | 不读取 | 检查总KD系数差异是否足以解释U0的收益 |

这三项不是可调的KD sweep；0.5/1.0由原D的两分支平均定义直接给出。不得增加第三个系数或按结果选择新的值。

### U0的严格语义

- 学生弱前向、原型、PAS、强增强、学生强前向和前驱教师U前向全部保留。
- Lu照常计算和记录，但总loss中的系数严格为零，warm-up后也为零。
- 原lambda_u调度另记为nominal，actual_lambda_u=0，不把名义值记录为实际参与值。
- PAS仍为Ku提供当前参照/可靠mask；无可靠参照的位置仍保留普通KD。
- U0不是纯监督，也不是彻底去除伪标签机制；只检验显式pseudo-CE的边际作用。
- 按独立随机流保持移除loss不会改变data/augmentation/GAS采样序列。

### L05/L10的严格语义

- 只读取当前train_labeled图像与标签；允许用U行数匹配optimizer预算，但不得读取U图像/GT。
- 只有学生和冻结t-1教师；teacher在标注view上预测，可靠参照来自当前GT。
- 直接调用冻结D的冲突定义和标注分支KD；没有PAS、伪标签、U-KD或EMA。
- 监督权重、GAS、标注呈现次数与原D相同。不得额外重复监督CE以伪装匹配。
- 这两个对照必须分别报告；不能将其中之一的失败隐藏，也不能拿U0仅胜过L05就声称无标注数据不可替代。

### 固定训练规则

完整继承原V0.1协议：UNet/GAS分类器，全部原可学习权重更新，FP32网络与原FP64 KD，禁AMP/TF32，Adam/学习率/调度/增强/采样/梯度来源/原型刷新规则不变。
- stage1: 100 epochs ×32 steps=3200。
- stage2: 100 epochs ×21 steps=2100。
- 每个新增臂共5300步；三臂共15900次成功更新。
- stage1从已验证公共stage0最后学生开始。
- stage2从本臂新stage1最后学生开始，Adam按原规则重置。
- 最终学生，不按val选择best、不早停、不调参。
- 禁止把新ID错误地传入旧字符串成员判断并走错分支；新runner显式枚举行为，并对目标恒等性作单元测试。

## 3. E数值收尾：独立条件，不阻塞上述三臂

只允许一次有针对性的数学数值修复。先离线解决已保存失败输入和合成见证，通过条件后才训练E_R1。不得用继续调大循环次数、跳过样本、r=p fallback或关闭KD来掩盖问题。

### 3.1 已确认的见证

读取V0.1的BRACKET_FAILURE_REPRODUCTION.json及SYNTHETIC_BRACKET_LIMIT.json。
旧代码已经按max|a|归一化，不能把“加归一化”当作新修复。
公开合成例：
p=[1,1e-30,8e-48], y=0,
q=[0.9999999933333333,3.3333333333333334e-9,3.3333333333333334e-9]。
归一化方向=[0,1,8e-18]，b=1e-30。
旧cap=2^60约1.15292e18，正残差约1.63214e-30。
独立CPU/100位参考根约1.2738962039568065e18，确实位于cap之外；所以该例不是无解。
包中的脚本只验证公开合成例，不是正式GPU求解器，也没有读取8423个私有失败像素。

### 3.2 必须保持的优化问题

保持p、q的定义、q的eps=1e-8正支持平滑、reliable mask、d_raw<0冲突判定、KL方向、detach、T=1及实际KD归一化不变。
只替换E的projector数值实现和相关数值诊断；原objectives.py与旧实现不修改。
目标仍是min KL(r||q), r在概率单纯形内，<a,r><=b，其中a=p-e_y，b=<a,p>。

### 3.3 建议采用的可证明括区（不是绝对残差救援）

在既有正比例归一化后，令m=min(a), d=a-m, v=b-m。
q0mass=sum_{d=0}q, delta=min_{d>0}d。
若v>0：
  E_eta[d] <= (sum q*d / q0mass)*exp(-eta*delta)。
因此可在对数域构造可行上界：
  eta_hi=[log(sum q*d)-log(q0mass)-log(v)+log(2)]/delta。
这里log(2)只给可行端留数学余量，不改变目标约束或残差门槛。

若v=0且存在最小值支撑：精确边界解是将q限制到argmin(a)后归一化；这是KKT/极限解，不是失败后任意fallback。
若a常量且原目标可行：保留q。
若检测出v<0、输入非有限或真正不合法：错误不能截断为0继续。查清浮点消差及p归一化语义，出具修复前后差异；不得悄悄改变公共p构造。

实现要求：
- 采用logsumexp或等价稳定的加权矩比较，避免对接近边界的b/min(a)作灾难性消差；明确描述与原约束等价的计算。
- 小正v不能用“很小”直接改成0；q正支撑、softmax下溢0及0log0边界分别处理。
- 针对实际动态范围选择有限数值区间及求解坐标；eta非常大时可在log1p(eta)坐标二分，避免巨大初始区间使固定线性二分失效。
- 只要可行端、目标最优性/高精度参照不能得到证据，不训练E。
- 生产不得逐像素Python/SciPy，不把整个网络转FP64；GPU分块不超过原65536。
- 新数值迭代规则在真实重新训练前冻结；原归一化约束精度1e-9不放宽。有限数值支撑之外显式故障。
- eta、log_eta和梯度范数等诊断的溢出也需正确表示，不能诊断写JSON失败导致训练再次中止；不把合法无限边界eta塞进普通有限JSON数字。

### 3.4 E资格门槛

1. 原实现重现公开合成例的cap耗尽，新实现返回有证书的可行解。
2. 读取已保存的真实失败像素向量，完整测试所有失败/非失败输入，不挑选部分像素。
3. 高精度独立参照验证小正v、精确边界、近one-hot、最小方向差、多类、置换、C=2退化以及已有普通输入。
4. 非冲突目标保持原q；未受故障影响的普通样本与旧解误差应在冻结容差内。不能要求已知错误解为golden；差异须由独立参照裁定。
5. 实际autograd验证detach及同一次logits的梯度，GPU与CPU参照一致；禁止仅测公式不测训练调用链。
6. 同一合成输入下C/D分支的loss、grad、GAS、optimizer更新不变；原熵边界修复保留。
7. 必要时最多8次成功真实诊断更新，隔离输出、旧checkpoints只读，绝不拼入正式轨迹。其余数值测试优先使用已保存向量，零网络前向。

若资格不通过：E标记E_NUMERICAL_QUALIFICATION_FAILED，终止E续接；仍执行并发布U0/L05/L10结果。不进入第三次数值救援。

### 3.5 资格通过后的唯一E训练

- E_R1从同一公共stage0最后权重重新开始stage1，不从旧1504/1507步checkpoint拼接。
- 因为新求解器可能改变此前多个数值目标，从头训练避免声称不成立的轨迹等价。
- stage1 3200 + stage2 2100 =5300步；stage2用新E_R1 stage1学生。
- 其余loss/SSL/GAS/增强/精度不变；E_R1是旧数学目标的数值修订实例，不是未变字节的旧E。
- 一次正式轨迹。硬件中断只能按同源码、完整状态的既有resume协议恢复；新的数值失败即停止，不再边跑边修。

## 4. 训练之前与训练之后的诊断

### 4.1 先使用已有聚合日志，零新增训练

报告原B、C、D各类PAS覆盖和拒绝、KD冲突比例、label/U分支有效质量、loss与梯度量级。高coverage不是高precision；原文件pseudo_label_accuracy=NOT_EVALUATED须保留。

可以由隔离evaluator使用原最终B/D学生和当前域val评估伪标签precision/coverage，但该结果只作事后描述，不生成新阈值或修改本轮配方。若需要过去阶段已删除权重，不补取更早模型或重建它。

### 4.2 新臂运行时必要记录

固定记录每5个epoch和阶段终点的聚合量：
- supervised / raw Lu / weighted Lu / labeled KD / U KD；
- background/rim/cup各自的预测占比、PAS覆盖；
- labeled和U各自冲突率、删除KD质量和梯度代理；
- 学生logit范围、p_max分位数、数值one-hot比例，分label/U记录；
- E求解边界分支数、解析括区数、实际迭代、最大残差、耗时；
- 有限性、teacher hash、GAS来源、两模型限制与tensor bytes。

伪标签precision只在当前域val由隔离evaluator计算，绝不读取train_unlabeled隐藏GT。明确模型posterior-mean评估与训练随机头/强视图不同。precision不用于训练或模型选择。

不新增逐步高开销全参数多次autograd；必要工程抽查有固定预算并单列。

## 5. 评价与判断：拆开四个问题

主指标沿用原METRIC_DEFINITION和逐病例rim/cup宏平均：
F=最终三域均值；H=最终两个历史域均值；N=各增量阶段进入域均值。
报告完整阶段×域×类别，BWT和每个历史域遗忘。最终学生、val-only、无teacher部署检查不变。

### 5.1 原V0.1状态

始终保持INCOMPLETE_TRAINING_MATRIX。新报告可以给出来源明确的“旧五完整臂＋新E_R1”补充矩阵，但不改旧终态。
原B-S支持条件的观察值保持不变；不能用新U0取得效果来给旧B或旧完整方法改判。

### 5.2 显式伪标签CE是否有价值

固定比较旧D与U0。报告D-U0的F/H/N及每阶段每类。
本轮开发信号要求：F或N至少一个提升0.005，另一个不下降超过0.005，同时当前任一rim/cup不下降超过0.02。
不满足则PSEUDO_CE_CONTRIBUTION_NOT_ESTABLISHED；不得笼统写“所有半监督学习无用”。
U0更好时可以说明本配置下显式pseudo-CE可能有害，但PAS参考仍用于U-KD，这不是完全无伪标签的方法。

### 5.3 无标注图像是否提供不可由KD系数解释的贡献

U0同时与L05、L10比较。保守的开发支持标准：
- F_U0 - max(F_L05,F_L10) >=0.005；
- N_U0不得比任一label-only对照低超过0.01；
- 每阶段当前rim/cup不得比任一label-only对照低超过0.02。
否则UNLABELED_KD_CONTRIBUTION_NOT_ESTABLISHED。
另报告H差异；两label-only对照都保留，不能只选有利者。
这组比较回答实际新增U-KD通路是否有帮助，不声称单独识别了所有表示/正则/数据量的因果作用。

### 5.4 E是否值得额外复杂度

仅E完整完成且部署正确时评价：
- F_E - max(F_C,F_D,F_U0,F_L05,F_L10) >=0.01；
- H_E >= H_B +0.01（原历史保留要求仍报告）；
- 原对B的逐阶段macro下降<=0.01、逐类下降<=0.02均报告。
若只胜过较弱C而未胜过D/新简单对照，SCD_INCREMENTAL_VALUE_NOT_ESTABLISHED。
数值资格失败或中途异常，标记NOT_EVALUATED，不填0、不以D代替E。

### 5.5 任何候选进入后续种子前的当前能力门槛

这是新的前瞻性续投筛查，不追溯修改旧判定：候选须在每个增量阶段对S满足macro下降<=0.01且rim/cup分别下降<=0.02，并且F比S高>=0.01。
即使整体F高，仍有类似旧D的stage2 cup约11个百分点损失，就不能宣称已经满足当前能力保护。
本轮不自动跑种子1/2；符合只输出FIXED_RECIPE_CANDIDATE_FOR_REPLICATION。
单种子、同开发split不是稳定性或SOTA证明；case bootstrap不是training-seed不确定性。

## 6. 预算、执行顺序和停止

先完成三臂及E资格的代码/配置冻结、必要合成回归和生产调用链见证。
然后优先启动U0、L05、L10。服务器/GPU必须实时只读确认可用，不能沿用旧“空闲”状态，不终止他人进程。
E仅在第3节全部通过后启动；其他三臂不得等待E数值修复反复循环。

本轮新正式成功更新：
- U0/L05/L10：15900。
- 条件E_R1：最多5300。
- 总上限：21200。
- 不重做common8000或旧五臂26500。

历史36008次正式更新保留；若本轮全部完成，跨尝试累计为57208，不能称为39800。
“旧五完整臂＋新E”的有效比较矩阵预算可单列39800，但不是历次实际计算成本。
工程回归/重放/warm-up均单列。无资格臂或失败臂不能计作完整5300。

不承诺45–60分钟。先用已记录runtime及本次实际吞吐量报告范围；明确epoch时间总和、并行wall time、资格成本和失败步骤，状态变化时即时修正估计。

数值E再失败就关闭E。三个简单臂任一工程失败单独记录，不让全局状态抹去其他已完成结果。禁止根据新val结果改配方再跑。

## 7. 必须交付

新目录中至少包括：
- R1_PREREGISTRATION.md/json、SOURCE_FREEZE、HISTORY_PROTECTION、INPUT_REUSE；
- ABLATION_CONTRACT：原D/U0/L05/L10每一项损失和实际系数；
- SOLVER_FAILURE_ANALYSIS、NUMERIC_EQUIVALENCE、CPU_HIGH_PRECISION/GPU_PARITY；
- PRIVATE_FAILURE_CORPUS_MANIFEST（仅脱敏hash，私有向量不公开）；
- QUALIFICATION、每臂真实父进程退出记录；
- STAGE_LINEAGE、UPDATED_BUDGET_AND_ATTEMPTS、MEMORY_AND_DEPLOYMENT；
- STAGE_DOMAIN_MATRIX、PER_CLASS、METHOD_SUMMARY、PSEUDO_CE_ABLATION、UNLABELED_KD_ABLATION；
- CUP_RETENTION_AND_ADAPTATION：单独展示stage1/2 cup及rim，不只均值；
- 数值/E组件状态、SSL支持状态、当前能力状态分别列出；
- FINAL_REPORT、NEXT_ACTION（不自动执行）、PUBLICATION_VERIFICATION。

公开结果不含原图、GT、身份、密码、完整私有向量或模型权重。旧源/旧结果逐文件保护。匿名访问与NAS归档核验是发布核验，不替代科学结果。

最终回复先给出：各臂是否完成、U0/L05/L10和条件E的F/H/N及每阶段rim/cup、无标注贡献判断、E增量判断、训练成本和明确下一步停止状态。不要将“又通过更多单元测试”作为主要研究结果。

## 8. 本计划的依据

已读取的公开源码和报告（统一固定引用，不跟随main）：
- https://github.com/DLwbm123/SSL_CL_seg/blob/3109e5a9d9dec2d75a4135b2f5c5b1293c99aa43/experiments/lcrseg/docs/single_teacher_scd_v0_1/FINAL_REPORT.md
- 同目录 BRACKET_FAILURE_REPRODUCTION.json、SYNTHETIC_BRACKET_LIMIT.json、SSL_DIAGNOSTICS.csv、STAGE_DOMAIN_MATRIX.csv。
- https://github.com/DLwbm123/SSL_CL_seg/blob/270e23985c8d78d0508fe6c4d43150f6ebffcc92/experiments/lcrseg/single_teacher_scd_v0_1/objectives.py
- 同版本 engine.py：原loss=Lsup+lambda_u*Lu+0.5*(Kl+Ku)。

解析括区界及本轮新归因臂属于本计划推导/实验设计，不是这些旧报告已经完成的结果。
