# LCTX Weight Memory V0.1.1：数值契约修订与固定矩阵收尾

## 0. 执行决策

执行代号：`LCTX_WM_NUMERICAL_RECOVERY_V0_1_1`。

这是 V0.1 的工程恢复与原矩阵收尾，不是新参数保护方法，不是新候选搜索。只修订约束诊断、失败证据保存及任务恢复/计数，不修改训练的有效权重、forward/backward、Adam、EMA、rank、k、学习率、数据流、混合目标和科学门槛。

用户将本文件交付 Codex 执行时，范围包括：有限的无 GT 故障核验、诊断与存档修复、必要资格、18个既有完成任务的无训练复核、唯一失败任务的新 attempt、17个未启动任务、全部权重封存后的 STATIC_SOURCE、完整统计与归档。GPU4/5/6/7共享授权沿用。不重新寻找旧KI，不新增方法、seed、域、标签预算、AMS或患者确认实验。

**重要：继续完成矩阵不以主候选能通过为前提。** 已公开的部分完整任务已经出现科学保护线反例。不得以工程修复之名删除这些结果、选择性重跑或降低科学门槛。

## 1. 已核对的来源与事实

仓库：`DLwbm123/SSL_CL_seg`。

- V0.1部分结果发布提交：`d614de8aaeccc08a8448f0e6460e8a59ed9395c3`。
- 正式训练源码：`50e094dd188c90c0117b62bb63bf961c096ea37d`，以原冻结文件再次核对。
- 原源码目录：`experiments/lcrseg/lctx_weight_memory_v0_1/`。
- 原文档目录：`experiments/lcrseg/docs/lctx_weight_memory_v0_1/`。
- 本次建议分支：`codex/lctx-weight-memory-v0-1-1-numerical-recovery`。
- 本次新增文档目录：`experiments/lcrseg/docs/lctx_weight_memory_v0_1_1/`。

读取原 FINAL_REPORT.md、RUN_LEDGER.csv、DECISIONS.json、AVAILABLE_PAIRED_EFFECTS.csv、AVAILABLE_CLASS_COSTS.csv、INCOMPLETE_EXECUTION_EVIDENCE.json，以及实际存在的源码/配置冻结文件。旧报告、旧运行目录、源权重和基底不覆盖。

原状态：18个训练/评价任务完成；1个失败；17个未启动。完成任务更新45500；失败任务已经执行420；合计45920。原科学矩阵95400步。STATIC_SOURCE尚未执行。原资格成本118个synthetic更新、24个丢弃的real-smoke更新单独保留。曾有一次CUDA冷启动问题，正式更新0，其来源也不删除。

失败任务：`O1_s62_LR_SRC_AB`，RIM→Drishti，epoch20，update420。恢复checkpoint位于update399，存在21步没有完整状态的尾段。不得把399伪装成420，不修剪旧日志后宣称直接resume，也不以打印的loss恢复模型和Adam。

### 1.1 当前科学证据不能只看seed61平均

公开 AVAILABLE_PAIRED_EFFECTS.csv 中，LR_SRC_A相对F_FULL：

| seed/order | Final百分点 | Incoming百分点 | Old百分点 |
|---|---:|---:|---:|
|61/O1|+2.478164|+1.266836|+3.689493|
|61/O2|-0.552853|-0.990338|-0.115367|
|62/O1|-1.545379|+2.454695|-5.545454|

这些都是已完成的配对任务，虽然不是完整三种子矩阵。

公开 AVAILABLE_CLASS_COSTS.csv 已有：

- 62/O1，LR_SRC_A−F_FULL，旧RIM rim：−0.07731558745756356。
- 61/O2，LR_SRC_A−LR_RAND，旧Drishti cup：−0.06875783410804343。
- 62/O1，LR_SRC_A−LR_RAND，旧RIM rim：−0.07962406526873261。

V0.1的VALUE和SOURCE_GEOMETRY包含单个seed/order/role/class差值不得低于−0.050的必要条件。只要上述任务有效且沿用，后续种子不能抵消这些极值条件的违反。旧正式终态仍保持`NOT_ADJUDICATED_INCOMPLETE_MATRIX`；新报告应区分“全矩阵尚未裁定”与“已有必要条件反例”。不要承诺补完后必然得到VALUE或SOURCE_GEOMETRY成功。

补齐的目的：完整量化效应、顺序与种子异质性、F_CONV/普通低秩/随机/源方向/双侧之间的权衡，建立完整参考；不是用新增任务救回既有失败的科学门槛。

## 2. 故障诊断：两个不同的数值对象

原`core.py`：

- `LowRankConv.delta()`按原FP32路径构造投影后的D。
- `LowRankConv.effective()`计算`W0 + D`，加法发生在FP32。
- `diagnostics()`把effective和W0都转到CPU FP64后相减得到Δ_add。
- 同时要求参数化D的投影残差和Δ_add的投影残差都小于1e-5。

注意：原检查已经把权重相减和残差计算转为FP64。问题不能通过给最终残差再加`.double()`解决；FP32权重加法中已损失的更新信息无法由后续升精度找回。

定义：

D = 原训练代码返回的FP32参数化残差；
W_eff = fl32(W0 + D)；
Δ_add = float64(W_eff) − float64(W0)；
E_add = Δ_add − float64(D)。

因此，即使UᵀD≈0，UᵀΔ_add仍可因E_add不为0。用很小的||D||做分母，会放大权重合并舍入。

保存的399步CPU探针中，`bottleneck.block.3.weight`：

- 参数化左残差：3.84662181e-08；
- 原合并权重反算左残差：2.08894503e-05；
- 更新范数约0.00323156591；源权重范数2.71912684；
- 范数比约841.43，FP32 unit roundoff乘此比约5.02e-05。

这是与舍入解释一致的证据，不是对丢失的420步CUDA状态的精确复原，也不是旧域不变或临床可忽略的证明。

### 2.1 必须保留的竞争解释

要检查：真实参数化投影是否失效；运行时FP32基底是否被修改；缓存FP64基底与训练时FP32基底是否一致；源权重是否改变；TF32/AMP/backend是否改变；effective是否包含投影外更新；错误是否只是小D下的权重加法舍入。

不要预先写“已确认纯舍入，无需检查”。只有新的精确状态通过下面的分层检查才可继续。

## 3. 只修诊断，不修改训练数学

冻结以下函数/行为的代码身份与语义：`basis`、`initial_A`、`LowRankConv.delta/effective/forward`、`configure`、训练`step`、`dense_ema`、合并`merge`、优化器构造、标签顺序、LCTX、source/basis/adapter初始化。

复用原源权重和thin V/U/Q/G基底的精确字节及hash。不要为“修数值”重新SVD/QR，不改变k或初始投影归一化。沿用实际训练环境的PyTorch/CUDA/设备类型及精度标志；不升级依赖，不启用/禁用新的TF32或AMP设置。

禁止以下看似方便但会改变实验的做法：

- 把训练的W0+D改为FP64再cast；
- 改成conv(W0,x)+conv(D,x)的双分支forward；
- 改成FP64训练或改变A/B存储精度；
- 为合并权重额外重投影；
- 把旧1e-5直接放大到1e-4；
- 只检查参数化D、完全不记录或约束实际合并舍入；
- 用||W0||代替所有投影残差的分母，掩盖真实D泄漏；
- 因AB表现差就删除AB或只给AB特殊门槛。

### 3.1 新诊断契约：解析约束与表示误差分开

本修改是可审计的数值契约修订。它改变了“舍入后的更新必须满足与未合并D同一相对阈值”的不合理要求；不是宣称旧门槛仍原封不动。科学效果/类别保护阈值完全不改。

每层捕获原FP32 D和原FP32 W_eff，随后仅在诊断中转FP64。D必须来自训练使用的函数，不能用重算的理想FP64 D替代它。可增加独立FP64参考计算帮助解释，但不拿参考计算替换生产路径。

**第一层：参数化约束。**

设N=||D||_F+1e-12。保留：

- LR_RAND：||DQ||_F/N ≤1e-5。
- LR_SRC_A：||DV||_F/N ≤1e-5。
- LR_SRC_AB：||DV||_F/N ≤1e-5，且||UᵀD||_F/N ≤1e-5。

使用原冻结参考基底，另记录运行时基底与参考基底的投影差异和正交性。LR_FREE、F_CONV、F_FULL没有对应强制约束，记N/A，不伪造0。

D精确为0时单独记录ZERO_UPDATE，绝不把没有学习解释为保护有效。检查是否非零用D，而不是只用舍入后Δ_add；后者可能为0却D不为0。

**第二层：独立的FP32加法误差预算。**

必须在运行恢复任务之前把预算公式与测试固定，不能观察失败值后继续提高常数。

可使用下述保守逐元素预算实现（全部在FP64诊断中计算）：

ε32=torch.finfo(float32).eps，ε64=torch.finfo(float64).eps，η32=torch.finfo(float32).tiny。

b_ij = ε32*(|W0_ij|+|D_ij|) + η32
       + 8*ε64*(|W_eff_ij|+|W0_ij|+|D_ij|)。

其中FP32 eps采用完整machine epsilon而非半eps，以保守覆盖一次有限值加法舍入；η32为可能的subnormal/flush边界提供极小绝对项，FP64项覆盖诊断中的加减数值误差。代码需检查dtype、finite、无overflow；不能将这份FP32预算用于FP16/BF16/TF32矩阵乘法误差。norm/reduction本身的FP64容差须按实现显式记录，不使用验证分数设定。

要求|E_add_ij|不超过独立b_ij及明确的FP64评估余量。记录β_add=||b||_F及||E_add||_F。可另用FP64精确和转回FP32/ULP对照验证预算，但不得用实际测得的E_add反过来定义允许预算。

**第三层：实际有效更新的约束误差包络。**

实际误差不是删除，而是以可解释预算判断：

||UᵀΔ_add||_F ≤ 1e-5*N + ||U||_2*β_add + β_eval64,L；
||Δ_add V||_F ≤ 1e-5*N + ||V||_2*β_add + β_eval64,R。

随机输入保护使用Q。基底可能只是近似正交，||U||_2/||V||_2不能不检查就强行写1；可以使用可证明的保守上界。

β_eval64涵盖Δ_add及矩阵乘积在FP64诊断中的可计算舍入界。例如由γ_n=n*u64/(1−n*u64)和绝对值矩阵乘积构造标准点积界，并包括提取Δ_add的加减误差和norm归约余量。若只采用保守FP64容差而非严格界，须说明它是数值工程包络而非形式化证明，并用独立参考和负例验证。不能因为第三层等式来自三角不等式，就省略独立的第二层预算与第一层约束。

原`right_leak/left_leak`相对Δ_add的数值全部保留为legacy监控字段，不再单独作为无舍入预算的fatal gate。若legacy>1e-5但三层均通过，记录`ROUNDOFF_LIMITED_REPRESENTATION`；不能写成严格数学零泄漏。

绝对值、相对更新范数、相对源权重范数、D/Δ_add范数、量化消失比例、basis误差、dtype/backend、逐项PASS/FAIL全部记录。严重真实错误仍必须停止。

## 4. 检查顺序与故障状态：先存档，再可能抛异常

原engine在epoch20/100末尾先调用diagnostics，再save。因此420步检查抛错时，latest只停在399。此顺序必须修复。

新阶段末流程：

1. 当前step的backward、optimizer、EMA完成，记录实际操作并明确该step的完成状态。
2. 保存原子化的POST_UPDATE_PENDING_DIAGNOSTICS checkpoint，必须包含当前position。
3. 对全部14层收集诊断，不在第一层失败时丢弃之前或后续记录。
4. 先写完整diagnostics.json与结构化异常证据，再根据结果抛异常。
5. 检查通过后，写入与checkpoint checksum绑定的diagnostics_pass状态。

checkpoint包括学生W0/A/B/buffers、EMA、Adam及参数组、本地LR位置、RNG、step/epoch/index、任务/源码/配置/source/basis身份、顺序摘要、操作计数、诊断pending状态。保存过程中不创建第三个完整模型对象。

failure JSON至少包括task_id、attempt_id、epoch/index/position、layer、constraint_side、parameterized与legacy残差、D/源权重范数、舍入预算与实测E、finite/源hash/basis检查、精确checkpoint路径与hash、最新日志位置、失败发生在optimizer/EMA/diagnostics中的哪一阶段。公开版去除私有绝对路径。

恢复PENDING checkpoint时，必须先运行未完成的诊断并确认通过，再执行下一更新。不能把PENDING当PASS，也不重复已保存的optimizer/EMA更新。final merge和部署封存前仍须通过原合并输出检查。

## 5. 必要资格：短、成组、无新评价GT

新增synthetic optimizer更新预算不超过120；新real-smoke更新预算0。优先复用原118/24资格证据，明确它们对应的旧源码；只对本次改动增加测试。不以未覆盖bug为由重跑整个旧资格矩阵。

必须覆盖：

- 小D/大W0，特别是bottleneck128×1152形状的量化放大场景；D=0、非常小的非零D、正常尺度D。改变的是合成张量尺度，不是正式训练超参。
- 原始FP32 D满足约束，legacy反算残差超线，但独立加法预算/实际包络通过。
- 漏掉右/左投影、污染基底、破坏源权重hash、注入投影方向分量、在effective中加入预算外扰动、NaN/Inf等负例必须失败。
- 对六臂短轨迹执行“旧训练核+不开诊断”与“相同训练核+新诊断”，逐步比较有效权重、梯度、Adam、EMA、loss和RNG/hash；在同一冻结后端上应保持原训练计算一致。不能因旧fatal条件在合成量化案例中触发，就要求旧诊断通过；比较的是训练路径。
- 新诊断调用前后student/EMA/Adam/RNG/train-eval mode/backend flags不变。
- 人为在epoch末诊断抛错时，精确post-update checkpoint和完整诊断先落盘；原子化写中断不覆盖最后一个完整checkpoint。
- PENDING恢复先重做检查，不重复optimizer/EMA；通过与不通过两条路径都覆盖。
- 已修的CUDA冷启动顺序保留；使用新进程验证相关存档/检查路径，不升级运行环境。

若资格揭示delta/forward/优化器确实需修改才能通过，不继续当前恢复计划混合新旧结果。记录`RECOVERY_NOT_DIAGNOSTIC_ONLY`，这属于需要单独版本的算法实现修复，而不是允许修改训练数学的通行证。不得通过删除AB或放宽真实参数化残差绕过。

## 6. 保存完成任务，失败任务重新开始，不追求不存在的420状态

### 6.1 18个完成任务

零训练复核其receipt、训练/评价退出、source/config/student/basis身份、frozen hash、未合并终态checkpoint及部署hash；用新诊断对可用完整checkpoint进行无GT、无模型前向权重核验。

复用原评价患者分数，不为修监控重读GT。原任务源SHA保持原值；新结果manifest另列`training_source_commit`和`reaudit_source_commit`。不可把旧receipt.source改为新SHA。

原20步/epoch20状态若未独立保留，不补造，也不因缺少历史监控快照全量重训：新终态核验覆盖终态，旧监控覆盖其已保存字段，准确披露覆盖范围即可。只要训练函数未变且既有任务有效，采用新监控不要求重训18个完成任务。

### 6.2 唯一失败任务

**选定策略：从同一SRC_CE和同一基底重新训练`O1_s62_LR_SRC_AB`整个2100步，使用新的attempt目录。**

原因：原420精确状态不存在；399恢复会涉及尾段分叉和额外证据解释。重新跑该任务只比从399恢复多399步，却有简单完整的源→目标轨迹。不能因为丢失21步状态而重跑全矩阵。

原失败420步保留为实际工程成本，永不放进最终科学矩阵中当独立样本。原399checkpoint和21步日志保留，不截断、不覆盖。

新attempt在399步额外保存完整状态，比较原399checkpoint的student/A/B/bases、EMA、Adam、数据位置与可比RNG/hash。若同一环境下匹配，记为前缀一致；若仅监控变化却导致训练状态不匹配，先调查，不自动归因于硬件或把它写成已复现。

新420步必须先保存精确状态，再同时输出legacy与新三层检查。该状态称“修复attempt的420步”，不能声称恢复了原丢失状态。只有新检查通过才继续本任务。

不启动单独的420步资格训练再从0重跑：399/420核验都在这次正式重跑的同一条轨迹上完成，避免重复成本。

### 6.3 调度余下17个任务

资格和18任务只读复核通过后，先派发失败任务的新attempt。在其420步新契约通过且证据封存后，恢复17个未启动任务的原队列；失败任务同时继续到2100步。这里等待的是工程检查，不是分数筛选。

保留六臂、两个顺序、seeds61/62/63、每目标100epoch、r8、k公式、alpha/r=1、14层、所有freeze/EMA/LR/LCTX设定。

不得根据已知seed61正面或seed62负面结果删任务、换seed、改变队列准入标准。后续真实错误仍保留精确状态并停止新派发；不无限自动重试。正常完成后将18旧成功+18新成功组成原36任务矩阵。

## 7. 预算：科学矩阵与物理计算分开

| 项目 | 已执行/新增更新 |
|---|---:|
| 已完成18任务，保留 |45500|
| 原失败attempt，保留但不进入最终矩阵 |420|
| 重跑失败任务 |2100 新增|
| 17个未启动任务 |47800 新增|
| 正常本次新增正式更新 |49900|
| 最终36成功任务科学矩阵 |95400|
| 含原失败成本的正式物理更新累计 |95820|

公式：45500+2100+47800=95400；原物理45920+新增49900=95820。

不要直接按95400−45920=49480当作恢复所需更新，因为旧失败420步不在新重跑轨迹里。不要把95820写成95400，或把重跑后的任务算作新seed。

这份恢复计划不新增源训练；已有源及基底作为不可变输入。新synthetic资格成本按实际记录在120上限内，旧118 synthetic与24 real-smoke仍分开列。若出现进一步故障，按真实操作记录，不宣称必定95820封顶，不擅自追加重试预算。

GPU4/5/6/7共享，每GPU最多一个本轮正式任务；发起前检查当前显存，不独占、不终止其他进程，不重复询问授权。

## 8. 完成后统一评价与裁定

在原36个目标任务全部有合格权重后，封存完整manifest，再执行原计划的6个STATIC_SOURCE目标域评价。它们不增加训练更新。未完成前不利用STATIC结果改变训练或阈值。

新的18任务沿用原固定epoch100单学生评价；旧18任务复用合格评价结果。保持patient-level rim/cup macro、物理域共享患者权重、2000次配对bootstrap、原分析seed2026091011、三种子先按两个顺序平均。完整矩阵完成前不对不平衡集合计算“总体六臂均值/显著性”。

同时呈现：

1. ENGINEERING：是否完成/数值误差包络是否通过/是否保留原训练函数。
2. VALUE：严格用V0.1原规则。
3. SOURCE_GEOMETRY：严格用V0.1原规则，分别对LR_FREE、LR_RAND。
4. SINGLE_VS_DUAL：完整A与AB差异，注明有效自由度不同，不宣称普遍定理。
5. 方法代价：F_CONV相对F_FULL，以及A对F_CONV，防止把冻结模块已有收益全归因于源子空间。
6. 类别风险：已观察的−7.73/−6.88/−7.96点反例必须出现在报告主文，补齐任务不能将它们平均消失。

若所有原完成任务有效，已有极值反例意味着原VALUE和SOURCE_GEOMETRY的全部条件无法同时满足。程序仍应等完整矩阵后按原规则生成正式判定，不能偷偷删除class_single或改变比较对象。可以报告正向均值与结构性价值，但不得伪造全门槛通过。

原INCOMPLETE_ENGINEERING报告保持不变；新报告写清“原attempt终态”与“修复后完整矩阵终态”，新工程成功不覆盖旧attempt失败。

## 9. 交付与停止

新增交付保持精简：

- PROTOCOL_AMENDMENT.md：诊断契约为何修订、失败任务重跑授权、科学门槛不变。
- NUMERICAL_CONTRACT.md：D/W_eff/E定义、浮点包络、逐层字段与负例。
- REPAIR_DIFF_AND_QUALIFICATION.json：允许改动、训练函数/配置/基底identity、非干扰与存档测试。
- RECOVERY_MANIFEST.json、RUN_LEDGER.csv：旧成功复用、失败attempt保留、新attempt与未启动任务映射。
- FAILURE_STATE_EVIDENCE.json：新399/420证据及原状态不能复原的边界。
- FINAL_REPORT.md及完整指标/配对/类别/成本表：原冻结判定，完整来源。

权重、RNG、患者输出和原始日志保留NAS；公开聚合和源码相对引用，敏感路径脱敏。不合并main，不触碰旧终态，不追加科学实验。

正常路径必须执行到原矩阵完成与STATIC_SOURCE结束，不只返回新诊断报告。若新契约揭示真实投影错误、非有限值、状态不一致或确实改变了训练计算，保留精确证据后停止该恢复；不以“不要再阻塞”为由继续无效比较。

## 10. 可直接提交给Codex的启动语句

请完整执行本文件LCTX_WM_NUMERICAL_RECOVERY_V0_1_1。只修诊断与存档，不改训练数学。

读取d614de8的部分结果、原正式源码50e094及相关冻结文件。必须先查看AVAILABLE_PAIRED_EFFECTS和AVAILABLE_CLASS_COSTS，承认主候选已有必要科学保护线反例，不以seed61均值宣称成功。

保留参数化投影残差1e-5；把FP32 W0+D后反算更新的残差改为有独立舍入预算的检查，原legacy数值完整保留。不能直接放宽1e-5，不能改FP64 forward，不能只算理想FP64 D代替生产D。

先保存post-update pending-diagnostics精确checkpoint和全部逐层诊断，再抛异常；恢复pending状态必须先重做检查。

18完成任务零训练复核并复用。O1_s62_LR_SRC_AB从同源同基底新attempt重跑2100步，旧420步完整保留为工程成本。新399步比对旧checkpoint，新420步保存精确状态并核验。通过后继续17个未启动任务，不删AB，不换seed，不改rank/k/LR或科学门槛。

正常新增49900正式更新；最终科学矩阵95400，正式物理累计95820；新资格单列。全部36权重封存后执行6个STATIC_SOURCE，完整发布结果和NAS归档后停止。GPU4/5/6/7共享授权沿用，不重新寻找旧KI。

## 11. 参考来源

项目证据均来自以下固定发布树，恢复时以精确文件哈希核对：

```text
https://github.com/DLwbm123/SSL_CL_seg/tree/d614de8aaeccc08a8448f0e6460e8a59ed9395c3/experiments/lcrseg/docs/lctx_weight_memory_v0_1
https://github.com/DLwbm123/SSL_CL_seg/blob/d614de8aaeccc08a8448f0e6460e8a59ed9395c3/experiments/lcrseg/lctx_weight_memory_v0_1/core.py
https://github.com/DLwbm123/SSL_CL_seg/blob/d614de8aaeccc08a8448f0e6460e8a59ed9395c3/experiments/lcrseg/lctx_weight_memory_v0_1/engine.py
https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html
https://docs.pytorch.org/docs/stable/notes/randomness.html
```

PyTorch文档用于解释有限精度与跨环境非位级复现边界，不是要求安装当前文档对应的新版本。本恢复必须保留原实际训练环境。本文浮点包络与恢复方案是提出的工程方案，尚未在用户NAS中的399/420状态上执行验证；不能把计划内容写成已完成测试。
