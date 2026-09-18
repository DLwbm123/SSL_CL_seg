# Prompt A — 现在交给 Codex：完整代码准备，不启动生产

请为 DLwbm123/SSL_CL_seg 准备新的有限研究 NKA_DOSE_V0_1。
我转发本段仅授权：读取既有代码/聚合元数据、实现新协议与代码、按本段新预算执行CPU生成测试、独立分支提交和推送。没有授权真实前缀张量读取、CUDA、真实smoke、正式训练、再评估或监测。

请同时读取附件 01_EXPERIMENT_PLAN.md 和 04_PROPOSED_MATRIX.json。后者是提案展开，不是已审核的运行清单；实际hash与证据必须由你从固定来源核验。

一、固定锚点与新分支

代码基准：145b3c1b3ab301031ea379c9d67452e57b28ae1d。
NKA结果：0da596f4050a3701f41ff74248ca9d4ba21222e2。
B2共同前缀的公开来源：3034b2199aa7d6e54ea67499f391a0dd4ea3d21e。
新分支建议 codex/nka-dose-v0-1；新增 experiments/lcrseg/nka_dose_v0_1/ 及对应docs。
先读AGENTS/合同、原NKA的alignment/trainer/authority/protocol/state/execution/qualification/reporting，以及结果中的环境/完整性/前缀证据。
保护用户工作区。不修改旧NKA/F5代码、科学清单、账本、结果和审批，不reset旧结果，不恢复旧DAG。

二、唯一正式矩阵

(C2随机8维、C3原生V8维) × lambda_align(0.5,2.0) × seed(163,164) × order(O1,O2)。
每个节点都是stage2；O1从指定B2已学RIM前缀训练Drishti2100步；O2从指定B2已学Drishti前缀训练RIM3200步。
精确16节点、42400正式科学/物理更新，source新训练0、stage1新训练0。

只导入旧C0四条与旧C2/C3各四条0.05结果，共12条historical_import。不重训这些节点，不训练高剂量C1，不把历史导入标成新执行。
新剂量必须从原B2 stage1前缀开始，不从NKA终点、其他剂量/臂的warmup或优化状态开始。
实际复用绑定同一软件/数值环境、基础B2代码、manifest/split、前缀和随机流。不能证明时BASELINE_REUSE_BLOCKED，不静默增加C0训练。

三、科学配置只改lambda_align

保持B2_C06完整options：KL=1、PAS=.6/.7、A/B初始lr=.0005、Adam而非AdamW、wd=4e-5、原调度/EMA/FP32/batch2/384几何及.2 warmup/.2 ramp。
L=L_sup+L_parent_constraint+ramp*(KL+lambda_align*L_align)。
L_align含原D/8归约；C2/C3均D=8。不要与遗留lambda_SWD字段混淆，不重复乘系数。
原生A-only投影、U对原A/B的权限、所选层、归一化、每类8–64样本、32方向、边界mask、类别等权全部不变。
C2只改辅助基，不改父保护V；C3引用已有V，无额外SVD。
不新增网络模块、历史记忆、在线梯度平衡/手术、动态权重、层数/秩/PAS搜索。

新study/dose/run身份与随机流分离。保留原NATIVE_KEY_ALIGNMENT_V0_1辅助随机namespace；剂量不进入种子。相同状态下C2各剂量随机基/数据/抽样一致。训练分叉后PAS/实际样本不同允许，不回放别的臂的teacher/mask。

四、最小代码适配与语义门控

旧KeyAlignmentTrainer正系数固定0.05并有exact-type检查，不能只改YAML，也不能松开旧研究批准。新增窄DoseTrainer/独立capability和恢复身份，尽量复用原数学核及公共一次optimizer更新路径；不改旧共享实现，不做生产权限monkeypatch。

D0本次要一次性交付有限production路径和资格测试定义，而不是仅写loss后再留生产adapter缺失。但没有新外部批准+真实用户启动确认时，所有真实/CUDA入口都拒绝。
canonical校验须完整重建并比较每个node/dose/domain/updates/options/prefix/import/环境/分析规则，不能只看总步数或互相一致的hash。修改后重新hash的12类负例应被拒绝。
同时拒绝把0.05、C0、C1、其他seed或完整stage1轨迹塞入新正式矩阵；0/.05只允许生成资格等价测试。

五、只读Adam局部对照，实际训练保持原求导顺序

在更新前的successful step编号525/1050/1575/2100（Drishti）和800/1600/2400/3200（RIM），同已有计算图额外求4组VJP：
gS=grad(L_sup+constraint)，gK=grad(原kl*lambda_U*ramp标量)，gA=grad(原ramp*lambda*L_align标量)，gU=grad(原完整unlabeled对象)。不得以数学等价的括号重建改变原浮点乘加顺序。
按原split_gradients/None语义组装g0=merge(gS,gK)及g_lambda=merge(gS,gU)。第四个VJP保证正式U标量的求导/归并顺序，不用分项和替代实际训练梯度。
retain_graph、detach仅用于诊断；不消费共同RNG、不新增前向、不更新.grad/m/v。
保留原引擎实际L/U求导和唯一optimizer.step。正式额外诊断VJP=256；64点各计算有/无辅助两个候选，共128次只读候选，不计为optimizer更新，算子/时间另计。

纯数值Adam公式从相同更新前参数、参数组、m/v和每参数step出发，匹配实际PyTorch2.2.1 Adam、coupled wd、bias correction、eps、foreach/fused等实际路径。
不得调用第二个真实optimizer.step或in-place functional Adam在正式模型/状态上做候选；不得为公式方便改正式optimizer参数。
grad=None不等于zero；未初始化状态/不同参数step/小lr均覆盖。只支持实际合同所需配置，其他配置明确报错。
报告rho_theta及rho_W、分子分母、全部参数/固定辅助可达上游范围、A/B/每层，以及真实步长。
有效W按完整B@(A-(A@V)@V.T)算候选，保留二阶交叉项。分母数值退化写null和原因，不凭epsilon制造大比例。
预测有辅助候选与随后唯一真实更新核对误差。先在生成参数oracle核验并固定容差，真实阶段不得放宽。
这只是当前剂量状态上的一步反事实，不是完整无辅助轨迹；不假造旧0.05/C0的Adam诊断。

六、资格、成本和恢复

新CPU账本：累计<=96次optimizer调用，最多3次尝试，每次预声明<=32次。已有旧64次不清空、不抵扣。新CPU oracle/连续恢复/预设失败全部计费。零更新元数据测试不伪计为训练。
至少覆盖：canonical拒绝、历史/环境绑定、旧/新0.05等价、计量开关不扰动、C2/C3新剂量可达梯度、原生恢复保留dose/basis/诊断、错dose/prefix拒绝、Adam oracle、完整有效W计算、报告schema、失败/尾段/封存保护。
CPU允许生成模型/参数及已绑定上游源代码，不加载真实模型、患者payload或签发生产批准。

未来CUDA定义必须完整实现但此轮不执行，固定40次：四个新cell各warmup1+连续2+恢复2+失败1=24；C2/C3旧0.05两步、新0.05计量开两步/关两步=12；4次生成小参数Adam oracle。所有对照副本都是生成模型，不伪造旧生产许可。
未来smoke32次：新四cell各8次，seed163/O1合法前缀、当前L-only、正式warmup语义，状态丢弃。
formal42400；真实总量42432；各预算独立。未分配的额度不是自动重试许可。

checkpoint绑定study/arm/dose/lambda/完整options/前缀/V/basis/RNG/游标/优化状态/支持/固定诊断/计量版本。恢复须为DoseTrainer，C2不得重新抽基，不允许跨剂量恢复。失败尝试计账，丢失tail或未闭合成本停止，不修改封存或借预算。
实际四个前缀及16个最终模型的验收路径复用已有可靠原语，但全程使用新study批准，不放宽旧前驱规则。训练只读当前L及U图像/geometry；旧域val仅封存隔离评价，不读U真值/test/future，不据评价早停。

七、冻结统计和决策

最终表28行：16新+12历史；三域84行。
主配对按三剂量(.05/.5/2.0)×三个contrast(C3-C0,C2-C0,C3-C2)×(4配对+2seed均值+1总均值)=63行。
主要新开发判定只允许lambda=.5或2.0；.05是历史描述，不冒充新候选/新确认。
G_perf：C3-C0平均Final>=.005、两个seed平均均>0、O2平均Old>=0、两顺序Incoming各>=-.005。
G_key：同lambda C3-C2平均Final>=.001（新开发差异门槛，不是显著性），且两个seed均>0。
只有同一lambda两门槛都过才建议另行完整轨迹审阅；两lambda均过选较小0.5。禁止按seed/order拼系数。只过G_perf不支持键特有主张。
两剂量都未过性能门槛则结束本剂量研究及当前NKA-SWD候选，不加更大权重/新层/改秩搜索。无论结局均不自动进入D2/D3。
全16节点后统一决定，无性能剪枝；DeltaForget=-DeltaOld不是独立证据。

八、交付并停止

提交完整代码、PLAN/EXECUTION_PLAN、PREFIX_BINDINGS/IMPORT_BINDINGS/环境绑定、基线与Adam公式资格证据、全部测试/失败成本、CODE_MANIFEST、真实可用CLI RUNBOOK、REVIEW_REQUEST。
生产最终导出路径要实际实现：FINAL_REPORT、28行FINAL_METRICS、84行DOMAIN_METRICS、63行PAIRED、GRADIENT_DIAGNOSTICS、64点ADAM_COUNTERFACTUAL、COST_AND_COMPLETION/PUBLIC_RESULTS/完成证明。
公开结果只含汇总，不含模型、患者/逐样本/private_val、凭据/私有路径。
新实验source/CUDA/smoke/formal全部标PENDING；不把CPU通过写成生产通过。

完成后push独立分支，报告完整SHA、改动边界、矩阵/费用、测试及累计调用、待生产项，停止在STOP_AWAITING_EXTERNAL_CODE_REVIEW。
不要执行附件Prompt B，不自行生成APPROVED或用户launch，不启动CUDA、真实前缀加载/smoke/D1或监测。
