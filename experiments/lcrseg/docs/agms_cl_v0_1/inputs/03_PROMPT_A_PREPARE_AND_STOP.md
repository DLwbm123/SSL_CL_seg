# Prompt A：准备AGMS核心代码，提交后等待外部审阅

将本文件与01_EXPERIMENT_PLAN.md、02_IMPLEMENTATION_SPEC.md、05_PROPOSED_PLAN.json一起交给Codex。
转发本段仅授权本次代码准备及有限CPU生成测试；不是实际训练授权。

---

请在 DLwbm123/SSL_CL_seg 新开独立研究 AGMS_CL_V0_1，完成代码、有限CPU资格与审阅材料，然后停止等待外部代码审阅。不要运行真实实验。

## 1. 先读上下文与证据，保护既有研究

读取适用AGENTS/合同和附件01/02/05，核对这些锚点：
- 旧B2/NKA/DOSE实现：d557bd245eb204ec3ca377a5d868a03dc31436a5。
- 刚结束的DOSE缩减结果：12e8bee31df6723a333f702c8a011e19053b628d，results/nka_dose_v0_1_s163_o1/。
- B2原前缀及完整结果：3034b2199aa7d6e54ea67499f391a0dd4ea3d21e，results/f5_confirmation_v1_p1/。
- NKA旧D1对照证据：0da596f4050a3701f41ff74248ca9d4ba21222e2。

旧DOSE实际4/4、8400更新，余12节点用户取消，原跨seed/O2门槛NOT_ASSESSED。不要改写为16/16，也不要恢复取消节点或继续SWD调参。

使用新分支 codex/agms-cl-v0-1；保护工作区，不reset或覆盖用户改动。旧Python、科学清单、结果、审批和全部账本保持不变。新包与docs独立，必要公开输入复制到新docs/inputs并固定原提交与hash。不要把结果分支中的文档变化误当作原执行代码变更。

## 2. 本次只实现可审阅的P0/P1

新包建议：experiments/lcrseg/agms_cl_v0_1/。
实现完整但默认禁止生产的model/trainer/protocol/authority/state/execution/reporting/qualification；本次不要再次交付只有loss、生产适配留白的版本。

完整研究方向是“参数保护+解剖粒度+多尺度+可信空间”。当前P1仅实现前三者核心。SCNP、MATCH、距离场、risk预测网络、RL、风格生成、多层SWD均PLAN_ONLY，不生成执行节点。

## 3. 精确矩阵

仅seed163、两个顺序、第二目标阶段：
O1的当前域Drishti_GS，2100次；O2的当前域RIM_ONE_r3，3200次。
从两个对应的原B2_C06第一目标阶段模型独立初始化。

A0：原B2，两个终点只历史导入，无正式A0训练节点。
A1：H父类监督，不加辅助头。
A2：M普通辅助深监督，不加H。
A3：M+H直接相加，H仍只看主teacher。
A4：M+H，H用等权多尺度概率与分歧门控。
A5：M+H，H用当前L风险加权多尺度概率与同一分歧门控；唯一主候选。

A1-A5各两个stage2，精确10新节点/26500正式更新，source=0/第一目标=0。
历史A0只有seed163/O1与O2，两顺序均值约65.74659%，不能导入旧跨seed均值67.9011%。
不从NKA/DOSE终点或别的臂warmup/EMA/head/optimizer开始。P1R/P2/P3为空执行集合。

## 4. 方法按02完整实现，不自行调参

底座保持NATIVE_LR_SRC_A_3DOMAIN_V1/B2_C06：原14层A-only、A/B lr=.0005、KL=1、PAS=.6/.7、CE+Dice、warmup/ramp=.2、原Adam与数据/RNG/精度。

M：dec3/dec2各一个1x1三类别头（64/32通道、有bias、294参数），aux lr=.001，原调度；两个原CE+Dice损失均值，lambda_DS=.25。头仅L训练，当前teacher .99 EMA；不能进入U参数白名单。

捕获原监督前向的中间特征：warmup单L，active原两次LCTX互补视图。aux输出先对齐到输出网格，再用同一mask收集anchor；不能把混合图直接配未混合标签。teacher-L/U同样复用现有骨干前向，新增readout另记；原主头valid3x3几何完全不变。

H：原主teacher的PAS细集合mF不改变规则；d0=q_rim+q_cup。
A1/A3：mH=G & ~mF & (d0>=.9)。
A4/A5：在上述基础上要求 dbar>=.9 且 sum alpha*(d_s-dbar)^2<=.01。
A4 alpha恒1/3；A5 alpha=.1+.7*softmax(-R/.1)。

R是三尺度在已有当前teacher-L上、按背景/disc平衡Brier误差的阶段内EMA；beta=.9，初值.25，当前步只读上一已提交R。没有合法L时不更新。R/alpha/q/masks全部detach，不读取val或U标签，不宣称共形校准。失败step不提交R或诊断。

H仅对原fine集合外的可信disc父类，使用 -logsumexp(logp_U[:,1:3])。每图除以合法geometry像素数，再对有效图像平均；lambda_H=.5，使用原ramp。复用原UL收集后的主logp，不增加clean-U骨干前向。

总损失：原主监督+约束+.25*M*LDS+ramp*(原KL+.5*H*LH)。
不把父类概率包含关系构造为恒零loss，不把两个属性bit称为纠错码，不另加细伪标签或改变原B2接受规则。

A2-A5头初始化相同，arm不入初始化seed；新头用独立命名空间，不改变父初始化、LCTX/UL和数据RNG。不同训练轨迹实际PAS集合可变，不回放别臂mask。

阶段退出丢弃head/EMA/risk/optimizer/prototype，仅合并主权重；部署只用原主输出。不得用aux平均推理分数冒充单学生。

## 5. P0与数据权限

本次准备阶段：真实权重、真实患者数据、CUDA、smoke、模型重新评估全部禁止。

为以后获审阅和用户授权后的P0实现只读入口：两个B2前缀，各最多16张不同的当前域L，0更新；mask先仅由图像/预测/geometry生成，再由独立L计数器评价。入口只用confidence-only机会统计，不冒称有完整当前阶段PAS。固定采样，不按结果挑图。

训练器仅当前L和授权当前U图像/geometry。不能打开U标签后丢弃。旧域val只在阶段封存后隔离评价，绝不反馈训练、R、checkpoint选择或门槛。

沿用既有服务器路径/环境/NAS规范；不要求用户重复提供已有信息，不升级依赖，不暴露私有路径/患者信息。

## 6. 诊断、预算与资格

P1固定四诊断点/阶段：Drishti525/1050/1575/2100，RIM800/1600/2400/3200，共40点。
四额外VJP：主监督、KL、DS、H；缺失项graph-zero并标not_applicable，总160。只读，不替换真实梯度合并。不要复制Dose的Adam双候选系统；本轮重点是新监督覆盖与正确性。

记录fine/coarse/ignore，当前L影子选择的父/细精度与覆盖，风险/alpha/分歧，loss和分组梯度，前向/读出/VJP/显存/时间。R和统计逐成功步提交，失败保留物理成本不提交成功证据。

新CPU预算<=96optimizer/最多3attempt/每次预声明<=32。建议完整套件28calls，可调整固定分配但必须先记录再执行。全部失败和oracle调用算成本，旧134次保留且分列，不运行旧suite。

未来CUDA计划36：A0-A5各warmup1+连续2+恢复2=30；原B2/A0额外各2步关闭等价=4；两个预设failure=2。
未来smoke24：六臂各4次当前L-only正式warmup，全部丢弃，不读U/val/test。
未来P1正式26500，真实总26524。当前不得执行任何这些未来环节。

至少测试：
- 完整canonical与重算hash的越界拒绝，不仅查总步数；每节点domain/updates/heads/风险/前缀准确。
- 原B2关闭等价、初始RNG/读取/EMA/optimizer不变；历史复用不成立则BASELINE_REUSE_BLOCKED，不补跑基线。
- .04/.48/.48进入父类、非细类；fine/coarse互斥；ignore/geometry/空类；alpha归一与下界；风险滞后与原子提交。
- aux LCTX源收集和坐标；头真实更新/EMA；U仅到AB；主头保持冻结；A-only残差。
- 原生新trainer连续/恢复/阶段exit；错臂/前缀/头schema/风险规则拒绝；恢复不能初始化新head/R。
- 部署丢弃aux后主输出等价；实际文件完整性/封存skip；故障、丢尾段、账本、输出完整性。
- 12/36/27/40报告schema；缩减范围门槛NOT_ASSESSED；不补造历史缺失诊断。

现有StageTrainer不会自动把新head纳入其groups/EMA；写清楚新模型和updateadapter，不能只挂nn.Module。若共享原语不兼容，报告明确边界，不能静默修改旧方法或绕过类型/权限检查。

## 7. 新权限与审阅

保留新study capability；真实操作必须要求真实外部APPROVED+独立用户launch，并绑定最终commit、传递代码树、科学/执行计划、前缀/导入/环境。

不要生成批准模板来通过检查，不将本prompt视为训练授权；旧NKA/DOSE的批准无效。approval/launch放reviewed checkout之外。审阅后不得通过更新README状态让HEAD漂移。

当前只执行CPU；真实前缀tensor gate/校准/CUDA/smoke/P0/P1全部PENDING。

## 8. 固定分析，别再开展候选择优循环

P_perf：A5-A0两顺序均值Final>=.003；每order Final>=-.002；O2 Old>=0；每order Incoming>=-.005。
P_joint：A5两顺序平均Final，比A1/A2/A3/A4中该均值最高的简单对照至少高.001。逐项保留A5-A1/A2/A3/A4，不能按order拼接最优对照。
这是事先设计的开发推进阈值，不是TMI录用/显著性标准。

全部10新节点完成后统一判断。任一通过只可建议下一轮审阅，不自动复验seed164/加SCNP/跑完整轨迹。未通过不改阈值、不挑另一个arm改名为主候选。用户缩减范围则独立记录，不伪报完整门槛。

## 9. 一次性交付与停止

代码及：
REVIEW_REQUEST.md、METHOD_SPEC、PLAN.json、EXECUTION_PLAN.json、PREFIX_BINDINGS、IMPORT_BINDINGS、FROZEN_OPTIONS、GEOMETRY_CONTRACT、RISK_STATE_CONTRACT、TEST_REPORT/所有尝试与账本、COST_SUMMARY、CODE_MANIFEST、实际可用CLI的RUNBOOK。

报告导出实际实现：P0_REPORT、FINAL_REPORT、FINAL_METRICS(12)、DOMAIN_METRICS(36)、PAIRED_COMPARISONS(27)、GRADIENT_DIAGNOSTICS(40)、覆盖/风险、COST_AND_COMPLETION、PUBLIC_RESULTS及完成证明。

旧baseline缺失的新指标写NOT_MEASURED_HISTORICAL，不从旧主loss或最终Dice推算边界/风险。公开仅聚合与必要复现信息，不传模型、逐患者信息、标签、private_val、凭据和私有路径。

推送后给完整SHA、变更文件、矩阵、测试与累计调用、真实数据optimizer更新0、仍待执行项。停止于：
STOP_AWAITING_EXTERNAL_CODE_REVIEW

不要执行Prompt B；不得启动P0真实读取、CUDA、smoke、正式训练或监测。等外部审阅通过，再按用户单独转发的运行prompt启动。
