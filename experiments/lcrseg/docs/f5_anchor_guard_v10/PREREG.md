# F5_ANCHOR_GUARD_V10 — 冻结阶段入口教师KL保护模块

## 授权、证据、正向底座
用户明确授权在正向方法上自主开展有依据的最小方法模块直到原双门槛，优先模块而非单纯改系数。V9完整正负结果公开验证8edbcc8f33df79836109478dc6ae8eeab8ddcddb后才注册此轮。V9 Final66.5536%，相对F5+1.1427pp，两seed都正并通过F5完整门槛；相对历史B2−1.3475pp，seed164 Old−5.9283pp、order2 Drishti cup最差−12.0699pp。该seed/order已见Drishti cup从阶段1的0.75538下降到阶段2的0.48957，−26.5814pp。A/B/R U16960次全部非零，模块激活真实。剩余旧域问题值得检验稳定性模块，但未证明EMA伪标签错误/冲突是退化因果，不把R方向当A/B方向。

## 单一模块、数据和生命周期
保持正向V9：原F5线性R、joint KL->所有可训练A/B/R、clean SWD->R-only，lambdaU0.25/SWD0.2/PAS0.7,0.5/Qk8D16/硬输入投影/原型/Adam/lr/日程不变。新增冻结阶段入口教师anchor：复制此轨迹该阶段入口student的有效dense parent与own F_prev，不使用本阶段更新R；eval、requires_grad=false、不进优化器/EMA更新，当前EMA与student共享同一immutable anchor。只有两个教师（当前EMA与入口anchor），没有第三教师、replay、历史图像或旧标签。每阶段重建本轨迹入口，不跨轨迹借用权重；第一阶段来自同源checkpoint+identity，第二阶段来自自己封存学生+own prefix。

对当前weak U（相同去重和geometry）每活跃步额外一次入口教师no_grad完整前向。KL保留掩码 m_KL=m_PAS & ~(max(q_anchor)>=原PAS_confidence0.7 & argmax(q_anchor)!=argmax(q_EMA))。仍用当前EMA概率作KL目标；拒绝而不强迫旧标签，低置信anchor/类别一致/原PAS无效边界分别按公式处理。原SWD继续用原PAS m，监督与prototype更新不变。仅KL引入此mask，无新loss/系数/旋转/梯度投影/范数cap/搜索/RL。入口教师在新域可能自信错误，拒绝会损害适应；old retention不获数学保证。

single StageTrainer增加一个默认identity kl_admission hook，将过滤仅用于KL，原损失/训练循环不复制。新实际模块/flag/threshold0.7/anchor源码、来源、入口tensor fingerprint绑定loss_contract；anchor作为冻结student/EMA state保存，真实恢复包含其参数，错误flag拒绝。teacher deepcopy后共享immutable anchor；部署仍只保留dense parent和固定D×D transform，丢弃anchor，无新增推理参数。

Learning without Forgetting（ECCV2016，https://arxiv.org/abs/1606.09282）提供“只用新任务数据保留旧能力”的背景，本轮是入口agreement拒绝改编，不声称复现论文蒸馏、2026论文、创新性/SOTA。只用当前U；不读取旧训练数据、封存test或新权限数据。

## 有限矩阵、资格与成本
两seed163/164×order1/2，4轨迹8阶段21200正式物理优化调用，source0。首S163/O1工程诊断两目标封存/EXIT0后其余3；不按患者效果剪枝。全8封存/4EXIT0后原20 seen-domain评价。必要对照复用V9，disabled guard两步native完整state/model/EMA/optimizer/cursor/prototype parity对冻结V9证明复用；原F5与历史B0/B2沿用冻结匿名结果，额外报告paired V10−V9。

独立create-only NAS协议先冻结PLAN/用户授权/code commit/config/此文。资格上限96CPU/32生成CUDA/4真实L，预计32/10/2：CPU两案例因子拟合32次；mask单元覆盖高置信一致、冲突、低置信、PAS无效、shape/nonfinite（0优化）；disabled guard和冻结V9两步完整parity4生成调用；enabled2步A/B/R实际U与merged L+U、entry anchor冻结/共享/不变、actual native入口预测与人工冲突目标触发拒绝/零KL梯度；保存checkpoint后2个继续调用、telemetry/anchor一致、错误实际flag拒绝；部署与own第二阶段1调用；故意after_optimizer失败1调用计费；真实当前L smoke2调用且zero U read。继承clean SWD路径parent阻断验证，共资格额外2生成VJP与2直接入口teacher前向，没有额外患者探针。不重置失败成本；所有相关资格PASS且绑commit才正式启动。

正式R固定诊断VJP上限64；每活跃U步1新增anchor forward，预计16960次，不新增正式VJP/optimizer调用。遥测anchor_full_forwards/anchor_candidate_pixels/anchor_rejected_pixels/anchor_gate_calls/anchor_rejection_calls和A/B/R非零计数用于激活检查，不从过滤频率推断性能/正确性。基础和anchor全前向成本据实际封存operation ledger记录。此前实际正式255073，本轮使总授权上限276273，actual=255073+本轮已有物理调用；此前资格672CPU/151生成CUDA/32真实L，预计704/161/34，上限768/183/36。源0，历轮负结果、故意失败和V5重启673重复/丢失保留；复制5300不双计、5缺失诊断NA不补算。正式失败不自动重试/扩预算，不以工程中断伪装下一科学轮。

## 成功与交付边界
原双门槛不变，同时相对匹配F5与历史B2：每seed平均orders后Final增益>0、总均值>=0.005；每order Old/Incoming>=−0.005；每seed/order/domain/rim或cup>=−0.05。报告B0、V9和全部负细胞/负轮。原F5复用依据V9已验证parity及本轮disabled V9完整parity；历史B2明确来源，协议不可比才补匹配对照。不得事后放宽/挑checkpoint/扩seed/data/PAS或系数、不用sealed test调参。反复开发患者不是独立泛化/SOTA/临床获益。

NAS实际NFS/容量/write-read与GPU4–7>=4GiB准入，现有storage wrapper及neutral stdin/env argv，默认匹配580.178.04库。不修改系统、冻结HDF5/清单/划分、旧配置/权重或健康无关进程。所有新大文件/缓存/权重/日志/tmp NAS。GitHub经本机7897代理，HTTPS显式proxy、防NO_PROXY绕过，不直连/强推，验证SHA和匿名公开访问。患者/逐患者结果/private config/原始日志/权重/密钥不入Git。完成先交付完整源代码/协议/匿名结果/激活/负发现/累计成本并验证，再依据已有授权选择下轮。小时f5 ACTIVE、RL PAUSED；仅原双门槛通过且交付验证或明确停止才暂停。
