# F5_ENTRY_GRADIENT_V12 — 合并梯度的入口知识半空间约束

用户已明确授权未达标时自主方法模块改进、以V10正向底座推进且持续小时监测。V11完整正负结果已公开验证f864bdf0a3feee6b534644a9fa7ed0153c1feb19，原B2门槛失败后注册本create-only有限轮次。V11 Final67.5628%（V10+0.6075pp），vsB2−0.3383pp；164 Old平均改善，但Drishti cup最终47.9215%（V10−5.0065pp），跨阶段下降27.7202pp，且RIM rim下降18.6459pp。不能仅按均值宣称成功或教师错误因果已证实。

## 一个模块，正向底座与边界
选择回到原V10拒绝底座，而不组合V11强制软目标恢复。F5线性R、joint KL->A/B/R、clean SWD->R-only，原lambdaU0.25/SWD0.2/PAS0.7,0.5/Qk8D16/硬输入投影/Adam/lr/日程/原型/数据均保留。自己的阶段入口dense parent+own F_prev作为共享immutable冻结anchor，与当前EMA共两个teacher；source+identity首阶段、own封存学生/线性prefix第二阶段；不使用第三教师/旧训练图像/旧标签/replay/RL，不增加推理参数。

每活跃U步复用原V10当前weak U入口anchor前向和已经计算的collected UL student logits。主KL仍是V10 PAS高置信入口/EMA类别冲突拒绝，原kept支持均值，原目标EMA；不把拒绝像素恢复为训练软目标。

额外参考标量 r=KL(q_entry || p_student_UL)，支持 m_ref=m_PAS & (max(q_entry)>=原PAS0.7)，同一个UL空间位置、当前U，entry概率和mask stop-gradient。它仅生成参考梯度h，不加入优化目标，无新lambda/温度/阈值。collected UL logits包含当前L donor和特征收集，这是与clean weak教师一致性同样的已有视图差异，不声称clean旧域损失。warmup未读取U、不建参考约束。

先得到原监督/约束L与原U的合并梯度g（所有可训练A/B/R拼接），h=grad(r)。若双精度dot(g,h)<0且||h||²>0，g_star=g−dot(g,h)/||h||²*h，否则保留g；数值零参考no-op。投影以double计算后转回参数dtype，无新学习率/系数。只改p.grad，原型提交、硬约束、Adam动量/权重衰减/EMA仍原样。h覆盖全部实际可训练参数，不按R稀疏方向代替A/B。参考teacher永不接收梯度。此半空间只是一阶原始梯度几何，不是Adam实际更新约束、有限步损失上界或旧域保持保证，current U可能不覆盖旧域且entry可能自信错误。不能把dot_after数值误差当成精确全步定理。

借鉴A-GEM Eq10/11（https://arxiv.org/html/1812.00420v2）：论文参考梯度来自历史episodic memory；本轮来自当前U的冻结入口知识，没有历史memory，因此是投影模块改编而非论文完整复现或新2026论文。默认单StageTrainer未改新训练循环；V12子类通过kl_objective/gradient_audit/transform_gradients接入。实际entry_gradient/参考公式/源码及入口来源fingerprint绑定loss_contract。参考loss/grad是单步临时数据，不写checkpoint；恢复从原cursor重新计算，原anchor/EMA/optimizer/telemetry保存，错误实际flag必须拒绝。

## 资格、正式矩阵和真实成本
任何优化前冻结PLAN/USER_AUTHORIZATION/CODE_COMMIT/private CONFIG/此文。资格上限96CPU/32生成CUDA/4真实当前L，预计32/10/2，含故意after_optimizer失败优化调用永久保留，不重置额度。CPU两case32步冲突保持入口维度同时学习正交维度；对齐/冲突/正交/零/shape/nonfinite/overflow/有限导数测试。native disabled对冻结V10完整model/EMA/optimizer/scheduler/cursor/prototypes及旧loss路径parity；启用主KL仍拒绝/同像素参考非零，强制实际native A/B/R合并梯度负向投影、教师冻结共享与线性R EMA、clean SWD R-only、真实checkpoint续训/telemetry与错误实际flag拒绝、部署/own stage2前缀、zeroU当前L smoke。预计5资格reference VJP、3额外生成VJP、1额外生成loss调用、2直接入口前向，无额外患者探针，实际成本据台账。

正式seed163/164×order1/2共4轨迹8目标阶段21200优化/source0。首S163/O1两阶段工程成功/EXIT0封存后其余3，不按效果剪枝；全8封存/4EXIT0后原20已见域统一评价。固定R诊断上限64VJP；每活跃U新增1 reference VJP共16960上限，分列entry_reference_vjps，不混入64诊断，不新增正式完整forward。entry_guard_calls/projected_calls/reference_pixels、dot_before/after/reference_norm按已有梯度统计，不能据激活推断临床/旧域保持。GPU4–7各>=4GiB显存余量。正式失败不自动重试、扩预算、改变运行中配置或加seed/数据/模块/系数搜索，工程中断不能冒充科学新轮。

之前正式实际297473；本轮使累计上限318673，实时actual=297473+本轮物理调用。资格此前736CPU/171生成CUDA/36真实L，预计768/181/38，上限832/203/40。所有历史负结果/资格失败、中断673回滚重复丢失及5缺失诊断NA永久保留，复制封存5300不双计、source0。相比原V10增加16960训练VJP而不改变optimizer调用预算，结果需要同时报告计算代价。

## 对照、成功与交付
V10为匹配底座，disabled native parity支撑复用，完整paired V12−V10，并报告V12−V11背景比较；原F5和历史B0/B2复用并明确来源，协议不可比补必要匹配对照。成功同时vs匹配F5与历史B2：每seed平均两order Final增益>0、总均值>=0.005；每order Old/Incoming>=−0.005；每seed/order/domain/rim或cup>=−0.05。报告全部负细胞/历轮失败，不能事后放宽门槛或选择checkpoint/用sealed test调参；重复开发患者不是独立泛化/SOTA/临床获益。

完成先公开完整源代码/冻结协议/匿名结果/激活/负发现/跨轮实际成本和报告，通过本机7897代理推送并验证SHA和匿名访问后才独立下一科学轮。患者/逐患者指标/权重/private config/原始日志/密钥NAS，不进Git。原NAS wrapper、实际NFS/容量/write-read及GPU准入，neutral父子stdin/env argv，匹配580.178.04默认库，不prepend旧580.173.02，不改冻结HDF5/清单/splits/系统/终止健康无关进程。小时f5 ACTIVE、RL PAUSED，仅原双门槛达标且完整公开交付验证后暂停，或遵从明确停止。
