# F5_ANCHOR_RESTORE_V11 — 冻结入口知识恢复路由

用户2026-10-11明确授权继续分析164旧域退化、以V10为底座直接开展最有依据的方法模块下一轮，不只调系数、小时监测直到超过B2且达到原门槛。V10完整正负交付和公开验证6f0ec91df3434b9e332471e0f309d8fe1083059f已完成。依据见ANALYSIS.md/DIAGNOSIS.json，因果尚未确认。

## 单一主改动与数据权限
V10原正向底座不变：F5线性R、联合KL->全部可训练A/B/R、clean SWD->R-only，lambdaU0.25/SWD0.2/PAS置信0.7和cos0.5/Qk8D16/硬输入投影/Adam/lr/日程/原型、教师和当前域数据边界原样。自己每阶段入口dense parent+own F_prev冻结teacher，与EMA共享immutable对象，不使用本阶段R/其他轨迹权重。首阶段source+identity，第二阶段本轨迹封存学生和own线性prefix。两个教师，无第三教师、replay、旧训练图像或旧标签；anchor部署丢弃，无新增推理参数。每活跃U继续仅1个原anchor no_grad forward，无额外正式前向/VJP。

唯一模块改变被V10拒绝的KL像素：r=m_PAS & (max(q_entry)>=原0.7) & (argmax(q_entry)!=argmax(q_EMA))；q_star=where(r,q_entry,q_EMA)，单一masked KL以原m_PAS支持求均值。两种目标和r均stop-gradient，监督L、SWD、原型/PAS几何不变。V10原KL按kept支持归一化，而本轮按原PAS支持归一化；因此模块也改变有效支持和保留EMA像素相对权重，必须披露，不能归因为标签本身或假装系数完全等效。不加新loss/temperature/系数/阈值搜索、Cayley、V7拒绝、V2投影、B cap/RL。未满足原PAS的高置信anchor不能额外纳入训练。

复用single StageTrainer，新增默认kl_objective调用原fine_kl+kl_admission，新子类只改KL目标路由；V10及其他默认路径数学行为不变。实际anchor_restore启用标记/公式/源码、入口来源/fingerprint绑定loss_contract；teacher参数不更新，checkpoint存原anchor和恢复原游标、EMA/optimizer/telemetry；错误实际flag恢复必须拒绝。

## 有限资格、矩阵、预算
独立create-only NAS目录，在任何优化调用前冻结PLAN/USER_AUTHORIZATION/CODE_COMMIT/private CONFIG和此协议。资格上限96CPU/32生成CUDA/4真实当前L；预计32/10/2，所有故意失败/实际错误永久计费不得重置。CPU32步实际两case软目标路由拟合；单元边界同类/高置信冲突/低置信/无效PAS/shape/nonfinite、目标detach。native disabled对冻结V10完整model/EMA/optimizer/scheduler/cursor/prototypes parity（原训练loss路径）；启用实际入口冲突软目标导数、同样像素原V10零导数、A/B/R真实U与L+U合并、SWD parent detach、冻结共享teacher与线性R EMA、真实checkpoint恢复/telemetry及错误flag拒绝、部署/own stage2 prefix、故意after_optimizer失败、真实当前L smoke zero U reads。预计额外生成VJP4（恢复KL、其参考导数、原零导数、SWD实际路由合计4）及直接anchor forward2，实际记录为准，不做额外患者探针。

正式seed163/164×order1/2共4轨迹8目标阶段21200物理优化/source0；先S163/O1两阶段工程成功/EXIT0封存后其他3条，不按患者效果剪枝；全8封存/4EXIT0后20已见域统一评价。正式固定R诊断VJP上限64，不新增正式VJP。anchor_restored_pixels/calls、candidate/rejected、A/B/R U利用已有计算记录，不能据激活推断恢复正确。正式失败不自动重试/扩预算、不改运行中配置/数据/seed或组合模块。

此前正式实际276273；本轮授权累计上限297473，实时actual=276273+本轮已调用。此前资格704CPU/161生成CUDA/34真实L，预计736/171/36，上限800/193/38；失败/中断和V5回滚673丢失/重复永久保留，复制封存5300不双计，缺失5诊断NA不补算。

## 对照、门槛与交付
disabled恢复冻结V10 native parity支撑复用V10匿名汇总，完整paired V11−V10；匹配F5、历史B0/B2复用并明确来源，协议不再可比补必要匹配对照。原双门槛同时相对F5及历史B2：每seed平均两order Final增益>0，总均值>=0.005；每order mean Old/Incoming>=−0.005；每seed/order/domain/rim或cup>=−0.05。报告B0/V10和全部负细胞/历史负轮，不能事后放宽门槛或选择checkpoint，以sealed test调参。反复开发患者不是独立泛化/SOTA/临床获益。入口知识恢复可能损害新域，也可能覆盖不到旧域，不能保证成功。

NAS NFS/容量/write-read/GPU4–7>=4GiB准入，原storage wrapper+neutral stdin/env父子argv，匹配580.178.04默认库，无旧580.173.02 prepend。所有大文件/缓存/权重/日志/tmp NAS，不改冻结数据/划分/系统或终止健康无关进程。GitHub经本机7897代理，显式Git HTTPS/HTTP/gh代理防NO_PROXY绕过、不直连/强推；患者/逐患者指标/权重/private config/原始log/密钥不进Git。完成先提交推送完整源代码/冻结协议/匿名正负结果/诊断/累计成本并验证SHA和匿名访问，再按既有授权独立下一轮。小时f5 ACTIVE、RL PAUSED，仅原双门槛达标且交付验证后暂停，或明确用户停止。
