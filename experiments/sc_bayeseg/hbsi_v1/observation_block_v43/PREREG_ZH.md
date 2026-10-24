# v43：观测评分与排序目标块移除干预（运行前冻结）

依据[第二份分析审阅决定](../reports_20261024/SECOND_REVIEW_DECISION_ZH.md)，本轮只回答：同一源域继续训练路径中，观测图像评分和ranking anchor这个联合目标块是否导致可逆的迁移性能代价。不是新U收益实验，不同时运行原双冻结候选，不认定既有根因。

单训练seed20261023，T从封存v23 L1600活动入口继承全部q/phi/RUNMC域状态、原先验、Adam和RNG，新增400更新到固定2000。对照A复用v25 A2000封存指标，不重训或重评分。原RUNMC训练4L患者80切片、batch1、data_seed163000；为匹配入口evidence保留原LU数据接口（14U/271切片），run_stage仍读U图像但目标不使用U，不能把输入计数称U训练。BMC仅用RUNMC记忆作未适应评价，禁止test/U隐藏GT和旧域训练图像。

令l=log p(c,R)−log q(c,R)，H×W=16384，s为原collapsed Gaussian观测log-score。原目标为 E[CE+Dice−.01(s+l)/HW+anchor]+(KLglobal+KLdomain)/(N_L HW)。新T为 E[CE+Dice−.01l/HW]+(KLglobal+KLdomain)/(N_L HW)。保留原2global×2local嵌套采样及其随机流、原S/b冻结；不新冻结参数组、不改学习率或Adam槽。移除的是评分与排序的联合块，不能分别归因两子项，也不是原ELBO或确定性baseline。

训练继续计算原score和anchor作为detach日志；它们没有损失反向路径，计算开销仍计入，不能声称删除全部图像计算。只在实现预检建立这些项的图，核对同样本代数恒等式。无需独立P0或梯度方向科学准入：一次固定批次原A前向复现、一次分解目标前向与retained autograd、一次实际T前向/backward，共3完整目标前向评估、2模型反向、0Adam。核对关键依赖/配置/data/evidence/source/prior、原A存档固定批次、随机流、有限梯度、retained梯度一致、来源payload/Adam/RNG恢复与S/b无梯度；失配保留失败停止，不自动重训A或retry。不做旧24副本审计。

正式400模型backward/Adam，0额外训练梯度探针。每步记录task、局部正则、观测score、anchor及同样本完整原目标标量；后者不是旧A的反事实轨迹。原A有3次基础梯度遥测，本T无这些额外反向；披露该计算差异。训练时score/anchor仍求值，但不反向，反向成本亦不同。

固定2000一次评估RUNMC/BMC各4val患者，原3推断重复/8×4MC、原spacing/恢复协议、每患者先封存概率再读val标签，2新cell。只读复用封存A指标及入口指标；逐患者配对报告Dice/NLL/Brier/HD95/空预测的均值、方向、范围和最差差值，不新增旧评分或访问旧GT。

每项必须同时满足才有单seed机制继续信号：T−A BMC Dice≥+.5pp；RUNMC Dice≥−.5pp；两域空预测数量不增加；两域平均HD95不超过A的110%。并列入口BMC44.07538%，未恢复到入口只能称部分恢复。此为新机制干预标准，所有旧U源≥1pp及既有FAILED原样保留。单seed、反复4val开发不是独立泛化、校准不确定性、精确后验或严格CL遗忘证据。

预算：400正式新增更新、1session/2cell；自首次启动硬3600秒（终止宽限15秒预留在24小时总限制内）、内部3000秒，预检300/训练1200/每eval450秒；全根2,000,000,000字节停止线，创建前按当前NAS实际registered-roots大小预留完整2GB，campaign与窗口29GB停止线取更严格余量。所有输出/cache/tmp NAS，独立timeout及全根guard，不靠心跳。历史clock/失败成本和存储不重置、不删历史。无自动retry、seed/权重/增强搜索。

若迁移恢复且源保持，优先研究评分与任务兼容；源提升但迁移仍下降不算解决；两域更差仅支持此块在当前路径有净价值；小差未达机制信号不搜索权重。FIX_Q/FIX_PHI只有结果支持时才考虑另立协议，不自动形成大网格，不把L-only成功设成新U研究永久前提。最终源码/匿名报告代理push、远端SHA和匿名HTTP200核验，患者身份、原始图像/概率/权重/梯度向量/原始日志/私有路径不发布。
