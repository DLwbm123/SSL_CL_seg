# v44：源域继续训练的q/phi更新许可干预（运行前冻结）

v43联合移除score_x与anchor后，T−A源/迁移仅−.01675/+.00789pp，迁移门失败，距入口仍−.98442pp。这个结果不支持该块的本轮净作用足以解释继续训练代价；不能据此断定各子项或历史影响无作用。现在检验两个既有候选更新策略是否能减少该代价，不再添加新辅助loss或调块权重。原v43保留FAILED。

两新臂各400更新到固定2000，共800：FIX_Q固定全局q均值/尺度，phi及RUNMC eta学习；FIX_PHI固定全部phi，q及eta学习。二者从同v23L1600 ACTIVE继承q/phi/域/原先验/Adam/RNG；不清动量、不改学习率/组/采样数，不用T2000作入口。完整原v23L目标（CE+Dice、image .01含score与c/R局部项、anchor1、global/domainKL）保持，不叠加v43块删除。原S/b冻结保持；冻结通过requires_grad=False/gradNone实现，继承的非空Adam槽必须原样保留。FIX_Q仍做原MC，不改为均值预测；FIX_PHI仍传播对输入theta/a的梯度，不no_grad整个识别器。

对照复用封存v25 A2000，不重训/评分；入口v23L1600只读旧指标。原RUNMC4L80切片、batch1、data_seed163000、2global×2local，同L曝光；为入口evidence继续LU适配器读取U图像，但损失不使用U。BMC用RUNMC记忆作未适应评价，无BMC训练、test/U隐藏GT或旧域训练图像。冻结权重不是完全冻结预测函数，phi/q/条件域相互依赖。

旧v21为BMC目标适应800步/freshAdam/v19目标；本轮为RUNMC继续400步/inheritedAdam/v23目标。不能混称同一实验或复活旧四臂；本轮仅两个更新许可条件效应，不足以识别完整q×phi交互或eta影响，不做DOMAIN_ONLY/组合网格/旧模块拼接评分。不能把某臂通过称唯一根因。

必要预检一次：原A固定批次完整目标复现一次前向，两冻结臂各一次完整目标前向/backward（合计3前向目标评估、2模型反向、0Adam），比较所有原目标标量和global RNG；来源payload/prior/config/evidence及关键计算依赖匹配；冻结组gradNone、其余模块与域有限非零梯度、S/b无梯度、来源payload/Adam/RNG恢复。失配保留失败停止，不自动重训A，不独立P0/副本Adam。正式每步检查冻结gradNone，在2000步finalize前核对冻结参数和继承Adam槽完全不变、另一模块与域确已改变；没有额外反向或新梯度诊断。旧A有3次额外基础梯度探针，新臂无，成本披露。

固定2000评估每臂RUNMC/BMC各4val患者、3推断重复、8×4MC，原spacing/恢复/每患者先seal概率后读val标签，共4新cell，不选中间终点/不旧评分。两臂各自相对A独立机制门：BMC Dice≥+.5pp、RUNMC≥−.5pp、无新增空、两域平均HD95不恶化>10%。并列入口BMC44.07538%，未达到仍仅部分恢复；报告全部患者配对方向/范围/最差与NLL/Brier/HD95/空预测。不改变旧U源≥1pp门，单seed反复4val非独立泛化、可靠不确定性、精确后验或严格CL遗忘证明。

预算硬5400秒、内部4800秒、预检300、每臂train1200、每eval450；800正式更新，根3GB停止线。运行前真实NAS挂载/容量/写读/GPU准入、campaign及窗口更严格29GB停止线全峰值预留，独立timeout+rootguard；历史/失败成本不重置，不删历史，不自动retry。

只有FIX_Q过：后继优先结构后验与域更新分离；只有FIX_PHI过：优先共享识别映射保留和域专属路径；两过：不按val赢家宣称根因；两不通过：结束该冻结诊断，不补组合或权重/seed搜索。之后回到明确语义关联的独立U机制，L-only达标不是永久前提。通过也只单seed开发机制信号，不能称SSL成功。

最终公开源码/匿名均值及差值范围/门槛/真实成本，代理push+远端SHA+匿名HTTP200。私有身份/影像/概率/权重/梯度向量/原始日志/私有路径不公开。唯一runmc每30分钟，不另建监测。
