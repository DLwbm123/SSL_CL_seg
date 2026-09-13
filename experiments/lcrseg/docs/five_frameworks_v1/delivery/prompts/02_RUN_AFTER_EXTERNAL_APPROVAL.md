# 给Codex｜第二轮：外部代码审阅通过后，执行五框架B–D有限实验

> 现在不要使用这个prompt。只有真实外部审阅批准、用户明确启动确认、准确commit与预算都已绑定后才执行。

## 输入必须满足

1. 外部审阅结论 `APPROVED_FOR_EXPERIMENTS` 及其真实证据。
2. 用户明确“允许启动这次B–D实验”。
3. reviewed_code_commit/tree hash、parent_binding、expanded_plan、dependency_lock与运行源码匹配。
4. 正式更新、source训练、smoke、probe/readout/VJP、GPU共享/存储额度已实际展开批准。
5. 真实父方法与合法D0来源已绑定，数据scope明确，test/hidden-U-GT仍关闭。

缺项则只输出具体缺项，不伪造approval，也不尝试跑起来再补审阅。不得使用模板true/空hash，不能自充外部reviewer。

## 执行B→C→D

读取README及全部docs，按已批准的统一定义运行，不在此轮重新设计方法。执行源码固定；文档发布提交不改变服务器运行代码指纹。

B：合成CUDA资格、至多24步discarded current-L真实smoke、数据/来源/几何/读出/恢复资格。利用实时GPU空闲显存，最多并发4个已授权trainer，禁止终止或占用他人独占资源。常规CUDA工程bug按影响范围修复并报告；科学定义改动需复审。

C0：父4配置，完整后续双域轨迹，按规则选基础父配置。
C1：四个新共同基线各8配置，基线也给予合理开发机会。
C2：五个框架各8配置，各1devseed×2orders，完整预算；全部五个必须获得尝试。不用单seed/类别下降、CI跨零、任意+0.005门槛中途杀候选。
C结束：封存每family与每baseline最优配置，生成SELECTED_CONFIGS.json；D不得再调这些参数。
D：五framework+五baseline，3个新优化seed×2orders；重新从各seed合法共同D0开始，不复用C的适配模型。

每个第二后续域从自己的第一后续域checkpoint继续。source按seed共享须验证无未来域暴露；旧source无法合法复用则只训练每seed一次。父旧optimizer/EMA不带入新stage。

## 执行期间

只读当前域训练数据；老域val仅独立evaluator在轨迹封存后使用。生成真实每步/每stage receipt和原子checkpoint；恢复重跑tail计physical成本，不与科学步混淆。所有GPU真实运行、失败/重试、full/readout-only forwards、VJP、EMA、L/U访问、显存、存储独立记账。

正式过程不得自动扩大框架数、rank搜索、种子数、数据集、病例或训练预算。遇到单框架工程问题，保护已完成任务与其他独立分支，不把局部bug强行变成全项目科学停止，也不静默删失败配置。实质错误修复后复核授权是否仍覆盖。

不可使用共享开发val逐epoch挑最好模型。固定stage-final学生部署；不用EMA或repair作为测试推理。无需每个epoch等待用户确认；已授权有限DAG连续执行即可。

## D完成后的结果与停机

写全ExperimentPlan要求的聚合文件、配对统计、患者条件区间、成本、失败列表和5框架排名。三域Final=(2Old+Incoming_final)/3，不用旧两域公式。先两order平均，再seed平均；不把6个单位当独立seed。

自动生成最多2个shortlist，并单独给相对强L-only/SSL基线的证据评估。若都无正面增量，仍给best-as-tested排名，但明确没有支持已成功论文主方法；不要改名洗成PASS，不重写历史结果。

生成每入选框架最多4对照的E阶段方案，但**不自动执行E/F**。独立test、额外数据集和全面官方baseline不在本次scope。

将公开代码/聚合报告/小receipt推送对应结果分支，核验GitHub SHA；原始患者、权重、原始路径/密钥留授权NAS。用真实状态报告运行是否结束、完成/失败任务数、总科学与物理更新、GPU进程是否退出。若平台支持用户授权的detached执行，可以用可审计有限controller，但启动状态不能冒充完成；不得承诺聊天模型会自行异步回访。

最终停在 `FIVE_FRAMEWORK_COMPARISON_COMPLETE_AWAITING_RESULT_REVIEW`；若矩阵未完整结束，则用真实INCOMPLETE状态并列缺项，不创建后台无限监控或新增调参。
