# V3L-AGREEMENT-SELECTION：固定教师的一致性筛选与同数量随机对照

## 证据、目标与边界

V3E/F/H/I预定联合改善条件均失败，不继续lambda/窗口/beta微网格。J确认训练L上的高置信错误；K已完成并发布789be9dd01d98179e9eaf85076271e39bb6e0058，50状态不变、50L预测精确复现J、1650唯一批次、0训练更新/U标签/val/test。

K显示FINE的U一致区保留99.9229%/99.9590%，几乎不能筛除共有错误。FIXED的U保留93.9296%/90.8460%，L一致区像素错误下降4.1440/8.9148pp、类平衡错误下降11.0459/26.4056pp，两域均5/5同向。固定教师是诊断后提出的探索性机制候选，未通过H性能条件；K训练L偏差、共有错误、clean视角与训练扰动差异仍可能使本轮失败。

假设：固定教师的部分高置信过时预测与当前student不一致，筛去这些位置可能改善新旧域联合结果。可区分解释是仅减少U像素数量/预测类别剂量；用逐步、逐图像、逐教师预测类别完全同数量的随机筛选对照检验。数量匹配不等于梯度范数、熵或真实类别匹配，不冒称已完全控制所有学习信号差异。

## 冻结矩阵

run_id=v3l_agreement_selection_20261002T110500Z。V3I执行2acdec5fd217bf31585f343befa525198be8f71a的完整ENTRY，复用V3E五个8000步REFUGE源，种子168..172，当前域RIM_ONE_r3/Drishti_GS，各H=1200。每seed/domain五臂，50终点、60000 main；固定顺序ORIGINAL、FINE_05、FIXED_05、AGREE_05、RANDOM_05，复用5步round-robin队列。

|臂|U权重|教师|筛选|
|---|---:|---|---|
|ORIGINAL|0|原.99 EMA|0U，与历史L-only一致|
|FINE_05|.5|原.99 EMA|原>0.7接纳|
|FIXED_05|.5|完整入口教师不变|原>0.7接纳|
|AGREE_05|.5|同一固定教师|>0.7且teacher/student clean argmax一致|
|RANDOM_05|.5|同一固定教师|从>0.7区域按图像、teacher预测类均匀随机保留，与配对AGREE同一步各类别保留数严格相同|

不扫阈值、不增加教师/辅助损失，不改LR、lambda、Adam、调度、数据流、当前L或U、置信>.7、KL目标。仍用soft teacher KL及原几何有效像素总数作分母，不按剩余接纳数重新归一化。每臂固定终点student评分，不选择最好checkpoint/seed。

## 筛选与匹配实现

共享Trainer.components仅新增可选只读observe_u回调，默认路径数值操作不变；共享u_loss仅新增可选布尔selection掩码，与原接纳相交，分母不变。所有历史调用不传该参数。新协议绑定现有Trainer.losses/update，不复制训练循环；其他三臂调用原函数。AGREE在同一U弱图像及teacher q上额外做一次student clean推理，torch.no_grad+readonly恢复RNG、buffer、mode、telemetry、last、provider读数，mask完全detach；随后仍走原LCTX U前向和优化。

每个保留的AGREE更新成功后记录当前两张U图各自teacher预测类0/1/2的接纳/保留数量与valid总数。每5步块AGREE先于RANDOM，因此随机臂只能使用已经成功记录的配对日程，不能用自身student重新定义剂量。RANDOM检查当前teacher类别接纳总体与日程完全相等，用独立CPU torch.Generator(seed=stable(V3L/random,seed,domain,step,image-index,class))均匀无放回选位置，不改变训练RNG。缺少日程、总体不同或保留数量越界立即失败，不补值/改协议。

同源图像顺序/geometry是原stateless stream且教师完整固定，保证两个臂的候选总体一致；每步每图每类的实际保留数硬断言相同。私有SELECTION_LEDGER记录24000个成功更新（12000配对），最终按seed/domain汇总数量与类别覆盖公开。日程是随机臂的外生执行依赖，恢复必须同时绑定完整trainer与该日程日志；本轮不实现隐式续跑。100步保存原完整trainer，300/600/1200导出沿用历史，仅1200评估。

## 主指标与预定决策

当前域case-mean rim/cup Dice macro与旧REFUGE同指标，逐域、逐种子完整配对；域/种子等权，旧域change相对ENTRY单列，不称全序列BWT。val从V3B起是反复查看的研发集，本轮是新机制探索，不能称独立患者泛化或新种子确认。test封存，无U隐藏真值，不重新划分。

主要AGREE−RANDOM：两项平均差都>0、至少4/5种子两项差均>=0且至少一项>0、两域均值两项均>=0，三条同时成立才selection_supported。AGREE−FIXED、AGREE−FINE采用完全相同的联合条件；三组参考均通过才practical_candidate。AGREE−ORIGINAL完整报告但不改变主比较。全部类别Dice、负域/种子、成本、每类保留率一并公开；不事后改容忍阈值、选单域或重新加权。

若只优于随机而不优于FIXED/FINE，只是机制信号，不称满意候选；若实用条件通过，另冻结至少5个新优化种子及少量有意义的参数敏感性确认。若失败或只移动权衡曲线，保留反证并重新分析机制，不能继续盲目调整一致性阈值。优化种子稳定不等于独立患者验证，未来最终test须另审历史用途/权限并冻结方法。

## 预算、检查与交付

准备起点2026-10-02T11:05Z；optimizer截止2026-10-03T01:05Z（14h），硬截止03:05Z（16h），不重置。main60000、smoke计划100上限120、source/development/controller0；工程replay reserve3006=ceil(.05×60120)，只有明确诊断后的有效修复才能使用，不自动重启；相同错误两次停止受影响路径。正常完成新增60100，本序列累计297300；既有源40000和G诊断VJP800单列，J/K均0。本轮smoke的梯度/预演与额外clean推理记录在操作成本中。

CPU检查：可选mask全保留精确等于原loss、原valid分母、零mask可微/被拒像素零梯度、精确预测类数量、随机确定性/RNG不变、非法数量拒绝、五臂矩阵与二维gate。GPU十个域/条件做现有实际Adam预演对照、只读特征完整状态检查、快照所有权/精确重复续步，加每条件五步计时；100物理smoke计入账本。固定teacher真实全状态不变、选点更新次数/同量断言通过后，按1.25×实测矩阵耗时+1800秒估算，必须完整适配剩余optimizer时间，否则停止，不缩矩阵。无需为已有底座另开新过拟合训练。

最终核验50端点、60评估文件/120域评分、60000main+100smoke、40条有U轨迹各1200U、ORIGINAL10条0U；30固定teacher全状态同入口，20动态1200EMA；20选点轨迹各1200成功筛选；12000对逐步逐图逐类数量精确相同。30个ORIGINAL/FINE/FIXED历史控制新旧域分数精确复现V3I（最大差0），否则先诊断，不发表机制胜出。所有物理attempt/success/失败成本保留，不能把预演当免费训练。

单GPU4按显存余量准入，NAS新create-only目录、NFS挂载/容量/真实读写探针通过，with_nas_storage.sh+原guardian中性argv，ps/nvidia-smi确认后结束当次唤醒。每小时持续监测，完成后发布ENDPOINTS.csv、SUMMARY.json、SELECTION_DECISION.json、SELECTION_AUDIT.json、TEACHER_AUDIT.json、CONTROL_REPLICATION.json、ALL_COSTS.json、COMPLETION_AUDIT.json与实质FINAL_INTERPRETATION.md到本目录reports及Downloads/V3L_AGREEMENT_SELECTION_20261002，经代理推送、remote SHA和匿名访问验证。患者、原始像素/标签、checkpoint和逐步私有日志仅留NAS。
