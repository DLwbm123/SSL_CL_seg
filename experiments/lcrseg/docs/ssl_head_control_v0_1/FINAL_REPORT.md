# 分类头对照与 PAS 等数量归因 V0.1：最终报告

完整固定队列已结束。工程完成；SSL增量未建立；线性头的监督参考改善未在两个新优化种子上通过预定复核。本轮到此停止，不继续同一val上的温度、阈值、增强、权重或种子搜索，不进入CL。

## 终态与执行范围

| 维度 | 终态 |
| --- | --- |
| ENGINEERING | COMPLETE |
| PAS_MATCHED_COUNT_EVIDENCE | DESCRIPTIVE_COMPLETE |
| HEAD_SUPERVISED_EFFECT | SUPERVISED_HEAD_REFERENCE_NOT_REPLICATED |
| SSL_INCREMENT | NOT_ESTABLISHED |
| PAS_INCREMENT | DESCRIPTIVE_COMPONENT_CONTRASTS_ONLY |
| CL_AND_SOTA | NOT_EVALUATED |

从405337e71d22aef011a041664fbc1b4b72b60408建立独立分支 codex/ssl-classifier-head-control-v0-1。实际执行源码始终为 f478cc204a87a807858c76a95fc4246d301eeff2；后续提交仅补充公开报告。冻结原始协议、HEAD_CONTRACT、来源和门槛保持不变。旧Foundation的SSL_FOUNDATION_NOT_ESTABLISHED及更早历史证据均保留。

P0四组已完成。P1六任务15900步；两个SSL候选均未过筛查，但监督参考改善通过，按唯一获准分支完成P2_SUP八任务21200步。正式合计37100步，与manifest预算、逐步日志、训练receipt及成功退出记录一致；历史累计168608步。未启动P2_SSL或旧seed12/13。

## P1：固定seed11筛查

主结果是最终epoch100学生，Q为两域等权rim/cup宏平均Dice；表格四舍五入只为阅读，所有判断使用完整精度。

| 臂 | RIM-ONE-r3 Dice | Drishti-GS Dice | Q | 相对LIN_SUP的Q差 |
| --- | --- | --- | --- | --- |
| LIN_SUP | 0.715601 | 0.717985 | 0.716793 | 0.000000 |
| LIN_MT_CONF | 0.707838 | 0.717039 | 0.712439 | -0.004354 |
| LIN_MT_PAS | 0.714813 | 0.713752 | 0.714282 | -0.002511 |

LIN_MT_CONF与LIN_MT_PAS的Q增量分别为−0.004354和−0.002511，均未达到+0.010；CONF也越过单域−0.005保护线。不能以EMA或某个更早epoch替代最终学生。

LIN_SUP相对更强旧监督参照SUP_G1的Q增量为+0.011428，且各域及rim/cup保护条件满足，因此自动进入监督分支复核。旧SUP_G0、SUP_G1、MT_PAS_G1的Q分别为0.696040805、0.705364552、0.712331688；这些是固定历史参照，未重训旧seed11任务。

## P2_SUP：新优化种子复核

| seed | LIN_SUP Q | SUP_G1 Q | Q差 | RIM域差 | Drishti域差 |
| --- | --- | --- | --- | --- | --- |
| 21 | 0.712062 | 0.673865 | 0.038197 | 0.016970 | 0.059424 |
| 22 | 0.696838 | 0.698891 | -0.002053 | -0.013910 | 0.009803 |

两个新种子的平均Q差为0.018072，通过均值+0.010门槛。但seed22的Q差为−0.002053，违反每seed必须为正；RIM域差−0.013910低于−0.005；同域cup差−0.032346低于−0.020。因此终态为SUPERVISED_HEAD_REFERENCE_NOT_REPLICATED。均值为正不能覆盖这些失败条件，也不把seed11混入确认统计。

## 全部最终rim/cup代价

| seed | 域 | 臂 | rim Dice | cup Dice | 宏平均Dice |
| --- | --- | --- | --- | --- | --- |
| 11 | Drishti_GS | LIN_MT_CONF | 0.688322 | 0.745756 | 0.717039 |
| 11 | Drishti_GS | LIN_MT_PAS | 0.679737 | 0.747766 | 0.713752 |
| 11 | Drishti_GS | LIN_SUP | 0.680344 | 0.755625 | 0.717985 |
| 11 | RIM_ONE_r3 | LIN_MT_CONF | 0.745594 | 0.670083 | 0.707838 |
| 11 | RIM_ONE_r3 | LIN_MT_PAS | 0.746305 | 0.683321 | 0.714813 |
| 11 | RIM_ONE_r3 | LIN_SUP | 0.742485 | 0.688718 | 0.715601 |
| 21 | Drishti_GS | LIN_SUP | 0.662910 | 0.770247 | 0.716579 |
| 21 | Drishti_GS | SUP_G1 | 0.574201 | 0.740108 | 0.657154 |
| 21 | RIM_ONE_r3 | LIN_SUP | 0.745081 | 0.670008 | 0.707545 |
| 21 | RIM_ONE_r3 | SUP_G1 | 0.739900 | 0.641250 | 0.690575 |
| 22 | Drishti_GS | LIN_SUP | 0.691916 | 0.732764 | 0.712340 |
| 22 | Drishti_GS | SUP_G1 | 0.661368 | 0.743706 | 0.702537 |
| 22 | RIM_ONE_r3 | LIN_SUP | 0.737970 | 0.624702 | 0.681336 |
| 22 | RIM_ONE_r3 | SUP_G1 | 0.733444 | 0.657048 | 0.695246 |

完整阶段曲线见EPOCH_CURVES.csv，学生与EMA分别保留；完整差值及head×SSL对照见HEAD_SSL_CONTRASTS.csv。

PAS相对同头CONF的seed11两域平均差为+0.001844：RIM域+0.006975，Drishti域−0.003287，方向不一致。它仍低于同头监督参考，没有获准SSL新种子复核，故只是单seed组件对照，不能声称PAS增益可复核。

## 等数量审计、概率和实际损失

P0完整解释见PAS_FIXED_MASS_AUDIT.md。PAS并未显示相对同数量confidence top-k的一致优势；部分Drishti rim相对随机的正差不等于分割增量或安全性。

以下均为最终学生clean evaluator的患者平均；错误前景数量是平均每例像素数。

| seed | 域 | 臂 | NLL | Brier | pmax>0.99比例 | 错误前景confidence | 错误前景数量 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | Drishti_GS | LIN_MT_CONF | 0.252735 | 0.095988 | 0.917974 | 0.893598 | 6715.560000 |
| 11 | Drishti_GS | LIN_MT_PAS | 0.265339 | 0.099981 | 0.917228 | 0.899860 | 6894.280000 |
| 11 | Drishti_GS | LIN_SUP | 0.261110 | 0.096848 | 0.921621 | 0.893485 | 6579.080000 |
| 11 | RIM_ONE_r3 | LIN_MT_CONF | 0.225690 | 0.066815 | 0.959130 | 0.909503 | 3658.075000 |
| 11 | RIM_ONE_r3 | LIN_MT_PAS | 0.235261 | 0.070819 | 0.957955 | 0.907476 | 4146.100000 |
| 11 | RIM_ONE_r3 | LIN_SUP | 0.238896 | 0.069185 | 0.961896 | 0.911904 | 3787.975000 |
| 21 | Drishti_GS | LIN_SUP | 0.318226 | 0.102390 | 0.932663 | 0.906677 | 6426.840000 |
| 21 | Drishti_GS | SUP_G1 | 0.694463 | 0.119324 | 0.949804 | 0.938189 | 7136.680000 |
| 21 | RIM_ONE_r3 | LIN_SUP | 0.276934 | 0.069861 | 0.969814 | 0.928020 | 4045.600000 |
| 21 | RIM_ONE_r3 | SUP_G1 | 0.372173 | 0.071144 | 0.956364 | 0.876405 | 4191.925000 |
| 22 | Drishti_GS | LIN_SUP | 0.264404 | 0.093054 | 0.927536 | 0.909103 | 6188.480000 |
| 22 | Drishti_GS | SUP_G1 | 0.331053 | 0.096538 | 0.940576 | 0.920472 | 6413.760000 |
| 22 | RIM_ONE_r3 | LIN_SUP | 0.235096 | 0.069797 | 0.958098 | 0.916908 | 3998.025000 |
| 22 | RIM_ONE_r3 | SUP_G1 | 0.381535 | 0.070782 | 0.963525 | 0.900401 | 3973.800000 |

线性头仍存在很高的confidence，seed11有约91.7%–96.2%的有效像素pmax>0.99。P2中线性头NLL均低于对应SUP_G1，但seed22 RIM Dice仍退化；概率尺度/校准改善不能替代Dice或SSL门槛。未拟合温度。

实际SSL训练按教师预测类记录接受数、MSE、weighted MSE及应用真实mask/分母/lambda后的解析局部logit梯度。四个SSL任务的背景接受像素占约85.0%–91.7%，但背景weighted MSE份额约7.0%–34.3%，因此单看像素coverage不能推出损失由背景支配。TRAINING_CLASS_LOSS.csv是实际训练统计；GRADIENT_LOGIT_DIAGNOSTICS.csv及CORRECTNESS_STRATA.csv是固定evaluator分层。局部logit梯度不是全网络参数梯度，不能据此宣称encoder梯度消失。完整分位点、10等宽bin可靠性表、有效分母和概率质量表全部提供。

## 工程、成本与部署

14个训练任务、14次单学生部署验证、70次epoch20/40/60/80/100诊断以及4次P0全部完成。32个子进程exit_code均为0；70次诊断的学生/EMA/RNG/训练原型状态均保持不变。未见正式失败或重试。实际训练37100次backward、37100次Adam更新、37100次EMA更新；GAS监督导数调用10600次是SUP_G1既有算法组成，不是新增诊断backward。额外真实诊断backward为0。

训练前的精确源码CPU/CUDA各8项测试通过；共5次开发/精确源码资格运行805次合成更新，另有隔离真实smoke8次更新，均不算入37100正式步。源码中的断点恢复测试逐值核验学生/EMA/Adam/head_mode/RNG/数据顺序，实际任务无恢复重跑。

14个任务的完整模型数均为学生+当前EMA共2份，部署进程均为1份学生。每份参数1936064字节；惰性sigma和grad_update各1728字节，grad_update不进入Adam；线性sigma冻结，所有任务sigma无梯度。峰值CUDA allocated约867.92 MiB，reserved约998.00 MiB。整个队列墙钟约36.11分钟，后台进程已退出。

训练学生L前向74200张次，学生U/EMA U各16640张次；训练原型416张次，固定诊断原型910张次，新学生/EMA诊断各11375张次，P0额外1300次sample-model；单学生部署另有455张次前向。完整计数、loader打开次数、子进程退出、状态及内存见TRAINING_AND_MEMORY_ACCOUNTING.json和CLOSEOUT_VERIFICATION.json。最终部署仅加载head_mode明确的学生checkpoint，复现所有14个任务的最终预测，无新增GT读取。

## 交付边界与停止

公开代码、协议、汇总指标、完整门槛、资格/部署/内存和退出证据。未发布GT、患者标识、逐病例像素、原始影像和checkpoint；这些保留在NAS的本轮create-only目录。SOURCE_AND_INPUT_LINEAGE.json保留启动前冻结来源，SERVER_INPUT_LINEAGE.json提供服务器输入核验。原始编译结果也保留在服务器，不被这份解释性报告覆盖。

这仅是固定data_split_seed0上的优化随机性复核，不是跨划分/外部域验证、CL或SOTA结论。本轮关闭这一固定配方的分类头参数化排查；没有启动新模块、额外种子、风险拟合、冻结层或CL。未读取test/隐藏GT或旧formal_03，未改变旧终态/锁，未合并main。
