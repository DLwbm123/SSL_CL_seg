# 正规空间潜变量：固定qR后的生成后验坐标更新

M/S/V与采样因素诊断表明，去KL不能解释主要不足，S/V均值拟合也不足；末层输出锥可行不证明整体可达。当前anatomy未与新R变量共同适配，因此实施正规模型内下一坐标步，而非延长相同局部失败设置。

唯一组合起点：原discreteL/S0600的全部model/Adam/scheduler(cursor8075)，加已完成spatial_residual_local/S0500正规qR和其已前进的RNG。原V只更新局部变量、未动模型/Adam；此组合是合法坐标VI当前状态。记录S0000中完整qR与RNG，绝不使用M/S去KL状态或oracle重构权重，不重设prior，不声称跨模型持续posterior递推。

仅anatomy三Bayesian层12个mean/logsd张量训练，其余88模型参数及其Adam冻结，qR两Gaussian字段全程严格冻结。原两L四翻转batch8，NL2/NU0，FP32/lr3e-4/clip1，原Adam历史保留，scheduler连续。每主步一Theta与一组局部样本，完整正规loss=−mean L_bound/N+全局Theta_KL/(2N)，qR解析KL计一次。没有额外CE、类别权重、教师或真实U/目标/新seed。

固定600main+首末各2golden=4工程，共604actual optimizer，额外诊断反向0，不调参/早停/自动重跑。每100完整checkpoint（包括qR），真实model cursor8075→8675，qR局部counter保持500；首末两个golden对核对model/Adam/scheduler/RNG与固定qR。活跃12 Adam各+600、冻结88参数Adam、prior/initial/qR不变。

0/300/600评价两既有流各2Theta×4local共48bound，评价恢复训练RNG；报告完整ELBO分项、Theta KL、R KL、标签条件均值8view各类Dice/CE、训练图像单独recognition（应精确不变），不读heldout。24项.8仍只是标签重构描述门限，不参与选择。单图ELBO不把全局KL算成证据。若改善也不称SSL/CL/图像泛化有效。

阶段完成核验604物理连续、600main4golden、七完整状态、两个golden对、各参数Adam/cursor/prior/qR与退出回执。训练失败保留完整partial及真实成本，只有准确恢复才继续，不盲重跑。20GB硬/18GB软四根、数据盘create-only中性入口、UTC03:30截止，窗口结束不终止健康任务。
