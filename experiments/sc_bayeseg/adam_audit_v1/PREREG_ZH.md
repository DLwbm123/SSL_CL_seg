# J初始状态Adam候选位移零更新重建

用户授权仅从已有split_v1 J/initial.pt重建原随机流下的完整梯度、全局裁剪和Adam候选位移，不写回参数、不调用optimizer，不做code适配、局部优化或早期轨迹训练。

使用原同两张Siemens可见L、NL2/NU0、128²、FP32确定性、原torch环境、所有J可训练参数及完整ELBO，恢复原Python/NumPy/Torch CPU/CUDA RNG。一次原单抽样前向；不得用上一轮冻结非语义参数的4次MC平均代替。核对loss、全部100份参数梯度的全局范数及随机统计与历史J added_step1一致。

原clip1使用CUDA foreach逐参数2范数→stack全局2范数，coef=min(1,1/(norm+1e-6))，只对返回的梯度副本作out-of-place缩放，不写p.grad。读取每个参数自己的exp_avg/exp_avg_sq/step及组超参，按torch2.7.1 CUDA默认foreach Adam非capturable路径，用out-of-place张量原语计算下一阶矩、偏差校正、分母和候选参数值，位移=candidate−current，计入FP32参数加法舍入。严禁调用Optimizer/step/functional Adam，也不把候选张量装入模型。

用独立逐tensor公式核对foreach候选数值；这验证两种公式一致，不是与真正optimizer.step逐位对照，原历史第1步权重并未独立存档。对未舍入公式额外分解历史一阶矩和本步梯度贡献，固定实际二阶矩分母；不是清空历史或替代优化器对照。

报告实际训练抽样label CE及确定性mu链CE的梯度与候选位移内积；同时给出−lr×clipped-full-gradient方向作为原始梯度参考（不执行SGD），报告尺度及单位范数方向，避免把不同步长当因果比较。

通过reverse-over-reverse JVP计算mu沿候选及原始梯度参考方向的一阶变化，不扰动参数。GT2 margin：Mbg=mu2；Mfg=mu2−max(mu1,mu3)；Mall=min(Mbg,Mfg)。梯度在初始胜出竞争者处取值，不当作跨越后重新选择竞争者的真实非线性变化。

子集预先固定：全部GT2；|Mall|≤1的近决策边界；|Mbg|≤1的近背景边界；GT2四邻域标签内边界（复制填充，无周期跨边缘）。报告各margin初值及方向变化的均值、分位数、降低比例；线性预测过零仅为JVP外推，不是实际应用位移后的分类。以上阈值仅用于可见L诊断，不用于训练筛选。

预计1次训练图前向、≤8次autograd.grad（完整梯度、两种label梯度、JVP），optimizer0。核对模型、载入Adam及checkpoint状态不变，原训练账本仍1204。本轮不启动监测。完成后中文Markdown报告、源码协议匿名聚合结果按全局规则交付，私有数据/权重/身份/路径/原始日志不公开。不因方向有害或无害而自动追加干预。
