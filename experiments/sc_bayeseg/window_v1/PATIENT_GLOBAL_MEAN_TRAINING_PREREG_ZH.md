# Mμ：G权重均值下的匹配训练控制

2026-10-21，独立假设、运行前冻结。前一推断消融先独立审计并交付；不追改旧Dμ、K1或U-Net结果。原Dμ权重均值推断的val Dice较4Theta混合增加约.6343pp，而NLL点估计略升，提示硬分割与概率风险可能有不同权衡。下一问题为训练期G权重采样是否值得保留。

## 唯一因素与匹配对照

新增一臂Mμ，5000固定更新。与原Dμ同R32/S0000完整模型/RNG、空optimizer、freshAdam、UNet特征+H均值路径、相同batch2患者均衡/四翻转/schedule5001..10000、lr3e-4/betas(.9,.999)/eps1e-8/wd0/foreachFalse/clip1、FP32 deterministic TF32False及CUBLAS设置。只把训练期G的2Theta采样改为各层固定mean_w/mean_b。局部c/R始终取均值。目标仍CE+原前景Dice+.5*(mu_c²+mu_R²)/128²；但CE/Dice现在作用于权重均值输出，而非原采样目标的无偏替代，必须称改变训练目标的控制，不叫纯ELBO或Bayesian增益。

主比较Mμ与“原Dμ训练后、G均值推断”的既有结果，两者均用同一权重均值推断规则，隔离训练期是否使用固定G或采样G。原Dμ的4Theta混合与U-Net+Dice为次要完整对照；所有控制复用旧指标，不重复训练、推断、评分。不得只对比有利终点。

G全部参数/prior/initial/variance、unused segmentation_head冻结；H后16variance输出行800标量保持原值、梯度/Adam矩0，shared hidden可变；有效可更新483776/优化器tensor槽484576。不用旧Dμ终态接训，不复制oracle状态。

## 数据、封存与指标

仍canonical320 seed0 Siemens20%L10患者230切片；23开发val患者46ED/ES相位544切片历史多轮使用。test24/U38/目标封存，无新seed扫描。训练GT仅loss；纯图像预测train20+val46相位全部native128FP32封存后标签评分，原deterministic恢复384归一化/processed_spacing/三维EDES→类→患者规则。保持GT缺类原Dice规则、NA/HD95空预测惩罚。

主终点val NLL：Mμ−原采样训练/均值推断Dμ。并列完整报告训练和val宏/三类Dice、NLL/Brier、HD95/空预测/失败病例数及患者配对bootstrap10000seed163，其他两对照同样完整。探索性多比较；患者CI非训练seed或MC CI；不以跨零称等效/非劣。固定5000，不按中间val加步/换源/改惩罚或挑最好。此轮无温度校准，原始风险只比原始风险。

## 预算、准入和停止

5000真实optimizer/反向、10000训练编码图像、5000G/10000图像G；0Theta/local采样。每0/1000/...5000六份完整modelAdamRNG。评价774编码图像、398G/774图像G，66native封存/恢复/标签相位。网络调用少于原Dμ训练10000G，不把更新数匹配说成算力匹配。概率理论202899456bytes，含六状态/日志预计<.40GB，启动预留.55GB，实际峰值与四根18GB软20GB硬先核。

合成准入1编码1G/1反向/0optimizer/0真实数据，检查总目标恒等、活跃梯度有限、G无梯度/800方差行梯度0、均值层引用；成本单列。完成独立审计5000连续账本、六全状态/Adam步、原初态RNG/同旧schedule、G/unused/800行与矩冻结、实际目标、66封存重算及全部患者对比，额外66插值/标签0新网络。保留所有工程失败，不从头重跑有效成本。

原首次clock_start_token87614098/original_started_unix1792485279.2578013及全部失败/占用不重置，独立23h45保存23h59硬停/四根18GB软20GB硬/各服务器新增<50GB/总<=24h。create-only、挂载容量真实写读/GPU余量/中性主子argv先核；不删历史腾空间。异常保存partial/model/Adam/RNG/成本准确接续，不能拆名绕限。此单臂到固定终点收口，再依据全部证据选择路线，不自动新训练seed/教师SSLCL/F/尺度族/test/U/目标。
