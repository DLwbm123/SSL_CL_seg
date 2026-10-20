# 有界空间后验接口诊断：R16→R32，固定CE+Dice且beta0

2026-10-20，因子结果后、运行前冻结。四臂各5000已结束；G_K0/G_D/G_DK0训练Dice85.8616/85.7120/85.4741%，U_D90.2017%，训练MYO78.5535/78.9451/78.9640%对85.3509%，仍有约6.4pp缺口。val G_DK0 69.0235%接近但低于U_D70.7044%，差−1.6809pp的患者区间[−4.2835,+1.2044]不支持宣布非劣；NLL/Brier也更差。Dice/KL改善不能证明当前空间接口已充分拟合。

只新增一条原G0冻结、UNet特征+H可训练的轨迹，将HR前空间池化16×16（8×8块）改为32×32（4×4块）；c分支仍全局均值、16维。R从4096维增至16384维，但HR仍共享原16→32卷积，活跃参数仍484576。生成器spatial_logits本来就插值任意R网格，不更改其weights/prior/logsd/F或其他模块，不调用要求R16的旧完整bound接口。原raw尺度范围不改。

控制为刚完成G_DK0：相同CE+Dice、beta0。使用该控制是为了隔离KL维数混杂，不以它val排名选择来源；不额外补beta1或先解冻G。beta0是非原ELBO的表达/优化能力诊断；若改善，不能直接当Bayesian模型或数学分辨率上限的证明。先验证空间接口，再另立概率目标如何恢复，不能临时改变本轮。

源为因子G_DK0/S0000全模型/RNG，断言其step0、freshAdam状态空、lambda1/beta0；所有参数初值相同，只改变池化网格。不是继承G_DK0终态，也不是数值q初态相同：空间池化改变q形状和初值输出。freshAdam3e−4、联合clip1/原betas eps wd/FP32确定性、同原schedule5001..10000/batch2/四翻转，固定5000步，6完整modelAdamRNG。

原训练2Theta2local/原训练和评价种子规则不改。Theta独立流可逐抽样复用，但局部R张量形状改变，使后续local RNG消费不同；相同seed不等相同c/R逐draw噪声。不把一次训练及患者bootstrap当局部MC或训练seed方差证明，报告此限制，不为追分扫描seed。softDice原定义不变：每图GT存在前景类等权、epsilon1e−6、全背景图Dice0，GT缺类CE仍惩罚。

数据仍Siemens20%L10训练患者230切片，23开发val患者544切片；两队列20+46共66相位预测全部封存后才本轮读标签评分。固定4Theta4local/batch2/原seed、native128 float32缓存→同GPU384恢复/归一化，原spacing/三维指标/EDES→患者聚合。复用G_DK0及U_D控制全部旧数值不重训/重推断/重评分。主看训练MYO R32−R16，完整报告训练/val三类与宏Dice/NLL/Brier/HD95失败/NA、对两控制的患者配对差及bootstrap10000seed163，禁止只报最有利指标。该val多轮开发使用，test24/U38/目标域不读取。

预算恰5000新optimizer/reverse/10000训练编码/10000Theta/20000G/40000图像G；评价774编码/1592Theta/6368G/12384图像G、66标签phase/66插值重构（非网络）。6状态+native概率预计<.4GB、启动预留.6GB。pro5000原GPU2/UUID/授权/data/bmw新create-only，原R_START首启clock及全部失败成本保留，<=24h/新增实际<50GB、独立原23h45保存/23h59硬停与四根18GB软20GB硬及新45/49GB。先核源/原剩余clock/容量挂载真实读写/GPU余量，中性主子argv ps+nvidia核。不删旧结果或失败、不终止别人。

工程准入PASS：R输出[1,16,32,32]、484576活跃参数全部梯度有限、G无梯度，native→384与原predict接口最大差0。合成1reverse/36G/10Theta/3编码图像/0optimizer/0真实val或test数组，独立列账。前因子必须EXIT0/AUDIT PASS后才启动，不从尚未审计结果抢跑。

非有限/源变化/边界/硬预算失败保存partial和成本准确接续，不盲重跑。完成独立审计5000连续账本/6状态/Adam、源S0000模型RNG一致/相同调度/仅R池化结构变、G冻结、66封存native与384恢复/全部指标差/EXIT FAILURE与worker guard退休；中文正负报告/匿名全指标/源码协议代理push核SHA+匿名200，更新唯一sc-bayeseg。结果后自主据证继续，不因负结果盲加步，不自动开启完整BayeSeg/F/SSL/CL。
