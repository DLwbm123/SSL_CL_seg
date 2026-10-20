# 匹配前景Dice监督与局部KL因子实验

2026-10-20，P0补充结果后、训练前冻结。P0显示端到端G训练MYO召回47.2251%、预测/GT体积比.60546，U-Net85.5868%/1.10417；验证G召回34.7578%、体积比.51238。G训练MYO拟合缺口先于开发差距。R logsd大量接近上界，但绝对KL和饱和不能证明KL主导。先保持结构/G0不变隔离目标，不扩R/解冻G/加U或CL。

## 因子与固定来源

原E2E G CE+KL、U CE两条已完成对照直接复用，无重训/重推断。新增固定顺序：G_K0(lambda0,beta0)、G_D(1,1)、G_DK0(1,0)、U_D(1,不适用)，各恰5000步，总20000新增optimizer/reverse。三G臂训练U-Net特征+H、G0全部冻结；U_D训练U-Net特征+直接头。所有臂同原patient_validation PRETRAIN/S5000完整模型和RNG，freshAdam，G的H从原R_START/R1 S0000，不接任何终态或oracle/G3。G0唯一原spatial_generator/S0600 cursor8675、全部prior/initial/方差不变。活跃参数G484576/U482500；unused backbone segmentation_head保持冻结。

G目标 E_sample pixelCE + lambda * foreground_softDice_loss(mean_sample_probability,Y) + beta*(KLc+KLR)/128²。样本原2Theta×2local共4；保留期望CE，不改为混合NLL。U目标pixelCE+同一定义Dice。Dice按每张图、每个GT存在前景类计算1−(2sum p*y+1e−6)/(sum p+sum y+1e−6)，每图GT存在三类等权后batch平均；GT缺类不进入该图Dice项，仍由CE惩罚假阳性；全背景图Dice项0。lambda/beta只0/1，无系数扫描。beta0/新增Dice均改变原纯ELBO目标，不能将其成功说成Bayesian增益或证明KL有害。

训练仍canonical320 seed0 Siemens20%L10患者230切片，患者均衡batch2/四翻转、绝对schedule5001..10000相同，128图像双线性/标签最近邻。Adam3e−4 betas(.9,.999) eps1e−8 wd0 foreachFalse、FP32确定性/无TF32、联合clip1、原H raw范围−4+3tanh(raw/3)，原训练Theta720163+step/local820163+step，无scheduler/earlystop/新初始化seed。每臂0/1000/2000/3000/4000/5000完整model/Adam/RNG，共24份，保存实际CE/Dice/KLc/KLR及连续账本。

## 固定评价及精度无损存储

四臂全部训练完成后，生成10训练患者20相位230切片及23开发val患者46相位544切片的纯图像预测，无增强，G原4Theta4local/batch2/seed620163+1000phase+batch。所有4×66=264相位预测封存后才读取本轮评分标签。训练图像是已见病例仅拟合评价，val多轮已用仅开发集；test24/U38/目标域始终不访问。

仅保存原生128网格的完整float32概率，评分时同GPU双线性恢复384并按类归一化，随后原三维phase_metrics。无量化/无删历史，既有来源模型合成检查native保存→恢复与原predict_probabilities输出最大误差0（U/G均0）；不是改为128网格Dice。原标签、processed_spacing/NA空预测/物理对角线惩罚/EDES→患者宏平均完全不变。全患者逐类Dice、NLL/Brier/HD95/空预测数，训练和val都报告。原G/U控制聚合直接复用，不再次评分。

固定报告每新臂−原匹配G/U、G_D−U_D和G_DK0−U_D、Dice效应在beta0/1下与KL去除效应在lambda0/1下，以及Dice×KL交互和G/U匹配Dice增益之差。各队列患者bootstrap10000/seed163，探索性多比较不称确认性证明，不能将患者×配置当独立患者。全臂结果不择优替换原负结果。

资源分流参考：相对原G的val宏平均>=+2pp、MYO>=+5pp、RV下降不超过1pp且空预测数不增加，可作为继续投入的探索门槛；不是超越U-Net/临床合格线。仍须看新U_D匹配差；若两者同升而差不缩，不能归功G。beta0训练升而val跌则提示拟合/泛化权衡，不自动永久删KL。若仍训练弱，再另立空间接口假设，不能在本轮临时扩容或加步。

## 预算、工程准入与完成

20000新optimizer/reverse/40000训练编码图像，三G臂30000训练Theta/60000批量G/120000图像G；评价3096编码图像/4776Theta/19104G/37152图像G，264标签相位读取及264插值恢复（后者非网络前向）。24完整状态+原生预测预计<1.2GB，启动预留1.8GB，source/tmp/cache/log/模型/概率全部pro5000原GPU2/授权/data/bmw新create-only。旧概率/失败不删除、不压缩改写。沿用R_START原首启clock及全部失败成本，<=24h/新增实际<50GB，独立原23h45保存23h59硬停、四根18GB软20GB硬和新45/49GB保持，训练每1000步及预测每相位存储余量检查。先核源/GPU显存/挂载容量真实写读/剩余clock，主子argv中性并ps+nvidia检查。

工程准入已通过：lambda0 beta1与原semantic_loss同噪声损失差4.77e−7（<=1e−6）、q梯度最大差0；Dice完美前景与全背景缺类规则通过；native128缓存恢复384与旧U/G预测逐像素最大差0。合成40G/12Theta/4编码图像/2q梯度请求/0optimizer/0真实val或test数组，成本单列，不冒充正式更新。

P0补充须EXIT0/AUDIT PASS后才启动。非有限/源G变化/边界/预算失败保存partial全状态与真实成本，准确接续不盲重跑。完成核验20000连续更新、24全状态/Adam步/源和RNG/控制schedule、全部G冻结、各臂真实lambda beta配置、264封存概率及同网格指标/交互复算、EXIT/FAILURE和worker/guard退休，中文正负报告/匿名全指标/源码协议经代理push核SHA+匿名200；私有身份/路径/像素/权重/RNG/概率/日志不公开。唯一sc-bayeseg每30min继续，不逐轮等确认。
