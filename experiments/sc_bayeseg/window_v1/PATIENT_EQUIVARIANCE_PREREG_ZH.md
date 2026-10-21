# 修正后的四臂空间一致性：运行前冻结协议

2026-10-21，用户明确解除旧18GB限制、保留新增实际<50,000,000,000字节及原始24h预算，并立即推进。此为独立目标干预，原实验和负结果不追改。沿用原clock_start_token87614098/original_started_unix1792485279.2578013，**不重置原时钟或已发生占用**。旧20GB累计硬限与18GB软限一并不再作为本新阶段停止线；新阶段采用新增路径和四根累计均45GB软/49GB硬的更保守守护，保证原链新增实际<50GB，原23h45保存/23h59硬停不变。历史协议/守护源码不改，只新阶段使用新限额模块。

## 假设与四臂

U_2V/U_EQ/G_2V/G_EQ按此固定顺序，各5000更新，共20000主更新/主反向。对照同样执行双视图前向、监督和JS计算，但JS系数0；不以旧单视图终态替代2V控制。U两臂从factor/U_D/S0000全model/RNG空Adam，G两臂从原R32/S0000全model/RNG空Adam，freshAdam，不续终态。原共享backbone起点核验，H/直接头差异保留，不称纯G架构因果。旧U_D/Dmu/MEAN_READ仅背景参考，不重训旧臂/评分，不接G3或GT oracle局部状态。

只canonical320seed0 Siemens20%L10训练患者230slice，两相位/患者；23重复开发患者46EDES544slice。test24/U38/目标封存，无新初始化seed扫描或教师/EMA/SSLCL/F/尺度族/阈值/拓扑项。固定5000终点、同原schedule5001..10000、batch2患者均衡和四翻转；Adam3e−4/.9,.999/eps1e−8/wd0/foreachFalse/clip1/FP32 deterministicTrue TF32False CUBLAS=:4096:8。

G仍局部c/R均值、R32；训练每步2Theta样本，以720163+step独立生成，每个Theta仅draw一次并同时用于两视图，不能分别draw。全G/prior/initial/variance严格冻结，unused backbone segmentation_head冻结。有效G更新483776/优化器槽484576，其中800方差输出标量值/grad/Adam矩必须0变化；shared hidden变化不等σ输出数值固定。U更新482500。仅输出概率一致性，不约束中间c/R。

## 双视图、原CE/Dice与新JS

第一视图完全复用原schedule输入/标签及原flips。每图第二视图用固定CPU生成器930163+step抽relative flip∈{1,2,3}，在第一视图上再翻转；两视图不同、四臂共享同一序列，与GT/开发结果无关。所有flip是可逆的精确离散变换。

**保留原CE和Dice定义：** 对每视图v，CE按Theta样本逐个算再平均；Dice在该视图两Theta平均概率上计算，前景GT存在类等权/epsilon1e−6/all背景Dice0，缺类由CE处理。不是逐Theta Dice平均，也不是平均概率CE。两视图监督项再取1/2平均。G均值罚为原.5*(||mu_c||²+||mu_R||²)/128²，按两视图取平均，不翻倍。

将第二视图输出逆relative flip回第一坐标。每Theta分别计算4类分类概率JS：.5 KL(pa||m)+.5 KL(pb||m)，m=(pa+pb)/2。按训练GT中每个实际存在类别（包括BG）的像素区域均值，再存在类等权、图像等权、Theta等权。缺类跳过，0log0以clamp1e−30实现；数值JS最小不得低于−2e−6，容差内负舍入如实记录，不临时改容差或加伪标签。该类别均衡仅作用JS，不改原监督CE/Dice。

总loss=.5*(sup_a+sup_b)+.5*(reg_a+reg_b)+lambda(t)*JS；EQ的lambda=.1*min(step/500,1)，2V为0。不扫系数或按开发集延长。lambda不是已知最优/理论KL，整个目标是监督加一致性，不称原ELBO或Bayesian增量。

记录每步CE/Dice/均值罚/JS/lambda/总loss/裁剪前梯度范数。两EQ臂各step1/1000/2000/3000/4000/5000另对base与加权JS各取一次autograd.grad，12个探针/24额外反向请求、0额外更新，保留范数比；不据此自动调lambda。对照JS=0与EQ主前向相同，梯度诊断额外开销单列，不称墙钟/算力完全相同。

## 封存评价与完整预算

四臂主读出均为单视图identity；U直接概率，G局部均值+G权重均值，0Theta/local，不加TTA。各完整train20相位+val46相位，264份identity native128 FP32。额外每臂val固定三个翻转各一次并逆变换，保存552份完整aligned native128 FP32，仅评价空间一致性，不平均成新分割、不GT选视图或权重。816全部预测和前向账本先封存，label0，才本轮标签评分；不再重复上轮未保存逐视图概率的局限。原GPU deterministic双线性恢复384归一化/processed_spacing/EDES→类→患者，原NA和空预测HD95惩罚不变。

正式训练：20000更新/主反向，80000编码图像、20000U批量logit、40000G/80000图像G、20000Theta，local0；额外24梯度请求。24完整modelAdamRNG（各0/1000/...5000）。评价9624编码图像、2464Ulogit、2464G/4812图像G，Theta/local0；264主预测+552aligned。所有文件FP32不量化，主缓存理论811597824 bytes，额外翻转1711276032 bytes，合计2522873856 bytes；含完整状态/源码/账本预计<3.0GB，启动预留4GB，保存期100MB余量，原链累计+4GB及新增路径+4GB都须<45GB，不删除历史腾空间。

原始评分264概率恢复，按相位共享GT只66读取；完整所有臂三类Dice/HD95/NA/空预测、NLL/Brier及自然权重类别条件/贡献风险均报。val每相位从同4份主预测构造GT×2V预测×EQ预测4³迁移，两管线全部46相位/三类，记录TP/FP/lost/gained、无TP但FP消失与真正TP丢失。新增真实RV失败定义为controlTP>0且EQTP==0，**不要求EQ预测完全为空**。

温度四臂20fits<=1280目标调用，沿用旧PRIVATE_FOLDS患者排序mod5/stride64/phase等权/logT[-3,3]/bounded xatol1e−5 maxiter64。fit排除评分患者，所有T封存；每次输出manifest先于该次heldout标签重读，argmax必须0。184校准概率恢复，按相位共享GT46读取，总112原始标签读取；raw评分已可见开发标签，不称完全未见/独立test。温度优化单列，不叫完全0优化。

## 主终点、分流和停止

主终点Delta_G=患者Dice(G_EQ−G_2V)；次要Delta_U、I=Delta_G−Delta_U、S=G_EQ−U_EQ及S0=G_2V−U_2V分别报告。固定23患者配对bootstrap10000seed163；多比较探索，不以患者×臂/切片/MC扩样本，CI跨0非等效非劣。完整训练和开发/所有类及风险都报，不择优抹负结果。空间主统计native128 TV和argmax disagreement，三flip/EDES平均到患者；一致性降低不等结构恢复。

工程晋级门槛同时满足：Delta_G点>=.005、其患者CI下界>0、G无新增controlTP>0→EQTP0的RV相位、I点>=0（即S不低于S0）。只支持进入另立重复验证协议，不是优越/非劣/临床证据，不自动展开新seed/SSLCL。点升但CI未决记未决；空间一致性升而MYO/RV召回降判当前约束不合格，不追加lambda。U/G同程度改善只称通用训练改进；若G仍无分割/风险/效率可信优势，则收口固定接口微调分支，不无限补洞。

独立审计：20000连续账本/24全状态、原初态model/RNG/空Adam与原schedule、实际目标2e−6容差、G/unused/800方差行与Adam矩、816封存账本和全部缓存；264主概率恢复/552aligned原生读取，按相位共享66GT，重算全部分割风险和184温度变换/20fit目标（不refit），迁移histogramdd+布尔计数独立验证，指标误差<1e−10。审计不重跑网络，不冒称独立网络复演。SOURCE/EXIT/无FAILURE及worker/guard退休联合通过后自动PUBLIC_RESULTS匿名，再中文报告源码协议push核remoteSHA/匿名200。

## 准入与故障

合成检查双翻转逆变换、JS同分布0/对称/有限梯度、缺类/全背景规则；relative identity且双输入相同时，新base损失和梯度必须回放原point_loss/原U监督<=2e−6，共用Theta值不变/两个样本只draw两次、方差行grad0、GgradNone。0真实图像标签/0optimizer，不运行真实预实验挑lambda。所有合成encoder/G/反向请求如实记账。

启动需原soft截止前剩余>=7200秒，并核实际挂载/容量/真实读写/GPU余量/create-only和精确源。身份绑定独立中性guard，原85500秒软保存/86340秒硬停，无需心跳介入；故障保存完整partial/modelAdamRNG/真实成本、准确接续，不从头重跑有效臂。用户最新解除旧18GB不等扩大数据/科学边界/重置24h；既有全部失败、旧限额阻塞与清理回执保留，不追改历史规则或源元数据。

准入实际回执：原监督base与梯度在合成同视图回放G/U均误差0；同Theta复用/JS对称及同分布0/反向有限/方差行grad0/GgradNone均PASS。原准入8encoder10G3Ulogit6Theta6反向请求，计数接入补充4encoder4G2Ulogit2Theta0反向；合计12合成encoder14G5Ulogit8Theta6反向请求，0optimizer/0真实图像GT。实际原U/G起点backbone逐项相同、两源step0/空Adam核验PASS。
