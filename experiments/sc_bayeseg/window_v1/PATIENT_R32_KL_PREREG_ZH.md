# R32局部KL恢复及维数归一化诊断（匹配两臂）

2026-10-20，运行前冻结。R32 CE+Dice/beta0已训练88.2608%、val70.8689%；相对R16训练MYO+2.8883pp，val宏+1.8455pp未决。相对U_D val+.1645pp CI[−1.5530,+2.0145]不证明非劣/优势，RV空预测2次、NLL.1053005高于U_D.0841448。此为去KL条件的能力证据，还需检验保留局部KL时是否仍有可用读出。

只新增两臂，每臂固定5000：R32_K1目标E_sample像素CE+前景Dice+(KLc+KLR)/128²；R32_KQ目标同前两项+KLc/128²+KLR/(4×128²)。保持标准Gaussian c/R先验、raw(-7,-1)族、所有G0参数/prior/initial/logsd、R32结构和学习参数数目不变。K1保留新模型各局部维度的完整KL，仍有额外Dice故不称原纯ELBO；KQ对R使用4096/16384=1/4维数比，是诊断性分块KL权重，**不是原模型普通ELBO**，也不等于改变了一个标准Gaussian先验。两个系数由维数规则事先给定，不扫描val选系数。

R从4096增至16384维，在每维KL相同条件下R总KL增4倍；原logsd上界导致R尺度KL/N下确界从.1419169增至.5676676。此事实不能证明梯度压制，减去该常数也不改优化。配对K1/KQ用于分离R维数下的正则强度效应，beta_c始终1，不改c分支或后验尺度；不同时解冻G/改先验/增加数据或训练seed。

两臂都源patient_r32_v1/S0000完整模型/RNG，断言step0、R32、源Adam空，不接S5000或oracle。freshAdam lr3e−4 betas(.9,.999)eps1e−8 wd0 foreachFalse，联合clip1、FP32确定性/无TF32，原患者均衡batch2四翻转/绝对schedule5001..10000，训练2Theta2local/原种子，Dice定义同前（每图GT存在前景类等权epsilon1e−6，全背景图Dice0，GT缺类CE处理）。此处两臂及原R32 beta0形状一致，可共用基础噪声；不同q轨迹的抽样值仍不相同。G冻结，仅UNet特征+H484576活跃参数，unused分割头冻结。每臂6完整model/Adam/RNG共12份，固定终点不按中间val改变臂/权重/步数。

复用R32 beta0、R16 G_D及U_D原数值，不重训/新推断/重评分。仍10训练患者20相位230切片、23开发val46相位544切片，无test24/U38/目标。两臂全部132相位image-only预测封存后本轮才评分标签；原4Theta4local/batch2/seed，native128float32缓存同GPU恢复384、原spacing/三维指标/EDES→患者。报告训练及val全类Dice/NLL/Brier/HD95 NA和失败惩罚、每新臂−R32 beta0及−U_D/−R16 G_D、KQ−K1配对患者bootstrap10000seed163，不只报Dice最高者。val多轮开发不能称未触碰最终测试，区间不覆盖训练seed/MC或保证等效。若KQ较好，只支持权重诊断不称Bayesian收益；若K1不能保留能力，如实报告，不临时删KL或移门槛。

预算恰10000新optimizer/reverse/20000训练编码、20000Theta/40000G/80000图像G；评价1548编码/3184Theta/12736G/24768图像G、132标签phase/132插值恢复（非网络）。12状态+native预测预计<.7GB，预留1.2GB，全部source/tmp/cache/log/model/概率pro5000原GPU2/授权/data/bmw create-only；原R_START首启clock及全部失败/历史成本不重置，<=24h/每服务器新增实际<50GB、独立原23h45保存23h59硬停、四根18GB软20GB硬及新45/49GB保持。挂载容量真实写读/GPU余量/原剩余clock和来源先核，中性主子argv ps+nvidia确认，不删历史/失败/别人数据。

工程准入PASS：K1同噪声与原g_loss(lambda1,beta1)损失及q梯度差0；K1−KQ梯度差等于.75×KLR/N梯度，最大残差2.18e−10。合成12G/6Theta/4q梯度请求/0编码/0optimizer/0真实val或test数组单列。R32前阶段须EXIT0/AUDIT PASS后才启动。

非有限/源G变动/边界或硬预算失败保存partial model/Adam/RNG和真实成本，准确接续不盲重跑。完成独立审计10000连续账本/12状态/源S0/RNG及空Adam/同schedule、实际KLc/R权重和目标、G冻结/R32结构、132封存与384重构指标及配对差、EXIT FAILURE/workerguard退休；中文正负报告/匿名全指标/源码协议代理push核SHA+匿名200，更新唯一sc-bayeseg和交接。结果后自主按证据继续，不直接进入SSL/CL/F或声称完整官方BayeSeg成功。
