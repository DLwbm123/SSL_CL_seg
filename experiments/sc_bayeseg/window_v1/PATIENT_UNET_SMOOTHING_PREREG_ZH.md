# U-Net+Dice固定标签平滑：简单监督风险对照

2026-10-21，独立假设，运行前冻结。权重均值校准先审计交付；不改旧U-Net、K1、Dμ、MEAN_READ或Mμ终点。已有完全均值模型Mμ经相同患者交叉校准后NLL仍低于旧U-Net，说明“条件G的原始NLL较低”不等于随机分布增量。需补强普通U-Net的简单监督正则基线，检验概率风险改善是否也可由明确的非Bayesian风险改变获得。

## 唯一变量与来源

只新增U_D_LS一臂5000固定主更新，从已审计factor U_D/S0000完整model/RNG/空optimizer开始，freshAdam；不重跑原U_D、不新初始化训练seed。网络仍原482500可更新标量的端到端U-Net/直接头，unused segmentation_head冻结。相同batch2患者均衡/四翻转/schedule5001..10000、Adam3e-4/.9,.999/eps1e-8/wd0/foreachFalse/clip1、FP32 deterministic TF32False/CUBLAS=:4096:8。

只把原CE替换为epsilon=.1的均匀四类label smoothing：target=.9*onehot+.1/4，loss=.9*CE(y)+.1*CE(uniform)+原Dice。Dice定义、GT缺类规则/epsilon1e-6保持。原始CE、uniformCE、smoothedCE及Dice独立记录。epsilon=.1是一个事先固定的监督风险探针，不来自扫描本开发集，也不声称与Dμ隐变量均值惩罚理论等价；本轮不试其他epsilon，不按结果改权重或延长预算。无G/H/F/先验/局部q/教师，不把结果叫Bayesian改进。

## 数据、预测、校准和对照

原canonical320 seed0 Siemens20%L10训练患者230切片，23开发患者46EDES相位544切片；test24/U38/目标封存。训练GT仅loss，66份train/val纯图像native128FP32预测全部封存后再标签评分。原GPU确定性384恢复归一化/processed_spacing/三维EDES→类→患者规则，完整Dice/LV/MYO/RV、NLL/Brier、HD95含NA/空预测惩罚与失败数；不称原扫描网格逆变换已审计。

本轮预先包含一次相同患者五折温度校准，避免仅比原始置信度：旧PRIVATE_FOLDS患者排序index mod5，每fold排除评分患者，stride64/phase等权/logT[-3,3]/scipy bounded xatol1e-5/maxiter64，5标量fits<=320实际函数调用，源final384概率温度变换。全部5温度先封存、每输出源概率+T清单在该次heldout标签重读前封存，argmax必须0变化。原始风险阶段标签已可见，不能冒称完全未触碰数据；fit只用其余患者，不能用当前患者GT选T。无其他温度范围/折/epsilon/阈值扫描。

主终点为校准后val NLL U_D_LS−原U_D；原始NLL及Brier、宏/三类Dice/结构失败为并列完整次要结果。K1/Dμ/MEAN_READ/Mμ原始和校准风险也完整对照，旧八模型40fits/概率/指标全部复用，不重训练、推断、评分或refit。患者配对bootstrap10000seed163，不把患者×配置当独立患者；CI跨0非等效/非劣，多轮开发探索不称确认性或临床证明。保持原U-Net+Dice作为主匹配对照，不择优抹旧失败。

## 固定预算、准入与审计

5000真实optimizer/反向、10000训练编码图像/5000直接logit前向，0G/Theta/local。0/1000/...5000六全modelAdamRNG。评价774编码图像/398直接logit前向。66native预测理论202899456bytes，含六状态/日志/校准预计<.40GB，启动预留.50GB；实际四根保守占用+预留必须<18GB，不为启动删历史。评分66缓存恢复/66标签，val风险与拟合quad复用同次标签；校准另46缓存恢复/46标签，共112恢复/112标签、5fits实际nfev单列非0优化。

合成准入1encoder/1直接logit/1反向/0optimizer/0真实数据，校验手工.9CE+.1uniform与已安装PyTorch label_smoothing=.1一致<=1e-6、目标分解、482500活跃梯度有限、unused头gradNone。不下载依赖。完成独立审计5000真实连续账本/六状态/同U_D初态RNG freshAdam/schedule/unused头冻结/目标系数，66原生预测重算全部结构和概率指标，46raw/cal风险、5fit目标（不refit）/折隔离/manifest/argmax0，审计额外66缓存恢复/66标签/46温度变换/5目标复核、0新网络。AUDIT PASS后audit直接写匿名PUBLIC_RESULTS，完整报告源码协议push验远端SHA+匿名200。

继续原首次clock_start_token87614098/original_started_unix1792485279.2578013及所有失败/占用，独立23h45保存23h59硬停/四根18GB软20GB硬/各服务器新增实际<50GB/原首启含失败续跑<=24h，不拆名重置。挂载容量真实写读/GPU余量/create-only/中性主子argv先核。故障保存partial/model/Adam/RNG和真实成本，准确接续而不盲从头重跑。到固定终点收口，不自动新seed/SSLCL/教师/F/尺度族/test/U/目标，不因简单基线负结果改epsilon追门槛。
