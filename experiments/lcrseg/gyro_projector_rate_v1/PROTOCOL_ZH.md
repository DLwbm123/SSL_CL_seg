# 原GYRO降低projector更新速率：同256图库G/M对照

用户12小时自主改进授权内独立轮次，GPU3、Siemens→Philips，与BayeSeg/CARE-Seg独立。先交付gamma0.3结果，再启动；不改任何历史终点，不进入后续域。

## 依据与单变量假设

同256图库gamma0.1→0.3，G的旧域+0.6562pp/新域-.0397pp，M旧+.4699pp/新-.0307pp。谱强度门槛通过，但G旧77.0692%仍低M77.3014%，并低原Dense1.6425pp，frame贡献未证实。G原native累计更新路径3.4178、projector14.5501；M分别3.6796/16.7128。不同参数维数/缩放的路径不能直接比较为因果或“吸收率”，但原phiLR.03是native.001的30倍，值得独立检验训练期projector过快适配的可能。

本轮只把projectorLR.03降为.003（matrix/vector两组一起），nativeLR.001不变。不是冻结projector或主干；原native所有允许参数及phi均训练。保留gamma.3、beta1/3的G和beta0的M，区分一般优化器变化和frame差异。不改谱公式、frame、BN更新、梯度clip或语义监督，不增加损失/模块、历史CE/KD。

## 两组与共同入口

两组G_U/M_U各从原S†源v1.2固定69步最后学生出发，native/projector/BN/frame相同，双动量空；不续跑已训练目标模型，原43600参考只读。复用同256图无标签合法源reservoir和一次入口评估，不重建库、不额外准备患者。Siemens入口79.4337143%/Philips73.2043174%，原源80.3674→S†的-.9337pp单列。

当前Philips原13L+原U、L2 CE/U14几何、同源回放8、P24/V2共48视图一次BN/全局核范数、K10/D128、seed163、增强、原native头/预处理、SGD Nesterov.9、矩阵wd.0005/向量0、clip1不变；每步采样与原256图库轮一致。原sigma.weight/grad_update常量保持固定，主干不冻结。EMA.999仅诊断，固定最后学生为主要终点。

## 调用和工程合同

独立cap4064=两组各2000正式+32工程，所有失败/恢复重放计入，不挪旧余额。工程30次两例CE学习+1次实际非空动量恢复重放+1次干净入口完整目标，权重丢弃。验证原native函数入口精确保持、双动量空、全部允许参数训练、L/U/phi梯度非零、黄金批与恢复精确、frame常量及teacher无更新；同时核对实际optimizer param_groups和每步LR日志，phi固定.003/native.001。单训练引擎复用，不复制训练loop。

NaN/Inf、数据访问、恢复、cap、显存/存储硬停受影响worker并保存现场。工程故障有依据最小修复并另记录所有失败成本，不盲重试或放宽资格。梯度比只诊断，不动态调gamma或据其停。正式每步成功保存latest（model/EMA/opt/RNG/reads），dirty不作为干净恢复点。终点2000，不中途验证择best。

## 冻结探索判断

G低phiLR相对同库gamma.3原phiLR：旧域至少+.5pp、新域不低超过.5pp；G低phiLR相对同期M低phiLR：旧域至少+.5pp、新域不低超过.5pp；G低phiLR相对原Dense两域均不低超过.5pp。任一失败，不称联合目标成立或frame有效。另报告M速率变化、各域绝对变化/LV/MYO/RV/EDES、源10L和目标13L训练、患者配对改善/下降及中位数、梯度范数/夹角和native/phi实际更新路径。

无noNewU组，不证明U类别语义贡献；原phiLR参照非同期，但同入口/同库/采样对应。验证集反复开发属探索，不是独立test或TMI创新证明；不把对照恶化作为算法胜利，不以双域均值掩盖明显代价。

## 磁盘、时间与交付

新子目录projector_rate_v1，预计新增<.3GB，不复制数据/库/模型下载；缓存/临时文件同目录。SpaceGuard每50实际调用及写前检查20GB硬cap/18GB提前保护/2GB原子峰值余量，保留initial/latest/final和必要失败现场、不保留每步历史副本。UTC2026-10-20 03:31:21后不发起新worker，不强杀正常固定作业。

完成后一次审计两组各2000/EXIT0、工程64/总4064无在途、共同入口和空动量、frame/常量、finite final/latest、患者/EDES及worker退休；既有记录聚合无新模型前向，完整源码/协议/匿名中文报告通过代理推送并匿名验证。test/U隐藏GT封存，原数据/源参考不改，不公开患者、图像、权重、库/frame/特征、原始日志/配置或第三方材料。
