# 原GYRO固定已有投影函数：U-Net全参数适应

12小时自主授权内独立轮次，GPU3 Siemens→Philips，与BayeSeg/CARE独立。先交付四视图阴性结果，再启动；不增加视图/锚强度序列，不进入后续域。

## 依据与干预合同

既有V2 gamma.45/beta1/3的G旧77.2298/新82.3091%，M gamma.3旧77.3014/新82.2955%；V4 G旧77.1876/新82.1009%，M旧77.2179/新82.1548%，未达到联合门槛。降低phiLR.03→.003也未改善G旧域，但不能据此断言投影坐标变化无关或是唯一原因。

本轮只固定已有source-calibrated projector函数：其权重requires_grad=False、BN eval并保持running统计，从optimizer移除phi组。U-Net编码器/解码器/原生头所有原允许参数继续训练，只有原sigma/grad_update常量不变；不是冻结主干。投影仍在计算图内，U几何梯度经过固定phi传到native；不detach输入，不新加网络/投影头、伪类参照、读出绑定、KD或历史标签CE。

“固定函数”包含权重与BN状态，不能区分两者的独立影响。它也会改变全局clip1下的梯度竞争、优化器组和有效更新，不能将任何改善单独归为“防吸收”。这属于原GYRO接入配置探索，已偏离原可训练projector设置，不是原论文逐项原样复现或TMI创新。

## 两组与不变项

G beta1/3/gamma.45，M beta0/gamma.3，显式rho均.3，实际梯度范数/方向不保证相同。共同S†源v1.2固定69步最后学生开始，原native和原phi/BN/frame入口值相同、native优化器动量空；不续跑目标模型，不重新校准phi。可训练phi的G参照prefactor_v1、M参照weight_v2，只读既有结果/患者记录，非同期比较披露。

回到V2（原弱+strong_view(step)），nativeLR.001、SGD Nesterov.9/wd/clip1、seed163、原native头/预处理/增强、K10/D128/frame、原Philips13L/原U、L2 CE+U14几何+历史8、P24/48视图一次全局谱项均与V2参照相同。固定phi的BN使用已有运行统计，不再进行本轮批统计归一化/更新，明确作为固定函数合同的一部分。复用只读256图库和一次入口评估，不重建库/增加标签或源准备，不碰test/U隐藏GT、源/历史参考不更新。

## 工程、成本与停止

独立cap4064=两组各32工程+2000正式，含所有失败/恢复重放。工程30两例CE学习+1实际非空动量恢复重放+1干净入口完整目标，全部丢弃。资格核对原生入口精确、全允许native参数训练、CE和U的encoder梯度非零、phi参数/BN不变且无phi optimizer组、黄金批/RNG/恢复精确、frame/teacher不更新及finite。phi参数梯度为0是本轮正确合同，不能沿用可训练phi的非零门槛，也不能伪造其梯度数值；合成合同额外核对固定phi输入梯度非零，外部train()后forward仍保持BN eval。

固定最后2000学生为主，EMA诊断，无中途best。NaN/Inf、访问/恢复/预算/存储/显存故障硬停相关worker保存现场；有依据最小修复另版本全计成本，不盲重试/扩cap。梯度比只诊断，clip1不变，不动态调权。成功步latest完整保存，dirty不作干净恢复点。沿用单训练引擎，只修改参数合同及preflight，不复制训练循环。

## 冻结科学判断

G固定phi相对G可训练phi：旧域至少+.5pp、新域不低超过.5pp；相对同期M固定phi同门槛；相对原Dense两域均不低超过.5pp、目标13L训练不低超过1pp。任一未达不称联合目标成立。M固定phi减可训练phi同报。固定功能可能保留旧域却牺牲新域，必须分别分析，不以均值掩盖代价、不靠对照变坏说成功。

无noNewU，不证明U类别语义收益。报告绝对两域变化/类/EDES/配对改善下降中位数、源10L/目标13L训练、native实际更新/梯度；phi参数和BN精确不变、phi更新路径0。重复开发验证属于探索，不是独立泛化或新颖性证明。

## 存储与交付

自主父目录fixed_projector_v1，预计新增<.3GB，缓存/temp同目录；不复制库/HDF5、不下载模型。SpaceGuard启动/保存前和每50调用检查新增20GB硬cap/18GB保护/2GB原子峰值，只保留必要initial/latest/final/失败现场。UTC2026-10-20 03:31:21后不派新worker，不强杀正常固定作业。

完成一次审计各2000/EXIT0+工程64，总4064无在途，finite final/latest/优化器RNG/reads、入口空动量、phi全state及其EMA精确不变、frame/常量、患者EDES和worker退休。从既有记录配对，无额外模型前向、不重审旧权重。源码/协议/匿名中文报告代理公开并匿名HTTP验证，不公开私有历史、患者/图像/权重/库/frame/特征/原始日志/配置/第三方全文代码。
