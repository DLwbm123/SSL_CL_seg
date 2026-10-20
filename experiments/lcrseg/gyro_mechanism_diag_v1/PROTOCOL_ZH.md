# GYRO机制诊断v1：零优化器更新

用户2026-10-20在暂停后明确要求立即执行刚讨论的机制诊断。本轮只解除诊断的暂停，不恢复训练、参数搜索、分组裁剪或旧12小时自主任务；BayeSeg/CARE独立，GPU3，不进入新域或修改原数据/源模型。全部checkpoint只读，零optimizer构造、零backward、零更新。

## 固定对象

选已完成source_coordinate_v1的SGD G_U/M_U固定2000步最后学生，同S† native/frame+源cal1000 phi/BN的初始状态。选择由既有共同入口、匹配G/M与明显保持—适应取舍决定，不按本诊断结果换组。旧Adam/其他轮不重新评估或审计。

## B：encoder—readout四格

encoder包含enc1/enc2/enc3/bottleneck及全部skip features；decoder/readout包括完整decoder及原头。仅交换decoder全state，即构造E1D0和E0D1，原native读取路径/插值保持原样，所有模块eval、requires_grad=False，projector/frame也不变。E0D0和E1D1复用已生成Siemens验证患者记录，不重新前向。新四个混合组在Siemens23患者/46EDES/544切片上评估，共2176 model-slices，批2。沿用原Dice/患者配对聚合，各类及ED/ES完整报告。

描述性比较ΔE=S10-S00、ΔD=S01-S00及交互I=S11-S10-S01+S00，单位pp。混合后协同失配可能产生损伤，不能作为可加、严格因果分解。若某混合格下降>=1pp而另一格非劣-.5pp，可称对应兼容性损伤更明显；不能据此证明其独立生物学原因。若两混合均差而完整终态较好，则报告共同适配补偿，不单独归罪于encoder或head。

## C：同图几何—功能对照

合法源L10患者/20EDES，每体固定取depth//2中心切片（20图，不按GT或当前预测挑图）；源256图bank中按已有库顺序取第一16个a=1的源U条目（16图，仅图像，无隐藏GT）。不复制数据/图库。只用源L合法标注，源U只比较入口与终态预测，不把旧预测当准确标签。源L中心切片Dice与全量源10L训练Dice不同，必须分别标注。EDES不当成独立患者，U库样本不当成独立重复试验。

同36图分别用入口、G终态、M终态：弱图完整native预测一次，共108 segmentation model-slices；再原strong_view(step=10000+固定样本序号)作encoder/projector-only一次，共108 encoder-only model-slices，按原V2单位化投影取均值质心和源阶段0最近锚点。原phi全函数固定，BN eval。统计弱图GAP相对变化、enc1/2/3/bottleneck/local decoder token cosine漂移、投影质心cosine漂移、锚点改变、argmax像素不一致、前景并集不一致；源L另报前景准确率变化、标准化真类margin及中心切片Dice。argmax/Dice不靠单纯提高置信度获益；margin仅辅助。

预设描述性标记：投影质心cosine漂移<=.01且源anchor不变称此图“投影质心稳定”；前景并集argmax不一致>=.1称“输出漂移”。交集只说明在该局部样本观察到投影稳定与输出漂移并存，不证明全部几何稳定，也不意味着输出漂移必然是错误（源L另看真值）。没有满足标记不证明二者等价。角色L/U分开聚合，相关系数仅描述、不作独立样本显著性或seed结论。原始逐图记录仅私有保存，不公开患者/图像/特征。

## 合同、成本与硬停止

预期真实完整分割前向model-slices=2176+108=2284，额外encoder-only=108；总真实encoder-equivalent2392。另9个32×32合成分割model-slices核对三模型自交换精确、手动encoder/skip/decoder路径等于原native路径，不使用患者图。全部前向预留/执行逐次记账，参数/帧/phi在诊断后逐张量不变；没有优化器实例，不算零native更新而忽略其他训练成本，实际任何optimizer更新均为0。旧端点和既有患者记录只读复用，不重复旧完成审计。

合成合同、维度/数据身份、非法GT访问、NaN/Inf、前向预算/存储/显存失败即停保留现场，不盲重跑、不静默追加前向或挑样。成功只输出完整原始诊断结果，不依据结果自动恢复训练/下一轮。每50前向调用及大写前SpaceGuard，预计新增<100MB；整个既有自主窗口新增20,000,000,000字节硬cap继续、18GB保护、2GB峰值、100MB小元数据预留。新产物create-only父目录mechanism_diag_v1_20261020，所有日志/cache/temp/code在其中；不写新权重/图像/特征缓存，不复制HDF5或下载模型。原freeze数据、test/U GT封存。

源码/完整协议/匿名聚合中文报告经显式GitHub代理公开并匿名HTTP验证；不公开患者身份/图像/权重/特征/图库/frame/私有配置/原始日志或第三方全文。实验与自动训练推进仍暂停，半小时任务不因此自动恢复。
