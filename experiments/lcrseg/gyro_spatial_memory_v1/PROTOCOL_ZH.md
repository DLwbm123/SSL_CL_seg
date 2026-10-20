# GYRO 空间凝缩闭环 v0.1.1 执行冻结

用户于2026-10-20明确授权：工程资格通过后启动A/B/C，再依据预设结果决定D/E；不保留历史原图。

本轮完整方法及全部对照/步数/门槛采用同目录 SPEC_ZH.md、PILOT_PLAN_ZH.md。唯一执行补充：原PIL strong_view前向含量化、直方图运算及detach，不能对生成输入求真实导数。凝缩C1的几何项使用精确原强增强前向，并以 `a(x).detach() + x - x.detach()` 定义单位直通输入Jacobian。该项是预声明的半梯度/直通近似，不声称对离散PIL运算求了精确导数。任务效用的一步native SGD展开仍保留完整二阶图像及mask超梯度；不改查询、真实数据增强前向、目标训练或损失权重。资格须确认两个效用梯度路径均非零以及几何直通梯度非零。所有组相同目标增强前向，不因结果改此定义。

来源阶段Siemens仍是合法当前数据，可流式读取训练L/U来形成压缩记忆；不会复制原图、原mask、历史image bank或逐图特征到本轮目录。进入Philips后，训练进程HDF5白名单仅含Philips L/U；独立评估进程才允许读取两域训练/验证。方法持久记忆只含共享P/G、32个优化空间码及两个固定frame，总额≤8MiB。原始受保护数据集不删除。

源codec正式500及质量门槛、凝缩各256 outer/256 functional更新、目标2000最后学生不变。总医疗调用上限11780，另列既有toy functional3，总11783。凝缩工程可能少于32但全部functional与optimizer更新计成本；失败/恢复重放全部计入，不透支正式成本补资格。失败可修复工程问题，科学学习/质量门槛失败不偷偷延长或调参。

独立实验原始时钟2026-10-20T04:58:04.671858+00:00；23h停止新阶段，23h45m独立watchdog终止本轮剩余受管进程（校验PID出生），24h硬上限。全部新增输出统一新根，47GB保护、49GB硬界，严格小于50GB。独立watchdog不依赖聊天/SSH。唯一原有30分钟心跳原位更新；不创建重复监测。

本轮是探索性dense GYRO变体，不宣称完整i-CTRL复现、U语义效应或TMI新颖性；完整正负结果保留。公开源码/协议/匿名聚合报告，不公开图像、mask、空间码、权重、患者记录、原始日志及第三方全文。

## 2026-10-20 确定性池化工程修复（不改科学协议）

C0已完成256outer+256functional，C1工程首次几何反向在CUDA adaptive_avg_pool2d的确定性限制处失败，仅消耗1次functional临时更新、0次optimizer更新。保留失败checkpoint/异常日志/原账本。原生384×384输入的bottleneck为48×48，8×8 AAP是互不重叠的6×6平均。自定义autograd保持原F.adaptive_avg_pool2d前向逐值不变，反向使用分块广播/36（无原子累加）；完整数学Jacobian不变，二阶反向可微。CUDA确定性开关保持开启，不降为warn_only。零训练测试检查前向、解析伴随、CPU gradcheck/gradgradcheck及CUDA二阶有限。

C1从共同干净入口重做有限工程，原失败1成本不清零；新资格30成本，累计31≤32工程上限。原codec/C0终点、预算、门槛、起始时钟不变，不重跑。控制器仅从C1继续，重新挂接独立watchdog沿用原始截止。该修复也应用所有后续target组的相同空间投影，不改变AAP前向或优化目标。

## 2026-10-20 目标访问白名单路径解析修复

C0/C1已各完成256outer+256functional，且均EXIT0。A工程首次读取合法Philips训练文件时，因白名单按data_root/relative计算、真实Accessor按统一safe_asset(data_root,relative)=data_root/h5/v1/relative解析而误拒；该尝试0optimizer/0functional目标更新，拦截现场保留。白名单现使用与Accessor相同的safe_asset和resolve，不增加任何角色/域权限；目标训练仍仅Philips L/U。零更新测试检查真实L/U读取，并确认所有源Siemens train-L及Philips val图像路径在打开前被拒，禁止内容未读取。修复后只从A工程继续，不重复codec/C0/C1，不改协议、预算或原始时钟。
