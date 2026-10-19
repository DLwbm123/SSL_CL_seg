# GYRO source-coordinate v1：已有projector源阶段准备

本轮处于用户2026-10-19 15:31:21至2026-10-20 03:31:21 UTC自主授权窗口；仅GPU3，与BayeSeg、CARE-Seg独立，Siemens→Philips，不进入后续域。协议/实现公开并匿名可读后才运行。所有历史终点与阴性结果保留。本轮不再搜索更小权重、更大beta或更多视图。

## 依据与单一假设

固定phi强尺度G旧78.3601564/新82.0049281%，弱六倍G旧76.1583814/新82.2901479%，M弱旧76.4621991/新82.3352608%。降低几何尺度恢复目标拟合却损失约2.2pp旧域；弱尺度三门槛均失败。原projector只经历短程源准备。检验在保留S†原生分割函数的条件下，继续原GYRO源阶段projector几何学习是否改善固定投影坐标带来的保持—适应取舍。不把坐标准备不足视为既定原因。

## 三阶段与公平入口

1. calibration：从S†源v1.2固定最后69步学生的完整native/projector/BN/frame加载，绝不随机重置phi。只训练现有projector，native所有参数和buffers及frame按构造只读。复用256张合法源图reservoir，无标签字段、无新HDF5读取/图像复制。每步seed=98163+step确定不放回16图，原弱图+strong_view(step)，32视图一次BN；单位化视图取均值质心，全部源阶段id0，原GYRO beta1/3、gamma1单谱目标。1000固定最后更新，projectorLR.03、原SGD Nesterov.9、矩阵wd.0005/vector0、clip1。没有CE/KD/验证选择。只phi的实际优化器调用照常计成本。source native函数精确不变，所以沿用已测S†两域入口，不新增患者评估。phi/BN允许变化，frame不变。
2. G_U：S† native/frame +同一calibration1000步phi/BN，phi权重/BN固定并从optimizer移除，但输入梯度仍到encoder；native所有原允许参数训练，仅原sigma.weight/grad_update常量固定。beta1/3、gamma.45。
3. M_U：同一复合入口、同一phi固定函数，beta0、gamma.3。G/M显式rho=.3，不能称实际梯度范数/方向匹配。两组native动量为空，不载入source phi动量；目标均固定2000步最后学生，EMA只诊断。

目标阶段其余配置严格保持原强固定phi轮：nativeLR.001、原SGD/Nesterov/wd/clip1、seed163、V2/P24/48viewimages、K10/D128、固定两阶段simplex frames、当前L2 CE+原U14几何+源回放8。原U-Net解码器与原头不变，推理不经过phi。当前Philips13L、原U/Siemens合法训练图库及冻结数据/test/U隐藏GT边界不变。源10L/目标13L训练指标分别报告。几何参照不命名器官，U的语义只通过当前真实L CE与共享编码器间接建立；无noNewU对照，不称U类别语义贡献。

## 预算、工程与硬停止

新独立cap5096=源正式1000+目标正式2×2000+工程3×32=5000+96；失败、丢弃、非空动量恢复重放全部计入。不挪用旧余额、不扩本轮cap。工程先calibration32并丢弃，再源正式1000，随后G/M各工程32与正式2000按顺序；不续跑旧目标模型。原单训练引擎复用，不复制训练循环。

每组工程30次两例目标学习+1实际非空动量恢复重放+1从干净入口完整目标。源两例是原无锚谱学习资格，正式完整目标为源beta1/3；源参数/函数入口/黄金批/恢复精确，native梯度0、phi非零且原native/frame精确不变，源L/U标签读取计数均0。目标资格为native CE/U梯度非零、phi参数梯度0/BN不变，无phioptimizer组，原native所有允许参数训练，入口与黄金批/恢复精确。不得放宽两例学习门槛。

NaN/Inf、非法访问、坏恢复、预算、显存/存储问题硬停相关worker保留现场。有依据最小工程修复另版本，所有失败成本记账，不盲重试、不强撑NaN。梯度比只诊断、不动态权重、不据比值停。成功步latest完整保存model/EMA/optimizer/RNG/reads；异常dirty不能作干净恢复点。截止后不派新worker，不强杀正常固定作业。

## 冻结判断与审计

固定最后学生。三项均达才称本轮联合目标通过：
- coordinate：G新phi相对既有强固定phi G，Siemens至少+.5pp、Philips不低超过.5pp。
- frame：同期G相对M，Siemens至少+.5pp、Philips不低超过.5pp。
- practical：G相对原五组Dense，两个验证域均不低超过.5pp，目标13L训练不低超过1pp。

M相对旧M也完整报告，不能靠对照变差说成功。旧强G/M参考fixed_projector_v1，Dense参考原五组，只读现有结果及患者记录，不重新评估/审计旧模型。非同期参照明确披露。共同原生入口Siemens79.4337143%、Philips73.2043174%；原43600源80.3674→S†的-.9337pp单列。

完成后一次审计：cal1000+32/G2000+32/M2000+32，5096事件/预留/执行/阶段账本一致无在途，所有worker退休；source从S†精确加载、native/frame不变且phi改变；target初始=复合入口、空动量、phi/BN及EMA精确保持新cal最终值、native有实际更新；final/latest有限且model/EMA/optimizer/RNG/reads一致。Siemens23/46、Philips31/62及源10L/20、目标13L/26身份精确。全部正负结果、两域绝对变化/类/EDES/患者配对人数中位数/训练口径及梯度更新完整报告，不以均值掩盖代价。重复开发验证是探索，不是独立泛化或TMI创新。

## 存储和交付

新增目录source_coordinate_v1在12小时自主父目录内，预计新增<.4GB。所有权重/日志/cache/temp/code包均在该目录；复用原数据、源、256图bank和已测入口。不下载或复制HDF5/模型。SpaceGuard启动/准备/写前及每50次更新检查，总新增20,000,000,000字节硬cap、18GB提前保护、2GB峰值余量、100MB本地/ops小元数据预留。仅GPU3，先核对挂载写读和足够显存，主子进程中性argv。每半小时监测，健康无变化安静，完成/异常/实质进展通知。源码/完整协议/匿名聚合报告经显式代理公开并匿名HTTP验证，不公开患者/图像/权重/库/frame/特征/原始日志/配置/第三方全文代码，不建PR。
