# 空间谱几何与一致性对照 v1（2026-10-20，运行前冻结）

## 范围与假设

用户明确授权S/G00/A-align两个新目标随机流，总更新不超过12256；校准、反向与评价另列预算。旧A/B/C/M/N、五组和源诊断结果/终点不变。不启动预测选点L/U矩阵、第二套源入口、frame/码/生成器重训、SG/T、F或其他条件后续。

两个目标流seed=20261021、20261022，不是独立完整源种子。共同S†为原69步student、共同源训练P、空SGD；不重建或修改旧源入口。所有组模型无generator/code，frame保留为只读兼容buffer但损失不使用。native原允许encoder/decoder/head全训练，sigma/grad_update固定；LR .001/Nesterov .9，矩阵wd .0005/向量0，global clip1。L2当前Philips CE归一化不变。最后student2000为主，EMA .999统一辅助，不中途验证/选best/补步。

|组|辅助训练|额外图像|矩阵|
|---|---|---|---|
|S|无|无|无|
|G00|.1[1−||(2/3)C||*/sqrt(64×32)]|L池仅图像14，标签不读取|同位置三视图48encoder图，64行D32/R4|
|A-align|λ_A mean_i(1−||c_i||²)|与G00严格同图/视图/位置/P|无谱分散，centroid权重仍.5/.25/.25|

各slot采样/两强视图/四象限位置复用已验证stateless规则，源初始化163不改、目标流贯穿所有目标采样。P参数和buffers全固定。PIL增强输入不是可学习量，本轮不使用输入STE；native参数通过encoder正常求导。G00/A-align的encoder暴露匹配，SVD额外计算单独记录，S算力更低，不称三组完全算力匹配。换视图时不额外监督补L图标签。目标训练HDF5仅Philips train-L图像/监督L标签；不构造U accessor，不读历史源HDF5/val/test/U隐藏GT；源权重和P仅共同入口只读使用。

## 一次共同入口校准（0更新）

另设固定校准stream20261024、step0..15共16个当前L池批次，每批同L2+L图14/三视图48；在共同入口共享一次encoder图，分别求G00原始几何与align对全部native encoder/bottleneck参数的梯度：λ_A=.1 sqrt(sum_b||g_G||² / sum_b||g_A||²)。两流共用此一个λ_A，不按流重校准、不用val、不更新参数。

硬停止：任一梯度/损失非finite、两累计梯度能量任一<1e-12、λ_A不在[1e-4,10]，或模型参数/buffers改变则停止；不截断权重、不换批次或放大系数追通过。保留每批梯度范数/累计量及固定λ_A。该匹配只控制入口encoder梯度尺度，不代表全程优化/裁剪/CE方向相同。

## 工程顺序与检查

1. CPU纯特征自检：weighted centroid align与加权pairwise距离等价、两目标梯度finite及slot随机流检查；不涉及医疗模型/更新。
2. 一次共同校准，输出CALIBRATION；未过不进入后续。
3. 断梯度短程：stream20261021，S与S_detach各32更新。从相同入口空opt开始，每步同L2；S_detach执行G00额外前向但stopgrad。逐步要求native参数/buffers、SGD动量、EMA精确相同，输入/标签及全局RNG状态相同。只比较学习状态，不要求读取计数相同；额外14图计数保留。额外几何前向仍须finite。工程终态丢弃。
4. 六个seed×组各最多32实际资格更新，全部通过并丢弃后才进入正式。复用已有原生训练引擎，不另写正式训练loop。共同入口/空opt/固定P与常量/golden loss及梯度精确/非空optimizer恢复检查；固定两例CE30步+一次恢复重放+一次全目标=32。CE下降门槛预先采用first>=1e-6时first−last>=min(1e-4,.01first)，first<1e-6时下降项NA且last<=first+1e-8；不换病例，不因背景/数值下限临时改资格。
5. 固定正式次序：seed20261021 S/G00/A-align，再20261022 S/G00/A-align；每组2000，随后独立进程评价，controller顺序推进。成功每步clean latest含model/EMA/opt/RNG/reads，dirty不可恢复。

## 硬预算、计账与资源

更新总上限12256 optimizer=正式6×2000=12000、资格6×32=192、S/S_detach短程2×32=64；functional/toy=0。工程/失败/恢复重放全计，同阶段不扩cap，不清账。校准不更新，不把求梯度称免费计算。

下表按实际样本暴露与顶层反向调用计硬cap；reserved先持久化、executed后登记、异常pending保留。所有旧共享资源只读，不复制数据/权重；不重复旧模型评价。

|类别|native完整分割图像cap|encoder-only图像cap|顶层反向/求梯度调用cap|
|---|---:|---:|---:|
|校准16批|0|768|32|
|S/S_detach短程|160|1632|64|
|六组资格|576|768|240|
|六组正式（含各自入口preflight）|24300|384600|12024|
|固定步梯度诊断（复用原图，不新增forward）|0|0|60|
|六组终点评价|17352|0|0|

评价单模型固定2892 native切片=student旧val544+新val660+源train230+目标train254+EMA旧val544+新val660，总6×2892；独立合法评价Siemens23/46、Philips31/62、源train10/20、目标train13/26，test/U隐藏GT封存。额外反向只在各正式step=0,499,999,1499,1999记录CE/辅助encoder与decoder/head梯度范数、内积/夹角、合并raw梯度范数、裁剪后实际梯度及真实SGD参数位移；30点×2 autograd=60，不额外forward。S辅助梯度0如实记录。不得增加中间验证或用诊断改训练。

本独立root create-only，原始时钟首次启动即锁定，8h硬界：7h不新阶段、7h45m TERM、8h前KILL；独立timeout覆盖controller及所有子进程，不依赖SSH/Codex。3.5GB保护+.5GB退出余量，新增实际严格<4GB（预计<500MB）；所有新log/cache/tmp/checkpoint/计账/代码在root，另2MB小元数据计入保护。每次forward/反向计账、优化器/保存/新stage检查时间与du；失败保留不盲重试/不重置时钟。仅指定GPU3，启动实际挂载/容量/写读/显存检查，中性父子argv并核ps+nvidia-smi。不删除历史或他人数据。NaN/非法访问/坏恢复/预算/资源任何一项硬停相关阶段并保留现场。

## 预声明分析与停止解释

主比较逐流student报告；G00−S要求两流旧域平均>=+.5pp且每流旧域>0，每流新域>=−.3pp。G00−A-align要求两流旧域平均>=+.3pp且每流旧域>0，每流新域>=−.3pp。均通过才支持优先发展谱结构；仅G00−S通过而谱比较未过，只能称谱结构未证明必要，不能声称已因果确定为纯一致性；复现门槛未过则不扩展本配方。EMA统一另列，不替代student主比较。

患者配对bootstrap10000次seed163逐流报告，均值/中位、改善下降、LV/MYO/RV、ED/ES、源/目标训练与val分开；两个训练流不把患者当训练重复数、不宣称完整源种子泛化。相对共同S†及原源模型的绝对变化均列，接入−.9337pp不能隐去；−.5边缘线不靠四舍五入变通过。

P0仅复用旧五组已生成患者记录，CPU补G00−S/G00−G11的类/EDES/患者损益与留一患者均值min/max；不删患者、不改指标/上述门槛，也不据此调整校准。新六组COMPLETE/EXIT0、受管进程全退役后一次新收据/CPU审计，核原始ledger/forward pending0、校准、断梯度、固定P/buffers/常量、共同入口空opt、finite final=latest/opt/RNG/reads、患者EDES、实际执行组/流和诊断次数；0额外模型forward。生成匿名中文完整报告，选择性代理push源码/协议/聚合/必要审计；远端SHA+匿名HTTP+local/remote DELIVERY/results_public齐才称交付。患者、图像、合成医学图像/mask、码/P/frame/G/分割权重、原始日志、服务器配置及第三方全文不公开。不建PR，不推私有历史。现有唯一30分钟监测跟进，健康静默；条件后续不展开。
