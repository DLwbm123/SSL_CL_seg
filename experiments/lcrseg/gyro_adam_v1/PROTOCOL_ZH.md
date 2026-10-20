# GYRO Adam v1：固定损失与原生读出的优化器配置检验

用户12小时自主窗口内，Siemens→Philips，GPU3，与BayeSeg/CARE-Seg独立，不进入后续域。公开协议/训练源码并匿名HTTP验证后启动；UTC2026-10-20 03:31:21后不派新worker、不强杀健康固定作业。所有旧轮终点及成本保留。

## 依据与唯一配置包

source_coordinate_v1固定phi G旧78.8271732/新81.8802303%、目标13L训练84.6930073，M78.6423057/82.0544228，三项门槛未过。occlusion_v1去Cutout G78.8913996/81.7902051、目标训练84.3091269，M78.7579104/81.9017786，去遮挡也未改善适应，三项门槛未过。本轮不再扩大beta/V、降低gamma或延长源准备。

已有全文审查记录原GYRO补充Table S2使用Adam，而本医学原生头适配迄今使用Nesterov SGD。检验原方法内一个固定Adam配置是否改善当前几何约束下的保持—适应取舍；不把SGD视为已证明的失败原因。替换native优化器为Adam，lr1e-4、betas(.9,.999)、eps1e-8、amsgrad=False、foreach=False，保留矩阵wd.0005/vector0（coupled L2，不是AdamW）。这是optimizer+标称LR的配置包，不单独归因于二阶矩，不声称实际步长与SGD匹配，不扩Adam学习率搜索序列。

## 入口、损失与匹配

- 同期G/M从同一S†原v1.2最后69步native/frame+source_coordinate cal1000最后phi/BN出发；所有Adam状态为空，不载入source或历史目标动量/矩状态。不续跑旧目标模型、不重新源准备。源/历史参考只读。
- 目标native编码器/decoder/原头所有原允许参数训练，仅原sigma.weight/grad_update常量固定；phi整个函数固定（权重requires_grad False、BN eval/统计固定、无phioptimizer组），phi输入梯度仍传encoder。
- 回到source_coordinate原有Cutout强视图，不沿用occlusion去遮挡改动。原GYRO损失不变：单位化V2视图均值质心、所属阶段最近simplex参照、混合质心一次全局核范数。G beta1/3/gamma.45，M beta0/gamma.3，显式rho=.3，不保证实际梯度方向/范数相等。
- 其余全保持source_coordinate：clip1、原光度增强及Cutout、seed163、K10/D128/frame/预处理、P24=当前L2 CE+原U14几何+源回放8，48视图一次前向，复用同一256合法源图bank、当前采样/弱强视图。推理仍原native头。无新模块、损失、KD、历史标签CE、伪类参照或读出绑定。
- 两组实际LR在配置、optimizer组、更新诊断及引擎日志中均为1e-4。新优化器自带完整clip/预算/更新诊断hooks，闭包引用真实Adam，不能沿用旧SGD opt记录错误LR。source/phi/frame不进Adam。
- 非同期SGD参照G/M均为source_coordinate_v1现有RESULTS和既有患者记录，只读、不重评/重审旧权重。Dense参照原五组。相同开发验证反复使用属探索，非独立泛化。无noNewU，不证明U类别语义贡献或TMI创新；语义仍由真实当前L与共享encoder间接建立，frame不命名器官。

共同原生入口Siemens79.4337143%、Philips73.2043174%直接复用，不新增入口评估或建库；原43600源80.3674→S†的-.9337pp单列。冻结Philips13L/26及原U、Siemens源L10/20；Siemens验证23/46、Philips31/62身份EDES匹配，源/目标训练口径分开。原数据/test/U隐藏GT与源参考不变。

## 预算、资格及硬停止

新独立cap4064=G/M各2000正式+各32工程；失败、丢弃、恢复重放全计，不扩cap、不挪旧余额。两组工程全部通过并丢弃，再按G/M顺序固定2000最后学生；EMA只诊断，不中途验证选best或调LR/权重/步数/门槛。

资格32：30次两例CE学习+1实际非空Adam恢复重放+1从干净入口完整GYRO目标。真实入口、全native训练、native CE/U梯度非零、phi梯度0/全state固定、frame/teacher无更新、黄金批/RNG/恢复精确及finite。两例学习仍须final<first-1e-4，不放宽。实际重放核对Adam所有state键step/exp_avg/exp_avg_sq逐张量相等，全部模型/EMA/optimizer组/RNG/reads精确，不能只检查SGD momentum_buffer。完整目标前恢复空Adam状态；工程权重全部丢弃。单一训练引擎philips_transfer_v1/run.py复用，不复制正式训练循环。

NaN/Inf/非法数据访问/坏恢复/预算/显存/存储硬停相关worker保留现场。有依据最小工程修复另版本并记所有失败/重放，不盲重试、不强撑NaN。成功步latest完整保存model/EMA/Adam状态/RNG/reads，dirty不能作干净恢复点。梯度比只诊断、不动态权重或按比值停。

## 冻结判断

三项均达才称联合目标通过，不靠对照变差：
1. optimizer：G Adam相对SGD source_coordinate G，Philips至少+.3pp、目标13L训练至少+.5pp、Siemens不低超过.5pp。
2. frame：同期G-M旧域至少+.5pp、新域不低超过.5pp。
3. practical：G相对原Dense两验证域均不低超过.5pp，目标13L训练不低超过1pp。

M Adam相对SGD M同时完整报告。两域绝对变化/类LV-MYO-RV/EDES/患者改善下降与中位数、源10L与目标13L训练、实际更新/梯度完整交付，均值不掩盖单域代价。旧阴性结论不因新探索改判。

## 一次完成审计、空间与交付

全轮COMPLETE且workers退休后一次审计G/M各2000 EXIT0、工程64、总4064事件/预留/执行/阶段账本一致无在途；同复合入口、空Adam状态、原native有更新，phi/BN及EMA精确保持cal最终值、frame/常量不变；final/latest模型/EMA/全部Adam状态/组/RNG/reads有限且一致。每参数Adam state键为step/exp_avg/exp_avg_sq，已更新参数step2000；实际optimizer组lr/fixed_lr=1e-4、betas/eps/amsgrad/foreach/weight_decay符合协议，逐步诊断optimizer=Adam、无phi组/phi路径0。患者EDES及训练集合精确。仅复用旧cal作初始化/复合入口合同，不重复旧完成审计或新增患者前向。

新root为自主父目录adam_v1，预计<.4GB，全部新权重/log/cache/temp/code包限制root子目录，cache重定位scratch；复用bank/数据/源/环境，不复制HDF5或下载模型论文，不重复入口评估。SpaceGuard启动/准备/写前及每50次更新检查，全窗口新增20,000,000,000字节硬cap、18GB保护、2GB原子峰值、100MB小元数据预留。只GPU3，显存足够可共享、不干扰他人；启动核对挂载写读/显存及主子中性argv。每半小时监测保持ACTIVE。完整源码/协议/匿名聚合中文报告显式代理push并匿名HTTP验证，remote DELIVERY/results_public和local收据齐；不公开患者/图像/权重/库/frame/特征/原始日志/配置/第三方全文代码，不建PR或整体推私有历史。
