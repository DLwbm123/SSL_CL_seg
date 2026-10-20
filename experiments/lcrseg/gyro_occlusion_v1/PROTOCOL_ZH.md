# GYRO occlusion v1：只去掉目标几何强视图中的Cutout

用户12小时自主授权内的独立有界探索：仅GPU3，Siemens→Philips，与BayeSeg/CARE-Seg独立，不进后续域。预先公开协议/训练源码并匿名HTTP验证后才启动；所有旧轮负结果和成本不变。授权UTC2026-10-20 03:31:21截止后不派新worker、不强杀健康固定作业。

## 依据、假设及边界

source_coordinate_v1已完成并一次审计：源phi1000更新保持S† native/frame不变，G旧78.8271732/新81.8802303%，M旧78.6423057/新82.0544228%。G较原强固定phi旧+.4670168/新-.1246978pp，G-M旧+.1848675/新-.1741925，coordinate/frame/practical三门槛均失败。源phi准备没有解除新域拟合代价（G目标13L训练84.6930073%，较Dense-2.1562336pp）。不再继续降低几何权重、更强beta或更多视图序列。

现有strong_view源码在光度增强后对半宽方块做Cutout，边缘裁切后遮挡约图像6.25%至25%；原GYRO质心把完整图与遮挡图作为同一实例视图。不假定遮挡实际命中器官，不额外读取GT定位。本轮检验这项遮挡不变性是否对医学分割适应造成代价；该假设尚未证明，光度增强自身也不保证语义正确。仅取消目标几何视图中的Cutout写入，不改原GYRO公式、架构或增加损失/模块/伪类参照/读出绑定/KD/历史CE。

## 唯一变量及公平匹配

- 入口复用source_coordinate_v1最终1000步phi/BN，与S†原v1.2固定69步最后学生native/frame组成完全相同入口，不再源准备，不续跑已训练目标模型。source/历史参考只读。
- 原phi整个函数固定（权重requires_grad=False、BN eval/统计固定、无phi optimizer），输入梯度仍流向encoder；目标U-Net编码器/decoder/原头全部原允许参数训练，仅原sigma.weight/grad_update常量固定。
- G beta1/3、gamma.45；同期M beta0、gamma.3，显式rho=.3但实际梯度范数/方向不保证相等。nativeLR.001、SGD Nesterov.9、矩阵wd.0005/vector0、clip1、seed163/K10/D128/阶段固定frame、V2/P24/48几何图不变。
- 每步当前L2 CE+原U14几何+源回放8，同一合法256图源库/目标流/弱视图；光度增强两次操作、操作类型/强度/符号及每图Cutout中心的RNG抽取均原样执行，唯独不把方块覆盖成128。保留中心抽取确保后续图的光度随机数与旧轮精确对应。强视图仍有量化及原光度操作；不把它误称完全无增强或所有像素恒等。
- 原GYRO单位化视图均值质心、所属阶段最近simplex参照、混合质心一次全局核范数均不变；CE仍只用相同当前L弱视图。复用source_coordinate旧G/M作非同期遮挡参照，仅已有聚合结果/患者配对记录，不重评/重审旧权重。两组新的G/M同期采样匹配。不降低对照质量，M也取消相同遮挡并完整报告其变化。
- 目标两组native动量均空；source phi动量不接入。推理仍原native头。原Philips13L/26及原U、Siemens源L10/20训练口径分开，验证Siemens23/46和Philips31/62。原生入口Siemens79.4337143%、Philips73.2043174%已测，直接复用，无新入口评估。原43600源80.3674→S†的-.9337pp单列。

U语义仍由当前真实L CE与共享encoder间接建立，任意frame不命名器官。本轮无noNewU，不宣称U类别语义因果贡献，不把无标签梯度通路/谱改善当分类正确。不是原GYRO医学CL已验证成功，也不是新TMI贡献证明；重复开发验证属探索，不是独立泛化。

## 预算、资格及停止

独立cap4064=两组各32工程+2000正式，所有失败、丢弃、恢复重放全计，不挪旧余额、不扩本轮cap。工程顺序两组各32全部通过并丢弃，再G/M各2000正式。32包括30两例CE学习+1实际非空动量恢复重放+1从干净入口完整新几何目标，门槛不放宽。资格核对复合入口精确、全native允许参数训练、native CE/U梯度非零、phi参数梯度0/全state固定/无phioptimizer、黄金批/RNG与恢复精确、常量/frame/teacher无更新、finite。

零optimizer合成合同：新augmentation开启legacy Cutout时与原strong_view逐值精确相同；关闭后mask及每图光度RNG精确一致，mask外像素精确一致，仅mask内可有不同；原固定phi输入梯度合同通过。实际完整目标检查在合法训练批上执行并计工程更新。不额外访问test/U隐藏GT或选择患者。

NaN/Inf、非法访问、坏恢复、预算/显存/存储硬停相关worker保留现场；有依据最小工程修复另版本并全计成本，不无限盲重试、不强撑NaN。梯度比只诊断、不在线调权/据比值停。成功步latest完整model/EMA/optimizer/RNG/reads，dirty不作恢复点。正式固定最后2000学生，EMA只诊断，不中途验证选best或改增强/步数/门槛。

## 冻结科学门槛

三项均通过才称联合目标通过，G/M完整报告：
1. augmentation：G无Cutout相对source_coordinate_v1 G，有标签训练拟合至少+.5pp、Philips验证至少+.3pp，且Siemens不低超过.5pp。
2. frame：同期G-M Siemens至少+.5pp、Philips不低超过.5pp。
3. practical：G相对原五组Dense两个验证域均不低超过.5pp，目标13L训练不低超过1pp。

M相对旧M变化、两域绝对变化/类LV-MYO-RV/EDES/患者改善下降数与中位数、源10L与目标13L训练、实际更新及梯度完整报告。不只以均值掩盖域代价、不靠对照变差称成功、不改旧轮结论。

## 一次完成审计、空间与交付

全轮COMPLETE且worker退休后一次审计：两组2000/EXIT0、工程64、总4064预留执行及阶段账本一致无在途；两目标共同复合入口精确、空动量、phi/BN及EMA精确保持source_coordinate cal最终值、frame/常量不变、native有实际更新；final/latest model/EMA/optimizer/RNG/reads有限且一致；每步loss权重G.45/M.3、nativeLR.001/无phi组/phi路径0、augmentation日志Cutout=False及RNG匹配；患者EDES与训练身份精确。不重新审计source_coordinate或其他旧轮权重，不新增患者前向，旧cal仅为本轮初始化与复合入口合同只读。

新增root在自主父目录occlusion_v1，预计<.3GB；不复制bank/数据/HDF5/模型、不重复入口评估，不下载模型/论文。所有新权重/日志/cache/temp/code包在root/scratch或root子目录，复用既有环境。SpaceGuard在启动/准备/写前和每50次调用检查，全窗口新增20,000,000,000字节硬cap、18GB保护、2GB原子峰值、100MB小元数据预留。GPU3足够显存可共享，不影响他人，启动前挂载写读及显存检查，主子中性argv并ps/nvidia-smi核对。每半小时监测保持ACTIVE。完成后源码/完整协议/匿名聚合中文报告显式代理push、匿名HTTP验证，remote DELIVERY/results_public及本地收据齐；不公开患者/图像/权重/库/frame/特征/原始日志/配置/第三方全文代码，不建PR或整体推私有历史。
