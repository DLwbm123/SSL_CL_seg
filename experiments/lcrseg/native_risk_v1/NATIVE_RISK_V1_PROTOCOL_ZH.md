# 源风险约束的原生解剖等变适配 V1 冻结协议

用户授权一次完整单种子验证；冻结于首次模型调用前。旧外观逆V1 SOURCE_STOP与既有APOS/DP失败全部保留，不复活、不回填。本轮不是延长旧训练或局部机制扫描。

## 唯一B/S/C及源阶段

B复用原已完成FixMatch43600/EMA163原生源（旧23患者Dice80.36740663770262%），没有新B源训练/评估，现有保存行只读复用。S/C共享唯一完整新源F，F从原源初始化训练2000步，非上轮失败F。B目标仅GT；S目标仅GT；C为完整GT+原生解剖+等变+当前L风险约束（唯一主候选）。三目标各2000，同容量残差、同批次/扰动/SGD与时钟口径；B少一份源机制训练成本如实披露，不优化baseline、不抹旧阴性。

源F资格32、源正式2000、固定训练诊断、源完整训练风险复核和源验证全部结束，三类门禁全PASS才允许任何目标基线缓存/资格/正式。通过后B/S/C资格各32全部PASS才三份正式；三份正式与最终当前L风险复核结束才三份旧/新验证。不读局部验证改方法，不挑checkpoint/seed。seed20271801，固定训练诊断seed20271802，不补seed。

## 结构和真实原生路径

原encoder/prehead/真实3×3分类头永久冻结，两阶段均eval模式，输入反传正常开放。唯一可训练模块为prehead16通道残差 h'=h+B3×3(ReLU(A1×1 h))，A16→4随机初始化，B4→16/padding1零初始化；A/B含bias。原生3×3/padding0分类conv与align_corners=True插值完整保持，无1×1近似、域路由或后验替换。初始化残差0严格源函数等价；非零最终适配器正常部署接受能力门禁，不能绕回零分支。

source使用当前已有Siemens train-L10患者/20volume/230slice；target仅Philips当前train-L13患者/26volume/254slice。每batch2不同患者，患者uniform，再patient内volume uniform，再slice uniform。冻结HDF5/manifest/split不修改，无U/test/新增患者/旧图replay。原源权重只读，源/目标期间都不回传原核心；优化器参数集合须精确等于adapter可训练清单。

## 不再盲逆，使用原生概率结构约束

V1固定8bin强度+4×4平滑偏置族仅作同GT坐标外观扰动。M输出bin宽 .125+softmax(v)，v8 U[-1,1]；b=.05(tanh(upsample a)-spatialmean)，a4×4 U[-1,1]；T=B_b(M_v x), B_b(t)=t+b(1−t²)。不裁剪、不几何变形/伪造器官，不输入供体，不宣称覆盖真实MRI物理变化。源/目标与diagnostic使用私有RNG独立stage键；所有对照相同stage/step同样本与扰动。

训练native只对concat(x,Tx)做一次4样本前向。L_GT=普通像素CE+每图存在前景类别Dice平均，两视图均值；空前景Dice0，CE保留，不改macro CE。L_A=.5 sum_{r32,64}mean[JSD(AvgPool_r p,AvgPool_r GT)/ln2]，使用原生概率，不训练新占据率头，不将粗尺度作为部署输出。L_E=mean_pixel JSD(p(x),p(Tx))/ln2，两视图共享GT；JSD=半数两KL，0log0为0，log数值下界1e−8。均值按batch×spatial，类内概率和为1。完整源F和目标C优化L_GT+L_A+L_E+风险Lagrangian，三个系数1。B/S只L_GT，其余公平计算并入账但不贡献参数更新。无盲像素重建、teacher伪GT、参数锚、DS/QP/后验内迭代。

## 源风险约束的精确定义

对每干净slice及存在前景类c，r_slice,c=该GT类像素上的普通CE均值 + 该类one-vs-rest soft Dice loss，Dice smoothing1e−5。训练组g=patient×ED/ES×存在前景类，R_g=该组所有存在类切片r的均值。无支持组在开始时排除，支持固定、不动态重分组。旧患者身份不用于target；组只是训练代理风险，不是验证体积Dice。

原生初始化对源230slice一次前向，缓存私有每slice三标量风险、类别存在、group索引、group初始风险与sampling概率；不存图像、mask、logit或feature。b_g=R_g(0)+.05max(R_g(0),.1)。p_g=(1/Npatient)*(1/Nvolume_of_patient)*(Npresent_slices_in_group/Ndepth_phase)，phase单volume已断言。采样patient无放回对每slot仍是uniform marginal。

Primal每批一次SGD：L=GT+A+E+(1/G)sum_g nu_g(R_g−b_g)，batch无偏估计为sum_{slot,class present}nu_g*(r−b_g)/(batch*p_g*G)。nu是detach buffer，不是optimizer参数；anchor标量不接梯度。

每批基于同一更新前模型的已detach风险进行一次对偶更新 nu_g←clip(nu_g+.01*sum_slot I_g(r−b_g)/(batch*p_g),0,10)，初始0；与primal为同一prestep状态的显式同时更新，保留真实SGD/Nesterov动量，更新后才EMA。无第二参数optimizer、无额外反向或内循环；ν饱和不改帽、不加步。每次计dual_computations，sourceF/targetC另计active_updates，B/S保持ν0。

最终仅一次全源L检查final EMA的每个组R_g−b_g，max<=1e−6才风险可行性PASS。不以SGD执行或低平均风险当约束证明。使用标量缓存不能推出未见旧患者安全。

## 适配缓存与数据隔离

源门禁过后才对当前Philips L建立B原源基线及共享F基线，各254slice一次。S/C共用F的当前L标量基线；target ν全部重置0，禁止继承源patient ν。目标loader仅允许cache/target_B.private.json或target_F.private.json，若传入source.private.json立即拒绝；HDF5只允许当前Philips L。源图片/标签/特征及源风险缓存均不读取；只继承冻结原核心与源F适配器权重。目标B/S仅GT，C使用同式当前L约束。最终三组各254slice当前L风险复核一次；C风险不通过也综合失败，但不追加训练。目标当前L约束不是旧域遗忘保证。

## 优化、资格和三类源门禁

SGD lr.001固定、momentum.9/Nesterov、矩阵WD.0005/向量WD0、全可训练参数clip1，固定EMA.999，每optimizer后更新，finalEMA唯一部署；不改EMA/学习率或选student。每组资格32：真实完整目标反向、golden精确重放、非空optimizer/dual/model/EMA/RNG恢复、单批固定审计目标下降>1e−4、冻结核心/末头、空optimizer初始断言。资格丢弃，不重跑。单批下降用固定目标，避免对偶变化造成假比较。

源1500/1750/2000对同固定8训练批与扰动测student和EMA分项，额外数据读取24batch。固定审计目标=GT+A+E+P；P为importance-weighted per-slice positive(r−b_g)平均，是组正违约的上界代理，不是最终全组可行性检查。审计目标不含随时间变动ν，避免把乘子变化误作稳定。两相邻abs(b−a)/max(1,abs(a))<=.01才训练充分性PASS，完整分项和EMA差披露；不追加步/放阈值。

源能力：F相对同原源旧23患者均值差>=−.20pp、患者配对95%CI下界>=−.30pp、下降>1pp人数<=2、>2pp=0、p10>=−.50pp，全通过。再加前述训练充分性、全源L风险可行性，三门禁全PASS才目标。任一失败SOURCE_STOP；依旧完整源结果交付，不能当适配性能阴性。B是已完成原源复用，不伪称新B训练充分性PASS。

## 全部调用预算

|阶段|optimizer|native样本前向预期|reverse预期|
|---|---:|---:|---:|
|源风险基线+最终全L|0|460|0|
|源资格32|32|148(cap160)|34(cap40)|
|正式源F|2000|8000|2012(cap2024)|
|源固定训练诊断|0|192|0|
|源val|0|544|0|
|目标风险基线B/F|0|508|0|
|目标三份资格|96|444(cap480)|102(cap120)|
|目标三份正式|6000|24000|6036(cap6072)|
|目标最终全L风险三份|0|762|0|
|目标val三份|0|3612|0|
|总计|8128|38670，上限39000|8184，上限8256|

正式每组0/999/1999同图GT/A/E/risk四次autograd.grad，48额外reverse计入；不会增加forward。所有调用事前记账；optimizer执行前预留、成功后提交，pending不是提交。资格同样保留重放真实调用。sourceF/targetC共4064次活跃dual更新；全部8128次dual计算事件，B/S不变ν0。独立cap：adapter39000、transform18000、spline36000、bias18000、anatomy72000、equivariance18000、risk_class120000；每native通过adapter除原生reference例外。risk_class只计前景3类的显式训练风险计算，不混GT CE/Dice同类算术。source_risk_calibration460、target1270独立，不借桶。

合成与医疗分开：native128/reverse64、adapter128/anatomy256/transform128/spline512/bias256/equivariance128/riskclass384/dual事件64；CPU/CUDA JSD/风险反向、无偏组估计、投影边界、target拒绝sourcecache、原生零残差恒等，以及四配置真实384完整反向。无医疗数据/optimizer。代表速度保守预计<20000s才进入医疗资格，运行时仍以原钟硬限制为准。

## 科学主门槛与理论边界

C旧绝对>=80.36740663770262%，新>=83%。对自身新源F和原源旧患者配对CI下界均>=−.30；两种参考>1<=2/>2=0/p10>=−.50且人数不劣B；C−B新均值及CI>=−.20，C−S新同样；C−B旧>=.30，C−S旧>=.15。C当前L风险也须可行。全部条件才单种子综合PASS，不换主终点/挑seed。旧23新31共享源、重复开发患者，ED/ES共同bootstrap10000seed163；不是独立测试或跨seed稳定性证据。

固定源核心+零残差只证明初始函数保持；训练后仍须患者实证。若源训练组约束实测满足，仅得到声明训练代理风险的预算界，不是未见患者Dice界。JSD约束只涉及所选强度干预，不能推出真实扫描仪因果识别/拓扑保证。无旧图适配不提供全旧域无遗忘定理。

## 资源、停止和交付

仅pro5000 GPU3既定UUID，/data/bmw是既有明确例外，新root与中性entry create-only；实际挂载/写读/显存准入后启动。原8h从首次合成运行起不重置；25200不新stage、27900stop、独立timeout28600TERM+30KILL。3.5GB写保护、预留原子保存150MB，含ops2MB严格<4GB；本轮4.05GB保守预留，累计30GB不清零，准入后9,426,325,728B<27GB。禁止删历史，不靠30min心跳兜底。

完成或STOP全worker/controller退休且AUDIT不存在，仅一次本轮audit.py零forward。完整报告/源码/匿名聚合summary/audit/status公开，代理remoteSHA/匿名HTTP200/localremoteDELIVERY/results_public齐再称交付。隐私患者ID/逐组行/图mask/权重/日志/配置不公开，不PR/强推。唯一30min健康静默监测；完整通知后删除等待用户，不自动后继。不扫描rank/λ/强度/EMA/头/迭代，不因失败追加预算。
