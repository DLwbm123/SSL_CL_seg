# 预训练表征与固定encoder持续分割：最后窗口独立完整路线 V3

2026-10-25约02:45北京时间预注册；没有本轮源/目标结果。上一完整小U-Net路线已60080真实opt完成：心脏C80.757835%、眼底C78.263434%，比DC²T数值低7.052165/4.836566pp，旧尾部均失败，眼底新非劣亦失败。该配方关闭，不扫参数锚、历史统计染色或蒸馏权重。新路线结构上改为公开ImageNet预训练ResNet34+高分辨率U-Net decoder，检验源表征能力与持续更新encoder的影响，不称DC²T复现/新颖方法。

## 授权、原钟、资源
仍在同一最后12小时窗口epoch1792856563至1792899763（北京时间10-25 11:42:43截止），准备/下载/全部故障与等待共钟；唯一ssl-12/30min、自主不审核。只pro5000 GPU3原UUID。用户新增实际存储<50GB，三根和新增下载/本地副本合计45GB预停，checkpoint峰值预计<18GB，下载约87MB；不删除历史、不重新计算原时钟、不使用旧图适配。前两轮合计80096optimizer、1275182native（其中32合成）、80096训练反向+448重要性反向（另4合成）；V1/V2审计和负结果全保留，不复跑。

## 数据与预训练
原冻结seed0、20%患者L/80%U、同一只读data_view（去test/U标签/额外GE25）。心脏4域Siemens/Philips/GE/Canon、眼底3域REFUGE/RIM_ONE_r3/Drishti_GS；缺第二REFUGE相机，不与论文同协议对比。始终完整384，不改HDF5/裁剪/标签/清单，不用验证选训练批或checkpoint。只在各站点作为当前站点时读取其trainL/trainU；推理旧val由独立固定末评估访问。

TorchVision0.22.1 ResNet34 IMAGENET1K_V1，官方 https://download.pytorch.org/models/resnet34-b627a593.pth；约21.8M参数、83.3MiB。文档 https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.resnet34.html 已核。只用分类预训练encoder，移除fc/avgpool；不下载医学额外数据。3×384输入；MRI将既有[-1,1]映射[0,1]后复制三通道；RGB仍[0,1]；均按ImageNet mean .485/.456/.406、std .229/.224/.225逐通道归一，不套分类224裁剪。encoder ResNet34多尺度64/64/128/256/512，decoder使用已有GroupNorm双卷积/转置上采样skip块512→256→128→64→32，最后上采样至384并16通道卷积、1×1四/三类头。所有encoder BN running stats固定eval，源及B允许其affine梯度，C冻结整个encoder（含affine）；encoder特征保持不等于分割无遗忘保证。

## 单种子、完整源与B/C序列
源单seed20272401，两个数据集分别从同一公开预训练encoder初始化，随机decoder；心脏10000源步、眼底4000源步。L4/U4，同一当前域；源监督CE+Dice（心脏三前景，眼底cup/disc union），同几何当前光度强弱视图各半。当前U均值教师，confidence>=.9，KL全像素分母、1000步ramp至1；EMA.999（C冻结encoder参数直接复制到EMA，不作会舍入漂移的重复衰减）。源阶段额外.1可靠旧源输出蒸馏，只让未改动旧源读取相同当前源L/U（L需GT正确、U需conf>=.7），不迁移小U-Net权重到新结构，不用旧源保护错误L标签。该先验可能限制改进，必须如实报告。

Adam/WD1e-5/clip5/FP32-TF32/阶段余弦至.1。源decoder lr5e-4、encoder lr1e-5；后续B decoder1e-4、encoder2e-6，C decoder1e-4且encoder全冻。B/C各站点4000步，从相同源末EMA分叉；B普通当前L+U均值教师，C在冻结encoder下再加.5前站点冻结模型在相同当前L/U输入上的可靠KL。无历史风格染色、无参数重要性锚/逐步投影/旧图回放/域路由/额外专家。该对照完整检验“更强源表征+稳定encoder+decoder适配”，不是逐参数扫描。

## 源门禁先于目标
源比较参照是上一轮未改变强源保存的warm_reference行（Siemens80.366967769、REFUGE89.900599247），不评估新架构随机头来构造容易的参照、不重复旧源val。固定末源完整val一次。心脏源绝对>=80.067406638%，眼底>=89.600599247%，各自配对CI下界>=−.30pp；>2pp退化=0、变化p10>=−.50pp。不得以源平均提高覆盖尾部失败，不挑中间checkpoint/换种子/加步/放阈值。任一失败SOURCE_STOP该数据集，不能进入适配，另一数据集继续。源训练充分性用固定训练批1500/末前500/末步记录监督+U目标及各分项，光度随机数固定并恢复全局RNG；末两次完整目标规范化变化 abs(b-a)/max(1,abs(a))<=.01且全部有限才PASS，否则SOURCE_STOP，不延长/补点。它只检验该固定批稳定，不声称全局最优，不以稳定代替患者能力门禁。

## 工程与完整调用账
source14000+10target×4000=54000正式opt，每阶段8弃资格（恢复/golden/全目标/固定两例下降），12阶段共96，总54096实际opt硬帽；原窗累计134192（若全完成）。native2,000,000样本硬帽，训练reverse55000（另合成4完整反向、0合成opt），importance0：本轮不测MAS/风格统计。复用lcrseg.engine.Trainer与Schedule；源/目标均L4 U4，所有EMA/legacy/source reference/统计/评估入native账；预期训练样本1,272,000内加资格、固定批描述诊断与评估仍<2M。参数或样本变大不是相同FLOPs，成本完整披露。每50步时钟/全窗空间保护、每1000步原子覆盖latest，只保留initial/latest/final及失败证据。任一工程失败停止该轨迹，原物理调用和时钟保留，不自动重播优化。

完整384、两数据集源/目标生产目标反向/BN状态固定/encoder冻结与无旧参考梯度合成自检必须PASS；每阶段8真实资格必须PASS才正式。encoder被冻结时明确要求其grad=None和参数版本不变，而不是要求所有参数有grad。正向/目标模型/EMA/RNG恢复与固定两例相对下降检查沿既有引擎。所有主/子进程中性入口，独立timeout至原截止前60秒TERM+30KILL，最后120秒不启新阶段。不得通过新名称绕过12h/50GB总窗。

## 结果与停止
源失败不是适配阴性。固定源和每站点末EMA评估全部已见站点，阶段矩阵/每结构/平均/配对CI/旧>1/>2/p10/新C−B非劣全报；患者尾部<=2/0/p10>=−.50及旧CI>=−.30、新NI均值与CI>=−.20、Philips>=83原阈值不变。2pp数值目标85.81/81.10不变，但预训练优势、不同val/三眼底站点、共享旧源/重复开发/单seed必须披露。当前序列完整后才分析改下一路线，不用局部目标val调参。

完成全退休后一次本轮零forward audit、完整匿名报告/源码/成本公开，7897代理remoteSHA/匿名200/localremoteDELIVERY齐。不公开数据/患者行/权重/私有配置日志；ImageNet权重只存私有root，不纳入Git。若本轮仍明显差或源失败，停止继续调本项目，不另开同族超参/步数/冻结范围扫描；到原窗口截止给出明确最终建议并停止唯一监测。
