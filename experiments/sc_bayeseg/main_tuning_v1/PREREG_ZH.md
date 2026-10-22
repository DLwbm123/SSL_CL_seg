# 用户原始Bayesian主线：三任务并行调参第一批

用户明确要求立即调自己的方法、并行三个任务，禁止用U-Net对照优化代替。主模型为sc_bayeseg.model.Model与Method：G/F六个BayesianLayer的权重均值/方差后验、局部q(x,m,ell,c)、规范化联合L/U目标、全局KL及posterior-as-prior；不是sslcl_v1的MAP G/F B分支。没有EMA、CE+Dice附加项、历史风格增强。本批原目标不改，只搜优化学习率。

三任务：M&Ms Siemens（单通道4类）、prostate RUNMC（单通道2类）、Fundus REFUGE（RGB3类）。均冻结已有split0、seed163、当前L/U，1000步源域短筛，各lr3e-5/1e-4/3e-4；9个配置，共9000正式更新+每配置2次golden资格=9018真实更新。原型原源Dice塌缩证据保留；这次优先源域语义可学习性，未达到源域主Dice≥.5且各前景类≥.2不进入跨域保持实验。该门只是可用源模型资格，不是方法成功；零分仍原样报告，不把下降loss称有效分割。

输入适配保持模型层级不变：K类采用K−1参考logit，G输出K−1，F输入K；C通道图像的x/m均C通道，F输出C，各通道独立原Student/Gaussian图像/结构因子，SAR logdet乘C，语义logit因子仍逐类独立，code encoder按(K−1)*16输入。RGB是明确的独立通道生成假设，不冒称无需适配即原公式或只复制灰度；旧单通道4类的默认loss及所有梯度须精确一致。内部观测统一128²，Fundus预测回384评估，不能与此前普通U-Net384分数当只差Bayesian的因果比较。

源训练沿原均匀切片/图像抽样batch L2/U2、无增强，完整原始目标NL/NU比例及单次KL，Adam wd0 clip1，同初始化/RNG/顺序，每任务内部只变lr。每臂2次真实golden恢复检查，G/F均值与方差梯度有限；更新后验证q→prior精确复制，再恢复初态，正式阶段prior保持不变。源阶段尚未实际做CL递推，不能称已验证持续学习。每250步保存可恢复状态；非有限或资格失败不隐式重跑。

评价仅源域val，corrected主输出保留3次局部修正及2×4后验样本，不挑终点。预测后读取GT，隐藏U/test禁用。按既有各任务患者/图像聚合及cup/disc/HD95/empty规则；全部候选公开，最多选每任务一个达到源资格的最高主Dice配置，之后另冻完整源及递推目标对照，不能直接报CL收益。

pro5000新root sc_bayeseg_main_tuning_20261022、ops x20261022_b1。GPU2/3/4/5最多四worker，三任务交错队列，不抢占他人。源数据只读复用，新增预计<1GB，前序会话预算一并保守计入，仍共享1792638157.570414原时钟，24h不重置；controller49GB85500s、worker85200s、外部86100s，小于50GB。一个30分钟监测统一分析推进。此前所有U-Net结果仅对照，相关FG/FH队列未启动且取消。
