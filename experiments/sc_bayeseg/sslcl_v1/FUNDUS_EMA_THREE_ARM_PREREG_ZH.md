# EMA .999 固定参数三臂完整对照（条件启动）

用户要求：另两个seed结果稳定才继续F0、FG、FH完整对照。此协议在读完seed164/165终点前冻结；启动必须FUNDUS_EMA_MULTISEED_PREREG的全部三seed门通过。未通过则不启动，不降低门槛。

全部采用EMA=.999，其他超参数默认，三个seed163/164/165，各自原源student/EMA/RNG与freshAdam、相同数据序列。F0复用h3 E999和h4 E164/E165，不重复训练。仅新增FG/FH×3seed×5000=30000更新。FG style_mode=wide、history=False；FH style_mode=history、history=True，L/U历史参与和既有概率/幅度不变。无附加调参，无源重训，无B/旧教师/Drishti/test/隐藏GT。

固定S5000学生评估，每新臂140图，共840终点评估；训练240000图像前向。公布全部三臂逐seed旧/新主Dice、cup/disc/rim、HD95、empty、Q10和均值。历史增量门仍为FH−F0旧均≥1pp、新均≥−.5pp，逐seed旧增益>0；逐seed每域cup/disc下降≤2pp、惩罚HD95恶化≤10%、empty不增加、Q10下降≤2pp。特异性FH−FG旧均≥.5pp且各seed旧增益>0；同样列出新域及质量保护，全部保护均须通过才能晋级。任何均值改善不覆盖个别seed严重退化。原默认参数三臂失败保留；本轮只检验新共同参数下的相对增量，不把重复val说成独立证据。

新建pro5000 runs/sslcl_fundus_ema_arms_20261022与ops/x20261022_h5，GPU2/3/4排队。源码不变复用h4引擎，新增资格0模型调用。原时钟1792638157.570414与24h不重置，累计所有前序run/ops/data <50GB，controller49GB85500s、worker85200s、外部timeout86100s；预估新增<.5GB。实际挂载写探针和GPU余量准入，中性argv。唯一30分钟监测；结果完成直接分析和公开匿名源码/协议/结果，不上传私人数据与checkpoint。
