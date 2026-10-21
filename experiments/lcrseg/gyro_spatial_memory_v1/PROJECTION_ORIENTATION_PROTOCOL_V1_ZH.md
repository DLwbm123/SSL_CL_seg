# 源训练投影方向依赖：独立最小辨别协议 v1

状态：FROZEN_FOR_AUTHORIZED_EXECUTION，2026-10-21。用户已明确恢复持续自主实验授权：自主选路线、冻结预算后实际启动，完成分析交付后继续选择下一独立路线，无需逐轮批准。本轮不得追改旧结果，不复跑NT或T配方；本轮失败停止本配方，不等于整个研究自动停止。

## 1. 证据、问题与新对照

旧G相对S在多条目标流可重复改善；最新NT轮G−S旧均+.900675pp、新+.323388pp，而NT−G旧−.320772pp、新−2.802593pp。NT相对NI的新域改善伴随相对NP的旧域代价，三项门槛全失败。旧NT的128维/分母/信息量混杂不能解释G为何有效。

本轮检验：G的收益是否依赖源训练投影与原生128通道的特定输入方向对应。仅三组：S=当前监督；G=CE+.1原P32谱；GR=CE+lambda_R旋转P32谱。两条新目标流seed20261101/20261102，共同源163、S†原69步入口不变。源模型不是独立重复，不称完全无源信息。

P的线性部分W为32×128、偏置b为32。CPU float64固定seed20261104生成128×128标准高斯矩阵，QR后按R对角正号固定Q列符号。仅一次Q，不挑选、不重抽。GR使用W_R=WQ、偏置仍为b。Q正交，W_R W_R^T=W W^T，因此保持投影奇异值和行Gram；改变输入特征通道对应。不得用输出正交旋转作为破坏对照（unit和谱对此不变）。保存私有Q/W_R，公开只聚合误差和范数。不读取新源图像/标签，不做源优化。对齐误差float64<=1e-10、float32行Gram相对误差<=1e-5，W_R与W相对差>=.1，否则工程/辨别停止，不换Q。

G与GR完全同32维、64行、确定性池化、unit eps1e-6、三视图centroid(.5,.25,.25)、R=1-||(2/3)C||*/sqrt(64*32)。保留所有旧native与G公式。GR保留源训练W的奇异谱和b，不能称无源信息。该实验比较固定随机方向对照，不足以证明源语义或原P为全局最优；单Q不代表所有随机投影。

## 2. 数据、训练与工程合同

目标只读Philips train-L：L2监督CE，补L池14仅image；不构造U accessor、不读补图label。三组执行相同48encoder辅助前向、相同两个谱标量，仅启用项接loss；S的辅助图不连接loss。额外前向暴露相同不等于反向FLOPs相同。slot采样/增强/位置沿既有固定独立流。无生成器、码、frame损失、T、历史图bank或在线旧teacher。源/val/test/U隐藏GT不得进入目标训练。只读旧入口与P，不复制数据权重。

原native允许encoder/decoder/head全训练，sigma及grad_update固定；P/Q/W_R/b全固定不进优化器。空SGD lr.001/Nesterov.9/matrixwd.0005/vector0/globalclip1。每组正式2000最后student主终点，EMA .999统一辅助；不挑best、不补步、不以EMA替换学生。每组train退出后独立eval。

一次校准：共同入口，训练stream20261103 step0..15，同16批48encoder，求grad(.1 R_G)与grad(R_GR)对encoder允许参数的平方和，lambda_R=sqrt(E_G/E_GR)。0更新；能量均>=1e-12、lambda_R在[1e-4,10]且参数/buffer/RNG不变才通过。两流共用；失败停、不改批或范围。只匹配入口RMS，非全轨迹尺度匹配；G-GR是预定校准配置对比，不是唯一方向因果隔离。

CPU自检0医疗模型：Q正交、行Gram保留、输入方向实际改变、输出旋转谱不变、梯度有限、slot前缀一致。S最小与S完整辅助各32更新共64，逐步model/opt/EMA/RNG/监督输入exact，全部丢弃。六资格各32：固定两例30步+非空恢复重放1步+全目标1步；golden/backward/恢复检查，门槛first>=1e-6时下降>=min(1e-4,.01first)，低floor非增且标NA，全部丢弃。资格通过才正式。

固定顺序：CPU投影形成→校准→detach→六资格→流01 S/G/GR train-eval→流02同序。正式固定step0/499/999/1499/1999共30点，复用forward记录CE/aux encoder和decoder梯度、内积/余弦、raw/clip、实际SGD位移，额外60autograd单列，不据此调训练。

## 3. 硬预算与原始时钟

总optimizer cap12256=正式12000+资格192+detach64；functional/toy0。任何失败、恢复、重放、临时调用全计原账/原钟，不扩cap。CPU投影形成0模型/0更新。

|阶段|native图像cap|额外encoder图像cap|reverse调用cap|
|---|---:|---:|---:|
|一次校准|0|768|32|
|detach|160|1632|64|
|六资格含golden/恢复|576|1152|240|
|六正式含入口preflight|24300|578000|12024|
|固定诊断|0|0|60|
|六终点评价|17352|0|0|

总native<=42388、额外encoder<=581552、reverse<=12420，generator/teacher0。预期成功实际：cal768encoder32reverse；detach128native1536encoder64reverse；资格480native864encoder210reverse；正式24072native576576encoder12018reverse；诊断60；评价17352。native内encoder不重复算额外encoder。每组评价2892=student旧val544+新val660+源train230+目标train254+EMA旧544+新660；患者角色沿冻结23/31验证、10/13训练，不评分旧模型/混合模型。

reserved先持久化、executed后提交，pending失败保留；更新与调用独立账。未使用cap余量不得追加研究阶段。NaN/非法数据/恢复不一致/未知pending/超资源硬停，保留失败现场，不盲重试。

新create-only root /data/bmw/SSL_CL_seg/runs/gyro_projection_orientation_v1_20261021；仅pro5000/GPU3 UUID GPU-9ea1d4ae-e403-cb3b-29b3-c63b083bf8ff、解释器/data/bmw/envs/x20261014/bin/python；明确/data/bmw存储例外。原始时钟从本root首次实际执行起<=8h，7h不新stage，原钟27899秒TERM+899秒KILL<8h。独立进程组timeout不依赖SSH/Codex。

新增实际严格<4GB：3.5GB写保护+.5GB退出余量，预计<500MB，2MBops计入；新code/log/cache/tmp/checkpoint全部root，旧资源只读。启动真实mount/free/write-read/显存核验；中性父子argv并检查ps/nvidia-smi，GPU共享不干扰他人。唯一30分钟监测只跟进，不替代硬限。

## 4. 预声明结论、患者代价与后续决策

G-S复现：旧域两流均值>=+.5pp、每流>0、新每流>=−.3pp。GR-S使用同门槛判断固定旋转配置是否也有实用改善。G-GR方向依赖门槛：旧均>=+.3pp、每流旧>0、新每流>=−.3pp。各比较报告逐流患者配对bootstrap10000 seed163；点估计门槛与区间证据分别陈述。

若GR-G在两流两域各>=−.3pp，只称该固定随机方向对照点估计非劣，CI下界未达到−.3不称统计非劣。G-GR通过且CI跨0只能称开发门槛证据，不能称源语义必要。GR-S通过而G-GR失败，则学习投影必要性未证明；不得靠重抽Q或调lambda追正结果。G-S未复现不换seed，降低整个机制结论。

相对入口旧域每流>=−.5pp绝对保持单列；原源80.367407→S†79.433714的−.933692pp接入代价仍计。逐组报告患者均值、中位、p10、>.5/1/2pp人数、改善/下降、LV/MYO/RV、EDES、train-vs-val、EMA、梯度/clip/位移。G与GR相对入口旧下降>1pp人数作患者代价对照，不以均值或单一人数阈值掩盖尾部。阈值不声称临床重要性。

全部结束/EXIT0/受管进程退休后只一次新CPU审计0模型前向，核精确12256/calls/入口空opt/冻结P与旋转buffer/finite final=latest含opt-RNG-reads/患者EDES及资源。旧模型不重评，旧终点不改。完整匿名中文报告、源码协议和必要审计选择性代理push，remoteSHA/匿名固定commitHTTP/local+remoteDELIVERY/results_public核验后称交付。

本轮有科学停止不自动扩展成多Q/多权重/维度扫描。依据结果自主选择下一独立且有限的研究问题，另冻协议/预算后推进；持续研究授权不解除数据、设备、旧结果和隐私边界。共享源/仅两目标流/反复开发及多重比较限制保留。公开禁患者图mask、P/Q/W_R/native权重、逐图特征、原日志和服务器配置；所有GitHub连接显式有效代理，不直连回退。
