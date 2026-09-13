# 01｜五框架代码实现规格

版本 V1；状态 **待 Codex 集成、外部代码审阅先行**。本文件中的“必须”约束工程语义，不是要求每个实验结果都优于基线。文献提供损失/思想，下面的组合与参数化是本研究设计，不能当作原论文已验证结论。

## 1. 父实现绑定：只绑定实际机制，不依赖方法昵称

从 `operator_inputs.template.json` 与现有项目入口解析：完整源commit、有效配置、网络、读出、A/B乘法顺序与形状、rank、可训练/冻结白名单、正交算子、阶段合并/冻结规则、优化器及调度、训练/推理域信息权限、跨阶段状态。

输出 `review/PARENT_BINDING.json`。实际张量决定哪个是输入侧/输出侧，不靠变量名。本文统一 ΔW=BA。父方法软正交则保留软约束，硬投影则保留硬投影。冻结B、只训练A与“A-only约束”不得混淆。

只有一条原始运行命令或一个原始run_id，也应沿该线索解析，不要求同时提供历史SOTA结果。不要再遍历全部历史档案。若仍无法绑定：继续独立五框架模块、合成测试与审阅包；把真实训练接口标为 `PARENT_BINDING_REQUIRED` 并报告一个具体缺项。**不得把 F_CONV、后来的SVD实验或自行移植O-LoRA冒充原始KI；不得由Codex自行换父方法。**

阅读既有 AGENTS/METHOD_SPEC/IMPLEMENTATION_CONTRACT；新授权以本轮任务边界为准，旧科学结果与终态不改。创建独立包，不覆盖旧方法源码/实验配置/结果文件。

## 2. 建议目录与接口

```text
experiments/lcrseg/five_frameworks_v1/
  contracts.py                  # ParentContract/DataScope/ResolvedStudy/权限
  parent_bridge.py              # 唯一接入真实父模型的适配层
  model.py                      # 原始features与native_readout，不复制一套UNet
  subspace.py                   # Q、累计线性参数F、R与阶段转移
  losses/{jml,kl,cwmi,convex,swd}.py
  reliability.py                # 当前teacher PAS/熵/原型
  recipes.py                    # LCTX/U互补混合/来源恢复
  gradients.py                  # L/U梯度合成与F3自定义反向
  frameworks/{f1,f2,f3,f4,f5}.py
  registry.py                   # 5候选+5基线+后续消融
  telemetry.py                  # 实际算子/数据/梯度/显存计数
  checkpoint.py                 # 精确恢复、原子保存、阶段封存
  planner.py                    # 从冻结配置展开所有任务
  train_stage.py                # 只训练当前域
  evaluate.py                   # 进程隔离的val评价
  controller.py                 # 有限DAG；审批后才可提交GPU任务
  analyze.py                    # 配对统计、排序、不强迫正结论
  cli.py
experiments/lcrseg/tests/five_frameworks_v1/
experiments/lcrseg/docs/five_frameworks_v1/review/
```

不可在生产逻辑中使用 `unittest.mock.patch`、全局monkeypatch或`import *`来偷换父函数。桥接层可以调用已核验的现有纯函数；必须明确参数、返回值和对象身份。

ParentBridge至少提供：
- `features(x, mode) -> h`：返回实际读出前、**最后一个非线性之后**的B×d×Hf×Wf特征；L路径保留对父允许参数的梯度。
- `native_readout(h, output_shape, mode) -> logits`：严格保留卷积、bias、padding、插值、align_corners及随机分类头语义。
- `effective_readout_kernel_at_entry()`：给Q的算子定义与来源；不能猜测随机分类头的均值读出就等于实际路径。
- `parameter_groups() -> input_factors/output_factors/other_allowed/frozen`。
- `stage_entry()/stage_exit()`、父更新约束、完全原生关闭模式。

若父读出不是可核验的线性卷积/仿射映射，先实现显式桥接并在审阅中报告；**不能把非线性换成线性以求方便**。与所需feature-sidecar不兼容时是结构修订项，不是整项性能失败。

## 3. 数据和共同训练配方

### 3.1 数据权限

默认 Fundus，固定标签语义：0 background，1 optic_disc_rim，2 optic_cup，255 ignore；真实manifest需验证。目标训练器只持有当前域 L图像+L标签、当前域U图像+geometry的读取能力。U loader不能打开标签再丢弃。禁止旧训练图像/未来图像/旧GT/hidden-U-GT/test；val仅隔离evaluator可读。

离线开发可在一整条轨迹封存后使用旧域val汇总选全局配置，这是**开发阶段选择**。在线阶段不能用旧域评分调整任何超参数或选checkpoint。不得混称为完全没有离线验证反馈。

### 3.2 LCTX

当前L batch两名不同患者，donor为batch反转。互补矩形与噪声沿用可绑定父配方，默认边长2/3、noise sd=.02。先两个学生前向，再按原始锚来源收集log-softmax。只对锚来源做原CE+Dice，不追加clean-L loss，不额外监督donor部分；另一患者以自身锚身份接受监督。

真实监督前景Dice保持父实现的图像/类别归约、ignore语义、精度与smooth，不因新框架任意重写。F2额外结构项在**来源恢复**后的概率图上计算。

### 3.3 U学习与共同曝光

所有U方法在同一当前U顺序上构造互补UL视图。L图像只作为U的上下文，这个额外对中的L区域**不再产生第二份GT loss**。教师在原图/弱视图U预测；学生两张混合图输出收集回同一U来源坐标。KL/JML在概率或log概率上可微，标签只用于分析时argmax。

LCTX的L/L路径与U/UL路径分开，避免增加U时悄悄替换原监督输入。每步一次优化器更新；多前向/多backward必须单独记账。U batch末尾重复来源应按唯一来源权重，不能重复放大U loss。

无U基线不得读U图像，连“为了匹配曝光的无效读取”也不做。计步可以用已知元数据样本数，不需要读U。不同方法前向成本不同，严格报告，不能宣称等step就是等FLOPs。

随机key包含 study/optimization_seed/order/stage/step/stream，不包含arm名称来人为改变共享增强；新增F2/F4/SWD随机流使用独立namespace，不能消耗共享stream。两顺序各自有自己的随机流，配对方法共享相同对应stream。

### 3.4 EMA与PAS

训练至多学生+**当前阶段EMA**两份完整模型；首域/域边界从新学生重建当前EMA，不留旧教师做蒸馏。EMA只用于目标，本轮部署final学生。原始EMA语义若有实质冲突，在审阅前显式修订，不静默延续历史教师。

PAS-inspired默认：教师端 confidence > .7，当前原型cosine > .5；不是JASCL整体复制，也不增加历史原型回放。原型由同一个teacher在当前L真实标签下的特征构建：每图每类先平均，再按有该类的图像平均、归一化；当前阶段EMA原型衰减.9。未出现类不造原型，退化为confidence-only，逐类标记missing_support。当前L teacher前向/原型更新成本记录。

原型使用teacher预测时对应的固定命名坐标（建议native readout输入）；与输出标签几何不同时，仅通过已核验映射建立对应，禁止盲目resize后称严格像素对齐。每步使用旧缓存进行U判定，在成功更新后提交本步已读L teacher的原型统计；resume恢复该顺序。首次初始化只用当前L。

q、mask、相似度与F3不确定性均stop-gradient。teacher-only admission是本研究变体，不称原PAS逐项复现。禁止学生必须高置信的额外门槛静默进入U。

## 4. 投影的可实现定义与跨阶段参数记忆

### 4.1 必须修正早期草图中的坐标混用

此前 `ΔW=(B+QR)A` 只有在该层B的输出坐标与Q完全一致时才合法。现有UNet末卷积之后还有归一化/非线性，**不能拿分类器前Q直接左乘这些卷积B**。

V1统一采用显式、坐标正确的feature-sidecar：

- h：native读出前特征。
- F_prev∈R^(d×d)：上一阶段已经学到的累计线性变换，当前阶段冻结，第一阶段为I。
- 当前G=I+Q R Qᵀ，R∈R^(k×k)，初始化0。
- 真正送入native读出的是 **F_prev G h**。

这保留父KI低秩更新、冻结读出原参数，不在非线性之间做虚假合并。额外F是声明的新模型参数记忆，不是父方法原来就有的状态。必须与dense-G等容量/相近容量控制比较、计参数与推理成本；不得隐藏新增d²标量。

### 4.2 Q的通道敏感性近似

native kernel切片为C_delta∈R^(K×d)，有效入口读出 C_eff,delta=C_delta F_prev。去掉公共logit方向 Π=I−11ᵀ/K：

H=Σ_delta C_eff,deltaᵀ Π C_eff,delta；Q=TopEig_k(H)。

在CPU FP64求解，保存所选Q、谱、rank、能量与cut-gap；只做一次/阶段，当前学习时Q冻结。重复特征值只要求投影矩阵一致/子空间角合理，不要求基向量逐元素一致。rank-ratio落到 `max(1,min(d-1,round(d*ratio)))`；若d<2须报结构不兼容。不得把rank自动提高到全空间来隐藏无效果投影。零谱/很小gap是诊断，不默认科研失败；零谱使用命名的deterministic legal basis并标注proxy_degenerate，不能说找到有效golden方向。

这是卷积的**通道代理**，不是完整patch算子行空间或零遗忘定理。偏置不影响这个导数空间，但native前向必须保留bias。若读出在L训练下也会变，Q仍是入口代理，记录偏移，不每步偷偷重算。

### 4.3 阶段合并

结束阶段后：**F_next=F_prev(I+QRQᵀ)**。释放Q/R及当前原型、协方差与优化器；下一阶段重新从当前入口参数提Q。F只有一个d×d矩阵，不随域数堆积。保留父方法本来允许的参数状态；不要声称父状态本身一定常数大小。

部署为一个学生+一个固定线性feature transform（可由桥接层在确实合法时折叠入读出）。不增加测试时teacher、内循环、患者缓存或域路由。全模型fold前后在独立进程验证预测误差；不只验证一张矩阵乘法。

EMA应在可比较的**有效矩阵G**而不是任意因子基坐标上更新。实现可在当前阶段用同Q的R-EMA（线性等价）或dense-G EMA；必须给出等价测试。F_prev固定共享只读，内存字节显式计数。阶段切换时从合并学生重建EMA。

## 5. 五个候选的确切定义

### F1｜Regional Soft Supervision in Readout Subspace

L：CE+Dice，训练父允许参数与R。U：`lambda_U*(KL + lambda_JML*masked_JML1)`，**直接只训练R**，parent features与所有U路径其他参数detach/冻结；这不阻止它们接收L梯度。

JML1：每图每前景类，s=Σm(p+q)，d=Σm|p−q|，loss=2d/(s+d)。双空类定义0，整图无有效像素从图像分母排除。teacher不硬化，mask同时乘p与q，拒绝像素不当背景。对来源恢复图评分；保存rim/cup有效质量和像素数。

JML与KL各自归约后再组合；不以softDice代替JML，不把negative entropy额外加入。增加区域信号与限制更新路径是两个因素，后续消融分开。

Q/R作用只在明确feature节点，不扩散到十四层的所有B。代码可保留同节点合法ΔW式分支供未来审阅，但本轮不同时搜索两套节点。

### F2｜Structure-Guided Plastic Subspace

没有有效的feature-sidecar学习，F固定I（不继承其他框架的F）；仍用父A/B。L：CE+Dice+lambda_structure*scaled_CWMI_struct；U：lambda_U*PAS-KL，默认只直接更新**父输出因子B**，输入因子A接受L+父约束。

阶段入口、优化器/teacher正式初始化前，用8个固定current-L batch校准结构项尺度：在同一h上分别求g_sup、g_struct，取各batch范数比的中位数，clip到[1e−3,1e3]。结构梯度全零时scale=1并显式记录，不称有效结构先验。

再用8个固定current-L clean batch，以该结构损失计算所选effective W的梯度G_i。探针只用于方向选择，不更新模型/Adam/EMA；必须把所有额外forward/VJP/L曝光计入实际成本。不可开真实val用于此校准。

当父输入侧本来是硬投影P_free时，取stack_i(G_i P_free)的top-r右奇异向量，初始化A，并匹配父初始A的F范数，B保持零，使入口函数不变。不显式构造所有d_in²稠密协方差；使用thin-SVD或固定随机sketch并锁定算法，内存不足不能静默换算法。

父是软约束：在同一G_i方向初始化后保留原软正交训练；如需要投影初始化，必须在代码审阅前定义对应父规则，不能自动硬化。

选择层按实际父adapter顺序：最后两层／所有父adapter。只能重置**当前新建的、B为零的adapter**，不能覆盖已经承载历史知识的A/B，也不能在warmup后重新随机化。如果父阶段没有新建零增量adapter，需报告结构不兼容并修订初始化策略，不能假造新adapter冒充原路径。

thin-SVD秩不足时保留父合法初始化并记录explicit_fallback；未得到有效方向不作为训练结果FAIL，但此run不支持“结构方向有效”的主张。

CWMI必须来自论文对应后端的结构部分。后端处理常量/空前景、float64协方差与logdet；不要重新加一次CE。使用完整有效区域，若存在ignore：仅对事前定义的all-valid crop做结构运算（最小128×128，采样不看标签内容），没有有效crop则该结构项为0、该图support=0并记录；不可填零后对全图小波计算再称masked精确等价。默认尺度2、方向4只作为本轮改造初值，先核验官方参数语义后冻结映射。

### F3｜Uncertainty-Spectral Update Allocation

L训练父参数+R；U为KL，通过R学习。U有效集合只用geometry，不沿用F1硬PAS掩码。与F1相比增加了不确定像素参与，这是有意的、必须有匹配覆盖的消融。

u=(1-beta)*H(q)/logK + beta*(1−cos(h_teacher,proto_pred))/2；缺失原型用熵替代该项。u停止梯度、clip[0,1]。没有正熵最小化项，也不是原UnCL完整目标。

s_j为Q对应入口谱，归一化到最大值1：D_vj=[1+kappa*u_v/(s_j+1e−6)]^−1。

在当前feature坐标：z=Qᵀh，a=Rz；**前向保持a不变，反向把∂L_U/∂a_v按D_v逐坐标乘**。不能用R参数的一个全局hook冒充逐位置过滤，也不能把D用于L分支。D来自源图但必须按混合来源映射到各view的feature-grid；不要用未对齐的原图u直接过滤混合视图。feature上的卷积反传已混合邻域信息，此操作不等价于逐输出像素雅可比投影，须如此陈述。

F3是显式自定义优化规则，不声称是某个标量损失的精确梯度。Adam可能部分抵消恒定的坐标缩放，因此D过滤的是梯度而不是最终参数增量；必须记录实际optimizer增量，不能保证最终更新按相同比例减小。用前向恒等测试+手工反向结果测试，不对这个identity-with-custom-backward做普通gradcheck来要求错误的一致性。

后续C_SCALAR消融以每位置mean_j(D_vj)控制同一覆盖；再按需要做最终R梯度范数匹配对照。不得称mean(D)已经精确匹配梯度范数。

### F4｜Anatomy/Subspace-Reachable Teacher Target Repair

L训练父参数+R；U只训练R，loss为PAS-KL到修正后的teacher目标。Q空间与F1相同。

teacher给h_bar与当前有效G_bar、原始q。定义u_bar=G_bar h_bar（位于F_prev之前），q(d)=softmax(native_readout(F_prev(u_bar+Q upsample(d))))。所有teacher参数和读出参数以冻结functional权重使用，**d是唯一内层可微量**。不能外层no_grad包住内层，也不能误把内层梯度加入teacher.grad。

d为B×k×64×64，bilinear升采样到真实feature-grid，再经实际native读出。优化E(d)=KL(q||q(d)) + lambda_shape*scaled_convex_loss(q(d)) + .1*mean(d²)。固定3步；每步detach重新requires_grad，避免二阶图累计。归一化gradient RMS后步长.05，投影到trust_fraction对应feature RMS幅度上限。最终q(d)完全detach。

shape_weight=0或steps=0必须精确退回原teacher目标与零修正，避免KL浮点微梯度经归一化放大。目标形状项缩放用与F2相同的当前L梯度范数规则（8batch），每阶段固定，额外探针记账。

形状只作用于p_disc=p_rim+p_cup和p_cup，不能对rim单独施加凸性；不得加入恒成立的cup≤disc伪约束。使用官方D-Convexity二阶项适配或经核验的指定组合，需源码/公式映射，不得以TV、拉普拉斯平滑、椭圆拟合冒充。有限步不是精确CGPM/全局最优/严格凸性保证。

形状损失在原U坐标计算，不在UL拼接的合成器官上计算。teacher PAS的接受集合由**原始q和原始特征**确定，不能用修正后更尖锐的q重新增加接受数。有效geometry按差分stencil保守侵蚀，不让填充边界制造形状约束。

只在训练目标生成中使用内循环；部署没有修正内循环。记录完整网络前向、readout-only前向、d-VJP及每步目标值。若内层发散，为工程异常停止该配置并修复/记录，不能静默退回q把失败样本删除。

### F5｜Class-Conditional Subspace Feature Alignment

L训练父参数+R，U的KL+lambda_SWD*SWD只更新R。输出KL仍使用共同UL混合来源路径。为了避免把混合接缝当类别特征，本轮**额外增加一次clean/appearance-strong U student前向**计算SWD；成本单列，消融保留相同路径。

在固定入口坐标Q中：z_U=Qᵀ(G_student h_U)；z_L=Qᵀ(G_teacher h_L)。teacher参考stop-gradient，Q为同一阶段常量。每个feature向量L2归一化（eps1e−6），不增加投影MLP。L由真实标签分类；U由原teacher预测类与PAS分类；几何映射核验。

每类选n_c=min(64,n_Lc,n_Uc)个特征，n_c<8该类不计入当步平均；无任何有效类返回graph-connected零。默认只rim/cup，避免背景主导；不把缺失类填零凑数。当前batch内采样，不用历史队列。

32个单位随机方向，按study/seed/order/stage/step/swd_stream锁定。等样本数下每个一维投影排序，计算mean((sort(z_U v)−sort(z_L v))²)，再方向平均、有效类别平均。原论文还有其他目标；这里只是类别条件经验SWD改造，不加uniformity、Gaussian近似或PatchCL损失。

## 6. 梯度与优化器必须准确

R-only U：parent features可在U前向no_grad以节省图；但R和后续固定readout对R的导数必须存在。固定readout权重requires_grad=False仍可对输入反传，不能对整个后段no_grad。

L与U分开构图/求导后按参数组一次合并再调用optimizer.step。未获U权限的参数的**U梯度**为None/0；它们仍可因L梯度、已有Adam动量或weight decay发生整体更新。不可用“总step后参数没变”错误测试隔离的U分量。

F2对A的U分量屏蔽不能只在模型forward detach一处却留下其他路径；对所有parent参数白名单逐项审计。父正交项只加一次。EMA仅成功optimizer更新后一次；AMP skip时不推进科学step、EMA或数据游标。

lr、weight_decay、Adam/SGD选择来自parent contract。PARENT_TUNING中的倍率是相对值。不要因参考代码方便静默切换AdamW。不同参数化同weight_decay不等于相同函数正则。F/R和A/B的actual lr与有效更新范数都记录。

## 7. Checkpoint与恢复

保存student完整状态（含F_prev/Q/R/A/B）、EMA、optimizer/scheduler/GradScaler、当前原型support、所有随机流、sampler游标、epoch与successful_step、F2尺度/方向探针完成状态、F4缩放、运算账本。目录先写临时文件，fsync+atomic rename，再写receipt。

推荐每步轻量持久化log，完整checkpoint固定间隔+阶段尾。崩溃后未进入checkpoint的已执行tail若重跑，计入physical_retry_updates，不偷偷改写原日志；科学轨迹以已恢复状态对应的一条序列定义。不得把持久化日志当成缺失模型状态。

DAG中第二目标域只能从同一family/config/seed/order的第一目标域封存学生开始。不得复用其他臂warmup、EMA或optimizer。首域common可以按来源证明共享；不同source种子不能混用。

## 8. 后端、依赖与源码

CWMI、D-Convexity必须由Codex读取作者版本并固定commit/文件sha/license、接口和修改说明。没有license时不大段vendoring到公开repo；优先记录外部源码安装与哈希，或经许可做独立实现并核验公式。许可证不清楚不妨碍其他四个框架继续代码实现，但相关backend不得伪装READY。

禁止用FFT-L2冒充CWMI、softDice冒充JML、TV冒充D-Convexity、均值MSE冒充SWD。不支持依赖时抛出清楚的MissingBackend异常，不能返回零loss。

`reference_impl`只是数学与接口参考，不能把玩具shape_loss测试当作D-Convexity验收。真实parent集成、backend parity、GPU测试与恢复测试由Codex完成。

## 9. CLI与审批

必须提供：
- `python -m ...cli inspect --metadata-only`
- `... plan --emit ...`（无真实数据）
- `... qualify --synthetic --device cpu`
- `... run --approval review_approval.json --phases B,C,D`
- `... report --run-root ...`（读取已完成记录，不新增训练）

run在第一次数据payload/ckpt tensor打开之前核对外部审批、用户启动确认、代码指纹/commit、parent binding、依赖lock、完整计划和预算。模板、空hash、代码改动、超范围phase均拒绝。审批是工作流校验不是安全签名；Codex不能自填“reviewer approved”。

所有代码分支默认 `CODE_ONLY`。push之后停在 `STOP_AWAITING_EXTERNAL_CODE_REVIEW`；不要因为CI绿色就自动提交GPU任务。
