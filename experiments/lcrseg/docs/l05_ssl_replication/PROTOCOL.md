# 单前驱持续分割：L05 保底复核 + L05_SSL 缺失因子格实验

## 0. 本文件的用途与边界

用户将本文件交给 Codex 并要求执行后，执行下述固定实验及明确列出的条件续跑；不再只提交草案。本文件不是宣称新方案有效，也不是修改历史实验结论。

优先级：有效性 > 创新性 > SOTA。SCD 投影组件到此停止：不再修求解器、不增加投影版本。保留 L05 作为当前具有单种子性能保护信号的简单基线；新增且仅新增一个配方 L05_SSL，补齐原损失组合中的缺失格。

仓库：DLwbm123/SSL_CL_seg
起点：f2c7bd366cf9bdf80ae52c35798ece6be2e3a60d
新分支：codex/l05-ssl-factorial-replication
建议新增命名空间：
- experiments/lcrseg/l05_ssl_replication/
- experiments/lcrseg/tests/l05_ssl_replication/
- experiments/lcrseg/docs/l05_ssl_replication/

先阅读已有 AGENTS.md、数据角色约束、NAS/GPU 运行约束及以下冻结材料。旧命名空间、报告、锁、源码和模型归档只读，不改 main。并行任务必须使用独立进程、输出和随机流；不得终止别人的任务。

## 1. 已知事实与本次假设

正式来源：
- R1 报告：08148e8411f4a56047373cc013c98760d13d3c8a / experiments/lcrseg/docs/single_teacher_scd_r1/FINAL_REPORT.md
- R1 实际源码：f94be07a8a3db1c062035a9bc2e542af9d4b156b
- R1 engine/config：experiments/lcrseg/single_teacher_scd_r1/{engine.py,config.json}
- 旧公共 stage0 与 S/A/B/C/D 源码：057ce07fa7bd02f8319e0c2bdd9d4adbceff23d8
- 旧数据与数学算子：experiments/lcrseg/single_teacher_scd_v0_1/{data.py,engine.py,objectives.py}
- 公开收据：f2c7bd366cf9bdf80ae52c35798ece6be2e3a60d

已有 seed0 描述性结果：
S: F=.585957181, H=.525668818, N=.704849150
D: F=.718089435, H=.742590645, N=.698910789
U0: F=.676784120, H=.698257901, N=.677446015
L05: F=.620432366, H=.565903763, N=.722308724
L10: F=.571301651, H=.504197862, N=.709716209
E_R1: F=.687037997, H=.704006347, N=.692652881

这里只给阅读近似值；一切门槛从已核验原始 CSV 读取全精度，不能按以上小数抄写硬阈值。

在固定基础 L_sup + .5 K_l^drop 上，已有：
- L05：pseudo CE=off, U-KD=off。
- U0：pseudo CE=off, U-KD=on，系数 .5。
- D：pseudo CE=on, U-KD=on，系数 .5。
- L05_SSL：pseudo CE=on, U-KD=off，是本次唯一缺失格。

D-U0 的正收益只识别了 U-KD 存在时 pseudo CE 的条件作用，不能据此假定 L05_SSL 一定优于 L05。U0 的历史收益伴随当前类别代价，也不能称 U-KD 在所有目标下无效。

本次假设：将历史约束限定到具有真实当前标签的像素，把当前无标注图像用于学生弱到强学习，可能比让旧教师同时约束无标注预测更好地平衡适应与保持。这个假设尚未验证；不得在报告中称为已证明的因果机制或新理论。

## 2. 固定三种配方

### S：监督对照
沿用旧 S 的全部语义和监督 GAS；不读取 U 图像，只使用冻结 U 数量匹配更新预算。

### L05：已入围保底基线，不改变配方
L = L_sup + .5 K_l^drop。
不读取 U 图像；不做 PAS、不估计原型、不做 SSL CE、无 U-KD。
K_l^drop 原样调用旧 D 的标注分支算子：
- p 为同次参与 KL 的学生 logit softmax，detach 后计算掩码；
- q 为冻结前驱教师概率，继承原 q 转換、归一化、EPS_Q 和 KL 方向；
- y 为当前 train_labeled 的真实标签；
- valid 为非 ignore 标签与合法几何交集；
- conflict = valid & [sum_c (p_c-q_c)(p_c-onehot(y)_c) < 0]；
- 只删除冲突 KD，不删除监督 CE，不重新按剩余像素数放大；
- label-KD 系数准确为 .5。

### L05_SSL：唯一新臂
L = L_sup + .5 K_l^drop + lambda_u(epoch) L_pseudoCE。

保留 L05 的所有标注监督与标注 KD 语义；对当前 U 图像复用旧 B/D 的学生弱到强 + 当前 PAS。
- 弱预测：当前学生 posterior-mean，no_grad，产生伪标签和 PAS mask。
- 强预测：当前学生，原 strong() 增强与原随机分类头模式。
- PAS 阈值均严格 > .7；原型仅当前 train_labeled；原型刷新 epoch 按原配置 11,16,...,96。
- lambda_u(e)=.5*min(1,max(0,(e-10)/10))，e 从 1 开始。
- U-CE 分母是原合法图像支持；mask/伪标签全部 detach。
- **前驱教师不得在 U 图像上执行 forward，不构造 q_u，不计算 K_u，U-KD 系数恒为 0。**
- 不能只在日志中把 U-KD 系数写成0，实际 loss 仍为 .5*(K_l+K_u)。
- 不调用 SCD project/solver；不增加 EMA 或第三份完整模型。

把 use_unlabeled_images、use_pseudo_ce、use_unlabeled_kd 明确拆成独立配置项。旧 R1 中 use_u 同时控制 U 前向和 teacher-U-KD；不得直接沿用该耦合开关实现新臂。

主方法推理只有当前学生。没有路由、历史专家库、额外校准头或后处理。

## 3. 数据划分种子与训练随机种子必须分开

本次固定 data_split_seed=0，仍读取 lcrseg_v1_seed0.csv / fundus_seed0.json。
optimization_seed 取 0、1、2，是初始化/训练采样/增强/分类头随机流的种子，不是另一套患者划分。

原因：本次明确检验训练随机性，且不通过切换患者角色混入旧开发暴露。结果只能称固定 split 上的多训练种子复核，不是多划分、外部或未见患者确认。

原代码多个位置硬编码 stable_seed(0,...)：初始化与 forward、batch_indices、load_batch 几何增强、strong 增强。必须通过新适配器统一加入 optimization_seed，保持：
- optimization_seed=0 与原随机映射逐字节一致；
- 同一 optimization_seed、同一 stage/epoch/step/stream 的 S/L05/L05_SSL 共用相同流，不能把 arm_id 放入随机键；
- optimization_seed=1/2 确实产生不同初始化及采样/增强/分类头随机序列；
- PAS/诊断的额外前向不改变监督随机流；
- 原冻结代码不全局改写，单进程单配方，不能产生跨并行任务的 monkeypatch 污染。

所有训练只读取当前域自己的 train_labeled，及新 SSL 臂允许的当前 train_unlabeled 图像。train-U 的隐藏 GT、标签路径及标签hash不进入训练对象。统计 valid geometry 不得用隐藏 GT 得到。

域顺序、类映射、384x384、读取 DATA/h5/v1/relative 及身份hash等保持旧约束。历史阶段 train 数据不回放。

## 4. 已完成结果如何复用

seed0 的公共 stage0、S、L05、D、U0 等完整结果仅用于固定初始化或评价参照：
- 训练 L05_SSL seed0 只可加载已核验公共 stage0 最后学生。
- stage2 教师必须为该 L05_SSL 本次 stage1 最后学生；不可加载旧 D 或 L05 stage1 来拼接。
- 已完成旧对照不正式重训、不改状态。
- 旧 S/L05 baseline 逐阶段、逐类评价从hash绑定的原CSV/报告复用；若需要重新核验预测，只由 evaluator 访问其最终学生，并与原数值相校验，不得训练读取。
- 数据、预处理、度量均不变时复用合法；缺少血缘证据应报告工程阻塞，不能凭相近分数认定复现。

不同 optimization_seed 不共享公共 stage0 权重。seed1和seed2各自从本种子初始化训练一个8000步公共stage0，再供本种子 S/L05/可选L05_SSL共享。

## 5. 训练设置：不另开搜索

保持原正式配方：
- UNet2D + 原官方随机分类头、GroupNorm、原通道与logit插值；全部原可训练权重更新；
- GAS仅来自当前监督 CE 梯度，不来自总loss、KD或U-CE；
- Adam lr=.001, weight_decay=4e-5, betas=(.9,.999), eps=1e-8；每阶段重置；polynomial power=.9；
- 每阶段100 epochs，batch size2；steps/epoch 为80、32、21，对应8000、3200、2100更新；
- 标注/无标注独立循环采样，不改变尾batch规则；
- 原几何和强增强、PAS阈值、KD温度1、EPS_Q均保持；不增加 Dice loss、类别权重、适配器、EMA、特征库或新阈值；
- 选择最后一步学生而不是 best-val；全程val只供独立评价，不回馈优化器或训练早停。

两份完整模型上限是每个实验进程的学生+冻结 t-1 教师，含CPU完整模型副本。优化器、激活、当前小原型和checkpoint缓冲分别诚实计量。归档保留不等于允许算法访问更早模型。

## 6. 执行顺序与自动续跑：一次授权完成整个有限计划

### P0：实现、回归、冻结
复用既有数据、训练、评估代码。只新增一条配方和 optimization_seed 适配；不修已结束的 SCD。
必要测试详见第9节。通过后发布精确source freeze，执行checkout保持干净。

### P1：补齐缺失格
执行 L05_SSL / optimization_seed0 的 stage1+2，共5300更新。
完整两阶段及部署后一次性评价。stage1出现较差指标也不能基于val改配方或提前换模型。
用全精度seed0数据计算第7节的 P_GATE，密封 gate 结果。

### P2：保底复核，**不依赖 P1 成功**
固定执行 optimization_seed1、2 的：
- 各自公共 stage0：8000更新；
- S stage1+2：5300；
- L05 stage1+2：5300。
每个种子18600，共37200更新。
P2在工程条件就绪后可与P1并行，P1科学未达标不能阻塞P2。任一新种子科学结果差，也仍完成另一固定种子；不挑种子。

### P3：唯一条件扩展
仅若P1的P_GATE全部通过，无需再次等待聊天确认，自动执行 L05_SSL optimization_seed1、2各5300更新，共10600。其初始化使用P2已训练的对应公共stage0。
不根据seed1结果改变seed2配方，也不在种子间选择超参数。
若P_GATE未通过：P3不启动，记 NOT_ADMITTED_BY_FROZEN_GATE，继续/完成P2；不寻找另一个SSL权重。

全部阶段结束后发布完整报告。最终停止仍然是项目预算终点，不代表每条任务都失败；必须区分工程完成、L05复核结论、SSL新增贡献结论。

## 7. 评价与预先固定的判定

R_{s,t,d,c}：optimization_seed=s、训练至阶段t后、在已见域d的val上按病例平均的类别Dice。
macro = (rim+cup)/2。F为最终三个域等权；H为最终两个历史域等权；N为stage1/domain1与stage2/domain2的平均。
继续原ignore=255及空支持口径。三个域不按病例总数混合。主评价不加入训练集分数。每个阶段最后学生独立评价所有已见域；未来域不提前评价。

### P_GATE：seed0 L05_SSL是否值得额外种子预算
全部满足：
1. F(L05_SSL)-F(L05) >= .010；
2. H(L05_SSL)-H(L05) >= -.010；
3. 每个增量阶段当前域 macro(L05_SSL)-macro(S) >= -.010；
4. 每个增量阶段的当前 rim/cup，L05_SSL-S >= -.020；
5. 完整训练/数据隔离/两模型/单学生部署通过。

这是新缺失格的前瞻性筛查，而不是修改SCD相对D的旧失败标准。D及其高F和cup代价始终列为描述性对照。P_GATE通过不代表超过所有旧臂，也不代表SOTA。

P_GATE没通过不按接近程度延长搜索；同时保留F/H/N和每项差值，不能只汇报FAIL标签。

### L05固定split训练种子复核
不让用于选择配方的seed0掩盖新种子结果。确认性判断只使用optimization_seed1和2的配对结果：
- 两个种子 F(L05)-F(S) 均 >0；
- 两个种子配对F差的平均 >=.010；
- 两个种子配对H差的平均 >=0；
- 每个种子、每个增量阶段当前macro差 >=-.010；
- 每个种子、每个增量阶段当前rim/cup差 >=-.020。
均满足记 L05_FIXED_SPLIT_REPLICATION_SUPPORTED，否则 NOT_ESTABLISHED，并逐项列出原始值。
这不等同于统计显著性、跨数据划分或独立患者确认。

### L05_SSL新增贡献复核（只有执行了P3才判定）
只以新optimization_seed1和2配对结果为确认依据：
- 两个种子F(L05_SSL)-F(L05)均 >0；
- 配对F差平均 >=.010；
- 配对H差平均 >=-.010；
- 每个种子/阶段当前macro相对S >=-.010；每个当前rim/cup相对S >=-.020；
- 工程与访问要求通过。
满足记 SSL_ADDITION_FIXED_SPLIT_SIGNAL，否则 NOT_ESTABLISHED。

单列三种子均值/SD、两新种子确认结果、逐seed矩阵与最差类别差。不得将像素、病例数当作训练随机种子样本量；不做“3个种子证明显著”的夸大声明。

## 8. 预算与记账

无论P_GATE结果，正式新增预算：
- P1: 5300
- P2: 37200
- 基础合计: 42500

若P3准入再加10600，正式上限53100。
已有历史正式尝试57208不改；全部任务完成时累计分别99708或110308。
资格测试、失败尝试、诊断重放都另记，不塞进有效训练预算，不删除失败成本。

实际训练分钟、GPU等待、各臂墙钟、并行总墙钟分开记录；不承诺固定完成分钟。GPU从已确认授权范围中选择，查实时占用；资源不足排队，不杀其他进程。资源未就绪是WAITING/BLOCKED工程状态，不是科学失败。

## 9. 最少但有效的工程验收

不是增加测试数量竞赛，而是覆盖本次改变：
1. objective常量张量测试：S、L05、L05_SSL公式准确，KL_l系数.5、U-KD恒0。
2. loss/gradient parity：optimization_seed0的S/L05与冻结实现匹配；可在小型fixture逐步比较参数/Adam/GAS/随机state。
3. L05_SSL在lambda_u=0时，与L05的训练更新匹配（PAS前向不得改变监督RNG/持久化缓冲）；正权重时学生U梯度非零，teacher梯度为零。
4. teacher_U_forward必须为0，给teacher-U入口放sentinel；新臂不允许调用SCD solver。
5. L05训练访问U图像计数=0；P臂只访问U图像而非GT；跨历史train/future/test访问失败。
6. optimization_seed0兼容性，以及1/2初始化、batch顺序、几何、strong、GAS分类头draw实际不同且同seed跨臂相同。
7. metadata始终split_seed0；不能仅把目录命名seed1/2就声称进行了随机种子复核。
8. 只读前驱hash；stage2必须本臂stage1；多种子之间不交叉父checkpoint。
9. resume保持优化器、调度位置、GAS、原型、随机流、更新计数；终态文件存在拒绝原目录重跑。
10. 独立DATA/h5/v1路径夹具、隐藏标签拒绝、单学生部署、最大完整模型数检查。
11. evaluator与原scorer全精度parity；阈值读原CSV全精度而不是展示小数。
12. 自动调度测试：P_GATE通过启动P3，失败跳过P3但P2必完成；seed1科学未过不跳seed2。

新增qualification在本地和服务器分别执行并记录精确源码和父子exit。历史测试因命名空间白名单需调整执行方法时，不修改原断言；公布执行范围，不把未跑写为已通过。工程错误只能在明确保留失败attempt后修复；任何修改科学目标、阈值、数据或增强均不在本次授权内。

## 10. 附带诊断不产生新调参权

复用原 epoch loss、覆盖率和类预测统计。记录：weighted/raw U CE、标注KD、当前/历史每类Dice、KD冲突比例，以及P臂teacher-U计数为零。

在各阶段最终模型密封后，隔离evaluator可额外对当前域val做固定弱视图PAS诊断，记录每类伪标签precision/recall/coverage及分母。优先复用已有最终验证forward/概率；必要的额外前向单列，不读train-U隐藏GT，不回馈训练，不增加门槛。原型刷新若为旧epoch96版本就原样说明，不事后换更佳原型。

固定因素交互只在P结果出来后计算：
CE在无U-KD背景下的作用 = P-L05；
CE在有U-KD背景下的作用 = D-U0；
U-KD在有CE背景下的作用 = D-P；
交互 = (D-U0)-(P-L05)。
这些是共享起点的完整训练配方差值，不是独立像素的因果效应。

## 11. 最终交付（合并交付，避免再拆成许多需确认的轮次）

必要文件：
- PROTOCOL.json/.md：配方、随机流、条件准入及预算在读新val结果前冻结。
- SOURCE_FREEZE.json、TEST_REPORT.json、INPUT_LINEAGE.json。
- SEED0_FACTORIAL_RESULT.csv：四格、L10/S/E等描述性参照与条件交互。
- REPLICATION_BY_SEED.csv、STAGE_DOMAIN_MATRIX.csv、PER_CLASS_METRICS.csv。
- GATE_ACCOUNTING.json：P_GATE、L05复核、SSL复核分别判断。
- TRAINING_AND_MEMORY_ACCOUNTING.json：正式与资格成本、双模型、teacher-U=0、部署证明。
- FINAL_REPORT.md：明确哪个已有配方得到复核，哪个新增贡献成立或不成立；不再只有一个笼统FAIL。
- PUBLICATION_VERIFICATION.json：真实远端SHA、NAS小型公开证据归档；source SHA与report SHA分开。

私有权重、图像、GT、患者标识、原始病例输出和访问凭据不上传GitHub。

最终决策：
- L05复核成立而SSL不成立：保留L05为监督持续学习基线，诚实停止本SSL增补，不改名宣称半监督新方法。
- SSL复核成立：冻结配方，下一步才讨论独立数据和更完整的JASCL对照；不得声称完整复现JASCL/UniMatch或达到SOTA。
- 两者都未成立：结束这一骨干/配方族的当前小网格开发，不自动引入投影、类别特殊规则或新的种子搜索。

不要只回复计划；用户转交本文件并要求执行后，完成所有已准入的固定任务并发布。不在5300步完成后仅以“未授权seed1/2”结束，因为本文件已明确授予训练随机种子1/2的有限范围权限；旧协议锁保持原样，本新协议独立授予权限。
