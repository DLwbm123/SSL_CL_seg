# 双骨干 QPrompt/GRQA 适配与 History-free RL 扩展：实验计划 V1

日期：2026-09-24  
研究标识：`SSLCL_QPROMPT_RL_V1`  
当前交付：计划、执行提示词、迁移清单工具；不是已实现的训练系统，不是启动授权。  
执行环境：新服务器经确认的 `remote-home` 下新建研究目录。不得继续沿用旧 NAS 的写入路径。

## 0. 本轮要回答什么

先回答：**QPrompt/GRQA 的组相对 query 对齐，在你的眼底分割任务上是否有独立收益，能否同时适用于 U-Net 和 ViT？** 再回答：**它是否能改善半监督持续分割，以及参数记忆是否提供额外保持收益？** 最后才进入上一轮讨论的、真正采样学习动作的 RL 控制器。

本计划刻意分开三个研究对象，不能合并命名或混淆结果：

| 轨道 | 实际方法 | 本计划定位 |
|---|---|---|
| Q-ViT | 最后一层 query 注入 + 论文公式 GRQA | 原文思想的医学任务重实现；不是原文自然场景成绩复现 |
| Q-UNet | 卷积 U-Net + 单层 query cross-attention head + 同一 GRQA 核 | 新的 CNN-to-query 适配；带注意力头，不能称纯卷积架构 |
| RL-Control | 采样监督动作，以候选学习后的标注反馈为 reward | 我们自己的后续 bandit/RL 假设；不是 QPrompt-R1 原方法 |

**首先执行R0，获准后做R1。R1结束可分别审阅R2（GRQA持续学习）或R3（真正RL控制器）的有限实验；不需要跑完R2大矩阵才有资格尝试RL。所有后续阶段默认关闭，不自动排队。** R1 不加入 LoRA、旧模型蒸馏、多尺度辅助头、历史原型库或旧 HKL 熵降权。R2 单独比较参数记忆的贡献，不把已有 A 侧底座当成已验证的优势。

## 1. 来源及不能静默补全的地方

### 1.1 原文已经明确

上传 PDF 第4页图2及式(2)–(3)：ViT 的前 L−1 个 block 处理图像，最后一个 block 才拼接 learnable queries；queries 和预测头保留在推理路径。第5页式(4)–(7)：用训练 GT 区域建立 EMA 原型，reward 为 query 与最相似原型的余弦相似度。第6页式(8)–(14)：同预测原型的 queries 为一组，组相对优势、裁剪概率比值和 reference 项共同构成 GRQA。第7页式(15)：Lseg + λimg Limg + λgrqa LGRQA；先2/3监督训练，后1/3启用对齐。第10页表5提供 clip=0.1、β=0.001、λimg=10、λgrqa=5 的实验设置。[S1]

2026-09-24 读取 PDF 第一页真实链接并检查其仓库：`straybird2333/QPrompt-R1` 默认分支根目录仅 README.md。当前没有可运行官方代码；若后续发布，先记录源码差异，不能在运行中切换实现。[S2]

### 1.2 需要公开记录的重实现选择

原文不足以完整确定 K、query 初始化、匹配损失细节、缺类行为、EMA 系数、detach 路径、具体优化器和全部训练预算。本计划给出一版预设值，它们是**研究者选择**，不是“从原论文恢复的事实”。实施方必须维护 `PAPER_TO_IMPLEMENTATION.md`：逐项记录原文页码/公式、实际代码位置、选择依据、偏差和验收测试。

特别注意第6页式(13)：本文按选定 c_i 计算 `r - log(r) - 1`，r=πref(i,c_i)/πθ(i,c_i)。主实现保留该公式，代码命名为 `paper_selected_k3_surrogate`；另外记录完整 categorical KL 作为诊断。不能静默把式(13)替换成全类别精确 KL，也不能把确定性 argmax 下的这个数值直接宣称为精确 KL。

另一条边界：论文 GRQA 的 πref 来自 EMA；RL-Control 中行为策略与 KL reference 必须分开，不能为了“统一实现”混成一个对象。

### 1.3 旧项目沿用什么

当前 canonical 数据代码规定 RGB 384×384、类别0/1/2及ignore255、L/U划分与hash，以及顺序 REFUGE→RIM→Drishti 和 REFUGE→Drishti→RIM。[S3][S4] 这些数据协议可以复用；不继承旧实验的启动授权、运行目录或性能结论。

当前已核对代码锚点：`f6b96973632b9fa8a2c3578e7fa7a18c65ff47ca`。该提交新增旧 HKL P0-B 终止报告：修正后84次前向完成，O1未通过冻结门槛，正式训练未启动。这个历史结论必须原样保留，新研究不能恢复旧任务或覆盖其终态。[S5]

## 2. 数据协议与评价边界

### 2.1 不改旧 split，不读取 U 的隐藏 GT

沿用 `manifests/training/lcrseg_v1_seed0.csv` 与 `splits/fundus_seed0.json`，不得为取得更好结果重新切分旧验证集。

- manifest SHA256：`0622f54f42f05d6ef87f9dc89ee9435cf8da03c6c30cd970db6ea167e00dd8a3`
- split SHA256：`f250d97aea1f36f21899f5dd40bb6c9a819e7755aee458c8ee27506496b46a88`
- 训练 L/U 数量：REFUGE 40/160，RIM_ONE_r3 16/63，Drishti_GS 10/41。[S3]

数量以真实 manifest 验证；不从文件名推测 patient_id。原始 GT 只有当前阶段 L 进入训练。U accessor 不保存 label 路径、不访问 label key。val 独立进程评价；test 默认不搬、不读取。无法验证患者分组时用 image-balanced 表述，不制造患者独立性。

### 2.2 数据何时迁移

**M1（R0/R1）**：只搬 RIM/Drishti 的 `train_labeled`、`val` 对应图像及标签，原始 manifest/split 元数据仍按原字节保留。此时不搬 U，也不搬 REFUGE payload。

**M2（R2获准后）**：增量补 RIM/Drishti 的 U 图像，REFUGE 的训练 L 和 val 图像/标签。源域训练预设仅用 REFUGE L，所以 REFUGE U 仍不搬。所有阶段均不搬 test payload、U标签、MRI数据或其他自然场景数据。

数据元文件可能列出未搬的split；这不表示可以访问其payload。每一run额外加载本次role/phase授权清单。真实训练仅能打开本域、本角色的文件；已搬到同一服务器不等于当前训练进程有权读取所有域。

### 2.3 指标

主指标：先逐图像算 rim、cup Dice 的宏平均，再域内等权平均。R1报告两域等权 Q；R2报告最终三域等权 Final、前两域 Old、最后新域 Incoming，以及逐域早期分数减最终分数的有符号 Forget（不截断）。同时完整报告rim/cup、disc_union、按域边界指标、无支持样本规则。不能用disc_union替代rim/cup主指标。

只用预定最终学生checkpoint作主结果，不临时切换EMA或best epoch。当前固定split已经多次暴露，所有结果标为development/optimization-seed confirmation，不宣称独立患者外部验证。可在后续新协议中另做独立划分或外部数据集；不自动搬外部数据。

## 3. 两种骨干与结构控制

### 3.1 U-Net线

保留当前U-Net body：16/32/64/128通道、GroupNorm、三次pool和转置卷积decoder。[S6]

- `U-DENSE`：body + 当前确定性3×3读出，保留实际valid-conv后插值的几何定义；不要误写1×1或GAS随机头。
- `U-QUERY`：同body，最终decoder特征经1×1映射为128维；mask embedding分辨率为H/4×W/4；query上下文再池化至H/8×W/8；12个learnable queries，经一层cross-attention（4 heads）+ FFN（宽512）获得图像条件化queries。类预测为C+1，mask为query与mask embedding的点积，再双线性恢复384×384。
- 不在384×384全分辨率token上做平方复杂度self-attention；不能为方便直接构造147456×147456矩阵。
- 这条是新CNN适配，额外query头属于推理结构。GRQA附加训练分支可去掉，query头本身不可去掉。

### 3.2 Transformer线

首轮选官方 **DINOv2 ViT-S/14，无register版**，只下载/迁移一个指定权重文件；官方提供该型号。[S7] 它不是论文主要使用的DINOv2-L，必须标为小型号医学适配，不能拿结果声称复现论文54 FPS或64.1 mIoU。

- `T-DENSE`：同一pretrained ViT + 轻量dense预测头。
- `T-QUERY`：前L−1层正常；仅最后block拼接12个queries。原CLS及image位置编码处理保持upstream语义；不得重复添加image位置编码或每层插入queries。mask读出采用两层×2转置卷积，再恢复空间尺寸，接近论文结构。
- 384不是14的倍数：预设图像右/下反射pad至392，label padding为ignore255；按392的几何处理patch/位置编码，再精确裁回384。记录边界有效性掩码。禁止把两条线GT分别裁成不同视野。
- ViT normalization、预训练来源与U-Net从头训练不同；报告重点是**各骨干内模块增量**，不把绝对差值解释成CNN/Transformer架构优劣。随机初始化ViT只作为后续单独批准的预训练归因，不默认加矩阵。

### 3.3 共同query分割语义（预注册实现选择）

K=12；C=3；第4个logit是`no-object`，**不是background**。GT以当前图像存在的语义类别mask构成匹配目标，含真实background，忽略255；Hungarian匹配后匹配query受class/BCE/Dice监督，未匹配query只受no-object分类监督。

预设matching cost=(class 1, mask-BCE 5, mask-Dice 5)，分割loss权重同此，no-object CE权重0.1。所有query臂完全相同。mask-dot-product缩放固定1/sqrt(d)；reward仍用不额外缩放的cosine。semantic输出定义：对每类求query的class probability×mask sigmoid之和，移除no-object后按类别归一化。所有概率分母、ignore、空类与全空图像行为编入测试。

Q-ViT原型使用最终image token在同一query坐标的normalized embedding；Q-UNet使用128维mask embedding。跨骨干只共享算法，不强行共享不同维度的原型或学习率。

## 4. GRQA核：明确张量、状态、梯度

1. 每阶段用当前L、无随机增强的一次完整前向初始化P；计入前向成本。跨域重置prototype bank，不能带入旧域原型。
2. 对每张图像、每个存在类，先L2-normalize pixel embedding，再按GT区域均值构成f_c。image loss用当前可微f_c和步骤开始时detached的P；当前batch先求loss，optimizer成功后再提交EMA更新。
3. query normalization后S=Q P^T；温度固定1。c_i、r_i、分组、A_i全部detach；πθ保持query梯度；πref用同一输入、同一bank快照及独立冻结EMA网络。
4. 组为**同一图像内、同一top-1原型的queries**，不跨患者拼组。std采用population定义；eps=1e-6；单元素/零方差组A=0，不强造优势。有效query的reduce分母与预注册K规则固定。
5. clip=0.1、β=0.001、λimg=10、λgrqa=5来自论文表5；bank EMA0.9、reference EMA0.99、K=12和detach规则是本轮选择。记录supported classes与无效组；缺全部原型或未初始化则该image GRQA为graph-connected zero。
6. 第6页式(13)使用log-space实现 `expm1(log_ref-log_cur) - (log_ref-log_cur)`，非有限值停止，不能将爆炸reward静默clip后继续。另记录全类别KL，不替换主公式。
7. 任何“ratio=1且loss值为0”都要同时检查梯度；不能把loss数值接近0解释为没有学习。reference不能alias student，advantage不能反向改变reward。
8. 当前阶段结束，部署只保留body/query head。删除训练reference和P不会改变学生预测（逐值/容差测试）；并不要求删除query头。

## 5. R0：代码准备、最小迁移、合成验收

现在交给Codex的prompt只执行R0。允许：读取指定源码/PDF/metadata、哈希必要数据字节、建立迁移allowlist、确认SSH与remote-home、在新独立目录复制文件、重建依赖、有限CPU合成检查。禁止：真实数据优化、CUDA模型实验、真实smoke、自动续跑旧任务、自动提交公开数据或伪造外部审阅。

R0交付：`PAPER_TO_IMPLEMENTATION.md`、结构/损失代码、两骨干配置、`CODE_MANIFEST.json`、`ENVIRONMENT_LOCK`、`MIGRATION_MANIFEST.private.json`、传输核验回执、预算DAG、CPU测试报告和审阅入口。源码在新分支/新repo保留来源hash，历史仓库只读；不要将旧主分支授权文件复制为新研究许可。

CPU测试初始上限64次合成segmentation optimizer调用，总计最多2次测试套件尝试；故意失败调用也计费。绝大多数数学测试无需optimizer。超限先报原因和新预算，不能删除ledger重置。R0迁移工具测试本身不训练网络，不占科学训练预算。

## 6. R1：先验证论文思想在两个骨干上是否成立（监督、无U）

### 6.1 固定矩阵

2骨干 × 2域（RIM、Drishti） × 1开发种子261 × 6臂 = **24个最终模型**：

| ID | 组成 | 目的 |
|---|---|---|
| D0 | dense head + Lseg | 原生dense读出对照 |
| Q0 | query head + Lseg | query架构收益 |
| Q1 | Q0 + λimg Limg | image/prototype正则贡献 |
| Q2 | Q1 + group-relative clipped reward，β=0 | group-relative部分 |
| Q3 | Q1 + 论文完整GRQA | 主候选 |
| Q4 | Q1 + 直接query–prototype相似度loss + 同式(13)reference项 | 区分组相对机制与普通alignment |

Q4直接loss=-mean_i S[i,detach(argmax S_i)]，权重5，reference项系数为5×0.001；不使用A。相同名义权重不是相同梯度尺度，报告实际梯度范数，不能过度归因。

每模型总进度3000更新。Q0–Q4共享**同骨干/域/种子**的2000步Lseg prefix，全状态复制后各做1000步suffix；D0独立3000步。prefix导入学生、optimizer、scheduler、scaler、RNG和计数，不仅复制权重。EMA/P在posttraining起点以相同协议初始化。每组合10,000次物理segmentation更新；4组合共 **40,000次**。实际执行任务为4个D0全程、4个prefix、20个suffix，共28个任务，不误报24个完整独立训练。

### 6.2 公平设置

batch2，384×384，固定几何与颜色增强。U-Net AdamW lr=1e-3；ViT backbone lr=1e-5、head lr=1e-4；各骨干内所有臂优化器相同。weight decay预设U-Net1e-4、ViT1e-2；全程poly调度power0.9，不在suffix重启。ViT与U-Net的这些超参是新研究预设，不由旧val结果反推。

backbone、head、query初始化使用独立固定RNG stream；增加query参数不能改变相同骨干的初始body权重或数据顺序。D0/Q0的架构差异仍需单独解释。

不做性能驱动网格搜索。资源不足可在任何科学训练前一次性固定精度/gradient accumulation；不得OOM后仅给某臂改变分辨率或有效batch。首次原生CUDA合成资格上限32 optimizer调用，L-only disposable smoke两骨干各8步，共16；均单独记账，状态不进入正式训练。

### 6.3 首轮报告与继续条件

关键delta：Q0−D0、Q3−Q0、Q3−Q1、Q3−Q2、Q3−Q4，所有骨干/域都列出。记录每类query利用率、组大小、单元素组比例、reward/std、clip fraction、selected surrogate/真实KL、GRQA梯度与Lseg梯度范数、prototype支持、训练耗时、GPU峰值、推理时延。

预设研发投入门槛（不是统计显著性标准）：每骨干Q3−Q0两域平均≥0.005，任一域不低于−0.010；同时报告Q3−Q4是否为正。两骨干都达标才提出“双骨干可迁移”的下一步；仅一条达标则如实限定，不静默删掉另一条。若Q3仅优于D0而不优于Q0，不把架构收益归为GRQA。

初次结果不作为TMI最终证据，不预设必须成功。评估门槛不通过则写负结果报告，不调阈值、不追加相同val搜索。是否进入R2需新的科研审阅。

## 7. R2：半监督三域持续学习与参数记忆交互（另行批准）

### 7.1 序列、source和六臂

两骨干、两顺序、开发seed261。每骨干独立训练dense/query两种REFUGE source，各6000步，仅用40张source L；共4个source、24,000更新。所有query目标臂共享同骨干source-SFT，不从RIM/Drishti单域终点假装source；source阶段不启用GRQA，使后续对比聚焦持续阶段。不能用旧KI或其他seed终点替代不匹配source。

目标臂：D_SSL、Q_SSL、Q_DIRECT、Q_GRQA、Q_SSL_MEM、Q_GRQA_MEM。前四臂检验dense/query/直接对齐/GRQA；后两臂与Q_SSL/Q_GRQA形成memory×GRQA交互。

目标域仅当前L/U。EMA teacher weak-view软目标，student强视图与GT几何保持一致；mask为当前teacher confidence>0.7、当前域prototype cosine>0.5和valid geometry，缺原型时按冻结规则退回confidence-only并报告。λU=0.5，前20%更新无U、后20%ramp。GRQA与Limg首先**只在L上计算**，最后1/3启用；U只进入已定义的soft KL。不要用伪标签更新GRQA bank并声称是原论文。

RIM每阶段3200、Drishti2100更新。2骨干×6臂×2顺序×(3200+2100)=127,200目标更新，加source24,000=**151,200更新**。这是R2 proposal，不是R0自动队列。

### 7.2 参数记忆：不默认LoRA

MEM臂在每阶段起点从当前已训练权重计算SVD。U-Net选14个body Conv2d；ViT选patch projection与各block的qkv/proj/fc1/fc2线性映射，名称和形状在R0生成清单核对。每层W0∈R^(m×d)，k=floor(min(m,d)/2)，至少1且小于d；V为前k个右奇异方向。

使用**不额外限制秩**的增量 D：W_eff=W0+D(I−VV^T)，D初始0。非MEM对应层同样使用D、但projector为I，从而可更新层集合和优化器对象对齐。其它query/readout/normalization参数在两臂中保持相同可训练语义；它们并未被此局部约束保护，不能宣称全模型不遗忘。phase末将增量合并，discard当前V/D，以合并权重进入下一域。

这是为检验参数记忆而选的新full-rank参照，不是已证实方法，也不继承旧LR_SRC_A数值。报告额外W0、V、optimizer/EMA内存；不宣称PEFT省内存。保护基底只从权重获取，不从旧图像/旧梯度/旧特征获取；不要把它称为真实遗忘测量。

### 7.3 确认与停止

seed261为开发。主候选由R2完整配对报告决定，不能按某个旧域的中途val选best checkpoint。研发门槛建议：Final平均增益≥0.005、两顺序方向为正、Old下降不超过0.010；细类退化完整公开。这些是投入门槛，不保证论文成立。

只有审阅后，固定Q_SSL/Q_GRQA/Q_SSL_MEM/Q_GRQA_MEM四臂，在新优化seed262/263两骨干两顺序确认。每seed/骨干一个query source6000；4个source=24,000，目标2骨干×2seed×4臂×2顺序×5300=169,600，合计**193,600更新**。新增seed不等于独立患者验证。D_SSL/Q_DIRECT的开发负结果继续保留；外部数据和正式方法比较另立预算。

## 8. R3：真正RL学习动作控制器（设计保留，默认关闭）

此轨道独立命名 `RL_CONTROL_BANDIT_V1`。完整持续学习扩展可从对应骨干Q_SSL_MEM出发；下面R3a/R3b先复用Q0 prefix验证单域决策。**不默认叠加GRQA**，先明确新收益来自哪里。动作取三种：忽略U区域、只学disc/background、学完整三类；作用于同一图像的一个预先定义候选区域，禁止跨图像随意凑组。

使用当前训练L的固定5-fold患者/图像分组，某次决策的L_fit与L_feedback不重叠；后者是训练反馈数据，不能作为验证集。少标注域每fold支持量必须公开，缺有效分组则不实施患者独立奖励。U的GT始终不读。

从相同全训练状态出发，采样G=4个动作（允许重复），分别执行h=1的临时候选更新，另执行一个只用L的anchor更新。reward=J(theta'_g,L_feedback)−J(theta'_0,L_feedback)，J为冻结的图像/类别平衡监督质量。奖励不包含旧域val。候选更新全部计入训练成本；不落地候选不等于免费。

行为策略phi_beh是采样时快照；importance ratio相对phi_beh，reference单独EMA。使用组相对优势、clip0.1、精确3-action categorical KL(β=0.001)，每decision固定4次策略更新。禁止把这段算法写成论文式(13)的原样复现。

初期以每20个学生更新一次决策为预设。执行学生动作从更新后的policy另行采样，不把反馈最高的候选模型直接当最终学生。若全部候选reward≤0，预设不学习这组策略、该次学生选择no-U；此绝对收益guard是本研究选择，明确记账，不声称得到无偏长期策略估计。退化组跳过policy更新并保留原因。

必须比较fixed、random、entropy-rule、用相同reward训练的non-RL动作预测器与RL；控制标注反馈次数、候选更新与总算力。单步方案称contextual bandit，不宣称跨域长期规划。RL策略参数可以跨域保留，rollout/feedback样本不能跨域积累。动态参数记忆仍是代理，不能伪造可直接观测的遗忘reward。

### R3a：先做有限reward可学习性诊断

可复用R1对应骨干/域Q0的固定2000步prefix，不能选best checkpoint。2骨干×2域×8个预先绑定训练状态/批次组合，每组合G=4候选加1个L-only anchor、h=1，共**160次临时学生optimizer更新**；committed学生更新0、policy更新0。不能说“零训练”，因为试探更新也耗费真实数据计算。

每个组合用不同fold的两个L_feedback批次评价同一候选，比较奖励排序一致性、绝对正收益比例及类支持。8个组合及feedback轮换清单在读结果前冻结；全负/退化奖励、不稳定排序或支持不足明确报告。其评价样本都是训练数据，不是新的验证集。这个小诊断可以与GRQA路线的效果结论分开决定是否继续，不必等待整个R2。

### R3b：双骨干RL pilot（另行批准，不能在R0自动实现后运行）

仍从R1每骨干/域固定Q0 prefix分叉，先验证单域SSL决策价值，暂不声称解决持续遗忘。6臂为fixed、random、entropy-rule、reward-regression、RL、fixed-extra-L。2骨干×2域×6臂=24个终点，每臂1200个常规学生更新，teacher/bank初始化和U ramp使用相同协议。

每20步一次决策，共60次。reward-regression与RL各有4个骨干/域组合，共8条轨迹；每次4个候选加1个anchor、h=1，所以临时候选更新=8×60×5=**2,400**。两种自适应臂每decision固定4次控制器optimizer调用，合计8×60×4=**1,920**；RL与regression各960。regression用同类反馈学习动作收益，不用policy-gradient；记录它与RL的动作采样分布不同，不能声称反馈内容完全相同。

fixed-extra-L每decision另做5个L更新，因此4×60×5=**1,200**额外committed更新；它是optimizer-call-matched额外计算基线，不是宣称GPU时间/FLOPs精确相等。额外L允许使用同角色反馈数据且计入访问；主要作用是排查更多训练/标注计算本身是否解释收益。所有臂实际forward/backward、GPU时间、标注访问分别报告。

R3b：常规committed 28,800 + extra-L committed 1,200 + probe 2,400 = **32,400次学生更新**；另有1,920控制器更新。R3a再加160探测更新。这些是待审阅上限/精确无失败计划，不构成启动许可；跳过退化组时实际policy调用可减少，不能补做来追到预算。失败尝试按真实次数保留。

R3a/R3b不验证真实旧域保持；转入完整history-free持续序列时必须另审source继承、参数记忆交互和有限预算。不得把单域RL pilot改善写成持续学习有效。

## 9. 累计预算与执行状态

| 阶段 | 科学segmentation更新 | 其它更新/成本 | 当前状态 |
|---|---:|---|---|
| R0 | 0 | CPU合成≤64；迁移工具0 | 可准备；无训练授权 |
| R1 | 40,000 | CUDA合成≤32、L-smoke16；前向另列 | 审阅后申请 |
| R2开发 | 151,200 | 资格/恢复成本新计划单列 | 未授权 |
| R2确认 | 193,600 | 资格/恢复成本新计划单列 | 未授权 |
| R3a | 160次probe、0 committed | 0 policy；反馈前向另列 | 默认拒绝执行 |
| R3b | 32,400（含2,400 probe） | 1,920控制器更新；不是全都属于RL | 默认拒绝执行 |

R1+R2开发+R2确认的科学更新合计384,800；R0和资格成本另计。所有阶段不能当作一次性批准。ledger分别记录physical attempted、successful、committed、failed及forward/backward，不合并CPU与正式训练。

一次失败消耗实际预算。丢失checkpoint尾部不自动重放；先做恢复评估和新授权。迁移文件断点续传不消耗optimizer，不得误归为训练恢复。

## 10. 新服务器最小迁移与目录

目标目录必须由执行环境解析，例如：

    ${REMOTE_HOME_ABS}/sslcl_qprompt_rl_v1/
      code/                # 必要新代码与许可证，不是旧repo整包
      data/                # canonical相对路径；按M1/M2增量搬运
      assets/              # 一个DINOv2权重、最小第三方源码/许可证、论文
      env/                 # 本机重建的环境和lock
      cache/               # 本研究的HF/Torch/pip cache
      tmp/                 # 合成临时状态和下载partial
      runs/                # 仅本研究任务
      receipts/            # private manifest、hash、transfer/env回执
      reports/             # 去标识汇总

`remote-home`是用户指定的存储位置，不自动等于SSH alias、$HOME、/root或旧NAS。通过已配置连接查询真实绝对路径及其realpath/mount/quota；识别不到先问，不猜。确认owner、可写性、空间与inode。数据量B、权重A、环境E和计划保留checkpoint上限C展开后，预留≥1.2(B+A+E+C)再执行。训练峰值显存另测，不能用磁盘量估计。

### 10.1 只搬这些

- 新模块、必要旧U-Net/data/evaluator定义、入口、配置、单元测试、依赖lock、许可证、来源映射。
- 实际import的依赖闭包。动态import必须用离线import smoke验证；不能仅凭AST断言闭包完整。避免旧`import *`牵出整套历史执行器；必要时最小port并保留归属。
- M1/M2 manifest选择的canonical HDF5图像/标签；每个文件按现有SHA校验。若容器文件还混有U标签，不能直接搬整个文件，应另报“拆分必要”，经确认导出image-only并产生新的派生manifest，绝不覆盖canonical版本。
- 一个DINOv2-S/14指定预训练权重；若旧服务器没有，目标机从官方来源下载单文件并校验，记录是新获取不是旧迁移。不要搬整个HF/Torch cache。
- 本研究实际启动需要的source checkpoint才可逐个列入；本计划默认旧checkpoint数量0。旧source不兼容新头，不默认复用。

不搬`.git`对象库、旧runs/protocols、日志、wandb/TensorBoard、图表、全部checkpoint、optimizer历史、旧虚拟环境、conda缓存、原始数据重复副本、test/U标签、SSH私钥/token/.env。报告中的源路径或患者ID视为private，不push GitHub。

### 10.2 传输流程

1. old端只读inventory，生成每文件source_relative/target_relative/bytes/SHA256/reason/role；review精确allowlist和总体积。
2. 对照SSH既有配置确认目标和host key，不关闭host-key检查，不复制私钥。两台服务器不能直连时优先目标端pull/ProxyJump；确需中转则只用加密受控暂存，先告知，不把患者数据绕经公共仓库或网盘。
3. `rsync --files-from` + `--from0` 精确文件列表，通过SSH复制；先dry-run。rsync不支持两个remote参数直接远程对远程复制，所以实际命令必须在old或new一端执行。[S8]
4. 使用目标研究目录内`.incoming/<manifest_digest>/`暂存；不使用`--delete`、`--remove-source-files`、`--inplace`。源文件永不删除，已有目标文件不覆盖。
5. 在目标逐文件核验SHA256、字节数和allowlist外文件；核验通过后逐文件无覆盖提升到code/data/assets。增量M2复用相同hash文件，只补missing。不同hash冲突硬停，不改成“以最新为准”。
6. M1/M2传输清单和完成回执分开保存；传输实际耗时、网络字节与失败重试可记录，但不谎称验证了数据语义。
7. 未绑定源/目标路径和确切allowlist前不执行复制。R0可以在确认这些参数后完成迁移，但仍不能开始真实训练。

### 10.3 环境迁移不是拷贝环境目录

新建Python/PyTorch环境；先记录GPU型号、driver、CUDA runtime、架构、Python和Torch版本，选择兼容版本后锁定。旧项目py38不是本研究必须沿用的约束。不要硬装论文/旧服务器CUDA版本，不改系统驱动或其他用户环境。

所有HF_HOME、TORCH_HOME、PIP_CACHE_DIR、TMPDIR显式指向remote-home研究目录，离线模型构建不得隐式下载。只迁移lockfile或必要wheelhouse，不复制venv/conda绝对路径环境。禁用在旧NAS路径写log/tmp/checkpoint的历史wrapper，写新storage guard而非全局字符串替换。

## 11. 原生验收与科研验收分离

CPU：公式值/梯度、detach、singleton、emptyclass/no-object、query matching、crop/pad、U-label拒绝、阶段权限、checkpoint全状态恢复、部署无reference/P等价、两骨干模块开关、精确prefix复用、预算计数。

CUDA原生资格：真实U-Net/ViT结构+生成数据、有限更新、保存/恢复、部署和OOM前置估计；不使用真实训练样本。之后L-only smoke仅检查接口、梯度/有限值，不根据其Dice调参；discard全部smoke状态。

失败记录必须区分：工程错误、原文实现不确定、原型无支持、信号不足、结果负向。不能把任务跑完、checkpoint存在、CPU测试通过当作科学成功。

R0结束输出`STOP_AWAITING_EXTERNAL_CODE_REVIEW`或具体的`BLOCKED_MISSING_TARGET/DEPENDENCY/DATA`。R1/R2结束报告完整矩阵与负结果，停止自动排队。不得把本plan或prompt自身当作外部review证据。

## 12. 来源索引

[S1] 用户上传的16页QPrompt-R1，ICLR 2026版本，PDF SHA256 `96b964dd69be73bc708bd45575238513435bc544d48004c5e07962dd685df9bf`。具体原文页码见第1节。  
[S2] `https://github.com/straybird2333/QPrompt-R1`，2026-09-24检查；根目录仅README.md，blob `7f0345d13371438e8e99a70058960904a84deb7a`。  
[S3] 用户repo `experiments/lcrseg/single_teacher_scd_v0_1/data.py`，blob `e7378184cdfe71cf75dc2891dc8fa5dae6bffdc3`。  
[S4] `experiments/lcrseg/five_frameworks_v1/native_data.py`，blob `3e11fdbf769c63f00576a1062c141df0243691e2`。  
[S5] commit `f6b96973632b9fa8a2c3578e7fa7a18c65ff47ca` 的 `P0_EXECUTION_REVIEW_20260920.md`；只保留历史，不继承执行权限。  
[S6] `experiments/lcrseg/di_dmpa_jascl/modeling.py` 与 `ssl_head_control_v0_1/core.py`，前文已核验的实际U-Net与linear head定义。  
[S7] 官方DINOv2源码与权重列表：`https://github.com/facebookresearch/dinov2`。实际执行commit、权重SHA与license由R0冻结，不使用浮动main。  
[S8] rsync官方手册：`https://download.samba.org/pub/rsync/rsync.1`，files-from/from0、SSH传输、断点与相对路径。  

除上述来源明确给出的内容外，实验臂、K、学习率、步数、门槛、MEM和RL-Control均是本计划的研究提案，不属于论文已验证结论。
