# 单前驱教师 SCD V0.1：真实训练筛查与 Codex 执行协议

## 0. 本次任务与授权边界

请实际实现并执行本协议，不要只返回计划、代码审阅草案或另一份离线 oracle 报告。

研究目标：固定类别空间的半监督域增量分割；最多一个当前学生和一个独立教师；推理只有当前学生。检验“监督兼容的最小历史目标修正”（Supervision-Compatible Distillation，暂用 SCD）是否优于普通蒸馏及直接删除冲突蒸馏。

本次允许：新命名空间的实现和必要回归；当前阶段 train_labeled / train_unlabeled 数据读取；新分割模型真实训练；独立 evaluator 使用已见域 val；seed0 完整固定矩阵；论文/源码差异审计；结果及脱敏证据发布。

本次不允许：执行 SHOR-UV V0.8.1；历史专家库；读取旧 SHOR/PPC/CARe-HR 概率、路由、oracle 和收益头作为新算法输入；回放过去阶段的图像/标签/病例特征；第三份教师或 shadow EMA；新增类别实验；Prostate/M&Ms；test GT；旧 formal_03；自动追加种子、超参数或新模块；修改旧锁/旧终态；合并 main。

仓库：DLwbm123/SSL_CL_seg。
已知起点：c4edcfc08bad784ad4c26e12aaaf6a395cad2f95。
新分支：codex/single-teacher-scd-v0-1-training。
若远端已有后续提交，用独立 worktree 从指定祖先创建新分支，不 reset/clean/覆盖用户工作。先核实起点及历史保护清单。

新增文件原则上限定：
- experiments/lcrseg/single_teacher_scd_v0_1/
- experiments/lcrseg/tests/single_teacher_scd_v0_1/
- experiments/lcrseg/docs/single_teacher_scd_v0_1/
配置放在新命名空间中。既有纯函数可以导入；不得解除旧 validator 的授权锁以复用其训练入口。新 runner 与新授权独立存在。

本次只有 seed0 的一轮科学筛查。完成固定矩阵后停止，无论结果如何。seed1/2 与外部确认仅写后续计划，不自动启动。

## 1. 源依据与新方法的区别

先读当前仓库 AGENTS、冻结数据协议、JASCL 模型适配代码、修复后的 Gate0 目标语义、canonical HDF5 绑定和既有逐病例 scorer。不得把旧研究 hard stop 当作本协议之外的普遍训练禁令，也不得修改旧授权。

论文参照：用户提供 Pandey 等《Continual Segmentation under Joint Nonstationarity》（arXiv:2605.20538v1）：
- §3.1 / pp.3–5：GAS；
- §3.2 / p.5：Mean Teacher、当前标注原型、置信度/相似度及有效像素 MSE；
- p.6 / 图3 p.7：还提及 L_proto；
- 附录H / 表19 p.45：不同场景冻结策略不同；
- 局限性p.9：严重域变化下仍可能遗忘。

本轮 A 是“预算匹配的 GAS+PAS/Mean-Teacher 组件参照”，不是完整 JASCL 论文复现。主动声明差异：Fundus 固定三类、公共监督 stage0 初始化、统一每步训练计划、全模型可训练、不进行历史原型回放。本轮不使用论文不同骨干/类别设置的表格数值作研究门槛。

B–E 借鉴 UniMatch 的当前学生弱到强自训练，但只用一条强图像视图，没有其完整双强视图/特征扰动/CutMix；不能命名为完整 UniMatch 复现。

SCD 的 KL 目标投影是本协议提出的待检验组件，不属于 JASCL 原文，不继承其理论结论。本轮无需再次调研或加入其他 SSL 模块。

已读取的模型锚点：
experiments/lcrseg/di_dmpa_jascl/modeling.py 中 LCRSegUNet2DJASCL，GroupNorm，通道16/32/64/128，16维 dec1 特征，官方随机3×3头。
官方 JASCL 源码固定：3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53。
该头 functional conv 的 padding/后续插值原行为不在本轮修复。复用已明确的适配行为，列入 MODEL_CONTRACT.json。

## 2. 固定实验矩阵

共享 stage0 是新训练的监督+GAS模型：只用 REFUGE 的 seed0 train_labeled，固定训练预算，不复用旧专家库；不使用无标注目标。stage0 最后一步 checkpoint 初始化全部实验臂。

stage1、stage2 运行：

| ID | 名称 | 当前半监督项 | 历史巩固项 | 第二份完整模型 |
| --- | --- | --- | --- | --- |
| S | SUP_GAS | 无 | 无 | 无 |
| A | GAS_PAS_MT_REFERENCE | 当前 Mean Teacher + 双端 PAS-valid 概率MSE | 无 | 当前EMA，仅此对照 |
| B | SELF_W2S_PAS | 学生弱预测 -> 单强视图CE | 无 | 无 |
| C | W2S_PAS_PLAIN_KD | 与B相同 | 普通前驱KL | 冻结t-1 |
| D | W2S_PAS_CONFLICT_DROP | 与B相同 | 冲突像素删除，否则普通KL | 冻结t-1 |
| E | W2S_PAS_SCD | 与B相同 | 冲突像素做本协议KL投影，其余普通KL | 冻结t-1 |

S补足“无标注目标是否有用”的对照，不属于为了扩大候选的新增方法。E为唯一主方法，D/C/B不能代替E取得主方法PASS；但允许报告简单方法有收益、因此放弃复杂组件。

C/D/E 在同一输入/权重状态下，除历史项目标/掩码外必须相同。完整独立训练后其伪标签、当前原型及前驱模型自然不同，不得跨臂强行共享以保持虚假的一致性。

## 3. 数据、预算、参数与身份

### 3.1 数据

仅 Fundus，固定顺序 REFUGE -> RIM_ONE_r3 -> Drishti_GS，固定 seed0 manifest/split，不重分患者。
固定标签：0 background，1 rim，2 cup；ignore=255；固定RGB 384×384，原uint8/255至[0,1]。
预期当前阶段 train_labeled/train_unlabeled：40/160、16/63、10/41。由冻结manifest核验，不以文件名或私人数据集里的“test”字符串猜role。

本轮不再使用路由开发实验的198行/177患者人群；使用每个阶段正式的训练标注与无标注 split。旧统计数据可以作为背景，不能替换本次测量。

训练数据 loader 只暴露当前阶段：
- labeled：image、允许的label、图像几何valid、当前阶段身份；
- unlabeled：image及几何valid；无label路径、隐藏GT、其hash或label tensor。
训练时不得通过诊断代码打开train_unlabeled的GT。

val只供隔离evaluator，评价已见域。test/未来域GT本轮完全不读。过去阶段val可以被evaluator评分，但不能给训练器提供样本、梯度、校准或早停选择。

### 3.2 数值训练计划（新协议，不沿用旧5295/13400步）

每阶段100个epoch。每步当前标注batch_size=2；有SSL时当前无标注batch_size=2。
定义每epoch步数：max(ceil(N_l/2),ceil(N_u/2))，drop_last=false。无标注序列每epoch独立固定shuffle；标注序列不足则循环并使用预注册seed流重新shuffle。
因此预期阶段0/1/2分别80/32/21步每epoch，即8000/3200/2100步。
S只使用无标注数量来匹配optimizer步数，实际不读取无标注图像；承认这是步数/标注呈现预算匹配，不是FLOPs匹配。

本轮stage0一次 + 6个实验臂各stage1和2，预期总optimizer成功更新：8000+6×(3200+2100)=39800。
完整三阶段等效每臂13300步，公共stage0实际只训练一次。原始warm-up/梯度诊断/集成训练另行记账，不能并入正式更新数。

optimizer：Adam，lr=1e-3，weight_decay=4e-5，betas=(0.9,0.999)，eps=1e-8。
每个新阶段重新初始化Adam动量；阶段内polynomial LR，power=0.9：第k次更新(从0开始)用 lr0*(1-k/K)^0.9。
全网络所有原本可学习权重可训练，包括编码器/解码器；原nontrainable GAS状态不硬改成可训练。不扫描冻结范围，不做layer-wise LR、SAM、梯度裁剪或新正则。
所有臂和stage0使用同一GAS头及原源码常数；学生需要梯度的监督/强视图前向 stochastic_classifier=true；B–E弱伪标签与C–E前驱教师 posterior mean(false)；评价false。
网络权重/前向/反向FP32，不启用AMP和TF32；小类别向量的log-softmax、KL、投影使用FP64，可分块计算。报告精度构成，不称整个网络FP64。

监督项只用CE，不另加Dice loss，保证当前CE参照方向可明确解释。所有臂共享。
λu最终=0.5；每阶段前10个epoch=0；epoch11开始，乘 min(1,(e-10)/10)，e为1-based epoch。
λKD=1.0，stage>=1时从第一步启用；C/D/E相同，固定不搜索。
τconf=0.7、τsim=0.7，均严格>；不按分数改阈值。
温度T=1，不用T>1或DKD等额外变体。

阶段0 100epoch只训练S目标，从最后一步得到公共初始化。A从这个学生初始化自己的EMA；B无第二模型；C/D/E各自复制成前驱教师。不要把A的stage0结果或旧B0专家替代公共stage0。

### 3.3 随机性与重放

把data order、geometry、strong augmentation、监督头采样、无标注头采样拆成可复现seed流，基于(seed,stage,epoch,step,stream)；不把arm名称作为核心随机流seed。额外teacher/solver调用不能改变下一批数据/增强随机序列。
种子间按既有合法split运行；本轮仅seed0。参数初始化及公共阶段0文件hash必须相同。
固定评估最后一步模型，不择优保存best；不根据旧域val选择checkpoint或更改lambda。可记录当前域val曲线，但不能影响训练计划。

## 4. 图像增强与当前PAS

几何：对图像和允许标签共同应用水平翻转p=.5、垂直翻转p=.5、k×90°旋转(k均匀取0,1,2,3)。不裁剪、不缩放、不elastic。共享同一几何的弱/强视图完全像素对齐。
弱视图：只有上述几何，RGB仍在[0,1]。
强视图：从弱视图出发，固定顺序brightness factor U(.8,1.2)，contrast factor U(.8,1.2)，saturation factor U(.8,1.2)，各确定应用；Gaussian blur以p=.5应用，kernel5、sigma U(.1,1.0)；Gaussian noise以p=.5应用，std=.02；最后clip[0,1]。不使用CutMix、copy-paste、feature dropout或第二强视图。
可使用同一个冻结torchvision/tensor实现，精确记录版本与border/色彩操作规则，所有臂B–E使用相同实现。增强不依据GT轮廓或域身份。

A的PAS一致性在弱视图，不使用此强视图；这是已声明的组件差异，C/D/E核心比较不受影响。

PAS原型只来自当前train_labeled：对每个case/class平均原始dec1特征，归一化该case中心，再对有该类的case等权平均并L2归一化成P_c（对应论文§3.2的case-level结构）。不使用旧版本的pixel-count pooled函数冒充这个定义。
当前数据原型在epoch11、16、21、…、96开始前重算。原型提取no_grad、posterior-mean、无随机几何。使用全部当前labeled样本，流式维护C×d统计，不保存病例特征。整个阶段只保留最新P，进入下一阶段清空重建。
某类当前真实无support：valid_c=false，该类伪标签不准入；记录但不捏造原型。不因此自动改阈值、旧域回填或调用更多模型。全局出现非法标签/缺文件属于工程阻塞。

B–E：p_w = softmax(S(x_u^w)).detach()；y_hat=argmax(p_w)。m = (max p_w >.7) & (cos(feature_w,P_y_hat)>.7) & prototype_valid & geometry_valid。
所有m/y_hat/P停止梯度。弱分支复用同一S对象，不clone/deepcopy模型。
L_SSL = sum_{i in Ωu} m_i CE(z_u^s_i,y_hat_i) / |Ωu|。分母是合法图像支持，不是通过PAS的像素数，不再额外除以C。
无有效m返回与学生图连接的零值。Ωu=0按空样本规则返回零并计数；不得读取隐藏GT生成Ωu。这里没有几何padding，正常Ωu为全图。

A：唯一第二模型是当前EMA(α=.99)，非前驱教师。采用student和EMA弱视图的各自概率confidence+同一当前原型相似度交集，L_SSL_A=sum_valid ||p_s-p_ema||²/max(n_valid,1)，空支持为连接学生图的零。学生和EMA的随机头采样保持官方PAS用法并单独固定seed；EMA权重每次成功step后更新，GAS状态处理与冻结参考helper一致并列入差异审计。不存在另一个t-1快照，不做历史原型L_proto。评价及进入下一阶段均采用A的学生，而非择优EMA。

## 5. 单前驱教师与GAS更新

C/D/E在stage t使用本臂stage t-1的最后学生权重作为唯一冻结T。T.eval()、requires_grad=false、不在optimizer中；参数、GroupNorm及GAS状态全程bitwise不变。q由T的posterior-mean输出生成。
在监督view上：p_l来自当前有梯度监督logits，q_l=T(x_l^w)，参照y_l是真实当前GT（排除255）。
在无标注view上：p_u来自同一次强视图有梯度logits，q_u=T(x_u^w)，参照y_hat/m来自当前学生弱分支。几何相同；teacher输入弱view以免强色彩噪声成为其额外误差因素。

冲突判断必须用实际参与KL的同一个student logits对应的p；不能用弱p来判断，却对另一次随机强logits求KL。p仅在目标构造支路detach，学生KL支路保持梯度。
在m=false的无标注位置，C/D/E都保留普通KD；没有可靠当前监督时不伪造“冲突参照”。

GAS平方梯度只能来自本步L_sup对官方mu.weight的梯度。不能从L_sup+L_SSL+L_KD的总.grad更新GAS。可用一次autograd.grad(L_sup,mu.weight,retain_graph=true)取得detach平方，随后总loss.backward/optimizer.step，再将本步监督平方写入下一步GAS状态。记录这次额外梯度抽取的成本，不宣称额外autograd=0。
若使用等价分段backward，需用独立测试证明GAS只含监督梯度；不得另做第二次随机监督forward造成不同样本gradient。

stage2教师必须来自本臂stage1新训练结果，不得重新加载旧B0/stage1，也不得继续持有公共stage0当教师。

## 6. 普通KD、冲突删除和SCD：唯一科学差异

教师q进行统一的数值正支持处理：q_tilde=(1-eps_q)*q+eps_q/C，eps_q=1e-8。C/D/E完全相同；q原概率、q_tilde和实际变化统计记录。不得为E独享平滑。

每个具有可靠当前参照y的像素：
a = stopgrad(p - one_hot(y))。
d_raw = <p-q_tilde, a>。
精确参照下d_raw<0即冲突；实现以FP64 logits softmax计算，不增加cosine阈值或可调余量。

C：每个合法位置L_i=KL(q_tilde||p)。
D：有可靠y且d_raw<0时L_i=0；否则与C相同。不重新按保留像素数归一化。
E：
- 没有可靠y，或d_raw>=0：r=q_tilde原样；
- 有冲突：求以下唯一凸问题：
    min_r KL(r||q_tilde)
    s.t. r_c>=0, sum r_c=1, <p-r,a> >=0。
  目标r、构造时p和y都detach；实际训练L_i=KL(r.detach()||p)。

共同归一化：
L_KD = .5*(sum_Ωl L_i/max(|Ωl|,1) + sum_Ωu L_i/max(|Ωu|,1))。
Ωl来自允许GT!=255与几何valid；Ωu只来自图像几何valid。
空分支为零，不按是否发生projection、是否drop或PAS覆盖再次放大权重。

总loss(B–E) = L_sup + λu(epoch)*L_SSL + λKD*L_KD（B历史项为0）。
S无SSL/KD；A用其单独声明的MSE-SSL。

### 6.1 求解与数值合同

固定p/y后，令b=<a,p>。约束为<a,r><=b。
若q_tilde可行则r=q_tilde。否则有 r_c(η)=softmax(log q_tilde_c - η*a_c)，η>=0，根为<a,r(η)>=b。可用按max|a|缩放a/b后的标量括区二分，缩放不改变约束集。
不得使用反方向的exp(+ηa)。不得在求解目标中把KL方向换成KL(q||r)。训练KL方向也不能换成KL(p||r)。
理论根定义与实现分开：CPU float64独立reference使用80次二分；生产GPU向量化/分块，使用最多60次倍增括区与固定48次二分，选可行端点。记录最大求解残差/括区次数/失败数，不按病例调迭代预算。
令max|a|=0时直接q；所有student/teacher值先检查finite。非法输入、无法括区或残差>1e-9（归一化约束）属于工程故障；不得悄悄r=p或drop救援。
对未修改目标保持其原字节；对conflict目标使用FP64计算。实际KD的log_softmax由同一个z.double()得到，确保p构造和loss梯度语义一致，再把梯度传回FP32网络。不是重新做FP64网络forward。
禁止逐像素Python/SciPy优化器进入正式训练。GPU块大小固定65536像素，可基于无GT合成microbenchmark在code freeze前降低以适配显存，选定后全部臂/阶段固定。

辅助文件 scd_projection_reference_check.py 是独立数学数值示例，可作测试参考，但不是生产runner，不能把其中的PASS说成项目GPU或数据集通过。

### 6.2 允许的理论表述

停止梯度时，T=1的logit梯度为p-r；与CE方向p-e_y局部内积非负。不能声称全参数梯度、Adam更新、Dice、所有像素合成方向或旧域风险都有保证。
二分类冲突情况下，投影常退化为r=p，即该像素KD梯度为0；测试此性质。三分类必须存在r既非q也非p的反例，证明实现不是总在drop。
正式阶段0没有前驱KD；不声称这一步解决历史域支持缺失或恢复更早模型已经遗失的知识。

## 7. 在线状态、推理与存储

每个独立训练实验臂的完整模型数量：S/B=1；A=学生+当前EMA=2；C/D/E=学生+前驱=2。A之外禁止EMA；任何臂禁止第三模型。
纯target张量、原型、GAS张量和optimizer不是额外模型，但都要报告bytes。CPU shadow完整权重也算完整模型状态，不能藏到CPU规避约束。

只滚动保存本臂student_latest完整训练checkpoint和prev_teacher权重；A第二slot是current EMA。primary不用best checkpoint，防止保存第三个模型版本。stage边界先完成评分/receipt，再替换teacher slot；新阶段不能访问t-2模型。
实验对照各臂有自己的模型属于实验总成本，不是部署时的专家库。在线每臂成本与实验总存储分开报告。公共stage0只用于初始化各臂的第一个增量阶段；不得在stage2读取。

旧研究全部档案保持原样且算法不可访问。新阶段历史权重不累积进新部署bundle。需要归档恢复证据时保留hash/配置/optimizer进度/矩阵，不把更早权重作为算法可读输入。不得删除旧研究用户数据。

推理独立脚本仅构建并加载当前学生；posterior mean，no prototype/gate/teacher/optimizer；输出分割。测试隐藏/移除本轮teacher路径后仍能推理；证明同一学生输出相等。
报告：student/teacher权重与缓冲区、梯度、optimizer、原型、目标临时张量bytes；max allocated/reserved GPU；CPU RSS；step/epoch总时间和投影成本；真实forward/autograd/backward/成功step数。不能用参数计数代替峰值显存。

## 8. 评价口径：先比同资源训练基线，不比多专家oracle

每个阶段只评价该臂最后一步学生，case-level Dice_c=2TP/(pred+truth)；GT255排除，双方空类=1。主指标为case-level(rim+cup)/2，然后域等权、种子等权，不把背景纳入主Dice。
使用已审阅逐病例评分语义及独立合成参照，不复用只提供pooled-confusion矩阵的旧logger作为主输出。meanIoU三类兼容项单独命名。全忽略case兼容score1/gain0并has_evaluable_gt=false，另报告有效样本敏感性；缺文件/坏标签不伪装all-ignore。

R[t,d]是学完阶段t后在已见域d的val宏平均前景Dice（0<=d<=t<=2）：
F=(R[2,0]+R[2,1]+R[2,2])/3。
H=(R[2,0]+R[2,1])/2。
N=(R[1,1]+R[2,2])/2。
BWT=((R[2,0]-R[0,0])+(R[2,1]-R[1,1]))/2。
Forgetting_d=max_{k=d..2}R[k,d]-R[2,d]，只汇总d=0,1；同时报告逐域轨迹，不把新域当历史遗忘。

seed0是开发筛查。既有Fundus受到研究者多轮开发暴露，即使fresh training/fold也不是外部确认。本轮test完全关闭。不要宣称论文官方复现成功、SOTA或独立患者泛化。
不使用旧0.17/0.27收益门槛，不把旧SHOR/PPC模型或oracle当同资源参照；它们访问不同历史信息。

### 8.1 seed0的固定判定

工程完整性/数据隔离/真实训练/内存约束为必要条件；不满足先记工程终态，不产生科学PASS。
以下数值为本轮预注册的实用筛查目标，不是数学常数：
1. SSL支持：N_B-N_S >= .005 且 F_B-F_S >= -.005。
2. 主效果：F_E - max(F_S,F_A,F_B,F_C,F_D) >= .010。
3. 历史保留：H_E-H_B >= .010。
4. 当前适应：对t=1,2分别 R_E[t,t]-R_B[t,t] >= -.010；当前rim/cup每类相对B下降均<=.020。
5. 投影确实在真实训练中激活，且同一输入状态下存在不同于q与p的三类目标；不把随机极小浮点残差当机制有效。该项只核实实现，不预设每个像素/每个step都改善。

全部成立：PASS_SINGLE_TEACHER_DEVELOPMENT_SCREEN。
1失败：SSL_SIGNAL_NOT_ESTABLISHED（其余结果仍完整交付；不隐藏E可能的巩固收益）。
1成立但2/3/4未全成立：SCD_INCREMENTAL_VALUE_NOT_ESTABLISHED。
2不成立且C/D与E相差<=.005，报告SIMPLE_BASELINE_COMPETITIVE说明，不改主终态或替E赢。
科学判定只在完整seed0矩阵全部结束后运行一次，不因stage1局部输赢提前删臂或改参数。
差值全精度，严格与阈值比较；不round到过线。报告所有门槛数值，而非只输出状态。
单seed可用于止损，不用于显著性或稳定性宣称。后续seed1/2方案可保持完全相同配置，但本轮不自动授权。

## 9. 预训练/部署前的工程验证（不是多轮科学gate）

在任何正式stage0更新前完成新源码freeze及本地/服务器资格；无需反复等待外部确认，只在真正无法解决的输入/权限/定义问题停下。

必须覆盖：
1. frozen scoring与新case-macro汇总的独立反例、ignore/all-ignore/空类与domain/seed等权。
2. canonical dataset binding使用di_dmpa_gate1.binding.safe_asset(DATA,relative)=DATA/h5/v1/relative；independent canonical-only/decoy fixture；image和label都验证hash/shape。
3. 当前labeled/unlabeled/val/test、跨阶段及跨seed患者身份防串；unlabeled禁止GT sentinel；stage2加载t-2拒绝。
4. 两模型上限、教师参数/缓冲区冻结、optimizer不含teacher、A EMA与CDE历史教师类型分离；最终单模型inference。
5. 同几何视图与标签对齐，strong不改几何，weak detach，PAS原型case-balanced/当前阶段来源/缺类；空m梯度安全。
6. supervised-only GAS抽取，加入SSL/KD不能直接污染GAS梯度缓冲；更新顺序明确。
7. KD/projection 两个KL方向；可行q恒等；实际蒸馏目标q_tilde=p时零KD（原始q=p后若统一平滑，允许明确记录的微小非零差异）；极端probability正支持；二分类退化；三分类非退化；eta非负；KKT/残差；与独立CPU/SciPy小问题结果比较。
8. detach目标的autograd实际为p-r；生产和reference一致；不采用另一次forward的p；梯度不流入旧teacher、weak伪标签、原型和solver。
9. D只清除可靠冲突位置，原分母不变；无可靠参照CDE同普通KD；λKD=0时CDE与B路径/更新一致（在相同状态和数据下，不是跨独立轨迹比较）。
10. 可复现步数、最后一个不完整batch、LR、GAS、RNG、原型resume，stage boundary释放旧teacher再复制新teacher；exact checkpoint resume对比。
11. 小型synthetic HDF5全链路训练/两次阶段切换/评估/恢复/失败receipt；夹具不能照抄executor路径错误。
12. 数值solver失败必须显式工程失败；不得自动跳过E样本、换标签、drop或改lambda。

真实smoke限定当前stage0 train_labeled中固定两例，最多100个优化step；验证loss确实下降、参数非零更新、teacher冻结逻辑和合法路径。该临时模型不能作为公共stage0。
另用全合成概率/当前训练数据无GT隐藏读取验证SSL非零梯度；真实首批无标注如果无PAS支持，记录但不要搜索“好batch”或降低阈值。空支持是可能的数据现象，不是要求每批过科学gate。
每个臂正式首个实际训练step记录GT/SSL/KD各分支梯度诊断并继续固定计划；不额外做旧域梯度优化或原PMGC虚拟步筛查。
不以测试数量作为完整性门槛；交付逐项覆盖表与真实未实现项，失败需明确工程状态。

## 10. 执行顺序与中断策略

P0：核实源码/协议/数据路径metadata/资源；写PREREGISTRATION和所有configs；code freeze；synthetic+真实smoke；通过后继续。
P1：seed0公共stage0监督训练100epoch，封存最后学生hash；独立评价REFUGE val。
P2：运行S/A/B/C/D/E各自stage1和stage2；同一臂顺序运行，相同stage0起点，独立新teacher递推；完成3×3下三角矩阵。
P3：统一汇总完整矩阵，生成所有指标、学习曲线、内存/成本和机制归因；单次终态；发布并停止。

默认仅使用此前用户允许的GPU4/5，经本轮只读检查空闲后，一卡最多一个训练进程。GPU索引映射必须验证；不杀别的用户进程、不自动占GPU6/7；若不可用则明确BLOCKED_ACCESS_OR_RESOURCE，不假设先前空闲仍有效。
以可靠服务器父进程记录child退出码；SSH中断不判训练成败。输入只读，输出NAS新目录，复用现有环境不为了版本统一重装。

允许同一exact code/config/data下的工程中断resume；恢复先核验日志/step/RNG/checkpoint，不重试到分数好为止。校准完整后若源码故障：停止受影响正式run，保存不完整证据，不在已有指标反馈下修改科学设计；补丁需新实现版本/完整回归/注明是工程修复。不得把路径问题称为科学FAIL。

若某臂运行失败，不以剩余方法表格宣布E优越；完整矩阵不足则INCOMPLETE_TRAINING_MATRIX。不可因为E stage1差就将它换成D或改变训练范围。
无新增的校准、grid、每像素正确性模型、域路由、uncertainty网络、数据合成或外部数据。

## 11. 必须交付的文件

公开只发布脱敏指标/代码/协议/哈希，不发图片、GT、私有病例ID、权重、密码或完整病例概率：

PREREGISTRATION.md/json
SOURCE_FREEZE.json
PAPER_TO_IMPLEMENTATION_DIFF.md
MODEL_AND_MEMORY_CONTRACT.json
DATA_AND_ROLE_AUDIT.json
EXPECTED_AND_ACTUAL_BUDGET.json
TEST_REPORT.json + A1_COVERAGE.json
TRAINING_COMMANDS.md
STAGE_LINEAGE.json
MODEL_STATE_ACCESS_AUDIT.json
TRAINING_COMPLETENESS.json
METRIC_DEFINITION.json
STAGE_DOMAIN_MATRIX.csv
PER_CLASS_METRICS.csv
PAIRED_METHOD_COMPARISONS.csv
SSL_DIAGNOSTICS.csv
DISTILLATION_DIAGNOSTICS.csv
SOLVER_DIAGNOSTICS.json
MEMORY_AND_RUNTIME.csv
GATE_ACCOUNTING.csv
FINAL_REPORT.md
STATUS.json
PUBLICATION_VERIFICATION.json
NEXT_STAGE_DRAFT.md

机制诊断至少区分labeled/unlabeled、rim/cup/background：
PAS coverage、有效类与空batch比例；原始KD冲突率；D被删除质量/梯度量；E投影比例、||r-q||、||r-p||、eta分布与残差；监督/SSL/KD梯度norm；solver开销；teacher hash未变。统计不用于训练门槛调节。
伪标签准确性只可在允许的val evaluator使用其GT做posthoc诊断；不得获取train_unlabeled隐藏GT来给precision。没有合法诊断则NOT_EVALUATED。

最终回答必须包含：
- 已完成哪些真实stage与optimizer更新，是否全部6臂完整；
- 每臂F/H/N/BWT/forgetting和当前逐类结果；
- E相对C/D是否有增量，B相对S是否有SSL收益；
- 每臂最大完整模型数、最终inference模型数、峰值显存与运行成本；
- 是否满足预注册筛查，未通过的具体条件；
- source SHA / report SHA / branch /匿名发布核验和NAS收据；
- seed1/2与外部数据均未启动。

## 12. 本轮之后如何止损

本轮的目的不是保证E被写成成功。
若B未从无标注数据得到信号，先承认当前SSL配置不够有效，不继续为E增加复杂巩固模块。
若C或D同样好，保留简单单前驱基线，不坚持SCD投影。
若E满足全部seed0门槛，只能进入固定配置多seed训练复核候选，仍不是SOTA或独立确认。
无论何种结果，本次没有继续搜索epoch/增广/lambda/投影边界的许可。交付能帮助决定“继续还是停止”的完整结果。

## 13. 来源定位（供复核，不是新模块清单）

- Pandey et al., Continual Segmentation under Joint Nonstationarity, arXiv:2605.20538v1，用户附件，第3–9、33–35、45页。
- Yang et al., Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation, CVPR2023 / arXiv:2208.09910。只借用同学生weak-to-strong，非完整UniMatch。
- DLwbm123/SSL_CL_seg @ c4edcfc08bad784ad4c26e12aaaf6a395cad2f95：di_dmpa_jascl/modeling.py、config.py、canonical binding和case-level scorer。
- prinshul/JASCL @ 3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53：官方随机分类器实现。

来源不覆盖的固定设计（SCD投影、训练网格、warm-up、单强增广、归一化、止损门槛）均是本协议的实验选择，不声称来自论文或已由论文证明。
