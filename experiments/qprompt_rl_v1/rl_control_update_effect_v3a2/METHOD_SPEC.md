# RL_CONTROL_UPDATE_EFFECT_STATE_V3A2
## 研究提案：原状态与动作条件化更新效应状态的受控比较

状态：PROPOSAL_NOT_EXECUTED。本文件不是已运行、外审通过或已授权推送的回执。

## 1. 决策与研究范围

继续 RL，但本轮不再启动完整闭环医学训练矩阵。问题固定为：决策前可合法计算的“候选动作将怎样改变下一步参数”是否比原16维概率统计，更能预测偏离 FINE 的五步反馈价值？

主要研究变量只有状态信息。两套状态使用同一策略学习器、先验、奖励、数据场景、模型、动作、反馈角色、优化预算和主要执行规则。新增数值重复、纯 FINE 可达和分组迁移评价均是所有方法共同采用的新评价协议，不能把相对旧 V3A 的变化全归因于新状态。

不加入 GRQA、CC、Limg、readout auxiliary loss、额外模型头、patch/region selector、旧域教师、旧图像 replay 或梯度投影。不改原始 KI 的 history-free、A侧约束、rank 或参数记忆。原始 KI 尚未绑定的字段保持未绑定，本轮不以 native reference 替代。

本轮是完整反事实反馈表上的策略价值与状态可预测性研究，不是长期 RL 或闭环分割有效性验证。它具有明确的后续 RL 用途，但全信息期望奖励优化不能被说成原 PPO 的原样复现。

## 2. 冻结来源

Repository: `DLwbm123/SSL_CL_seg`

Base commit: `e5fa35589817224d53fe37cd0e836c871a7a48d6`

建议新分支：`codex/rl-control-update-effect-v3a2`。

建议代码目录：`experiments/qprompt_rl_v1/rl_control_update_effect_v3a2/`。

保留 V2 与 V3A 的所有原始结果、失败状态、私有日志和 ledger。新协议不得追溯更改旧研究的门槛或结论。精确来源见末尾 S1–S7。

## 3. 从 V3A 得到的限制性依据

V3A h5 平均 FI_POLICY−FINE 为负，FI_POLICY 与打乱奖励、常数优化对照的平均价值接近；每个 cell 只有一个监督 prefix 和12或14个不同的 U 决策图像组。[S1,S2]

每个场景入口重新复制 teacher=student、局部 t=0；原状态含时间、学习率比例与 teacher/student 同弱视图差异。因此部分分量在入口处恒定或预期近零。必须做实际方差审计，不能据此断言所有16维都无用。[S3,S4]

当前学生是 AdamW，不是无动量 SGD。单独的原始梯度范数/余弦没有包含所有影响下一步参数位移的信息。[S5,S6]

本轮假设：加入当前参数、优化器状态和合法 L_fit/U 梯度所决定的动作差异，可能改善决策可预测性。假设尚未成立，不能声称一定提分。梯度指导训练权重存在既有研究先例；本轮不宣称这项概念本身全新，也不复现使用独立验证集的其它方法。[S7]

## 4. 学生、动作与目标全部冻结

两个学生：UNET_QUERY_128；DINOV2_VITS14_QUERY。

三个动作：0=SKIP，1=COARSE_DISC_BG，2=FINE_BG_RIM_CUP。

学生目标仍为 Lset + 0.5*LU(a)。Lset 使用既有 matching loss；非fine动作和fine动作的 KL、teacher confidence >0.7、按全图 valid 像素数归一化、coarse 合并 disc/bg 的规则全部继承 V2。

候选第一步之外的四步均为 FINE：
- skip branch: [0,2,2,2,2]
- coarse branch: [1,2,2,2,2]
- fine branch: [2,2,2,2,2]

每条分支从独立恢复的相同根状态开始。每步更新 branch-local student、AdamW moments/steps、local scheduler、teacher EMA=.99、RNG 与 cursor。五步后整体丢弃；根状态和其它分支完全不变。

原模型拟合精度 BF16、FP32 master/moments、FP32 feedback 不变。学习率从对应2000步 prefix 的真实 entry_lr 读取，采用 V2 LocalSchedule；五步使用其局部索引0–4，不能错误跳到旧场景 origin_t。所有 optimizer flags、参数组和原调度证据逐项保存。

feedback quality J 沿用 V2 clean、per-image class-balanced NLL + foreground soft Dice 的负值。相对于同场景、同重复编号的 FINE 分支：d_h(a)=J(theta_h^a,F)-J(theta_h^fine,F)，h=1,5。

h5 是唯一主要 horizon。h1 只报告诊断，不训练另一套 h1 主候选，不择优。

## 5. 开发与迁移场景

### 5.1 prefix

开发：seed261，2骨干×2域，共4个已核验 QUERY_PREFIX step2000。

前瞻策略迁移评价：seed262/263，2骨干×2域，共8个已核验 QUERY_PREFIX step2000。

这些优化种子以前使用过，不能称为“从未见过的新种子”或独立患者确认。这里的新内容是在当前协议与策略冻结后测量的场景反馈。

不重训 prefix；不以 V2/R1.6 最终学生、EMA endpoint 或 best checkpoint 替代。真实 tensor/hash/optimizer/schema 缺失时隔离受影响 cell，不猜路径或借用另一个 seed。

### 5.2 U池分组：首步和后续四步一起约束

每域构建一个开发 U pool（16个不同图像）与一个迁移 U pool（16个不同图像），二者不重叠。

开发 pool 包含原 V3A 16场景的全部首步 U 身份（预计12或14个）。用固定 hash 顺序从其余合法 U 补足16个。迁移 pool 从剩余 U 中按独立固定 hash 顺序取16个。

身份采用现有 U image-only payload SHA，先做重复字节合并，不能用序号冒充身份。若已证实的合法 distinct U 不足32，不偷偷允许重叠，报告具体数据限制。

开发场景首步 U 继续使用原16场景对应身份；四个后续 U 全部从开发 pool 生成。迁移场景首步 U 为迁移 pool 的16个身份一一覆盖；四个后续 U 也只能来自迁移 pool。

因此，开发反事实分支不能在后续步骤偷偷看到迁移池图像。这是一个新的共同场景协议，所以所有分支都重新测量，不把旧 V3A h5 数值拼入新表。

同一域/seed 的两个骨干共用相同场景序列、图像顺序、增强种子与 fold 角色。

### 5.3 16场景与反馈折

每 root cell 16场景。场景首步原始 schedule index 固定为 20*floor(60*j/16), j=0..15。保留该 seed 的 L-fit/online/audit 角色、原几何/光度随机流；按5.2替换该场景5步 U 索引。

五步内 fit/online/audit 角色不轮换，三类角色互斥；同动作比较共用全部随机流。场景 origin_t 只是抽样来源，根模型的训练进度仍是0。

用 payload digest 检查 U 独立；对 L 不夸大独立性。所有反馈图像曾被各自的监督 prefix 学习过，且开发/迁移共享同一 L cohort。禁止声称 patient-independent、external validation 或 unbiased long-term return。

开发64场景，迁移128场景，共192场景。

## 6. 核心新增状态：EFFECT40

### 6.1 两套输入及容量控制

BASE40 = [原BASE16, 24个零]。

EFFECT40 = [原BASE16, 24个更新效应特征]。

两套神经网络都为40→32→3、Tanh，总参数1411。相同 fold/init seed 采用完全相同的参数初始化；不是 BASE 用小网络、EFFECT 用大网络。零填充不表示两者可用信息完全相同，它用于控制名义架构和训练自由度。

### 6.2 根状态的三个动作梯度

在首步、任何 online/audit 访问和候选实际更新之前，使用完全相同的 augmented L-fit batch2、U弱/强视图和固定 teacher 目标，分别计算：

g_a = grad_theta[Lset + 0.5*LU(a)], a=0,1,2。

可共用前向计算图，但要分别对完整目标作VJP；不把一次单独反向和浮点相加后的梯度未经校验当作原真实更新的精确实现。

SKIP 的梯度图只有 Lset。保留每个参数的 grad=None 与精确零梯度区别。不能因为方便填零就让本来不参与更新的参数受到weight decay或Adam step递增。

### 6.3 只读 AdamW 下一步预演

从根参数theta与对应param-group的实际AdamW状态，求 theta_plus[a]。常见的非AMSGrad形式为：

m' = beta1*m + (1-beta1)*g_a
v' = beta2*v + (1-beta2)*g_a^2
n' = n+1
theta_plus = (1-lr*weight_decay)*theta - lr*(m'/(1-beta1^n'))/(sqrt(v'/(1-beta2^n'))+eps)

实际实现必须以目标环境已锁定版本的 optimizer 语义为准，覆盖 flags、偏差修正、参数组学习率、None、step tensor设备等。禁止把 SGD 的 -lr*g 叫作 AdamW 更新效应。

预演不能修改、安装或优化真实参数；临时结果只用于统计特征，用完删除。使用 functional API 时，注意其可能原地修改传入参数和状态；全部输入必须为不共享存储的 scratch clone。[S6]

预演是完整的优化器等价向量计算，不是免费操作。单独计数 virtual_adamw_previews，不伪报为0计算，也不与真实optimizer.step合并。

定义 Delta_a=theta_plus[a]-theta，delta_a=Delta_a-Delta_FINE。后者是与FINE相比的首步参数变化。参数/增量采用FP32预演；统计点积/范数采用FP64分块归约，避免大向量先低精度累加。不能在数学公式中忽略不一致的参与参数而假定weight decay完全抵消。

### 6.4 当前 L-fit 的clean quality方向

只用首步拟合batch相同的两个L图像身份，读取其无增强版本，计算：

g_Q = grad_theta[-J(theta, L_fit_clean_batch2)]。

这里所有图像都属于当前 L_fit，绝不读取 F_online/F_audit、旧域图像或 U标签。J 的形式虽与feedback评分相同，但数据角色不同。这是合法的决策前有标注计算，不是独立泛化证据。

g_Q路径使用eval模式、FP32且启用autograd，只对上述L_fit求梯度；不能调用现有inference_mode评分包装后声称取得梯度。真正的online/audit评分仍严格no_grad。前后恢复模型mode、buffers与RNG。

禁止用候选更新后的 J、实际theta_h、reward或audit梯度作为状态输入。禁止将g_Q用于直接更新学生。本步不作二阶反传。

### 6.5 固定24维

划分两个不重叠参数块：BODY = `body.*`（UNet）或`backbone.*`（DINO）；HEAD = 其余原本可训练参数。每个参数恰好属于一个块。

对每个 a in {SKIP,COARSE}、每个块 b，固定六维：

1. `score = -dot(g_Q_b, delta_a_b)`：clean-fit质量的一阶相对变化预测。
2. `direction = cos(-g_Q_b, delta_a_b)`；分母为0则置0并将第5维标0。
3. `step_ratio = log1p(norm(delta_a_b)/(norm(Delta_FINE_b)+1e-12))`。
4. `grad_ratio = log1p(norm(g_a_b-g_FINE_b)/(norm(g_FINE_b)+1e-12))`。
5. `direction_valid`：g_Q与delta的范数均非零。
6. `reference_norms_valid`：Delta_FINE与g_FINE范数均非零。

6×2动作×2参数块=24维，拼接原16维得到40维。完整索引在JSON中。

这个一阶预测主要对应当前clean L_fit的即时变化，不是h5 audit真值，也不保证动作排序正确。h1只用于解释一阶近似与延迟反馈的差异。

### 6.6 特征重复、归一化与访问隔离

每场景从相同root独立提取三次完整BASE/EFFECT特征，先保存每次值与数值方差，再取算术平均作为唯一输入。每次提取四次VJP（3个动作目标＋1个clean-fit目标）和3次虚拟AdamW预演。

主要神经输入与ridge输入均只用训练折/训练seed261数据估计连续列的mean/std。std下限1e-8，训练中完全恒定列在训练和测试都置零；布尔有效位保持0/1，不拟合阈值，不按audit clip异常值。BASE最后24维保持零。

特征进程没有online/audit标签访问权，也不能打开这些场景的分支反馈文件。转移阶段所有特征、预测动作和概率先冻结哈希，再允许评分器生成新反馈。

特征本身携带训练数据衍生信息，完整向量/梯度/优化器状态只存私有位置，公开仅匿名均值、方差和成本。

## 7. 相同的策略与执行规则

所有神经策略使用普通softmax：
pi=softmax(log([.05,.05,.90])+f_phi(z))。

不加epsilon混合，不设置任一动作的概率下界。输出层初始零，初始分布是注册prior。

训练目标：
L=-mean_s sum_a pi(a|z_s)*stopgrad(d_online,h5(a)/scale) + .001*mean_s KL(pi||prior)。

scale沿用V2 seed261对应backbone/domain的固定值与SHA，不按本轮audit重新估计。finite大logit使用log_softmax稳定计算KL，不能因概率下溢产生0*log(0)。

优化器 Adam，lr=.003，betas=(.9,.999)，eps=1e-8，weight_decay=0；每fit完整full-batch固定256步。不能挑best epoch或根据OOF调beta/lr。

主要执行策略为确定性argmax。用pre-softmax/log-policy scores取最大值，在绝对1e-8的并列范围内按FINE→COARSE→SKIP；因此可以真正执行100% FINE。普通softmax期望价值作次要结果，不能事后在两种规则中挑正者。

这些共同规则不同于V3A的混合概率评价。只有新实验内部EFFECT对BASE的配对差异才检验状态信息；不能将全部变化归因于新特征。

KL是软偏好不是安全保证；argmax也不是可靠干预证书。不得宣称“有prior所以不会伤害FINE”。

## 8. 八个预注册方法

1. FINE：永远动作2，所有相对价值定义为0。
2. NC_OPT：无上下文，按训练折平均normalized收益精确求解KL正则化常数策略。pi ∝ prior*exp(mean_d/beta)。不再使用概率下界；主要动作按第7节取argmax。
3. BASE_RIDGE：原状态零填充，ridge预测三动作normalized收益，fine列固定0。训练列标准化，目标MSE按N*3平均，固定ridge=.01，截距不惩罚。
4. EFFECT_RIDGE：相同ridge，用更新效应状态。
5. BASE_POLICY：相同40→32→3网络，输入BASE40。
6. EFFECT_POLICY：输入EFFECT40；唯一主候选。
7. EFFECT_SHUFFLE：同网络、同初始化，固定打乱训练行与完整三动作reward向量之间的对应；保存置乱数组hash。仅一个固定置乱负对照，不进行显著性置换检验。
8. EFFECT_RULE：只用合法first-order score，两块相加后选择预测值>0的最大非fine动作，否则fine；并列遵循同一规则。无reward训练、不调阈值。

ridge的predicted value按相同prior、beta转为解析概率并执行argmax；禁止为某种学习器单独选温度。STATIC_PRIOR的随机期望值可描述性附报，不需要另一个医学分支。

新增梯度信息若仅让RULE或ridge变好，应描述为任务效应信息有用，不能宣称RL特有优势。

## 9. 训练/预测/评分顺序

### 9.1 开发

完成seed261的64场景特征和三重复完整反馈表。用三重复均值作为训练目标。每cell基于首步U身份做固定4折OOF，同U的重复场景绝不跨折。

每fold训练BASE_POLICY/EFFECT_POLICY/EFFECT_SHUFFLE各1个：4cells×4folds×3=48 fits。

每fold拟合NC_OPT/BASE_RIDGE/EFFECT_RIDGE各1个。所有训练只使用online奖励；audit只在对应预测冻结后评分。

OOF用于解释，不用于选择超参数、输入子集、horizon或主要候选。

### 9.2 冻结最终策略

无论开发OOF正负，都用每cell全部16个seed261场景重新从相同初始化规则训练3个神经模型：4×3=12 fits。另有每cell的3个解析/回归fit。

总神经fit=60，总controller更新=15,360。总解析/回归fit=60。

冻结全部模型、scaler、奖励scale、配置、特征schema、决策规则与源代码SHA。不能在seed262结果后修改seed263。

### 9.3 前瞻迁移评价

seed262/263对应的8个prefix，仅提取合法决策前特征。相同backbone/domain使用在seed261冻结的模型，不在262/263再做策略fit。

先生成并封存所有8方法的128场景动作与概率；之后独立进程才生成/读取这些场景三重复分支的online/audit质量。

迁移online奖励只用于事后诊断，不训练或修正预测；audit永不进入控制器。

这检验“对新prefix与预划U池的策略迁移”，不是先让策略在每个新seed在线学习后的上限。若OOF有效而迁移无效，要区分状态信息的局部价值与跨prefix稳定性。

## 10. 三重复测量与数值解释

全部192场景×3动作×3次完整重复×5步=8,640真实学生optimizer调用；不将已付旧V3A成本当成本轮新的计费。1728条完整分支全部临时丢弃。

每个重复从同一完整root/RNG恢复，保持data/augmentation随机种子不变；重复测的是执行数值波动，不是新的采样增强。不以模型输出为依据更换图像或重复次数。

动作执行顺序固定循环平衡：重复0=[0,1,2]，重复1=[1,2,0]，重复2=[2,0,1]。所有counterfactual均从root重启，不互相继承。

每个预测冻结后，分别在三套audit反馈上计算其原始价值与配对contrast，再取均值。报告每场景差值跨度、每cell价值跨度和总体配对跨度。

对任一比较X，注册数值解释带：band_X=max(1e-6, 5*max_{r,r'} abs(V_X,r-V_X,r'))。cell和总体分别计算；同时保留原V3A样式的worst-scene重复误差，不用均值掩盖局部噪声。

这不是置信区间、标准误或患者独立性证明。只有三次重复也不能精确估计所有CUDA随机性。数值噪声大时标记INCONCLUSIVE_NUMERICS，不把结果置零、不删除cell、不说算法无效。全部已注册cell继续完成；数据/状态错误仍必须修复或隔离。

精度、CUDA确定性策略保持全组一致。不因某域表现差独自切FP32；若必须改数学路径或精度语义，原新表受影响部分作废并按新协议处理，不混报。

## 11. 主要评价与解释

唯一主要数据集：seed262/263的8个cell、每cell16个场景、h5、已密封的确定性策略。

主要对比有固定次序：
A. EFFECT_POLICY−BASE_POLICY：新增状态是否有增量。
B. EFFECT_POLICY−FINE：实际干预是否有净价值。

两个对比都必须满足才能将新状态策略提议进入闭环pilot，不能只击败有害BASE。

先每场景三重复平均，再cell内均值，再等seed/等backbone/等domain平均。不得按U图像量或模型规模隐式加权。学习方法、oracle和online selector的值分开呈现。

描述性研发投入条件（不是统计显著性）：
- 总体A、B均至少+1e-5反馈质量单位，并且各自超过对应数值解释带；
- B在两个backbone各自平均均为正；8cell至少6个的B超过本cell数值解释带；
- EFFECT_POLICY对NC_OPT及EFFECT_SHUFFLE的总体增量均超过数值解释带；
- 完整报告对EFFECT_RIDGE/EFFECT_RULE的差异。若未优于二者，不提出RL专属优势；
- 不允许省略数值不确定cell后重新宣称通过。

1e-5是事前决定是否值得再投入的工程/科研量级目标，不是可换算的Dice门槛；通过也不能保证闭环Dice改善。

必须报告干预率、不同动作的效用、负向干预质量、返回FINE的比例、scaler外推幅度、原16维/新24维的有效变化，以及标准化前后的列方差。

不得把贪心与softmax两个评价口径、h1/h5两个horizon或不同方法中较好的一个改成主要结果。

## 12. 资格与工程恢复

先做只生成数据的资格：
- 拷贝/恢复每个optimizer tensor，CPU step无共享storage；
- 实际AdamW与只读preview在同一梯度、同一state下的theta/m/v/step核验，特别比较delta_a-delta_fine，不能只用大theta的相对误差遮住小更新误差；
- 如preview使用functional，检查其scratch不泄漏到根模型；
- 三动作grad=None、零U mask、参数组lr、weight decay、已存在n=2000 moments；
- first-order score符号：已知二次目标的线性近似方向；
- 特征提取前后根权重/optimizer/teacher/RNG/mode/.grad/buffers/cursor全部相同；
- 代码权限测试：effect extractor打开online/audit或候选reward文件时明确拒绝；
- 拟合scaler不读测试特征分布；模型不读测试奖励；
- FINE在主要argmax执行中可完全达到；零输出初始prior正确；
- 合成始终FINE最优、状态切换、状态无信息三类行为正/负对照；
- 新分支checkpoint/报告atomic发布、结束标记、deadline与重复恢复。

AdamW preview身份资格的预定目标：每个step向量与真实克隆更新的L2误差≤1e-5*norm(actual_step)+1e-8，并报告每参数块最大绝对误差与relative-action-delta误差。保留实际误差，不根据反馈成绩放宽；不符合时先修计算路径，不把不一致preview称为“精确”。合成资格不足以保证真实分支确定性，后者由三重复单独评价。

最多64个合成student optimizer调用、4096个合成controller调用，不用真实L smoke。精确当前库API由已安装版本解析，不能为了方便升级环境；不要求一定存在main文档里的新API名称。

发生相同版本同一错误两次，暂停受影响路径，回归后恢复，不让全部worker反复触发错误。math/action/reward/state/precision变化必须记录scientific-path修改及作废范围。无性能驱动调参，未完成完整矩阵不得作完整比较。

## 13. 完整预算

真实学生：64开发×45=2,880；128迁移×45=5,760；合计8,640。所有为临时分支，保留医学模型更新0、新医学终点0。

特征：192场景×3重复=576次提取；4VJP/次，共2,304次；3虚拟AdamW/次，共1,728次。使用共享图但全目标VJP语义固定。记录真实student/teacher前向、反向时间、峰值内存，不能把虚拟预演藏在“0optimizer成本”里。目标最多每次5个student前向（weak U、base用L-eval、augL-fit、strongU、cleanL-fit）和1个teacher前向；未融合的重复调用如实计费。

神经fit：60×256=15,360。解析/回归fit60，预测次数按实际保存。

新增上限：学生恢复/重放864；controller恢复/重放1536；virtual preview恢复173；额外feature提取58（相应VJP/前向单记）。合成student64、controller4096。真实L smoke0。

预算不能互相转换成新增种子、horizon、loss、fit步数或medical endpoints。第一次CUDA资格开始最多12小时，完整队列完成可提前结束。窗口到达保存完整断点、封存未完成事实，不重置时钟。不存在为“跑满12小时”自动加实验。

新run逻辑空间上限32GiB，借用的不可变旧prefix只读引用，不复制全部历史checkpoint。原始梯度/虚拟全参数张量每场景用完删除，不批量跨场景缓存。合法可用GPU内调度，不抢占其他用户进程。

## 14. 交付与后续

至少交付：METHOD_SPEC.md、EXECUTION_PLAN.json、FEATURE_SCHEMA.json、数据/前缀/分组私有绑定及匿名摘要、ADAMW_PREVIEW_PARITY、STATE_INVARIANCE_AUDIT、FEATURE_VARIATION、DEVELOPMENT_OOF、FINAL_POLICY_LOCK、TRANSFER_PREDICTION_LOCK、HORIZON_TRANSFER、REPEAT_ERROR、TRANSFER_POLICY_VALUES、MAIN_CONTRASTS、ALL_COSTS、PATCH_LOG、FINAL_INTERPRETATION。

最终明确回答：新特征是否在入口有变化；它与h1/h5收益关系如何；同模型下是否改善预测；只是在训练/OOF有效还是可迁移；简单规则/回归是否已经解释全部收益；结果是否超过数值分辨率；是否有策略回到FINE而无净增益；是否满足提议闭环pilot的预注册标准。

即便全部通过，也只新提议后续小规模真实闭环FINE/BASE/EFFECT/强非RL对照，不自动执行。所有原始KI绑定和历史参数记忆保留约束不变。

用户当前请求的是实验计划。执行须当前明确授权；push须另行确认。已有V3A的publication permission不授予本研究运行或发布权限。没有原始代码审阅就不得写“外部审阅通过”。

## 15. 来源

S1–S5均来自上面的固定commit，路径分别为：
- S1 `experiments/qprompt_rl_v1/rl_control_fine_intervention_v3a/reports/FINAL_INTERPRETATION.md`
- S2 `experiments/qprompt_rl_v1/rl_control_fine_intervention_v3a/reports/OOF_POLICY_VALUES.csv`
- S3 `experiments/qprompt_rl_v1/rl_control_fine_intervention_v3a/execution.py`
- S4 `experiments/qprompt_rl_v1/rl_control_bandit_v2/method.py`
- S5 `experiments/qprompt_rl_v1/r1_12h/core.py`

算法背景链接（不是用户当前环境的版本锁）：
```text
S6 https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html
S6 https://docs.pytorch.org/docs/main/generated/torch.optim.functional.adamw.html
S7 https://proceedings.mlr.press/v80/ren18a.html
```

本计划中的新状态、数据分组、argmax协议、重复次数、规模及门槛都是研究提案，不是上述来源已验证的医学收益。
