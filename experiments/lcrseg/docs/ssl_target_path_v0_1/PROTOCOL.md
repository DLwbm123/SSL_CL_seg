# SSL_CL 下一轮：教师监督路径双因素检验 V0.1

**执行代号：`SSL_TARGET_PATH_V0_1`**  
**定位：基础机制对照，不是新方法发布，不是 JASCL/FixMatch 完整复现。**  
**优先级：有效性 → 简洁性与可解释性 → 创新性 → SOTA。**

## 0. 执行摘要和本轮授权

从以下已公开提交建立独立分支：

```text
repository: DLwbm123/SSL_CL_seg
base_commit: 60843e85c110f31863299639822b080cf4c4ba9b
new_branch: codex/ssl-teacher-target-path-v0-1
source_namespace: experiments/lcrseg/ssl_target_path_v0_1/
test_namespace: experiments/lcrseg/tests/ssl_target_path_v0_1/
doc_namespace: experiments/lcrseg/docs/ssl_target_path_v0_1/
```

本轮检验两个清楚的问题：

1. 双端置信度掩码是否把教师可能纠正学生的像素一起排除了？
2. 在相同软目标下，概率 MSE 与软交叉熵的差别，是否转化为最终分割收益？

执行一个固定的 **2×2 因素矩阵 + 监督对照**，在两个域、三个预先指定优化种子上全部训练完成。不得先看一个种子再决定是否运行其余种子。

- 两个域独立：`RIM_ONE_r3`、`Drishti_GS`。不构成持续学习序列。
- `data_split_seed=0`，`optimization_seeds=[31,32,33]`。
- 五臂：`SUP`、`J_MSE`、`T_MSE`、`J_SCE`、`T_SCE`。
- 唯一预先指定主候选：`T_SCE`；另外三种 SSL 配方是机制对照，不允许外层换冠军。
- 每任务 100 epochs；RIM 3,200 更新，Drishti 2,100 更新。
- 30 个独立任务，正式更新总预算 **79,500**。本轮不共享或拼接已训练 warm-up，不复用旧权重。
- 全部训练后统一作主结论。早期指标只记录，不选择 checkpoint、不改配方。
- 不恢复 PAS、GAS、L05、SCD、历史 KD、路由或分类头搜索；不进入 CL。
- 最多当前学生与当前 EMA 两个完整模型；最终部署只有学生。
- 不访问 test、train-unlabeled 隐藏 GT、旧 formal_03 或历史训练回放。
- 允许本文件明确规定的训练、当前 train-labeled / train-unlabeled 图像读取和隔离 val 评价；不修改旧锁。

**这不是承诺能够成功，而是基于新诊断的一次有限检验。** 不允许把局部梯度变大、被接受纠错像素变多或单个种子改善，直接判为分割有效。

## 1. 决策依据：事实、解释和假设必须分开

### 1.1 已有结果事实

最新 `ssl_head_control_v0_1/FINAL_REPORT.md` 已完整完成 14 个任务、37,100 步训练和 70 次固定诊断，无正式工程失败。

seed11：

| 配方 | 两域平均最终学生 Dice Q | 相对 LIN_SUP |
|---|---:|---:|
| LIN_SUP | 0.716793（展示精度） | 0 |
| LIN_MT_CONF | 0.712439 | −0.004354 |
| LIN_MT_PAS | 0.714282 | −0.002511 |

线性监督头的 seed21/22 相对 SUP_G1 平均增益为约 +0.018072，但 seed22 平均差约 −0.002053，RIM 差约 −0.013910，RIM cup 差约 −0.032346。因此旧复核未通过。**不把这项结果误述为线性头平均无效或全面劣于旧头。**

等数量 PAS 审计没有显示稳定的原型筛选优势。例如旧 G1/Drishti/rim/training_like 下：PAS precision 0.794709，同数量 confidence top-k 为 0.807888，random 为 0.770301。报告中的小幅反例和其它全部组仍须保留。

### 1.2 最值得检验的新证据

公开 `CORRECTNESS_STRATA.csv` 的 LIN_MT_PAS、Drishti、epoch100：

| 评价条件 | 筛选前“教师对、学生错”平均像素/患者 | 实际 PAS 接受的该类像素 | 实际 PAS 接受的“两者同样错”像素 |
|---|---:|---:|---:|
| clean | 135.60 | 0 | 6,879.56 |
| training_like | 674.42 | 104.96 | 6,384.35 |

training_like 的纠错机会保留比例约为 15.56%。这些是 val 的患者平均、预测 draw 先在患者内汇总；不是独立像素样本数，也不是 train-U 精确发生率。

这支持检验“纠错监督可能被筛掉”的假设，但不证明它是唯一原因。需要强调：

- 现有 joint-confidence 是“学生和教师分别都高置信”，不是要求类别预测相同。
- 丢失可能发生于 teacher-confidence、student-confidence 或 PAS 相似度条件，旧汇总不能唯一分解；新诊断必须分解。
- 两者同样错且软分布也相同的像素，任何单纯一致性损失都不能凭空获得正确标签。
- 对象是 val 诊断；不把高 confidence 当成教师正确性保证。

### 1.3 本轮假设

在保留同一线性头、EMA、数据、弱扰动、阈值与训练预算的情况下，去掉“学生也必须高置信”的准入条件，并用对 softmax 饱和不再额外乘 Jacobian 的软交叉熵，可能提高有益的教师纠错信号。

**不预设一定成立；更强纠错梯度也可能更强地传播教师错误。**

## 2. 与参考论文的关系

JASCL 附件第5页公式(3)–(5)包含原型验证和学生/教师有效集合交集上的概率 MSE。本轮旧式对照沿用 joint confidence 和 MSE，但移除了已单独审查的 PAS；不是指认 JASCL 实现错误。

FixMatch 的参考思想是：目标由弱视图的高置信预测产生，用于监督另一视图；目标侧可信度不等同于要求受监督分支已经高置信。本轮不采用其完整算法：使用 EMA 教师、软标签、现有 Gaussian 扰动，而非原论文完整 hard-label 强增强方案。因此只能称为“目标侧筛选思想的受控检验”。

不加入 BCP、CutMix、强颜色增强、新骨干、Dice loss 或额外预训练。本轮若同时改这些因素，就无法解释两项主假设。

## 3. 固定实验矩阵

J = joint mask；T = teacher-only mask；SCE = soft cross-entropy。

| arm | 无标注掩码 | 无标注损失 | 角色 |
|---|---|---|---|
| SUP | 无；不打开 U 图像 | 0 | 匹配监督对照，仍维护 EMA 作对称诊断 |
| J_MSE | 双端 confidence | 概率 MSE | 当前旧式基线；对应旧 LIN_MT_CONF 的学习行为 |
| T_MSE | 仅教师 confidence | 概率 MSE | 只改变监督准入 |
| J_SCE | 双端 confidence | 软交叉熵 | 只改变目标损失 |
| T_SCE | 仅教师 confidence | 软交叉熵 | 唯一预指定主候选 |

两项因素的训练轨迹自然会不同。固定输入的数学对照和完整训练的配方效应分别报告，不把后者说成单像素因果证明。

## 4. 数学与实现合同

### 4.1 模型输出

对当前域 U 图像，同一几何变换得到 x。沿用旧输入扰动：

```text
teacher input = x
student input = clamp(x + Normal(0, 0.02^2), 0, 1)
```

GAS 关闭。无随机头、无 dropout 新增、无图像 mixing。

设：

```text
z = student(student_input)
p = softmax(z)
v = stopgrad(teacher(teacher_input))
q = stopgrad(softmax(v))
```

teacher 固定 eval，参数不进入 optimizer、无梯度；使用当前 EMA 而不是前驱教师。

### 4.2 掩码

```text
m_T = geometry_valid & (max_c(q_c) > 0.7)
m_J = m_T & (max_c(stopgrad(p_c)) > 0.7)
```

所有比较严格大于；所有掩码 detached。

- 不要求学生和教师 argmax 一致。
- 不使用原型相似度、类别阈值、样本 ID、真实域身份或 GT 选择 mask。
- train-U geometry 仅来自合法图像/变换支持，不能借隐藏 GT 构建 ignore mask。
- 当前 labeled GT=255 排除于监督 loss；val GT valid-mask 只供评价，不改变待评估掩码。

### 4.3 无标注损失

每个 batch 采用各臂自己的 accepted-pixel 分母，与旧式基线一致；不再除以类别数：

```text
D = max(sum(m), 1)
MSE = sum_i m_i * sum_c (p_ic - q_ic)^2 / D
SCE = -sum_i m_i * sum_c q_ic * log_softmax(z_i)_c / D
```

SCE 梯度等价于 KL(q||p)，但值多一个教师熵常数；训练日志必须同时报告 SCE 与去熵后的 KL，不能直接拿原始 SCE 与 MSE 数值判断谁更难优化。

实现 SCE 用 `log_softmax`，不先对 p 取 log，不夹小 p、不锐化 q、不让 q 获得梯度。教师熵诊断使用 `xlogy` 或等价 0log0=0 安全计算。

empty mask 返回连接学生计算图的严格零；不退回 all pixels，不跳 batch，不改变实际步数。

### 4.4 总目标

```text
loss = supervised_CE + lambda_cons(epoch) * unsupervised_loss
lambda_cons(e) = 0.5 * min(1, max(0, (e - 20) / 20))
```

epoch 从1开始，1–20无 U 图像访问。SUP始终不访问 U 图像；只可用其数量 metadata 对齐步数。

不加入Dice loss、class weights、KL温度、温度拟合、自适应loss权重、梯度裁剪或投影。

### 4.5 数学性质与不能声称的结论

温度1、固定 detached q 和 mask 时，单像素未加权梯度：

```text
g_MSE = 2 * (diag(p) - p p^T) * (p - q)
g_SCE = p - q
```

MSE在p饱和时可能进一步抑制修正；SCE没有该额外Jacobian因子。

合成见证（独立算术，不是模型实验）：

```text
p = [0.999, 0.001]
q = [0.001, 0.999]
g_MSE = [0.003988008, -0.003988008]
g_SCE = [0.998, -0.998]
```

范数比约250.25。但当p=q时两者都为零；教师错时SCE也可能加强错误。局部logit梯度不等于全网络参数梯度，更不保证Dice上升。

本实验保持相同名义权重，因此“loss替换”的配方效应同时含梯度形状和尺度差别；不得仅凭结果把两者完全分离。不在结果后增加一个调到刚好相等的MSE系数。

## 5. 模型与初始化

固定使用 `ssl_head_control_v0_1` 已验证的 raw linear 3×3 分类头：

- UNet通道16/32/64/128、GroupNorm和原解码器不变。
- 输出头无特征归一化、无权重归一化、无乘10、无GAS。
- 保留3×3、无bias、无padding和原插值几何。
- `sigma`、`grad_update` 为兼容惰性状态；冻结并计入内存，不进入Adam，不估计GAS。
- 所有实际有效的网络权重可训练，不新增冻结层。
- 每个(domain, optimization_seed)内五臂初始化字节一致；前20epoch监督warm-up也应一致。
- 原官方构造器固定head seed的行为沿用，明确记录“优化种子改变主干初始化/采样流，兼容head初始权重相同”的边界，不偷偷改旧seed定义。
- 不加载旧最终模型、旧EMA、最佳epoch模型或外部预训练权重。

选择线性头只是固定一个透明、无采样的控制条件，不声称它已经稳定胜过GAS头。旧头比较已收尾，不重开。

## 6. 数据和优化合同

固定 `data_split_seed=0`，复用已核验 manifest 与 split；先记录实际 SHA：

| 域 | train-labeled | train-U | val | steps/epoch | formal updates/task |
|---|---:|---:|---:|---:|---:|
| RIM_ONE_r3 | 16 | 63 | 40 | 32 | 3,200 |
| Drishti_GS | 10 | 41 | 25 | 21 | 2,100 |

- 每域独立从随机初始化训练，域间不共享权重或原型。
- batch size L=2、U=2；`steps=max(ceil(NL/2),ceil(NU/2))`；短流循环，末尾batch不丢。
- Adam lr=1e-3，betas=(0.9,0.999)，eps=1e-8，weight_decay=4e-5。
- 第k步 lr=`1e-3*(1-k/K)^0.9`，k从0开始，K为该任务全部更新。
- EMA=0.99；有效浮点参数按旧规则更新；惰性状态按旧head实现复制。
- geometry沿用成对H/V flip各0.5、rot90；input noise仅学生U使用0.02。
- 不改增强、label fraction、重采样策略、训练epoch、head或任何阈值。
- 维持PyTorch/CUDA及确定性设置，禁AMP/TF32等未登记变更。
- 每个随机流包含 optimization_seed、domain、epoch、step、stream、sample position；不含arm。
- 使用既有canonical `safe_asset(DATA, relative) = DATA/h5/v1/relative`，不发明fallback路径。

## 7. 执行顺序与预算

### P0：源码和合成资格

通过以下最小但有针对性的验证，然后实际训练，不另立无限review gates：

1. J_MSE固定输入前向、loss、mask与旧LIN_MT_CONF一致；保持旧源码不改。
2. T-mask包含J-mask；teacher高置信/student低置信时仅T接受；teacher低置信时两者都拒绝。
3. 两者高置信且类别不同仍允许J；不要误写成agreement mask。
4. SCE的autograd与`(p-q)*m*lambda/D`一致；MSE与解析式一致。
5. 上述饱和输入、p=q、q含零、空mask、全ignore-L、非法类别/形状均有测试。
6. teacher/EMA/target/mask无梯度；全零U系数回退监督梯度，不退回其它loss。
7. 有效分母是accepted pixels，不除C、不按类别归一化。
8. 在固定合成HDF5上五臂warm-up的student/EMA/optimizer/采样序列一致。
9. canonical-only路径成功，错误根目录单独存在失败。
10. 中断/恢复包含head mode、student、EMA、optimizer、RNG、顺序和counter；部署无teacher仍可运行。
11. 验证诊断不修改训练state/RNG；不创建第三份完整模型或CPU shadow。
12. 混合/非整数类型/非finite输入明确失败；不是科学失败。

本地和服务器在相同精确源码上测试；不要求凑测试数量。资格总合成成功更新上限1,000，全部尝试计数。真实梯度smoke最多每域5步、合计10步，只用该域排序前两例train-labeled；可隐藏其标签作为伪U，但GT仅用于隔离正确性核验。smoke权重不用于正式初始化，失败不挑另两例。

### P1：固定30个完整任务

预先登记全部：`seeds31/32/33 × domains2 × arms5`。

- 全部运行，不以seed31表现决定seed32/33准入。
- 主候选T_SCE不变，不能用某个更好的消融替换主候选。
- 任务间独立，可并行。默认使用已有授权的GPU4/5；GPU6/7仅在服务器已有明确用户授权时使用。
- 每卡一个训练进程，至少4GiB可用；不停止其他用户进程。
- 最终epoch100学生固定为主输出；不按val选best，不切EMA，不TTA。
- 每臂在20/40/60/80/100做固定诊断。epoch20尚未训练U，记录权重0。
- 30×5=150个固定快照；额外资格/诊断前向不混入正式更新。

预算：

```text
new_formal_updates = 3 * 5 * (3200 + 2100) = 79,500
previous_reported_formal_updates = 168,608
all_attempt_formal_total_if_no_new_failure = 248,108
```

不共享计算后的warm-up checkpoint，不把逻辑重复预算冒充实际；仅共享确定的初始化来源。每条trajectory完整跑完100epoch。

真实工程错误停止受影响trajectory；其它已冻结独立任务可继续。不可见val结果后改源码重跑该臂、补样本或挑种子。报告INCOMPLETE而非科学FAIL；另案修复须明确新授权。

### P2：统一判定和收尾

完成全部可执行固定任务后再生成主统计。无任何自动CL、更多seed、更多loss、强增强、teacher设计或温度搜索。

## 8. 必需诊断：检查纠错通路，不再堆无关文件

### 8.1 同网络、同输出的掩码分解

在每个诊断快照同一组p/q上，全部计算：

```text
raw geometry
teacher-only confidence
joint confidence
```

不重新运行网络来比较掩码。先固定预测/掩码及哈希，再读取val GT计算正确性；不使用GT决定掩码、像素数或阈值。

按教师预测类和真实类分别报告计数；background/rim/cup不能混成一个高精度数字。至少包含：

- both_correct
- teacher_correct_student_wrong
- teacher_wrong_student_correct
- both_wrong_same_class
- both_wrong_different_class

报告每类原始支持、T-mask接受、J-mask接受，及每一步损失的纠错机会保留率。特别分开：

```text
teacher-confidence rejects / raw teacher_correct_student_wrong
student-confidence further rejects / T-eligible teacher_correct_student_wrong
```

不再用旧PAS汇总推断是哪个条件导致丢失。

### 8.2 软分布与局部梯度

对同一p/q计算四种目标的解析logit梯度。计算两种版本：

- 未加权单像素向量，用于比较损失几何；
- 应用真实mask、该batch分母和lambda后的向量，用于记录实际局部权重。

必须报告teacher与student argmax相同但分布不同的数量，不能把“同样错”全部当成零梯度。单独记录p≈q的分布距离。

在eval GT下可以计算局部CE方向`g_GT=p-e_y`，与`g_U`的dot/cosine；它只能解释单像素一阶方向。**禁止将val GT构造训练梯度或更新模型。**

额外真实诊断backward为0；生产训练自身的loss.backward照常计数。解析式用合成autograd测试验证。

报告teacher-only扩展后是否同时增加Tw/Sc（潜在损伤），避免只看“纠错机会变多”。支持为空时显式undefined，不填完美分数。

### 8.3 分割和概率质量

每个快照：student/EMA分开；两域分别；rim/cup分别；主指标沿用每患者rim/cup宏平均Dice，GT255语义不改。

记录NLL、Brier、confidence、pmax>0.99、错误前景confidence，以及precision/coverage/accepted-correct recall。二者都是本次val评价，不是隐藏U标签质量。

`teacher-only正确率`不够时不可调阈值；如实说明更强SCE可能传播错误。

### 8.4 训练类别贡献

实际训练按教师预测类记录accepted count、raw loss、weighted loss以及解析local logit gradient。SCE同时记录target entropy和KL值。不从像素占比直接推断背景参数梯度支配。

不要新增risk head、校准器、原型统计、模型复制或后验病例规则。

## 9. 评价和科学判定

设`R[s,d,a,c]`为seed s、domain d、arm a、class c的最终学生患者平均Dice：

```text
D[s,d,a] = (R_rim + R_cup)/2
Q[s,a] = mean_d D[s,d,a]
Delta[s] = Q[s,T_SCE] - Q[s,SUP]
Delta_old[s] = Q[s,T_SCE] - Q[s,J_MSE]
```

三种子都预先指定，不设一个被选中的发现seed；报告全部差值、均值、sample SD和范围。当前split有多轮研究者暴露，训练新种子不是独立患者确认。

### 9.1 工程维度

全部30个完整任务、150个固定诊断、30个单学生部署与input/step/teacher隔离核验完成才标`ENGINEERING_COMPLETE`。否则缺失单列，不补算、不以中间epoch代最终。

### 9.2 主候选实践有效性门槛（预先固定）

以下同时满足才标`TARGET_PATH_SSL_DEVELOPMENT_SIGNAL`：

1. `mean_s Delta[s] >= 0.010`。
2. 每个预定seed的`Delta[s] > 0`。
3. `mean_s Delta_old[s] >= 0.005`。
4. 每个seed/domain，T_SCE相对SUP的宏平均变化 >= −0.005。
5. 每个seed/domain/rim或cup，T_SCE相对SUP变化 >= −0.020。
6. 无内存、权限或训练完整性违规。

未满足则逐项报告，不降低门槛、不把弱正信号说成零作用；均值正而某类别/seed不稳定时，标明取舍。科学未通过不是工程失败。

本轮不要求T_SCE超过所有历史随机种子的最佳数字，也不借较弱的历史baseline宣称SOTA。历史LIN_SUP/SUP_G1分布只作上下文。

### 9.3 因素贡献必须单列

每个seed/domain及三seed汇总：

```text
mask_effect_MSE = T_MSE - J_MSE
mask_effect_SCE = T_SCE - J_SCE
loss_effect_joint = J_SCE - J_MSE
loss_effect_teacher = T_SCE - T_MSE
interaction = (T_SCE-J_SCE) - (T_MSE-J_MSE)
```

这些是完整训练配方效应；不把交互等式称为因果识别。若简单T_MSE等臂同样好或更好，保留其观察但不把它替换成预注册T_SCE的胜出结果。

独立列出`CORRECTION_OPPORTUNITY_RETENTION`、`TARGET_ERROR_PROPAGATION`等描述性机制结论。局部诊断变好而Dice不升，必须写“机制指标不能替代模型收益”。

### 9.4 不确定性

若存有必要per-patient metrics，可用固定2000次患者paired bootstrap给cohort不确定性，种子内保持两策略配对、跨种子同患者identity一起重采样；每域内抽样，不把4个draw或像素当独立患者。

训练随机性由3个seed的差值单列，不能用患者bootstrap充当重训练置信区间。不开启事后显著性测试来推翻预定实用门槛。

## 10. 内存、部署和隐私

- 每个任务最多student+currentEMA两份完整模型；SUP也维护EMA，供对称对照。
- CPU full shadow/checkpoint clone也属于完整权重，不得藏在CPU规避两份模型约束。
- 诊断只读训练状态，前后student/EMA/RNG/head/optimizer hash相同；不保留跨任务域索引模型库。
- 最终部署checkpoint只加载student，teacher路径隐藏时仍复现最终预测；一次前向，不路由、不TTA。
- 算法在将来CL中如何使用t−1教师不在本轮解决，不允许额外复制前驱后形成三模型。
- raw图像、GT、模型权重、患者ID、逐病例像素和凭据保留NAS，不公开GitHub。
- public只含源码、协议、汇总、测试/资源/来源与退出证据；不复制全部旧档案增加无关审计开销。

## 11. 工程交付（合并同类，避免重复大文件）

至少交付：

```text
PROTOCOL.md + PROTOCOL.json
SOURCE_AND_INPUT_LINEAGE.json
TEST_REPORT.json
FINAL_METRICS.csv
FACTORIAL_CONTRASTS.csv
CORRECTION_PATH_DIAGNOSTICS.csv
CLASS_LOSS_AND_GRADIENT.csv
PROBABILITY_AND_PSEUDO_QUALITY.csv
TRAINING_AND_MEMORY_ACCOUNTING.json
DECISION.json
FINAL_REPORT.md
PUBLICATION_VERIFICATION.json
```

FINAL_REPORT必须首页回答：是否增益、是否稳定、哪些类别代价、旧监督通路与新通路差别是否支持假设。不得首页只报“测试全部通过”。

实际计数包括全部正式/资格/失败尝试的optimizer、backward、forward、EMA、诊断和I/O，报告current total和historical total。不承诺固定分钟数；使用已观察吞吐更新剩余区间，并标明共享GPU负载。

## 12. 本轮之后的明确分流

- T_SCE未建立目标收益：结束当前微型骨干、同视图轻扰动EMA体系内的这组监督路径检验。不追加阈值、权重、head、seed、PAS变体或损失投影。
- 只有机制指标改善：只作为诊断结论，不进入CL。
- T_SCE建立开发信号：冻结配方；后续另行设计独立患者/协议确认，以及最多一个历史教师的CL训练架构。不能本轮顺带宣称抗遗忘。
- 简单对照更好：不为了“新颖”强留SCE，保留完整矩阵；不得事后更名成为主方法成功。

不承诺这条路线成功；也不因一次固定实验失败宣称半监督学习普遍无效。

## 13. 来源

以下链接是本计划的可复核依据；数学梯度推导与前瞻阈值为本计划的设计，不是原文结论。

1. 最新完整报告：
   https://github.com/DLwbm123/SSL_CL_seg/blob/60843e85c110f31863299639822b080cf4c4ba9b/experiments/lcrseg/docs/ssl_head_control_v0_1/FINAL_REPORT.md
2. 等数量PAS审计：
   https://github.com/DLwbm123/SSL_CL_seg/blob/60843e85c110f31863299639822b080cf4c4ba9b/experiments/lcrseg/docs/ssl_head_control_v0_1/PAS_FIXED_MASS_AUDIT.md
3. 正确性分层：
   https://github.com/DLwbm123/SSL_CL_seg/blob/60843e85c110f31863299639822b080cf4c4ba9b/experiments/lcrseg/docs/ssl_head_control_v0_1/CORRECTNESS_STRATA.csv
4. 旧head-control核心实现：
   https://github.com/DLwbm123/SSL_CL_seg/blob/f478cc204a87a807858c76a95fc4246d301eeff2/experiments/lcrseg/ssl_head_control_v0_1/core.py
5. Pandey等，Continual Segmentation under Joint Nonstationarity，用户附件，arXiv:2605.20538v1，第5页公式(3)–(5)。
6. Sohn等，FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence，NeurIPS 2020，arXiv:2001.07685。这里只引用目标侧高置信监督不同视图的思想，不继承其分类实验成绩。
7. Tarvainen & Valpola，Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results，NeurIPS 2017。这里只引用EMA教师机制，不保证Fundus有效。

## 14. 发给Codex的启动指令

```text
完整阅读SSL_CL_Teacher_Target_Path_V0_1_Codex_Plan.md并执行。

base: 60843e85c110f31863299639822b080cf4c4ba9b
branch: codex/ssl-teacher-target-path-v0-1

本次新问题是监督准入和损失形式，不重判旧分类头/PAS结果。
执行SUP及joint/teacher-only × probability-MSE/soft-CE固定矩阵。
主候选预指定T_SCE；split_seed0、optimization_seeds31/32/33、两域独立。
全部30任务100epoch，共79500正式更新。三个种子均获准，
不得因seed31表现提前科学停止或更换候选。

只改变矩阵规定的两个因素。保持线性头、Gaussian0.02、
confidence>0.7、EMA0.99、Adam/步数/分母/权重日程不变。
不加PAS、GAS、强增强、Dice、前驱KD、投影或路由。

完成必要资格后实际训练。所有主指标固定为最终学生。
必做教师对学生错/反向错误/共同错误的掩码与局部梯度诊断，
GT仅在隔离val评价，不驱动训练或选择阈值。

训练最多student+EMA，部署一份student。旧终态/锁不变，
不读test/隐藏GT，不回放历史训练数据，不合并main，不自动CL。
全部已登记任务、分项结果、GitHub发布和NAS归档完成后停止。
```
