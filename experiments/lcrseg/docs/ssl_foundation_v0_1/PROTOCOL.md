# SSL_CL 下一步执行计划：单域半监督基础有效性验证 V0.1

> **用途：交给 Codex 实际执行的独立实验计划。**
>
> **优先级：有效性 → 创新性 → SOTA。**
>
> **本文件状态：PLANNED_NOT_EXECUTED。** 本文件不表示新实验、代码测试或服务器核验已经完成。
>
> 本轮结束 L05／L05_SSL／SCD 的继续开发，转而回答一个更基础的问题：**排除跨域顺序、历史蒸馏和冲突控制后，当前域的无标注数据能否通过 Mean Teacher／PAS 带来可复核的分割收益？**

---

## 0. 执行摘要与明确授权范围

建立独立分支，完成下列有限流程；前置资格通过后应实际训练，不只交付计划或合成测试：

1. 固定协议、核验输入、完成最小必要工程资格。
2. 在 **RIM_ONE_r3 与 Drishti_GS 两个独立单域任务**上，执行六个固定组件对照，优化种子为 11。
3. 必须完成分割评价及伪标签 precision／coverage 诊断，不能再次仅报告 coverage。
4. 用预先固定的规则选择至多一个 SSL 配方；若有配方准入，自动用优化种子 12、13 完成规定的确认矩阵，不再等待额外确认。
5. 分开报告工程完成、SSL 有效性、PAS 增量、GAS 配置差异，以及证据适用范围。完成全部已准入任务后收尾，不追加调参。

**本轮不运行持续学习序列。** 不训练新的前驱蒸馏、不研究冻结层、不运行新的数据集、不重开历史方法、不读取 test。冻结范围实验只写下一阶段草案，不能在本轮结果出来后临时启动。

| 项目 | 固定内容 |
|---|---|
| 仓库 | `DLwbm123/SSL_CL_seg` |
| 基线提交 | `6eb48fd845f6192698c4d39893b2524447afb3df` |
| 新分支 | `codex/single-domain-ssl-foundation-v0-1` |
| 建议代码命名空间 | `experiments/lcrseg/ssl_foundation_v0_1/` |
| 建议测试命名空间 | `experiments/lcrseg/tests/ssl_foundation_v0_1/` |
| 报告命名空间 | `experiments/lcrseg/docs/ssl_foundation_v0_1/` |
| 数据划分 | `data_split_seed=0`，原始患者角色不变 |
| 筛查优化种子 | `optimization_seed=11` |
| 条件确认优化种子 | `optimization_seed=12,13` |
| 正式更新预算 | 筛查 31,800 步；确认最多 42,400 步；合计最多 **74,200 步** |
| 模型上限 | 每个训练任务最多当前学生与其当前 EMA 两份完整模型；推理一份 |

种子编号只是预先固定的编号，不代表这些种子预期表现更好。不能遇到坏结果换种子。

---

## 1. 立项依据：将已有证据、论文内容与新实验选择分开

### 1.1 已有项目结果，不重新裁定

最新 L05_SSL 研究的结论如下，来源为文末 [R1]：

- L05 在两个新优化种子上的 F 增益为 `+0.007136805006` 和 `-0.041538650720`，固定配方复核未成立。
- L05_SSL 相对 L05 的 seed0 F／H 变化为 `-0.068748918335 / -0.088358784333`。
- 已有显式 CE 与 U-KD 呈现显著的配方交互；一个条件下的正效果不能直接外推到另一个条件。
- 前驱研究记录的正式尝试累计为 `99,708` 步。本轮更新另列；这个数字不是所有更早项目工作的无遗漏总算力。
- 伪标签 precision 仍未评估。因此不能声称已证明唯一根因是伪标签错误、GAS、冻结范围或某个类别。

保留全部旧状态、锁、报告、分支和 NAS 产物。不得把本轮成功用来覆盖旧失败，也不再重训 L05、L10、L05_SSL、SCD 或多专家路由。

### 1.2 附件 JASCL 实际支持的设计依据

附件：Pandey 等，*Continual Segmentation under Joint Nonstationarity*，arXiv:2605.20538v1，63 页。

本次附件 SHA-256：

```text
dc5cdfc70cdda70e6175ce055632a0cf00431bcdbf88fd2801f6f5cf37742b93
```

| 位置 | 附件支持的内容 | 本轮如何使用／不如何使用 |
|---|---|---|
| 第 3 页 §3.1、式 (1) | 分类器上的梯度自适应随机扰动 GAS | 做开／关组件对照，不预设开启必然更好 |
| 第 5 页 §3.2、式 (3)–(5) | Mean Teacher；逐样本类别原型；confidence 与 prototype similarity 联合过滤；两模型有效集合交集上的概率 MSE | 构成本轮 MT_CONF／MT_PAS 的明确参照 |
| 第 6 页及第 7 页图 3 | 完整方法还涉及原型回放 `L_proto` | 本轮无历史阶段、不做原型回放；不能称完整 JASCL 复现 |
| 第 8 页及第 33 页 | 文中采用 `tau_conf=tau_sim=0.7`，并报告敏感性分析 | 本轮固定 0.7，不据新 val 搜索阈值 |
| 第 35 页、第 45 页表 19 | 医学、自然场景等设置的训练轮数和冻结范围不同；多项医学增量设置是 “All but last” | 不把它解释成 Fundus 必须冻结某层；本轮从随机初始化训练单域，全部有效权重可训练 |
| 第 9 页、第 38–40 页 | 严重分布变化与无标注数据质量存在局限 | 不宣称局部公式或理论文字保证本项目成功 |

论文没有为当前 2D Fundus 数据给出这一份完整配方。下面的域选择、初始化、预算、输入噪声、原型刷新、优化器、筛查阈值与确认流程，均是**本计划的前瞻性实验选择**，不是论文原文。

Mean Teacher 的外部参照 [R3] 支持用 EMA 权重构建一致性目标；其分类基准结果不构成本项目的效果预期。

### 1.3 本轮不做的推断

- 不由上一轮失败推出 JASCL／Mean Teacher／所有蒸馏都无效。
- 不以附录 “All but last” 自动决定本项目的层名。
- 不声称 PAS 一定优于置信度过滤；这正是待检验的组件问题。
- 不为了获得 PASS 换骨干、提高标注比例、修改患者划分或替换评价指标。

---

## 2. 三个研究问题与实验边界

**Q1：SSL 基础有效性。** 在相同单域、相同标签预算、相同初始化与标注呈现次数下，EMA 概率一致性能否提高最终学生的宏平均前景 Dice？

**Q2：PAS 增量。** 在同一 GAS 设置下，原型验证相对 confidence-only 是否增加分割收益，或改善有实际覆盖的伪标签质量？两类证据分别报告，不互相替代。

**Q3：GAS 依赖。** 在同一分类器函数与训练配置下，GAS 开／关是否改变监督和 SSL 的效果？不得把 GAS_OFF 命名成标准线性卷积 U-Net。

本轮不回答遗忘、历史保持、类增量、联合非平稳或 SOTA。两个域分别从头训练，**RIM 的模型不初始化 Drishti，Drishti 不初始化 RIM**。

---

## 3. 数据与访问权限

### 3.1 固定人群

沿用原始 `data_split_seed=0` 的 manifest 和 split，按域独立读取：

| 单域任务 | train_labeled | train_unlabeled | val 预期 | epoch 步数 | 100 epoch 更新数 |
|---|---:|---:|---:|---:|---:|
| RIM_ONE_r3 | 16 | 63 | 40 | 32 | 3,200 |
| Drishti_GS | 10 | 41 | 25 | 21 | 2,100 |

以上数量是待前置核验的输入要求，不能跳过核验便写成“本轮已核实”。病例／患者身份、HDF5 文件哈希、图像尺寸及标签映射全部绑定既有文件。

两域被选中，是因为它们是此前少标注适应中的两个目标域；不是根据新实验结果选择较容易获胜的域。REFUGE 不参与本轮训练或初始化。

### 3.2 单域隔离

每个训练任务只可获得本域的 `train_labeled` 图像与 GT，以及本域的 `train_unlabeled` 图像。

- 原 `train_unlabeled` 隐藏 GT、GT 路径及标签哈希不得构造或读取。
- 原 val 不得并入 U；原 test 的图像和 GT 均不读。
- 不加载过去训练的公共 stage0、L05、SHOR、B0 或其他专家权重。
- 不使用其他域的训练数据、旧特征、原型或概率缓存。
- val GT 只进入只读 evaluator；监督优化、掩码和原型不能接收它。

资源路径继续使用既有 `di_dmpa_gate1.binding.safe_asset(DATA, relative)`，即 `DATA/h5/v1/relative`。不要全局改变 `DATA` 或 generic path resolver。

### 3.3 评价隔离不是独立数据确认

本轮 val 患者已在前面研究中产生过反馈。新初始化、新分支与新种子不能取消这种开发暴露。本轮最多建立固定患者划分上的开发／训练随机性证据，不得声称未见患者确认。

---

## 4. 六个固定实验臂

“G0／G1”只表示同一个官方分类头的 GAS 采样关闭／开启，不替换其归一化、温度或卷积几何。

| ID | 分类头采样 | 训练目标 | U 图像参与训练 | 原型是否参与损失掩码 |
|---|---|---|---|---|
| `SUP_G0` | 关闭 | supervised CE | 否 | 否 |
| `MT_CONF_G0` | 关闭 | CE + 双模型 confidence 有效集合上的概率 MSE | 是 | 否 |
| `MT_PAS_G0` | 关闭 | CE + 双模型 confidence/prototype 有效集合上的概率 MSE | 是 | 是 |
| `SUP_G1` | 开启 | supervised CE | 否 | 否 |
| `MT_CONF_G1` | 开启 | 与 MT_CONF_G0 相同 | 是 | 否 |
| `MT_PAS_G1` | 开启 | 与 MT_PAS_G0 相同 | 是 | 是 |

**学生最终模型是统一主输出。** 六个臂都维护一份 EMA，用于统一的 EMA-only 评价对照；SUP 的 EMA 不向学生提供损失，也不读取 U。这样可区别“训练真正利用 U 的改善”和“只在评价时做权重平均的改善”。

不另加未过滤 MT、硬伪标签 CE、强增强 CE、BCP、CPS、UniMatch 完整分支、Dice loss、历史 KD、投影、风险头或参数冻结。确认阶段也不添加新方法。

---

## 5. 骨干、初始化、训练预算和随机性

### 5.1 固定结构

复用项目可核验的 UNet2D 主干（通道 16/32/64/128、GroupNorm）和 pinned JASCL 分类头。官方参照提交：

```text
prinshul/JASCL
3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53
Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT/models/deeplabv3/deeplab.py
```

官方头还包含输入／权重归一化、`temp=10`、无 padding 的 functional convolution；项目 wrapper 将 logits 插值回原图尺寸。[R4][R5]

这些行为在六臂之间保持一致，不顺手修改。GAS_OFF 只是 `stochastic_classifier=False`，不是“普通线性头”。若这套头在单域也不成立，应报告边界，而不是本轮换头救结果。

`grad_update` 不进入 optimizer。官方存在未实际用于 forward 的 `sigma` 权重时，记录其存在、梯度和字节，不把它算作有效不确定性估计。不得凭 `model.training` 隐式决定采样。

### 5.2 初始化

全部单域任务随机初始化，不使用训练权重。每个 `(domain, optimization_seed)` 的六个臂，其起始学生权重必须完全一致；EMA 初始化等于学生。

官方分类器构造中存在固定随机种子设置，需用作用域隔离恢复外部 RNG，并记录其实际行为。不能声称所有分类头参数都随 optimization_seed 改变；必须证明 backbone 初始化和训练随机流实际改变。禁止为了“让种子看起来不同”静默重写官方初始化。

### 5.3 固定训练配方（本计划选择）

| 项目 | 值 |
|---|---|
| 训练周期 | 每单域任务 100 epochs |
| label / U batch size | 各 2，不 drop 最后一个小 batch |
| 每 epoch 更新数 | `max(ceil(N_l/2), ceil(N_u/2))` |
| 标注采样 | 周期性独立打乱并循环补足更新数 |
| SUP 的 U 访问 | 只读计数以匹配更新数；不得打开 U 图像 |
| optimizer | Adam，lr=0.001，betas=(0.9,0.999)，eps=1e-8，weight_decay=4e-5 |
| LR | 从头初始化；`lr=0.001*(1-k/K)^0.9`，k 为零基 update |
| 可训练范围 | 全部实际可学习的学生网络权重；不冻结随机特征 |
| supervised loss | 有效像素 CE；不另加类别权重或 Dice loss |
| EMA decay | 固定 0.99，每次成功 optimizer 更新后更新 |
| consistency 权重 | 前 20 epochs 为 0；21–40 线性升到 0.5；之后 0.5 |
| 权重精确定义 | `lambda_cons(e)=0.5*min(1,max(0,(e-20)/20))`，e 从 1 开始 |
| 最终评价 | epoch100 最后一步学生；不选择 best epoch |
| 精度 | 模型前向 FP32；不用 AMP/TF32；沿用已验证确定性后端配置 |

前 20 epochs 不打开 U 图像、不计算 U 损失；EMA 仍正常更新。相同 GAS 设置下三个臂的 warm-up 应当一致。每个任务按完整 100 epochs 计数；本轮不另外实现共享 warm-up 分叉优化，以免增加来源／恢复复杂度。

### 5.4 匹配随机流

采样、几何变换、U 输入噪声、学生分类头噪声、教师分类头噪声分别使用独立 stateless key：

```text
(optimization_seed, domain, epoch, step, stream, sample_position)
```

同一组件在所有实验臂的 key 不含 arm 名；GAS_OFF 不消费 GAS_ON 的其他流。诊断使用独立命名的流，不能改变训练 RNG。

六臂匹配优化步数与真实标注呈现序列，不宣称 FLOPs 或墙钟时间相同。真实前向、激活峰值、EMA 字节和耗时全部另报。

---

## 6. Mean Teacher 与 PAS 的精确定义

### 6.1 视图与几何对齐

标注数据沿用配对的水平／垂直翻转（各 0.5）与均匀 `rot90`，图像、标签及 geometry mask 同步；不附加本轮未定义的图像操作。

U 图像先进行同一套几何变换，得到 `x_base`：

```text
teacher_input = x_base
student_input = clip(x_base + 0.02 * standard_normal, 0, 1)
```

噪声独立于模型噪声，形状为图像各通道各像素；两个输入的像素坐标不变。MT_CONF 与 MT_PAS 完全同一视图定义。这里的轻量输入扰动是本计划适配选择，不声称是 JASCL 原作者在 Fundus 上的设置。

不使用上一轮复杂 brightness/contrast/blur 链，也不把它称为强增强伪标签 CE。

### 6.2 概率与梯度

得到学生 logits `z_s`、教师 logits `z_t`：

```text
p_s = softmax(z_s, class_dim)
p_t = stopgrad(softmax(z_t, class_dim))
```

只调用一次 softmax。掩码使用 detached 概率与特征；损失里的 `p_s` 不能 detach。教师参数不接收梯度，不进入 optimizer。

G1 的学生与教师 U 前向都使用官方采样头，随机流独立；G0 两者都使用确定性头。主分割评价始终使用确定性头。

### 6.3 当前 EMA，不是历史教师

成功的学生更新后：

\[
\theta_T\leftarrow 0.99\theta_T+0.01\theta_S.
\]

浮点网络权重做 EMA；离散计数缓冲区如存在则复制。GAS 的 `grad_update` 作为梯度敏感度状态，在学生更新该状态之后复制到 EMA，而不是当作参数重要性历史库。此项是本计划显式 buffer 规则，须列入与原项目 helper 的差异，不能静默混用。

G1 的学生 `grad_update` 只使用本步 supervised CE 对分类器均值权重的平方梯度；不能从加入 consistency 的总梯度更新。G0 不注入噪声，GAS buffer 保持零。所有臂使用相同生效时序：本步 forward 使用旧 buffer，成功 update 后写入新 buffer。

SUP 中保留 EMA 是为了可比的 EMA 评价，它不得形成额外监督。

### 6.4 原型定义

只使用本单域当前 `train_labeled`。采用固定的最终 16 维 decoder 特征，逐图逐类先求平均，再归一化；有该类的图像等权平均：

\[
u_{j,c}=\frac{1}{|\Omega_{j,c}|}\sum_{i\in\Omega_{j,c}}F_j(i),\qquad
P_c=\frac{1}{n_c}\sum_{j:c\text{存在}}\frac{u_{j,c}}{\|u_{j,c}\|_2}.
\]

相似度使用 cosine，显式除以 `||P_c||`。不将其误实现为按像素总数加权的全局中心，不混用 teacher 与 student 的两个库。**本轮共享学生特征生成的单一原型集合供两模型验证，这是明确的坐标选择，不宣称论文排除了其他实现。**

MT_PAS 在 epochs `21,26,31,...,96` 开始前刷新原型：学生 eval、no_grad、确定性分类头、无随机增强，只读取本域标注训练数据。

缺失类或零范数类标记 unsupported；不能伪造原型。该类 PAS 位置不接受；保留科学覆盖结果，不自动降阈值或补历史原型。

### 6.5 掩码必须按各自预测类验证

每个像素：

\[
\hat c_s=\arg\max p_s,\quad \hat c_t=\arg\max p_t.
\]

Confidence-only：

\[
m_{conf}=geometry\land(\max p_s>0.7)\land(\max p_t>0.7).
\]

PAS：

\[
m_{pas}=m_{conf}\land support_{\hat c_s}\land support_{\hat c_t}
\land(cos(F_s,P_{\hat c_s})>0.7)
\land(cos(F_t,P_{\hat c_t})>0.7).
\]

**不要额外要求学生、教师 argmax 相同。** 附件式 (5) 要求有效集合交集，并未在该式加预测类别相等；两者是否相等作为诊断项报告，而非新增训练门槛。

### 6.6 损失与空集合

\[
L_{cons}=\frac{\sum_i m_i\sum_c(p_{s,ic}-p_{t,ic})^2}{\max(1,\sum_i m_i)},
\qquad L=L_{sup}+\lambda_{cons}(e)L_{cons}.
\]

按照附件式 (5) 在有效像素上平均，不额外除以 C，不改成上一轮按总图像面积平均，不转成 hard-label CE。

空集合返回与学生计算图相连的零值；无 NaN、无教师梯度。空集合／某前景类无覆盖是需要报告的科学现象，不允许边看结果边修门槛。

---

## 7. 伪标签质量评价是强制交付项

### 7.1 固定时间与不干预训练

在 epochs `20,40,60,80,100` 做本域 val 的只读诊断。评估过程不得改变权重、EMA、GAS、optimizer、训练原型或训练随机流，不产生任何优化步骤，不依据结果调整训练。

无需保留五份完整模型。优先在预定边界只读复用当前两份模型，或使用滚动 checkpoint；评价服务只输出结果文件，不把 val GT 或评价值作为优化器、scheduler、checkpoint selector 的输入。若通过缓存隔离评分，只允许 evaluator-only 临时预测缓存，不形成训练可访问的历史库。

每个快照用本域 train_labeled、该快照学生的确定性特征**重算诊断专用原型**，绝不写回训练原型。这样 SUP、MT_CONF、MT_PAS 都能比较同一定义的候选 mask。MT_PAS 另报实际训练中使用的最近刷新原型的诊断，区分“新鲜诊断库”和“实际训练库”。epoch20 尚无训练库时记 `NOT_STARTED`，不能填成零 precision。

### 7.2 每个病例需要的统计

诊断区分两种模式：

- `posterior_mean_clean`：两模型均确定性、无 U 输入噪声，便于理解模型本身质量。
- `training_like`：与该臂 U 训练的输入噪声及分类头采样一致，固定四次独立诊断流，报告各次值和病例内均值。

四次诊断不是四个独立患者。默认 precision 评价的伪标签是 teacher argmax；student 另列，不能混合分子。

对背景、rim、cup 分别记录：

1. 预测该类像素数、GT 该类像素数；
2. 接受该类像素数；
3. 接受且正确像素数；
4. precision = 接受且正确 / 接受；
5. prediction-conditioned coverage = 接受 / 预测该类；
6. image-support coverage = 接受 / 有效像素；
7. accepted correct recall = 接受且正确 / GT该类；
8. student/teacher disagreement；
9. joint valid 覆盖及空集合病例数。

分母为零则数值留空并给 `UNDEFINED_NO_SUPPORT`，不能记成 precision=1。GT=255 从评价分子分母排除，但不得使用 GT validity 来修改训练 U mask。

必须同时报告 raw、confidence、PAS 三种过滤统计；SUP 的这些是 post-hoc 诊断，不是其训练使用过的机制。

汇总时同时给出患者等权统计、pooled counts 及其实际分母；训练种子不是患者，像素也不是独立样本。precision 与分割收益分开评价。严禁再以高 coverage 代替高 precision。

### 7.3 必需输出与失败处理

`PSEUDO_QUALITY_BY_CLASS.csv`、`PSEUDO_QUALITY_AGGREGATE.csv`、`PAS_COVERAGE_PRECISION.md` 是本轮必需文件。它们不再是可省略的 optional diagnostic。

若诊断无法执行，记录 `INCOMPLETE_DIAGNOSTICS`。即使训练完成，也不能写“半监督基础验证完整完成”。不要通过读取 train-U 隐藏标签来补齐。

---

## 8. 主分割指标、EMA 对照与固定评价器

每个单域、每个种子分别报告最终学生：

\[
D_{fg}=\tfrac12(D_{rim}+D_{cup}).
\]

使用既有已验证的逐病例评分语义及全精度 CSV；各病例等权。两个单域任务的总体分数为域等权平均：

\[
Q_r=\tfrac12(D_{r,\mathrm{RIM}}+D_{r,\mathrm{Drishti}}).
\]

这里 `Q` 不是持续学习 F。**不生成 BWT、遗忘、历史 H 或三阶段 N，不把不存在的序列指标填为 0。**

GT=255、空类、全忽略图像沿用旧评分 contract：合法空支持的兼容 Dice=1；全忽略病例标记 `has_evaluable_gt=false`，另报 evaluable-only sensitivity。缺失／损坏文件不是合法全忽略图像，不得填充值。

最终 EMA 网络在全部六臂另报同样的 deterministic 指标，写入 `EMA_VS_STUDENT.csv`。主判定始终使用学生；不能看到 EMA 更高后切换部署对象。若收益只发生在 EMA，结论为 `EMA_ONLY_SIGNAL`（描述性），而非主学生成功。

本轮不按 val 选 epoch100 以外的模型。中间曲线仅用于诊断，不能事后报告 best-of-epoch 冒充主指标。

---

## 9. 执行顺序与筛查规则

### P0：资格与预冻结

固定源代码、配置、患者 allowlist、种子流、预算、评价器和本文件的哈希。运行第 12 节资格后实际开始训练，不增加另一轮必须由用户重复确认的纯文档阶段。

### P1：六配方 × 两域 × optimization_seed11

全部 12 个单域训练完成及部署密封后，再计算筛查。不得看到一个配方早期不好就少训练、换配置或提前开始另一配方确认。

每个 SSL 候选 `r` 的对照是同 GAS 设置的 SUP，令：

\[
\Delta_d(r)=D_{r,d}-D_{SUP_g,d},\qquad
\overline\Delta(r)=\tfrac12\sum_d\Delta_d(r).
\]

四个候选为 `MT_CONF_G0/G1`、`MT_PAS_G0/G1`。所有候选使用同一套前瞻性筛查条件：

| 条件 | 固定门槛 |
|---|---|
| 同 GAS 监督对照上的平均实际收益 | `mean_domain_delta >= 0.010` |
| 不让一个域严重退步换取均值 | 两域各自 `delta >= -0.005` |
| 前景类别保护 | 两域各rim/cup相对同 GAS SUP 的差值均 `>= -0.020` |
| 不只战胜一个很弱的 SUP | `Q_r >= max(Q_SUP_G0, Q_SUP_G1) + 0.005` |
| 工程与质量诊断 | 本臂两域完整、可部署、必需质量诊断完成 |

0.01／0.005／0.02 是资源准入门槛，不是理论常数或临床安全界。使用未四舍五入数值，不能事后更改。

若没有候选通过：完成所有 P1 结果与失败归因，`SSL_FOUNDATION_NOT_ESTABLISHED`，不进入 P2。保留各组件的真实方向，不宣称所有 SSL 无效。

若多个候选通过，至多选择一个。固定排序：

1. `min_domain_delta` 最大；
2. `Q_r` 最大；
3. 精确相同时优先 `MT_CONF`（较简单）；
4. 再相同时优先 `G0`；
5. 最后按固定 ID 字典序。

不使用伪标签 precision 来临时重排获胜者。PAS 即使 precision 高，也不能代替分割收益。

### P2：自动完成新优化种子12/13，不因第一枚新种子结果改变任务

选择候选的 GAS 设置为 `g*`，其 SSL 类型为 `k*`。在两个新种子、两个域上执行固定四臂：

```text
SUP_G0
SUP_G1
MT_CONF_G*
MT_PAS_G*
```

合计 16 个单域训练。执行两个 SUP 是为了确认结果不是仅相对弱参照成立；执行两种 MT 是为了在新种子上独立区分 SSL 基础收益和 PAS 增量。**P1 选中的唯一主候选保持不变，不能在 P2 看到另一臂更高后更换主方法。**

即使第一个新种子不满足科学门槛，也完成第二个既定种子和所有已准入对照。合法工程失败保留 `INCOMPLETE`，不得替换种子。

---

## 10. P2 的独立判定：只使用新优化种子

旧筛查种子11只列描述性表格，不能抵消种子12/13的确认失败。

### 10.1 SSL 基础是否建立

主候选须同时满足：

- 每个新种子相对同 GAS SUP 的两域平均增益均 `>0`；
- 两新种子、两域等权的平均增益 `>=0.010`；
- 四个 seed-domain 单元的 macro 差值均 `>=-0.010`；
- 八个 seed-domain-class 差值均 `>=-0.020`；
- 相对 P1 已固定的较强 SUP 设置（按 P1 两域 Q 选定，平局 G0），两个新种子的总体平均增益 `>=0.005`；
- 最终学生单模型部署与所有必需诊断完整。

满足则：`SINGLE_DOMAIN_SSL_FOUNDATION_ESTABLISHED_ON_FIXED_SPLIT`。

不满足则：`SINGLE_DOMAIN_SSL_REPLICATION_NOT_ESTABLISHED`。

无论哪一种，都不是独立患者确认、持续学习成功或 SOTA。

### 10.2 PAS 是否有新增价值

在 `g*` 下，用同一新种子、同一域比较 `MT_PAS-MT_CONF`。若两新种子平均增益均为正、总体平均增益 `>=0.005`，且每个域类差值 `>=-0.020`，才标记 `PAS_INCREMENTAL_DICE_SIGNAL`；否则 `PAS_INCREMENTAL_DICE_NOT_ESTABLISHED`。

另外单独说明 PAS 的 precision／coverage／recall 是否改善。不要以删除大量像素换得precision提高就声称训练效果更好，也不要以预测质量改善代替Dice增量。

若 confidence-only 已建立 SSL，而 PAS 没有增量，后续保留更简单的 confidence-only；不为了维持“JASCL-inspired 创新”坚持 PAS。

若 P1 主候选确认失败而另一个固定对照较好，该对照只列 `EXPLORATORY_ALTERNATIVE`，不临时升格为预注册成功；本轮不再启动第三轮验证。

### 10.3 GAS 与理论边界

报告 G0/G1 同臂差值及 interaction，但不以一个分数否定或证明论文全部 GAS 理论。GAS_OFF 的 head normalization、temp 和几何仍未变化；本轮没有与真正线性分类头作比较。

---

## 11. 预算与调度

| 阶段 | 计算 | 正式更新 |
|---|---|---:|
| P1 RIM | 6 × 3,200 | 19,200 |
| P1 Drishti | 6 × 2,100 | 12,600 |
| P1总计 | 12个训练任务 | **31,800** |
| 条件P2 RIM | 2种子 × 4臂 × 3,200 | 25,600 |
| 条件P2 Drishti | 2种子 × 4臂 × 2,100 | 16,800 |
| P2总计 | 16个训练任务 | **42,400** |
| 新正式更新上限 | P1+P2 | **74,200** |

按前驱报告的累计口径，只有 P1 完成时为 `99,708+31,800=131,508`；P1/P2全部完成时为 `173,908`。失败尝试的成功更新必须另加，不能只记有效最终轨迹。

资格／诊断不挤入正式预算，实际前向、backward与更新单列。不要虚报“零额外成本”。SUP 的 EMA、原型诊断、训练中质量评价和不同 GAS 工作量均计入资源。

使用现有明确授权 GPU；默认仅4/5，不因繁忙扩大许可或终止他人进程。每GPU至多一条本研究训练任务。可并行同阶段不同域／臂，但每条任务均独立日志、输出目录与模型状态。

不预先承诺固定分钟数。用实测吞吐更新 ETA，并区分训练时间之和、并行墙钟时间、排队与评价耗时。

---

## 12. 必要工程资格（做完就训练，不追求测试数字）

复用有效 data、metrics、seed、deployment 纯组件，不重写整套平台，也不修改旧 freeze/authorization 文件以通过旧门槛。本轮单独注册权限。

必须覆盖：

1. **部署路径：**正确 `h5/v1` 布局成功，错误root失败，双位置decoy不能被错误读取；沿用已验证loader。
2. **角色隔离：**训练接口无法获得val/test/其他域/隐藏U-GT；监督臂U打开次数为0。
3. **单域身份：**无跨域parent和训练权重加载，全部任务来自随机初始化；两个域不是顺序阶段。
4. **配对随机性：**六臂初值相同；同GAS组前20epoch等价；新种子实际改变有效随机流；诊断前后训练RNG不变。
5. **概率损失：**单次softmax、不除以C、有效像素分母、空mask图连接零、学生梯度存在、教师及掩码无梯度。
6. **PAS：**逐图归一化后等权原型、严格大于阈值、各自argmax选原型、联合mask且不强加类别相等、缺类与零范数正确处理。
7. **GAS与EMA：**监督梯度来源、G0不采样、teacher不进optimizer、参数EMA与GASbuffer复制规则，teacher更新发生在成功step之后。
8. **最小集成：**独立生成HDF5，实际模型执行至少warm-up边界、原型刷新、EMA、保存／恢复、评价和单模型部署；不用mock替代真实loader或loss。
9. **真数据梯度smoke：**每域固定前两条当前标注和U图像，最多8次诊断更新／域；不挑高覆盖样本、不读隐藏GT、不作正式初始化。报告真实loss梯度，不以参数变化替代分支梯度。
10. **诊断质量：**手工GT/预测见证验证precision、coverage、recall分母以及零支持；四次采样不按四个患者统计。
11. **状态保持：**只读诊断前后学生/教师/GAS/buffer hash、模式、随机状态一致；诊断原型不能写回训练。
12. **来源与历史保护：**新source SHA、config SHA、data SHA与实际执行一致，旧文件不修改，主分支不合并。

真实资格只判断实现与数据边界是否正确。不得要求微小真实smoke先证明未来Dice改善，亦不得因为初期PAS覆盖较低而修改配方。发现合法空集合应记录后继续；NaN、非法标签、梯度实现错误则属于工程阻塞。

未完成测试项目逐项说明，不能拿“0 skipped”掩盖没有实现的检查。科学结果不好不触发代码补救。

---

## 13. 收尾产物与报告结构

至少交付以下内容；可复用既有schema，避免为文档另建复杂框架：

```text
PROTOCOL.md / PROTOCOL.json
SOURCE_AND_INPUT_LINEAGE.json
PAPER_TO_IMPLEMENTATION.md
TEST_REPORT.json
RUN_LEDGER.csv
SINGLE_DOMAIN_METRICS.csv
PER_CLASS_METRICS.csv
EPOCH_CURVES.csv
EMA_VS_STUDENT.csv
PSEUDO_QUALITY_BY_CLASS.csv
PSEUDO_QUALITY_AGGREGATE.csv
PAS_COVERAGE_PRECISION.md
SCREEN_DECISION.json
CONFIRMATION_BY_SEED.csv
COMPONENT_CONTRASTS.csv
MEMORY_RUNTIME_AND_ACCESS.json
DEPLOYMENT_REPORT.json
FINAL_REPORT.md
STATUS.json
NEXT_STAGE_DRAFT.md
PUBLICATION_VERIFICATION.json
```

无P2准入时，确认文件保留schema和 `NOT_ADMITTED`，不造零值或虚构种子结果。

最终报告按五段写：

1. 工程是否完整、实际任务和更新数；
2. 最终学生相对两个SUP的真实效果；
3. teacher/EMA只是平均效应，还是学生也受益；
4. PAS质量和分割贡献，GAS差异；
5. 本轮到底支持什么、不支持什么、后续是否有资格研究CL。

公开仅发布代码、固定协议和聚合结果。模型、图像、GT、患者ID、全量private预测、凭证留NAS。公布准确 source SHA、report SHA 与发布验证；匿名HTTP成功不等于独立科学审查。

---

## 14. 合理停止与下一阶段边界

### 14.1 本轮通过

固定已确认SSL配方，不再在当前val调参。下一阶段可以准备两件事：

- 该配方在新的患者证据／数据设置下的确认；
- 从已训练而非随机的基础模型出发，比较明确模块边界的全量更新与受限更新。

冻结实验必须另有规范：哪个模块、哪些parameter/buffer、教师是什么角色、如何避免全冻结造成伪稳定，均需提前固定。“All but last”不能自动翻译成某个字符串路径。

本轮EMA是**当前平均教师**。后续若另需冻结 `t-1`，不得直接把它加成第三模型。要在下一协议中明确选择教师角色或分时执行方式，再训练。当前成功不等于两模型持续学习问题已经解决。

### 14.2 本轮未通过

不恢复L05/SCD，不自动改loss、阈值、温度、backbone、学习率、种子或标签比例。基于完成的质量与训练结果报告有限归因，例如：

- mask不足还是接受错误较多（有实测才写）；
- EMA平均有无收益；
- 分类头GAS设置的差别；
- 单域从头训练本身的能力上限。

只关闭这套固定单域适配配方的准入，不宣称一般SSL不可能有效。接下来若需新骨干／真正线性头／更忠实官方协议，必须是新的有依据问题，而非本轮自动救分。

---

## 15. 来源与引用（供Codex审阅追溯）

**[R1] 项目最新完整报告，固定提交。**

https://github.com/DLwbm123/SSL_CL_seg/blob/1b18257be20fc0fa9b560c4b6c93865221b799ae/experiments/lcrseg/docs/l05_ssl_replication/FINAL_REPORT.md

**[R2] 附件原文。** Pandey 等，*Continual Segmentation under Joint Nonstationarity*，2026，arXiv:2605.20538v1。以本次63页附件及上列SHA为详细规范参照；重点第5页式(3)–(5)、第6–7页、第35页及第45页表19。网页入口：

https://arxiv.org/abs/2605.20538

**[R3] Mean Teacher 原始研究。** Tarvainen and Valpola，*Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results*，NeurIPS 2017。用作权重EMA一致性框架来源，不迁移其分类数值结论：

https://proceedings.neurips.cc/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html

**[R4] pinned 官方随机分类器。**

https://github.com/prinshul/JASCL/blob/3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53/Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT/models/deeplabv3/deeplab.py

**[R5] 项目骨干适配与数据接口（只读参考，不改旧锁）。**

https://github.com/DLwbm123/SSL_CL_seg/blob/6eb48fd845f6192698c4d39893b2524447afb3df/experiments/lcrseg/di_dmpa_jascl/modeling.py

https://github.com/DLwbm123/SSL_CL_seg/blob/6eb48fd845f6192698c4d39893b2524447afb3df/experiments/lcrseg/single_teacher_scd_v0_1/data.py

---

## 16. 可直接发送给 Codex 的启动指令

```text
请完整阅读并执行附件：
SSL_CL_Single_Domain_SSL_Foundation_V0_1_Codex_Plan.md

从 6eb48fd845f6192698c4d39893b2524447afb3df 建立：
codex/single-domain-ssl-foundation-v0-1

关闭旧L05/L05_SSL/SCD的继续开发。
本轮不运行持续学习序列，也不加载旧训练权重。

在固定data_split_seed0的RIM_ONE_r3和Drishti_GS上，
分别随机初始化，执行SUP、MT_CONF、MT_PAS × GAS_OFF/ON。
optimization_seed11完成六配方两域，共31800正式更新。
每个单域只使用自己的训练标注与无标注数据。

必须完成学生最终Dice和按类伪标签precision/coverage/recall，
诊断不能再省略或用coverage冒充precision。
主输出固定为最终学生，EMA另报，不能择优切换。

若按附件规则有候选准入，自动完成optimization_seed12/13：
两域上的SUP_G0、SUP_G1及所选GAS设置下的MT_CONF/MT_PAS，
新增最多42400步，无需再次确认；总更新不超过74200。

最多当前学生+当前EMA两份完整模型，不额外保存前驱教师。
不加KD、投影、风险头、伪标签CE或新特征；不调阈值。
不读test/隐藏GT，不合并main，不修改历史终态或旧锁。

资格通过后实际训练，不仅返回计划。
完成全部已准入任务，分开给出工程完成、SSL基础、PAS增量
和GAS诊断结果，公开报告、核验来源并归档NAS。
不自动启动冻结层实验或新的持续学习方法。
```

**最终原则：这轮争取的是一个经配对对照和新训练种子支持的半监督学习基础，不是为旧方法换名字，也不是再做一个复杂巩固模块。**
