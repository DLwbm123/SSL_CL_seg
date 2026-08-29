# LCR-Seg 方法验收测试 V0.1

**文件名：** `METHOD_ACCEPTANCE_TESTS_V0_1.md`  
**依赖：** `METHOD_SPEC_V0_1.md`、`IMPLEMENTATION_CONTRACT_V0_1.md`  
**版本：** 0.1  
**状态：** V0–V3 合入和正式实验前的验收标准  
**最后更新：** 2026-08-18

---

## 0. 测试分级

### 0.1 Hard Engineering Gate

必须全部通过。任一失败都表示实现不可信，禁止启动正式实验。

### 0.2 Method Sanity Gate

用于确认方法量的行为符合定义。失败通常意味着公式、阈值或实现存在问题，应先修复或解释。

### 0.3 Research Hypothesis Gate

用于判断方法假设是否得到数据支持。失败不一定是代码错误，但必须触发方法复审，不能通过继续堆模块掩盖。

---

## 1. 测试环境与可复现性

每次验收记录：

```text
git commit
Python/PyTorch/CUDA/cuDNN
GPU
resolved config
dataset manifest hash
split hash
preprocess version
HDF5 inventory hash
seed
```

最小种子：

```text
unit tests: fixed seed 123
golden batch: fixed seed 20260818
smoke/overfit: seed 0
research gate: seeds 0,1,2
```

---

# Part A. Hard Engineering Gates

## 2. 数据与 loader 测试

### A-01 无患者交叉

文件建议：

```text
test_no_patient_overlap.py
```

对每个数据集、每个 seed、每个 site：

\[
\text{train}\cap\text{val}
=
\text{train}\cap\text{test}
=
\text{val}\cap\text{test}
=
\varnothing.
\]

M&Ms 额外要求同一患者 ED/ES 不跨集合。

**通过标准：** 零冲突。

---

### A-02 hidden label 不泄漏

文件建议：

```text
test_no_hidden_label_leakage.py
```

检查：

- train-unlabeled manifest 的 label path 为空；
- `UnlabeledBatch` 无 `label`；
- training package 不 import diagnostics manifest；
-训练 config 不包含 hidden label path；
- DataLoader 返回值不可通过 nested field 访问 GT。

**通过标准：** 任一泄漏路径均抛出明确异常。

---

### A-03 HDF5 多 worker 读取

```text
test_h5_worker_reading.py
```

分别运行：

```text
num_workers=0
num_workers=2
num_workers=4
```

连续读取至少 100 batch。

**通过标准：**

- 无 deadlock；
- 无 invalid file handle；
- shape/dtype 一致；
- case ID 不重复异常；
-内存不持续增长。

---

### A-04 weak/strong 几何一致

```text
test_weak_strong_geometry_alignment.py
```

使用含单点/方块标记的 synthetic image，执行 transform 后验证 weak 与 strong 的几何位置一致。

**通过标准：**

- 几何坐标完全一致；
- strong 只在 appearance/cutout 上不同；
- cutout valid mask 精确覆盖遮挡区域。

---

## 3. 模型与张量合同

### A-05 模型输出尺寸

```text
test_model_output_contract.py
```

输入：

- prostate `[2,1,256,256]`；
- M&Ms `[2,1,384,384]`；
- fundus `[2,3,384,384]`。

检查：

```text
logits = [B,C,H,W]
relation_features = [B,128,H/4,W/4]
```

**通过标准：** 完全符合合同，feature channel L2 norm 接近 1。

容差：

\[
|\|z_i\|_2-1|<10^{-4}
\]

对非零 feature。

---

### A-06 relation probability

```text
test_relation_probability_normalization.py
```

检查：

\[
\sum_cq_{i,c}=1.
\]

**通过标准：**

- max absolute error < `1e-5`；
- 无 NaN/Inf；
- temperature 非法时抛异常；
- invalid anchor class 不获得非零概率；
-所有 anchor invalid 时显式失败。

---

### A-07 feature/anchor scale invariance

```text
test_relation_scale_invariance.py
```

对 feature 和 anchor 分别乘正标量后重新 normalize。

**通过标准：** relation probability 最大差异 < `1e-5`。

---

## 4. AnchorBank 测试

### A-08 anchor 非参数

```text
test_anchor_not_parameter.py
```

**通过标准：**

- anchor 不出现在 `named_parameters()`；
-出现在 method state/checkpoint；
- optimizer 不含 anchor；
- backward 后 anchor.grad 不存在。

---

### A-09 current/old storage 独立

```text
test_anchor_storage_independence.py
```

修改 current anchor 后 old anchor 不变。

**通过标准：**

```python
current.data_ptr() != old.data_ptr()
```

并且数值独立。

---

### A-10 空类别不更新

```text
test_anchor_empty_class_no_update.py
```

构造 batch 中缺少类别 \(c\)。

**通过标准：**

- anchor[c] 原样保持；
- count 不变；
-无零向量覆盖；
-日志记录 skipped class。

---

### A-11 anchor norm 与 EMA

```text
test_anchor_update_math.py
```

使用可手算 feature/weight。

**通过标准：**

- weighted mean 正确；
- EMA 正确；
-更新后 norm=1±`1e-5`；
- support 小于阈值时不更新；
- labeled/unlabeled counts 正确。

---

### A-12 背景边界排除

```text
test_background_boundary_exclusion.py
```

synthetic 前景 mask 周围指定宽度区域不得用于 background anchor。

**通过标准：** 采样索引与预期完全一致。

---

## 5. 伪标签与 Learnability

### A-13 classifier-easy 分支

构造：

- segmentation confidence 高；
- segmentation class 与 relation class 一致。

**通过标准：**

```text
source = classifier
valid = true
label = segmentation top1
```

---

### A-14 anchor-recoverable 分支

构造：

- segmentation confidence 低；
- relation confidence/margin 高；
- spatial agreement 高。

**通过标准：**

```text
source = anchor
valid = true
label = relation top1
```

---

### A-15 deferred 分支

构造两个分支均不满足。

**通过标准：**

```text
source = deferred
valid = false
label = ignore_index
L_i = 0
```

不得改为 background。

---

### A-16 progressive participation 单调行为

```text
test_progressive_participation.py
```

固定 percentile rank：

- 早期 \(\rho=0.05\)；
- 中期 \(\rho=0.5\)；
- 后期 \(\rho=0.95\)。

检查低 rank 像素权重随进度增加，高 rank 像素始终不低。

**通过标准：**

- early：high-rank weight > low-rank weight；
- late：low-rank weight 明显增加；
-所有权重在 `[0,1]`；
-无端点 singularity。

---

### A-17 小类 rank fallback

当某预测类像素少于 `min_rank_pixels`。

**通过标准：**

- 使用 global rank；
-结果确定性；
-无空 tensor/NaN。

---

### A-18 \(L_i\) detach

```text
test_learnability_detached.py
```

**通过标准：**

```python
assert learnability.requires_grad is False
```

反向后 learnability 计算路径无梯度累积。

---

## 6. Compatibility

### A-19 relation 相同

令：

\[
q^{cur}=q^{old}.
\]

**通过标准：**

- JS≈0；
-主类一致；
- compatibility 接近 old-margin×spatial 的上限。

---

### A-20 relation 冲突

令 current/old 主类不同。

**通过标准：**

\[
C_i=0.
\]

---

### A-21 JS 单调性

构造逐渐偏离的 current distribution。

**通过标准：**

\[
d_{JS}\uparrow\Rightarrow C_i\downarrow.
\]

数值严格单调或在浮点容差内非增。

---

### A-22 old margin

保持 current/old 主类相同，降低 old top1-top2 margin。

**通过标准：** \(C_i\) 单调下降。

---

### A-23 第一站点

**通过标准：**

- old model=None；
- \(C_i=0\)；
- relation loss differentiable zero；
-无 old anchor 访问；
-训练可完成 backward。

---

### A-24 \(C_i\) detach

```python
assert compatibility.requires_grad is False
```

---

## 7. Loss 与梯度

### A-25 assimilation 空集合

所有 pseudo-label deferred。

**通过标准：**

- `loss_assim == 0`；
-有限值；
-可 backward；
-不会除零。

---

### A-26 relation 空集合

所有 \(C_i=0\)。

**通过标准：**

- `loss_relation == 0`；
-有限值；
-可 backward。

---

### A-27 relation identity

\[
q^{cur,s}=q^{old,w}.
\]

**通过标准：**

\[
\mathcal L_{rel}<10^{-6}
\]

在 float32 测试中。

---

### A-28 old model 无梯度

执行一个完整 incremental training step + backward。

**通过标准：**

```python
all(p.grad is None for p in old_model.parameters())
```

且 old model 参数 checksum 不变。

---

### A-29 current model 有梯度

在有效 supervised/assimilation/relation 情况下：

- segmentation head；
- backbone；
- projection head；

均有有限梯度。

---

### A-30 reliability 不可投机优化

对只含 assimilation/relation 的 synthetic step backward。

**通过标准：**

- \(L_i,C_i\) 无梯度；
-没有 trainable reliability parameter；
- loss 只能通过 current model prediction 下降。

---

### A-31 cutout mask

被 strong cutout 遮挡区域：

- 不进入 assimilation；
-不进入 relation KD。

**通过标准：** 将遮挡区 logits 改成任意值，loss 不变。

---

## 8. Continual 生命周期

### A-32 begin_site 深复制

**通过标准：**

- current/old 参数 storage 不同；
-初始化值相同；
- current 可训练；
- old 冻结。

---

### A-33 old 状态稳定

在当前站点训练 10 step 后：

- old model parameter checksum 不变；
- old anchor checksum 不变。

---

### A-34 current anchor 更新

在有有效 labeled/unlabeled 支持时：

- current anchor 发生预期变化；
- old anchor 不变。

---

### A-35 checkpoint 完整性

保存后检查必需 key。

**通过标准：** 所有 contract key 存在，schema/version 匹配。

---

### A-36 checkpoint 恢复一致

恢复后固定 batch：

- logits；
- relation；
- \(L_i\)；
- \(C_i\)；
-各 loss；
- anchor state；

最大绝对差异：

```text
float32: 1e-6
AMP path: 1e-4
```

---

### A-37 RNG 恢复

checkpoint 恢复后下一次 augmentation/sample order 与未中断运行一致。

**通过标准：** golden sequence hash 一致。

---

## 9. Golden Batch 回归

固定：

```text
seed = 20260818
one labeled batch
one unlabeled batch
one previous checkpoint
deterministic transforms
```

保存：

```text
golden/model_logits.sha256
golden/relation_probabilities.sha256
golden/learnability.npy
golden/compatibility.npy
golden/losses.json
golden/anchor_state.pt
golden/valid_counts.json
```

代码重构后：

- loss 相对误差 < `1e-5`；
- map 最大误差 < `1e-5` float32；
- counts 完全一致。

有意改变方法时必须更新版本与 golden baseline，禁止静默覆盖。

---

# Part B. Method Sanity Gates

## 10. 两病例过拟合

每个数据集选择两个有标注病例。

### B-01 Supervised overfit

**目标：**

- training Dice > 0.95；
- CE 显著下降；
-无 NaN；
-所有前景类可学习。

眼底 cup 等极小结构可使用 class-wise Dice，最低前景类建议 >0.85。

### B-02 Relation overfit

在同一两病例上训练 projection/anchor relation。

**目标：**

- relation prediction accuracy > 0.90；
- segmentation/relation agreement > 0.90；
-每类 anchor valid；
-前景 anchor 间 cosine 不全部接近 1。

失败时禁止进入 \(L_i,C_i\) 实验。

---

## 11. Tiny SSL smoke

单站点小子集：

```text
2 labeled + 4 unlabeled
10–20 epochs
```

检查：

- pseudo-label coverage 非零；
- classifier/anchor/deferred 三类计数合理；
- \(L_i\) 不全部为 0 或 1；
- anchor 更新但不爆炸；
- static SSL 不明显低于 supervised。

### 建议门槛

```text
0.05 < mean pseudo coverage < 0.95
0.05 < mean L_i on valid pixels < 0.95
```

这不是论文结果，只用于发现饱和/失效实现。

---

## 12. Tiny continual smoke

选择两个站点，每站点少量病例，训练 5–10 epoch。

检查：

1. 第一站点 checkpoint 可加载；
2. 第二站点 old model/anchor 冻结；
3. relation loss 非零；
4. site matrix 为 `2×2`；
5. second-site current performance 可提升；
6. first-site performance 可评估；
7. resume 能从中间 epoch 继续。

---

## 13. 可靠性分数分布

### B-03 \(L_i\) 非退化

在 dev set 统计：

- min/max；
- p10/p50/p90；
-按 class；
-边界/内部；
-训练进度。

**通过标准：**

- 不长期全部为 0；
-不长期全部为 1；
-各类至少有有效覆盖；
-后期低间隔区域参与率增加。

### B-04 \(C_i\) 非退化

增量站点：

- 不全部为 0；
-不全部为 1；
-域偏移较大站点的均值可低于相近站点；
- relation JS 有有效分布。

若全 0：检查 anchor/空间不一致或阈值过严。  
若全 1：检查 JS、agreement 或 old margin 是否失效。

---

## 14. Anchor 诊断

### B-05 anchor support

每类输出：

```text
labeled support
unlabeled support
number of updates
drift from site start
```

**通过标准：**

-所有真实存在类别有 labeled support；
- unlabeled support 受 \(L_i,C_i\) 控制；
-无单 batch 背景将 anchor 大幅覆盖；
- anchor drift 有限且非零。

### B-06 relation vs segmentation

在 labeled validation 上：

- segmentation head accuracy；
- relation head accuracy；
- agreement；
- disagreement 区域可视化。

关系分支必须提供额外语义信息，而不是随机或完全复制 segmentation head。

---

# Part C. Research Hypothesis Gates

这些门槛决定方法是否值得进入大规模 TMI 实验。

## 15. Learnability calibration

使用 hidden GT 的独立 analysis 进程，将 \(L_i\) 分成 10 个等频 bin。

计算：

\[
\operatorname{Acc}_b
=
P(\widetilde y_i=y_i\mid L_i\in b).
\]

### C-01 单调性

推荐目标：

- Spearman \(\rho(L_i,\text{correct}) > 0.5\)；
-最高 bin accuracy 显著高于最低 bin；
-最高 bin 不低于 confidence-only 最高 bin。

更理想：

\[
\rho>0.7.
\]

### C-02 coverage–accuracy curve

比较：

- confidence-only；
- logit margin only；
- relation margin only；
-完整 \(L_i\)。

在相同 coverage 下，完整 \(L_i\) 应具有更高 pseudo-label accuracy。

若不成立，先修改 \(L_i\) 定义，不得直接加入 multi-agent/RIC。

---

## 16. Compatibility calibration

将 \(C_i\) 分成 10 个等频 bin，使用 hidden GT 计算：

\[
\operatorname{OldAcc}_b
=
P(\widehat y_i^{old}=y_i\mid C_i\in b).
\]

### C-03 单调性

推荐目标：

- Spearman \(\rho(C_i,\text{old correct})>0.4\)；
-最高 bin old-model accuracy 高于总体至少 5 个百分点；
-最低 bin 明显包含更多 old-model 错误。

更理想：

\[
\rho>0.6.
\]

若 \(C_i\) 与历史正确率无关，compatibility routing 的方法依据不成立。

---

## 17. 四象限分析

使用固定分析阈值，例如各分数中位数或预注册阈值，仅用于分析。

报告：

| Quadrant | 比例 | 当前伪标签准确率 | old accuracy | 边界比例 |
|---|---:|---:|---:|---:|
| high L / high C |  |  |  |  |
| high L / low C |  |  |  |  |
| low L / high C |  |  |  |  |
| low L / low C |  |  |  |  |

预期：

- high-L 区域当前 pseudo-label accuracy 更高；
- high-C 区域 old-model accuracy 更高；
- high-L/low-C 体现新域可塑区域；
- low-L/high-C 体现历史恢复区域。

若四象限没有可区分语义，方法图和叙事需要复审。

---

## 18. Uniform KD vs compatibility routing

在 prostate 前两个站点、seed 0，固定预算比较：

1. Sequential SSL；
2. Sequential SSL + uniform logit/relation KD；
3. LCR-Seg V0.1。

### C-04 稳定性—可塑性门槛

推荐目标：

- LCR-Seg 的 previous-site Dice 高于 Sequential SSL；
- incoming-site Dice 不比 Sequential SSL 下降超过 2 个百分点；
- LCR-Seg 相比 uniform KD，incoming-site Dice 更高或相当；
- previous-site Dice 不显著低于 uniform KD。

核心不是单一总分，而是显示 compatibility routing 减少历史 negative transfer。

---

## 19. Relation KD vs feature L2

比较：

- full feature L2；
- uniform relation KL；
- compatibility relation KL。

预期：

- feature L2 对 incoming-site adaptation 抑制更强；
- relation KL 具有更好稳定性—可塑性平衡；
- compatibility relation KL 进一步改善。

若 relation KL 不优于 feature L2，需要检查 relation space 是否真正受监督。

---

## 20. 梯度冲突诊断

抽样计算：

\[
g_a=\nabla\mathcal L_{assim},
\qquad
g_c=\nabla\mathcal L_{rel}.
\]

统计：

\[
\cos(g_a,g_c).
\]

比较：

- uniform relation KD；
- compatibility-conditioned relation KD。

### C-05 预期

compatibility routing 应降低负余弦比例，或减少强负冲突的幅度。

该分析是证据，不是额外训练模块。V0.1 不加入 PCGrad/PGA。

---

## 21. 长序列抗遗忘门槛

在 prostate 主序列：

```text
A → B → C → D → E
F unseen
```

至少 seed 0 完整跑通后检查：

- complete site matrix；
- final average；
- previous-site average；
- incoming performance；
- BWT；
- FWT；
-最早站点 forgetting。

### C-06 是否触发 RIC

只有满足以下条件才考虑 V5：

1. \(L_i,C_i\) calibration 已通过；
2. compatibility routing 优于 uniform KD；
3. 但最早站点仍有明显、可重复遗忘；
4.遗忘主要来自当前输入未覆盖历史外观，而非实现错误。

若 calibration 未通过，禁止用 RIC 掩盖前置问题。

---

## 22. 是否触发 multi-agent

在 M&Ms 和 fundus 检查：

- 类内 relation feature 聚类；
- silhouette score；
-单 anchor 到 feature 的距离分布；
-背景/前景多峰性；
- relation error by site。

### C-07 进入条件

仅当：

- `K=2/4` 在开发 seed 中改善 relation calibration；
-至少两个数据集出现一致趋势；
-额外 memory/compute 可控；

才进入 V4。

---

# Part D. 实验流程验收

## 23. Baseline 一致性

所有 baseline 与 proposed MUST：

-相同 split；
-相同 backbone；
-相同训练 step；
-相同 labeled/unlabeled batch ratio；
-相同 supervised loss；
-相同 augmentation family；
-相同 final-checkpoint 规则。

### D-01 预算审计

生成自动报告：

```text
method
trainable params
forward passes per batch
steps/site
effective batch
GPU hours
peak memory
stored state size
```

---

## 24. Site matrix

每训练完一个 site \(i\)，评估所有配置 site \(j\)：

\[
R_{i,j}.
\]

### D-02 矩阵完整性

**通过标准：**

- shape 与 site 数一致；
- diagonal、lower、upper cells 不缺失；
-模型 checkpoint 与矩阵行一一对应；
- patient-level aggregation 正确；
- Dice/ASD/HD95 同时保存。

---

## 25. 断点恢复长实验

在一个 site 中途强制停止，恢复后完成训练。

### D-03 恢复等价

与不中断运行比较：

- final metric 差异在 deterministic 容差内；
- site matrix 相同；
- anchor state 相同；
- global/site step 连续；
- scheduler 连续。

---

## 26. 结果目录验收

每个实验必须包含：

```text
config.yaml
git_commit.txt
environment.txt
manifest_hash.txt
split_hash.txt
train_log.csv
per_case_metrics.csv
site_matrix_dice.csv
site_matrix_asd.csv
site_matrix_hd95.csv
checkpoint_final.pt
analysis/
  learnability_bins.csv
  compatibility_bins.csv
  quadrant_stats.csv
  anchor_stats.csv
  gradient_cosine.csv
```

缺失关键文件不得标记为完成。

---

# Part E. 失败处置

## 27. 立即停止条件

出现以下任一情况立即停止当前 run：

- NaN/Inf loss；
- old model 参数变化；
- old anchor 变化；
- hidden GT 出现在训练 batch；
- checkpoint schema 不完整；
- relation probability 不归一；
- anchor 全部 invalid；
-所有 \(L_i\) 或 \(C_i\) 长期饱和且无说明；
- site split 泄漏；
- HDF5 读取错误。

---

## 28. BLOCKER 报告模板

```markdown
# BLOCKER

## Scope
dataset / site / method version / commit

## Symptom
exact error or failed acceptance gate

## Reproduction
exact command and seed

## Evidence
logs, tensor stats, stack trace, hashes

## Confirmed facts
what is known

## Unknowns
what is not known

## Prohibited workaround
what must not be guessed or silently changed

## Proposed next action
smallest evidence-driven fix
```

---

# Part F. 最终签署清单

## 29. Codex 自检

- [ ] 所有 Hard Engineering Gates 通过；
- [ ] golden batch 通过；
- [ ] checkpoint resume 通过；
- [ ] 2-case overfit 通过；
- [ ] tiny SSL/continual smoke 通过；
- [ ] old model/anchor 无变化；
- [ ] \(L_i,C_i\) detach；
- [ ] hidden label 无泄漏；
- [ ] 结果目录完整；
- [ ] `STATUS.md` 已更新；
- [ ] 未实现 V4/V5。

## 30. GPT Pro 方法审核

- [ ] 公式与代码逐项对应；
- [ ] tensor space 未混用；
- [ ] current/old anchor 生命周期正确；
- [ ] weak/strong 对齐正确；
- [ ] pseudo-label 来源正确；
- [ ] \(L_i,C_i\) 无投机梯度；
- [ ] relation KL 方向正确；
- [ ]空集合和第一站点处理正确；
- [ ] calibration 分析支持分数含义；
- [ ]消融能够验证统一主线。

## 31. 正式实验启动条件

只有在以下三项同时满足后，才启动三个数据集多 seed 正式实验：

1. Hard Engineering Gates 全部通过；
2. Learnability 与 Compatibility calibration 至少在 development protocol 上成立；
3. compatibility routing 在前两个 prostate 站点表现出合理稳定性—可塑性权衡。

正式实验前记录冻结版本：

```text
method_version = 0.1
method_spec_sha256 = ...
implementation_contract_sha256 = ...
acceptance_tests_sha256 = ...
git_commit = ...
```

冻结后任何方法行为改变都必须升版本，不得在同一结果目录中混用。
