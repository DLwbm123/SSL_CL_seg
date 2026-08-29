# LCR-Seg V0.2a 协议修订与下一步实验计划

**版本：** V0.2a  
**日期：** 2026-08-27  
**状态：** `protocol_amendment_required_before_downstream_runs`  
**适用范围：** Fundus seed 0 的路由校准实验，以及通过门槛后才允许启动的 Prostate A→B pilot。

---

## 0. 执行结论

上一版 V0.2 将 R0 定义为：

```text
unit assimilation + uniform relation KD
```

但用于比较的旧参考实际是：

```text
continuous V0.1 learnability weighting + uniform relation KD
```

因此，两者不是同一个方法。已有的 literal R0 运行不能作为“共享路径等价性”验证，也不能作为原 R0→R3 因子实验的正式对照。

本修订作出以下决定：

1. **旧 V0.1 uniform-relation artifact 被正式定义为新的 R0。**
2. 已完成的 literal V0.2 R0 不删除、不覆盖，重新标记为辅助消融：
   ```text
   U0 = unit-all assimilation + uniform relation KD
   ```
3. 正式 R0–R3 改成一个无混杂的 \(2\times2\) 因子设计：
   - 因子 A：当前无标注知识如何进入训练；
   - 因子 C：历史 relation KD 如何被否决。
4. 在任何正式 R1–R3 运行之前，必须先完成 V0.1→V0.2a 的**共享路径桥接验证**。
5. 不加入 multi-agent、RIC、梯度投影、第三个 teacher、额外对比损失或其他未注册模块。
6. 当前 formal Fundus gate 仍是：
   ```text
   not_evaluated
   ```
   不能把 literal R0 的停止解释为 V0.2 方法失败。

---

## 1. 已冻结证据

### 1.1 旧 V0.1 uniform-relation 参考

```text
run:
/home/jiangsuiyang/SSL_CL/runs/
fundus_seed0_lcrseg_uniform_relation_kd_full200e

method:
lcrseg_v0_1

assimilation:
continuous V0.1 learnability weighting

consolidation:
uniform relation KD
```

验证集结果：

| 指标 | 数值 |
|---|---:|
| Final average Dice | 0.6551054533 |
| BWT | -0.1184621104 |
| Incoming Dice | 0.7340801936 |
| Previous-site Dice | 0.6759458886 |

### 1.2 已完成 literal V0.2 R0

```text
run:
/home/jiangsuiyang/SSL_CL/runs/
fundus_seed0_lcrseg_v0_2_r0_uniform_full200e

assimilation:
unit weight for every valid pseudo-label

consolidation:
uniform relation KD
```

验证集结果：

| 指标 | 数值 |
|---|---:|
| Final average Dice | 0.6309953259 |
| BWT | -0.1111381492 |
| Incoming Dice | 0.7050874253 |
| Previous-site Dice | 0.6687593930 |

该运行已经完成 13,400/13,400 steps，工程上有效，但其语义不等价于旧参考。

### 1.3 对已有 literal R0 的重新分类

后续报告中将其固定命名为：

```text
U0_unit_all_uniform_relation
```

其用途是回答：

> 完全取消 continuous learnability weighting，改为所有有效伪标签单位权重，会发生什么？

它是一个有价值的辅助消融，但不属于新的正式 R0–R3 因子实验。

---

## 2. 新的正式 R0–R3 定义

### 2.1 两个实验因子

#### 因子 A：Assimilation policy

**A0 — Legacy Continuous**

沿用 V0.1 的完整 continuous learnability weighting：

\[
\mathcal L_{\mathrm{assim}}^{A0}
=
\frac{
\sum_i
L_i\,
\mathbb I_i^{\mathrm{valid}}
\operatorname{CE}(p_i^s,\tilde y_i)
}{
\sum_i
L_i\mathbb I_i^{\mathrm{valid}}+\epsilon
}.
\]

强制要求：

- 调用 V0.1 已验证的同一计算函数；
- 使用相同 `valid pseudo-label mask`；
- \(L_i\) 继续 `detach`；
- 不重写一个“近似等价”版本；
- 该模式必须能够重现旧 uniform-relation artifact。

**A1 — Class-wise Progressive Admission**

仍计算同一个 \(L_i\)，但 \(L_i\) 不再直接决定梯度幅度，而只用于类别内准入：

\[
\pi(\rho)=0.40+0.40\rho,
\qquad
\rho\in[0,1].
\]

对每个预测类别 \(c\)，在当前 batch 的有效伪标签中选择 \(L_i\) 最高的 \(\pi(\rho)\) 比例：

\[
m_i^{\mathrm{adm}}
=
\mathbb I
\left[
L_i
\geq
Q_c(1-\pi(\rho))
\right].
\]

然后：

\[
\mathcal L_{\mathrm{assim}}^{A1}
=
\frac{
\sum_i
m_i^{\mathrm{adm}}
\operatorname{CE}(p_i^s,\tilde y_i)
}{
\sum_i m_i^{\mathrm{adm}}+\epsilon
}.
\]

固定规则：

- schedule 按每个 site 内部的训练进度线性变化；
- 开始覆盖 40%，结束覆盖 80%；
- 按预测类别分别求分位数；
- 某类别有效像素少于 32 时，回退到当前 batch 的全局分位数；
- 若类别存在有效伪标签，至少保留 1 个像素；
- admitted pixels 使用单位权重；
- hidden GT 不参与准入；
- 记录每类、每 epoch 的实际覆盖率。

---

#### 因子 C：Consolidation policy

**C0 — Uniform Relation KD**

所有有效 relation-grid 像素使用单位权重：

\[
\mathcal L_{\mathrm{rel}}^{C0}
=
\frac{
\sum_i
\operatorname{KL}
\left(
\operatorname{sg}[q_i^-]
\parallel
q_i^{t,s}
\right)
}{
N_{\mathrm{valid}}+\epsilon
}.
\]

该分支必须与旧 V0.1 `use_compatibility=false` 完全一致。

**C1 — Calibrated Teacher-Validity Rejection**

V0.1 compatibility 存在一个结构性问题：其门控包含 current–old agreement 和 JS divergence。当 current model 已经偏离一个正确的 old model 时，这种分歧恰好是 KD 应介入的位置，而不是应关闭 KD 的位置。

因此，V0.2a 不再用 current–old agreement 估计历史 teacher 是否可靠，而定义一个**只依赖 frozen old model 的 teacher-validity score**。

对 relation-grid 像素 \(i\)：

\[
V_i^{\mathrm{raw}}
=
\left[
V_i^{\mathrm{margin}}
\cdot
V_i^{\mathrm{certainty}}
\cdot
V_i^{\mathrm{spatial}}
\right]^{1/3}.
\]

其中：

\[
V_i^{\mathrm{margin}}
=
\sigma\left(
\frac{\Delta_i^-}{\tau_m}
\right),
\]

\[
V_i^{\mathrm{certainty}}
=
1-\frac{H(p_i^-)}{\log C},
\]

\[
V_i^{\mathrm{spatial}}
\in[0,1]
\]

为 frozen old model 的局部空间一致性。

强制约束：

- \(V_i^{\mathrm{raw}}\) 不允许读取 current-model tensor；
- current–old JS 和 prediction agreement 只做诊断，不进入 teacher-validity gate；
- old model 和 raw validity 全部 detached；
- 第一站点没有 old model，C1 自动退化为无 relation loss，与 C0 一致。

---

### 2.2 Teacher-validity calibration

在每个新 site 开始训练前，使用该 site 的 `train_labeled` 数据，对 frozen old model 进行一次只读前向：

1. 计算每个 labeled pixel 的 \(V_i^{\mathrm{raw}}\)；
2. 以 old model 预测类别作为条件类别；
3. 目标为：
   \[
   y_i^{\mathrm{valid}}
   =
   \mathbb I[\hat y_i^-=y_i].
   \]
4. 每类最多均匀采样 100,000 个像素；
5. 每类至少需要 2,048 个像素，否则回退到 global calibrator；
6. 构建 20 个等频 bin；
7. 每个 bin 使用 Laplace smoothing 估计正确率；
8. 使用 pooled-adjacent-violators algorithm（PAVA）强制映射单调非降；
9. 保存 piecewise-constant calibration table；
10. calibration 仅在 site 开始时计算一次，当前 site 训练期间冻结。

得到：

\[
\hat V_i
=
g_{\hat y_i^-}(V_i^{\mathrm{raw}}).
\]

禁止：

- 使用 train-unlabeled hidden GT；
- 使用 val/test GT；
- 每 epoch 根据验证集重调；
- 用额外可训练神经网络拟合 calibrator。

---

### 2.3 Rejection-only relation weighting

固定参数：

```yaml
teacher_validity_threshold: 0.70
rejected_weight_floor: 0.50
max_rejected_fraction_per_predicted_class: 0.20
```

初始候选：

\[
r_i
=
\mathbb I[\hat V_i<0.70].
\]

若某个 old-predicted class 的候选拒绝比例超过 20%，只保留该类别 \(\hat V_i\) 最低的 20% 为 rejected。

最终权重：

\[
w_i^{\mathrm{rel}}
=
\begin{cases}
0.50,& i\text{ 被拒绝},\\
1.00,& \text{其他有效像素}.
\end{cases}
\]

relation loss 为：

\[
\mathcal L_{\mathrm{rel}}^{C1}
=
\frac{
\sum_i
w_i^{\mathrm{rel}}
\operatorname{KL}
\left(
\operatorname{sg}[q_i^-]
\parallel
q_i^{t,s}
\right)
}{
\sum_iw_i^{\mathrm{rel}}+\epsilon
}.
\]

该设计的原则是：

> 默认保留 relation KD，只对 incoming site 上有标注证据表明 old teacher 明显不可靠的区域进行有限降权。

---

### 2.4 正式 \(2\times2\) 因子表

| 变体 | Assimilation | Consolidation |
|---|---|---|
| **R0** | A0 Legacy Continuous | C0 Uniform Relation |
| **R1** | A1 Progressive Admission | C0 Uniform Relation |
| **R2** | A0 Legacy Continuous | C1 Teacher-Validity Rejection |
| **R3** | A1 Progressive Admission | C1 Teacher-Validity Rejection |

辅助消融：

| 变体 | Assimilation | Consolidation | 状态 |
|---|---|---|---|
| **U0** | Unit weight for all valid pseudo-labels | Uniform Relation | 已完成，不重跑 |

这样可以分别估计：

\[
\text{Assimilation main effect}
=
\frac{(R1-R0)+(R3-R2)}{2},
\]

\[
\text{Consolidation main effect}
=
\frac{(R2-R0)+(R3-R1)}{2},
\]

\[
\text{Interaction}
=
R3-R2-R1+R0.
\]

---

## 3. 代码协议修订

禁止继续用多个布尔量隐式组合出不同语义。改为显式枚举：

```yaml
protocol_id: lcrseg_v0_2a

assimilation_mode:
  legacy_continuous_v01
  progressive_admission
  unit_all

consolidation_mode:
  uniform_relation
  calibrated_teacher_rejection
```

每次运行必须在 `config.yaml`、checkpoint 和 branch report 中保存：

```text
protocol_id
assimilation_mode
consolidation_mode
learnability_formula_version
teacher_validity_formula_version
calibrator_version
progressive_schedule
rejection_threshold
rejection_floor
rejection_cap
```

不允许：

- `use_learnability=true/false` 同时承担不同含义；
- `progressive_admission=false` 被自动解释为 unit assimilation；
- 未记录 resolved method defaults；
- 通过 run name 猜测实际语义。

---

## 4. 正式运行前的共享路径桥接

### 4.1 Formal R0 artifact

新的正式 R0 直接采用已经冻结的旧 artifact：

```text
/home/jiangsuiyang/SSL_CL/runs/
fundus_seed0_lcrseg_uniform_relation_kd_full200e
```

不需要再次浪费 13,400 steps，但新 V0.2a 代码必须证明其 R0 分支与旧 V0.1 计算路径一致。

### 4.2 Golden bridge

从同一 batch、同一 checkpoint 和同一 RNG state 比较：

```text
legacy lcrseg_v0_1:
use_learnability=true
use_compatibility=false

v0_2a R0:
assimilation_mode=legacy_continuous_v01
consolidation_mode=uniform_relation
```

比较：

```text
pseudo-label mask
learnability map
assimilation numerator/denominator
relation distribution
relation numerator/denominator
total loss
anchor update proposal
all logged branch counts
```

容差：

```text
tensor max_abs_error <= 1e-6
loss abs_error <= 1e-7
integer counts exact
```

### 4.3 500-step paired bridge

使用旧 R0 的 REFUGE site-end checkpoint，复制为两个完全相同的起点，在 RIM-ONE-r3 上分别运行：

```text
legacy V0.1 path
V0.2a R0 path
```

要求：

- 相同 500 optimizer steps；
- 相同 dataloader order；
- 相同 augmentation RNG；
- 相同 AMP 行为；
- 每 50 steps 比较 loss 和 branch counts；
- step 500 比较 current model、anchor bank、optimizer 和 scheduler。

容差：

```text
max parameter abs error <= 1e-6
max anchor abs error <= 1e-6
loss abs error <= 1e-6
optimizer step count exact
scheduler state exact
```

若失败：

```text
HARD_STOP_R0_BRIDGE_MISMATCH
```

不得启动 R1、R2、R3。

若通过，正式 R0 使用旧完整 artifact，不再重跑 full 200 epochs。

---

## 5. 新增单元测试

至少新增：

```text
test_assimilation_mode_is_explicit_enum.py
test_legacy_continuous_calls_v01_path.py
test_legacy_continuous_golden_equivalence.py
test_progressive_admission_classwise_fraction.py
test_progressive_admission_schedule.py
test_progressive_admission_no_hidden_gt.py
test_teacher_validity_independent_of_current_model.py
test_teacher_validity_old_model_only.py
test_teacher_validity_calibrator_monotonic.py
test_teacher_validity_calibrator_class_fallback.py
test_rejection_cap_per_class.py
test_rejection_floor.py
test_relation_ess_not_collapsed.py
test_protocol_fields_in_checkpoint.py
test_u0_not_registered_as_formal_r0.py
```

关键测试：

### 5.1 Teacher validity 独立性

固定 old model 和输入，任意扰动 current model 参数：

\[
V_i^{\mathrm{raw}}(\theta_t^{(1)})
=
V_i^{\mathrm{raw}}(\theta_t^{(2)}).
\]

必须数值一致。

### 5.2 Calibrator 单调性

若：

\[
V_a^{\mathrm{raw}}\leq V_b^{\mathrm{raw}},
\]

则：

\[
g(V_a^{\mathrm{raw}})
\leq
g(V_b^{\mathrm{raw}}).
\]

### 5.3 Rejection cap

任何 predicted class：

\[
\frac{N_{\mathrm{rejected},c}}{N_{\mathrm{valid},c}}
\leq0.20.
\]

### 5.4 Effective sample size

relation weights 的：

\[
N_{\mathrm{eff}}
=
\frac{(\sum_iw_i)^2}{\sum_iw_i^2}
\]

必须被记录。

---

## 6. 快速 pilot gate

共享路径桥接通过后，不立即启动全部正式运行。

### 6.1 R1–R3 1,000-step pilot

固定：

```text
dataset: fundus
seed: 0
site transition: REFUGE -> RIM-ONE-r3
start checkpoint: corresponding REFUGE site-end checkpoint
steps: 1000
```

R1–R3 分别运行 1,000 steps，仅用于工程与分支验收，不用于选择超参数。

必须满足：

- 无 NaN/Inf；
- 无 old-model gradient；
- historical anchors 不更新；
- R1/R3 的实际 admission coverage 与 schedule 误差不超过 5 个百分点；
- R2/R3 每类 rejection 不超过 20%；
- R2/R3 relation ESS 不低于 uniform branch 的 80%；
- calibrator 文件完整；
- hidden GT usage 为 0；
- current–old JS 不进入 gate 的代码路径；
- checkpoint resume 精确通过。

pilot 只允许发现工程问题，不允许根据 pilot validation Dice 修改固定参数。

---

## 7. Fundus 正式运行

pilot 全部通过后，按顺序运行：

```text
R1 -> R2 -> R3
```

固定：

```text
dataset: fundus
seed: 0
site order: REFUGE -> RIM-ONE-r3 -> Drishti-GS
evaluation role: val
steps: 13,400
epochs/site: 200
manifest SHA-256:
0622f54f42f05d6ef87f9dc89ee9435cf8da03c6c30cd970db6ea167e00dd8a3
split SHA-256:
f250d97aea1f36f21899f5dd40bb6c9a819e7755aee458c8ee27506496b46a88
```

建议 run names：

```text
fundus_seed0_lcrseg_v0_2a_r1_progressive_uniform_full200e
fundus_seed0_lcrseg_v0_2a_r2_legacy_teacherreject_full200e
fundus_seed0_lcrseg_v0_2a_r3_progressive_teacherreject_full200e
```

不覆盖：

```text
fundus_seed0_lcrseg_uniform_relation_kd_full200e
fundus_seed0_lcrseg_v0_2_r0_uniform_full200e
```

---

## 8. 必须输出的分析

### 8.1 主结果

报告 R0、R1、R2、R3 和 U0：

```text
Final average Dice
BWT
Incoming Dice
Previous-site Dice
完整 site matrix
per-site forgetting
```

### 8.2 因子效应

对每个指标计算：

```text
assimilation main effect
consolidation main effect
interaction
```

### 8.3 Progressive admission

按 site、epoch、class 报告：

```text
target coverage
realized coverage
pseudo-label accuracy
admitted pixel count
deferred pixel count
boundary/interior ratio
```

hidden GT 仅可在训练完成后的独立 post-hoc process 中计算 pseudo-label accuracy。

### 8.4 Teacher validity

对每个 site 和 class 报告：

```text
raw validity histogram
calibrated validity histogram
20-bin empirical correctness
Brier score before/after calibration
ECE before/after calibration
rejected fraction
rejected-region old-model correctness
retained-region old-model correctness
relation ESS
```

### 8.5 诊断但不参与 gate

继续记录：

```text
current-old JS
current-old semantic agreement
assimilation/relation gradient cosine
four-quadrant statistics
```

但它们不得参与 C1 的训练权重。

---

## 9. Fundus progression gate

### 9.1 工程 gate

必须全部通过：

- 所有新增测试通过；
- R0 bridge 通过；
- R1–R3 全部 13,400 steps；
- 无 NaN；
- 无 hidden-label leakage；
- old model 无梯度；
- historical anchor 无更新；
- no AMP silent failure；
- checkpoint resume 通过；
- rejection cap 与 ESS 门槛通过。

### 9.2 研究 progression gate

以正式 R0 为基准：

```text
R0 Final average = 0.6551054533
R0 BWT           = -0.1184621104
R0 Incoming      = 0.7340801936
R0 Previous      = 0.6759458886
```

R3 至少满足：

\[
\text{Final}_{R3}
\geq
\text{Final}_{R0}+0.003,
\]

\[
\text{BWT}_{R3}
\geq
\text{BWT}_{R0}+0.005,
\]

\[
\text{Incoming}_{R3}
\geq
\text{Incoming}_{R0}-0.010,
\]

\[
\text{Previous}_{R3}
\geq
\text{Previous}_{R0}.
\]

同时：

- R3 在 R0–R3 中是 Pareto non-dominated；
- calibrated retained pixels 的 old-model correctness 高于 rejected pixels；
- relation ESS ≥ uniform 的 80%；
- R3 不得通过大幅牺牲 optic cup 或任一前景类获得平均提升。

### 9.3 后续决定

- 若 R3 通过：启动 Prostate RUNMC→BMC pilot。
- 若 R3 未通过，但 R1 或 R2 单独通过：停止自动扩展，输出“单因素有效、组合无正交增益”的报告，由用户决定是否简化方法。
- 若 R1、R2、R3 均未超过 R0：冻结 V0.2a 失败结果，不加入 multi-agent 或 RIC。
- 不根据 U0 的结果修改预注册门槛。

---

## 10. Prostate pilot 条件

只有 Fundus R3 progression gate 通过后，才运行：

```text
RUNMC -> BMC
seed 0
20% labeled
相同 method hyperparameters
不重新调 teacher-validity threshold
不重新调 admission schedule
```

需要输出：

```text
RUNMC final Dice
BMC incoming Dice
RUNMC forgetting
BWT
teacher-validity calibration
progressive admission coverage
relation ESS
```

Prostate pilot 未通过时，不启动完整 A→B→C→D→E。

---

## 11. 状态与产物

新增：

```text
reports/experiment_status/
PROTOCOL_AMENDMENT_V0_2A.md
PROTOCOL_AMENDMENT_V0_2A.json
V0_2A_R0_BRIDGE_REPORT.md
V0_2A_R0_BRIDGE_REPORT.json
V0_2A_FUNDUS_COMPLETION.md
V0_2A_FUNDUS_COMPLETION.json

reports/analysis/v0_2a/
factorial_effects.csv
admission_coverage.csv
teacher_validity_calibration.csv
teacher_validity_calibration.json
relation_effective_sample_size.csv
classwise_results.csv
regionwise_results.csv
gradient_diagnostics.csv
```

每个 run 保存：

```text
config.yaml
protocol.json
parent_artifact.json
environment.txt
train_log.csv
branch_coverage.csv
site_matrix_dice.csv
per_case_metrics.csv
checkpoint_site_*.pt
checkpoint_final.pt
```

---

## 12. 明确禁止事项

本轮禁止：

- 把 literal U0 改名或伪装成 formal R0；
- 删除任何已有 run；
- 跳过 R0 bridge；
- 用 hidden GT 训练 calibrator；
- 使用 current–old agreement 或 JS 作为 teacher-validity gate；
- 根据 R1/R2 中间结果修改 R3 参数；
- 增加 multi-agent；
- 增加 RIC；
- 增加 EWC、gradient surgery 或第三个 teacher；
- 在 Fundus gate 前启动 Prostate；
- 把未运行的实验写成已完成。

---

## 13. 最短执行路径

```text
协议修订与显式枚举
        ↓
新增测试
        ↓
golden bridge
        ↓
500-step paired bridge
        ↓
R1/R2/R3 1000-step pilot
        ↓
R1/R2/R3 Fundus full run
        ↓
2×2 factorial analysis
        ↓
Fundus progression gate
        ↓
条件式 Prostate A→B
```
