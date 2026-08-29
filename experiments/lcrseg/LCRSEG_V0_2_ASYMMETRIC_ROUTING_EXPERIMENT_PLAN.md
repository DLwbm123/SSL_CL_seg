# LCR-Seg V0.2 下一步实验计划

## 1. 计划目的

本计划用于在不扩展网络结构、不引入额外持续学习模块的前提下，修正 LCR-Seg V0.1 中已经被实验定位到的路由问题。

当前工程结论是：

- 数据、统一训练引擎、checkpoint/resume、HDF5 DataLoader、V0–V3 张量与梯度不变量均已通过；
- 单 anchor relation field 和 historical relation consolidation 具有有效信号；
- Current Learnability \(L_i\) 能预测伪标签正确率，但作为连续损失权重时偏向当前站点学习，并未改善整体稳定性；
- Historical Compatibility \(C_i\) 与旧模型正确率存在总体关联，但高分段不单调；
- uniform relation KD 明显优于 compatibility-weighted relation KD；
- 因此，当前失败点是 **\(L_i\) 与 \(C_i\) 的对称连续加权方式**，而不是 relation field 本身。

本轮只验证一个预注册假设：

\[
\boxed{
\text{严格选择当前知识，保守否决历史知识}
}
\]

具体地：

- \(L_i\) 不再直接缩放每个伪标签像素的梯度，而用于决定该像素是否进入当前域知识吸收；
- relation KD 默认保留；
- \(C_i\) 不再对全体历史监督进行连续衰减，而只对经当前站点标注数据校准后仍明显不可靠的区域施加有下限的降权。

内部版本命名：

> **LCR-Seg V0.2: Asymmetric Reliability Routing**

---

## 2. 已有事实与基准线

所有数值均来自冻结的 Fundus seed-0 验证结果，训练预算统一为 13,400 steps。

| 方法 | Final avg | BWT | Incoming | Previous |
|---|---:|---:|---:|---:|
| Sequential-SSL | 0.6473 | -0.1557 | 0.7511 | 0.6552 |
| Uniform-KD/LwF | 0.6540 | -0.1372 | 0.7455 | 0.6747 |
| LCR-Seg V0.1 full | 0.6211 | -0.1596 | 0.7275 | 0.6369 |
| LCR V0.1 without learnability | 0.6379 | -0.0965 | 0.7022 | 0.6764 |
| LCR V0.1 with uniform relation KD | **0.6551** | -0.1185 | 0.7341 | **0.6759** |
| LCR V0.1 without relation consolidation | 0.5897 | -0.2193 | 0.7360 | 0.5963 |

现有最强的 LCR 内部参考点是：

\[
\boxed{
\text{R0 reference: unit assimilation + uniform relation KD}
}
\]

其目标值为：

\[
\text{Final}=0.6551,\quad
\text{BWT}=-0.1185,\quad
\text{Incoming}=0.7341,\quad
\text{Previous}=0.6759.
\]

V0.2 必须优先超过这个参考点，而不是只超过 V0.1 full。

---

## 3. 本轮范围

### 3.1 固定内容

以下内容全部保持不变：

- 数据根目录：`/home/jiangsuiyang/SSL_CL`
- 代码根目录：`/home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg`
- 运行根目录：`/home/jiangsuiyang/SSL_CL/runs`
- Python：`/home/jiangsuiyang/anaconda3/envs/py38/bin/python`
- Fundus seed-0 split、manifest 与 checksum；
- 2D U-Net、projection head、single-anchor \(K=1\)；
- weak/strong augmentation；
- optimizer、scheduler、200 epochs/site；
- 13,400 optimizer-step 预算；
- final checkpoint 作为主结果；
- validation role 用于方法比较；
- hidden GT 只允许在 checkpoint 冻结后的 analysis 进程使用。

### 3.2 本轮允许修改

仅允许修改：

1. \(L_i\) 如何控制当前伪标签参与训练；
2. \(C_i\) 如何控制 historical relation KD；
3. 对应的日志、校准器、分析脚本与单元测试。

### 3.3 明确禁止

本轮不允许加入：

- multi-agent 或 \(K>1\)；
- RIC、EWC、MAS 或其他参数正则；
- 第三个 teacher 或 EMA teacher；
- replay、diffusion、VAE、style encoder；
- 通道分解、额外 contrastive loss、triplet loss；
- 新 backbone；
- 数据或 split 修改；
- hidden GT 参与训练、阈值估计或校准器拟合；
- 在 Fundus gate 通过前启动完整 Prostate 序列；
- 修改或覆盖已有 V0.1 run artifacts。

---

## 4. 研究假设

### H1：Learnability 更适合作为准入信号，而不是梯度幅度

V0.1 中：

\[
L_i\uparrow
\Rightarrow
P(\tilde y_i=y_i)\uparrow,
\]

说明 \(L_i\) 能预测伪标签可靠性。

但：

\[
\text{伪标签正确率校准}
\neq
\text{最优梯度权重}.
\]

高 \(L_i\) 像素可能主要位于背景或容易的结构内部。持续对它们赋予更大梯度，会强化当前站点的易学模式，而不一定改善历史保持或边界学习。

因此 V0.2 将 \(L_i\) 改为 **class-wise progressive admission**。

### H2：Relation KD 应默认保留，Compatibility 只负责低端否决

消融已经表明：

\[
\mathcal L_{\mathrm{rel}}
\]

具有明显抗遗忘作用，而当前 \(C_i\) 的全范围连续加权会削弱有效历史监督。

因此 V0.2 默认保留 uniform relation KD，只对经标注数据校准后仍明显不可信的少量区域降低权重，并保持非零权重下限。

### H3：旧模型在新站点上的可靠性可以由当前少量标注数据校准

每个站点有 20% labeled data。可直接在这些可见标签上估计：

\[
P(\hat y_i^{old}=y_i\mid C_i).
\]

该校准只使用当前站点可见标注，不使用 train-unlabeled hidden GT，因此属于合法训练信息。

### H4：全局 Compatibility 失配可能来自类别与空间组成偏差

需要区分：

- background / disc rim / optic cup；
- interior / boundary；
- 大区域 / 小连通区域。

若 class-wise 或 region-wise calibration 明显优于 global calibration，则 V0.1 的失败主要来自像素组成偏差，而不是 relation field 无效。

---

## 5. Phase A：先完成 V0.1 路由诊断

在改动方法公式前，先对已有以下 checkpoint 做统一分析：

```text
fundus_seed0_lcrseg_v0_1_full200e
fundus_seed0_lcrseg_uniform_relation_kd_full200e
fundus_seed0_lcrseg_no_learnability_full200e
fundus_seed0_sequential_ssl_full200e
```

### 5.1 类别分解

分别针对：

```text
0 background
1 optic disc rim
2 optic cup
```

输出：

- \(L_i\) 十分位与 pseudo-label accuracy；
- \(C_i\) 十分位与 old-model correctness；
- pseudo-label coverage；
- relation-KD weight；
- effective pixel count。

### 5.2 空间分解

基于 hidden GT 只在 post-hoc analysis 中定义：

- interior：距离真实边界大于 3 像素；
- boundary band：距离真实边界不超过 3 像素；
- small component：预测连通区域面积低于该类别训练集第 10 百分位。

输出每类区域中的：

- 像素比例；
- \(L_i\)、\(C_i\) 分布；
- pseudo-label accuracy；
- old-model correctness；
- relation JS；
- 当前/旧模型 agreement。

### 5.3 有效样本量

对 assimilation 和 consolidation 权重分别计算：

\[
N_{\mathrm{eff}}
=
\frac{\left(\sum_i w_i\right)^2}
{\sum_i w_i^2+\epsilon}.
\]

报告：

- `weighted_pixel_count`；
- `effective_pixel_count`；
- `N_eff / valid_count`；
- class-wise 与 global 结果。

### 5.4 梯度诊断

在固定 golden batches 上分别计算：

\[
g_{\mathrm{assim}}
=
\nabla_\theta \mathcal L_{\mathrm{assim}},
\qquad
g_{\mathrm{rel}}
=
\nabla_\theta \mathcal L_{\mathrm{rel}}.
\]

输出：

- \(\|g_{\mathrm{assim}}\|_2\)；
- \(\|g_{\mathrm{rel}}\|_2\)；
- \(\cos(g_{\mathrm{assim}},g_{\mathrm{rel}})\)；
- 高/低 \(L_i\) 和高/低 \(C_i\) 区域的梯度贡献。

不得在本阶段加入 gradient surgery。

### 5.5 Phase A 产物

```text
reports/experiment_status/V0_1_ROUTING_DIAGNOSTIC.md
reports/analysis/v0_1_routing/classwise_calibration.csv
reports/analysis/v0_1_routing/regionwise_calibration.csv
reports/analysis/v0_1_routing/effective_sample_size.csv
reports/analysis/v0_1_routing/gradient_diagnostics.csv
reports/analysis/v0_1_routing/*.png
```

---

## 6. V0.2 方法定义

## 6.1 Learnability：Class-Wise Progressive Admission

保持 V0.1 中 \(L_i\) 的计算公式和 detach 规则不变，只改变其使用方式。

对当前 weak-view 产生的有效伪标签集合：

\[
\mathcal V_c
=
\{i:\tilde y_i=c,\ i\text{ satisfies the existing valid pseudo-label rule}\}.
\]

定义站点内训练进度：

\[
\rho=\frac{\text{current site optimizer step}}
{\text{total optimizer steps of current site}}
\in[0,1].
\]

预注册参与比例：

\[
\pi(\rho)
=
0.4+0.4\rho.
\]

因此：

- 站点训练开始时，每个预测类别只保留最高 \(40\%\) 的 \(L_i\) 像素；
- 站点训练结束时，逐步扩大到最高 \(80\%\)；
- 不允许覆盖率重新升到 V0.1 的约 \(98\%\)。

对每个类别 \(c\)，按 detached \(L_i\) 排序，保留：

\[
k_c
=
\max\left(1,\left\lceil \pi(\rho)|\mathcal V_c|\right\rceil\right)
\]

个最高分像素。若 \(|\mathcal V_c|=0\)，则该类不产生 assimilation loss。

准入掩码：

\[
m_i^{\mathrm{assim}}
=
\mathbb I[i\in\operatorname{TopK}_{\mathcal V_c}(L_i,k_c)].
\]

新的 assimilation loss：

\[
\mathcal L_{\mathrm{assim}}^{v0.2}
=
\frac{
\sum_i
m_i^{\mathrm{assim}}
\operatorname{CE}(p_i^s,\tilde y_i)
}{
\sum_i m_i^{\mathrm{assim}}+\epsilon
}.
\]

关键约束：

- 选中像素使用 unit weight，不再乘 \(L_i\)；
- `m_assim` 必须 detach；
- class-wise 选择，禁止 global top-k；
- pseudo-label 生成分支不变；
- weak/strong 几何对齐规则不变；
- cutout/invalid pixels 不得参与排序或损失。

---

## 6.2 Compatibility：Labeled-Calibrated Rejection-Only Consolidation

保持 V0.1 raw compatibility \(C_i^{raw}\) 的定义不变，但不再直接作为 relation KD 的连续权重。

### 6.2.1 校准数据

每个增量站点只使用当前站点的 `train_labeled`：

1. 当前模型与 frozen old model 在 weak/no-random-geometry calibration view 上推理；
2. 计算 \(C_i^{raw}\)；
3. 计算 old relation prediction：
   \[
   \hat y_i^{old}=\arg\max q_i^{old};
   \]
4. 利用可见 GT 计算：
   \[
   t_i=\mathbb I[\hat y_i^{old}=y_i].
   \]

禁止读取 `train_unlabeled` 的 hidden GT。

### 6.2.2 单调校准器

对每个 old-predicted class：

- 收集 \((C_i^{raw},t_i)\)；
- 构造 10 个等频 bins；
- 使用加权 PAVA 拟合单调非降的 piecewise-constant mapping：
  \[
  \widehat C_i=g_c(C_i^{raw});
  \]
- Laplace smoothing：每个 bin 的正确数与错误数各加 1；
- 若某类有效像素少于 500，回退到 global calibrator；
- 若 global 有效像素也少于 500，当前校准周期使用 uniform relation KD。

校准器：

- 不可训练；
- 不进入 optimizer；
- 每 10 epochs 更新一次；
- 当前站点前 10 epochs 使用 uniform relation KD；
- 状态必须进入 checkpoint；
- resume 后映射完全一致。

### 6.2.3 保守否决规则

对于 relation-valid pixels，先依据校准概率得到候选拒绝集合：

\[
\mathcal R_c
=
\{i:\hat y_i^{old}=c,\ \widehat C_i<0.7\}.
\]

每类最多拒绝其 relation-valid pixels 的最低 \(20\%\)。若候选超过 20%，仅选择 \(\widehat C_i\) 最低的 20%。

最终 consolidation weight：

\[
w_i^{\mathrm{cons}}
=
\begin{cases}
0.5,& i\in\mathcal R_c^{cap},\\
1.0,& \text{otherwise}.
\end{cases}
\]

注意：

- 不允许权重为 0；
- 不允许对高分区域继续放大到 1 以上；
- 不允许对全部像素使用 \(\widehat C_i\) 连续乘法；
- 若校准器不可用，全部使用 1；
- 第一站点没有 old model，relation loss 为 0。

新的 relation loss：

\[
\mathcal L_{\mathrm{rel}}^{v0.2}
=
\frac{
\sum_i
w_i^{\mathrm{cons}}
\operatorname{KL}
\left(
\operatorname{sg}[q_i^{old}]
\|
q_i^{current,strong}
\right)
}{
\sum_iw_i^{\mathrm{cons}}+\epsilon
}.
\]

---

## 6.3 V0.2 总目标

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{sup}}
+
\lambda_a\mathcal L_{\mathrm{assim}}^{v0.2}
+
\lambda_c\mathcal L_{\mathrm{rel}}^{v0.2}.
}
\]

保持 V0.1 的 \(\lambda_a,\lambda_c\)、optimizer、scheduler 与训练预算不变。

---

## 7. 预注册的四个实验变体

所有实验使用 Fundus seed-0、13,400 steps、validation role。

| ID | Assimilation | Consolidation | 目的 |
|---|---|---|---|
| R0 | 所有现有 valid pseudo-label，unit weight | uniform relation KD | 复现当前最强内部参考 |
| R1 | class-wise progressive \(L_i\) admission | uniform relation KD | 单独验证 learnability 准入 |
| R2 | 所有现有 valid pseudo-label，unit weight | calibrated rejection-only KD | 单独验证 compatibility 否决 |
| R3 | class-wise progressive \(L_i\) admission | calibrated rejection-only KD | V0.2 full |

建议配置与运行目录：

```text
configs/experiments/lcrseg_v0_2_r0_uniform.yaml
configs/experiments/lcrseg_v0_2_r1_learnability_admission.yaml
configs/experiments/lcrseg_v0_2_r2_compatibility_reject.yaml
configs/experiments/lcrseg_v0_2_r3_asymmetric_full.yaml

runs/fundus_seed0_lcrseg_v0_2_r0_uniform_full200e
runs/fundus_seed0_lcrseg_v0_2_r1_learnability_full200e
runs/fundus_seed0_lcrseg_v0_2_r2_compatibility_full200e
runs/fundus_seed0_lcrseg_v0_2_r3_asymmetric_full200e
```

如果代码改动触及 shared loss/routing path，R0 必须完整重跑；不得直接把旧 run 当作新代码的等价结果。

---

## 8. 工程测试

至少新增：

```text
test_v0_1_regression_unchanged.py
test_classwise_admission_fraction.py
test_admission_progress_monotonicity.py
test_admission_is_classwise_not_global.py
test_admission_weights_detached.py
test_calibrator_uses_labeled_only.py
test_calibrator_pava_monotonic.py
test_calibrator_checkpoint_resume.py
test_compatibility_rejection_cap.py
test_consolidation_weight_floor.py
test_uniform_fallback_when_calibrator_unavailable.py
test_v0_2_empty_class_safety.py
test_v0_2_empty_relation_safety.py
test_v0_2_golden_batch.py
```

必须继续通过原有全部测试。

### 8.1 Regression gate

旧的 V0.1 golden batch 必须保持：

```text
all recorded tensor/loss errors == 0
```

若 V0.2 改动破坏 V0.1 复现，停止长实验。

### 8.2 新 golden batch

为 R3 固定：

- seed；
- labeled batch；
- unlabeled batch；
- current/old checkpoint；
- calibrator state；
- progress；
- augmentation parameters。

记录：

```text
logits
relation probabilities
raw L
admission mask
raw C
calibrated C
rejection mask
consolidation weights
L_sup
L_assim
L_rel
total loss
selected pixel counts by class
rejected pixel counts by class
```

---

## 9. 训练日志要求

每 100 optimizer steps 记录：

```text
site
epoch
site_progress
loss_sup
loss_assim
loss_rel
pseudo_valid_count
assim_selected_count
assim_selected_fraction
assim_selected_fraction_by_class
relation_valid_count
compat_rejected_count
compat_rejected_fraction
compat_rejected_fraction_by_class
consolidation_weight_mean
consolidation_effective_sample_size
raw_C_mean
calibrated_C_mean
calibrator_status
calibrator_last_update_epoch
gradient_norm_assim
gradient_norm_rel
gradient_cosine
```

每个站点结束时保存：

- model checkpoint；
- current/historical anchor state；
- calibrator state；
- site matrix；
- calibration CSV；
- branch statistics；
- loss/gradient summary。

---

## 10. Fundus seed-0 研究门槛

R3 必须同时满足：

### 10.1 性能门槛

\[
\text{Final avg}\geq 0.6551,
\]

\[
\text{BWT}>-0.1185,
\]

\[
\text{Incoming}\geq0.7241,
\]

即相对 R0 的 incoming Dice 下降不得超过 0.01。

此外：

\[
\text{Previous}\geq0.6709,
\]

即相对 R0 的 previous-site Dice 下降不得超过 0.005。

### 10.2 机制门槛

- class-wise admission coverage 随 progress 从约 40% 平滑增加至约 80%；
- 任一类别最终覆盖不得长期超过 90%；
- 校准后的 class-wise compatibility correctness curve 非降；
- 每类 compatibility rejection 比例不超过 20%；
- consolidation weight 的 \(N_{\mathrm{eff}}\) 不低于 uniform relation KD 的 70%；
- hidden GT 未进入训练进程；
- R1 相比 R0 不得同时降低 Incoming 和 Final；
- R2 相比 R0 应改善 BWT 或 Previous，且 Incoming 下降不超过 0.005；
- R3 至少在 Final/BWT/Previous 中两项优于 R0，并满足上述容差。

若门槛未通过：

```text
FUNDUS_V0_2_RESEARCH_GATE_NOT_MET
```

停止，不运行 Prostate，不加入 V4/V5。

---

## 11. 条件式 Prostate A→B Pilot

只有 Fundus R3 gate 通过后，才运行：

```text
A = RUNMC
B = BMC
20% labeled
seed = 0
200 epochs/site
```

比较：

```text
Sequential-SSL
R0 unit assimilation + uniform relation KD
R3 asymmetric routing
```

仅运行 A→B，不运行完整 A→E。

Prostate pilot gate：

- R3 在 site A 的 retained Dice 高于 R0；
- R3 的 BWT 优于 R0；
- site B incoming Dice 相对 R0 下降不超过 0.01；
- 无数值异常、无 hidden-label leakage；
- calibration/rejection 行为与 Fundus 方向一致。

通过后才制定完整 Prostate 与 M&Ms 计划。

---

## 12. 输出产物

```text
reports/experiment_status/V0_1_ROUTING_DIAGNOSTIC.md
reports/experiment_status/LCRSEG_V0_2_COMPLETION.md
reports/experiment_status/LCRSEG_V0_2_GATE.json

reports/analysis/v0_1_routing/
reports/analysis/v0_2_r0/
reports/analysis/v0_2_r1/
reports/analysis/v0_2_r2/
reports/analysis/v0_2_r3/

configs/experiments/lcrseg_v0_2_r0_uniform.yaml
configs/experiments/lcrseg_v0_2_r1_learnability_admission.yaml
configs/experiments/lcrseg_v0_2_r2_compatibility_reject.yaml
configs/experiments/lcrseg_v0_2_r3_asymmetric_full.yaml
```

`LCRSEG_V0_2_COMPLETION.md` 必须明确：

1. 改动文件；
2. 原有测试和新增测试结果；
3. V0.1 regression 与 V0.2 golden batch；
4. Phase A 诊断结论；
5. R0–R3 完整结果；
6. site matrices；
7. class-wise/region-wise calibration；
8. effective sample size；
9. gradient diagnostics；
10. gate 是否通过；
11. 是否运行 Prostate A→B；
12. 所有失败、异常与未完成项。

---

## 13. 执行顺序

```text
M0 读取报告并冻结 V0.2 preregistration
M1 完成 V0.1 class/region/ESS/gradient audit
M2 实现 progressive learnability admission
M3 实现 labeled-only compatibility calibrator
M4 实现 rejection-only consolidation
M5 单元测试、V0.1 regression、R3 golden batch
M6 运行 Fundus R0
M7 运行 Fundus R1
M8 运行 Fundus R2
M9 运行 Fundus R3
M10 汇总 gate
M11 仅在 gate 通过后运行 Prostate A→B pilot
M12 输出 completion report
```

不得跳过 M1、M5 或 M10。
