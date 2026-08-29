# LCR-Seg 实现合同 V0.1

**文件名：** `IMPLEMENTATION_CONTRACT_V0_1.md`  
**依赖规范：** `METHOD_SPEC_V0_1.md`  
**版本：** 0.1  
**状态：** Codex 实现前强制合同  
**最后更新：** 2026-08-18

---

## 0. 合同语言

- **MUST / 必须**：违反即视为实现错误。
- **MUST NOT / 禁止**：出现即停止实验。
- **SHOULD / 应当**：除非有记录充分的工程原因，不得偏离。
- **MAY / 可以**：不改变方法定义的可选实现。

任何方法公式、张量合同、梯度规则或状态生命周期的更改，必须同时更新：

1. `METHOD_SPEC_V0_1.md`；
2. 本文件；
3. `METHOD_ACCEPTANCE_TESTS_V0_1.md`；
4. 对应单元测试；
5. `STATUS.md` 中的版本说明。

---

## 1. 推荐代码布局

```text
experiments/lcrseg/
├── AGENTS.md
├── METHOD_SPEC_V0_1.md
├── IMPLEMENTATION_CONTRACT_V0_1.md
├── METHOD_ACCEPTANCE_TESTS_V0_1.md
├── configs/
│   ├── data/
│   ├── model/
│   ├── method/
│   └── experiment/
├── lcrseg/
│   ├── contracts.py
│   ├── data/
│   │   ├── h5_dataset.py
│   │   ├── continual_sampler.py
│   │   ├── batch_types.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── unet.py
│   │   ├── projection_head.py
│   │   └── outputs.py
│   ├── methods/
│   │   ├── base.py
│   │   ├── supervised.py
│   │   ├── sequential_ssl.py
│   │   ├── uniform_kd.py
│   │   ├── lcrseg_v0_1.py
│   │   └── components/
│   │       ├── anchor_bank.py
│   │       ├── relation_field.py
│   │       ├── pseudo_label.py
│   │       ├── learnability.py
│   │       ├── compatibility.py
│   │       └── routing.py
│   ├── engine/
│   │   ├── trainer.py
│   │   ├── continual_runner.py
│   │   ├── evaluator.py
│   │   └── checkpoint.py
│   ├── evaluation/
│   └── analysis/
└── tests/
```

### 1.1 训练引擎复用

所有方法 MUST 复用同一个：

- model factory；
- dataset 和 transforms；
- optimizer/scheduler builder；
- continual runner；
- evaluator；
- checkpoint manager。

禁止为 LCR-Seg 复制一套新的训练循环。

---

## 2. Python 类型合同

### 2.1 模型输出

```python
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class SegModelOutput:
    logits: torch.Tensor
    # [B, C, H, W], unnormalized

    relation_features: torch.Tensor
    # [B, D, h, w], L2-normalized on D
```

要求：

- `logits.dtype` 为训练 mixed precision 所允许的浮点类型；
- `relation_features` 在最后一个 feature channel 维度做 L2 normalize；
- 不返回已 softmax 的 segmentation output；
- baseline 仍使用同一 `SegModelOutput`，但可忽略 relation feature。

### 2.2 训练 batch

```python
@dataclass
class LabeledBatch:
    image: torch.Tensor         # [B_l, Cin, H, W]
    label: torch.Tensor         # [B_l, H, W], long
    valid_mask: torch.Tensor    # [B_l, 1, H, W], bool
    case_id: list[str]
    patient_id: list[str]
    site: list[str]

@dataclass
class UnlabeledBatch:
    weak_image: torch.Tensor        # [B_u, Cin, H, W]
    strong_image: torch.Tensor      # [B_u, Cin, H, W]
    strong_valid_mask: torch.Tensor # [B_u, 1, H, W], bool
    case_id: list[str]
    patient_id: list[str]
    site: list[str]
```

训练 `UnlabeledBatch` MUST NOT 包含：

- `label`；
- `hidden_label`；
- diagnostics path；
-任何可从 batch 访问 GT 的字段。

### 2.3 方法 step 输出

```python
@dataclass
class MethodStepOutput:
    total_loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    scalars: dict[str, float]
    maps: dict[str, torch.Tensor] | None
```

`losses` 至少包含：

```text
loss_sup
loss_seg_ce
loss_seg_dice
loss_anchor_sup
loss_assim
loss_relation
```

即使某损失在当前阶段不适用，也必须返回 device-compatible scalar zero。

---

## 3. 模型结构合同

### 3.1 输入输出

| 张量 | 形状 |
|---|---|
| input image | `[B, Cin, H, W]` |
| segmentation logits | `[B, C, H, W]` |
| decoder relation feature input | `[B, F, H/4, W/4]` |
| projected relation feature | `[B, D, H/4, W/4]` |
| anchor bank | `[C, 1, D]` |
| relation logits | `[B, C, H/4, W/4]` |
| relation probability | `[B, C, H/4, W/4]` |
| learnability map | `[B, 1, H/4, W/4]` |
| compatibility map | `[B, 1, H/4, W/4]` |
| pseudo-label relation grid | `[B, H/4, W/4]` |

若输入不能被 4 整除，DataLoader/crop MUST 在进入模型前处理，模型内部不得静默裁剪标签。

### 3.2 Projection head

默认实现：

```text
Conv2d(F, D, kernel_size=1, bias=False)
GroupNorm(num_groups=min(8, D), num_channels=D)
ReLU(inplace=False)
Conv2d(D, D, kernel_size=1, bias=True)
L2 normalization along channel
```

要求：

- `D = 128` 默认；
- 不使用 BatchNorm，避免小 batch 和跨站点统计漂移；
- projection head 属于模型 checkpoint；
- old model 包含冻结 projection head；
- 不将 anchors 注册为 projection head 参数。

### 3.3 插值规则

- label downsample：nearest；
- pseudo-label upsample：nearest；
- continuous maps \(L_i,C_i\) upsample：bilinear，`align_corners=False`；
- relation feature resize：除非规范明确要求，禁止额外 resize；
- segmentation logits 不得通过 nearest resize。

---

## 4. 方法接口

```python
from abc import ABC, abstractmethod
from pathlib import Path

class ContinualSegMethod(ABC):
    @abstractmethod
    def begin_site(
        self,
        site_id: str,
        previous_checkpoint: Path | None,
        total_steps: int,
    ) -> None:
        ...

    @abstractmethod
    def training_step(
        self,
        labeled_batch: LabeledBatch,
        unlabeled_batch: UnlabeledBatch,
        global_step: int,
        site_step: int,
    ) -> MethodStepOutput:
        ...

    @abstractmethod
    def after_optimizer_step(self) -> None:
        ...

    @abstractmethod
    def end_site(self, site_id: str) -> dict:
        ...

    @abstractmethod
    def method_state_dict(self) -> dict:
        ...

    @abstractmethod
    def load_method_state_dict(self, state: dict) -> None:
        ...
```

### 4.1 `begin_site`

MUST：

- 初始化 current model；
- 加载/创建 old model；
- 设置 old model `eval()`；
- `requires_grad_(False)`；
- 加载 old/current anchor bank；
- 重置 current-site accumulators；
- 固定 total steps；
- 记录 site ID 和 method version。

### 4.2 `training_step`

MUST：

1. forward labeled current model；
2. weak current forward；
3. strong current forward；
4. incremental site 时 old weak forward under `torch.no_grad()`；
5. 计算监督 relation field；
6. 计算候选伪标签；
7. 计算 detached \(L_i\)；
8. 计算 detached \(C_i\)；
9. 计算三类损失；
10. 返回 loss 与日志；
11. 不在该函数中调用 optimizer step。

### 4.3 `after_optimizer_step`

MUST 在 optimizer step 之后：

- no-grad 更新 current anchor bank；
- 更新 counts/statistics；
- 不更新 old anchor bank；
- 不读取 hidden GT。

### 4.4 `end_site`

MUST：

- 验证 anchors；
- 保存 final checkpoint；
- 输出 site summary；
- 将 current bank 作为下一阶段 historical bank；
- 不删除 old checkpoint，直到新的 checkpoint 完整验证。

---

## 5. AnchorBank 合同

### 5.1 状态

```python
@dataclass
class AnchorBankState:
    anchors: torch.Tensor
    # [C, K, D], V0.1 K=1

    valid: torch.Tensor
    # [C, K], bool

    counts_total: torch.Tensor
    # [C, K], float64/int64

    counts_labeled: torch.Tensor
    counts_unlabeled: torch.Tensor

    last_update_step: torch.Tensor
    # [C, K], int64
```

### 5.2 注册方式

MUST 使用 `register_buffer` 或独立可序列化状态。

MUST NOT：

- `nn.Parameter(anchors)`；
- 加入 optimizer parameter groups；
- 对 anchor update 反向传播；
- 在 current/old bank 之间共享同一 tensor storage。

### 5.3 更新 API

```python
@torch.no_grad()
def update(
    features: torch.Tensor,  # [B,D,h,w]
    labels: torch.Tensor,    # [B,h,w]
    weights: torch.Tensor,   # [B,1,h,w]
    source: str,             # labeled | unlabeled
    step: int,
) -> dict[str, int]:
    ...
```

要求：

- per class capped sampling；
- 忽略 invalid label；
- 忽略非有限 feature；
- 若 support < `min_support_pixels`，不更新；
- batch 无某类时保持原值；
- 更新后 L2 norm 接近 1；
- 背景执行 boundary exclusion；
- 返回每类更新像素数。

### 5.4 Bootstrap

第一站点：

- bootstrap 前 `anchor_valid=False`；
- bootstrap 期间不执行 anchor-recoverable pseudo-labeling；
- classifier confidence SSL 可以运行；
-满足最小 support 后才设 valid；
-若某前景类别在 labeled subset 中完全不存在，标记 BLOCKER。

---

## 6. RelationField 合同

```python
@dataclass
class RelationOutput:
    logits: torch.Tensor       # [B,C,h,w]
    probabilities: torch.Tensor
    predicted_class: torch.Tensor  # [B,h,w]
    top1: torch.Tensor         # [B,1,h,w]
    top2: torch.Tensor         # [B,1,h,w]
    margin: torch.Tensor       # [B,1,h,w]
    valid_class_mask: torch.Tensor # [C]
```

计算要求：

- feature 与 anchor 都 L2 normalize；
- cosine similarity；
- invalid anchor class 的 logits 设为有限大负数；
- 至少一个 valid class，否则抛出显式异常；
- probabilities 每像素和为 1；
- 不允许 NaN；
- temperature > 0。

---

## 7. 伪标签合同

```python
@dataclass
class PseudoLabelOutput:
    labels: torch.Tensor       # [B,h,w], long
    valid: torch.Tensor        # [B,1,h,w], bool
    source: torch.Tensor       # 0 deferred, 1 classifier, 2 anchor
    source_weight: torch.Tensor# [B,1,h,w]
    spatial_weight: torch.Tensor
```

### 7.1 计算顺序

1. current weak segmentation probabilities；
2. current weak relation probabilities；
3. classifier-easy mask；
4. preliminary anchor-recoverable mask；
5. preliminary candidate labels；
6. spatial agreement；
7. final anchor-recoverable mask；
8. deferred mask。

禁止先根据 hidden GT 修正候选标签。

### 7.2 无效标签

无效位置内部使用 `ignore_index=-100`，不得把 deferred 像素改成 background。

---

## 8. Learnability 合同

```python
@dataclass
class LearnabilityOutput:
    score: torch.Tensor            # [B,1,h,w]
    robust_progress_index: torch.Tensor
    percentile_rank: torch.Tensor
    progress_weight: torch.Tensor
    relation_weight: torch.Tensor
    spatial_weight: torch.Tensor
    source_weight: torch.Tensor
```

MUST：

- 所有输入 weak prediction detach；
- 最终 `score.detach()`；
- score clamp 到 `[0,1]`；
- class-wise rank deterministic；
- 小类 fallback 到 global rank；
- 无效 pseudo-label score=0；
- strong view 不参与 score 的反向传播。

MUST NOT：

- 将 \(L_i\) 作为 trainable head；
- 对 \(L_i\) 直接添加监督；
- 用 diagnostics GT 计算训练时 rank 或阈值。

---

## 9. Compatibility 合同

```python
@dataclass
class CompatibilityOutput:
    score: torch.Tensor        # [B,1,h,w]
    js_divergence: torch.Tensor
    old_margin_weight: torch.Tensor
    agreement: torch.Tensor
    spatial_weight: torch.Tensor
```

MUST：

- old model forward 在 `torch.no_grad()`；
- old anchor frozen；
- current weak relation 用 detach 版本计算 C；
- final score detach；
- 第一站点返回全零；
- current/old relation class count 必须一致；
- JS 使用数值稳定实现；
- 主类不一致时 `agreement=0`。

---

## 10. Loss 合同

### 10.1 监督损失

```python
loss_sup = loss_seg_ce + lambda_dice * loss_seg_dice \
           + lambda_anchor_sup * loss_anchor_sup
```

- segmentation CE 使用 full-resolution visible GT；
- Dice 只用 visible GT；
- relation CE 使用 nearest downsample GT；
- invalid/crop padding 使用 valid mask 排除。

### 10.2 Assimilation loss

```python
loss_assim = weighted_mean(
    pixel_ce(strong_logits, pseudo_labels_up),
    learnability_up * strong_valid_mask
)
```

要求：

- pseudo label detach；
- \(L_i\) detach；
- denominator 为空时返回 `strong_logits.sum() * 0.0`；
-不使用 unlabeled Dice；
-不在 deferred 区域计算 CE。

### 10.3 Relation loss

```python
loss_relation = T_d**2 * weighted_mean(
    KL(old_relation_weak.detach(), current_relation_strong),
    compatibility * strong_valid_mask_down
)
```

要求：

- KL 方向为 old teacher 到 current student；
- old probability detach；
- C detach；
-第一站点为 differentiable zero；
- denominator 为空时无 NaN；
- relation strong view 与 weak teacher 空间对齐。

### 10.4 Ramp

```python
lambda_assim_eff = lambda_assim * linear_ramp(site_step)
lambda_relation_eff = lambda_relation * linear_ramp(site_step)
```

- assimilation ramp 在 bootstrap 完成后开始；
- relation ramp 每个增量站点开始时重置；
- ramp 参数写入 checkpoint/config。

---

## 11. 梯度合同

### 11.1 Old model

MUST：

```python
old_model.eval()
old_model.requires_grad_(False)

with torch.no_grad():
    old_output = old_model(weak_image)
```

验收要求：

```python
assert all(p.grad is None for p in old_model.parameters())
```

### 11.2 Weak pseudo-label path

MUST：

```python
weak_prob = current_weak.logits.detach().softmax(dim=1)
weak_relation = relation_output.probabilities.detach()
```

### 11.3 Reliability weight

MUST：

```python
learnability_weight = learnability.score.detach()
compatibility_weight = compatibility.score.detach()
```

### 11.4 Anchor update

MUST 在 `torch.no_grad()` 中执行，且在 optimizer step 后更新。

### 11.5 禁止的梯度路径

禁止：

- gradient 进入 old model；
- gradient 进入 old/current anchor buffers；
- gradient 从 loss 反向进入 pseudo-label selection；
- gradient 从 loss 反向进入 \(L_i\) 或 \(C_i\)；
- current 与 old model parameter storage alias；
- current 与 old anchor storage alias。

---

## 12. 增强与空间对齐合同

### 12.1 Transform 输出

Transform MUST 返回：

```python
{
    "weak_image": ...,
    "strong_image": ...,
    "strong_valid_mask": ...,
    "geometry_record": ...
}
```

### 12.2 V0.1 允许的强增强

- MRI：intensity/gamma、noise、blur、bias-field-like perturbation、cutout；
- fundus：brightness/contrast/saturation/gamma、noise、blur、cutout；
- 共享 flip/rotation/scale/translation 作为 base geometry。

### 12.3 V0.1 禁止

- weak 与 strong 各自独立几何变换；
- 未记录 elastic deformation；
- 不能同步作用于 mask/map 的几何操作；
- cutout 后仍对被遮挡区域计算 relation KD。

---

## 13. Continual 生命周期合同

### 13.1 第一站点

```text
old_model = None
old_anchor_bank = None
compatibility = 0
loss_relation = 0
```

bootstrap 后运行 learnability SSL。

### 13.2 增量站点开始

MUST：

1. load previous final checkpoint；
2. deep-copy current model；
3. deep-copy old model；
4. freeze old model；
5. deep-copy old anchors 到 current anchors；
6. verify no storage alias；
7. reset current optimizer/scheduler per experiment config；
8. reset current-site accumulators；
9. keep global site matrix history。

### 13.3 站点结束

MUST：

- evaluate all seen sites and configured unseen site；
- save final checkpoint；
- validate checkpoint reload；
- save current anchors as next historical anchors；
- save site matrix row；
- save per-case metrics；
- update `STATUS.md`。

---

## 14. Checkpoint 合同

Checkpoint MUST 包含：

```text
schema_version
method_name
method_version
git_commit
config_resolved
site_id
site_index
epoch
site_step
global_step

current_model_state
optimizer_state
scheduler_state
scaler_state

current_anchor_state
historical_anchor_state
bootstrap_state
method_statistics

rng_python
rng_numpy
rng_torch_cpu
rng_torch_cuda

data_split_hash
preprocess_version
manifest_hash
```

若启用 old model，可由 `previous_checkpoint` 重建，但恢复中必须验证其哈希。为简化断点恢复，MAY 直接存储 old model state。

### 14.1 恢复一致性

同一 checkpoint 恢复后，对固定 golden batch：

- logits；
- relation probabilities；
- \(L_i\)；
- \(C_i\)；
-各 loss；
- anchor state；

必须在规定容差内一致。

---

## 15. Config 合同

建议：

```yaml
experiment:
  seed: 0
  deterministic: true
  final_checkpoint_only: true

data:
  data_root: ...
  manifest: ...
  split_file: ...
  preprocess_version: v1
  label_fraction: 0.20
  site_order: [...]

model:
  name: unet2d
  base_channels: 16
  num_classes: ...
  relation_dim: 128

method:
  name: lcrseg_v0_1
  version: 0.1
  # 其余字段与 METHOD_SPEC 默认超参数一致

training:
  optimizer: adam
  lr: 5.0e-4
  weight_decay: 1.0e-5
  epochs_per_site: 200
  scheduler: cosine
  labeled_unlabeled_ratio: [1, 2]
  amp: true
```

所有 resolved config 必须保存到结果目录。命令行 override 必须写入最终 config，不得只存在 shell history。

---

## 16. 日志合同

每个 step 记录：

```text
site_id
epoch
site_step
global_step
lr

loss_total
loss_sup
loss_seg_ce
loss_seg_dice
loss_anchor_sup
loss_assim
loss_relation

pseudo_valid_ratio
pseudo_classifier_ratio
pseudo_anchor_ratio
pseudo_deferred_ratio

learnability_mean
learnability_p10/p50/p90
compatibility_mean
compatibility_p10/p50/p90
relation_js_mean

anchor_updates_per_class
anchor_drift_per_class
```

低频记录：

```text
gradient_cosine_assim_relation
quadrant_ratios
relation_accuracy_labeled
seg_relation_agreement
```

日志不得写入患者原始图像或可识别信息。

---

## 17. 数值稳定性合同

MUST：

- 所有分母加 `eps`；
- JS/KL 先 clamp probability；
- mixed precision 中 relation/JS 可提升为 float32；
- empty valid set 返回 differentiable zero；
- anchor norm 小于 eps 时保留旧 anchor并记录 warning；
- 每 step 检测 total loss 非有限；
-出现 NaN/Inf 立即停止并保存 failure bundle。

Failure bundle 至少包含：

```text
resolved config
site/epoch/step
tensor shapes
tensor min/max/mean
valid pixel counts
anchor validity/counts
stack trace
last checkpoint
```

禁止静默 `nan_to_num` 后继续长训练。

---

## 18. 数据泄漏合同

训练 package MUST NOT import：

```text
manifests/diagnostics/
hidden_labels/
diagnostic_evaluator.py
```

hidden GT 只允许由独立 analysis CLI 读取。

训练时若 unlabeled batch 出现 `label` key，必须抛出异常，而不是忽略。

---

## 19. 版本化与 Git 合同

建议 milestone：

```text
M0 contracts + baseline interface
M1 relation field + anchor bank
M2 pseudo-label + learnability
M3 compatibility
M4 continuous routing
M5 checkpoint + golden regression
```

每个 milestone：

1. 独立 commit；
2. 运行相应测试；
3. 更新 `STATUS.md`；
4. 不混入无关格式化；
5. 输出 diff 摘要；
6. 未通过测试不得进入下一 milestone。

---

## 20. AGENTS.md 最低内容

```markdown
# LCR-Seg Engineering Rules

- Treat METHOD_SPEC_V0_1.md as the method source of truth.
- Do not change equations, tensor contracts, detach rules, or lifecycle semantics without updating all three specification files.
- Do not implement K>1 or RIC in V0.1.
- Never give gradients to the old model or anchor buffers.
- Detach pseudo-labels, learnability, and compatibility weights.
- Never expose hidden diagnostic labels to training loaders.
- Reuse one training engine for all baselines and LCR-Seg.
- Run unit, integration, golden-batch, checkpoint-resume, and two-case overfit tests before long experiments.
- Stop and report a BLOCKER rather than inventing data semantics or silently changing protocol.
```

---

## 21. 实现完成定义

代码只有同时满足以下条件才可声称“LCR-Seg V0.1 已实现”：

- V0–V3 全部代码存在；
- 三个规范文件一致；
-所有 hard engineering tests 通过；
- checkpoint 完整恢复；
- first-site 与 incremental-site smoke test 通过；
- old model 无梯度；
- reliability weights detach；
- anchor 生命周期正确；
- hidden-label leakage 测试通过；
- fixed golden batch 回归通过；
-可生成完整 site matrix 与 \(L_i,C_i\) 分析输出；
- 尚未运行的正式实验不得被描述为已完成。
