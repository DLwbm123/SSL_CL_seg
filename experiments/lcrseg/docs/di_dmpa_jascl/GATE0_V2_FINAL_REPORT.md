# JASCL Gate 0 v2：半监督目标语义修复与三种子审阅报告

日期：2026-08-30。状态：**PASS，停止并等待独立审阅**。

该 PASS 仅表示本轮指定的 Gate 0 正确性门槛通过，不表示 B0 优于 C0，
也不表示 DI-DMPA 已实现或获准启动。

## 1. 提交与授权边界

- 仓库：`DLwbm123/SSL_CL_seg`。
- 审阅分支：`codex/gate0-pas-probability-mse`，不合并 main。
- 冻结 main/v1：`46e892960240543c946c570a9378d409b226384b`。
- 本次全部正式训练及前置测试的源码：`fb55e8022bc379e2515a46214c6fdf45ea818de6`。
- 官方 JASCL：`3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53`，tracked source 未修改。
- 最终交付提交另含报告及文档目录内的调度/汇总脚本；以分支 HEAD 为交付 SHA，
  不把它伪称为已运行的训练源码 SHA。远端训练 checkout 保持上述 fb55 提交。

解析后的配置 SHA-256：

```text
C0 074b9afeef0d7acbbda9a9e03c4bc479248a200ca1e4cf6e1b0eccc30116b000
B0 37876e4a5dae85a31ff8d8a211e975e745cffe4c395f89407585bb9ede682b4c
```

始终保持 `method_registered=false`、`di_dmpa_training_launched=false`。
multi-prototype、domain-indexed bank、transport、soft fusion、history gate、
multi-prototype loss、prototype inference、constant-patch regularization 全部关闭。
没有启动 Prostate、MnMS、DI-DMPA 或 full sweep。

## 2. v1 撤销与归档

旧 overall PASS 已撤销。v1 的工程/泄漏检查及已测试的六步 resume 仍是历史事实，
但其无标注一致性梯度为零，不能作为半监督 JASCL 结果。

仅可称为 inert-unlabeled / labeled-cycle control；其无标注 forward 仍消耗
classifier RNG，因此不是简单的 pure supervised，也不是 v2 的 compute/RNG-matched C0。

原状态字节保存在 `GATE0_STATUS_V1_ARCHIVED.json`，原报告及矩阵在
`gate0_results_v1_zero_u_grad/`。原 `gate0_results/`、冻结协议以及远端 v1
运行目录未被本轮覆盖。审阅说明见 `GATE0_V1_ZERO_U_GRAD_REVIEW.md`。
`V1_PRESERVATION_AUDIT.json` 记录了归档内 23 个文件与冻结提交的逐文件字节一致性。

## 3. 固定 benchmark 与模型

仅运行 Fundus，域顺序为 **REFUGE → RIM_ONE_r3 → Drishti_GS**，读取自冻结
LCRSeg 协议/配置并进行哈希验证，不按审计报告中的列举顺序推断。

- 三通道，C=3；原标签 255/128/0 映射到 background / optic-disc rim / optic cup。
- 原始中心裁剪 800×800，存储尺寸 384×384，RGB 缩放至 [0,1]。
- 映射后的 ignore label=255；不把 C 或 C−1 当 void。
- 每个种子的 labeled/unlabeled 数分别为 REFUGE 40/160、RIM_ONE_r3 16/63、Drishti_GS 10/41。
- val 仅评价及 checkpoint selection；test 仅阶段评价，不用于训练或阈值选择。
- 模型是冻结的 LCRSeg **UNet2D + 官方 3×3 ProbabilisticClassifier**，不是 DeepLab backbone。
  官方 classifier 的 functional-conv 有效无 padding 行为与随后的 logit 插值也未偷偷修复。

Fundus、Prostate、MnMS 的完整独立协议继续保存在 `DOMAIN_PROTOCOL.yaml`；本轮未运行后两者。

## 4. 修复内容

G0-R11：以 detached bool mask 表达 PAS validity。confidence 与 cosine similarity
均严格大于 0.7，先在 feature 分辨率计算，再 nearest 到 loss 分辨率；
最终 mask 为 student 与 teacher 的交集，不通过伪类别编码无效像素。

无标注目标为：

```text
L_cons = sum_{i in V} ||softmax(z_s(i)) - stopgrad(softmax(z_t(i)))||² / |V|
L_total = L_sup + lambda_u * L_cons
```

不额外除以 C；student 与最终 L_cons 不 detach；teacher/mask/prototype 不接收梯度。
空 mask 返回连接 student 图的零值。`upstream_pas_labels` 仅保留为 legacy 接口，
正式目标不调用它。

G0-R12：`stochastic_classifier` 是必须显式传入的参数。训练 student/teacher PAS
均为 true；validation、checkpoint selection、formal test 均为 false，即 posterior mean。
不根据 `self.training` 猜测 classifier 是否采样。

G0-R13：实际计算 `autograd.grad(L_cons, student)`，而非仅检查 optimizer.step 或
参数是否变化；原误导性的 TinySegNet parity 测试重命名为监督确定性 smoke test。

G0-R14：状态编译器读取真实 test/resume/gradient/leakage/evaluation 报告、JUnit、
transcript/checkpoint 哈希、六条运行的日志、元数据与矩阵；缺失或不匹配则拒绝 PASS。

保持 Adam、LR=1e-3、weight decay=4e-5、polynomial power=0.9、100 epochs/domain、
batch size=2、prototype start=25、PAS interval=5、EMA=0.99 及其更新频率、
原 labeled augmentation 和非随机增强的 unlabeled view。GAS 只使用正式 supervised phase 梯度。

## 5. 真实 batch 梯度审计

正式运行前，使用只读 v1 checkpoint 与每域第一个固定无标注 batch。
没有搜索高覆盖 batch、重试筛选、降低阈值或读取 test GT。

| 域 | joint valid pixels | coverage | L_cons | student 原始无标注梯度范数 |
|---|---:|---:|---:|---:|
| REFUGE | 93,190 | 31.60% | 0.110272 | 1.329544 |
| RIM_ONE_r3 | 87,158 | 29.55% | 0.068462 | 0.182580 |
| Drishti_GS | 57,551 | 19.51% | 0.075765 | 0.872878 |

三域均 `consistency_requires_grad=true`、teacher nonnull-gradient count=0、
prototype requires-grad=false、hidden-GT usage=none；总梯度减监督梯度的范数也均非零。
完整 checkpoint、SHA、batch ID 与 RNG seed 见 `PAS_GRADIENT_AUDIT.json`。

部分原始 case ID 带有 `test` 字样，这不是本轮 role。
实际角色由哈希冻结 manifest 的 `primary_20pct_split` 决定；审计使用的是
train_unlabeled/train_labeled，不是 final-test 标签。

## 6. 测试、resume 与评价随机性

- 远端主测试：52 passed，0 failed，0 skipped；JUnit/transcript 随报告交付。
- 本地完整回归：264 passed，4 个 remote-data-only skips。
- 四种 resume 路径：mid-supervised、mid-PAS、加载 best 前、加载 best 后/下一域初始化前。
  比较 student、EMA、optimizer、scheduler、GAS、prototype、全部 RNG、sampler、stage state、
  matrices、best metric、deterministic logits；所有比较最大绝对差为 0，容差仍为 atol=rtol=1e-6。
- 扩展 resume 使用**真实 UNet/JASCL CUDA 模型与生产状态机，但输入是哈希固定的 16×16
  synthetic HDF5 fixture**。不把它描述为完整真实 Fundus 100-epoch 中断复跑；
  真实 Fundus 梯度与六次完整训练是独立证据。
- 同一 v1 checkpoint、REFUGE val：20 次 stochastic single-draw mean Dice
  均值 0.898206、样本 SD 0.000972、范围 [0.896448,0.899744]。
- 两次 posterior-mean Dice 均为 0.894940，logit 最大绝对差为 0，classifier RNG 不变。
  MC-16 为可选诊断，未运行。正式评价方式是预先指定的 posterior mean，不按分数择优。

## 7. 正式六次运行与配对结果

C0 仅 λ=0，B0 仅 λ=0.5；其余配置完全相同。即使 C0 λ=0，也完整执行所有
unlabeled forwards、classifier RNG sampling、PAS、labeled-cycle batch 与 optimizer step。
C0 日志中的无标注梯度是未乘 λ 的 `grad(L_cons)`；它不会进入 λ=0 的总更新。

先完成 seed 0 配对及其最终门槛，再启动 seed 1/2。
每次均为 3 domains × 100 epochs、5,295 optimizer steps，其中 1,995 为 PAS/labeled-cycle steps。
六次总计 31,770 steps、11,970 PAS batches，全部 `.complete`、exit=0。

以下分数为最终阶段对三个已见域的等权平均，单位为百分数；± 为三种子样本 SD。

| 指标 | C0 | B0 | 配对 B0−C0（百分点，均值±SD） |
|---|---:|---:|---:|
| Mean Dice（含背景） | 74.28 ± 3.87 | 73.80 ± 4.95 | −0.48 ± 1.60 |
| Foreground Dice | 62.40 ± 5.72 | 61.78 ± 7.23 | −0.62 ± 2.34 |
| Mean IoU | 63.63 ± 3.83 | 62.97 ± 4.79 | −0.66 ± 1.27 |

| Seed | C0 Mean Dice | B0 Mean Dice | B0−C0（百分点） |
|---|---:|---:|---:|
| 0 | 73.2641 | 74.0424 | +0.7783 |
| 1 | 71.0255 | 68.7408 | −2.2847 |
| 2 | 78.5595 | 78.6313 | +0.0718 |

**B0 未获得三种子平均性能提升。** 本轮没有为此修改阈值、λ、epoch 或 schedule。
三种子样本量有限，不据此做方法优越性或显著性声明。

完整 lower-triangular 三项矩阵、每阶段 current/historical domain 指标在
`gate0_results_v2/{C0,B0}/seed{0,1,2}/`；每目录含 JSON、三项 CSV、运行及阶段元数据。
`V2_TRAINING_DIAGNOSTICS.json` 另含逐种子/domain 的统计与 backward transfer。
forgetting 定义为该域从首次出现到最后阶段的最大分数减最终分数，报告所有历史域，
不将新域记成历史遗忘。

## 8. PAS 覆盖、梯度与 validation-only precision

下表对每个 seed 先求训练 batch 均值，再跨三个 seed 等权平均。
precision 是阶段末、posterior-mean、joint PAS-valid validation pixels 上的 teacher
pseudo-label precision，使用阶段末当前域 prototype；不反馈训练、checkpoint 或阈值选择。
它不是 train-unlabeled hidden-GT precision，也不替代完整 test 指标。

| 配置 | 域 | 平均训练 coverage | 原始 grad(L_cons) 平均范数 | 平均 validation precision |
|---|---|---:|---:|---:|
| C0 | REFUGE | 19.01% | 0.9077 | 91.71% |
| C0 | RIM_ONE_r3 | 37.39% | 0.5814 | 95.58% |
| C0 | Drishti_GS | 40.22% | 1.2770 | 91.90% |
| B0 | REFUGE | 19.54% | 0.6436 | 95.41% |
| B0 | RIM_ONE_r3 | 44.38% | 0.4101 | 96.20% |
| B0 | Drishti_GS | 46.92% | 1.2428 | 95.00% |

六次正式运行中零 coverage batch=0、零原始无标注梯度 batch=0；
最小原始无标注梯度范数为 3.7583e-6。全部 teacher 梯度计数为 0，hidden-GT usage=none。
三个 seed 在首次 PAS 前 520 个监督 steps 的配对 loss 差异为 0；第一个 PAS batch
的 loss、coverage、raw gradient 差异也为 0，支持 compute/RNG-matched 初始化路径。

## 9. GPU 利用率与完整性

发现容器只有 16 核 CPU quota，而 seed-0 两进程各约 56 个计算线程，导致线程争抢。
用户要求提高 GPU 利用率后，先做资源等价测试，再将 seed 1/2 设为每进程 1 个
CPU 计算线程、每 GPU 两个独立进程。模型、精度、batch、超参数与训练代码不变。

资源验证额外 52/52 测试通过；四类跨线程状态比较通过，三域真实梯度审计数值差异为 0。
调整后 20 秒采样，两张 3090 平均利用率为 78.7% / 72.2%，峰值 94% / 92%。
seed-0 单次约 29.4–29.5 分钟，后四次并发单次约 8.96–9.63 分钟；
这是实际运行耗时，不是控制了 seed/阶段/竞争条件的严格加速比实验。
详情见 `RESOURCE_UTILIZATION_REPORT.md`。

六次运行均无训练 exception、NaN、非有限梯度；24 个 best/final checkpoint 的
tensor finiteness、完整 classifier、student/EMA keys、source/config 与 SHA 检查通过。
最终运行全部退出，两卡均无训练进程。未为占满 GPU 而启动任何未授权实验。

保留的警告是原 scheduler 调用顺序、epoch 参数弃用及 PyTorch CUDA NLL 的
deterministic warn-only；不宣称跨硬件或任意轨迹普遍 bitwise deterministic。
初次开发测试失败、GAS 零初始化采样退化、传输 early EOF、诊断命令兼容问题均登记于
`V2_DEVELOPMENT_FAILURES_AND_WARNINGS.md`，没有隐去或覆盖失败证据。

## 10. 审阅入口与停止点

- 最终机器状态：`GATE0_STATUS.json`（errors=[]，next_action=STOP_FOR_INDEPENDENT_REVIEW）。
- 修复逐项登记：`GATE0_REPAIR_LEDGER.md`，G0-R11—R14。
- 冻结协议：`DOMAIN_PROTOCOL.yaml`。
- 真实 batch / 随机性：`PAS_GRADIENT_AUDIT.json`、`EVAL_STOCHASTICITY_AUDIT.json`。
- 测试 / resume / 泄漏：三个相应 JSON 报告、`pytest.xml`、`pytest_output.txt`。
- 阶段门槛留档：`SEED0_PAIR_GATE.json`，不是提前写成整体 PASS 的状态。
- 六次矩阵与元数据：`gate0_results_v2/`。
- 配对结论：`C0_VS_B0_PAIRED_COMPARISON.json`、`V2_TRAINING_DIAGNOSTICS.json`。
- 资源等价证据：`resource_audit_threads1/`。
- 复现及调度：`EXACT_COMMANDS.md`、`launch_remaining_seeds.sh`。
- 配置：`../../configs/gate0_repaired_v2/fundus_lambda_u0.yaml` 与 `fundus_pas_probmse.yaml`。

原始 train.jsonl、完整 checkpoints 位于各远端 `run_dir`，不随源码报告上传；
其路径、哈希、统计及运行元数据已记录，未删除。

本轮到此停止。`method_off_switch_parity_status=NOT_APPLICABLE_METHOD_NOT_IMPLEMENTED`，
不声称未实现的方法已通过 parity；不自动进入 DI-DMPA，不合并 main。
