# SSLCL 五框架统一开发与筛选包 V1

日期：2026-09-13。状态：**SPECIFICATION_AND_REFERENCE_KERNELS / CODE_REVIEW_FIRST**。

## 你现在应做什么

把整个目录放到 Codex 可读取的工作区，给它 `prompts/01_CODE_IMPLEMENTATION_AND_PUSH.md`。这轮只实现、做合成测试、生成实验计划并推送到独立 GitHub 分支，然后停止。**不读真实图像/标签，不加载真实 checkpoint tensor，不跑真实 smoke，不启动实验。**

Codex 推送后，把仓库、分支、commit SHA、PR（如已创建）和 `review/REVIEW_INDEX.md` 路径带回当前对话。审阅代码通过后，才使用 `prompts/02_RUN_AFTER_EXTERNAL_APPROVAL.md`。

本包不会自行推送仓库，也不会自行创建服务器任务。这里没有假定已经存在一个经核实的 KI 父实现。

## 目标

一次统一开发五个候选，每个框架都有合理调参机会；所有五个都进入锁定配置的多优化种子比较，然后按结果选择最多两个深入研究候选。排名第一不自动等于有效的新方法。若结果不支持论文主张，仍输出完整排名与不足，不伪造“成功主方法”。

### 优先级（是实现顺序，不是排除其他框架）

1. **F1：判别子空间内区域软监督**——GOLD-inspired 前向残差 + PAS + KL/JML。
2. **F2：结构引导可塑子空间**——当前 L 的 CWMI 结构梯度选择/初始化合法低秩方向。
3. **F3：不确定性谱更新分配**——UnCL-inspired 连续可靠性与子空间坐标的自定义反向过滤。
4. **F4：解剖与子空间可达目标修正**——D-Convexity 结构项 + 有限步教师目标修正。
5. **F5：类别条件子空间分布对齐**——SWSEG-inspired 经验 sliced-Wasserstein，PAS 给类别对应。

不把五个框架叠加为一个模型。每个框架独立沿自己的域序列更新。

## 文件导览

| 文件 | 用途 |
|---|---|
| `docs/01_IMPLEMENTATION_SPEC.md` | 公共训练路径、模型状态、五种数学定义、梯度权限、代码接口 |
| `docs/02_EXPERIMENT_PLAN.md` | 数据序列、40个框架配置、基线调参、预算、筛选规则 |
| `docs/03_TEST_AND_REVIEW_CHECKLIST.md` | 合成、CUDA、恢复、数据隔离、逐框架验收与审阅包 |
| `docs/04_DESIGN_DECISIONS_AND_SOURCES.md` | 与上一轮草图的明确修订、文献借鉴边界、证据来源 |
| `docs/05_OPERATOR_HANDOFF.md` | 用户需要提供什么；何时允许实验 |
| `prompts/01_CODE_IMPLEMENTATION_AND_PUSH.md` | **现在给 Codex 的 prompt** |
| `prompts/02_RUN_AFTER_EXTERNAL_APPROVAL.md` | **审阅批准后才给 Codex 的 prompt** |
| `prompts/03_REVIEW_REQUEST.md` | 把 Codex 的交付带回本对话的审阅模板 |
| `configs/operator_inputs.template.json` | 父实现与服务器数据位置的私有配置模板 |
| `configs/study_plan.json` | 机器可读的搜索空间、顺序和阶段 |
| `configs/review_approval.template.json` | 默认拒绝运行的外部审批模板 |
| `tools/make_plan.py` | 不读数据、不能训练的配置/任务占位清单生成器 |
| `reference_impl/` | 独立 PyTorch 参考算子与合成测试，**不是完整五框架训练器** |
| `manifests/` | 已展开的配置、任务占位、示例预算和本地测试记录 |
| `sources/SSL_CL_seg_loss_review_2023-2026.md` | 用户提供的综述原文件 |

## 本包已经做了什么／还没有做什么

已提供：五框架的工程规格，公平的共同基线与选择方案，原生可运行的 JML/KL、坐标投影、跨阶段线性参数合并、梯度过滤、结构梯度方向、SWD、目标修正接口、当前 PAS 参考算子；提供默认锁定的执行审批校验和元数据计划生成脚本。

**尚未提供/尚未验证：**真实 KI 父绑定、真实训练器接入、CWMI 与 D-Convexity 的正式后端、真实 CUDA/混合精度正确性、数据加载、分布式训练、断点恢复、实际单学生部署。Codex 必须完成这些工作再提审。参考测试通过不能冒充五框架已经实现完毕。

运行参考测试：

```bash
cd reference_impl
PYTHONPATH=. python -m pytest -q
```

生成计划（仅元数据）：

```bash
python tools/make_plan.py --example-stage-updates 3200 2100
```

## 默认实验规模与预算边界

C 阶段：4个父配置、4类基线各8个配置、5个框架各8个配置，各跑1个开发优化种子×2个顺序，共 **152 条后续域轨迹**。
D 阶段：5个框架各自最优配置 + 5种基线各自最优配置，3个新优化种子×2个顺序，共 **60 条轨迹**。
每条含2个后续域：C+D共 **212 条轨迹／424 个后续域训练阶段**，不含共享首域训练。

若经父配置确认两个后续域预算恰为3200与2100步，后续域科学更新示例为 **1,123,600**。实际预算必须由父契约与数据调度重新展开，另计至多4份合法首域训练、资格/真实smoke、F2梯度探针、F4读出内循环和F5额外前向。**本示例不是启动许可，也不是墙钟估算。**

E 阶段最多两个入选框架的贡献消融只生成计划；D结束后根据结果另外审阅，不在首轮自动执行。独立测试集、外部数据集与官方CL方法全面对比也不在本轮自动执行范围。

不按单种子负差、类别负差或区间跨零自动杀掉框架；只有实质工程有效性错误阻止相关任务。
