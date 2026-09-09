# KI 父版本溯源与接入解阻 Prompt V0.2

请执行 `KI_PARENT_RECOVERY_V0_2`。本任务是恢复原方法的工程身份与接入关系，不是重新设计 KI，也不是换用一个能运行的替代方法。

## 1. 依据、授权与不可改变事项

项目：`DLwbm123/SSL_CL_seg`。
已完成 G0 审计提交：`75d89bc52bb775fa08f985f9ea03227e0f25a1e6`。
审计目录：`experiments/lcrseg/docs/ki_ams_v0_1/`。
Anchored Mix 源码：`800cabe5e5dd69612738aaba45ad35121d63c0f2`。
Anchored Mix 报告：`1321dd84f67f2de9bd5ceaf513703bda250cc850`。
旧附件：`SSL_CL_KI_AMS_Next_Stage_Plan_v0_1.md`；仓库内有对应 `PROTOCOL.md`。

先阅读上述审计中的 FINAL_REPORT.md、COMPATIBILITY_REPORT.md、KI_PARENT_FREEZE.json、DATA_ACCESS.md 和旧协议。复用其已完成的搜索，不原样重复扫描 39 个分支端点。

GPU 4/5/6/7 共享使用授权沿用，不再次索要相同授权；发现阶段不占用 GPU，不杀死其他进程。未来实际准入时重新检查显存，不能使用旧快照作为实时保证。

旧 `BLOCKED_KI_PARENT_UNVERIFIED` 是当时真实审计状态，保持该提交及其文件不变。旧 106000 次正式更新和 `VALUE_REPRODUCED` 不改、不重计。不读 test/hidden GT，不改变患者划分；`formal_03` 等既有禁读范围继续排除。本任务不授权 G2/G3、额外方法或超参搜索。

在独立分支和输出目录执行，例如 `codex/ki-parent-recovery-v0-2`、`experiments/lcrseg/docs/ki_parent_recovery_v0_2/`。先记录当前工作区与实际分支状态，不 checkout 覆盖未提交工作，不 reset/clean/强推。

## 2. 显式修订 G0：分离“原方法身份”与“历史成绩证明”

先生成 PROTOCOL_AMENDMENT.md，记录本节对旧附件 §2.1 的修订。不得静默把旧 gate 改成通过。

以下仍是硬前提：
1. 找到真实存在的原方法源码实体，绑定精确 Git commit，或冻结可核验的源码快照及其内容哈希。
2. 找到对应有效配置与入口，并有原运行元数据、项目交接或其他可追溯记录证明这就是用户所指的 KI；不能仅凭含有 LoRA/A-only 字样认领。
3. 静态解析实际 A/B 维度、乘法顺序、rank、正交操作、冻结/训练参数、阶段转移、训练/推理域信息、参数记忆、优化器与每域更新预算。
4. 满足原方法及当前实验的数据和部署边界；有合法 D0 入场状态，或能按已核实的原方法配方新训练 D0。

“找不到原始 SOTA 汇总表/历史逐项分数”不再单独构成禁止新配对实验的条件。若上述原方法身份和有效配置均已核实，只缺历史成绩绑定：允许完成工程资格；新 G1 必须包含重新训练的匹配 KI_REF，禁止使用未绑定的旧数值或旧权重充当对照。

此时明确记录：`implementation_identity_verified=true`、`historical_performance_verified=false`。报告只评价新配对实验，不能声称复现历史 SOTA。

如果缺的是源码本体、原方法身份或决定训练行为的配置，而不只是历史成绩，则仍不得启动 G1。不得从 O-LoRA 论文、通用 continual runner 或 MIX runner 猜测并新造父方法。

## 3. 搜索采用“运行证据反查源码”，不是只找方法名称

先建立 SEARCH_LEDGER.csv，记录已查范围、引用的既有审计、此次新增范围、命令/查询、结果、访问限制及失败。已查目录无新增线索时不重扫。

### 3.1 优先检查项目运行元数据

从已确认属于本项目的根开始：
- `/home/jiangsuiyang/SSL_CL`
- `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg`

这些是已知项目定位根，不是已验证 KI 路径。先核查实际存在性、属主和既有访问权限。不要扫描整个 `/`、其他用户目录或无关项目。

查找原任务的启动脚本、日志头部、任务 ledger、run manifest、source/input lineage、源码归档清单、配置快照、Hydra 配置、slurm 脚本及聚合结果说明。提取 cwd、repo、commit、entrypoint、config、run_id、checkpoint/结果目录引用等字段，沿显式引用反查。

只读取必要的项目文本元数据，不读取图像、HDF5、标签、逐患者预测或 checkpoint 张量；不批量 torch.load。凭据文件、SSH 私钥和无关 shell 历史不读；日志中的 token/患者标识不进入公共输出。

别名搜索可覆盖 key_isolation/key-isolation、A-only/orth_A、history-free、LoRA/lora_A/lora_B、low-rank、adapter、orthogonal、protected basis 及中文对应词。命中只用于候选发现；按真实代码语义验明身份。后来的计划或审计中出现 KI，不是旧 KI 曾经实现的证据。

### 3.2 补足历史源码搜索，不把标题搜索当作代码历史搜索

在真实 Git worktree 内检查已授权 refs/tags、worktree、本地可用的 stash/reflog 线索；沿相关路径和候选差异检查历史代码内容，而不只 grep commit subject。先去重文件/blob，避免对所有版本盲目全量导出。

检查项目内允许的未跟踪源码、合法源码快照和归档目录索引；仅按候选证据提取少量源码/配置文本到新审计目录，不覆盖原文件，不展开数据或权重归档。无 .git 的 source-only 目录不自动判废；可以建立内容寻址的冻结清单，但不得把新冻结的 commit 冒称当时运行的历史 commit。

`formal_03` 和其他既有禁读路径继续排除；若线索指向那里，只记录定位线索和限制，不自行解禁。

### 3.3 跨仓库只扩展到有证据相关的候选

沿项目 remote、脚本、配置或交接记录里实际出现的其他仓库/目录检查。必要时通过已连接的 GitHub 工具检索用户可访问的相关候选，但不因仓库名含 CL 就当作父版本。

当前会话或项目交接里若有原 KI 的链接，优先复用；不能假设自己能读取其他不可见的 ChatGPT 对话。不可访问的路径记录 `ACCESS_UNAVAILABLE`，而不是“确认不存在”。

有界停止：所有明确候选线索检查完且一轮新增检索没有产生新的源码/配置/运行记录引用，即结束恢复。报告覆盖与未覆盖的范围，不无限递归，也不重复相同搜索。

## 4. 建立候选证据链并检查真正的兼容性

每个候选记录 source、effective_config、entrypoint、run_metadata、historical_results、data/task、identity_evidence、missing、accept/reject reason。

证据链应当是：
`源码身份 → 有效配置/入口 → 原运行记录 → 可选的历史聚合成绩`。

可接受旧运行元数据和源码内容哈希证明源代码身份；文件夹名称、修改时间接近或一个相似分数都不能单独完成绑定。找不到 commit 就如实填 null，记录源码快照哈希和证据等级，不伪造 SHA。

特别核查：
- A-only 是只对哪个实际子空间施加约束，不等于 B 必须被冻结。
- A/B 变量名称不同不等于不同方法；必须核对维度、乘法和约束算子。
- 正交是软惩罚还是硬投影，不对软惩罚要求不存在的精确零干扰。
- 原方法是否真的使用参数记忆、是否依赖旧样本/原型/历史教师、状态是否随任务增长。
- 找到的是当前医学域增量分割父方法，还是分类/另一骨干/另一任务的研究版本。后者不是可直接运行的父版本，不能直接移植并标作原 KI。
- D0 不猜测。来源不明的旧 checkpoint 不复用；只有原方法配方与数据边界完整时，才可把新 D0 训练列入冻结预算。

历史成绩未绑定不应阻止真实源码的静态审查与必要合成资格，但不允许借此偷换方法。

## 5. 按发现结果分支执行

### A. 原方法身份、配置和接入语义已核实

生成新的 KI_PARENT_FREEZE.json，历史成绩是否已验证单独列出。按旧 G0 完成必要工程资格，关闭 AMS 应还原已冻结的 KI_REF 行为。

若唯一缺项是旧成绩，则采用新的匹配 KI_REF，不使用旧分数。若 D0 谱系未验证，按明确配方新训练并单独计数，不使用来历不明权重。

冻结 SCHEDULE_ADAPTATION、实际 TASK_MATRIX 和 BUDGET；所有硬前提及测试通过后，沿用户已有 G0/条件 G1 授权继续，无需再索要同样的 GPU/执行确认。

G1 仍限旧协议的五个核心配方，KI_REF 不等价时增为六臂；两顺序、两个预定优化种子。预算从父配置计算，不硬填 106000；首域和合成/smoke 成本分列。不得扩展 G2/G3。

若环境不能完成执行，如实交付实际进度/退出状态，不能把启动算完成，不能自动开启替代实验。

### B. 找到真实候选，但原方法身份或必要配置无法确定

交付 `SOURCE_FOUND_IDENTITY_OR_CONFIG_INCOMPLETE`。列出具体候选路径、已经确定的字段、唯一/最少的阻塞字段及其预期所在记录。允许做不依赖这些缺项的静态工作，但不猜超参数、不启动正式训练。

最终不要笼统重问“请提供仓库、配置、结果目录”三件套；明确到例如“已经找到 X/某入口和 Y/某配置，尚缺原 run_id 对这一配置的绑定”。不可通过选最高分的记录认领父方法。

### C. 找到的是另一任务/骨干中的原 KI

交付 `PARENT_FOUND_PORT_REQUIRED`：冻结真实源路径，说明任务/架构/输出与数据接口差异，生成保持原机制的最小移植差异说明。本轮不把移植自动当作已有 G1 接入，不改 head/backbone/rank 以强行兼容，也不开始新训练。

### D. 没有定位到任何可核实父源码

交付 `PARENT_NOT_LOCATED_IN_ACCESSIBLE_SCOPE`。明确已搜索哪些范围、排除了哪些线索，以及“当前计划中的原 KI 父实现尚未绑定”的事实。

不能写成方法不存在，不能写成方法无效，不能伪造 PASS。输出只缺的最小原始线索；不自动换成全参数 Sequential-SSL/O-LoRA，不重跑旧 106000 次更新，不自动另开单域实验。

## 6. 最少交付

在新目录保存：
- PROTOCOL_AMENDMENT.md
- SEARCH_LEDGER.csv（引用已有审计与本次新增搜索）
- CANDIDATE_LINEAGE.csv（没有候选也保留表头）
- KI_PARENT_FREEZE.json（身份、历史成绩、兼容性分项状态）
- RECOVERY_REPORT.md

只有确实具备资格时才生成有效 TASK_MATRIX.csv、BUDGET.json 和测试通过证据。实际操作计数精确填写；未执行不得填 PASS，不可把未知计数填零。

公共发布仅含必要源码差异、脱敏元数据和聚合说明；私有日志、患者信息、权重与原始路径映射按既有 NAS 边界保留。不覆盖旧审计报告；新报告通过引用说明恢复进展。

最终回答首先给出真正原因、是否找到原实现、确切源码/配置定位、历史成绩验证状态和是否满足 G1；最后说明实际执行了什么。不要仅复述旧 `BLOCKED`，也不要只返回一份准备开展的搜索计划。
