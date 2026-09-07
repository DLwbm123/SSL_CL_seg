# SSL_CL_seg — CARe-HR V0.7.1：语义修订与动作空间容量诊断

## 给 Codex 的执行提示词

你正在处理 GitHub 仓库 `DLwbm123/SSL_CL_seg`。

本轮任务不是直接训练 CARe-HR，也不是重新运行旧 SHOR/PPC 实验。请执行一个新的、范围有限的工作包：

**CARe-HR V0.7.1 Semantic Repair + Action-Space Capacity Diagnostic**。

目标是回答：保持现有历史专家、候选区域、混合系数和主预算不变时，局部修正的动作空间，是否足以承载 V0.7 草案要求的历史收益？

本提示词授权新工作包的源码开发、合成测试、限定开发数据上的冻结推理及 evaluator-only oracle 诊断。它不代表旧 V0.7 已通过独立代码审阅，不解除旧锁，不授权任何分割模型训练、真实数据 router/risk-head 拟合、旧测试集再评价或后续阶段自动执行。

执行到本轮报告与停止点，不要只交付计划文档。

---

## 1. 起点、范围与历史保护

- 仓库：`DLwbm123/SSL_CL_seg`。
- 已核验基线提交：`61c1e302fae515fe51adf4e777887a1a489859be`。
- 原分支：`codex/care-hr-v0-7-code-review`。
- 新分支：`codex/care-hr-v0-7-1-capacity-audit`。
- 新实现目录：`experiments/lcrseg/care_hr_v0_7_1/`。
- 新测试目录：`experiments/lcrseg/tests/care_hr_v0_7_1/`。
- 新公开文档目录：`experiments/lcrseg/docs/care_hr_v0_7_1/`。

先核对远端和本地 SHA、工作区及现有协议。不得覆盖用户未提交工作，不得 reset/clean 原工作区。新工作使用隔离 worktree。若远端已有新提交，记录差异；不得静默把实验换到未审阅版本。仍以给定不可变提交为基线；存在明确冲突的更新授权则停止并报告。

阅读根及 experiments/lcrseg 下的 AGENTS、方法/实现约定、V0.7 草案与锁、V0.6B 预注册/状态/清单。既有指令若要求额外真实执行权限而现有用户授权不能满足，保留阻塞，不伪造外部批准。

旧 V0.7 源码、测试、设计草案和 REVIEW_LOCK 逐字节保留；SHOR V0.3.1/V0.4、RC V0.5、PPC V0.6A/V0.6B 的状态及产物全部保持不变。不得合并 main。新诊断使用独立命名空间和诊断入口，不能通过把旧锁改成 true 来执行。

本轮禁止：
- 分割训练、微调、LoRA、adapter、EMA/GAS/prototype 更新、optimizer/backward；
- 真实数据上的 Ridge/PAV/temperature/risk-head 拟合；合成数组上的单元测试拟合另行计数；
- 搜索新超参数、新预算、新连通区域规则；
- V0.4 formal_03 私有测试产物读取；
- 任何 own-seed val/test 分割 GT、train_unlabeled hidden GT 读取；
- Prostate、M&Ms、外部确认实验及后续阶段自动启动。

## 2. A0：修订指标语义，不改变主动作空间

主指标统一命名为：

`macro_fg_dice = (dice_rim + dice_cup) / 2`

对任意最终动作 a，定义：

`gain_macro_fg(a) = macro_fg_dice(after_a) - macro_fg_dice(current)`

`gain_rim(a) = dice_rim(after_a) - dice_rim(current)`

`gain_cup(a) = dice_cup(after_a) - dice_cup(current)`

`harm(a) = max(0, -gain_macro_fg(a), -gain_rim(a), -gain_cup(a))`

合并 rim/cup 后的二值前景 Dice 仅作为 `union_fg_dice` 辅助指标；禁止与宏平均前景 Dice 继续混用 `gain_fg` 名称。

复用并核对冻结评价器的 ignore-label、空类别、空有效图像处理规则。若旧评价器规则互不一致，先阻塞并列出差异，不自行选择一个方便的规则。所有 GT valid-mask 只在 evaluator 内影响计分，不能反向影响 proposal、面积预算或路由。

为 lambda=0.50、0.75 分别生成动作目标并记录 lambda；不得把相同特征、不同 lambda 的不同目标混成一个不带动作条件的监督问题。本轮不拟合这些目标。

主动作空间严格继承 V0.7：
- rim=1、cup=2，4-connected add/remove proposals；
- 最少 8 像素，固定排序和重叠消解，最多 12 个 proposals；
- 单病例最多接受 4 个，同 target_class 最多 3 个；
- 累计 proposal mask 的并集面积不超过当前预测前景面积的 15%，且不超过图像面积的 2%；
- 一次动作使用一个共同 lambda，取 0.50 或 0.75，不允许每个 proposal 自选 lambda；
- 区域外概率逐字节保留，不重新归一化，不形态学后处理；
- no-op 必须合法。

本轮不要给零当前前景添加预算下限，不要拆分过大连通块，也不要按 GT 剔除像素。它们是本轮要测量的限制，不能边测边修。

## 3. A0：修订纯数学组件与接口检查

在新命名空间实现精确的患者级单侧 conformal 秩组件：

- n 是校准患者数，不是 proposal 数或 seed-row 数；跨患者可交换性必须另行论证，去重本身不提供该假设；
- `k = ceil((n + 1) * (1 - alpha))`；
- k<=n 时取排序后的第 k 个残差；k>n 时返回明确的无穷界/证据不足状态；禁止截断为有限最大残差后仍声明保证；
- 患者内重复行/多个区域须按声明的患者级 score 聚合；
- 明确这只是数值组件修订，不构成整套方法的风险控制证明。

本轮不做真实 conformal calibration。普通 OOF 残差拼接、独立 gain/harm 两个边界、多个动作选择和域偏移不能自动继承单个 split-conformal 的保证。将这些限制写入下一阶段草案。

同时拒绝所有长度不匹配的数组，禁止 zip 静默截断；检查 shape、有限值、概率合法性、重复 proposal、面积字段一致性、重叠区域、有限预测数与投票数范围。明确 distinction：proposal mask 面积、实际硬标签变化面积、概率变化面积。

## 4. A1：必须通过的合成测试与回归

先在不读真实数据的条件下完成以下测试：

1. 语义反例：current=[1,1,2,2]、truth=revised=[1,2,2,2]；union Dice gain=0，而 macro foreground gain=4/15。
2. no-op 的各项 gain/harm 为零，概率输出与 current 逐字节相同。
3. 接受区域之外逐字节不变；输入概率和 proposal mask 不被原地修改。
4. 多区域最终 Dice 必须从组合后预测或精确充分统计量计算，不得相加单区域 Dice gain。
5. 多个单区域 harm 分别低于阈值时，组合 harm 仍可能超限的反例。
6. 区域重叠消解、排序、并集面积、12/4/3 数量上限、两项面积预算全部测试边界值。
7. current foreground=0 时，原主预算只能 no-op，并记录为可诊断限制，不能静默放宽。
8. lambda=0.5/0.75 的目标分别计算；原 dtype 的混合与 argmax tie 规则保持一致。
9. 11 个患者残差为 1..11、alpha=0.1 时，正确第 k 阶结果为 11；补充 k>n、重复患者、极小样本及非有限输入测试。
10. 新 oracle 在小图上的组合枚举，与暴力修改整张预测再评价逐项一致。
11. 固定 lambda 下，严格动作集包含于取消面积约束动作集，后者包含于进一步取消数量约束动作集；各自最优 macro Dice 不得反向下降。
12. 跨 seed 相同患者的映射及统计不能变成独立患者；错误输入/GT 哨兵不能进入 proposal/policy。
13. 原 V0.7、PPC V0.6B 相关测试与历史文件哈希保护仍通过。

不要以达到某个测试数量代替覆盖。报告实际 passed/failed/error/skipped、精确命令、环境与被测源码 SHA。远端数据测试的缺失必须显式说明，不能统称全部通过。

## 5. A2：真实开发诊断的数据与血缘

使用 V0.6B 已冻结的同一 segmentation-value population：预期 198 个 seed-case 行、177 名患者、seed 0/1/2、最终 stage2。

本轮 current 指冻结的 B0/seed/stage2 分割专家，不是 Gate0 中 lambda_u=0 的 C0 训练对照。新报告优先使用描述性策略名称，避免跨协议复用 C 编号造成混淆。

只允许行自身 seed 的 role 为 train_labeled 的 GT。交叉 seed 重复患者必须有统一患者映射。不能因为某患者在另一个 seed 有标注，就开放当前 seed 的 val/test GT。

注意：本轮只是基于已训练模型的开发集结构诊断。outer-fold 身份不能使训练过这些样本的分割专家变成全链路未见患者模型；不得称为泛化性能或独立测试。

历史专家选择：
- 逐行使用 V0.6B 正式冻结 C6 route。
- C6 选择历史专家 h 时，只用该 h 生成主 proposal；不得按 GT 换成另一个专家。
- C6 留在 current 时，主动作集只能 no-op。
- 不重拟合 PPC、不改变其参数、不把 rejected top1 当已接受历史路由。
- 核验冻结 stitched route 的 155/198 历史分配和原清单 route hash；若原接口有不同明确语义，先记录冲突并阻塞，不猜测补齐。

优先复用经哈希核验的允许概率缓存。若缺少缓存，只允许对这 198 行用清单中原 B0/seed/stage 的冻结专家重新物化概率：eval、posterior mean、原预处理和原数值环境；不得使用新 checkpoint。最多 3 个专家/行，即 594 次 sample-expert 前向评价，batch forward 调用另计。若私有模型/数据/必要参考无法获得，输出 BLOCKED_REQUIRED_FROZEN_INPUT_MISSING，不替换为其他材料。

读取范围从既有清单解析为 allowlist，不扫描整个 NAS。旧 formal_03 明确排除。混合缓存若包含 GT，必须使用经审查的盲输入加载路径，避免提前载入标签。

在第一次本轮分割 GT 访问之前：
1. 冻结并公开源码、协议、scope、输入标识、动作定义、门槛及确定性 tie-break；
2. 核验远端 source commit；在该精确代码上通过 A1；
3. 盲阶段物化并密封 current/history 概率、routes、proposals、所有允许动作的索引规则和预算统计；
4. 隔离 evaluator 核验 ACTION_SPACE_SEAL 后，才开启本轮限定 GT 访问。

True domain 仅用于 evaluator 的分组报告，不进入路由/proposal API。私有患者标识、图片、掩码、概率、绝对私有路径和逐病例文件不上传 GitHub。

## 6. A3：精确动作空间 oracle

对病例 i，定义 A_i 为主预算下所有合法 proposal 子集和共同 lambda 所形成的动作，并包含 no-op。

不应用待学习的 gain/harm/consensus 接受门槛；这是有意放宽选择器后的结构上界，不是完整可部署策略。不得把 oracle 选择当作真实 router。

### O_CAP：主动作空间的乐观收益上界

`a*_i = argmax_{a in A_i} macro_fg_dice(after_a, GT_i)`

同时输出每个 lambda 单独的最优结果，以及允许 oracle 逐病例选 lambda 的乐观 envelope。后者比按 fold 冻结一个 lambda 更宽松，必须明确标记；它失败可以排除更受限策略，它成功不证明实际策略可达到。

12 个 proposals 中最多接受 4 个，因此主动作候选数在进一步预算筛选前至多为：

`1 + 2 * (C(12,1)+C(12,2)+C(12,3)+C(12,4)) = 1587`。

必须穷举，不准用贪心选择冒充上界。lambda=1.0 不能替代主 oracle，也不是 lambda=0.5/0.75 动作集的上界。

### O_SAFE0：严格逐例、逐类不退化的 oracle

在同一 A_i 内额外要求：

`gain_rim(a) >= 0 and gain_cup(a) >= 0`，然后最大化 macro foreground Dice。

约束用未舍入的精确计分语义判断；不要以容差偷偷允许损害。no-op 保证动作集非空。

该约束比草案域平均安全门槛更严格。因此：O_SAFE0 高表示存在安全容量；O_SAFE0 低不等于所有满足原平均安全门槛的策略都不可能。

### 两个固定的嵌套诊断，不能升级为主方法

- O_NO_AREA：同一冻结 h、同一 12 个 proposals、同一 lambda 集合；仅取消两项面积限制，保留最多 4 个和每类 3 个。
- O_FREE_SUBSET：在 O_NO_AREA 基础上取消接受数量/每类数量限制，仍只允许原最多 12 个 proposals 的子集，不新增/切割 proposal。每例最多 `1+2*(2^12-1)=8191` 个动作。

两者只解释受限容量来自哪里。不能看到结果后选其中一个当作 V0.7.1 主结果，也不能用它们的通过救援 O_CAP 失败。

所有 oracle 的确定性 tie-break 在 GT 访问前固定为：macro foreground Dice 高优先；完全相等时选 mask 面积更小、区域更少、lambda 更小、proposal_id 元组字典序更小；no-op 优先于无收益修改。不要按 rounded metric 或新容差造 tie。

### 计算实现

复用已经生成的预测，禁止逐动作进行网络前向。对非重叠 proposal 和固定 lambda，缓存相对 current 的 int64 confusion-count 增量；组合增量后重新计算整例指标。Dice gain 本身不可加。保留像素暴力版本作为合成独立校验，不在所有真实动作上复制大图。

## 7. 对照、统计和失败归因

同一 198 行上报告：current、冻结 Ridge hard、冻结 SHOR、冻结 PPC C6，以及上述四个 oracle。只能从允许的 PPC 血缘和缓存中取得旧对照，不能访问 V0.4 私有测试结果。

先确认 current 和 PPC 的逐例输出及聚合结果与冻结参考一致；公开参考的整体 macro Dice 为 current≈0.6016731401004847、PPC≈0.802187012880861。精确门槛比较使用原机器可读未舍入值。若同口径重算不一致，标记 BLOCKED_BASELINE_MISMATCH，不能继续解释科学增益。

主聚合完整复用并记录 V0.6B 的 case/class/seed/domain 权重，不从逐病例宏平均偷偷切换到全局 confusion Dice。另给患者等权敏感性统计，不混写为主结果。所有策略严格配对同一行顺序。

至少输出：
- 整体/历史域收益、各域与各 seed 的收益、当前域及类别下降；
- 相对冻结 PPC 的整体和历史收益差；
- 实际 proposal 数、合法动作数、no-op 比例、零当前前景比例；
- 面积预算拒绝数、数量预算拒绝数、实际改动 mask 面积和 hard-label 面积；
- O_CAP→O_NO_AREA→O_FREE_SUBSET 的嵌套容量差；
- O_CAP 与 O_SAFE0 的差和逐例收益/伤害 Pareto 数据；
- 模型/专家参数量与缓存字节、实际新增和复用前向量、计算耗时及峰值内存。

本轮没有 router 重拟合，因此不得把固定动作的病例统计写成 200 次重拟合稳定性，相关 p90/feasibility 项应为 NOT_EVALUATED。没有合适的可学习规则时，oracle 的安全也不能称为已完成风险校准。无需为了填满表格实施额外 bootstrap 拟合。

## 8. 本轮预注册判定与硬停止

下面只是动作空间筛查，不是旧 V0.7 或整个 CARe-HR 的方法通过。

沿用草案两个必要价值目标：三域平均增益 0.17，历史域平均增益 0.27。完整草案其他门槛须保留并逐项列出本轮可计算项，不能删除；未实施的拟合稳定性不能填 PASS。

按以下顺序判定：

1. 任一权限、数据血缘、实现、回归、密封或基线一致性失败：相应 BLOCKED，不能写科学 FAIL。
2. 若逐病例允许自由选 lambda 的 O_CAP 乐观 envelope，整体收益<0.17 或历史收益<0.27：
   `FAIL_FROZEN_ACTION_SPACE_CAPACITY`。
   结论仅限本冻结开发群体与动作空间。即使放宽预算 oracle 达标，也不改变主失败；停止，不拟合风险模型。
3. 若 O_SAFE0 envelope 同时达到 0.17/0.27，两个历史域收益均>0，三个 seed 的整体收益均>0：
   `PASS_ACTION_SPACE_SCREEN_ONLY`。
   只说明存在值得研究的安全动作容量，不证明 lambda 可盲选、oracle 可学习或方法已成功；停止并交付下一阶段草案。
4. 其余情况：
   `CAPACITY_PRESENT_SAFETY_UNRESOLVED`。
   记录 Pareto 取舍与受限环节，不把更严格 safe oracle 的失败当成原安全目标不可能，也不自动扩预算/跑拟合。

三个科学终态都停止本轮。不得在看到 GT 后降低价值目标、改预算、换专家、增大 proposals 数或换 oracle 搜索方式再争取 PASS。

工程中断不等于科学失败。保留实际进程退出与 artifact 完整性；只有冻结代码/输入不变、未生成终态、已有块校验通过时，才允许同一诊断的幂等恢复。GT 开放后若需要修改计分/动作/判定代码，先停止并报告 incomplete，不现场改代码重判。不要用 SSH 传输退出码冒充计算程序退出码。

## 9. 交付与下一阶段草案

公开交付至少包括：
- CODE_REVIEW_RESPONSE.md：每个问题、修订位置、回归证据；
- PREREGISTRATION.md / .json：明确新诊断范围及非授权范围；
- SYNTHETIC_TEST_REPORT.json：实际测试与精确被测 SHA；
- SOURCE_LINEAGE.json / MANIFEST.json：源码、输入、模型、动作空间和输出血缘；
- ACTION_SPACE_SEAL 的脱敏摘要及核验结果；
- CAPACITY_METRICS.csv、BUDGET_ATTRIBUTION.csv、CAPACITY_STATUS.json；
- CAPACITY_FINAL_REPORT.md：主判定、所有对照、限制、错误与警告；
- EXACT_COMMANDS.md：可复核命令，无密钥和敏感路径；
- NEXT_STAGE_DRAFT.md：只有设计，没有启动许可。

逐病例指标、Pareto、动作索引、图片、标签和概率保留私有，公开文件不泄露患者身份。报告 source commit、report commit，不把报告 HEAD 假称实际执行源码。推送新分支并核对远端 SHA，main 不变。

下一阶段草案仅在容量结果支持时建议：患者分组 nested development evaluation；按 lambda 区分收益目标；对最终组合动作而非单 proposal 定义/校准风险；保留 current/Ridge-hard/SHOR/PPC/简单局部融合强对照；报告患者级尾部风险和重新拟合稳定性。外层 fold 必须隔离新风险模型的拟合、标准化、超参数选择与校准，但不能因此把既有分割模型的 in-sample 开发样本称为独立测试。

还须写清：历史 GT 用于风险头拟合是否改变原“不回访历史数据”的任务约束；零 optimizer 并不等于没有监督拟合。既有 Fundus 已进入开发使用，未来确认应使用新的、预先隔离的数据，不重用旧 formal_03。

最终回复只报告事实：status、分支、source/report SHA、测试结果、真实输入/前向/GT计数、四类 oracle 指标、预算归因、文件位置、下一阶段准入结论。若阻塞，列出缺少的具体输入/权限与已经完成的部分。无真实结果时禁止填占位性能或宣称 PASS。

---

## 编制依据（供审阅，不是额外执行授权）

仓库固定版本：61c1e302fae515fe51adf4e777887a1a489859be。

- `experiments/lcrseg/docs/care_hr_v0_7_review/CARE_HR_V0_7_DESIGN_DRAFT.json`
- `experiments/lcrseg/docs/care_hr_v0_7_review/CARE_HR_V0_7_REVIEW_LOCK.json`
- `experiments/lcrseg/care_hr_v0_7/targets.py`、`policy.py`、`conformal.py`、`proposals.py`
- `experiments/lcrseg/docs/ppc_shor_v0_6b/PPC_SHOR_V0_6B_PREREGISTRATION.md`、`PPC_SHOR_V0_6B_FINAL_REPORT.md`、`PPC_SHOR_V0_6B_METRICS.csv`
- Angelopoulos 与 Bates，A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification，arXiv:2107.07511，Theorem D.1：有限样本秩、无穷边界与可交换性前提。

本文是新的实验建议和执行约束，不是已经运行的实验报告。
