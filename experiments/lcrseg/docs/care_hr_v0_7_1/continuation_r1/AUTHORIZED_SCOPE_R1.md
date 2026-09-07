# SSL_CL_seg — CARe-HR V0.7.1 R1：评分语义裁定与容量诊断续接

## 交给 Codex 的完整执行指令

你正在处理 `DLwbm123/SSL_CL_seg`。继续完成 CARe-HR V0.7.1 的 A0–A3，不新增方法，不启动风险模型拟合或分割训练。

本指令明确解决此前的 `BLOCKED_EVALUATOR_SEMANTICS_MISMATCH`：**本轮统一采用冻结 PPC-SHOR V0.6B 实际调用的 `shor_v0_4_test.case_metrics` 计分语义。旧 V0.7 的 targets helper 不是新的评分规范。** 在新命名空间修订 ignore-mask 和 macro-foreground 目标，不修改旧实现或历史结果。

本指令是本次用户转交执行的 R1 补充协议。对于旧提示词中的“旧评分器不一致即停止等待确认”，本补充已经给出明确决定；对于既有保护、数据范围、动作空间、oracle 定义、门槛及禁止训练事项，继续保持。不得把本指令伪造为旧 V0.7 全面审阅通过或已签署的其他外部授权。

本次范围内的编码、测试和限定开发诊断，在前置门槛满足后连续执行到真实容量报告，不仅提交又一份计划或合成测试结果。遇到本指令未解决的真实权限、输入或数值完整性问题，再按对应 BLOCKED 终止；不得把已明确裁定的相同语义差异重新当作未决问题。

---

## 1. 版本、分支与不可变历史

- 仓库：`DLwbm123/SSL_CL_seg`。
- 续接分支：`codex/care-hr-v0-7-1-capacity-audit`。
- 本次起点／predecessor report commit：`2f28c63d6dedb057fce96f131038f1889804ac10`。
- 前一轮被测 source commit：`c98a4e8e46d133d1b51360410c43873a2c283de0`。
- 更早 V0.7 基线：`61c1e302fae515fe51adf4e777887a1a489859be`。
- 实现范围：`experiments/lcrseg/care_hr_v0_7_1/`。
- 测试范围：`experiments/lcrseg/tests/care_hr_v0_7_1/`。
- 新协议、测试证据及结果目录：`experiments/lcrseg/docs/care_hr_v0_7_1/continuation_r1/`。

先读取根及 `experiments/lcrseg` 下的 AGENTS、既有协议、旧 V0.7 锁、本轮前驱报告及附件采用记录。核验分支、HEAD、工作区；使用隔离 worktree，不覆盖未提交工作，不 reset/clean，不强推。若远端与给定起点不同，先记录新增提交，不能静默更换被测版本。

前驱 `CAPACITY_FINAL_REPORT.md`、`EVALUATOR_SEMANTICS_AUDIT.json`、`SYNTHETIC_TEST_REPORT.json`、其他已发布前驱证据及其状态逐字节保留。82 个已保护历史文件仍核验 SHA-256；新一轮不能降低旧保护范围。前驱审计证明的差异是历史事实，不回写为“不存在”。

可以在 V0.7.1 新命名空间新增 scorer、targets、oracle、续接 executor 和测试；需要修改该命名空间原有纯函数时必须列明必要性与 diff。旧 V0.7、SHOR、RC、PPC 的源码、测试、状态、锁、真实产物不可修改，不合并 main。

原 `semantic_audit.py` 可以原样保留，用于重现已确认的历史差异及预期退出码 2；新增独立的 resolution/parity audit。不要简单把原脚本中的 BLOCKED 文本改成 PASS，也不要要求新 scorer 与已知有缺陷的 V0.7 helper 相等。新的通过条件是“裁定已登记＋新 scorer 与冻结 V0.6B 参照一致＋全部回归通过”。

前驱报告记载真实输入、前向、GT 读取均为 0，没有创建 ACTION_SPACE_SEAL；这只是公开前驱证据。本次需检查可用执行收据，不把前驱 129 个测试当作新源码已通过测试。使用新的 run_id 和新输出目录，不覆盖前驱终态，不冒充同一已完成结果重跑。

## 2. 本次评分语义裁定：SCORING_CONTRACT_R1

在任何本轮真实 GT 读取前，写出并冻结 `SCORING_CONTRACT_R1.json` 和解释文档。

### 2.1 输入合法性与 ignore mask

- 预测 hard label 只能是整数 `{0,1,2}`；GT 只能是整数 `{0,1,2,255}`。
- 0=background、1=rim、2=cup；255 仅是 GT ignore label，不是第四类。
- 预测与 GT 必须为相同形状的非空二维数组。非法类别、错误 shape、缺失 GT 或损坏文件不能被转换为全 ignore 图像。
- 从概率得到 hard prediction 时，沿用原 dtype、混合运算与 `numpy.argmax` 的最低索引 tie 规则；概率有限性、类别数、形状、取值范围与和的合法性先验证。
- `valid = (GT != 255)`，且它只存在于隔离 evaluator。
- 对每个类别，prediction support、truth support、intersection、union 都必须同时与 valid 取交。
- 仅修改 ignore 像素上的合法预测，不得改变任何 Dice、IoU、gain 或 harm。
- 禁止将 ignore 像素重新编码为背景后参与评分；禁止用预测标签构造 valid。

### 2.2 类别级 Dice、IoU 与空类别

在 valid 内计算整数计数：

`D_c = 2*TP_c / (P_c + T_c)`；若 `P_c + T_c == 0`，令 `D_c = 1.0`。

`IoU_c = TP_c / U_c`；若 `U_c == 0`，令 `IoU_c = 1.0`。

不添加 smoothing/epsilon，不更换分母，不采用库的默认“忽略空 GT 类别”逻辑。

因此：
- 真值和预测在某类上都为空：该类 Dice=1；
- 真值为空但预测非空：该类 Dice=0，不能跳过这个 false positive；
- 真值非空但预测为空：该类 Dice=0。

主指标：`macro_fg_dice = (dice_rim + dice_cup)/2`，不含背景。

兼容字段 `foreground_dice` 只可明确作为 `macro_fg_dice` 的别名；不能再指 union foreground。

兼容 `mean_iou` 保持原评分器的三个类别 IoU 等权平均，包含背景。若增加 foreground IoU，必须用另一个明确的字段，不能覆盖 mean_iou。

辅助 `union_fg_dice`：先合并 `{1,2}` 为一个二值前景，再应用同一 valid mask 与空 support 约定；不进入主收益定义。

### 2.3 全无有效像素：保持数值兼容，显式标记无标注证据

当一张真实存在且形状合法的 GT 全部为 255 时：
- 兼容评分与 V0.6B 完全一致：三个类别 Dice/IoU、macro_fg_dice、mean_iou、union_fg_dice 均为 1.0；
- 对所有动作，before=after，各项 gain=0、harm=0；
- oracle 按原先固定的 tie-break 选择 no-op，不允许因 ignore 像素变化制造收益；
- 输出 `n_valid_pixels=0`、`has_evaluable_gt=false`、`all_pixels_ignored=true`、`metric_from_empty_support_convention=true`。

这里的 1.0 是兼容计分约定，不是“模型预测完美”；这里的 harm=0 不是“观察证明安全”。

**主兼容聚合保留这些冻结行及原有权重**，不得删除、补样、换样、改变原 cohort 或偷偷重新归一化，以免改变 V0.6B 对照和 0.17/0.27 门槛的含义。

额外输出预先声明的 `EVALUABLE_ONLY_SENSITIVITY.csv`：每个原 seed/domain 组内去掉 `n_valid_pixels=0` 的行后重算同一组内统计，再使用原来的组间权重；每项明确列出保留/排除行数、患者数和分母。如果某个原 seed/domain 组没有任何可评价行，相关敏感性汇总记为 UNDEFINED，不静默丢掉整组重算。

敏感性结果不得替换主结果、选择策略、改变门槛或“救援”失败。安全、纠正 precision 等有实际标注证据的额外计数，不把全 ignore 行当作观测成功；明确分母。整批没有有效标注或规定的某个 seed/domain 完全没有有效标注时，只能给出兼容描述值及 `BLOCKED_NO_EVALUABLE_SUPPORT`，不能给科学容量 PASS/FAIL。

仅仅遇到部分全 ignore 行，不是新的评分语义冲突；本条已经给出处理方式。

### 2.4 动作目标与最终组合评分

对完整动作 a：

`gain_macro_fg = macro_fg_dice(after_a) - macro_fg_dice(current)`

`gain_rim = dice_rim(after_a) - dice_rim(current)`

`gain_cup = dice_cup(after_a) - dice_cup(current)`

`harm = max(0.0, -gain_macro_fg, -gain_rim, -gain_cup)`

目标分别记录 lambda=0.50、0.75 与 action_id；不混为无动作条件的拟合目标。本轮不拟合这些目标。

对多个 proposal，先得到最终组合预测再评分，或累加 int64 confusion-count 增量后重新计算整例指标；禁止相加单 proposal 的 Dice gain/harm。可明确采用 truth-row/prediction-column 的 3x3 confusion matrix，但必须在代码和测试中固定方向。

### 2.5 盲动作绝不使用 valid mask

valid mask、GT 为空的标志、真实域身份均不得进入路由、proposal 生成、acceptance、面积预算或动作枚举。

主预算的当前前景仍由 **整幅 current prediction** 的 `{1,2}` 像素数得到；图像面积仍是 H*W；proposal 面积仍是原 mask 并集面积，包含其中可能恰好处于 ignore 区的像素。

不因 GT=255 给 proposal 裁边、退还预算、降低 proposal 面积、变更路由或删除病例。全 ignore 的 no-op 必须由 evaluator 的零收益 tie-break 产生，不能在盲阶段读取 GT 后提前返回。

## 3. 完成 A0/A1：不是只修三个反例

先在合成数据上完成评分实现、target API、组合 oracle 与以下所有测试。可以复用前驱 conformal、actions 等纯组件，避免无关重写。

### 3.1 语义与数值见证

下面的短序列是便于阅读的写法；实际测试均包装为 `1xN` 二维数组，避免与 shape 检查相冲突。

1. `current=[1,0]`, `revised=[1,1]`, `truth=[1,255]`：所有主 gain=0，harm=0。
2. `current=[1,1]`, `revised=[0,0]`, `truth=[255,255]`：兼容 score 前后均为 1；gain/harm=0；has_evaluable_gt=false；oracle no-op。
3. `current=[1,1,2,2]`, `truth=revised=[1,2,2,2]`：union gain=0；macro gain=4/15（浮点输出按冻结运算）；不再使用 union 作为主目标。
4. `current=[1,1]`, `revised=[1,2]`, `truth=[1,1]`：cup before=1、after=0，不能跳过空 GT cup；harm=1。
5. no-op 对所有合法场景输出逐字节不变，主 gains/harm 为零。
6. 混合 ignore、全背景、空 cup/rim、仅一像素有效、非法 GT 类别、非法预测类别、缺失/空数组/shape 不匹配均有明确测试。
7. 可穷举两像素数组：current/revised 各取 `{0,1,2}^2`，truth 取 `{0,1,2,255}^2`，共 1296 个组合，检查新分数、差分及 harm 与冻结评分器一致。报告组合数，不冒称 1296 个独立 pytest 用例。

冻结参照来自前驱所核验的 `shor_v0_4_test.py:case_metrics`；其文件 SHA-256 为 `7b62332879985d831efdb90fa50a3f303a6e86bd1d97a04c59b451ca1a6fd997`，使用前须复核。允许沿用前驱 AST 提取纯函数的独立 golden 方式，避免为评分测试导入具有真实执行副作用的完整旧 executor。

### 3.2 组合损害与 oracle 精确性

补上前驱标注 NOT_IMPLEMENTED/PARTIAL 的内容，不以通过数量代替覆盖。

组合损害具体测试：40x40 图像含 800 个 rim、800 个 cup；current 与 truth 一致。两个互不重叠、彼此不四连通的 8 像素 rim 区域分别改成背景（例如按行展平索引 `[0:8]` 与 `[16:24]`）。每个单区域 harm 约为 0.00502512562814，均 <=0.01；两个合并后 harm 约为 0.01010101010101，超过 0.01。最终必须按组合后结果计算，不把它当成仍满足单区域 harm 限制。

还需测试：
- 所有合法动作的 int64 confusion 增量评分与独立逐像素暴力评分逐项一致；
- 固定 lambda 下 O_CAP 的动作集包含于 O_NO_AREA，后者包含于 O_FREE_SUBSET，对应最优宏 Dice 不反向下降；
- O_SAFE0 含 no-op，逐类非下降判断不加新的安全容差；
- ignore 像素变化不改变任何评分，但 GT/valid 的改变也绝不影响盲动作枚举结果或预算；
- lambda=0.5/0.75 分开计分，保持原混合 dtype 和 argmax tie；
- 12/4/3 限制、两项面积预算、交叠消解、计数边界及 current foreground=0 仅允许 no-op；
- 输入、概率、proposal 不原地修改，区域外逐字节不变，length mismatch 不得 zip 截断；
- 患者级 conformal 秩、k>n 无穷界及输入异常回归保持；本轮不增加真实校准。

A1 的患者重复/跨 seed 验证先用合成映射与接口哨兵；真实人口与跨 seed 血缘的核验属于 A2 盲 metadata preflight，不要求为了声称 A1 完成而提前读取真实 GT。

### 3.3 通过条件与版本证明

- 旧 95 项前驱回归、新 V0.7.1 全部合成项和历史保护核验都要在新精确源码上执行；不把历史 129 passed 当成新源码证据。
- 报告每项 A1 的具体覆盖状态，不把未实现项写成无 skip 即完成。
- 新 resolution audit 记录 `historical_difference_detected=true`、`resolution=ADOPT_FROZEN_V0_6B_SCORER`、`unresolved_semantics_count=0`，且只有真实 parity 通过后才退出 0。
- 原历史差异审计预期退出 2可以保留；它不等于新的 resolution gate 失败。
- 纯 case metrics 在同样的 hard prediction/GT 上应逐值完全一致；合成算术恒等式可以用明确测试容差验证数学期望，但不能把这个容差带入安全门槛或 candidate selection。
- 真实主聚合沿用原分组、权重、行顺序和运算；独立重算浮点序列化检查预先固定 `rtol=0, atol=1e-12`，不得看到真实差值后放宽。科学判定仍用未舍入值，不加容差。

## 4. A2：限定开发数据、固定专家和完整盲密封

预期人群保持 198 个 seed-case 行、177 名患者，seed=0/1/2，各 66 行，最终 stage2。真实存在性与私有缓存仍需核验；前驱报告没有验证它们。

仅允许行自身 seed 的 `train_labeled` GT；不读取 own-seed val/test GT、train_unlabeled hidden GT；不得因另一个 seed 有标注就开放当前行 GT。

本轮 current 是清单中的 B0/seed/stage2 expert，不是 Gate0 的 lambda_u=0 控制。只用冻结清单中的模型和 route：

- baseline snapshot manifest SHA-256：`e6d3c833865a3c6773d28851c11e7da48b6496661eac34e34478ff01714a6f1d`；
- 逐行 PPC C6 stitched route SHA-256：`c03c457d0b5ab7a529dd8ec07076b06c2fe41a22e7439265f24ab046687a41e7`；
- C6 historical assignments：155/198；
- 其他输入 SHA 和 cache SHA 从前驱 `SOURCE_LINEAGE.json` 及原正式清单解析核验，不扫描整个 NAS、不重新拟合路由。

C6 选择历史 h，只用该 h 生成 proposals；C6 留在 stage2，该行所有主及嵌套空间只允许 no-op。不用 GT 或诊断 domain 换专家，不把被拒 top1 偷当历史路由。

优先复用允许的、经哈希核验的概率缓存。缺少缓存时，只允许对同一 198 行和原三专家做冻结 posterior-mean、eval、无增强推理；最多 594 个实际新增 sample-expert forward evaluations，batch 调用另计。重建失败输入、重复计算和诊断前向也必须记录，不能把“最多 594”解释成无限次重试只统计唯一结果。原概率环境、预处理和确定性设置保持。

GPU 4/5 与 NAS 的前驱可用性只是历史观测。执行时只读确认当前资源占用和允许输入；不停止或占用他人作业。无法获得必要输入时输出具体 BLOCKED，不替换新 checkpoint 或新样本。

第一次本轮 segmentation GT/true-domain 数据访问前：
1. 发布并核验 R1 协议、评分 contract、源码 freeze、全部 A1 证据；记录 source commit，不能把后续 report SHA 伪称被测代码。
2. 从允许 metadata 建立行序、patient 映射、缓存/模型/route allowlist；盲 executor 不接收真实 GT/domain。
3. 盲生成并密封 probabilities、hard predictions、routes、proposals、固定动作枚举定义/索引、mask 面积和预算计数。
4. 写 `ACTION_SPACE_SEAL.json`，核验内容哈希；然后原子创建本次 evaluator 访问 reservation。
5. 只有隔离 evaluator 核验密封后才能打开限定 GT 和 true-domain，用于评分和分组。

GT/valid mask 不得通过 shared dict、提前加载的 archive、cache sidecar 或 logging callback 进入盲进程。既有混合文件若含 GT，应使用经审查的分离读取路径，不能先把全部内容载入内存再声称“没有使用”。

## 5. A3：保留原四类 oracle，精确枚举

原 proposal 定义不变：rim/cup 的 4-connected add/remove；最少 8 像素；原排序和重叠消解；最多 12 个。一次动作使用共同 lambda，取 0.50 或 0.75；包含唯一 no-op；区域外概率逐字节不变。

主预算不变：最多接受 4 个 proposal、每 target_class 最多 3 个、累计原 mask 并集面积 <= 当前整幅预测前景面积的 15%，且 <= 图像面积的 2%。不加前景面积下限，不裁分大片 proposal，不加 morphology。

四类 evaluator-only oracle：

1. **O_CAP**：原预算下最大化 macro_fg_dice；不施加待学习 gain/harm/consensus 门控。分别报告固定 lambda 结果和逐病例自由选 lambda 的乐观 envelope。上限 1587 个动作，必须穷举。
2. **O_SAFE0**：同一动作集，额外要求最终组合 `gain_rim>=0 AND gain_cup>=0`，然后最大化 macro Dice。no-op 永远合法。它比原域平均安全条件更严格；失败不能证明所有平均安全策略都不可能。
3. **O_NO_AREA**：仅取消两项面积限制，仍最多 4 个、每类最多 3 个；同专家、同 proposals、同 lambda。
4. **O_FREE_SUBSET**：再取消接受数量和每类数量限制，仍限定原最多 12 个 proposal 子集；最多 8191 个动作。

后三者的得分都从最终组合预测或精确 confusion statistics 得到。不对每个动作做模型前向。lambda=1.0 不替代这组 oracle，也不加作新的救援候选。

固定 tie-break：macro Dice 高优先；完全相等时 mask 面积小、区域数少、lambda 小、proposal_id 元组字典序小；无收益时 no-op 胜出。不按 rounded metric、新容差或看到 GT 后的新准则打破 tie。

两个放宽空间只作容量归因，不能升级为主方法，不改变 O_CAP 的科学判定。oracle 使用 GT，不是可部署策略，也不是训练完成的方法。

## 6. 同口径基线、聚合和本次输出

同一批行上报告：当前专家、冻结 Ridge hard、冻结 SHOR、冻结 PPC C6，以及四类 oracle；oracle 单列。对照来源只限允许的 PPC 血缘和缓存，禁止读取 V0.4 `formal_03`。

先完成允许的逐病例基线一致性和分组聚合核验。公开历史参考（尚不是本轮重算结果）为：

- current macro_fg_dice：0.6016731401004847；
- Ridge hard：0.8156293208569301；
- frozen SHOR：0.8052579174291238；
- frozen PPC C6：0.802187012880861。

必须复用 V0.6B 实际完整聚合调用链的 case/class/seed/domain 权重及跨 seed 处理，不仅抄 case_metrics 后自创聚合。不以 global confusion Dice 替换 mean-of-case Dice。不根据公开 rounded 数值放宽差值容差。基线不一致，先 `BLOCKED_BASELINE_MISMATCH`，不继续解释 oracle 科学增益。

至少输出：
- 各策略整体、历史、各域、各 seed 的宏 Dice/gain 和每类 score/drop；
- 相对冻结 PPC 的整体/历史 gain 差；
- O_CAP 固定 lambda 与 envelope，O_SAFE0 固定 lambda 与 envelope；
- O_CAP -> O_NO_AREA -> O_FREE_SUBSET 容量差；收益/损害 Pareto 诊断；
- proposal 数、合法动作数、no-op 频率、零 current foreground 比例、面积/数量拒绝原因；
- 原 mask 面积、实际 hard-label 改动面积、probability 改动面积分别报告；GT-valid 改动面积仅在 evaluator 另列，不回传；
- valid/ignored pixels、all-ignore 行数和患者数、各 seed/domain 标注覆盖及兼容计分说明；
- 主兼容统计、患者等权敏感性、evaluable-only 敏感性分别命名，明确每项权重与分母；
- 参数/存储/缓存成本、实际新增与复用的 sample-expert 前向数、batch 前向数、GT/domain 读取数、实际耗时和峰值内存。

真实拟合次数必须保持 0，包括 Ridge、PAV、temperature、risk-head、conformal 校准；继承测试中的纯合成拟合另计，nested wrapper 不重复求和。分割 optimizer/update/EMA/GAS/prototype 更新计数为 0。不把未评估指标填成 0。

本轮没有 refit，因此原 bootstrap p90、feasible refits 等稳定性指标写 NOT_EVALUATED；不得为了填表补跑真实拟合，也不得把 oracle 的逐例非下降当成校准保证。

## 7. 判定与停止：不改目标、不补跑风险头

前提是权限、A1、输入血缘、密封、有效标注支持、基线一致性全部通过。

- 若 O_CAP 逐病例自由选 lambda 的乐观 envelope，整体 gain<0.17 或历史 gain<0.27：`FAIL_FROZEN_ACTION_SPACE_CAPACITY`。
- 否则，若 O_SAFE0 envelope 整体 gain>=0.17、历史 gain>=0.27，两个历史域各自 gain>0，三个 seed 各自整体 gain>0：`PASS_ACTION_SPACE_SCREEN_ONLY`。
- 其余：`CAPACITY_PRESENT_SAFETY_UNRESOLVED`。

这些是本冻结开发群体与动作空间的筛查，不是独立测试、不是完整 CARe-HR PASS。0.17/0.27 为 Dice 原始比例增益，即 17/27 个百分点。旧完整草案的其他门槛仍逐项列出，但未执行项不填 PASS。

既有 Fundus 已被开发使用，当前专家可能训练过这些行；本轮不能用 outer-fold 名称声称全链路未见患者泛化。

所有科学终态均停止。不用 O_NO_AREA、O_FREE_SUBSET、evaluable-only 敏感性或事后新阈值救援主结论。

GT 开放前的合成修复允许重新冻结后重测；GT 开放后若评分、动作、选择、预算、样本或科学判定代码需要修改，应保留当前产物并停为 BLOCKED/INCOMPLETE，不现场改代码重判。只允许在源码、输入、协议未变且无终态时，以验证块校验的方式幂等恢复纯执行中断。SSH 退出码不冒充服务器计算子进程退出码。

## 8. 发布和最终回复

在 `docs/care_hr_v0_7_1/continuation_r1/` 至少发布：

- `SCORING_RESOLUTION.md`、`SCORING_CONTRACT_R1.json`；
- `SEMANTIC_PARITY_REPORT.json`、完整 A1 coverage 与本地/服务器测试摘要；
- `R1_PREREGISTRATION.json`、输入/模型/route/动作密封的公开哈希清单；
- `CAPACITY_FINAL_REPORT.md`、`CAPACITY_STATUS.json`；
- `CAPACITY_METRICS.csv`、`BUDGET_ATTRIBUTION.csv`、`EVALUATION_COVERAGE.csv`、`EVALUABLE_ONLY_SENSITIVITY.csv`；
- 实际命令、服务器父进程退出收据摘要、runtime counters、历史保护和发布核验；
- `NEXT_STAGE_DRAFT.md`：只记录容量结果支持的后续问题，不执行下一阶段。

不上传原始图像、GT、概率、模型权重、私有患者标识、逐病例私有材料或服务器凭据。逐病例与 Pareto 原始数据保留私有，公开提供脱敏汇总和可核验哈希。

最终回复必须给出：
1. R1 的明确终态，旧 blocked 状态保持不变；
2. 新 source SHA、新 report SHA、branch、远端与匿名公开核验；
3. 新源码上的 tests passed/failed/errors/skipped 和逐项 A1 coverage；
4. 本轮真实 rows/patients、缓存、前向、GT/domain 读取和真实拟合计数；
5. 四类 oracle 真正测得的结果，或具体未执行阻塞；
6. 全 ignore 数据实际是否存在及处理记录，不从合成见证推断真实发生率；
7. 不修改旧锁、不改历史、不合 main、不启动风险拟合的核验证据。

请执行到本轮完整结果及停止点。不要要求用户再次确认本指令已明确的 ignore=255、空类别、全 ignore、macro foreground 规则；也不要仅因发现既有 V0.7 仍不一致而再次阻塞。
