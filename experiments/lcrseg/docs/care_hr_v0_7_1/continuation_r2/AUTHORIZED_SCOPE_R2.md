# SSL_CL_seg — CARe-HR V0.7.1 R2：冻结动作的 evaluator 路径修复与完整评分

## 给 Codex 的执行指令

继续处理 `DLwbm123/SSL_CL_seg`。本指令由用户转交执行时，授权一个新的、范围受限的 **R2 engineering continuation**：修正 R1 evaluator 的数据资源绑定，补上独立的部署布局回归，在新源码、新输出目录和新访问 reservation 下，对同一批已密封动作完成基线和四类 oracle 评分。

**不要只提交下一份草案。前置条件通过后，执行到本次真实容量报告并停止。** 不需要再确认已经裁定的评分规则、既定数据范围或本指令明确授权的步骤。真实缺少权限、输入校验失败或未授权变更则按相应状态停止，不能编造证据。

这不是把 R1 改判为成功，不是删除 reservation 后重跑旧 attempt，不是新增方法、重新选择策略或拟合风险模型。原 R1 终态 `INCOMPLETE_EVALUATION` 和此前的 `BLOCKED_EVALUATOR_SEMANTICS_MISMATCH` 永久保留。

---

## 1. 固定起点与本次范围

- 仓库：`DLwbm123/SSL_CL_seg`
- 分支：`codex/care-hr-v0-7-1-capacity-audit`
- 起点／R1 report commit：`3505ffdfdeddb75d55e6fc982b78e20dafe6f816`
- R1 被测源码及动作生成源码：`9bbbacd25f3c3abf885205009eb14d698b36f32b`
- R1 A1 publication：`13ccd8f46392a144862146b2423c1286e8d124cf`
- R1 盲密封 publication：`d4d8ca318a8eb0631b5cc3146b7e50f1b3ba1af3`
- R1 ACTION_SPACE_SEAL 文件 SHA-256：`9b186f8c4e80236987700fcb74cf90a5b41071b63df128a5b1e28142df7141a4`
- 实现与测试仍在 `experiments/lcrseg/care_hr_v0_7_1/`、`experiments/lcrseg/tests/care_hr_v0_7_1/`。
- 本轮新增公开协议、证据、报告仅写入 `experiments/lcrseg/docs/care_hr_v0_7_1/continuation_r2/`。
- 采用新的私有 NAS run_id/output root；不得写入 R1 run root。

先读取根及 experiments/lcrseg 的 AGENTS、本轮前驱报告、EXECUTION_AUDIT、R1 预注册及评分 contract、旧锁、原输入 expectations、R1 action seal 和既有附件采用记录。不要从 main 开始；不要混用已发表报告 HEAD 与实际执行源码 SHA。

使用隔离 worktree。核验给定提交及当前分支；不 reset/clean 用户工作区，不覆盖未提交修改，不强推。如远端已增加提交，报告差异并证明其与给定起点及授权范围的关系，不静默替换实验版本。

保护原 99 个文件，并扩展到 R1 已发布源码、测试、协议和报告。原则上只新增 R2 adapter/executor/qualification/tests；直接复用 R1 的科学纯函数，旧 `execute_r1.py` 也保持原样作为失败实现的证据。可采用 `io_r2.py`、`execute_r2.py`、`run_checks_r2.py`、`parent_r2.py` 等新文件，不要求照抄文件名，但不得修改旧科学逻辑。

本次绝对禁止：
- 修改 scoring_r1、oracle_r1、summary_r1、actions、blind_r1 等已冻结科学逻辑，修改阈值、聚合、tie-break、proposal 或预算；
- 根据 R1 已枚举数量，增加新预算、重新选历史专家、换样本或增加 oracle；
- 新模型前向、模型构造、分割训练、optimizer/backward/EMA/GAS/prototype 更新；
- 真实 Ridge/PAV/temperature/router/risk-head/conformal 拟合；
- 读取旧 V0.4 formal_03、own-seed val/test 或 hidden-unlabeled GT；
- 修改旧 REVIEW_LOCK、旧状态、旧 reservation 或合并 main。

R1 已有可验证概率缓存，本次新增 sample-expert forward 的允许数量为 **0**。不重新加载 checkpoint tensor 只为重复成本统计；引用 R1 参数量/缓存成本时标明历史观测，不冒充新测量。若冻结缓存损坏/缺失，停止，不用重新推理作为替代。

## 2. 接受 R1 事实，不重置暴露历史

R1 的公开及私有收据需逐项核验：
- 161 passed / 0 failed/errors/skipped，A1 的 13 项覆盖通过，是旧源码证据；不是 R2 自动通过。
- 198 seed-case 行，177 名患者，各 seed 66 行；9 个概率缓存；594 个历史 sample-expert 输出；新前向为 0。
- 首次 GT payload open 尝试 1，成功读取/解码 0；seed0 的 66 行真实域 metadata 已 materialize，循环处理 1 行。
- 完成基线比较 0、完成 oracle 行 0；真实 all-ignore 情况未知。
- evaluator 父进程实际 child exit=1；这是已有终态，不能按未终止进程 resume。

新报告必须写出 R1 和 R2 各自的暴露计数。R2 可以记录本次新访问计数从零开始，但不能写“整个研究从未读取真实域身份”。66 行 metadata 已暴露是事实；由于动作空间及评分规则不变，本次是记录该暴露后的工程续接，不是新的独立盲测试或外部确认。

## 3. 唯一的数据路径裁定与修复范围

R1 错误位置：

```python
payload = safe_path(DATA, row['label_h5_relpath']).read_bytes()
```

GT asset 的正确绑定已经存在：

```python
from di_dmpa_gate1.binding import safe_asset
asset = safe_asset(DATA, row['label_h5_relpath'])
# DATA / 'h5/v1' / relative
```

### 3.1 不要全局替换 generic safe_path

`safe_path` 对协议目录中的 npy/npz/json 等相对路径仍是另一种合法约定。不要给所有路径自动添加 h5/v1；也不要把 DATA 全局改为 DATA/h5/v1，导致 manifests/training 路径变成错误路径。

新增专用 dataset-asset adapter，直接调用冻结 safe_asset，或使用一层极薄的类型/包含性检查包装。数据集 assets、training manifests、sealed protocol artifacts、run outputs 必须显式区分，禁止混用 resolver。

拒绝绝对 relative、`..`、空资源名及越界解析。对合法继承的 symlink data root，先解析注册的 h5/v1 root，再校验目标位于该 root 内；不要仅做字符串前缀比较。不得移动数据、创建兼容 symlink、扫描磁盘找同名文件，或在 DATA/relative 与 DATA/h5/v1/relative 之间按哪个存在来 fallback。输入地址由冻结 manifest 与既有 binding 唯一确定。

对本轮实际打开的 GT，保持“读原文件 bytes -> 与 row['label_sha256'] 比较 -> h5py 以只读方式从相同 bytes 解码 label”的顺序。stat 成功不等于 hash 正确、HDF5 有效或标签已解码。文件存在性检查后仍必须进行 payload hash、dataset key、shape、dtype、标签取值校验；不以全 ignore 占位代替读取失败。

### 3.2 允许改动的其他部分

仅允许为本次恢复新增／改动新 adapter 中的：
- R1 sealed input root 与 R2 output root 的分离；
- 新旧源码身份验证与旧 seal 引用；
- 路径与 HDF5 加载前置检查；
- 实际访问计数和失败记录；
- 将同一批正确解码 GT 交给原评分、原基线比较、原 oracle、原 summary/terminal；
- 新协议、回归、发布收据与输出位置。

若必须复制 R1 evaluator 的 orchestration 到 execute_r2，新旧函数 diff 逐段分类；不得借复制改变任何科学运算顺序。禁止 monkeypatch 全局 safe_path 去偷偷改变所有输入的绑定，禁止修改已提交的 execute_r1 来消除失败证据。

写 `SCOPED_DIFF_AUDIT.json`：每个新文件、复用的科学函数及 SHA，允许变化的调用点，不变的阈值/排序/聚合，以及无科学变更的证明。发现真实科学 bug 不在本次修复权限内时，如实报告，不用“路径恢复”名义顺带修算法。

## 4. 独立部署布局回归：必须能识别 R1 原错

仅有 resolver 单测不够。补全生产 loader 到 evaluator 的集成测试。夹具布局必须由测试规范独立构造，不能调用被测 resolver 来决定应该把 fixture 放在哪里。

### 4.1 必须新增的见证

1. **canonical-only 成功**：只在 `fixture_root/h5/v1/<rel>` 创建合成 HDF5；manifest 保存 `<rel>`，不是带 h5/v1 的绝对/完整路径。实际 R2 loader 必须成功读取且标签、hash 正确。
2. **wrong-root-only 拒绝**：只在 `fixture_root/<rel>` 建立文件。真实 R2 adapter 不得 fallback 到它。
3. **双路径诱饵**：两处都存在、内容不同，manifest hash 绑定 canonical 文件。必须打开 canonical；修改或删除诱饵不能改变结果。
4. **旧错误被检出**：在独立 canonical-only fixture 上，明确执行 R1 的旧读取表达式，观察 FileNotFoundError；然后相同 fixture 经 R2 生产 loader 成功。单独记录 expected-failure witness，不把它伪装成 R2 pytest failure/skip。
5. **多根约定并存**：`DATA/manifests/training` 和 `protocol_root/candidate_*` 不得被加 h5/v1；只 dataset assets 使用继承 binding。
6. **路径安全**：绝对 relative、`..`、目录冒充文件、越界 symlink 拒绝；合法已注册 data-root symlink 可以解析。fixture 仅使用合成文件。
7. **负向数据测试**：缺失、hash 不匹配、不可读文件/模拟 read 错误、缺少 label key、错误 shape/dtype、非法 label、损坏 HDF5。全 ignore 必须是合法解码的 GT，而不是异常 fallback。
8. **计数反例**：失败 open 应只增加 attempts，不增加成功读取/解码；hash mismatch 不增加 hash_verified/decode_success；66 行 materialization 即使循环只走 1 行也要记录 66，而非 1。模拟异常必须检查父进程真实 exit 和终态。
9. **生产通路端到端**：复用实际 loader、reservation/seal guards、原评分函数、原 baseline parity、原 oracle、原输出序列化，完成生成 HDF5 的成功运行及故障运行。不得 monkeypatch 掉 path resolution 或 HDF5 decode 来声称 E2E 通过；不得放宽正式 198/177、shape、hash 门槛来让真实运行通过。小规模合成注入只作为明确的测试接口，不能成为 CLI 的任意真实人群参数。
10. **seal 只读测试**：R2 写路径只允许新 output root；原 R1 action/input 文件、旧 reservation/terminal 保持字节不变。新旧 source 不同必须通过双来源新封装验证，而不是篡改旧 seal。

完整执行既有 161 项回归（如测试收集结构有变化需逐项对应，不减少覆盖）和新增项；本地与服务器都在 R2 精确源码上执行。测试总数按实际报告，不预填“必须 N passed”。保留 13 项 A1 覆盖，不把数量增加等同于部署通过。

可以在纯合成工程阶段修复、重新冻结并重测，不必每次等待用户确认。最终新源码 freeze、本地/服务器 tests 和 independent-layout witness 全部通过后，才能进入真实 R2 流程。

## 5. 复用原动作内容，创建 R2 新密封，不重新生成方法决策

### 5.1 只读复核 R1 成果

从允许的 R1 私有输出与原 PPC formal_02 血缘读取精确列举文件，不扫描 NAS，不接触 formal_03。核验：
- 上述 parent ACTION_SPACE_SEAL SHA-256；其所有 sealed files 与 original_files 逐项哈希；
- 原 9 份概率缓存、行序、patient mapping、C6 route、冻结 controls；
- 198/177/66 per seed、C6 历史 155 与当前 43；
- R1 proposals=949、O_CAP actions=15796、O_NO_AREA=26374、O_FREE_SUBSET=159232；
- 最大每例 strict/free action 数为 597/8191；严格 no-op-only 67，free no-op-only 43；zero-current-foreground 0。

这些是**要重新验证的 R1 历史观测值**，不是本轮新前向或新 oracle 结果。任何一个数据内容/identity 不匹配，输出 `BLOCKED_PARENT_SEAL_OR_INPUT_MISMATCH`；不要重新枚举、改 expert、补样或重新推理来生成一份“能过”的 seal。

默认直接读取原来密封的 prepared arrays/actions。为了输出封装允许字节一致的只读副本或内容地址引用；不要求重算全部动作，更不重做统计拟合。重新检查冻结公式或统计数量只用于 parity，不能生成新的候选设计。

### 5.2 双来源新 seal

R1 evaluate 原来要求 `seal['source_commit'] == source_state()`。R2 的源码不同，不能通过把旧 seal.source_commit 改成新源码绕过。

在 R2 新输出目录写新的 `ACTION_SPACE_SEAL.json` 或等价的新封装，至少绑定：

```text
schema_version = explicit_R2_version
action_generation_source_commit = 9bbbacd25f3c3abf885205009eb14d698b36f32b
evaluator_source_commit = actual_R2_source_SHA
parent_action_space_seal_sha256 = 9b186f8c4e80236987700fcb74cf90a5b41071b63df128a5b1e28142df7141a4
parent_action_content_manifest = original exact file map
new_protocol_sha256 = R2 protocol hash
scoring_contract_sha256 = unchanged R1 contract hash
scientific_kernel_hashes = unchanged modules/functions
new_A1_and_layout_qualification = exact-source evidence hashes
prior_exposure = R1 factual access counts
new_model_forwards = 0
new_real_fits = 0
```

新 seal 的来源/授权封装可以变化，**动作、概率、预测、路由、病例以及科学定义的内容身份必须不变**。分开记录 `sealed_input_root` 和 `run_output_root`；旧目录只能读，新的 reservation/public/private_evaluation/counters 均写在 R2 根下。不得用复制旧 reservation 或覆盖旧输出伪装新 attempt。

## 6. R2 分阶段执行顺序

### E0：新源码与工程证据

登记 `R2_RECOVERY_PREREGISTRATION.json` 和本指令采用记录，明确只修路径、访问边界和计数。发布新 source freeze，确认远端可达；服务器干净 checkout 固定该源码。发布本地/服务器新 A1 与独立布局证据，区分 evidence commit 和 tested source commit。

报告提交推进 branch 不意味着运行 checkout 的源码可切换；不同证据提交中的 source 字段必须仍指向实际被测源码。

### E1：父密封核验和新引用密封

按第 5 节完成 R1 内容校验和 R2 双来源 seal，发布脱敏的 `ACTION_SPACE_REUSE_QUALIFICATION.json`。通过后才开放限定的 evaluator metadata preflight。

### E2：全部 198 行的 stat-only 数据绑定 preflight

使用独立 loader-only 进程，从冻结 training manifests 只投影选中的 `(seed, case_id, patient_id, primary_20pct_split, label_h5_relpath, label_sha256)` 等必需字段。严格按 own-seed train_labeled 和 parent sealed row identity 检查。不要在此投影 `site_or_vendor`、真实域字段或逐病例历史分数。

通过同一个 R2 生产 asset adapter，为所有 198 行解析地址；做 canonical-root containment、is_file/stat、size>0 等 metadata 检查，固定顺序，记录重复物理资产与 patient/seed 的映射。所有 metadata 访问单独计数，不称为“零真实输入读取”。

这里不能 `read_bytes`、`digest(label_file)`、打开 h5py label 或读取像素；GT 文件全字节 hash 验证属于正式 payload 访问，不是 stat-only。路径名可能包含来源线索，只允许 loader 使用，不传回动作生成器；不得据其选择或剔除病例。

输出私有 `RESOLVED_ASSET_ALLOWLIST.json`，公开只给汇总和内容哈希。另写 create-only `ASSET_BINDING_ADMISSION.json`，同时绑定既已发布的 R2 seal hash、allowlist hash 和 preflight 结果；原 parent seal 与已发布的 R2 seal 都不改。正式 reservation 再引用这份 admission。预检如果有任何 missing/permission/unsafe mismatch，在 GT payload 开放前一次性汇总允许范围内的异常并停止；不要轮流猜路径或寻替代文件。全部 stat 通过仍不能宣称 label 内容已经校验。

本指令已允许这种“动作已密封后、GT payload 开放前”的限定路径 metadata 预检，无需再为同一个既有 h5/v1 绑定等待确认。

### E3：一次新的完整 evaluator

E0–E2 全部通过，原子创建新的 R2 `EVALUATOR_ACCESS_RESERVATION.json`。在新进程中重新核对源码、parent seal、新 seal、允许 metadata、输入哈希，并打开限定 GT/真实域字段。

从第一个固定病例开始：正确原文件读取 -> sha256 -> HDF5 label decode -> shape/dtype/value 校验 -> 传给原评分接口。第一例同时承担正式加载 smoke 的作用，不另开一个“看分数后修代码”的真实 pilot。成功后不等待人工确认，按原固定顺序继续全部 198 行；每个输入的内容验证均不可省略。

保留原 R1 的逐病例和完整分组 baseline parity：主对照 current、Ridge hard、SHOR、PPC C6；沿用已冻结 `rtol=0, atol=1e-12`，不根据观察误差放宽。应完成 198*4=792 个 case-policy 比较；若每个比较包含 4 个分数字段，单独报 3168 个 scalar comparisons，不再混用这两个计数单位。原 C7 仅按已有权限用于诊断 oracle gap，不改为新可部署对照。

在所有必要基线一致性条件通过后执行原四类 oracle 的固定 lambda 0.50/0.75 与 envelope，输出原汇总、预算归因、覆盖、各敏感性结果与 Pareto 私有文件。当前执行的 scientific functions、动作顺序、tie-break、浮点求值和门槛不能修改。

整个过程新前向=0、真实拟合=0；通常无需 GPU，执行 CPU evaluator，不因用户曾授权 GPU4/5 就占用 GPU或终止他人任务。使用既有可用服务器与 NAS，执行时复核允许的路径，不把历史可用性当作当前保证。

### E4：报告并停止

四类 oracle 完成后按原 terminal 判定，发布 R2 报告及来源证据并停止。不启动风险拟合、下游训练或外部测试。

若正式 evaluator 报错，保留新 reservation、全部原始输出和真实父进程退出码，记录 `INCOMPLETE_EVALUATION` 或具体基线/输入阻塞。本轮不允许看过真实 GT/domain 后现场修源码、改人群或重新判定。明确的 R1 终态不能被删除；R2 若形成终态也不能通过换目录静默重跑。

## 7. 评分、oracle 与科学判断全部沿用 R1

R1 scoring contract 的 ignore=255、空类别 Dice=1、全 ignore 兼容分数=1/gain=0 并标记无证据、macro foreground 和 union foreground 分开、背景包含在 mean_iou 的约定保持。

GT valid-mask 只能影响 evaluator 的计分，不能影响 proposal、面积预算或路由。全 ignore 行保留主兼容聚合权重，evaluable-only 敏感性单列；没有有效标注支持的完整 seed/domain 分组按原规则 blocked，不捏造成功，也不更换病例。

四类 oracle 的 action sets 与 SAFE0 逐类非下降保持。只允许原 CAP、NO_AREA、FREE_SUBSET 的嵌套归因；lambda=1 不能替换原空间，不能把放宽预算的最优结果升级为主方法。

权限、来源、基线、完整评价及标注支持都通过之后：
- O_CAP envelope 的整体 gain<0.17 或历史 gain<0.27：`FAIL_FROZEN_ACTION_SPACE_CAPACITY`。
- 否则若 O_SAFE0 envelope 整体 gain>=0.17、历史 gain>=0.27，两个历史域各自增益>0，三个 seed 整体各自增益>0：`PASS_ACTION_SPACE_SCREEN_ONLY`。
- 其余：`CAPACITY_PRESENT_SAFETY_UNRESOLVED`。

这些仍是同一冻结开发人群上的容量筛查。不得称未见患者泛化、独立外部测试或完整 CARe-HR 成功。原完整草案的其他门槛逐项保留；不重拟合就不填写 bootstrap/refit stability PASS。

## 8. 必须修正访问计数，保留旧 raw counter

R2 从实际 I/O 边界记录，而不是先加一个含糊 GT_reads 再推断成功。至少分开：

```text
asset_metadata_rows_checked
asset_unique_files_stat_checked
GT_payload_open_attempts
GT_payload_open_successes
GT_payload_reads_completed
GT_file_hash_checks_passed
GT_label_decode_attempts
GT_label_decodes_completed
GT_rows_bound_to_validated_labels
GT_unique_files_decoded
true_domain_records_materialized
true_domain_rows_processed
baseline_case_policy_comparisons_completed
baseline_scalar_comparisons_completed
oracle_case_rows_completed
new_sample_expert_forwards
reused_sample_expert_outputs
real_fits_by_kind
```

字段定义写入 schema。若跨 seed 重复使用同一已解码原文件，区分物理文件解码与 seed-row 使用，不人为把一次读写成多次；策略必须事先固定且 own-seed 权限逐行验证。

记录 case identity 前不得声称该行已经 scored。status 中分开 cohort_expected、metadata_revealed、GT_validated、baseline_completed、oracle_completed。R1 的 GT_reads=1、domain_reads=1 原文件不改，仅在新 lineage 中引用其解释。旧66域行暴露与R2新增访问分开，不静默归零。

在异常和 finally 中也可靠落盘 counters 与父进程 receipt；不能把 SSH 返回码替代实际 evaluator child returncode。原始异常可能含患者ID/路径，只存私有 NAS；公开日志仅含错误类型、脱敏原因、事件阶段和私有traceback hash。

## 9. 交付文件与最终回复

`continuation_r2/` 至少交付：
- `R2_RECOVERY_PREREGISTRATION.json` 与采用说明；
- `SCOPED_DIFF_AUDIT.json`、科学核与所有保护文件 SHA 核验；
- `DEPLOYMENT_LAYOUT_REGRESSION.json`：canonical/decoy/旧错见证/E2E/负向数据/计数覆盖；
- R2 精确 source 上的本地与服务器完整测试结果、实际命令、父进程收据摘要；
- `ACTION_SPACE_REUSE_QUALIFICATION.json`、R2 新封装 seal 的公开摘要；
- `ASSET_BINDING_PREFLIGHT.json`：198行预检覆盖、异常数、metadata与payload计数区分；
- `EXECUTION_AUDIT.json`、新的 runtime counters 与 parent child exit；
- `BASELINE_PARITY.json`；
- 原要求的 `CAPACITY_METRICS.csv`、`BUDGET_ATTRIBUTION.csv`、`ORACLE_CAPACITY_ATTRIBUTION.csv`、`EVALUATION_COVERAGE.csv`、各敏感性 CSV 和 `DRAFT_GATE_ACCOUNTING.csv`；
- `CAPACITY_STATUS.json`、`CAPACITY_FINAL_REPORT.md`；
- 有且仅有结果支持的 `NEXT_STAGE_DRAFT.md`，不执行下一阶段。

结果未得到必须填明确 NOT_EVALUATED 或 UNKNOWN，不伪造零。原始图像、GT、模型、概率、病例标识、逐例动作/分数/Pareto和凭据不公开；保留私有清单与哈希。GitHub仅发布脱敏汇总和可核验证据。

最终回复说明：
1. R2 明确终态及原因；R1终态保持不变。
2. 新 source SHA、evidence/report SHA、分支，远端及匿名访问核验。
3. 精确源码上的测试与独立部署布局见证，不只报 passed 总数。
4. parent action seal 与 R2 evaluator source 的双来源关系，无动作变化的核验。
5. 198行真实路径预检与完整评分进度；尝试/成功/解码/域materialization计数；新前向/真实拟合为零。
6. current/Ridge/SHOR/PPC 与四类 oracle 真正得到的指标及原门槛判断；若中断则说明未完成项。
7. 历史文件、R1数据及旧锁未修改，无main合并，无后续拟合。

执行到以上边界。不要为了“更谨慎”只发新草案又等待同一授权；也不要为了“尽快出结果”省去独立路径测试、输入hash或baseline parity。

---

## 公开核验依据（固定提交）

```text
https://github.com/DLwbm123/SSL_CL_seg/blob/3505ffdfdeddb75d55e6fc982b78e20dafe6f816/experiments/lcrseg/docs/care_hr_v0_7_1/continuation_r1/CAPACITY_FINAL_REPORT.md
https://github.com/DLwbm123/SSL_CL_seg/blob/3505ffdfdeddb75d55e6fc982b78e20dafe6f816/experiments/lcrseg/docs/care_hr_v0_7_1/continuation_r1/EXECUTION_AUDIT.json
https://github.com/DLwbm123/SSL_CL_seg/blob/3505ffdfdeddb75d55e6fc982b78e20dafe6f816/experiments/lcrseg/docs/care_hr_v0_7_1/continuation_r1/NEXT_STAGE_DRAFT.md
https://github.com/DLwbm123/SSL_CL_seg/blob/9bbbacd25f3c3abf885205009eb14d698b36f32b/experiments/lcrseg/di_dmpa_gate1/binding.py
https://github.com/DLwbm123/SSL_CL_seg/blob/9bbbacd25f3c3abf885205009eb14d698b36f32b/experiments/lcrseg/care_hr_v0_7_1/execute_r1.py
https://github.com/DLwbm123/SSL_CL_seg/blob/9bbbacd25f3c3abf885205009eb14d698b36f32b/experiments/lcrseg/care_hr_v0_7_1/io_r1.py
```

本提示词依据公开源码、报告与前序已裁定协议编写；制定者未进入服务器、未读取私有GT、未重跑R1/R2完整测试。实际工程资格由Codex按上述流程执行、记录并核验。
