# F5_CONFIRMATION_V1：外部代码审阅 R1

## 结论

**CHANGES_REQUESTED**。审阅对象为 `4ac9a27b52a9d381dce6bdfaf96bba23b17245f8`，不批准执行实验；保持 `STOP_AWAITING_EXTERNAL_CODE_REVIEW`。

科学矩阵和冻结配方不需要改变。问题集中于生产完整性验收、原生资格测试覆盖、成本采集及交付闭环。

## 审阅范围与限制

经GitHub连接器读取并核对新包的protocol/execution/tests/CLI、REVIEW_REQUEST、RUNBOOK、TEST_REPORT，以及复用的integration/checkpoint/native_state/reliability/native_runner路径。重点核对原生target_task调用链、封存/恢复、授权和报告。

没有在用户服务器执行命令，没有读取患者或真实权重，没有执行CUDA/真实smoke/训练，也没有重新运行仓库完整9组CPU测试。已发表TEST_REPORT记录9组PASS、累计15次合成optimizer调用；这是仓库证据而非本地复跑结果。

本地独立执行的是从所读代码抽取的3个标准库元数据函数、4个定向fixture；真实与合成optimizer调用均为0。抽取函数测试不等于原生训练验收。容器未能取得完整仓库进行全套运行。

## 核对一致的内容

- matrix()按B0/F5 seed163/164及B2 seed162/163/164生成14轨迹、28目标阶段、74200更新；历史导入与source不作为新训练节点。
- bound_options、options_for检查冻结完整配置；B2_C06阈值0.6/0.7、F5_C02阈值0.7/0.5、父A/B与R学习率区别均保留。
- F5嵌套损失、R-only直接U梯度、Final/Old/Incoming/Forget、seed内先平均两顺序与G1–G3仍符合先前计划。
- 新review/user-launch门控、零新增source和有限矩阵、lost-tail拒绝逻辑方向正确。源张量/CUDA/smoke正确标为待执行。

## R1｜高优先级：封存模型没有完整性验收

`execution.stage_state`在receipt字段和调用数满足条件后，只要求student.pt存在且非空；不读取payload、不核对student_hash/transform_hash。`run`的SEALED_SKIP和`report`的完成判定复用这一状态。原target_task在后继入口检查前驱权重，不能覆盖没有后继的末阶段文件，也不能替代整个完成验收。

本地反例：student.pt内容为`NOT A MODEL CHECKPOINT`，receipt甚至没有两个hash字段，stage_state仍返回SEALED。这不证明服务器已有模型损坏，但明确证明当前检查无法识别这种状态。

要求将元数据封存与受授权的真实模型完整性检查分层，实现实际文件hash/schema/identity/finite/transform检查，以及可审计的完成证明。纯metadata report保留不读张量，但无证明时不写验证完成。

## R2｜高优先级：资格测试缺少原生恢复与阶段切换

`qualify(cuda)`仅对B0/B2/F5各连续update四次。不存在native_state.resume、checkpoint等价继续、合并前后全模型预测对照、真实第二阶段构建路径。CPU测试的merge_current_ema_resume使用SyntheticParentBridge，不是实际14卷积原生桥接。

`require_qualification`只检查顶层PASS/commit/plan/12或24条账本；未验证逐用例集合或结果。定向元数据fixture中，rows=[]仍被它接受。fixture没有创建生产审批或permit，不说明实际有人伪造资格报告。

要求编写上述原生合成CUDA有限用例（当前不执行）、将用例和预期调用数写入plan且保持<=60，逐项核验；同时绑定资格与正式执行运行环境。不能仅通过扩大循环次数解决。

## R3｜中优先级、启动前补齐：成本采集在新控制器中丢失

原NativeRunner.worker在target_task外使用NativeOperations并采集CUDA峰值。新execution.run直接调用target_task，没有这层采集；report也未导出这些字段。因此训练器telemetry尚在，但operation_counts及阶段显存峰值不在新记录中。恢复后的耗时还应按所有会话累计，而不是只看最终会话。

应复用只读采集，保证不改变数学/RNG、不增加训练forward/VJP。成本属于必须从首次执行保留的信息，不能在实验结束后从optimizer步数推算。

## R4｜中优先级：交付接口与约定不符

execution.report当前只写PUBLIC_RESULTS.json；没有实现先前约定的FINAL_REPORT.md、FINAL_METRICS.csv、STAGE_METRICS.csv、PAIRED_COMPARISONS.csv、COST_AND_COMPLETION.json。run最后写通用COMPLETE，而非完成后待科研审阅的状态，也未形成明确完整性证明。

保留JSON为中间产物，补上报告导出、historical_import/new_execution标记、schema测试，以及完整通过后的COMPLETE_P1_AWAITING_SCIENTIFIC_REVIEW。公开发布与本地完成分开，不声称未实际执行的发布。

## 定向反例结果

| 用例 | 观察 |
|---|---|
| 无效模型文件且缺少声明hash | 被判SEALED；确认R1 |
| optimizer ledger领先checkpoint一步 | 正确拒绝lost-tail |
| invocation断号 | 正确拒绝损坏账本 |
| 空资格用例rows + 表层PASS/commit/plan/计数 | 被require_qualification接受；确认缺少用例级检查 |

## 处置

只做R1–R4修补，不改科学矩阵、冻结参数和预算，不启动任何生产资格或实验。新提交需重新绑定审阅；本记录不能作为APPROVED_FOR_EXPERIMENTS使用。

## 主要证据

所有代码证据固定在被审SHA，定位使用函数名：

- [execution.py](https://github.com/DLwbm123/SSL_CL_seg/blob/4ac9a27b52a9d381dce6bdfaf96bba23b17245f8/experiments/lcrseg/f5_confirmation_v1/execution.py)：stage_state、qualify、require_qualification、run、report、preflight。
- [protocol.py](https://github.com/DLwbm123/SSL_CL_seg/blob/4ac9a27b52a9d381dce6bdfaf96bba23b17245f8/experiments/lcrseg/f5_confirmation_v1/protocol.py)：matrix、bound_options、metrics、paired、decisions。
- [tests.py](https://github.com/DLwbm123/SSL_CL_seg/blob/4ac9a27b52a9d381dce6bdfaf96bba23b17245f8/experiments/lcrseg/f5_confirmation_v1/tests.py)：trainer、merge_current_ema_resume、failure_budget_sealed。
- [native_runner.py](https://github.com/DLwbm123/SSL_CL_seg/blob/4ac9a27b52a9d381dce6bdfaf96bba23b17245f8/experiments/lcrseg/five_frameworks_v1/native_runner.py)：NativeRunner.worker的外层采集与target_task的实际训练。
- [TEST_REPORT.json](https://github.com/DLwbm123/SSL_CL_seg/blob/4ac9a27b52a9d381dce6bdfaf96bba23b17245f8/experiments/lcrseg/docs/f5_confirmation_v1/TEST_REPORT.json)：已发表CPU结果与明确的生产限制。
- [RUNBOOK.md](https://github.com/DLwbm123/SSL_CL_seg/blob/4ac9a27b52a9d381dce6bdfaf96bba23b17245f8/experiments/lcrseg/docs/f5_confirmation_v1/RUNBOOK.md)：当前入口和待执行资格说明。
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)：不同版本/平台及CPU/GPU之间不保证逐bit一致；原生恢复比较需绑定实际受控运行环境。
