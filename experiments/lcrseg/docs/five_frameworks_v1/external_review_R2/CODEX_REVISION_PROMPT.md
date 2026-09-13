# 给 Codex：R2 小范围修订，修复后推送并停止

请先读同包 REVIEW_REPORT.md。固定基础提交是
`bac21b6585ec284fb6e93f213e280c1492f6e57a`，同一分支
`codex/sslcl-five-frameworks-v1-review`。

## 本轮决定与边界

R01–R06 的修订已被本次外部审阅接受。不要退回重做六个问题，不要再改五个框架的数学定义、损失组合、梯度权限、候选网格或 212/424 计划上限。

本轮唯一必改问题为 **R07：integration.py 中校验后的 manifest 与 ExecutionPermit 共享外部可变对象**。状态保持 CODE_ONLY。没有真实实验、真实图像/标签/checkpoint tensor 读取、真实 smoke 或 GPU 运行授权。

不得将本报告当成 APPROVED_FOR_EXPERIMENTS。不得修改审批文件伪造用户运行确认。

## 执行任务

### A. 修复清单和许可的快照语义

- CurrentDomainDataAdapter 在构造时持有独立、稳定的清单与行记录；digest、校验和以后 reader 使用同一快照。
- 调用方修改原 manifest、L/U 列表和 nested row，reader 要么仍使用批准值，要么在任何 reader callback 前拒绝。
- ExecutionPermit 将 bindings、authorized_manifest_digests、budget、phases 深复制并冻结；不能因传入的 dict/list 后来被修改而扩大允许清单或预算。
- 不仅复制顶层 dict。不得继续通过公共属性暴露可修改的内部 list/dict；返回只读视图或安全副本。
- 保持私有 seal 对象身份，不做会无意复制 seal 的整体 deepcopy。
- 无需每个样本都重新 hash 整份清单；优先明确的数据所有权和不可变记录，不增加复杂安全框架。
- 对 NativeParentBridge 的嵌套 metadata/参数白名单做同类局部检查，必要时统一快照函数，但不要扩展算法范围。

### B. 回归

把 test_external_review_R2.py 放入现有测试目录。仓库回归必须使用真实模块 import：**不要设置 SSLCL_R2_SNAPSHOT**。该环境变量仅供外部审阅者在受限沙箱中验证固定源文件 helper。

修订前外部局部回归记录是 7 通过、4 失败；四个失败都是 R07 的不同别名修改路径，不是你原 162 项中的失败。无需再次重跑旧 R1 的整个失败版本，也无需重训任何历史实验。

必须覆盖：
- L 和 U 输入原字典修改后不改变读取目标；包括 nested row 和列表的替换。
- 修改构造 permit 时传入的 allowed-digest 列表，不增加许可。
- 修改构造 permit 时传入的预算字典，不增加实际 cap。
- 通过对外属性/返回值不能意外改写内部校验快照。
- 合法、不修改的 manifest、许可和 reader 路径仍工作。
- 当前无注册真实 runner、手改审批 JSON 不能启动真实工作。

新增测试中使用私有 seal 只能与已有 unit-test fixture 一样在测试内构造；不能导出为用户审批文件或接给真实启动器。

修订完成后只进行必要 CPU 合成回归，记录实际范围/调用成本，复用已有环境。保留原 162 项覆盖，不以删掉负例、改成 skip 或放松不相关断言过关。新增接口使原测试需要调整时，应解释具体理由。合成成本与真实/科学成本分开报告，不能把最后一次资格数冒充所有调试总成本。

### C. 父接入与运行计划

本轮不要求等待真实父入口才修 R07。若工作区/用户已经给出确定的源码、配置或原始入口，可以补来源的 metadata-only 绑定材料；没有就保留 PARENT_BINDING_REQUIRED，列出原缺项，不再做无界历史搜索。

不猜 A/B 顺序、约束形式和源模型来源；不拿 F_CONV 或源 SVD 模型代替原 KI。不能通过改 synthetic 标志、注册一个占位 runner 或改 JSON 来宣布真实集成完成。

CUDA、真实 L smoke、源训练与 C/D/E 都不在本轮授权范围。

### D. 推送交付

同分支新增提交，不 amend/force-push，不合并 main。不上传数据、权重、患者身份或凭据。

更新 REVIEW_RESPONSE_R2.md、实际测试日志、CODE_MANIFEST、未审批 REVIEW_LOCK，以及必要的所有权/数据访问说明。历史 R1/R2 报告保留；不要覆盖原有决策解释。

回传：

```text
status = STOP_AWAITING_EXTERNAL_CODE_REVIEW
base_commit = bac21b6585ec284fb6e93f213e280c1492f6e57a
review_commit = <actual>
branch = codex/sslcl-five-frameworks-v1-review
R01_R06 = CLOSED_PRESERVED
R07 = <actual outcome>
CPU_tests = <actual scope and result>
CUDA = NOT_RUN
real_data_reads = 0
real_checkpoint_tensor_reads = 0
real_smoke_updates = 0
formal_optimizer_updates = 0
parent_binding = <actual>
real_runner_registered = false
plan = 212 trajectories / 424 target stages upper bound, unchanged
review_index = <exact path>
push_verified = <actual>
```

推送后停止。该修订不改变“合理调参、不要求所有设置最优”的实验原则。
