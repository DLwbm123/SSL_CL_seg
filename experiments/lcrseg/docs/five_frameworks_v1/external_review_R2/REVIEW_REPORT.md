# SSLCL Five Frameworks — R2 external review

日期：2026-09-13  
审阅提交：`bac21b6585ec284fb6e93f213e280c1492f6e57a`  
前次提交：`b909a9131dc7137232ac8eb3a6922fe66520b932`  
分支：`codex/sslcl-five-frameworks-v1-review`

## 决定

**R01–R06：依据已审源码及本次提交的回归证据，接受这次修订。**

**整体：CHANGES_REQUESTED，限定为新增桥接问题 R07；不批准真实实验。**

这不是把原六项再次判为失败，不要求重新设计五个框架。保留既定候选、搜索网格、五框架全部复核、212 条轨迹／424 个目标阶段的计划上限，不增加“所有种子、类别、顺序均为正”的效益 gate。

新问题影响代码准备接入的真实数据边界；当前 REAL_RUNNERS 为空，未观察到真实数据越界或实验启动。修复应在注册真实 runner 前完成，可与父桥接的正常准备工作合并，不需重启历史审计。

## 1. 审阅范围与独立验证边界

通过 GitHub 连接器读取固定提交的 R1 回复、源码、依赖清单、回归代码和测试日志，并查看两提交的 compare 元数据。重点包括 checkpoint、semantics、train_stage、numerics、model、parent_bridge、integration、losses、evaluate 及 test_revision_r2。

仓库报告：162 passed，3 个由故障注入的 optimizer replacement 引起的 warnings；CUDA NOT_RUN。此为提交方运行记录，本审阅没有独立完整重跑这 162 项，不能把它写成外部审阅者自己的整套运行结果。

本地用连接器返回的原文重建了 5 个源文件：integration.py、evaluate.py、numerics.py、model.py、gate.py。5 个文件的 SHA256 均与该提交 CODE_MANIFEST.json 相同，详见 SOURCE_VERIFICATION.json。

随后对其中 gate/integration/evaluate/numerics 执行局部 AST helper 回归：只移除模块级相对 import；被测试函数和类的正文不改动。未调用依赖缺失的父模型、DAG 执行或真实启动入口。此为局部函数验证，不是完整生产包集成验证；model.py 在本地只做字节核验与静态审阅。

本地结果：**11 项检查，7 通过，4 失败**。四项失败均对应 R07 的可变引用问题。结果与完整日志分别见 LOCAL_TEST_REPORT.json、LOCAL_HELPER_TESTS.txt、LOCAL_HELPER_TESTS.xml。

本地环境：Python 3.13.5，PyTorch 2.10.0+cpu；不同于提交方的 Python 3.12.9、PyTorch 2.6.0。没有验证 CUDA、AMP、作者全部后端数值对照、真实主干或完整第二阶段独立进程恢复。

所有新 reader 都是记录字符串的测试回调，不打开图像、标签或 checkpoint。此轮没有真实数据、真实 checkpoint tensor、真实 smoke、正式更新或远端写操作。

## 2. R01–R06 关闭依据

| 原问题 | 本轮接受的修复 | 证据范围 |
|---|---|---|
| R01 第二阶段恢复丢失 A/B 训练权限 | configure_stage_training 与 stage_entry 分离；Model.for_resume 不重做 eigensolve；StageTrainer.for_resume 不重做探针或原型初始化，再加载状态 | model.py、parent_bridge.py、train_stage.py、checkpoint.py；提交的 10 个框架/基线独立进程恢复测试。完整轨迹结果依赖提交方测试，未在本地整套重跑 |
| R02 恢复没有绑定实际配置和数据来源 | semantics.binding 从实际对象形成配置、provider、入口参数、参数分组和模式指纹；restore 在加载状态前比较 | semantics.py、checkpoint.py；配置、seed/order/stage/source 等变更拒绝测试 |
| R03 非有限损失/更新状态被提交 | 检查活跃损失；validate_commit 检查优化器、有效矩阵、候选 EMA 和原型；异常后要求恢复，科学游标不继续 | train_stage.py、numerics.py；提交故障注入测试；本地独立复核有限梯度但无限 loss 和有限因子乘积溢出两个 helper 反例 |
| R04 F2 把非法投影当成支持不足 | 只捕获 InsufficientProbeSupport / RankDeficientProbe；所有候选 A 先验证后写入 | kernels.py、train_stage.py；非法 P/G 与无 A mutation 的新增回归；支持不足仍允许显式回退 |
| R05 教师路径误传 student mode | 模型 role 和 checked_mode 显式传递；teacher 原型、U 和修正路径使用 teacher；部署 eval | model.py、reliability.py、losses.py、train_stage.py；behaviorally-distinct bridge 测试。真实随机头仍待接入核验 |
| R06 ignore 评价 | Dice 使用 target!=255 与显式 mask 交集；全 ignore 无得分；partial-ignore 辅助距离/拓扑返回 NOT_EVALUABLE | evaluate.py、提交测试；本地 ignore 不变性、全 ignore、显式 mask、全有效图 4 个回归通过 |

F2 structure-only 方向探针保留；无需新增组合梯度搜索臂。CWMI CPU 来源对照和新增按设备构造仅按提交证据接受为代码准备工作；CUDA 未运行，不做 GPU 正确性声明。

## 3. R07 — 校验后的清单与执行许可仍与外部可变对象共享引用

**严重级别：P1，真实数据接入前必须修复。**  
**当前影响：局部桥接已可复现；尚无真实 runner，故没有据此声称发生真实读取或泄漏。**

### 3.1 清单在构造时校验，但使用时读取的是仍可变的原对象

位置：`experiments/lcrseg/five_frameworks_v1/integration.py:96–110`，特别是第 102 行：

```python
self.manifest=manifest
```

构造时 validate_current_manifest 与 authorized digest 检查没有问题。问题在于，调用者的 manifest、嵌套 L/U 列表和各样本字典仍被共享。后续 labeled()/unlabeled() 不重新核对这些实际值。

局部复现：

1. 建立只允许当前域、路径为 `approved-U-0` 的测试清单与许可。
2. 构造 CurrentDomainDataAdapter。
3. 修改调用者原字典：image 改成 `unapproved-history-or-future-image`，domain 改成 `not-current`。
4. 调用 unlabeled(0)。
5. 未被拒绝，reader 收到新路径；adapter 中缓存的 manifest_digest 仍是旧值。

同样可以影响 L 的图像和 label 路径。这里的路径全是符号字符串，reader 只记录实参，没有打开任何真实文件。

### 3.2 frozen dataclass 没有深度冻结许可中的 dict/list

位置：`integration.py:85–93`。

ExecutionPermit 使用 frozen=True，但 bindings、budget 是引用传入的普通 dict，authorized_manifest_digests 也是 list。构造后修改调用方的 bindings 即可把另一份原本未批准清单的 digest 加入许可，构造新的 DataAdapter 不再拒绝。

另一个局部检查将创建许可时的 `max_formal_optimizer_updates=0` 通过调用方 budget 字典改成 1000000；permit.budget 也随之改变。本检查不声称实际 runner 已违反预算，因为当前没有注册该 runner；它证明这个对象不是其声明含义下的不可变预算快照。

这两点是**同一个根因：校验身份没有绑定到之后实际使用的稳定快照**。不要求为此设计抵御恶意同进程 Python 代码的安全沙箱，也不要求新的签名系统。

## 4. 修复规格

1. 校验、计算 digest 和实际读者引用必须来自同一份由 adapter 自己持有的快照。避免校验一个 dict、之后继续读取调用者可变 dict。
2. 推荐采用深复制后递归冻结的清单/样本记录；也可以采用私有不可变记录加只读副本。调用者修改原列表、原 row、嵌套字段，不应改变未来 reader 实参。读取入口可以在发现变化时拒绝，但必须在 callback 前拒绝。
3. ExecutionPermit 的 bindings、允许 digest 集合、phases 和 budget 均需要复制并冻结。对外暴露只读结构或副本，不暴露可写内部记录。
4. 保持 `_seal` 的对象身份，不对整个许可进行会复制 seal 的无差别 deepcopy。保证预检发出的许可和合法 fixture 在正常路径仍可使用。
5. 不必每次读取都重新 hash 全部清单；不可变快照通常足以解决此处问题。真实路径规范化、软链接和 reader 的权限仍属于后续真实接入契约。
6. 为 L/U 原清单修改、嵌套允许列表修改、原 budget 修改、直接暴露可写记录、合法无修改路径分别增加回归。
7. 不注册真实 runner，不读真实数据，不扩大预算，不新增框架或 efficacy gate。

可选的同类检查：NativeParentBridge 对 metadata 目前也只做浅复制，且 semantic_metadata 返回原对象。真实绑定时应避免参数白名单/坐标声明被意外修改；这作为同类数据所有权检查，不另增方法实验。

## 5. 修复后与真实实验之间的剩余工作

真实父方法仍未绑定；通用 NativeParentBridge callbacks 不等于已经接上用户原始 KI。实际 StageTrainer/恢复器仍明确是 synthetic 路径。下一步必须以被用户认可的源码/配置/入口建立真实桥接，不能将 synthetic=False 改为 True 绕过限制，也不能用 F_CONV/SVD 静默替代。

若当前已有明确源码入口、配置或用户指认的版本，可直接做来源绑定；不要求一定找回某个旧命令文本才能工作，更不要求历史 SOTA 表。不存在有效线索时，记录原缺项，不再无限搜索旧日志。

之后需按分级授权完成：真实桥接和数据/评价接入的代码审阅；明确预算的 CUDA 合成资格；获准的当前 L smoke；源/目标预算及数据身份冻结；然后才是 C/D 全部五框架实验。此报告不授权这些运行。

## 6. 证据入口（固定提交）

仓库：`https://github.com/DLwbm123/SSL_CL_seg`  
提交：`bac21b6585ec284fb6e93f213e280c1492f6e57a`

- `experiments/lcrseg/docs/five_frameworks_v1/review/REVIEW_RESPONSE_R1.md`
- `experiments/lcrseg/docs/five_frameworks_v1/review/CODE_MANIFEST.json`
- `experiments/lcrseg/docs/five_frameworks_v1/review/TEST_LOG_SUMMARY.txt`
- `experiments/lcrseg/docs/five_frameworks_v1/review/CPU_COST_R2.json`
- `experiments/lcrseg/tests/five_frameworks_v1/test_revision_r2.py`
- `experiments/lcrseg/five_frameworks_v1/integration.py`

**总结：关闭原六项，限定修复新增 R07，保留研究设计；当前仍是 CODE_ONLY，而非实验放行。**
