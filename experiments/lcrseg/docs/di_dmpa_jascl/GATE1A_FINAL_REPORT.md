# Gate 1A 独立审阅报告

日期：2026-08-30（Asia/Shanghai）。

## 结论

**BLOCKED_NUMERICAL_FAILURE**。正式执行在 **B0-EMA、seed 1、
stage 0 / REFUGE、validation** 特征提取时发现 `decoder.dec1`
零范数向量：观测最小范数为 **0.0**，触发冻结规则 `norm <= 1e-12`。

程序按预注册停止，另一 GPU 分片被终止。退出码 **2**，进程已退出，
两张 GPU 已释放。**聚类任务数为 0；A1–A6 均未计算；selected_K=null。**
不能将本轮记为多模态不成立，也不能退回 K=1 冒充已完成判定。
Gate 1 overall 仍为 incomplete；没有 PASS_CORE_ADMISSION。

## 不应超出证据的解释

检查发生在完整的 384×384、16-D 特征图上，先于锁定坐标的采集。
因此当前证据**没有证明零向量位于已注册的采样坐标中**。
原始异常记录了最小范数和调用位置，未记录具体 case ID、像素坐标
或零向量数量；本轮没有为了定位而再次运行模型。

该异常不是已捕获的 NaN/Inf：有限性检查在零范数检查之前通过。
目前没有确认实现缺陷，不能以此为由放宽阈值、过滤像素、
重新采样、改变 feature source 或重跑。完整图检查的适用范围、
零向量处理和后续授权交由独立审阅决定；冻结预注册没有改动。

## 四面板完成情况

下表的“特征单元”指 seed/domain/role，**不是**18个前景准入单元。
每面板预期 3 seeds × 3 domains × 2 roles = 18 个特征单元。

| Panel | 已完成特征单元 | 几何前景单元 | 状态 |
| --- | ---: | ---: | --- |
| B0-EMA | 7/18 | 0/18 | seed 1 REFUGE val 数值阻断 |
| B0-student | 9/18 | 0/18 | 因全局 hard stop 取消后续 |
| C0-EMA | 0/18 | 0/18 | 全局阻断，未开始 |
| C0-student | 0/18 | 0/18 | 全局阻断，未开始 |

共保存 **16** 份 feature-unit manifest、**48** 个原始 class feature cache。
未启动任何 K=1/2/3/5 聚类、bootstrap geometry 或 boundary/interior
数值评价；不存在可用于控制面板性能比较或主面板救援的结果。

| K | A1 | A2 | A3 | A4 | A5 | A6 | 准入 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 未计算 | 未计算 | 未计算 | 未计算 | 未计算 | 未判定 | blocked |
| 3 | 未计算 | 未计算 | 未计算 | 未计算 | 未计算 | 未判定 | blocked |
| 5 | 未计算 | 未计算 | 未计算 | 未计算 | 未计算 | 未判定 | blocked |

A6 排除 background 的规则仍然冻结，但未执行最终准入编译。
`passing_K=null` 不等同于“所有 K 已测试且失败”。

## 已通过的检查

- 已发布执行源码上的 **47 tests passed，0 failed，0 skipped，0 warnings**。
  包括真实 checkpoint 的只读小坐标集成测试；集成测试是 B0 seed 0
  stage 0 的 student/EMA、首个注册 labeled case 的8个固定坐标，
  不读标签数组、不拟合 prototype、不推断机制是否成立。
- 18 个 stage-best checkpoint、raw/canonical config、
  DOMAIN_PROTOCOL、3个 manifest 和3个 split 的输入校验通过。
- 完整18组共享采样计划已生成并锁定，四面板绑定同一哈希：
  `96277c841165d8cd84a63d3d4a9301b27c50dcdd87d0b0d215c162d95c345e24`。
  原始 JSON 大小 **44,276,416 bytes**；没有按特征重新采样。
- 已结束或失败的4个 checkpoint task 均留下 student/EMA、
  classifier/GAS/buffer 的前后 bitwise 一致证据：
  seed0 的3阶段完整完成；seed1/stage0 虽提取失败，不变性仍通过。
- 另一分片在 B0/seed1/stage1 运行途中收到 SIGTERM，
  **没有**该 task 的 in-memory after-state audit。
  不声称18个 checkpoint 全部完成了内存态审计。
- 停止后重新核验：**18个磁盘 checkpoint 全部未改变**；
  原始 attempt manifest 中 **92个产物**的 SHA 全部一致。

上述 PASS 是测试、输入或文件不变性 PASS，**不是 Gate 1A 准入 PASS**。

## 身份与路径

| 身份 | SHA |
| --- | --- |
| Frozen preregistration | `cfb62554f1e6a2a36850547485b1857dc9a28a20` |
| Authorization commit | `25ec97c988af290a4fb7a637c4b7cdfe462deb87` |
| Diagnostic code commit | `8f4a71a5ea8d145183a3007ccd398ab79387478e` |
| Preregistration MD raw SHA-256 | `32acdc5c24bcc5763daa6cb3650fea91f46da7ae3845b1fd0615c781619fbf0a` |
| Preregistration JSON raw SHA-256 | `6f50bd9df404d987aa70e2035a5c3f3853aa59ce49d21ffface34172cf754cbf` |
| BASELINE_FREEZE raw SHA-256 | `e171a1d476ca626830541e80dbb1dff763ae02716a09463bb70ea5892da8231a` |
| 原始 attempt artifact manifest SHA-256 | `c26edceea102da568421e0327a7cc10fabb2ceee16fc936dd8adeb439eab8ee9` |

工作分支：`codex/di-dmpa-gate1-diagnostics`。Report commit 为首次加入本报告
字节的独立 Git commit，推送后另报完整 SHA，不冒充执行源码提交。

正式 attempt（保留原始失败、日志和部分输出）：

```text
/root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_formal_8f4a71a_attempt1
```

主机上的 post-run 字节校验记录：

```text
/root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_report_evidence_8f4a71a_attempt1/GATE1A_POSTRUN_INTEGRITY_AUDIT.json
```

## 交付索引

- [机器可读状态](GATE1A_STATUS.json)
- [原始失败 attempt](gate1a_results/gate1a_formal_8f4a71a_attempt1/)
- [共享采样计划](gate1a_results/gate1a_formal_8f4a71a_attempt1/SHARED_GEOMETRY_SAMPLING_PLAN.json)
- [采样审计](gate1a_results/gate1a_formal_8f4a71a_attempt1/SAMPLING_PLAN_AUDIT.json)
- [原始运行 metadata](gate1a_results/gate1a_formal_8f4a71a_attempt1/GATE1A_RUN_METADATA.json)
- [输入审计](gate1a_results/gate1a_formal_8f4a71a_attempt1/GATE1A_INPUT_AUDIT.json)
- [部分 feature-cache manifest](gate1a_results/gate1a_formal_8f4a71a_attempt1/FEATURE_CACHE_MANIFEST.json)
- [部分模型不变性审计](gate1a_results/gate1a_formal_8f4a71a_attempt1/GATE1A_MODEL_IMMUTABILITY_AUDIT.json)
- [停止后完整性审计](gate1a_results/publication_evidence/GATE1A_POSTRUN_INTEGRITY_AUDIT.json)
- [停止后源文件/输入哈希绑定及在线查询警告](gate1a_results/publication_evidence/GATE1A_POSTRUN_SOURCE_BINDING.json)
- [单元/集成测试报告](gate1a_results/gate1a_formal_8f4a71a_attempt1/GATE1A_UNIT_INTEGRATION_TEST_REPORT.json)
- [真实 checkpoint 集成记录](gate1a_results/gate1a_formal_8f4a71a_attempt1/REAL_CHECKPOINT_EXTRACTION_INTEGRATION.json)
- [pytest JUnit](gate1a_results/gate1a_formal_8f4a71a_attempt1/pytest.xml)、[transcript](gate1a_results/gate1a_formal_8f4a71a_attempt1/pytest_output.txt)
- [完整失败/警告记录](GATE1A_FAILURES_AND_WARNINGS.md)
- [准确命令](GATE1A_EXACT_COMMANDS.md)
- [交付哈希 manifest](GATE1A_ARTIFACT_MANIFEST.json)

原始48个 `.npy` feature tensor 留在云主机，不提交 Git；
shape/dtype/SHA 和对应 manifest 全部保留。原始 attempt 的所有非 tensor
产物原样归档。四个 geometry CSV 因在聚类前阻断而未生成，
没有用合成测试结果或空表伪装正式结果。

## 授权边界与停止

`method_registered=false`、`di_dmpa_training_launched=false`；
全部新方法开关 false；model/transport optimizer steps 均为0。
hidden-GT training usage 与 test GT usage 均为 none。

没有运行 Gate 1B、Gate 1C、transport optimizer、正式 gradient-conflict、
teacher-noise、最终 theory quantities、Gate 2、Prostate、MnMS 或训练；
没有合并 main。**STOP_FOR_INDEPENDENT_REVIEW**。
