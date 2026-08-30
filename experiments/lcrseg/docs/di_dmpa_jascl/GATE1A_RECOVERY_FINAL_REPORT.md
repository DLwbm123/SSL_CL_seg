# Gate 1A sampled-norm recovery：独立审阅报告

日期：2026-08-30，Asia/Shanghai。

## 结论：修复已验证，但正式 attempt2 因注册零向量阻断

**BLOCKED_NUMERICAL_FAILURE**。已知 seed1 失败单元的完整注册坐标定位通过；随后正式 attempt2 在另一单元发现**真正位于冻结注册坐标的零向量**，已按协议停止。

| 失败定位 | 精确值 |
| --- | --- |
| Panel / source | B0-EMA / ema_teacher |
| Seed / stage / domain / role | 2 / 0 / REFUGE / train_labeled |
| Case | REFUGE_test_n0128 |
| Class | 1，optic_disc_rim |
| Registered coordinate | y=125, x=212 |
| Selected norm / invalid count | 0.0 / 1（当前失败 case） |
| Checkpoint ID | B0/seed2/stage0 |
| Checkpoint SHA256 | `244c87368f252a660bf0d1934bf0ccf512790dc698d04ede01196d14c34064ac` |
| Sampling unit SHA256 | `5574b03230f7e724747174387edd1e96c70f565d52cfb61f9a53414ceb57a8d9` |

该坐标确实存在于 attempt1 的冻结采样计划中，已只读核对；没有读取标签重建坐标。case ID 中的 `test` 是原有名称的一部分，**冻结协议角色为 train_labeled**，不是本轮 test role。

没有删除、替换或重采样该向量；没有 eps 归一化、修改 feature tap、阈值、K、面板或准入条件；没有重跑。**0/432 geometry jobs，A1–A6 全部未计算，selected_K=null**。这不是 FAIL_MULTI_MODALITY_NOT_SUPPORTED，也不是 K1 fallback，更不是 Gate1 PASS。

## 已知 seed1 单元定位：PASS

状态：**PASS_FALSE_POSITIVE_FULL_MAP_SCOPE_CONFIRMED**。

- B0 seed1 stage0 REFUGE EMA val：100个 case，14,745,600个完整图向量。
- 1个 case（REFUGE_test_n0133）存在2个有限零向量，坐标 (196,260)、(245,192)；完整图 NaN/Inf=0。
- 三类各142,500个注册向量，共427,500个；注册零向量、非有限值和 full-zero/registered 交集全部为0。
- 三类最小注册范数分别为 0.2751399874687195、0.11928451342689131、0.5845266097488644。
- batch_size=8、原 checkpoint、原采样计划、原 forward seed、eval/no_grad、AMP off、stochastic_classifier=false。
- student/EMA/classifier/GAS/buffers 和 checkpoint 前后未变。

这份后续定位不覆盖或改写 attempt1：其原始 **BLOCKED_NUMERICAL_FAILURE**、0聚类、A1–A6未计算、selected_K=null 均永久保留。

## Attempt2 完成量与数值审计

每面板18个 feature units = seed/domain/role；不是18个前景准入单元。四面板几何前景单元均为0/18。

| Panel | Feature units | 状态 |
| --- | ---: | --- |
| B0-EMA | 14/18 | 注册零向量 hard stop |
| B0-student | 16/18 | 全局停止，未完整 |
| C0-EMA | 2/18 | 全局停止，未完整 |
| C0-student | 2/18 | 全局停止，未完整 |

共34/72个完整 feature units、102个新生成的原始 class caches。未复用 attempt1 的48个 partial caches。四个正式 geometry CSV 未生成，不用空表或合成结果伪装正式输出。

已审计的34个完整单元和首个失败 case 共1,457个 case/source/role 出现次数、214,843,392个完整图向量。观察到5个完整图有限零向量，其中1个位于注册 class1 坐标；非有限值=0。累计各类已检查1,973,497个注册向量。**这些是已审计部分的计数，不是72单元的最终全覆盖统计**；失败 batch 中未继续审计的其他 case 不在计数中。

原始 norm audit 将完整单元汇总与 `failed_case_summary` 分开保存。因此不能只读完整单元的注册零计数0而忽略失败 case 的 class1零计数1。[恢复状态 JSON](GATE1A_RECOVERY_STATUS.json)另给出了两者合计。

## 取消、不变性和文件完整性

- 失败 shard0 退出1；另一 shard1 完成当前 checkpoint guard 后退出0，在 C0/seed0/stage2 之前取消。主进程退出2。
- STOP_REQUESTED.json 已保存；固定600秒超时未触发，**无强制 SIGTERM**。
- 本轮启动的9个 checkpoint task 全部有前后状态审计且 bitwise unchanged，包括失败的 B0/seed2/stage0；8个提取完整完成，1个提取失败。
- 不声称18个 checkpoint 全部完成了特征提取内存审计。停止后重新核验18个磁盘 checkpoint，全部未变。
- 原 attempt1 manifest 的92项文件、attempt2 manifest 的174项文件均重新核对 SHA 通过。
- 原 preregistration MD/JSON、采样计划字节和 geometry normalize/Kmeans/bootstrap/metric/adjudication 源码未改变。

## 测试与发布顺序

两轮开发期 synthetic tests 均64 passed。发布的精确 recovery code 上，**65 passed，0 failed，0 skipped，0 warnings**，包含已知 seed1单元全部注册坐标的真实只读集成测试。测试通过不代表机制准入通过。

| 身份 | 完整 Git SHA |
| --- | --- |
| Original preregistration | `cfb62554f1e6a2a36850547485b1857dc9a28a20` |
| Original authorization | `25ec97c988af290a4fb7a637c4b7cdfe462deb87` |
| Attempt1 report / recovery branch base | `945b484072cb9f2757be98df34e5d72844596e84` |
| Scope clarification | `e8336da9d7364f4b67912d03791195445318afc3` |
| Recovery diagnostic code | `a89716ddbd2eccbe76c574e97e520d424aa923ab` |

Clarification 先独立 push/核对，之后才实现并发布 recovery code；精确 code push/核对后才启动真实定位。定位 PASS 后才启动唯一 attempt2。Report commit 是首次加入本报告字节的独立提交，推送后另报完整 SHA，不混同执行源码。

共享采样计划 raw SHA256：
`96277c841165d8cd84a63d3d4a9301b27c50dcdd87d0b0d215c162d95c345e24`。

原 attempt2 manifest raw SHA256：
`15e7beaf67ad55bd1c18b494f53c7f06fc6ea92ff161f99ccf62b6a648736ea3`。

## 路径与交付

```text
/root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_formal_a89716ddbd2eccbe76c574e97e520d424aa923ab_attempt2
```

- [作用域澄清 MD](GATE1A_NUMERICAL_SCOPE_CLARIFICATION_V1.md) / [JSON](GATE1A_NUMERICAL_SCOPE_CLARIFICATION_V1.json)
- [Known-failure localization audit](gate1a_recovery_results/gate1a_known_failure_a89716d_attempt1/GATE1A_KNOWN_FAILURE_LOCALIZATION_AUDIT.json)
- [Recovery unit/integration report](gate1a_recovery_results/gate1a_recovery_tests_a89716d_attempt1/GATE1A_RECOVERY_UNIT_INTEGRATION_TEST_REPORT.json)
- [Pytest XML](gate1a_recovery_results/gate1a_recovery_tests_a89716d_attempt1/GATE1A_RECOVERY_PYTEST.xml) / [stdout](gate1a_recovery_results/gate1a_recovery_tests_a89716d_attempt1/GATE1A_RECOVERY_PYTEST_OUTPUT.txt)
- [原始 attempt2 状态](gate1a_recovery_results/gate1a_formal_a89716ddbd2eccbe76c574e97e520d424aa923ab_attempt2/GATE1A_STATUS.json) / [RUN metadata](gate1a_recovery_results/gate1a_formal_a89716ddbd2eccbe76c574e97e520d424aa923ab_attempt2/GATE1A_RUN_METADATA.json)
- [Registered norm audit](gate1a_recovery_results/gate1a_formal_a89716ddbd2eccbe76c574e97e520d424aa923ab_attempt2/GATE1A_REGISTERED_NORM_AUDIT.json) / [full-map zero diagnostic](gate1a_recovery_results/gate1a_formal_a89716ddbd2eccbe76c574e97e520d424aa923ab_attempt2/GATE1A_FULL_MAP_ZERO_DIAGNOSTIC.json)
- [完整失败坐标与上下文](gate1a_recovery_results/gate1a_formal_a89716ddbd2eccbe76c574e97e520d424aa923ab_attempt2/numerical_failures/B0-EMA_seed2_stage0_train_labeled.json)
- [Model immutability audit](gate1a_recovery_results/gate1a_formal_a89716ddbd2eccbe76c574e97e520d424aa923ab_attempt2/GATE1A_MODEL_IMMUTABILITY_AUDIT.json) / [feature-cache manifest](gate1a_recovery_results/gate1a_formal_a89716ddbd2eccbe76c574e97e520d424aa923ab_attempt2/FEATURE_CACHE_MANIFEST.json)
- [Post-run integrity audit](gate1a_recovery_results/gate1a_recovery_postrun_a89716d_attempt1/GATE1A_RECOVERY_POSTRUN_INTEGRITY_AUDIT.json)
- [全部失败/警告](GATE1A_RECOVERY_FAILURES_AND_WARNINGS.md) / [准确命令](GATE1A_RECOVERY_EXACT_COMMANDS.md) / [交付哈希 manifest](GATE1A_RECOVERY_ARTIFACT_MANIFEST.json)

原始 attempt2、定位和测试证据原样归档；102份 raw feature tensor 只在云端保留，其路径/shape/dtype/SHA仍在 manifest。两份复用计划不重复提交44MB内容，而引用已经提交的 attempt1 同一字节文件；云端两份复制均为只读并核对哈希。

## Hard stop

全部 method flags=false；method_registered=false；di_dmpa_training_launched=false；model/transport optimizer steps=0；hidden-GT training usage=test GT usage=none。未运行 Gate1B/C、transport/reliability、正式 gradient-conflict/teacher-noise、theory final、training、Prostate/MnMS、Gate2，未合并 main。

**STOP_FOR_INDEPENDENT_REVIEW**。注册零向量不能再解释为本轮允许的“未采样完整图零向量”；不以新过滤规则、换源或控制面板挽救本轮。
