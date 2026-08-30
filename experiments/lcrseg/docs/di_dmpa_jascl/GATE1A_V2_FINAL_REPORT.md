# Gate 1A v2 — null-aware spherical feature model 最终报告

日期：2026-08-30（Asia/Shanghai）。本报告仅关闭 Gate 1A v1 并交付唯一一次 Gate 1A v2 机制准入诊断，不授权后续训练或诊断。

## 结论

**`PASS_MULTI_MODALITY_SUPPORTED`；`passing_K=[2,3,5]`；按预注册选择最小通过值 `selected_K=2`。**

唯一主面板 B0-EMA 的18个前景单元满足全部 A1–A6。四个面板分别完成，未合并72个前景单元、未跨面板平均或投票、未使用控制面板选择 K 或 feature source。`primary_feature_source=ema_teacher`，`feature_source_selection_performed=false`。

这是冻结 Fundus / B0-EMA / null-aware 几何模型下的 Gate 1A 准入结果，不是整个 Gate 1 通过，也不是 DI-DMPA 的分割性能收益或临床有效性结论。**Gate 1B/1C 未运行，方法未注册，训练未启动。下一步固定为 `STOP_FOR_INDEPENDENT_REVIEW`。**

机器原始状态：[GATE1A_V2_STATUS.json](GATE1A_V2_STATUS.json)。完整逐单元结果、条件/保守指标及 CSV 均保存在 [原始 v2 attempt 归档](gate1a_v2_results/gate1a_v2_8ae5d7532f90aee5d53c0d966706ef64c18a19ac_attempt1/)。下表仅为显示而舍入，准入使用未舍入原值。

## B0-EMA 唯一主准入结果

A1 要求至少12/18个前景单元的验证集 `R95_null_worst_case` 严格下降；A2 要求9个 seed-domain 前景 macro 相对降幅中位数至少10%；A3 要求 active directional cluster 中 occupancy≥0.05 的比例至少90%；A4 要求五次固定 bootstrap/Hungarian matched cosine 中位数至少0.85；A5 要求至少2/3域跨种子前景 macro 半径严格下降；A6 排除背景。

| K | A1 改善单元 | A2 半径降幅中位数 | A3 occupancy 合格比例 | A4 matched cosine | A5 改善域 | A6 排除背景 | A1–A6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 18/18 | 14.494354% | 100%（36 clusters） | 0.9987574804 | 3/3 | 是 | 全通过，选中 |
| 3 | 18/18 | 19.239903% | 100%（54 clusters） | 0.9981140638 | 3/3 | 是 | 全通过 |
| 5 | 18/18 | 27.915391% | 98.888889%（90 clusters） | 0.9986168936 | 3/3 | 是 | 全通过 |

K=2 的 A2 原值为 `0.14494353698204337`；其9个 seed-domain 相对降幅、三个域的原始半径差及所有 A1–A6 布尔条件均在机器状态中。域顺序从冻结协议读取为 `REFUGE → RIM_ONE_r3 → Drishti_GS`。前景类别为1（optic_disc_rim）和2（optic_cup），背景0只报告，不参与准入。

## 三个独立控制面板

每个控制面板仍独立包含18个前景单元，使用相同 cases、coordinates、weights、UID order、bootstrap draws、solver seeds 和 K。以下仅描述统计，**不对控制面板施加准入判断，不用于挽救或替换 B0-EMA**。

| 面板 | K | A1 型改善计数 | A2 型降幅中位数 | A3 型 occupancy 比例 | A4 型 matched cosine | A5 型改善域 |
| --- | --- | --- | --- | --- | --- | --- |
| B0-student | 2 | 18/18 | 27.162630% | 100% | 0.9999306804 | 3/3 |
| B0-student | 3 | 18/18 | 39.944038% | 100% | 0.9999198643 | 3/3 |
| B0-student | 5 | 18/18 | 56.500135% | 97.777778% | 0.9999150345 | 3/3 |
| C0-EMA | 2 | 18/18 | 14.947939% | 100% | 0.9985304082 | 3/3 |
| C0-EMA | 3 | 18/18 | 20.961558% | 100% | 0.9987820774 | 3/3 |
| C0-EMA | 5 | 18/18 | 29.472860% | 100% | 0.9981112337 | 3/3 |
| C0-student | 2 | 18/18 | 25.983537% | 100% | 0.9999045348 | 3/3 |
| C0-student | 3 | 18/18 | 40.512749% | 98.148148% | 0.9999200145 | 3/3 |
| C0-student | 5 | 18/18 | 54.048592% | 94.444444% | 0.9998758463 | 3/3 |

控制面板分别提供表示源敏感性、SSL objective 依赖和联合控制的信息；这些观察不改变预注册主路径。未来方法性能仍须同时对比 B0 和 C0。

## 完整 support census 与 null 保留

先完成全部72个 feature units（4 panels × 3 seeds × 3 domains × 2 roles），并核验18个 checkpoint 和全部缓存后，才打开 geometry barrier。barrier 中 `clustering_jobs_started_before_this_barrier=0`；独立 postrun 文件时序核验同样通过。

总计 **11,161,740 条注册观测行，11,161,718 active，22 null，删除/替换/重采样0条**。这是四面板、各 stage/role/class 的缓存行总数，不是独立病例数，也不是准入的分母。每个 panel 的相同注册坐标仍作为独立面板观测保存。

| 面板 | 注册行 | Active | Null | class0 null mass | class1 null mass | class2 null mass |
| --- | --- | --- | --- | --- | --- | --- |
| B0-EMA | 2,790,435 | 2,790,418 | 17 | 0 | 4.985868221162339e-6 | 4.144431327335740e-6 |
| B0-student | 2,790,435 | 2,790,435 | 0 | 0 | 0 | 0 |
| C0-EMA | 2,790,435 | 2,790,430 | 5 | 0 | 4.817006585664777e-6 | 3.968253968253968e-7 |
| C0-student | 2,790,435 | 2,790,435 | 0 | 0 | 0 | 0 |

表中各 class null mass 为该面板18个 feature units 的原权重 null mass 等权平均（含 train_labeled/val），不是 pooled pixel fraction，也未跨面板平均。最大 case-class null fraction：B0-EMA `0.0035714285714285713`，C0-EMA `0.000851063829787234`。四面板均无全-null case-class 单元，full-map/registered 非有限值均为0。

每个 class cache 原样保留 N 行 `directions.npy`、`active_mask.npy`、`raw_norms.npy`，共216个 class caches / 648个数组。null 行以全零占位且 mask=false，绝不进入归一化或 cosine；worst-case cosine/spherical Euclidean 距离均为2。原始权重及全部 UID 进入同一 weighted ECDF。

真实集成复现了 B0 seed2 stage0 EMA / REFUGE / train_labeled / `REFUGE_test_n0128` / class1 / `(125,212)`：原 batch 的15,096条对应注册行全部保留，已知零点 `raw_norm=0`、`active_mask=false`、距离2，未拟合原型，模型及 checkpoint 不变。

648个原型数量间 Q 恒等式检查全部通过，最大绝对误差 `6.245004513516506e-17`。所有输出均同时保留 conditional directional 与 null-worst-case 指标；A1/A2/A5 仅使用后者的半径。

## 执行、测试与不可变性

- 正式 attempt 恰好1次，退出码0；72/72 features，432/432 geometry jobs，无缺失/重复。四面板各108 jobs，包含背景及 K=1 reference。
- 432 original fits + 2,160 fixed bootstrap fits = 2,592 fits；每 fit 五个固定 restarts，共12,960 restart 记录。
- exact-code 测试 **98 passed / 0 failed / 0 skipped，11.41s**，包含真实已知零点集成；此前纯合成开发测试32项与97项均通过，另行归档，不计作额外正式 attempt。
- 当前模型和18个磁盘 checkpoint 前后逐位/字节不变；冻结采样 plan 的 SHA256 一致，无旧 partial feature cache 复用。
- postrun 校验正式 manifest 内 **1,201个文件 / 1,614,319,773 bytes** 全部匹配；同时重新核验 v1 attempt1 的92个文件、attempt2 的174个文件，原 manifest SHA 和全部成员字节均不变。
- 两张 RTX3090 并行提取；batch=8、float32 forward、float64 norms/clustering/statistics、eval/no_grad、AMP off、stochastic classifier off；16个 CPU geometry workers、每 worker 单线程 BLAS。未改变冻结 solver 或训练配置。
- `model_optimizer_steps=0`、`transport_optimizer_steps=0`、`hidden_gt_training_usage=none`、`test_gt_usage=none`。无训练对象、反向传播或阈值再选择。

### 必须保留的警告

**79/12,960 个 restart 在冻结的100次迭代上限时未满足收敛条件；其中6/2,592个最终选中 fit 未收敛，全部 K=5。**没有 inactive slots 或无方向支持 fit。所有原始输出原样保留；未加迭代、reroll、替换 restart 或改变准入条件。这是优化近似性的审阅风险，不能描述为“所有拟合均收敛”。

六个选中 fit 的完整定位、全部79条 restart 记录、开发和传输事件见 [失败与警告](GATE1A_V2_FAILURES_AND_WARNINGS.md) 及 [原始警告审计](gate1a_v2_results/postrun_8ae5d75_attempt1/GATE1A_V2_FIT_WARNINGS_AUDIT.json)。K=2/3 的选中 fit 均收敛，但不据此重新选择 K；K=2 来自预注册的 smallest-passing-K 规则。

## v1 closure 与冻结身份

v1 的 `all registered features norm>1e-12` 假设被确切注册零点否定，根因为 `POST_RELU_FEATURE_SUPPORT_INCLUDES_EXACT_ZERO_ATOM`。这不是新的 guard bug。closure 状态为 `CLOSED_V1_FEATURE_SUPPORT_ASSUMPTION_FALSIFIED`；禁止 v1 attempt3。

两次 v1 attempt 永久保留 `BLOCKED_NUMERICAL_FAILURE`、geometry=0、A1–A6 uncomputed、selected_K=null；v2 不覆盖或重命名它们。完整身份见 [GATE1A_V1_CLOSURE.md](GATE1A_V1_CLOSURE.md)。

| 顺序 | 独立发布 | Commit |
| --- | --- | --- |
| 分支基点 | v1 attempt2 report | `606a5c53a37d0e4c9605415e8b38a1f177d1604f` |
| 1 | v1 closure | `b61f6db0ca9e746d005937e7dfc51c45078e1d80` |
| 2 | v2 preregistration | `eaae37bbaa7546679d9e6893023afbeeef0ab5c6` |
| 3 | v2 execution authorization | `e8f558dcc3fb6054a3f757c1295bd07ede2a002b` |
| 4 | exact diagnostic code | `8ae5d7532f90aee5d53c0d966706ef64c18a19ac` |
| 5 | 本次报告 | 首次添加本文件的 commit；与 diagnostic code SHA 分开解析 |

v2 preregistration MD SHA256：`9e051c6f270fa673d7f8078eaceb4f0d916d5b929a1c92765ff111f67bcbc2fd`。

v2 preregistration JSON SHA256：`b97847425cf5ef612aa646e98b1a21fde31fae4dccfa45a9e9b2d1481497a50f`。

冻结 sampling-plan SHA256：`96277c841165d8cd84a63d3d4a9301b27c50dcdd87d0b0d215c162d95c345e24`。

原始正式 artifact manifest SHA256：`a483757ccd6918d06df83cf91ba89c8641a90a3eb4a40fb8749eaf84d7ee617d`。

所有发布均按顺序独立 push 并远端核验；预注册和授权核验发生在新 checkpoint tensor/forward 之前，exact code 核验发生在真实集成之前。`main` 保持 `46e892960240543c946c570a9378d409b226384b`，本分支未从 main 创建，也未合并 main。

## 交付索引与停止边界

云端完整原始目录：

```text
/root/LCRSeg/runs/di_dmpa_gate1_v2/eaae37bbaa7546679d9e6893023afbeeef0ab5c6/gate1a_v2_8ae5d7532f90aee5d53c0d966706ef64c18a19ac_attempt1
```

- [预注册 MD](DI_DMPA_GATE1A_V2_PREREGISTRATION.md) / [JSON](DI_DMPA_GATE1A_V2_PREREGISTRATION.json)，[执行授权 MD](GATE1A_V2_EXECUTION_AUTHORIZATION.md) / [JSON](GATE1A_V2_EXECUTION_AUTHORIZATION.json)。
- [完整原始诊断、census、cache/model manifest、CSV、pytest、metadata、input audit](gate1a_v2_results/gate1a_v2_8ae5d7532f90aee5d53c0d966706ef64c18a19ac_attempt1/)。
- [postrun 完整性审计](gate1a_v2_results/postrun_8ae5d75_attempt1/GATE1A_V2_POSTRUN_INTEGRITY_AUDIT.json)。
- [exact commands](GATE1A_V2_EXACT_COMMANDS.md)、[交付 manifest](GATE1A_V2_ARTIFACT_MANIFEST.json)。原始 attempt manifest 在原始归档中保持不变；外层交付 manifest 用于公开副本验证。

公开仓库不包含原始特征数组或 checkpoint；云端保留，manifest 记录其路径、shape/dtype、原始权重/UID hash、大小和 SHA256。44,276,416-byte sampling plan 的重复副本不再次发布，字节身份仍冻结且可从云端核验。未改写任何原始 attempt 的报告、日志或状态。

**到此停止。** `method_registered=false`、`di_dmpa_training_launched=false`、Gate1B=false、Gate1C=false。未启动 transport、reliability、gradient-conflict、teacher-noise、theory final、Prostate、MnMS、Gate2 或 main merge。后续任何执行需要独立审阅后的新授权。
