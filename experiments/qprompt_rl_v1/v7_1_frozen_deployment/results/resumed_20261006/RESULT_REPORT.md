# V7.1 Frozen-policy deployment diagnosis

## 中文结论

本轮已完成全部 8 个正式端点和统一评估，判定为 **DEPLOYMENT_RECOVERY_ONLY（仅支持部署修复）**。

- 相对 V7 贪心部署：四对新域和旧域差值均为正；新域平均 **+1.6767 pp**，旧域平均 **+2.8406 pp**。支持本轮随机部署较贪心部署改善的描述性结论，不意味着已消除全部训练—部署差异。
- 相对同实现匹配均匀采样：新域平均 **−0.0148 pp**，仅 **1/4** 对为正；旧域平均 **+1.5273 pp**。**未建立预登记的、具有实际意义的学习优势**，也不能据此证明两者完全等价。
- 相对 ALL_U 的旧域保持门槛通过：四对旧域平均 **+0.1424 pp**，最差 **−0.9063 pp**。但 GRPO_SAMPLE 相对共同源的绝对遗忘仍为 **9.8050–18.4248 pp**，不能写成“没有遗忘”。
- 最佳非 RL 方法价值门槛未通过：新域平均 **−0.8345 pp**，最差 **−1.8021 pp**。相对原匹配 RANDOM 的平均改善 **+1.5012 pp**，仍不能替代匹配均匀对照或最佳非 RL 门槛。
- 学生更新正好 **21,232**（资格 32 + 正式 21,200），控制器更新 **0**。首次失败的零更新诊断成本保留；本次恢复未新增训练分支、未调参、未扩大预算。
- 本次恢复从开始至报告完成约 **37.44 分钟**；首次失败诊断另有 **26.74 作业秒**，已计入汇总成本。并行作业耗时之和不能当成实际经过时间。

四对结果只来自一个固定源模型、两个域和两个控制器种子，不构成独立源训练重复或独立患者泛化确认。所有门槛和逐对正负结果均保留。原 V7 判定不变，没有启动下一轮实验。

完整机制数值见 [MECHANISM_SUMMARY.md](MECHANISM_SUMMARY.md)；保留的历史失败见相邻 `stopped_20261006` 报告，实际执行时间见 RUN_TIMING.json。


Status: **DEPLOYMENT_RECOVERY_ONLY**. Eight of eight final endpoints; 21,232 student updates, zero controller updates. Two independent source-to-target adaptations, not sequential three-domain training. NATIVE_LR_SRC_A reference, not a claim of exact original KI reproduction.

## All new endpoints

| Domain | Method | Controller | Macro % | Rim % | Cup % | Disc union % | REFUGE % | Absolute forgetting pp | Old vs ALL_U pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RIM_ONE_r3 | GRPO_SAMPLE | 601 | 69.8040 | 72.0636 | 67.5444 | 88.0650 | 73.3202 | 9.8050 | 1.9507 |
| RIM_ONE_r3 | UNIFORM_MATCHED | 601 | 69.2662 | 72.2088 | 66.3235 | 88.3062 | 70.6812 | 12.4440 | -0.6882 |
| RIM_ONE_r3 | GRPO_SAMPLE | 602 | 68.5200 | 72.3513 | 64.6887 | 88.5133 | 70.8458 | 12.2794 | -0.5236 |
| RIM_ONE_r3 | UNIFORM_MATCHED | 602 | 68.9896 | 72.4742 | 65.5049 | 88.4789 | 67.7014 | 15.4238 | -3.6680 |
| Drishti_GS | GRPO_SAMPLE | 601 | 73.9104 | 71.4795 | 76.3412 | 93.4555 | 65.6555 | 17.4697 | 0.0487 |
| Drishti_GS | UNIFORM_MATCHED | 601 | 74.0306 | 71.6845 | 76.3766 | 93.3372 | 65.2823 | 17.8429 | -0.3245 |
| Drishti_GS | GRPO_SAMPLE | 602 | 74.5403 | 72.2418 | 76.8388 | 93.6194 | 64.7004 | 18.4248 | -0.9063 |
| Drishti_GS | UNIFORM_MATCHED | 602 | 74.5477 | 72.2649 | 76.8305 | 93.6264 | 64.7480 | 18.3772 | -0.8588 |

## All four paired comparisons (pp)

| Domain | Seed | Sample − uniform new | Sample − uniform old | Sample − greedy new | Sample − greedy old | Sample − best non-RL |
|---|---:|---:|---:|---:|---:|---:|
| RIM_ONE_r3 | 601 | +0.5379 | +2.6390 | +2.9917 | +6.6608 | -0.5181 |
| RIM_ONE_r3 | 602 | -0.4696 | +3.1444 | +0.3331 | +2.2878 | -1.8021 |
| Drishti_GS | 601 | -0.1202 | +0.3733 | +1.5369 | +1.0848 | -0.8239 |
| Drishti_GS | 602 | -0.0074 | -0.0476 | +1.8451 | +1.3291 | -0.1940 |

Overall equal-pair means: {"deploy_new_pp": 1.676715066135609, "deploy_old_pp": 2.8406072101810427, "learn_new_pp": -0.014829036161620457, "learn_old_pp": 1.5272613668336903, "vs_best_nonRL_pp": -0.8345034206982072, "vs_random_pp": 1.5012108429574567, "old_vs_allU_pp": 0.14238753547064775}. Domain and controller means and every gate are in DECISION.json. 0.005 Dice equals 0.5 pp.

## Gates

- D2 / mean_new_at_least_0_5pp: **False**.
- D2 / positive_new_at_least_3_of_4: **False**.
- D2 / each_seed_new_mean_positive: **False**.
- D2 / mean_old_at_least_minus0_5pp: **True**.
- D2 / min_old_at_least_minus1pp: **True**.
- D3 / mean_best_at_least_0_5pp: **False**.
- D3 / mean_random_at_least_0_5pp: **True**.
- D3 / min_best_at_least_minus0_25pp: **False**.
- D3 / each_seed_best_mean_positive: **False**.
- D3 / mean_old_allU_at_least_minus0_5pp: **True**.
- D3 / min_old_allU_at_least_minus1pp: **True**.

## Interpretation and limits

No practically meaningful learning advantage was established under the prespecified matched-uniform screen; this is not proof of equivalence.

Stochastic deployment improved mean new-domain Dice over greedy deployment descriptively. All pairwise new/old outcomes remain visible. A positive deployment comparison alone supports only a deployment-recovery explanation. Frozen final policies still differ from evolving training behavior policies; not all train/deploy mismatch has been removed.

Relative ALL_U retention and absolute forgetting from the source are distinct. One fixed REFUGE source, two targets and two controller seeds are not four independent source or clinical repeats. Historical development patients cannot confirm independent-patient generalization. No statistical significance claim is made. No sealed test accessed.

## Mechanism and costs

See MECHANISM_AUDIT.md and all three diagnostic JSON files for policy/exposure/reward results and missing-state limitations. COSTS.json separates new work, per-job deployment overhead and prior learning cost (42,400 student and 1,696 controller updates). No speedup claim follows from selecting 25% U at fixed student steps.

## Historical controls and stop

All 18 V7 controls are included in HISTORICAL_V7_RESULTS.csv, with patient-role, native-code, source and evaluation provenance in PROVENANCE_AUDIT.json. Historical training commit 37ac930652dc2da6585342a437d014ae3b4b995d; public report baseline 03224a49cef09a3325abb7d32ffd8762e678e880. V7 original decision remains unchanged. This round ends here; no next-round experiments launched.
