# 原源完整接入与决策敏感保护：三流完整结果

结论：工程审计 PASS，综合科学门槛 FAIL。决策保护在三个流中都明显改善旧域保持，但新域下降约一个百分点；并未优于全特征保护。保留这一正向机制证据与明确代价，不将它包装成完整方法成功。

## 固定 EMA 主终点

A原源CE、B仅固定末头、C全patch保护、D决策空间保护。所有组由原FixMatch43600 EMA直接初始化；无S†校准、polar、旧图/U。

|流|组|student旧%|student新%|EMA旧%|EMA新%|源下降>1/>2人数|源p10 pp|
|---|---|---:|---:|---:|---:|---|---:|
|20270601|A|77.516161|82.361644|80.390247|83.378369|6/3|-2.147988|
|20270601|B|77.502621|82.361966|80.393418|83.381192|6/3|-2.144189|
|20270601|C|81.103586|82.572746|81.622533|82.407454|1/1|-0.446764|
|20270601|D|80.878240|82.571018|81.443108|82.378096|1/1|-0.361148|
|20270602|A|77.737048|82.630783|80.571473|83.313973|5/3|-1.999865|
|20270602|B|77.762410|82.626646|80.583636|83.305734|5/2|-1.987072|
|20270602|C|81.100193|82.757811|81.603353|82.335685|1/1|-0.478373|
|20270602|D|80.877736|82.821389|81.402882|82.318334|2/1|-0.425781|
|20270603|A|77.509862|82.444010|80.488850|83.213632|6/4|-2.196184|
|20270603|B|77.546241|82.437014|80.495471|83.209887|6/4|-2.157958|
|20270603|C|80.898639|82.644821|81.610027|82.407017|2/1|-0.487824|
|20270603|D|80.645461|82.615976|81.442751|82.364284|2/1|-0.447515|

原源旧域参照80.36740663770262%。D三个流旧均值/旧源CI门槛均通过；>1人数1/2/2、p10均高于-.50明显改善，但>2人数仍各1，且新域均低于83。

## 同流配对差及95%患者区间

|流|对比|旧域pp [CI]|新域pp [CI]|
|---|---|---|---|
|20270601|D−A|+1.052860 [+0.440503, +1.707558]|-1.000272 [-1.597071, -0.381512]|
|20270601|D−B|+1.049690 [+0.437369, +1.700561]|-1.003095 [-1.599888, -0.383599]|
|20270601|D−C|-0.179425 [-0.352462, -0.053014]|-0.029358 [-0.083686, +0.021672]|
|20270602|D−A|+0.831408 [+0.222432, +1.495147]|-0.995639 [-1.647352, -0.323301]|
|20270602|D−B|+0.819246 [+0.210384, +1.481557]|-0.987400 [-1.640357, -0.317396]|
|20270602|D−C|-0.200472 [-0.380177, -0.069280]|-0.017351 [-0.074948, +0.035260]|
|20270603|D−A|+0.953900 [+0.335698, +1.623247]|-0.849347 [-1.464549, -0.176039]|
|20270603|D−B|+0.947279 [+0.329932, +1.611206]|-0.845603 [-1.461942, -0.174018]|
|20270603|D−C|-0.167276 [-0.321878, -0.050854]|-0.042733 [-0.102513, +0.014541]|

D−B旧均值 +0.938738pp，三个患者配对CI均为正；新域三个CI均为负，不能称兼顾新旧。D−C新均值 −0.029814pp，未实现选择性可塑性收益；旧域三个CI均为负。A与B接近，故本配置下单独固定末头不是新域损失的主要证据来源。

## 下一步如何帮助完整方法

本轮说明实际读出保护可以恢复旧域均值并改善大部分尾部，但双侧漂移约束存在明显稳定性—可塑性代价。不能从这些结果证明所有损失都由有益margin变化被抑制造成。按照事先方法蓝图，后继完整候选改为可靠支持上的单侧margin侵蚀保护，保留GT CE，不扫描本轮λ/rank/EMA，不回填主终点。相同可靠支持、辅助聚合与梯度尺度口径下比较DS、标准centered-logit蒸馏和完整margin候选；再以单视图margin隔离可靠性作用。

## 冻结判定、P0与成本

```json
{
  "gates": {
    "20270601": {
      "deployment": false,
      "checks": {
        "old_absolute": true,
        "old_interval": true,
        "new_absolute": false,
        "new_vs_A": false,
        "tail_over_1": true,
        "tail_over_2": false,
        "tail_p10": true,
        "tail_vs_A_B": true
      },
      "increment_point": true,
      "increment_new": false,
      "selectivity_new": false,
      "selectivity_old": false
    },
    "20270602": {
      "deployment": false,
      "checks": {
        "old_absolute": true,
        "old_interval": true,
        "new_absolute": false,
        "new_vs_A": false,
        "tail_over_1": true,
        "tail_over_2": false,
        "tail_p10": true,
        "tail_vs_A_B": true
      },
      "increment_point": true,
      "increment_new": false,
      "selectivity_new": false,
      "selectivity_old": false
    },
    "20270603": {
      "deployment": false,
      "checks": {
        "old_absolute": true,
        "old_interval": true,
        "new_absolute": false,
        "new_vs_A": false,
        "tail_over_1": true,
        "tail_over_2": false,
        "tail_p10": true,
        "tail_vs_A_B": true
      },
      "increment_point": true,
      "increment_new": false,
      "selectivity_new": false,
      "selectivity_old": false
    }
  },
  "increment_mean_old_pp": 0.9387384937184109,
  "selectivity_mean_new_pp": -0.029813616152004428,
  "composite_pass": false,
  "P0": {
    "status": "PASS",
    "gates": {
      "gradient_calibration": true,
      "foreground_coverage": true,
      "foreground_loss_share": true,
      "gradient_separation": true,
      "full_path_gradient": true
    },
    "weights": {
      "DS": 0.5188853059792543,
      "FULL": 0.1619075108617199
    },
    "gradient_squared": {
      "CE": 14.943203343718954,
      "DS": 3.468813725550162,
      "FULL": 35.62785954054693
    },
    "gradient_cosine": 0.8248729490691197,
    "normalized_gradient_distance": 0.5918226946153389,
    "coverage_by_class": [
      0.9990019055905611,
      0.8176795580110497,
      0.5117712158559283,
      0.42260883444395675
    ],
    "covered_patient_counts": [
      12,
      10,
      10,
      9
    ],
    "foreground_loss_share": {
      "DS": 0.1385693946311217,
      "FULL": 0.10362893565755776
    },
    "DS_gradient_groups": {
      "encoder": 2.0034246307568364,
      "decoder": 1.4653890947933244
    },
    "centered_readout_error": 5.424022674560547e-06,
    "model_reference_RNG_unchanged": true,
    "probe_discarded": true
  },
  "audit": {
    "status": "PASS",
    "cost": {
      "total": 24544,
      "optimizer": 24544,
      "functional": 0,
      "pending": 0,
      "by_phase": {
        "P0_ENGINEERING:engineering": 32,
        "PROBE:probe": 64,
        "DETACH_B:detach": 32,
        "DETACH_D:detach": 32,
        "20270601_A:engineering": 32,
        "20270601_B:engineering": 32,
        "20270601_C:engineering": 32,
        "20270601_D:engineering": 32,
        "20270602_A:engineering": 32,
        "20270602_B:engineering": 32,
        "20270602_C:engineering": 32,
        "20270602_D:engineering": 32,
        "20270603_A:engineering": 32,
        "20270603_B:engineering": 32,
        "20270603_C:engineering": 32,
        "20270603_D:engineering": 32,
        "20270601_A:formal": 2000,
        "20270601_B:formal": 2000,
        "20270601_C:formal": 2000,
        "20270601_D:formal": 2000,
        "20270602_A:formal": 2000,
        "20270602_B:formal": 2000,
        "20270602_C:formal": 2000,
        "20270602_D:formal": 2000,
        "20270603_A:formal": 2000,
        "20270603_B:formal": 2000,
        "20270603_C:formal": 2000,
        "20270603_D:formal": 2000
      }
    },
    "calls": {
      "executed": {
        "p0_native": 418,
        "p0_reverse": 148,
        "detach_native": 256,
        "detach_reverse": 64,
        "engineering_native": 1008,
        "engineering_reverse": 408,
        "formal_native": 96000,
        "formal_reverse": 24108,
        "evaluation_native": 34704,
        "evaluation_reverse": 0
      },
      "by_group": {
        "P0_ENGINEERING:p0_native": 84,
        "P0_ENGINEERING:p0_reverse": 34,
        "PROBE:p0_native": 334,
        "PROBE:p0_reverse": 114,
        "DETACH_B:detach_native": 128,
        "DETACH:detach_reverse": 64,
        "DETACH_D:detach_native": 128,
        "20270601_A:engineering_native": 84,
        "20270601_A:engineering_reverse": 34,
        "20270601_B:engineering_native": 84,
        "20270601_B:engineering_reverse": 34,
        "20270601_C:engineering_native": 84,
        "20270601_C:engineering_reverse": 34,
        "20270601_D:engineering_native": 84,
        "20270601_D:engineering_reverse": 34,
        "20270602_A:engineering_native": 84,
        "20270602_A:engineering_reverse": 34,
        "20270602_B:engineering_native": 84,
        "20270602_B:engineering_reverse": 34,
        "20270602_C:engineering_native": 84,
        "20270602_C:engineering_reverse": 34,
        "20270602_D:engineering_native": 84,
        "20270602_D:engineering_reverse": 34,
        "20270603_A:engineering_native": 84,
        "20270603_A:engineering_reverse": 34,
        "20270603_B:engineering_native": 84,
        "20270603_B:engineering_reverse": 34,
        "20270603_C:engineering_native": 84,
        "20270603_C:engineering_reverse": 34,
        "20270603_D:engineering_native": 84,
        "20270603_D:engineering_reverse": 34,
        "20270601_A:formal_native": 8000,
        "20270601_A:formal_reverse": 2009,
        "20270601_A:evaluation_native": 2892,
        "20270601_B:formal_native": 8000,
        "20270601_B:formal_reverse": 2009,
        "20270601_B:evaluation_native": 2892,
        "20270601_C:formal_native": 8000,
        "20270601_C:formal_reverse": 2009,
        "20270601_C:evaluation_native": 2892,
        "20270601_D:formal_native": 8000,
        "20270601_D:formal_reverse": 2009,
        "20270601_D:evaluation_native": 2892,
        "20270602_A:formal_native": 8000,
        "20270602_A:formal_reverse": 2009,
        "20270602_A:evaluation_native": 2892,
        "20270602_B:formal_native": 8000,
        "20270602_B:formal_reverse": 2009,
        "20270602_B:evaluation_native": 2892,
        "20270602_C:formal_native": 8000,
        "20270602_C:formal_reverse": 2009,
        "20270602_C:evaluation_native": 2892,
        "20270602_D:formal_native": 8000,
        "20270602_D:formal_reverse": 2009,
        "20270602_D:evaluation_native": 2892,
        "20270603_A:formal_native": 8000,
        "20270603_A:formal_reverse": 2009,
        "20270603_A:evaluation_native": 2892,
        "20270603_B:formal_native": 8000,
        "20270603_B:formal_reverse": 2009,
        "20270603_B:evaluation_native": 2892,
        "20270603_C:formal_native": 8000,
        "20270603_C:formal_reverse": 2009,
        "20270603_C:evaluation_native": 2892,
        "20270603_D:formal_native": 8000,
        "20270603_D:formal_reverse": 2009,
        "20270603_D:evaluation_native": 2892
      },
      "pending": null,
      "caps": {
        "p0_native": 2000,
        "p0_reverse": 512,
        "detach_native": 512,
        "detach_reverse": 80,
        "engineering_native": 4096,
        "engineering_reverse": 512,
        "formal_native": 96000,
        "formal_reverse": 24200,
        "evaluation_native": 34704,
        "evaluation_reverse": 0
      }
    },
    "workers_retired": true,
    "source_entry_exact": true,
    "audit_model_forwards": 0,
    "allocated_bytes_including_metadata": 413832320,
    "elapsed_seconds": 6535.680000066757
  }
}
```

共24544 optimizer；实际native132386、reverse24728；源SVD1、CPU/CUDA合成SVD2/反向4，正式无SVD/polar。6535.680秒，审计约414MB；初次入口import失败发生于任何医疗调用前，保留记录并在同一原钟内修复依赖路径。一次最终审计零forward。

共享源163、23旧/31新开发验证患者；三个目标流不等于69/93名独立患者。10000次患者配对bootstrap seed163，ED/ES共同抽样；区间不包含源随机、重复开发与多重比较。无封存测试访问。完整逐类/EDES、学生和EMA结果及匿名配对指标在SOURCE_DECISION_PUBLIC_SUMMARY.json。
