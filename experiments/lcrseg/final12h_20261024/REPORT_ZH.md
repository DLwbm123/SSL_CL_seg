# 最后12小时双数据集验证

固定单种子；当前L+U训练；适配不访问旧图；全部是重复使用的开发验证。眼底缺第二REFUGE相机，不称DC²T复现或直接超越。

|数据集|完整|B站点均值%|C站点均值%|论文均值%|C数值距论文≤2pp|旧尾部|最后新域非劣|
|---|---|---:|---:|---:|---|---|---|
|mnms|否|未完成|未完成|87.81|不可判定|不可判定|不可判定|
|fundus|否|未完成|未完成|83.10|不可判定|不可判定|不可判定|

决策：INCOMPLETE_OR_SOURCE_STOP_NO_SUCCESS_CLAIM

完整阶段、患者配对CI/下降人数/p10、源门禁和物理成本：

```json
{
  "terminal": {
    "status": "FINISHED_WITH_STOPS",
    "source_ok": {
      "mnms": false,
      "fundus": false
    },
    "failed": {},
    "counts": {
      "native": 241904,
      "reverse": 20016,
      "importance_reverse": 64,
      "optimizer": 20016
    },
    "end_epoch": 1792858995.6900084,
    "elapsed_seconds": 2432.6900086402893,
    "allocated_bytes": 46149632
  },
  "audit_model_forward": 0,
  "optimizer_ledger": 20016,
  "single_seed": 20272401,
  "independent_test": false,
  "current_U_used": true,
  "historical_raw_images_in_adaptation": false,
  "datasets": {
    "mnms": {
      "source": {
        "gate": {
          "status": "SOURCE_STOP",
          "contrast": {
            "mean_pp": -2.219063429431637,
            "CI95_pp": [
              -3.7046802508499197,
              -0.7014453439878785
            ],
            "over1": 15,
            "over2": 11,
            "p10_pp": -5.68233738343932,
            "n": 23
          },
          "min_Dice_percent": 80.06740663770262
        },
        "metrics": {
          "Siemens": {
            "Dice_percent": 78.14790433935951,
            "classes_percent": [
              85.00066759353247,
              75.7031799551158,
              73.7398654694303
            ],
            "n": 23,
            "p10_Dice_percent": 68.05320113645993
          }
        }
      },
      "stages": {},
      "paper_reference_mean": 87.81,
      "direct_paper_comparison": false,
      "comparability_reason": "different frozen development-validation split",
      "complete": false
    },
    "fundus": {
      "source": {
        "gate": {
          "status": "SOURCE_STOP",
          "contrast": {
            "mean_pp": -0.02341761619348383,
            "CI95_pp": [
              -0.23076121061885949,
              0.1746169797111098
            ],
            "over1": 16,
            "over2": 3,
            "p10_pp": -1.23194519208506,
            "n": 100
          },
          "min_Dice_percent": 89.29755965874404
        },
        "metrics": {
          "REFUGE": {
            "Dice_percent": 89.87718163036087,
            "classes_percent": [
              85.65960057684774,
              94.09476268387395
            ],
            "n": 100,
            "p10_Dice_percent": 85.32656101376486
          }
        }
      },
      "stages": {},
      "paper_reference_mean": 83.1,
      "direct_paper_comparison": false,
      "comparability_reason": "three sites; second REFUGE camera absent; development validation",
      "complete": false
    }
  },
  "decision": "INCOMPLETE_OR_SOURCE_STOP_NO_SUCCESS_CLAIM"
}
```

## 首轮源能力完整结论与后继依据

|数据集|本轮warm参考Dice%|追加源训练后Dice%|配对差pp [95%CI]|下降>1pp/>2pp|变化p10 pp|结论|
|---|---:|---:|---|---|---:|---|
|心脏Siemens|80.366968|78.147904|−2.219063 [−3.704680,−.701445]|15/11|−5.682337|源能力失败|
|眼底REFUGE|89.900599|89.877182|−.023418 [−.230761,+.174617]|16/3|−1.231945|均值接近不覆盖图像键尾部失败|

两个源分别完成12000/8000正式更新、各8资格，均EXIT0；实际20016 optimizer、241904 native样本（含32合成）、20016训练反向+64重要性反向，另4合成反向；总历时2432.690秒，终态根46149632B。原控制器/worker退休。唯一终态审计PASS，0新模型调用；没有进入任一目标适配。不能称完整持续学习阴性或已接近论文。

已核对实际原生暖启动加载、头几何与标签语义；代码按本轮明确的Dice+CE/均值教师/Adam源再训练执行，未发现足以解释全部结果的加载错误。源重训会改变优化目标、优化器和训练轨迹，当前证据支持停止这套源再优化，不能唯一归因为过拟合、U有害或某种网络错误。

剩余窗口继续的独立完整路线应从未改动的既有强源直接开始，使用参数/函数保持门禁证明源能力精确继承，不利用失败源权重、不放宽旧门槛、不重挑checkpoint；随后再验证完整当前L+U的B/C序列。原始80.37/89.90源已经存在，这一对照能区分“源再训练损坏”与“无旧图适配本身是否仍有可用性能”。该路线必须另冻协议/预算，不回填本轮SOURCE_STOP，不声称新学到更强源。
