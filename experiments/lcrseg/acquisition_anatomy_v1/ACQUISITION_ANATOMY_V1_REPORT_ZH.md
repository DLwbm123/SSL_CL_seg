# 外观逆校正与解剖表示 V1：完整源阶段结果

**B/F均完成2000步及唯一源域评估；源门禁STOP，适配未启动。** 一次零forward终态审计PASS，原控制器及worker全部退休。没有目标性能结论。

## 源能力与患者尾部

原FixMatch源Dice80.367407%。全部变化相对同一原源，23患者ED/ES配对bootstrap10000、seed163；Dice单位%，变化和CI单位pp。

|指标|B 普通源|F 完整源|门槛|
|---|---:|---:|---:|
|Dice|80.358878|80.586098|均值差>=−.20pp|
|相对原源均值差|−.008528|+.218691|两者PASS|
|患者配对95%CI|[−.368480,+.367107]|[−.251966,+.744238]|下界>=−.30；B FAIL/F PASS|
|下降>1pp人数|3|2|<=2；B FAIL/F PASS|
|下降>2pp人数|0|1|=0；B PASS/F FAIL|
|患者变化p10|−1.177256|−.606015|>=−.50；两者FAIL|
|完整源能力门禁|FAIL|FAIL|所有项同时通过|

F−B均值+.227219pp，95%CI[−.138572,+.764803]，跨0，不能宣称已证明总体提升。F相对B的患者变化p10−.577251pp，下降>1pp为1人、>2pp0人；这些不能替代对原源的绝对尾部门禁。

|分项Dice %|B|F|
|---|---:|---:|
|LV|87.554691|87.564220|
|MYO|76.363636|76.468606|
|RV|77.158308|77.725468|
|ED|84.626241|84.479817|
|ES|76.091515|76.692378|

## 固定训练批充分性门禁

|EMA固定批总目标|1500|1750|2000|相邻规范化变化|结论|
|---|---:|---:|---:|---|---|
|B|.086059|.079189|.073203|.006870/.005986|PASS|
|F|.678610|.646946|.627701|.031664/.019245|FAIL|

规范化变化严格按冻结公式 abs(b−a)/max(1,abs(a))，两次均<=.01；不是实际损失的相对百分比。B达到预声明稳定标准但不表示全局最优；F未满足训练充分性标准，禁止延长2000步或放宽门槛。

F的EMA任务损失.087853→.081035→.074792；解剖损失.086556→.063126→.048798；逆校正损失.504201→.502785→.504111。总目标仍下降主要涉及任务/解剖项，逆校正项在约.50附近，不能从此断定真实扫描仪校正已学会，也不能据此证明唯一因果机制。源F均值未下降，但一名患者下降超过2pp且p10未通过；均值改善不等于患者尾部安全。

## 成本与停止

实际4064optimizer（两源4000+资格64）、17768医疗native样本前向、4080reverse；合成另24native/7reverse。管理历时3768.988881秒，终态审计保守占用78,763,136B，均在原8h/<4GB内。适配0opt/0forward，未读取目标训练或测试。本轮不补seed、不重训、不重评、不重审计、不自动后继。

单种子、共享源和重复开发验证，CI不覆盖训练随机性/多重比较；不是独立测试。后验路线未复活。完整机器可读结果如下。

# 外观逆校正与解剖表示 V1 完整结果

工程审计PASS；未进入/未完成适配，不能作适配性能结论。

```json
{
  "status": "SOURCE_STOP",
  "audit": {
    "status": "PASS",
    "audit_model_forward": 0,
    "workers_retired": true,
    "counters": {
      "calls": {
        "transform_samples": 8336,
        "spline_samples": 52200,
        "bias_samples": 26100,
        "source_qualification_samples": 296,
        "appearance_samples": 17764,
        "anatomy_samples": 33352,
        "gt_occupancy_samples": 16672,
        "source_qualification_reverse": 68,
        "source_formal_samples": 16000,
        "source_formal_reverse": 4012,
        "source_diagnostic_samples": 384,
        "source_evaluation_samples": 1088
      },
      "optimizer": {
        "source_B_qualification": 32,
        "source_F_qualification": 32,
        "source_B_formal": 2000,
        "source_F_formal": 2000
      },
      "pending_optimizer": null
    },
    "resources": {
      "elapsed_seconds": 3768.988880634308,
      "allocated_bytes_including_ops": 78763136
    },
    "source_complete": true,
    "target_complete": false,
    "synthetic_calls": {
      "reverse": 7,
      "appearance_samples": 30,
      "spline_samples": 96,
      "bias_samples": 44,
      "native_samples": 24,
      "anatomy_samples": 44,
      "transform_samples": 14,
      "gt_occupancy_samples": 20
    }
  },
  "source_gate": {
    "status": "STOP",
    "checks": {
      "B_training_sufficient": true,
      "B_source_capability": false,
      "F_training_sufficient": false,
      "F_source_capability": false
    },
    "source_vs_original": {
      "B": {
        "mean_pp": -0.008528310595755794,
        "CI95_pp": [
          -0.3684804685821973,
          0.3671070759016977
        ],
        "p10_pp": -1.1772561737208376,
        "over1": 3,
        "over2": 0,
        "patients": 23
      },
      "F": {
        "mean_pp": 0.218691082858781,
        "CI95_pp": [
          -0.25196649455523884,
          0.7442384255741262
        ],
        "p10_pp": -0.6060150140712084,
        "over1": 2,
        "over2": 1,
        "patients": 23
      }
    },
    "training_diagnostics": {
      "B": {
        "last_three_EMA_objective": [
          0.086058621889606,
          0.07918853526825842,
          0.07320272531251248
        ],
        "relative_changes": [
          0.006870086621347582,
          0.0059858099557459354
        ],
        "components": [
          {
            "step": 1500,
            "batch_reads": 8,
            "native_samples": 64,
            "student": {
              "task_loss": 0.06813339014843223,
              "anatomy_loss": 1.3862943649291992,
              "inverse_loss": 0.4991658255457878,
              "total_loss": 0.06813339014843223,
              "correction_MSE": 1.272122962525657e-08
            },
            "EMA": {
              "task_loss": 0.086058621889606,
              "anatomy_loss": 1.3862943649291992,
              "inverse_loss": 0.49949314817786217,
              "total_loss": 0.086058621889606,
              "correction_MSE": 2.165175802537611e-09
            }
          },
          {
            "step": 1750,
            "batch_reads": 8,
            "native_samples": 64,
            "student": {
              "task_loss": 0.05910539695832995,
              "anatomy_loss": 1.3862943649291992,
              "inverse_loss": 0.49903279170393944,
              "total_loss": 0.05910539695832995,
              "correction_MSE": 1.937018523356926e-08
            },
            "EMA": {
              "task_loss": 0.07918853526825842,
              "anatomy_loss": 1.3862943649291992,
              "inverse_loss": 0.4994005784392357,
              "total_loss": 0.07918853526825842,
              "correction_MSE": 4.161680999814621e-09
            }
          },
          {
            "step": 2000,
            "batch_reads": 8,
            "native_samples": 64,
            "student": {
              "task_loss": 0.056055227181786904,
              "anatomy_loss": 1.3862943649291992,
              "inverse_loss": 0.4991893619298935,
              "total_loss": 0.056055227181786904,
              "correction_MSE": 1.6674160385754533e-08
            },
            "EMA": {
              "task_loss": 0.07320272531251248,
              "anatomy_loss": 1.3862943649291992,
              "inverse_loss": 0.49934251233935356,
              "total_loss": 0.07320272531251248,
              "correction_MSE": 6.1838665299518425e-09
            }
          }
        ]
      },
      "F": {
        "last_three_EMA_objective": [
          0.6786095276474953,
          0.6469459906220436,
          0.6277012228965759
        ],
        "relative_changes": [
          0.03166353702545166,
          0.019244767725467682
        ],
        "components": [
          {
            "step": 1500,
            "batch_reads": 8,
            "native_samples": 64,
            "student": {
              "task_loss": 0.0756601577904803,
              "anatomy_loss": 0.027514207758940756,
              "inverse_loss": 0.5007116533815861,
              "total_loss": 0.6038860194385052,
              "correction_MSE": 1.4868385164845677e-05
            },
            "EMA": {
              "task_loss": 0.08785275878062748,
              "anatomy_loss": 0.0865559671074152,
              "inverse_loss": 0.5042007975280285,
              "total_loss": 0.6786095276474953,
              "correction_MSE": 5.821404784001061e-06
            }
          },
          {
            "step": 1750,
            "batch_reads": 8,
            "native_samples": 64,
            "student": {
              "task_loss": 0.06037480270060769,
              "anatomy_loss": 0.024795937119051814,
              "inverse_loss": 0.4953358359634876,
              "total_loss": 0.5805065631866455,
              "correction_MSE": 8.466456961286895e-06
            },
            "EMA": {
              "task_loss": 0.081034696082952,
              "anatomy_loss": 0.06312606437131763,
              "inverse_loss": 0.5027852356433868,
              "total_loss": 0.6469459906220436,
              "correction_MSE": 6.352860339120525e-06
            }
          },
          {
            "step": 2000,
            "batch_reads": 8,
            "native_samples": 64,
            "student": {
              "task_loss": 0.05708194706357972,
              "anatomy_loss": 0.022125854913610965,
              "inverse_loss": 0.503950797021389,
              "total_loss": 0.5831585861742496,
              "correction_MSE": 1.7831920558819547e-05
            },
            "EMA": {
              "task_loss": 0.07479233996855328,
              "anatomy_loss": 0.048798185773193836,
              "inverse_loss": 0.5041106976568699,
              "total_loss": 0.6277012228965759,
              "correction_MSE": 7.702635059558816e-06
            }
          }
        ]
      }
    },
    "original_source_score_percent": 80.36740663770262,
    "thresholds_unchanged": true
  },
  "metrics": {
    "source": {
      "B": {
        "Siemens": {
          "primary_dice": 0.8035887832710685,
          "classes": {
            "LV": 0.8755469133223458,
            "MYO": 0.763636357652282,
            "RV": 0.7715830788385772
          },
          "phases": {
            "ED": {
              "primary_dice": 0.8462624120825464,
              "classes": {
                "LV": 0.9069944182071109,
                "MYO": 0.7946168091021747,
                "RV": 0.8371760089383542
              }
            },
            "ES": {
              "primary_dice": 0.7609151544595902,
              "classes": {
                "LV": 0.8440994084375811,
                "MYO": 0.7326559062023894,
                "RV": 0.7059901487388008
              }
            }
          },
          "patients": 23,
          "volumes": 46,
          "forward_slices": 544
        }
      },
      "F": {
        "Siemens": {
          "primary_dice": 0.8058609772056137,
          "classes": {
            "LV": 0.8756421985718481,
            "MYO": 0.764686056135414,
            "RV": 0.7772546769095792
          },
          "phases": {
            "ED": {
              "primary_dice": 0.8447981722694875,
              "classes": {
                "LV": 0.9049964932220464,
                "MYO": 0.794965096331089,
                "RV": 0.834432927255327
              }
            },
            "ES": {
              "primary_dice": 0.7669237821417402,
              "classes": {
                "LV": 0.8462879039216503,
                "MYO": 0.7344070159397388,
                "RV": 0.7200764265638316
              }
            }
          },
          "patients": 23,
          "volumes": 46,
          "forward_slices": 544
        }
      }
    }
  },
  "patient_tails": {},
  "contrasts_C_minus": {},
  "checks": {},
  "composite_pass": false,
  "primary": "C final EMA .999",
  "single_seed": 20271701,
  "development_validation_reused": true,
  "independent_test": false
}
```

单种子、共享源及重复开发患者。训练稳定门禁不证明全局最优；外观扰动不等于真实扫描仪变化。适配不读取旧图。失败成本保留，不重训重评重审计。
