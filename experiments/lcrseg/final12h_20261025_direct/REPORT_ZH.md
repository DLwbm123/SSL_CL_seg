# 最后12小时双数据集验证

固定单种子；当前L+U训练；适配不访问旧图；全部是重复使用的开发验证。眼底缺第二REFUGE相机，不称DC²T复现或直接超越。

|数据集|完整|B站点均值%|C站点均值%|论文均值%|C数值距论文≤2pp|旧尾部|最后新域非劣|
|---|---|---:|---:|---:|---|---|---|
|mnms|是|78.9363|80.7578|87.81|False|False|True|
|fundus|是|80.2213|78.2634|83.10|False|False|False|

决策：ABANDON_CURRENT_PROJECT_PERFORMANCE_ROUTE

完整阶段、患者配对CI/下降人数/p10、源门禁和物理成本：

```json
{
  "terminal": {
    "status": "COMPLETE",
    "source_ok": {
      "mnms": true,
      "fundus": true
    },
    "failed": [],
    "counts": {
      "native": 1033278,
      "reverse": 60080,
      "importance_reverse": 384,
      "optimizer": 60080
    },
    "previous_costs": {
      "optimizer": 20016,
      "native": 241904,
      "reverse": 20016,
      "importance_reverse": 64,
      "synthetic_native": 32,
      "synthetic_reverse": 4
    },
    "window_optimizer": 80096,
    "end_epoch": 1792865672.3167067,
    "elapsed_seconds": 9109.316706895828,
    "round_elapsed_seconds": 5255.149872541428,
    "allocated_bytes": 270983168,
    "window_allocated_bytes": 317235200,
    "failure_details_private": false
  },
  "audit_model_forward": 0,
  "optimizer_ledger": 60080,
  "single_seed": 20272401,
  "independent_test": false,
  "current_U_used": true,
  "historical_raw_images_in_adaptation": false,
  "datasets": {
    "mnms": {
      "source": {
        "gate": {
          "status": "PASS_IDENTITY_REUSE",
          "parameter_and_function_exact": true,
          "reused_original_evaluation": true,
          "new_source_optimizer": 0,
          "source_training_sufficiency": "NOT_APPLICABLE_NO_NEW_TRAINING",
          "contrast": {
            "mean_pp": 0.0,
            "CI95_pp": [
              0.0,
              0.0
            ],
            "over1": 0,
            "over2": 0,
            "p10_pp": 0.0,
            "n": 23
          },
          "reference_Dice_percent": 80.36696776879118,
          "RNG_unchanged": true
        },
        "metrics": {
          "Siemens": {
            "Dice_percent": 80.36696776879118,
            "classes_percent": [
              87.1438554785903,
              76.608915232926,
              77.34813259485726
            ],
            "n": 23,
            "p10_Dice_percent": 69.88909318285174
          }
        }
      },
      "stages": {
        "mnms_1_B": {
          "Siemens": {
            "Dice_percent": 68.78021734003302,
            "classes_percent": [
              78.38109969428807,
              67.74245566830429,
              60.217096657506666
            ],
            "n": 23,
            "p10_Dice_percent": 45.20588271326948
          },
          "Philips": {
            "Dice_percent": 83.44464945843532,
            "classes_percent": [
              86.97964941450881,
              80.98741417854349,
              82.36688478225366
            ],
            "n": 31,
            "p10_Dice_percent": 79.59778152242365
          }
        },
        "mnms_1_C": {
          "Siemens": {
            "Dice_percent": 76.47606541063273,
            "classes_percent": [
              83.38201455372472,
              73.39481265633589,
              72.65136902183755
            ],
            "n": 23,
            "p10_Dice_percent": 59.59011225007356
          },
          "Philips": {
            "Dice_percent": 83.76719758890242,
            "classes_percent": [
              88.04917920480173,
              80.99216214466949,
              82.26025141723602
            ],
            "n": 31,
            "p10_Dice_percent": 77.38583551021178
          }
        },
        "mnms_2_B": {
          "Siemens": {
            "Dice_percent": 66.5305854050762,
            "classes_percent": [
              76.65969973084555,
              64.9144935483079,
              58.017562936075166
            ],
            "n": 23,
            "p10_Dice_percent": 35.06677805781229
          },
          "Philips": {
            "Dice_percent": 80.93640784857065,
            "classes_percent": [
              85.60806505435887,
              77.66710897850172,
              79.5340495128514
            ],
            "n": 31,
            "p10_Dice_percent": 74.66328000635374
          },
          "GE": {
            "Dice_percent": 81.10610323787132,
            "classes_percent": [
              87.62561175613284,
              79.26629233268106,
              76.42640562480011
            ],
            "n": 12,
            "p10_Dice_percent": 69.36462774041523
          }
        },
        "mnms_2_C": {
          "Siemens": {
            "Dice_percent": 74.31120420640391,
            "classes_percent": [
              83.10609047218111,
              71.49613246173323,
              68.33138968529745
            ],
            "n": 23,
            "p10_Dice_percent": 58.86092602245083
          },
          "Philips": {
            "Dice_percent": 81.56631905375659,
            "classes_percent": [
              86.16548135773773,
              77.85815516229805,
              80.67532064123402
            ],
            "n": 31,
            "p10_Dice_percent": 73.71969998979581
          },
          "GE": {
            "Dice_percent": 81.49135941748312,
            "classes_percent": [
              88.91815741761894,
              79.79725228410479,
              75.75866855072564
            ],
            "n": 12,
            "p10_Dice_percent": 74.42367624660578
          }
        },
        "mnms_3_B": {
          "Siemens": {
            "Dice_percent": 71.65665867732629,
            "classes_percent": [
              79.36903560903494,
              68.59697254200246,
              67.0039678809414
            ],
            "n": 23,
            "p10_Dice_percent": 53.392410183629636
          },
          "Philips": {
            "Dice_percent": 80.05276738822087,
            "classes_percent": [
              85.3825121594708,
              77.02853929543696,
              77.74725070975485
            ],
            "n": 31,
            "p10_Dice_percent": 72.1687170165204
          },
          "GE": {
            "Dice_percent": 80.59411709842398,
            "classes_percent": [
              87.91950188441923,
              78.84742346408711,
              75.01542594676552
            ],
            "n": 12,
            "p10_Dice_percent": 73.42970787980533
          },
          "Canon": {
            "Dice_percent": 83.44159975305705,
            "classes_percent": [
              90.70557780095719,
              79.93535710821446,
              79.68386434999948
            ],
            "n": 12,
            "p10_Dice_percent": 79.13517126442788
          }
        },
        "mnms_3_C": {
          "Siemens": {
            "Dice_percent": 74.39369353429824,
            "classes_percent": [
              81.95361111256886,
              70.56764689085159,
              70.65982259947425
            ],
            "n": 23,
            "p10_Dice_percent": 59.407089849149074
          },
          "Philips": {
            "Dice_percent": 82.12838322852795,
            "classes_percent": [
              86.13952370453195,
              78.03033181221147,
              82.21529416884043
            ],
            "n": 31,
            "p10_Dice_percent": 75.84391270377046
          },
          "GE": {
            "Dice_percent": 81.47014995491926,
            "classes_percent": [
              87.85166426179028,
              79.13878407393416,
              77.42000152903334
            ],
            "n": 12,
            "p10_Dice_percent": 73.56696041519548
          },
          "Canon": {
            "Dice_percent": 85.03911233852585,
            "classes_percent": [
              91.78373497618638,
              81.14130089150434,
              82.19230114788691
            ],
            "n": 12,
            "p10_Dice_percent": 79.62555460706072
          }
        }
      },
      "paper_reference_mean": 87.81,
      "direct_paper_comparison": false,
      "comparability_reason": "different frozen development-validation split",
      "final": {
        "B": {
          "Siemens": {
            "Dice_percent": 71.65665867732629,
            "classes_percent": [
              79.36903560903494,
              68.59697254200246,
              67.0039678809414
            ],
            "n": 23,
            "p10_Dice_percent": 53.392410183629636
          },
          "Philips": {
            "Dice_percent": 80.05276738822087,
            "classes_percent": [
              85.3825121594708,
              77.02853929543696,
              77.74725070975485
            ],
            "n": 31,
            "p10_Dice_percent": 72.1687170165204
          },
          "GE": {
            "Dice_percent": 80.59411709842398,
            "classes_percent": [
              87.91950188441923,
              78.84742346408711,
              75.01542594676552
            ],
            "n": 12,
            "p10_Dice_percent": 73.42970787980533
          },
          "Canon": {
            "Dice_percent": 83.44159975305705,
            "classes_percent": [
              90.70557780095719,
              79.93535710821446,
              79.68386434999948
            ],
            "n": 12,
            "p10_Dice_percent": 79.13517126442788
          }
        },
        "C": {
          "Siemens": {
            "Dice_percent": 74.39369353429824,
            "classes_percent": [
              81.95361111256886,
              70.56764689085159,
              70.65982259947425
            ],
            "n": 23,
            "p10_Dice_percent": 59.407089849149074
          },
          "Philips": {
            "Dice_percent": 82.12838322852795,
            "classes_percent": [
              86.13952370453195,
              78.03033181221147,
              82.21529416884043
            ],
            "n": 31,
            "p10_Dice_percent": 75.84391270377046
          },
          "GE": {
            "Dice_percent": 81.47014995491926,
            "classes_percent": [
              87.85166426179028,
              79.13878407393416,
              77.42000152903334
            ],
            "n": 12,
            "p10_Dice_percent": 73.56696041519548
          },
          "Canon": {
            "Dice_percent": 85.03911233852585,
            "classes_percent": [
              91.78373497618638,
              81.14130089150434,
              82.19230114788691
            ],
            "n": 12,
            "p10_Dice_percent": 79.62555460706072
          }
        }
      },
      "final_mean": {
        "B": 78.93628572925705,
        "C": 80.75783476406784
      },
      "C_minus_B": {
        "Siemens": {
          "mean_pp": 2.737034856971956,
          "CI95_pp": [
            1.2613789992237554,
            4.2917836722593945
          ],
          "over1": 4,
          "over2": 3,
          "p10_pp": -1.9945353075009464,
          "n": 23
        },
        "Philips": {
          "mean_pp": 2.0756158403070772,
          "CI95_pp": [
            0.8210900900651125,
            3.3115182950121076
          ],
          "over1": 3,
          "over2": 3,
          "p10_pp": 0.1412707447556505,
          "n": 31
        },
        "GE": {
          "mean_pp": 0.8760328564953014,
          "CI95_pp": [
            -1.4488300178028621,
            2.9805936833442512
          ],
          "over1": 3,
          "over2": 2,
          "p10_pp": -4.384907767628762,
          "n": 12
        },
        "Canon": {
          "mean_pp": 1.5975125854688261,
          "CI95_pp": [
            0.9715868000688076,
            2.2579098163033597
          ],
          "over1": 0,
          "over2": 0,
          "p10_pp": 0.38172563068781584,
          "n": 12
        }
      },
      "tails_C_relative_arrival": {
        "Siemens": {
          "mean_pp": -5.973274234492949,
          "CI95_pp": [
            -8.177746435881998,
            -4.013227961593725
          ],
          "over1": 21,
          "over2": 20,
          "p10_pp": -12.952497887542593,
          "n": 23
        },
        "Philips": {
          "mean_pp": -1.6388143603744576,
          "CI95_pp": [
            -2.4650462676277605,
            -0.8192010794824982
          ],
          "over1": 20,
          "over2": 13,
          "p10_pp": -5.29891641229856,
          "n": 31
        },
        "GE": {
          "mean_pp": -0.0212094625638601,
          "CI95_pp": [
            -1.452482772296728,
            1.7502377976617616
          ],
          "over1": 4,
          "over2": 2,
          "p10_pp": -2.076414852343469,
          "n": 12
        }
      },
      "source_warm_vs_final_C": {
        "mean_pp": -5.973274234492949,
        "CI95_pp": [
          -8.177746435881998,
          -4.013227961593725
        ],
        "over1": 21,
        "over2": 20,
        "p10_pp": -12.952497887542593,
        "n": 23
      },
      "numerically_within_2pp": false,
      "old_tail_pass": false,
      "final_new_noninferiority_pass": true,
      "joint_safety_pass": false,
      "arrival_new_noninferiority": {
        "Philips": {
          "mean_pp": 0.3225481304671107,
          "CI95_pp": [
            -0.6159484868732612,
            1.501605903944366
          ],
          "over1": 10,
          "over2": 4,
          "p10_pp": -2.2937331520594517,
          "n": 31
        },
        "GE": {
          "mean_pp": 0.3852561796117806,
          "CI95_pp": [
            -2.0478086007395326,
            2.38292161198111
          ],
          "over1": 2,
          "over2": 2,
          "p10_pp": -3.518556214027951,
          "n": 12
        },
        "Canon": {
          "mean_pp": 1.5975125854688261,
          "CI95_pp": [
            0.9715868000688076,
            2.2579098163033597
          ],
          "over1": 0,
          "over2": 0,
          "p10_pp": 0.38172563068781584,
          "n": 12
        }
      },
      "complete": true
    },
    "fundus": {
      "source": {
        "gate": {
          "status": "PASS_IDENTITY_REUSE",
          "parameter_and_function_exact": true,
          "reused_original_evaluation": true,
          "new_source_optimizer": 0,
          "source_training_sufficiency": "NOT_APPLICABLE_NO_NEW_TRAINING",
          "contrast": {
            "mean_pp": 0.0,
            "CI95_pp": [
              0.0,
              0.0
            ],
            "over1": 0,
            "over2": 0,
            "p10_pp": 0.0,
            "n": 100
          },
          "reference_Dice_percent": 89.90059924655434,
          "RNG_unchanged": true
        },
        "metrics": {
          "REFUGE": {
            "Dice_percent": 89.90059924655434,
            "classes_percent": [
              85.58811431595454,
              94.21308417715406
            ],
            "n": 100,
            "p10_Dice_percent": 85.47581537628591
          }
        }
      },
      "stages": {
        "fundus_1_B": {
          "REFUGE": {
            "Dice_percent": 84.13386343050409,
            "classes_percent": [
              78.1900865389441,
              90.07764032206401
            ],
            "n": 100,
            "p10_Dice_percent": 77.79987856682314
          },
          "RIM_ONE_r3": {
            "Dice_percent": 81.272341772938,
            "classes_percent": [
              74.42736287741522,
              88.11732066846079
            ],
            "n": 40,
            "p10_Dice_percent": 71.00664088549617
          }
        },
        "fundus_1_C": {
          "REFUGE": {
            "Dice_percent": 84.29155285182121,
            "classes_percent": [
              78.53252421696781,
              90.05058148667467
            ],
            "n": 100,
            "p10_Dice_percent": 76.43512049577231
          },
          "RIM_ONE_r3": {
            "Dice_percent": 80.90846778348163,
            "classes_percent": [
              73.75747225661448,
              88.05946331034879
            ],
            "n": 40,
            "p10_Dice_percent": 64.34987442876732
          }
        },
        "fundus_2_B": {
          "REFUGE": {
            "Dice_percent": 86.51186211112098,
            "classes_percent": [
              80.87090330724153,
              92.15282091500043
            ],
            "n": 100,
            "p10_Dice_percent": 79.83405991856682
          },
          "RIM_ONE_r3": {
            "Dice_percent": 76.87363374178051,
            "classes_percent": [
              69.33116254406998,
              84.41610493949105
            ],
            "n": 40,
            "p10_Dice_percent": 55.00163770398031
          },
          "Drishti_GS": {
            "Dice_percent": 77.27828153494147,
            "classes_percent": [
              62.2672842912818,
              92.28927877860112
            ],
            "n": 25,
            "p10_Dice_percent": 67.94371801918221
          }
        },
        "fundus_2_C": {
          "REFUGE": {
            "Dice_percent": 84.38072812969672,
            "classes_percent": [
              82.13872891787587,
              86.6227273415176
            ],
            "n": 100,
            "p10_Dice_percent": 74.75606992298268
          },
          "RIM_ONE_r3": {
            "Dice_percent": 76.62764979001503,
            "classes_percent": [
              67.66167420780666,
              85.59362537222341
            ],
            "n": 40,
            "p10_Dice_percent": 56.871059492709996
          },
          "Drishti_GS": {
            "Dice_percent": 73.78192400083957,
            "classes_percent": [
              60.69100019885608,
              86.8728478028231
            ],
            "n": 25,
            "p10_Dice_percent": 59.71931620238203
          }
        }
      },
      "paper_reference_mean": 83.1,
      "direct_paper_comparison": false,
      "comparability_reason": "three sites; second REFUGE camera absent; development validation",
      "final": {
        "B": {
          "REFUGE": {
            "Dice_percent": 86.51186211112098,
            "classes_percent": [
              80.87090330724153,
              92.15282091500043
            ],
            "n": 100,
            "p10_Dice_percent": 79.83405991856682
          },
          "RIM_ONE_r3": {
            "Dice_percent": 76.87363374178051,
            "classes_percent": [
              69.33116254406998,
              84.41610493949105
            ],
            "n": 40,
            "p10_Dice_percent": 55.00163770398031
          },
          "Drishti_GS": {
            "Dice_percent": 77.27828153494147,
            "classes_percent": [
              62.2672842912818,
              92.28927877860112
            ],
            "n": 25,
            "p10_Dice_percent": 67.94371801918221
          }
        },
        "C": {
          "REFUGE": {
            "Dice_percent": 84.38072812969672,
            "classes_percent": [
              82.13872891787587,
              86.6227273415176
            ],
            "n": 100,
            "p10_Dice_percent": 74.75606992298268
          },
          "RIM_ONE_r3": {
            "Dice_percent": 76.62764979001503,
            "classes_percent": [
              67.66167420780666,
              85.59362537222341
            ],
            "n": 40,
            "p10_Dice_percent": 56.871059492709996
          },
          "Drishti_GS": {
            "Dice_percent": 73.78192400083957,
            "classes_percent": [
              60.69100019885608,
              86.8728478028231
            ],
            "n": 25,
            "p10_Dice_percent": 59.71931620238203
          }
        }
      },
      "final_mean": {
        "B": 80.221259129281,
        "C": 78.26343397351711
      },
      "C_minus_B": {
        "REFUGE": {
          "mean_pp": -2.1311339814242602,
          "CI95_pp": [
            -2.97377701311626,
            -1.290233588444839
          ],
          "over1": 56,
          "over2": 47,
          "p10_pp": -7.732304619646128,
          "n": 100
        },
        "RIM_ONE_r3": {
          "mean_pp": -0.24598395176548316,
          "CI95_pp": [
            -1.9403393585231663,
            1.496371409459898
          ],
          "over1": 16,
          "over2": 13,
          "p10_pp": -6.7290136678427075,
          "n": 40
        },
        "Drishti_GS": {
          "mean_pp": -3.4963575341018998,
          "CI95_pp": [
            -6.293464021955047,
            -0.5973671890755717
          ],
          "over1": 17,
          "over2": 16,
          "p10_pp": -13.254490843451713,
          "n": 25
        }
      },
      "tails_C_relative_arrival": {
        "REFUGE": {
          "mean_pp": -5.519871116857599,
          "CI95_pp": [
            -6.690782820091132,
            -4.440290765472601
          ],
          "over1": 81,
          "over2": 74,
          "p10_pp": -12.114224066744427,
          "n": 100
        },
        "RIM_ONE_r3": {
          "mean_pp": -4.280817993466579,
          "CI95_pp": [
            -6.369816331019974,
            -2.2728664299752928
          ],
          "over1": 26,
          "over2": 25,
          "p10_pp": -10.945730257053793,
          "n": 40
        }
      },
      "source_warm_vs_final_C": {
        "mean_pp": -5.519871116857599,
        "CI95_pp": [
          -6.690782820091132,
          -4.440290765472601
        ],
        "over1": 81,
        "over2": 74,
        "p10_pp": -12.114224066744427,
        "n": 100
      },
      "numerically_within_2pp": false,
      "old_tail_pass": false,
      "final_new_noninferiority_pass": false,
      "joint_safety_pass": false,
      "arrival_new_noninferiority": {
        "RIM_ONE_r3": {
          "mean_pp": -0.36387398945638977,
          "CI95_pp": [
            -1.6522831019634079,
            1.034184701884171
          ],
          "over1": 18,
          "over2": 11,
          "p10_pp": -4.471020214822823,
          "n": 40
        },
        "Drishti_GS": {
          "mean_pp": -3.4963575341018998,
          "CI95_pp": [
            -6.293464021955047,
            -0.5973671890755717
          ],
          "over1": 17,
          "over2": 16,
          "p10_pp": -13.254490843451713,
          "n": 25
        }
      },
      "complete": true
    }
  },
  "previous_window_costs": {
    "optimizer": 20016,
    "native": 241904,
    "reverse": 20016,
    "importance_reverse": 64,
    "synthetic_native": 32,
    "synthetic_reverse": 4
  },
  "cumulative_optimizer": 80096,
  "decision": "ABANDON_CURRENT_PROJECT_PERFORMANCE_ROUTE"
}
```

## 完整结果的路线判断

|数据集|B平均Dice%|C平均Dice%|C−B pp|论文平均%|C与论文数值差pp|
|---|---:|---:|---:|---:|---:|
|心脏四站点|78.936286|80.757835|+1.821549|87.81|−7.052165|
|眼底三站点|80.221259|78.263434|−1.957825|83.10|−4.836566|

心脏C相对B四域均值更好，但Siemens最终74.393694%，对原源−5.973274pp，旧23患者>1/>2=21/20、p10−12.952498pp；Philips对自身到达阶段−1.638814pp，>1/>2=20/13。最后Canon新域C−B +1.597513pp、CI[+.971587,+2.257910]非劣通过，不能覆盖旧尾部失败。Philips/GE各到达时C−B的CI下界也低于−.20，均值正不代替新非劣。

眼底C对B更差：REFUGE−2.131134pp、Drishti−3.496358pp，两配对CI均负；最后新域非劣失败。REFUGE相对原源−5.519871pp，图像键>1/>2=81/74、p10−12.114224；RIM相对到达模型−4.280818pp，26/25、p10−10.945730。图像键不是核实患者，不能称临床患者风险。历史moments/KD/MAS组合不能通用地弥补跨站点遗忘，不能唯一归因于任何单项。

全部10目标各6000正式+8资格完成，60080 optimizer/1033278 native/60080训练反向+384重要性反向，无新失败；两源0新优化和0新val，源恒等检查不是适配成功。轮次5255.150秒，窗口累计9109.317秒、80096opt、317235200B（不含小型本地交付副本）。唯一终态审计PASS，0新模型调用，所有worker/controller退休。两个数据集都未达数值2pp目标，旧尾部与新域非劣也未同时达标，当前配方关闭，不扫权重/强度。

窗口尚有约9小时，若继续，只值得独立检验一个结构上不同的完整路线：公开预训练的更强分割表征，以较小encoder学习率建立新源，源门禁未过仍不适配；在源通过的数据集上对比普通全网微调与保持encoder表征不变的decoder适配/参考蒸馏。其目的为区分当前小型随机预训练U-Net的能力上限和共享表示漂移；不再用历史统计染色或参数锚延续局部修补。需另冻模型预训练来源、源门禁、完整序列/对照和预算，承认ImageNet先验与计算配置并不等同DC²T。
