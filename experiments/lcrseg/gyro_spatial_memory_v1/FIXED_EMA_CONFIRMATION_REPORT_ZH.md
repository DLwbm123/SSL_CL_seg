# 固定EMA部署：三个新目标流确认

固定第2000步EMA(.999)主部署，最后student配对对照。共享源163/S†69与开发验证患者，不是独立源或最终测试。原源旧域80.36740663770262%，源自身报告primary EMA。

## 绝对性能

|流|学生旧%|学生新%|EMA旧%|EMA新%|综合|
|---|---:|---:|---:|---:|---|
|20270501|79.389510|83.471053|80.950047|83.293866|False|
|20270502|79.899442|83.375523|80.978531|83.278634|False|
|20270503|79.574368|83.072987|81.034085|83.315949|False|

## 冻结门槛与实测

```json
{
  "composite_pass": false,
  "per_stream": {
    "20270501": {
      "composite_pass": false,
      "checks": {
        "old_absolute": true,
        "old_interval": true,
        "new_point": true,
        "new_interval": false,
        "new_absolute": true,
        "tail_over_1": false,
        "tail_over_2": false,
        "tail_p10": false,
        "tail_no_worse_than_student": true
      },
      "EMA_old_percent": 80.9500470186399,
      "EMA_new_percent": 83.2938664428635,
      "old_source_delta_pp": 0.582640380937314,
      "old_source_CI": [
        -0.2868021789860686,
        1.5079455158278943
      ],
      "new_student_delta_pp": -0.17718669930124642,
      "new_student_CI": [
        -0.6246162783369706,
        0.3868835745317013
      ],
      "source_tail_counts": {
        "0.5": 7,
        "1": 5,
        "2": 1
      },
      "source_tail_p10_pp": -1.3967303869857206
    },
    "20270502": {
      "composite_pass": false,
      "checks": {
        "old_absolute": true,
        "old_interval": false,
        "new_point": true,
        "new_interval": false,
        "new_absolute": true,
        "tail_over_1": false,
        "tail_over_2": false,
        "tail_p10": false,
        "tail_no_worse_than_student": true
      },
      "EMA_old_percent": 80.97853052286814,
      "EMA_new_percent": 83.27863395909817,
      "old_source_delta_pp": 0.6111238851655441,
      "old_source_CI": [
        -0.30217655678854843,
        1.5873234008946704
      ],
      "new_student_delta_pp": -0.0968891809944893,
      "new_student_CI": [
        -0.5277974133797272,
        0.4414373951984055
      ],
      "source_tail_counts": {
        "0.5": 7,
        "1": 5,
        "2": 1
      },
      "source_tail_p10_pp": -1.3957835232732285
    },
    "20270503": {
      "composite_pass": false,
      "checks": {
        "old_absolute": true,
        "old_interval": true,
        "new_point": true,
        "new_interval": false,
        "new_absolute": true,
        "tail_over_1": false,
        "tail_over_2": false,
        "tail_p10": false,
        "tail_no_worse_than_student": true
      },
      "EMA_old_percent": 81.03408512681274,
      "EMA_new_percent": 83.31594911687705,
      "old_source_delta_pp": 0.6666784891101373,
      "old_source_CI": [
        -0.19132299597520372,
        1.5875330327579977
      ],
      "new_student_delta_pp": 0.24296245331892105,
      "new_student_CI": [
        -0.29632773328928974,
        0.9347303251411457
      ],
      "source_tail_counts": {
        "0.5": 7,
        "1": 3,
        "2": 1
      },
      "source_tail_p10_pp": -1.3289720433594487
    }
  }
}
```

## 原源参照患者尾部

|流/模型|源差均值pp [95%CI]|p10pp|下降>.5/1/2pp|改善/下降|入口差pp|
|---|---|---:|---|---|---:|
|20270501/student|-0.977897 [-2.108309,+0.122777]|-3.202801|13/13/9|6/17|-0.044204|
|20270501/EMA|+0.582640 [-0.286802,+1.507946]|-1.396730|7/5/1|13/10|+1.516333|
|20270502/student|-0.467964 [-1.577939,+0.644979]|-2.915016|12/11/3|9/14|+0.465728|
|20270502/EMA|+0.611124 [-0.302177,+1.587323]|-1.395784|7/5/1|13/10|+1.544816|
|20270503/student|-0.793039 [-1.903625,+0.311625]|-3.157715|13/12/8|8/15|+0.140654|
|20270503/EMA|+0.666678 [-0.191323,+1.587533]|-1.328972|7/3/1|13/10|+1.600371|

## 完整配对与类别/EDES

```json
{
  "EMA_minus_student": {
    "20270501": {
      "Siemens": {
        "mean_pp": 1.5605371771301102,
        "median_pp": 1.4018384111153193,
        "p10_pp": 0.39500466615145724,
        "increased": 23,
        "decreased": 0,
        "decline_over_pp": {
          "0.5": 0,
          "1": 0,
          "2": 0
        },
        "leave_one_mean_range_pp": [
          1.4572154812525688,
          1.6292767611196852
        ],
        "paired": {
          "delta_B_minus_A_pp": 1.5605371771301082,
          "classes_pp": {
            "LV": 0.9360020691679615,
            "MYO": 1.7662782550977967,
            "RV": 1.979331207124566
          },
          "patients": 23,
          "increased": 23,
          "decreased": 0,
          "unchanged": 0,
          "median_patient_delta_pp": 1.4018384111153102,
          "patient_bootstrap95_pp": [
            1.1684947828132515,
            1.9711651250736932
          ]
        },
        "phase_pp": {
          "ED": 1.4671615933898097,
          "ES": 1.6539127608704018
        }
      },
      "Philips": {
        "mean_pp": -0.17718669930124642,
        "median_pp": -0.4779516780897275,
        "p10_pp": -0.9664392395253185,
        "increased": 8,
        "decreased": 23,
        "decline_over_pp": {
          "0.5": 15,
          "1": 2,
          "2": 1
        },
        "leave_one_mean_range_pp": [
          -0.3752254801596011,
          -0.10976142245761747
        ],
        "paired": {
          "delta_B_minus_A_pp": -0.17718669930124403,
          "classes_pp": {
            "LV": -0.23457458127642744,
            "MYO": -0.2920828395270221,
            "RV": -0.00490267710028256
          },
          "patients": 31,
          "increased": 8,
          "decreased": 23,
          "unchanged": 0,
          "median_patient_delta_pp": -0.47795167808971917,
          "patient_bootstrap95_pp": [
            -0.6246162783369706,
            0.3868835745317013
          ]
        },
        "phase_pp": {
          "ED": 0.011964468602176925,
          "ES": -0.3663378672046802
        }
      }
    },
    "20270502": {
      "Siemens": {
        "mean_pp": 1.0790883543117196,
        "median_pp": 0.8303151987038238,
        "p10_pp": 0.06282264777937475,
        "increased": 23,
        "decreased": 0,
        "decline_over_pp": {
          "0.5": 0,
          "1": 0,
          "2": 0
        },
        "leave_one_mean_range_pp": [
          1.0111761432187374,
          1.1276550644185095
        ],
        "paired": {
          "delta_B_minus_A_pp": 1.0790883543117218,
          "classes_pp": {
            "LV": 0.7189558395775629,
            "MYO": 1.2596590094706441,
            "RV": 1.2586502138869586
          },
          "patients": 23,
          "increased": 23,
          "decreased": 0,
          "unchanged": 0,
          "median_patient_delta_pp": 0.8303151987038367,
          "patient_bootstrap95_pp": [
            0.7418940630690072,
            1.4373826809379702
          ]
        },
        "phase_pp": {
          "ED": 1.0111110602889495,
          "ES": 1.1470656483344888
        }
      },
      "Philips": {
        "mean_pp": -0.0968891809944893,
        "median_pp": -0.4281556495814409,
        "p10_pp": -0.8500361449485849,
        "increased": 9,
        "decreased": 22,
        "decline_over_pp": {
          "0.5": 14,
          "1": 2,
          "2": 1
        },
        "leave_one_mean_range_pp": [
          -0.29640587359776893,
          -0.019547099650686364
        ],
        "paired": {
          "delta_B_minus_A_pp": -0.09688918099449104,
          "classes_pp": {
            "LV": -0.02542571490276358,
            "MYO": -0.3864701526544091,
            "RV": 0.12122832457369956
          },
          "patients": 31,
          "increased": 9,
          "decreased": 22,
          "unchanged": 0,
          "median_patient_delta_pp": -0.42815564958144464,
          "patient_bootstrap95_pp": [
            -0.5277974133797272,
            0.4414373951984055
          ]
        },
        "phase_pp": {
          "ED": -0.1233102414508691,
          "ES": -0.07046812053812523
        }
      }
    },
    "20270503": {
      "Siemens": {
        "mean_pp": 1.4597170328792748,
        "median_pp": 1.1182869280230823,
        "p10_pp": 0.22401301241406385,
        "increased": 22,
        "decreased": 1,
        "decline_over_pp": {
          "0.5": 0,
          "1": 0,
          "2": 0
        },
        "leave_one_mean_range_pp": [
          1.3757077984792756,
          1.5359883676068125
        ],
        "paired": {
          "delta_B_minus_A_pp": 1.459717032879277,
          "classes_pp": {
            "LV": 1.0671015641619876,
            "MYO": 1.3735278855388717,
            "RV": 1.9385216489369719
          },
          "patients": 23,
          "increased": 22,
          "decreased": 1,
          "unchanged": 0,
          "median_patient_delta_pp": 1.1182869280230747,
          "patient_bootstrap95_pp": [
            1.0386192532391822,
            1.8845327503603466
          ]
        },
        "phase_pp": {
          "ED": 1.3693215863167518,
          "ES": 1.5501124794417898
        }
      },
      "Philips": {
        "mean_pp": 0.24296245331892105,
        "median_pp": -0.06120437290964009,
        "p10_pp": -0.7628190955499825,
        "increased": 13,
        "decreased": 18,
        "decline_over_pp": {
          "0.5": 8,
          "1": 2,
          "2": 1
        },
        "leave_one_mean_range_pp": [
          -0.022382200945138743,
          0.33401896740086084
        ],
        "paired": {
          "delta_B_minus_A_pp": 0.24296245331891977,
          "classes_pp": {
            "LV": 0.1052273489833598,
            "MYO": -0.1396132118172658,
            "RV": 0.7632732227906652
          },
          "patients": 31,
          "increased": 13,
          "decreased": 18,
          "unchanged": 0,
          "median_patient_delta_pp": -0.06120437290964009,
          "patient_bootstrap95_pp": [
            -0.29632773328928974,
            0.9347303251411457
          ]
        },
        "phase_pp": {
          "ED": 0.41072850880340406,
          "ES": 0.07519639783445387
        }
      }
    }
  },
  "source_comparisons": {
    "20270501": {
      "student": {
        "versus_original_source": {
          "mean_pp": -0.9778967961927962,
          "median_pp": -1.239430617769166,
          "p10_pp": -3.2028009344563535,
          "increased": 6,
          "decreased": 17,
          "decline_over_pp": {
            "0.5": 13,
            "1": 13,
            "2": 9
          },
          "leave_one_mean_range_pp": [
            -1.2409357632067868,
            -0.6421296237898789
          ],
          "paired": {
            "delta_B_minus_A_pp": -0.9778967961927971,
            "classes_pp": {
              "LV": -0.7913861690040725,
              "MYO": -1.3750489074055343,
              "RV": -0.7672553121687834
            },
            "patients": 23,
            "increased": 6,
            "decreased": 17,
            "unchanged": 0,
            "median_patient_delta_pp": -1.2394306177691605,
            "patient_bootstrap95_pp": [
              -2.1083088733984447,
              0.1227774478925218
            ]
          },
          "phase_pp": {
            "ED": -1.3541747254891656,
            "ES": -0.6016188668964362
          }
        },
        "versus_entry": {
          "mean_pp": -0.04420442970973475,
          "median_pp": -0.4884121490471016,
          "p10_pp": -2.341509761926126,
          "increased": 10,
          "decreased": 13,
          "decline_over_pp": {
            "0.5": 11,
            "1": 10,
            "2": 3
          },
          "leave_one_mean_range_pp": [
            -0.36928619953035474,
            0.24878375619843512
          ],
          "paired": {
            "delta_B_minus_A_pp": -0.04420442970973356,
            "classes_pp": {
              "LV": -0.7506607515851922,
              "MYO": -0.8551639796802427,
              "RV": 1.4732114421362341
            },
            "patients": 23,
            "increased": 10,
            "decreased": 13,
            "unchanged": 0,
            "median_patient_delta_pp": -0.4884121490470887,
            "patient_bootstrap95_pp": [
              -1.1722018041324795,
              1.1558555903483867
            ]
          }
        }
      },
      "EMA": {
        "versus_original_source": {
          "mean_pp": 0.582640380937314,
          "median_pp": 0.6557672829772265,
          "p10_pp": -1.3967303869857206,
          "increased": 13,
          "decreased": 10,
          "decline_over_pp": {
            "0.5": 7,
            "1": 5,
            "2": 1
          },
          "leave_one_mean_range_pp": [
            0.2871192674032038,
            0.8150858574626899
          ],
          "paired": {
            "delta_B_minus_A_pp": 0.582640380937311,
            "classes_pp": {
              "LV": 0.14461590016388895,
              "MYO": 0.39122934769226225,
              "RV": 1.2120758949557824
            },
            "patients": 23,
            "increased": 13,
            "decreased": 10,
            "unchanged": 0,
            "median_patient_delta_pp": 0.6557672829772228,
            "patient_bootstrap95_pp": [
              -0.2868021789860686,
              1.5079455158278943
            ]
          },
          "phase_pp": {
            "ED": 0.11298686790064405,
            "ES": 1.0522938939739657
          }
        },
        "versus_entry": {
          "mean_pp": 1.5163327474203752,
          "median_pp": 1.1363694678887537,
          "p10_pp": -0.7112072836627602,
          "increased": 15,
          "decreased": 8,
          "decline_over_pp": {
            "0.5": 3,
            "1": 2,
            "2": 1
          },
          "leave_one_mean_range_pp": [
            1.1587688310796358,
            1.7059992374510038
          ],
          "paired": {
            "delta_B_minus_A_pp": 1.5163327474203743,
            "classes_pp": {
              "LV": 0.18534131758276923,
              "MYO": 0.9111142754175539,
              "RV": 3.4525426492607996
            },
            "patients": 23,
            "increased": 15,
            "decreased": 8,
            "unchanged": 0,
            "median_patient_delta_pp": 1.1363694678887426,
            "patient_bootstrap95_pp": [
              0.5638178220026522,
              2.6129572106191414
            ]
          }
        }
      }
    },
    "20270502": {
      "student": {
        "versus_original_source": {
          "mean_pp": -0.46796446914617545,
          "median_pp": -0.89648199496416,
          "p10_pp": -2.9150159300239986,
          "increased": 9,
          "decreased": 14,
          "decline_over_pp": {
            "0.5": 12,
            "1": 11,
            "2": 3
          },
          "leave_one_mean_range_pp": [
            -0.7941755350279421,
            -0.15483087451344524
          ],
          "paired": {
            "delta_B_minus_A_pp": -0.46796446914617806,
            "classes_pp": {
              "LV": -0.3923499293638808,
              "MYO": -0.8176803810952008,
              "RV": -0.19386309697945217
            },
            "patients": 23,
            "increased": 9,
            "decreased": 14,
            "unchanged": 0,
            "median_patient_delta_pp": -0.8964819949641619,
            "patient_bootstrap95_pp": [
              -1.577939401544409,
              0.6449790293959748
            ]
          },
          "phase_pp": {
            "ED": -0.9668456448409746,
            "ES": 0.03091670654860046
          }
        },
        "versus_entry": {
          "mean_pp": 0.46572789733688585,
          "median_pp": 0.0034798800092827875,
          "p10_pp": -1.7097987772098833,
          "increased": 13,
          "decreased": 10,
          "decline_over_pp": {
            "0.5": 10,
            "1": 6,
            "2": 2
          },
          "leave_one_mean_range_pp": [
            0.0774740286484898,
            0.7360825054748686
          ],
          "paired": {
            "delta_B_minus_A_pp": 0.4657278973368854,
            "classes_pp": {
              "LV": -0.35162451194500055,
              "MYO": -0.29779545336990926,
              "RV": 2.0466036573255657
            },
            "patients": 23,
            "increased": 13,
            "decreased": 10,
            "unchanged": 0,
            "median_patient_delta_pp": 0.0034798800092920392,
            "patient_bootstrap95_pp": [
              -0.6508236512057379,
              1.697528511571302
            ]
          }
        }
      },
      "EMA": {
        "versus_original_source": {
          "mean_pp": 0.6111238851655441,
          "median_pp": 0.7524745618017414,
          "p10_pp": -1.3957835232732285,
          "increased": 13,
          "decreased": 10,
          "decline_over_pp": {
            "0.5": 7,
            "1": 5,
            "2": 1
          },
          "leave_one_mean_range_pp": [
            0.3038120198569037,
            0.8595217568909761
          ],
          "paired": {
            "delta_B_minus_A_pp": 0.6111238851655441,
            "classes_pp": {
              "LV": 0.32660591021368207,
              "MYO": 0.44197862837544344,
              "RV": 1.0647871169075065
            },
            "patients": 23,
            "increased": 13,
            "decreased": 10,
            "unchanged": 0,
            "median_patient_delta_pp": 0.7524745618017248,
            "patient_bootstrap95_pp": [
              -0.30217655678854843,
              1.5873234008946704
            ]
          },
          "phase_pp": {
            "ED": 0.04426541544797491,
            "ES": 1.1779823548830892
          }
        },
        "versus_entry": {
          "mean_pp": 1.5448162516486057,
          "median_pp": 1.1796449845139612,
          "p10_pp": -0.7122811953180919,
          "increased": 15,
          "decreased": 8,
          "decline_over_pp": {
            "0.5": 3,
            "1": 2,
            "2": 1
          },
          "leave_one_mean_range_pp": [
            1.175461583533336,
            1.7504351368792903
          ],
          "paired": {
            "delta_B_minus_A_pp": 1.5448162516486073,
            "classes_pp": {
              "LV": 0.3673313276325624,
              "MYO": 0.9618635561007352,
              "RV": 3.305253871212524
            },
            "patients": 23,
            "increased": 15,
            "decreased": 8,
            "unchanged": 0,
            "median_patient_delta_pp": 1.1796449845139743,
            "patient_bootstrap95_pp": [
              0.5650970481294081,
              2.674073784768977
            ]
          }
        }
      }
    },
    "20270503": {
      "student": {
        "versus_original_source": {
          "mean_pp": -0.7930385437691376,
          "median_pp": -1.0640093730992617,
          "p10_pp": -3.1577147872799793,
          "increased": 8,
          "decreased": 15,
          "decline_over_pp": {
            "0.5": 13,
            "1": 12,
            "2": 8
          },
          "leave_one_mean_range_pp": [
            -1.072600442071912,
            -0.4889330992039008
          ],
          "paired": {
            "delta_B_minus_A_pp": -0.7930385437691395,
            "classes_pp": {
              "LV": -0.7730825387145029,
              "MYO": -0.9126675293591796,
              "RV": -0.6933655632337361
            },
            "patients": 23,
            "increased": 8,
            "decreased": 15,
            "unchanged": 0,
            "median_patient_delta_pp": -1.0640093730992692,
            "patient_bootstrap95_pp": [
              -1.90362483851145,
              0.31162515923072465
            ]
          },
          "phase_pp": {
            "ED": -1.1967074293511737,
            "ES": -0.389369658187122
          }
        },
        "versus_entry": {
          "mean_pp": 0.14065382271392385,
          "median_pp": -0.6076638708157445,
          "p10_pp": -1.9240145117633745,
          "increased": 11,
          "decreased": 12,
          "decline_over_pp": {
            "0.5": 12,
            "1": 9,
            "2": 2
          },
          "leave_one_mean_range_pp": [
            -0.20095087839547982,
            0.40198028078441317
          ],
          "paired": {
            "delta_B_minus_A_pp": 0.14065382271392363,
            "classes_pp": {
              "LV": -0.7323571212956227,
              "MYO": -0.39278260163388795,
              "RV": 1.5471011910712817
            },
            "patients": 23,
            "increased": 11,
            "decreased": 12,
            "unchanged": 0,
            "median_patient_delta_pp": -0.6076638708157361,
            "patient_bootstrap95_pp": [
              -0.9594850657231762,
              1.3408745030497333
            ]
          }
        }
      },
      "EMA": {
        "versus_original_source": {
          "mean_pp": 0.6666784891101373,
          "median_pp": 0.8336363074035846,
          "p10_pp": -1.3289720433594487,
          "increased": 13,
          "decreased": 10,
          "decline_over_pp": {
            "0.5": 7,
            "1": 3,
            "2": 1
          },
          "leave_one_mean_range_pp": [
            0.36948254053587504,
            0.8909857109119472
          ],
          "paired": {
            "delta_B_minus_A_pp": 0.6666784891101374,
            "classes_pp": {
              "LV": 0.29401902544748465,
              "MYO": 0.460860356179692,
              "RV": 1.2451560857032355
            },
            "patients": 23,
            "increased": 13,
            "decreased": 10,
            "unchanged": 0,
            "median_patient_delta_pp": 0.8336363074035735,
            "patient_bootstrap95_pp": [
              -0.19132299597520372,
              1.5875330327579977
            ]
          },
          "phase_pp": {
            "ED": 0.17261415696557814,
            "ES": 1.1607428212546678
          }
        },
        "versus_entry": {
          "mean_pp": 1.6003708555931981,
          "median_pp": 1.2248242741778492,
          "p10_pp": -0.7186369876884967,
          "increased": 15,
          "decreased": 8,
          "decline_over_pp": {
            "0.5": 3,
            "1": 2,
            "2": 1
          },
          "leave_one_mean_range_pp": [
            1.2411321042123065,
            1.7818990909002608
          ],
          "paired": {
            "delta_B_minus_A_pp": 1.600370855593201,
            "classes_pp": {
              "LV": 0.33474444286636496,
              "MYO": 0.9807452839049837,
              "RV": 3.4856228400082534
            },
            "patients": 23,
            "increased": 15,
            "decreased": 8,
            "unchanged": 0,
            "median_patient_delta_pp": 1.2248242741778548,
            "patient_bootstrap95_pp": [
              0.6518840789995882,
              2.6955743238636027
            ]
          }
        }
      }
    }
  }
}
```

## 成本与工程审计

```json
{
  "status": "PASS",
  "cost": {
    "total": 6096,
    "optimizer": 6096,
    "functional": 0,
    "pending": 0,
    "by_phase": {
      "20270501_GRP:engineering": 32,
      "20270502_GRP:engineering": 32,
      "20270503_GRP:engineering": 32,
      "20270501_GRP:formal": 2000,
      "20270502_GRP:formal": 2000,
      "20270503_GRP:formal": 2000
    }
  },
  "calls": {
    "executed": {
      "engineering_native": 240,
      "engineering_encoder": 432,
      "engineering_reverse": 105,
      "formal_native": 12036,
      "formal_encoder": 288288,
      "formal_reverse": 6009,
      "diagnostic_reverse": 30,
      "evaluation_native": 8676
    },
    "by_group": {
      "20270501_GRP:engineering_native": 80,
      "20270501_GRP:engineering_encoder": 144,
      "20270501_GRP:engineering_reverse": 35,
      "20270502_GRP:engineering_native": 80,
      "20270502_GRP:engineering_encoder": 144,
      "20270502_GRP:engineering_reverse": 35,
      "20270503_GRP:engineering_native": 80,
      "20270503_GRP:engineering_encoder": 144,
      "20270503_GRP:engineering_reverse": 35,
      "20270501_GRP:formal_native": 4012,
      "20270501_GRP:formal_encoder": 96096,
      "20270501_GRP:formal_reverse": 2003,
      "20270501_GRP:diagnostic_reverse": 10,
      "20270501_GRP:evaluation_native": 2892,
      "20270502_GRP:formal_native": 4012,
      "20270502_GRP:formal_encoder": 96096,
      "20270502_GRP:formal_reverse": 2003,
      "20270502_GRP:diagnostic_reverse": 10,
      "20270502_GRP:evaluation_native": 2892,
      "20270503_GRP:formal_native": 4012,
      "20270503_GRP:formal_encoder": 96096,
      "20270503_GRP:formal_reverse": 2003,
      "20270503_GRP:diagnostic_reverse": 10,
      "20270503_GRP:evaluation_native": 2892
    },
    "pending": null,
    "caps": {
      "engineering_native": 288,
      "engineering_encoder": 576,
      "engineering_reverse": 120,
      "formal_native": 12200,
      "formal_encoder": 289000,
      "formal_reverse": 6012,
      "diagnostic_reverse": 30,
      "evaluation_native": 8676
    }
  },
  "polar": {
    "executed_steps": 6096,
    "executed_SVD": 48768,
    "pending": null,
    "cap_steps": 6096,
    "cap_SVD": 48768
  },
  "numerical_selfcheck_SVD": 10,
  "total_SVD": 48778,
  "geometry_SVD_calls": 12030,
  "total_including_geometry_SVD": 60808,
  "workers_retired": true,
  "common_entry_empty_opt": true,
  "frozen_projection_constants": true,
  "finite_final_latest_opt_RNG_reads": true,
  "patient_EDES_exact": true,
  "source_reference": {
    "status": "PASS",
    "patients": 23,
    "volumes": 46,
    "source_primary": "EMA",
    "source_percent": 80.36740663770259,
    "patient_EDES_match": true,
    "additional_model_forwards": 0
  },
  "EMA_formula_exact": true,
  "audit_model_forwards": 0,
  "allocated_bytes_including_metadata": 112657536,
  "managed_elapsed_seconds": 2621.56769990921
}
```

每流单独判断，所有流全门槛才PASS。CI为10000次患者配对重采样seed163，保留患者内ED/ES，不覆盖训练随机/共享源/重复开发和多重比较。旧-.30、新-.20pp非劣界限不是临床安全界限；均值恢复不代表逐患者恢复。无补种子、扫描alpha、事后checkpoint选择或放宽门槛。
