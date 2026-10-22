# 弱视图停止梯度参照与全支持均衡：两流完整结果

固定student2000主终点，EMA使用相同.999更新；这是独立弱视图梯度路由干预，不是MuLoRA复现。患者bootstrap不覆盖训练随机性与反复开发。

## 完整结论

主GRPA−GRP旧域−0.613895/−0.724158pp，均−0.669027，两患者CI均负；新域−0.226132/−0.261846pp，均−0.243989，两CI跨0。主增量、两尾部与综合门槛FAIL，入口保持第一流FAIL、第二流PASS。旧患者下降16/23、17/23；入口>1pp损失11/7→14/10，>2pp4/1→8/4，p10−2.658138/−1.624526→−3.199674/−2.107431。不能将新域点非劣达线解释为旧域保护。

EMA旧均−0.329515、新均−0.173934pp，同样未恢复。停止当前弱视图detach及其补偿缩放/混合扫描。数值自检证明梯度路径按设计改变，不证明“弱视图漂移”因果；本结果拒绝当前设计预测，不能推广为所有停止梯度方法无效。弱图仍通过共享模型的监督及strong更新而变化。

8192optimizer、65546总polar/数值SVD、4合成反向；3501.300秒、约150.51MB，一次审计PASS、全部worker退出，无新增评估。共同源163/S†69，原源到入口−0.933692pp仍计入；23旧/31新患者EDES配对、10000bootstrap seed163，不覆盖训练随机、反复开发、多重比较，无独立最终测试。

## 训练集诊断

|流|组|源训练Dice%|目标训练Dice%|
|---|---|---|---|
|20270121|GRP|89.315323|91.320723|
|20270121|GRPA|89.050384|91.274468|
|20270122|GRP|89.523653|91.353383|
|20270122|GRPA|89.207848|91.298807|

## 门槛

```json
{
  "student": {
    "composite_pass": false,
    "increment_pass": false,
    "patient_tail_pass": [
      false,
      false
    ],
    "absolute_retention_pass": [
      false,
      true
    ],
    "old_pp": [
      -0.6138954349909853,
      -0.7241584503734977
    ],
    "new_pp": [
      -0.2261323993600431,
      -0.2618463603229041
    ],
    "old_mean_pp": -0.6690269426822415,
    "new_mean_pp": -0.2439893798414736
  },
  "EMA": {
    "composite_pass": false,
    "increment_pass": false,
    "patient_tail_pass": [
      false,
      false
    ],
    "absolute_retention_pass": [
      true,
      true
    ],
    "old_pp": [
      -0.3258644561051926,
      -0.3331653569196325
    ],
    "new_pp": [
      -0.18655759642125624,
      -0.16131104869166574
    ],
    "old_mean_pp": -0.32951490651241255,
    "new_mean_pp": -0.173934322556461
  }
}
```

## 终点

|流|组|旧/新学生%|旧/新EMA%|
|---|---|---|---|
|20270121|GRP|79.400155/83.268321|80.940475/83.300645|
|20270121|GRPA|78.786259/83.042188|80.614610/83.114087|
|20270122|GRP|80.032582/83.569204|81.088656/83.285138|
|20270122|GRPA|79.308424/83.307357|80.755491/83.123827|

## 配对代价

|流/模式/域|比较|均值pp [95%患者CI]|中位/p10|改善/下降|下降>.5/1/2pp|LV/MYO/RV|ED/ES|
|---|---|---|---|---|---|---|---|
|20270121/student/Siemens|GRPA_minus_GRP|-0.613895 [-0.976318,-0.285176]|-0.417401/-1.495390|7/16|9/6/2|-0.506024/-0.352655/-0.983007|-0.400474/-0.827317|
|20270121/student/Philips|GRPA_minus_GRP|-0.226132 [-0.669949,+0.106555]|-0.016389/-0.254010|15/16|3/2/2|-0.322751/-0.248052/-0.107595|-0.291565/-0.160700|
|20270121/EMA/Siemens|GRPA_minus_GRP|-0.325864 [-0.547114,-0.119845]|-0.139371/-1.252508|7/16|7/4/0|-0.293662/-0.217902/-0.466029|-0.180172/-0.471557|
|20270121/EMA/Philips|GRPA_minus_GRP|-0.186558 [-0.593152,+0.076291]|+0.017734/-0.272728|17/14|2/2/1|-0.250842/-0.182448/-0.126384|-0.199623/-0.173492|
|20270122/student/Siemens|GRPA_minus_GRP|-0.724158 [-1.123995,-0.362413]|-0.434930/-2.028636|6/17|11/7/3|-0.465141/-0.382212/-1.325123|-0.522532/-0.925785|
|20270122/student/Philips|GRPA_minus_GRP|-0.261846 [-0.733795,+0.082922]|+0.053222/-0.323710|19/12|3/3/2|-0.326498/-0.241480/-0.217561|-0.264037/-0.259655|
|20270122/EMA/Siemens|GRPA_minus_GRP|-0.333165 [-0.549389,-0.126890]|-0.120106/-1.186623|7/16|7/4/0|-0.235505/-0.221256/-0.542735|-0.208392/-0.457939|
|20270122/EMA/Philips|GRPA_minus_GRP|-0.161311 [-0.497836,+0.066044]|+0.024029/-0.264431|19/12|2/2/1|-0.175849/-0.153453/-0.154631|-0.201908/-0.120714|

## 相对入口患者损失

|流/模式/域|组|均值/中位/p10 pp|下降>.5/1/2pp|旧相对原源pp|
|---|---|---|---|---|
|20270121/student/Siemens|GRP|-0.033559/-0.976903/-2.658138|13/11/4|-0.967252|
|20270121/student/Siemens|GRPA|-0.647455/-1.122460/-3.199674|14/14/8|-1.581147|
|20270121/student/Philips|GRP|+10.064003/+10.318070/+2.934556|2/2/2|NA|
|20270121/student/Philips|GRPA|+9.837871/+10.361586/+3.089210|2/2/2|NA|
|20270121/EMA/Siemens|GRP|+1.506761/+0.876518/-0.730900|3/2/1|+0.573068|
|20270121/EMA/Siemens|GRPA|+1.180896/+0.738334/-0.965933|4/3/1|+0.247204|
|20270121/EMA/Philips|GRP|+10.096327/+9.743751/+2.767204|2/2/0|NA|
|20270121/EMA/Philips|GRPA|+9.909770/+9.719604/+2.911128|2/2/1|NA|
|20270122/student/Siemens|GRP|+0.598868/-0.006357/-1.624526|10/7/1|-0.334825|
|20270122/student/Siemens|GRPA|-0.125291/-0.278889/-2.107431|11/10/4|-1.058983|
|20270122/student/Philips|GRP|+10.364886/+10.404286/+2.524771|2/2/2|NA|
|20270122/student/Philips|GRPA|+10.103040/+10.661330/+2.688069|2/2/2|NA|
|20270122/EMA/Siemens|GRP|+1.654942/+1.169285/-0.696029|3/2/1|+0.721249|
|20270122/EMA/Siemens|GRPA|+1.321776/+1.055066/-0.811279|4/1/1|+0.388084|
|20270122/EMA/Philips|GRP|+10.080821/+9.792625/+2.504707|2/2/1|NA|
|20270122/EMA/Philips|GRPA|+9.919510/+9.821867/+2.525890|2/2/1|NA|

## 更新干预与成本

```json
{
  "polar": {
    "20270121_GRP": {
      "matrices": 16000,
      "mean_stable_rank": 2.401054784476757,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.5821425753030925,
      "max_absolute_energy_error": 7.828930392861366e-09
    },
    "20270121_GRPA": {
      "matrices": 16000,
      "mean_stable_rank": 2.4645124355480075,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.5832511373385787,
      "max_absolute_energy_error": 6.995833246037364e-09
    },
    "20270122_GRP": {
      "matrices": 16000,
      "mean_stable_rank": 2.393545012757182,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.5814490263704211,
      "max_absolute_energy_error": 8.70932126417756e-09
    },
    "20270122_GRPA": {
      "matrices": 16000,
      "mean_stable_rank": 2.4568038189411165,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.5828578376267105,
      "max_absolute_energy_error": 7.310518412850797e-09
    }
  },
  "audit": {
    "status": "PASS",
    "cost": {
      "total": 8192,
      "optimizer": 8192,
      "functional": 0,
      "pending": 0,
      "by_phase": {
        "DETACH_S:engineering": 32,
        "DETACH_STOP:engineering": 32,
        "20270121_GRP:engineering": 32,
        "20270121_GRPA:engineering": 32,
        "20270122_GRP:engineering": 32,
        "20270122_GRPA:engineering": 32,
        "20270121_GRP:formal": 2000,
        "20270121_GRPA:formal": 2000,
        "20270122_GRP:formal": 2000,
        "20270122_GRPA:formal": 2000
      }
    },
    "calls": {
      "executed": {
        "detach_native": 128,
        "detach_encoder": 1536,
        "detach_reverse": 64,
        "engineering_native": 320,
        "engineering_encoder": 576,
        "engineering_reverse": 140,
        "formal_native": 16048,
        "formal_encoder": 384384,
        "formal_reverse": 8012,
        "diagnostic_reverse": 40,
        "evaluation_native": 11568
      },
      "by_group": {
        "DETACH:detach_native": 128,
        "DETACH:detach_reverse": 64,
        "DETACH:detach_encoder": 1536,
        "20270121_GRP:engineering_native": 80,
        "20270121_GRP:engineering_encoder": 144,
        "20270121_GRP:engineering_reverse": 35,
        "20270121_GRPA:engineering_native": 80,
        "20270121_GRPA:engineering_encoder": 144,
        "20270121_GRPA:engineering_reverse": 35,
        "20270122_GRP:engineering_native": 80,
        "20270122_GRP:engineering_encoder": 144,
        "20270122_GRP:engineering_reverse": 35,
        "20270122_GRPA:engineering_native": 80,
        "20270122_GRPA:engineering_encoder": 144,
        "20270122_GRPA:engineering_reverse": 35,
        "20270121_GRP:formal_native": 4012,
        "20270121_GRP:formal_encoder": 96096,
        "20270121_GRP:formal_reverse": 2003,
        "20270121_GRP:diagnostic_reverse": 10,
        "20270121_GRP:evaluation_native": 2892,
        "20270121_GRPA:formal_native": 4012,
        "20270121_GRPA:formal_encoder": 96096,
        "20270121_GRPA:formal_reverse": 2003,
        "20270121_GRPA:diagnostic_reverse": 10,
        "20270121_GRPA:evaluation_native": 2892,
        "20270122_GRP:formal_native": 4012,
        "20270122_GRP:formal_encoder": 96096,
        "20270122_GRP:formal_reverse": 2003,
        "20270122_GRP:diagnostic_reverse": 10,
        "20270122_GRP:evaluation_native": 2892,
        "20270122_GRPA:formal_native": 4012,
        "20270122_GRPA:formal_encoder": 96096,
        "20270122_GRPA:formal_reverse": 2003,
        "20270122_GRPA:diagnostic_reverse": 10,
        "20270122_GRPA:evaluation_native": 2892
      },
      "pending": null,
      "caps": {
        "detach_native": 160,
        "detach_encoder": 1632,
        "detach_reverse": 64,
        "engineering_native": 384,
        "engineering_encoder": 768,
        "engineering_reverse": 160,
        "formal_native": 16200,
        "formal_encoder": 385000,
        "formal_reverse": 8016,
        "diagnostic_reverse": 40,
        "evaluation_native": 11568
      }
    },
    "polar": {
      "executed_steps": 8192,
      "executed_SVD": 65536,
      "pending": null,
      "cap_steps": 8192,
      "cap_SVD": 65536
    },
    "numerical_selfcheck_SVD": 10,
    "total_SVD": 65546,
    "workers_retired": true,
    "common_entry_empty_opt": true,
    "frozen_projection_constants": true,
    "finite_final_latest_opt_RNG_reads": true,
    "patient_EDES_exact": true,
    "weak_anchor_gradient_selfcheck": true,
    "synthetic_reverse_calls": 4,
    "audit_model_forwards": 0,
    "allocated_bytes_including_metadata": 150779008,
    "managed_elapsed_seconds": 3501.300168275833
  }
}
```

能量只匹配同一状态下普通SGD提议；不同轨迹不保证相同能量。保留普通momentum；P只变换encoder矩阵，A组只停止当前弱视图centroid分支梯度，监督CE保持原样，decoder仍正常更新。停止弱视图梯度可能削弱新域适应；不将平均增益解释为患者安全。仅两流配对主比较，无交互估计。所有旧失败保留，下一问题另立冻结协议。
