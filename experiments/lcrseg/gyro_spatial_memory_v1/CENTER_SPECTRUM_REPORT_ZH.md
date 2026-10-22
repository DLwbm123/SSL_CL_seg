# 中心化谱目标与全支持均衡：两流完整结果

固定student2000主终点，EMA使用相同.999更新；这是独立centroid中心化谱干预，不是MuLoRA复现。患者bootstrap不覆盖训练随机性与反复开发。

## 结论与后续决策

主综合、增量与两条流患者尾部门槛均FAIL；绝对入口均值保持门槛两流PASS，不能覆盖尾部失败。GRPC−GRP旧域−0.030418/−0.112756 pp，均−0.071587；新域−0.082443/−0.079376 pp，均−0.080909。旧域首流患者CI跨0，次流CI为[−0.201410,−0.021396]；两新域CI均跨0。不能把小幅点下降全部描述成显著退化。

相对入口，旧域下降>1pp人数GRP 9/10→GRPC 10/10，>2pp为3/4→3/4；p10为−2.404042/−2.419424→−2.399423/−2.285413。p10略改善但受损人数未达至少减少2人的预设门槛。相对配对GRP，旧域15/23、16/23患者下降，新域17/31、16/31下降。EMA旧均−0.016574、新均−0.007452，同样未过综合门槛。

中心化实现确已生效：GRPC输入行均值最大6.34e-8，能量分解误差<3.2e-16；未中心化组共同分量约38.3%能量，候选轨迹约35.9%。这些统计不证明共同分量有害，移除它没有改善主终点或尾部。停止中心化、均值混合及能量/权重补偿扫描。

下一独立主线问题拟检验三视图先均值是否抵消有用变化：以固定sqrt(3w)加权堆叠192行替代64行centroid，保持原增强、投影、CE和lambda_R。该假设不由本轮负结果证明；新轮另冻结协议与预算，使用新配对流，不改变本轮结论。

本轮8192真实更新、7054.786秒（约117.58分钟），运行结束153,621,632字节；最终一次零forward审计PASS。原源→入口−0.933692 pp仍计入；患者bootstrap不能覆盖训练随机性、重复开发或多重比较，未使用独立最终测试。

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
      true,
      true
    ],
    "old_pp": [
      -0.03041784434758361,
      -0.11275578408142359
    ],
    "new_pp": [
      -0.08244271724725491,
      -0.07937610578109876
    ],
    "old_mean_pp": -0.0715868142145036,
    "new_mean_pp": -0.08090941151417683
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
      -0.022377022148835902,
      -0.010770870607971748
    ],
    "new_pp": [
      -0.00864048534078431,
      -0.006262901034383911
    ],
    "old_mean_pp": -0.016573946378403825,
    "new_mean_pp": -0.00745169318758411
  }
}
```

## 终点

|流|组|旧/新学生%|旧/新EMA%|
|---|---|---|---|
|20270201|GRP|79.382086/83.375056|80.969375/83.364492|
|20270201|GRPC|79.351668/83.292613|80.946998/83.355851|
|20270202|GRP|79.872529/83.476440|81.055312/83.545572|
|20270202|GRPC|79.759773/83.397064|81.044541/83.539309|

## 配对代价

|流/模式/域|比较|均值pp [95%患者CI]|中位/p10|改善/下降|下降>.5/1/2pp|LV/MYO/RV|ED/ES|
|---|---|---|---|---|---|---|---|
|20270201/student/Siemens|GRPC_minus_GRP|-0.030418 [-0.140972,+0.095923]|-0.044714/-0.283618|8/15|1/0/0|-0.115439/-0.053832/+0.078018|-0.102071/+0.041235|
|20270201/student/Philips|GRPC_minus_GRP|-0.082443 [-0.183164,+0.006552]|-0.009835/-0.526421|14/17|4/0/0|-0.085734/-0.064877/-0.096718|-0.098160/-0.066725|
|20270201/EMA/Siemens|GRPC_minus_GRP|-0.022377 [-0.095176,+0.059058]|-0.020262/-0.178779|10/13|0/0/0|-0.064344/-0.023060/+0.020274|-0.060010/+0.015256|
|20270201/EMA/Philips|GRPC_minus_GRP|-0.008640 [-0.056547,+0.033661]|+0.022069/-0.167536|20/11|0/0/0|-0.030468/+0.006239/-0.001692|-0.028414/+0.011134|
|20270202/student/Siemens|GRPC_minus_GRP|-0.112756 [-0.201410,-0.021396]|-0.122893/-0.353462|7/16|1/0/0|-0.147195/-0.078830/-0.112243|-0.165209/-0.060303|
|20270202/student/Philips|GRPC_minus_GRP|-0.079376 [-0.187478,+0.009607]|-0.001372/-0.376602|15/16|2/1/0|-0.079366/-0.056473/-0.102289|-0.104431/-0.054321|
|20270202/EMA/Siemens|GRPC_minus_GRP|-0.010771 [-0.076633,+0.066560]|-0.007540/-0.142331|11/12|0/0/0|-0.054409/-0.006141/+0.028237|-0.069268/+0.047726|
|20270202/EMA/Philips|GRPC_minus_GRP|-0.006263 [-0.058649,+0.038172]|-0.009014/-0.112257|15/16|1/0/0|-0.024906/+0.012479/-0.006362|-0.045949/+0.033423|

## 相对入口患者损失

|流/模式/域|组|均值/中位/p10 pp|下降>.5/1/2pp|旧相对原源pp|
|---|---|---|---|---|
|20270201/student/Siemens|GRP|-0.051628/-0.631505/-2.404042|12/9/3|-0.985321|
|20270201/student/Siemens|GRPC|-0.082046/-0.849543/-2.399423|12/10/3|-1.015739|
|20270201/student/Philips|GRP|+10.170739/+10.162331/+2.575502|2/2/2|NA|
|20270201/student/Philips|GRPC|+10.088296/+10.296362/+2.453729|2/2/2|NA|
|20270201/EMA/Siemens|GRP|+1.535661/+1.126262/-0.755509|3/2/1|+0.601969|
|20270201/EMA/Siemens|GRPC|+1.513284/+1.040439/-0.728055|3/2/1|+0.579592|
|20270201/EMA/Philips|GRP|+10.160174/+9.807374/+2.757430|2/2/0|NA|
|20270201/EMA/Philips|GRPC|+10.151534/+9.817178/+2.647350|2/2/0|NA|
|20270202/student/Siemens|GRP|+0.438815/+0.174810/-2.419424|11/10/4|-0.494878|
|20270202/student/Siemens|GRPC|+0.326059/+0.140089/-2.285413|11/10/4|-0.607634|
|20270202/student/Philips|GRP|+10.272123/+10.187814/+3.045174|2/2/2|NA|
|20270202/student/Philips|GRPC|+10.192747/+10.499829/+2.898890|2/2/2|NA|
|20270202/EMA/Siemens|GRP|+1.621598/+1.185955/-0.648446|3/2/1|+0.687906|
|20270202/EMA/Siemens|GRPC|+1.610827/+1.039759/-0.663032|3/2/1|+0.677135|
|20270202/EMA/Philips|GRP|+10.341254/+9.858299/+2.496534|1/1/0|NA|
|20270202/EMA/Philips|GRPC|+10.334991/+9.835192/+2.432405|2/1/0|NA|

## 更新干预与成本

```json
{
  "polar": {
    "20270201_GRP": {
      "matrices": 16000,
      "mean_stable_rank": 2.3896564274355767,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.5833980273380875,
      "max_absolute_energy_error": 8.229108061641455e-09
    },
    "20270201_GRPC": {
      "matrices": 16000,
      "mean_stable_rank": 2.170760494597256,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.5649068861454726,
      "max_absolute_energy_error": 6.606569513678551e-09
    },
    "20270202_GRP": {
      "matrices": 16000,
      "mean_stable_rank": 2.3873492255583404,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.5816010555662214,
      "max_absolute_energy_error": 7.021299097687006e-09
    },
    "20270202_GRPC": {
      "matrices": 16000,
      "mean_stable_rank": 2.175589152894914,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.564067256435752,
      "max_absolute_energy_error": 6.832124199718237e-09
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
        "20270201_GRP:engineering": 32,
        "20270201_GRPC:engineering": 32,
        "20270202_GRP:engineering": 32,
        "20270202_GRPC:engineering": 32,
        "20270201_GRP:formal": 2000,
        "20270201_GRPC:formal": 2000,
        "20270202_GRP:formal": 2000,
        "20270202_GRPC:formal": 2000
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
        "20270201_GRP:engineering_native": 80,
        "20270201_GRP:engineering_encoder": 144,
        "20270201_GRP:engineering_reverse": 35,
        "20270201_GRPC:engineering_native": 80,
        "20270201_GRPC:engineering_encoder": 144,
        "20270201_GRPC:engineering_reverse": 35,
        "20270202_GRP:engineering_native": 80,
        "20270202_GRP:engineering_encoder": 144,
        "20270202_GRP:engineering_reverse": 35,
        "20270202_GRPC:engineering_native": 80,
        "20270202_GRPC:engineering_encoder": 144,
        "20270202_GRPC:engineering_reverse": 35,
        "20270201_GRP:formal_native": 4012,
        "20270201_GRP:formal_encoder": 96096,
        "20270201_GRP:formal_reverse": 2003,
        "20270201_GRP:diagnostic_reverse": 10,
        "20270201_GRP:evaluation_native": 2892,
        "20270201_GRPC:formal_native": 4012,
        "20270201_GRPC:formal_encoder": 96096,
        "20270201_GRPC:formal_reverse": 2003,
        "20270201_GRPC:diagnostic_reverse": 10,
        "20270201_GRPC:evaluation_native": 2892,
        "20270202_GRP:formal_native": 4012,
        "20270202_GRP:formal_encoder": 96096,
        "20270202_GRP:formal_reverse": 2003,
        "20270202_GRP:diagnostic_reverse": 10,
        "20270202_GRP:evaluation_native": 2892,
        "20270202_GRPC:formal_native": 4012,
        "20270202_GRPC:formal_encoder": 96096,
        "20270202_GRPC:formal_reverse": 2003,
        "20270202_GRPC:diagnostic_reverse": 10,
        "20270202_GRPC:evaluation_native": 2892
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
    "numerical_selfcheck_SVD": 18,
    "total_SVD": 65554,
    "geometry_SVD_calls": 16104,
    "total_including_geometry_SVD": 81658,
    "workers_retired": true,
    "common_entry_empty_opt": true,
    "frozen_projection_constants": true,
    "finite_final_latest_opt_RNG_reads": true,
    "patient_EDES_exact": true,
    "center_spectrum_gradient_selfcheck": true,
    "synthetic_reverse_calls": 2,
    "audit_model_forwards": 0,
    "allocated_bytes_including_metadata": 153629824,
    "managed_elapsed_seconds": 7054.786010026932
  }
}
```

能量只匹配同一状态下普通SGD提议；不同轨迹不保证相同能量。保留普通momentum；P只变换encoder矩阵，C组在谱输入按64行去均值，均值参与反向、不重归一化，监督CE保持原样，decoder仍正常更新。去均值可能移除任务相关共同信息并削弱适应；不将平均增益解释为患者安全。仅两流配对主比较，无交互估计。所有旧失败保留，下一问题另立冻结协议。
