# 加权视图堆叠谱与全支持均衡：两流完整结果

固定student2000主终点，EMA使用相同.999更新；这是独立加权视图堆叠谱干预，不是MuLoRA复现。患者bootstrap不覆盖训练随机性与反复开发。

## 最终判断

主综合、主增量和两流患者尾部门槛均FAIL；绝对入口均值保持两流PASS，不能覆盖前述失败。GRPV−GRP旧均−0.423305pp、新均−0.614070pp，四个配对患者CI均完全低于零。EMA旧均−0.211922、新均−0.435648，同方向失败；不以EMA替换原主终点。

相对入口旧域>1pp人数10/7→10/10，>2pp人数2/2→5/4，p10−1.920997/−1.772859→−2.837697/−2.769557。相对同期GRP，旧18/23、16/23患者下降，新两流均22/31下降。保留视图二阶矩没有兑现本配方的预期，不能进一步推出所有多视图方法无效。

停止堆叠权重/行数/尺度扫描。按用户新授权，下一步执行已于567c14b7冻结的固定EMA部署确认，三个新训练流，共6096更新；旧/新绝对性能和原源患者尾部为主，不再继续零碎变体。该确认共享源与开发验证患者，不是独立最终测试。

原GRP对照的EMA旧81.046541/81.004922、新83.337646/83.284954%，支持开展预注册确认，但不能用本轮结果追改确认门槛。保留原源旧80.367407%及S†入口79.433714%的区别。

本轮8192实际更新，6240.350539秒（104.01分钟），结束153048192字节，最终一次零forward审计PASS。候选几何矩阵192×32、原64×32；相同调用数不等于相同计算量。

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
      -0.4196291932760471,
      -0.42698158391818114
    ],
    "new_pp": [
      -0.6406241232702209,
      -0.5875154914034444
    ],
    "old_mean_pp": -0.4233053885971141,
    "new_mean_pp": -0.6140698073368327
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
      -0.21300803044989527,
      -0.21083679415527612
    ],
    "new_pp": [
      -0.4475548923326887,
      -0.4237401229332012
    ],
    "old_mean_pp": -0.21192241230258568,
    "new_mean_pp": -0.4356475076329449
  }
}
```

## 终点

|流|组|旧/新学生%|旧/新EMA%|
|---|---|---|---|
|20270401|GRP|79.694484/83.507557|81.046541/83.337646|
|20270401|GRPV|79.274854/82.866933|80.833533/82.890092|
|20270402|GRP|80.032823/83.371551|81.004922/83.284954|
|20270402|GRPV|79.605841/82.784035|80.794085/82.861214|

## 配对代价

|流/模式/域|比较|均值pp [95%患者CI]|中位/p10|改善/下降|下降>.5/1/2pp|LV/MYO/RV|ED/ES|
|---|---|---|---|---|---|---|---|
|20270401/student/Siemens|GRPV_minus_GRP|-0.419629 [-0.734705,-0.137081]|-0.349017/-1.363008|5/18|7/3/1|-0.112632/-0.316447/-0.829809|-0.517526/-0.321733|
|20270401/student/Philips|GRPV_minus_GRP|-0.640624 [-1.399737,-0.078816]|-0.185867/-0.540058|9/22|4/3/3|-0.610925/-0.544698/-0.766250|-0.580300/-0.700948|
|20270401/EMA/Siemens|GRPV_minus_GRP|-0.213008 [-0.475890,+0.011563]|-0.146913/-0.839575|8/15|3/2/1|-0.077717/-0.235338/-0.325969|-0.307139/-0.118877|
|20270401/EMA/Philips|GRPV_minus_GRP|-0.447555 [-0.923749,-0.097411]|-0.123009/-0.527155|8/23|4/3/2|-0.355804/-0.354975/-0.631886|-0.369494/-0.525616|
|20270402/student/Siemens|GRPV_minus_GRP|-0.426982 [-0.741426,-0.135715]|-0.227429/-1.460935|7/16|8/5/1|-0.140300/-0.236399/-0.904246|-0.490358/-0.363605|
|20270402/student/Philips|GRPV_minus_GRP|-0.587515 [-1.269928,-0.080117]|-0.147326/-0.846574|9/22|7/3/3|-0.491869/-0.485107/-0.785571|-0.564506/-0.610525|
|20270402/EMA/Siemens|GRPV_minus_GRP|-0.210837 [-0.442845,-0.011004]|-0.133124/-0.830936|8/15|3/2/0|-0.104627/-0.167966/-0.359917|-0.241972/-0.179701|
|20270402/EMA/Philips|GRPV_minus_GRP|-0.423740 [-0.874181,-0.092910]|-0.109077/-0.610087|9/22|5/3/2|-0.278468/-0.333419/-0.659334|-0.356668/-0.490813|

## 相对入口患者损失

|流/模式/域|组|均值/中位/p10 pp|下降>.5/1/2pp|旧相对原源pp|
|---|---|---|---|---|
|20270401/student/Siemens|GRP|+0.260769/+0.003168/-1.920997|10/10/2|-0.672923|
|20270401/student/Siemens|GRPV|-0.158860/-0.202798/-2.837697|11/10/5|-1.092552|
|20270401/student/Philips|GRP|+10.303239/+10.435263/+2.102723|2/2/2|NA|
|20270401/student/Philips|GRPV|+9.662615/+10.752327/+2.362802|2/2/2|NA|
|20270401/EMA/Siemens|GRP|+1.612827/+1.172462/-0.696217|3/1/1|+0.679134|
|20270401/EMA/Siemens|GRPV|+1.399819/+0.766222/-0.763468|6/2/1|+0.466126|
|20270401/EMA/Philips|GRP|+10.133329/+9.985735/+2.615312|2/2/0|NA|
|20270401/EMA/Philips|GRPV|+9.685774/+9.885843/+2.492522|2/2/2|NA|
|20270402/student/Siemens|GRP|+0.599109/+0.078770/-1.772859|11/7/2|-0.334584|
|20270402/student/Siemens|GRPV|+0.172127/-0.356471/-2.769557|11/10/4|-0.761565|
|20270402/student/Philips|GRP|+10.167233/+9.819753/+2.957368|2/2/2|NA|
|20270402/student/Philips|GRPV|+9.579718/+10.055441/+3.095774|2/2/2|NA|
|20270402/EMA/Siemens|GRP|+1.571208/+1.229651/-0.704695|3/2/1|+0.637515|
|20270402/EMA/Siemens|GRPV|+1.360371/+0.602023/-0.704413|5/1/1|+0.426678|
|20270402/EMA/Philips|GRP|+10.080637/+9.860004/+3.004919|2/2/0|NA|
|20270402/EMA/Philips|GRPV|+9.656896/+9.719298/+3.029780|2/2/2|NA|

## 更新干预与成本

```json
{
  "polar": {
    "20270401_GRP": {
      "matrices": 16000,
      "mean_stable_rank": 2.3941385436430576,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.581200305826962,
      "max_absolute_energy_error": 8.13452061265707e-09
    },
    "20270401_GRPV": {
      "matrices": 16000,
      "mean_stable_rank": 2.3342829974144697,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.5740643438119442,
      "max_absolute_energy_error": 7.3196133598685265e-09
    },
    "20270402_GRP": {
      "matrices": 16000,
      "mean_stable_rank": 2.3965566982775925,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.5831712129563094,
      "max_absolute_energy_error": 6.555637810379267e-09
    },
    "20270402_GRPV": {
      "matrices": 16000,
      "mean_stable_rank": 2.3431969496756793,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.5763357165949419,
      "max_absolute_energy_error": 8.247297955676913e-09
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
        "20270401_GRP:engineering": 32,
        "20270401_GRPV:engineering": 32,
        "20270402_GRP:engineering": 32,
        "20270402_GRPV:engineering": 32,
        "20270401_GRP:formal": 2000,
        "20270401_GRPV:formal": 2000,
        "20270402_GRP:formal": 2000,
        "20270402_GRPV:formal": 2000
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
        "20270401_GRP:engineering_native": 80,
        "20270401_GRP:engineering_encoder": 144,
        "20270401_GRP:engineering_reverse": 35,
        "20270401_GRPV:engineering_native": 80,
        "20270401_GRPV:engineering_encoder": 144,
        "20270401_GRPV:engineering_reverse": 35,
        "20270402_GRP:engineering_native": 80,
        "20270402_GRP:engineering_encoder": 144,
        "20270402_GRP:engineering_reverse": 35,
        "20270402_GRPV:engineering_native": 80,
        "20270402_GRPV:engineering_encoder": 144,
        "20270402_GRPV:engineering_reverse": 35,
        "20270401_GRP:formal_native": 4012,
        "20270401_GRP:formal_encoder": 96096,
        "20270401_GRP:formal_reverse": 2003,
        "20270401_GRP:diagnostic_reverse": 10,
        "20270401_GRP:evaluation_native": 2892,
        "20270401_GRPV:formal_native": 4012,
        "20270401_GRPV:formal_encoder": 96096,
        "20270401_GRPV:formal_reverse": 2003,
        "20270401_GRPV:diagnostic_reverse": 10,
        "20270401_GRPV:evaluation_native": 2892,
        "20270402_GRP:formal_native": 4012,
        "20270402_GRP:formal_encoder": 96096,
        "20270402_GRP:formal_reverse": 2003,
        "20270402_GRP:diagnostic_reverse": 10,
        "20270402_GRP:evaluation_native": 2892,
        "20270402_GRPV:formal_native": 4012,
        "20270402_GRPV:formal_encoder": 96096,
        "20270402_GRPV:formal_reverse": 2003,
        "20270402_GRPV:diagnostic_reverse": 10,
        "20270402_GRPV:evaluation_native": 2892
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
    "view_stack_gradient_selfcheck": true,
    "synthetic_reverse_calls": 6,
    "audit_model_forwards": 0,
    "allocated_bytes_including_metadata": 153056384,
    "managed_elapsed_seconds": 6240.35053896904
  }
}
```

能量只匹配同一状态下普通SGD提议；不同轨迹不保证相同能量。保留普通momentum；P只变换encoder矩阵，V组堆叠sqrt(3w)加权三视图192行，保留全视图梯度、不中心化，监督CE保持原样，decoder仍正常更新。堆叠保留视图二阶矩，可能削弱均值所隐含的对齐并保留增强噪声；不将平均增益解释为患者安全。仅两流配对主比较，无交互估计。所有旧失败保留，下一问题另立冻结协议。
