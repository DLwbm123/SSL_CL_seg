# 输出梯度范数匹配：两流完整结论

六组完成，一次零forward最终审计PASS。固定student2000主终点。**范数匹配没有挽救类别平均CE：综合、主增量、患者尾部、绝对保持和恢复门槛均失败。** 不换EMA、不放宽阈值、不继续类别权重/混合比例扫描。

|比较|旧域第一/第二流 pp|新域第一/第二流 pp|
|---|---:|---:|
|匹配宏CE−原CE（GRPBN−GRP）|−.989634 / −.598300|−3.991713 / −3.815670|
|匹配宏CE−原宏CE（GRPBN−GRPB）|+.452713 / +.533881|−.553577 / −.736069|

主比较旧均−.793967pp（两CI跨0）、新均−3.903691pp（两CI全负）。不能把旧CI跨0说成无损或等效。对原宏CE，旧恢复均+.493297pp，未过预声明+.5且两CI跨0；新域进一步下降，两CI均全负。预声明同时恢复旧/新条件FAIL。

## 患者尾部与类别

GRPBN相对GRP旧域下降16/23、15/23人，新域两流各28/31人下降。相对训练入口，旧下降>1pp人数GRP8/11→GRPBN14/13；>2pp人数3/3→11/11；p10从−2.112246/−2.326093变成−3.775502/−3.570471。旧入口均值−.694567/−.562038，均越过−.5绝对保持边界。

对原宏CE，旧均值的恢复主要来自LV（+1.558894/+1.712096pp）；MYO−.120400/+.219215、RV−.080355/−.329667。新域对原宏CE：MYO−.955450/−.902780、RV−1.063585/−1.699217，两流各25/31人进一步下降。更均衡的梯度范数不等于更均衡的器官效果。

目标train固定终点：GRPBN83.234689/83.271876%，低于GRPB86.462165/87.180778及GRP91.313543/91.392638。EMA主增量旧均−1.342878、新均−2.963532pp，亦不支持该干预，不改变学生主终点。

## 机制诊断与停止结论

两流匹配系数平均约.165149/.158662，范围.003620—1/.007217—1；此汇总含工程与正式调用，每流2038条事件，另64detach条。最大匹配相对误差低于6e-8。实际输出梯度范数匹配通过，不能说实现没有生效。

每组5个固定诊断点：原宏CE的10点全部raw parameter gradient norm>1；匹配后仅第二流step0>1，普通CE也仅该点>1。干预消除了这些诊断点的频繁裁剪现象，但新域训练及验证进一步恶化。因此“仅修复输出梯度尺度即可消除宏CE损害”的本轮预测不成立。不能外推所有梯度归一化都无效，也不能证明每步参数梯度/clip系数完全匹配。

终止此宏CE及输出范数匹配路线，不追加参数梯度归一化、权重或混合扫描。下一独立问题回到原CE，检查几何强增强的大块遮挡：现有代码在相同光度变换后写入128灰度遮挡区域，仍将相同空间位置纳入三视图centroid。去遮挡是否减少几何目标的局部语义破坏只是待检验假设；采用保留光度/RNG/位置/模型调用的最小2×2对照，另立冻结协议，不改本轮结果。

## 成本与限制

12256 optimizer、98058 SVD（含10数值）、4次合成logits反向，医疗forward/encoder/reverse精确守账，6资格全PASS，全部worker退出。原钟5810.734994秒（约97分钟），输出约224MB，8h/4GB内。核共同入口、空opt、final/latest finite optRNG、固定投影、reads、患者EDES与30诊断通过。

仍为共享源入口、23旧/31新验证患者上的两条训练随机流；患者bootstrap10000seed163不覆盖训练随机性、多重比较或反复开发。原源到入口−.933692pp仍计，不宣称临床安全或独立源复现。源图、mask、患者ID、权重和私有日志不公开。

---

# 输出梯度范数匹配的类别平均CE：两流完整结果

固定student2000主终点，EMA使用相同.999更新；这是独立输出梯度范数匹配/等能量encoder位移干预，不是MuLoRA复现。患者bootstrap不覆盖训练随机性与反复开发。

## 门槛

```json
{
  "student": {
    "composite_pass": false,
    "increment_pass": false,
    "recovery_from_macro_pass": false,
    "patient_tail_pass": [
      false,
      false
    ],
    "absolute_retention_pass": [
      false,
      false
    ],
    "old_pp": [
      -0.9896337060786837,
      -0.5983003069959555
    ],
    "new_pp": [
      -3.9917129028128175,
      -3.8156696079233954
    ],
    "old_mean_pp": -0.7939670065373197,
    "new_mean_pp": -3.9036912553681065,
    "GRPBN_minus_GRPB_pp": {
      "Siemens": [
        0.4527127346841083,
        0.5338813414434014
      ],
      "Philips": [
        -0.5535767320480786,
        -0.736068706661437
      ]
    }
  },
  "EMA": {
    "composite_pass": false,
    "increment_pass": false,
    "recovery_from_macro_pass": false,
    "patient_tail_pass": [
      false,
      false
    ],
    "absolute_retention_pass": [
      true,
      true
    ],
    "old_pp": [
      -1.3686292065370278,
      -1.3171269723763483
    ],
    "new_pp": [
      -2.9218898868029406,
      -3.005173780265829
    ],
    "old_mean_pp": -1.342878089456688,
    "new_mean_pp": -2.9635318335343848,
    "GRPBN_minus_GRPB_pp": {
      "Siemens": [
        -0.4141985344538555,
        -0.4167207530341418
      ],
      "Philips": [
        -0.6315439170312379,
        -0.6203106783484801
      ]
    }
  }
}
```

## 终点

|流|组|旧/新学生%|旧/新EMA%|
|---|---|---|---|
|20261225|GRP|79.728781/83.481494|81.092498/83.351029|
|20261225|GRPB|78.286434/80.043358|80.138067/81.060683|
|20261225|GRPBN|78.739147/79.489782|79.723869/80.429139|
|20261226|GRP|79.469976/83.369911|81.048513/83.444152|
|20261226|GRPB|78.337795/80.290310|80.148106/81.059289|
|20261226|GRPBN|78.871676/79.554241|79.731386/80.438978|

## 配对代价

|流/模式/域|比较|均值pp [95%患者CI]|中位/p10|改善/下降|下降>.5/1/2pp|LV/MYO/RV|ED/ES|
|---|---|---|---|---|---|---|---|
|20261225/student/Siemens|GRPB_minus_GRP|-1.442346 [-2.201295,-0.650558]|-1.610836/-3.856561|6/17|16/16/10|-1.125892/-1.990753/-1.210394|-1.187588/-1.697105|
|20261225/student/Siemens|GRPBN_minus_GRP|-0.989634 [-2.234842,+0.272661]|-1.397308/-4.678421|7/16|15/14/10|+0.433002/-2.111153/-1.290750|-1.258237/-0.721030|
|20261225/student/Siemens|GRPBN_minus_GRPB|+0.452713 [-0.205104,+1.161064]|-0.235931/-1.360184|10/13|8/3/0|+1.558894/-0.120400/-0.080355|-0.070649/+0.976075|
|20261225/student/Philips|GRPB_minus_GRP|-3.438136 [-4.392089,-2.352509]|-3.906395/-6.008994|3/28|27/27/26|-0.244141/-4.852460/-5.217808|-3.102196/-3.774076|
|20261225/student/Philips|GRPBN_minus_GRP|-3.991713 [-5.305162,-2.450272]|-4.199829/-7.510333|3/28|27/27/26|+0.114164/-5.807910/-6.281393|-3.621835/-4.361591|
|20261225/student/Philips|GRPBN_minus_GRPB|-0.553577 [-0.991056,-0.055335]|-0.808265/-1.923148|6/25|17/11/2|+0.358305/-0.955450/-1.063585|-0.519639/-0.587514|
|20261225/EMA/Siemens|GRPB_minus_GRP|-0.954431 [-1.816734,-0.075744]|-1.439361/-3.488957|7/16|15/12/8|-0.484350/-1.824658/-0.554284|-1.121033/-0.787829|
|20261225/EMA/Siemens|GRPBN_minus_GRP|-1.368629 [-2.468898,-0.240956]|-1.821893/-4.439535|7/16|16/14/10|-0.416513/-2.008856/-1.680519|-1.595926/-1.141332|
|20261225/EMA/Siemens|GRPBN_minus_GRPB|-0.414199 [-0.709529,-0.106157]|-0.553300/-1.160721|5/18|12/7/0|+0.067838/-0.184197/-1.126236|-0.474894/-0.353503|
|20261225/EMA/Philips|GRPB_minus_GRP|-2.290346 [-3.372666,-1.002484]|-3.001710/-5.347669|4/27|27/24/20|+0.305454/-3.758713/-3.417779|-2.190892/-2.389800|
|20261225/EMA/Philips|GRPBN_minus_GRP|-2.921890 [-4.199750,-1.403006]|-3.731853/-6.004110|4/27|26/26/24|+0.147318/-4.342493/-4.570494|-2.732054/-3.111725|
|20261225/EMA/Philips|GRPBN_minus_GRPB|-0.631544 [-0.888863,-0.355183]|-0.683067/-1.585922|3/28|20/7/1|-0.158136/-0.583780/-1.152715|-0.541162/-0.721926|
|20261226/student/Siemens|GRPB_minus_GRP|-1.132182 [-1.941047,-0.281323]|-1.745085/-3.549325|6/17|15/14/8|-0.985056/-1.558082/-0.853407|-0.871200/-1.393163|
|20261226/student/Siemens|GRPBN_minus_GRP|-0.598300 [-1.944299,+0.768061]|-0.829324/-4.299067|8/15|14/11/9|+0.727039/-1.338866/-1.183074|-0.915520/-0.281080|
|20261226/student/Siemens|GRPBN_minus_GRPB|+0.533881 [-0.177104,+1.285746]|+0.057172/-1.319921|12/11|9/4/0|+1.712096/+0.219215/-0.329667|-0.044320/+1.112083|
|20261226/student/Philips|GRPB_minus_GRP|-3.079601 [-4.039697,-1.960898]|-3.682251/-5.854724|3/28|28/28/25|-0.547152/-4.436241/-4.255410|-2.707797/-3.451404|
|20261226/student/Philips|GRPBN_minus_GRP|-3.815670 [-5.206891,-2.173509]|-4.331079/-7.323723|3/28|27/27/27|-0.153361/-5.339021/-5.954627|-3.286673/-4.344667|
|20261226/student/Philips|GRPBN_minus_GRPB|-0.736069 [-1.266883,-0.136855]|-0.944240/-2.212749|6/25|21/15/6|+0.393791/-0.902780/-1.699217|-0.578875/-0.893262|
|20261226/EMA/Siemens|GRPB_minus_GRP|-0.900406 [-1.803678,+0.010910]|-1.426329/-3.616475|7/16|15/13/8|-0.400915/-1.579393/-0.720911|-1.161476/-0.639336|
|20261226/EMA/Siemens|GRPBN_minus_GRP|-1.317127 [-2.476954,-0.135447]|-1.581876/-4.644809|7/16|16/14/10|-0.319233/-1.755820/-1.876329|-1.651399/-0.982855|
|20261226/EMA/Siemens|GRPBN_minus_GRPB|-0.416721 [-0.727002,-0.086814]|-0.508488/-1.260506|5/18|12/6/0|+0.081682/-0.176427/-1.155418|-0.489923/-0.343518|
|20261226/EMA/Philips|GRPB_minus_GRP|-2.384863 [-3.433955,-1.151083]|-3.236938/-5.523874|4/27|27/24/20|+0.086504/-3.743574/-3.497519|-2.304888/-2.464838|
|20261226/EMA/Philips|GRPBN_minus_GRP|-3.005174 [-4.233672,-1.549863]|-3.540882/-6.056764|4/27|27/26/23|-0.051702/-4.314583/-4.649236|-2.818648/-3.191699|
|20261226/EMA/Philips|GRPBN_minus_GRPB|-0.620311 [-0.862415,-0.372046]|-0.653860/-1.554686|4/27|22/6/0|-0.138207/-0.571009/-1.151717|-0.513760/-0.726861|

## 相对入口患者损失

|流/模式/域|组|均值/中位/p10 pp|下降>.5/1/2pp|旧相对原源pp|
|---|---|---|---|---|
|20261225/student/Siemens|GRP|+0.295067/-0.011657/-2.112246|10/8/3|-0.638626|
|20261225/student/Siemens|GRPB|-1.147280/-1.992960/-4.580053|14/13/11|-2.080972|
|20261225/student/Siemens|GRPBN|-0.694567/-1.981951/-3.775502|14/14/11|-1.628259|
|20261225/student/Philips|GRP|+10.277177/+10.348922/+2.750385|2/2/2|NA|
|20261225/student/Philips|GRPB|+6.839041/+6.180134/-2.358953|5/5/4|NA|
|20261225/student/Philips|GRPBN|+6.285464/+5.957873/-2.613606|7/5/4|NA|
|20261225/EMA/Siemens|GRP|+1.658783/+1.250612/-0.628911|3/2/1|+0.725091|
|20261225/EMA/Siemens|GRPB|+0.704353/-0.096177/-2.312080|11/9/4|-0.229340|
|20261225/EMA/Siemens|GRPBN|+0.290154/-0.619780/-3.122260|12/10/7|-0.643538|
|20261225/EMA/Philips|GRP|+10.146712/+9.844730/+2.823591|2/2/0|NA|
|20261225/EMA/Philips|GRPB|+7.856366/+7.039172/+0.076491|3/3/1|NA|
|20261225/EMA/Philips|GRPBN|+7.224822/+6.774862/-0.352490|3/3/3|NA|
|20261226/student/Siemens|GRP|+0.036262/-0.639376/-2.326093|12/11/3|-0.897430|
|20261226/student/Siemens|GRPB|-1.095920/-2.129377/-4.682787|14/13/12|-2.029612|
|20261226/student/Siemens|GRPBN|-0.562038/-1.563017/-3.570471|14/13/11|-1.495731|
|20261226/student/Philips|GRP|+10.165594/+10.285798/+3.007966|2/2/2|NA|
|20261226/student/Philips|GRPB|+7.085993/+6.949461/-1.984490|5/5/3|NA|
|20261226/student/Philips|GRPBN|+6.349924/+5.847216/-2.614366|6/6/4|NA|
|20261226/EMA/Siemens|GRP|+1.614798/+1.301857/-0.627378|3/2/1|+0.681106|
|20261226/EMA/Siemens|GRPB|+0.714392/-0.206063/-2.266347|10/9/4|-0.219300|
|20261226/EMA/Siemens|GRPBN|+0.297671/-0.387233/-3.262378|11/10/7|-0.636021|
|20261226/EMA/Philips|GRP|+10.239835/+9.835028/+2.871083|1/1/0|NA|
|20261226/EMA/Philips|GRPB|+7.854972/+7.297594/-0.222841|3/3/2|NA|
|20261226/EMA/Philips|GRPBN|+7.234661/+6.560278/-0.390986|3/3/3|NA|

## 更新干预与成本

```json
{
  "polar": {
    "20261225_GRP": {
      "matrices": 16000,
      "mean_stable_rank": 2.3684491093158724,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.5795554811879993,
      "max_absolute_energy_error": 7.414200808852911e-09
    },
    "20261225_GRPB": {
      "matrices": 16000,
      "mean_stable_rank": 2.5386576237902045,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.5838151957057417,
      "max_absolute_energy_error": 7.53789208829403e-09
    },
    "20261225_GRPBN": {
      "matrices": 16000,
      "mean_stable_rank": 2.3367555025964974,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.5752535767536611,
      "max_absolute_energy_error": 7.4214767664670944e-09
    },
    "20261226_GRP": {
      "matrices": 16000,
      "mean_stable_rank": 2.382216158345342,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.5823329901769757,
      "max_absolute_energy_error": 7.283233571797609e-09
    },
    "20261226_GRPB": {
      "matrices": 16000,
      "mean_stable_rank": 2.549326876178384,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.5863518425617367,
      "max_absolute_energy_error": 8.221832104027271e-09
    },
    "20261226_GRPBN": {
      "matrices": 16000,
      "mean_stable_rank": 2.3497544598430395,
      "mean_retained_rank": 59.125,
      "mean_direction_cosine": 0.5779742487631738,
      "max_absolute_energy_error": 7.4214767664670944e-09
    }
  },
  "audit": {
    "status": "PASS",
    "cost": {
      "total": 12256,
      "optimizer": 12256,
      "functional": 0,
      "pending": 0,
      "by_phase": {
        "DETACH_S:engineering": 32,
        "DETACH_STOP:engineering": 32,
        "20261225_GRP:engineering": 32,
        "20261225_GRPB:engineering": 32,
        "20261225_GRPBN:engineering": 32,
        "20261226_GRP:engineering": 32,
        "20261226_GRPB:engineering": 32,
        "20261226_GRPBN:engineering": 32,
        "20261225_GRP:formal": 2000,
        "20261225_GRPB:formal": 2000,
        "20261225_GRPBN:formal": 2000,
        "20261226_GRP:formal": 2000,
        "20261226_GRPB:formal": 2000,
        "20261226_GRPBN:formal": 2000
      }
    },
    "calls": {
      "executed": {
        "detach_native": 128,
        "detach_encoder": 1536,
        "detach_reverse": 64,
        "engineering_native": 480,
        "engineering_encoder": 864,
        "engineering_reverse": 210,
        "formal_native": 24072,
        "formal_encoder": 576576,
        "formal_reverse": 12018,
        "diagnostic_reverse": 60,
        "evaluation_native": 17352
      },
      "by_group": {
        "DETACH:detach_native": 128,
        "DETACH:detach_reverse": 64,
        "DETACH:detach_encoder": 1536,
        "20261225_GRP:engineering_native": 80,
        "20261225_GRP:engineering_encoder": 144,
        "20261225_GRP:engineering_reverse": 35,
        "20261225_GRPB:engineering_native": 80,
        "20261225_GRPB:engineering_encoder": 144,
        "20261225_GRPB:engineering_reverse": 35,
        "20261225_GRPBN:engineering_native": 80,
        "20261225_GRPBN:engineering_encoder": 144,
        "20261225_GRPBN:engineering_reverse": 35,
        "20261226_GRP:engineering_native": 80,
        "20261226_GRP:engineering_encoder": 144,
        "20261226_GRP:engineering_reverse": 35,
        "20261226_GRPB:engineering_native": 80,
        "20261226_GRPB:engineering_encoder": 144,
        "20261226_GRPB:engineering_reverse": 35,
        "20261226_GRPBN:engineering_native": 80,
        "20261226_GRPBN:engineering_encoder": 144,
        "20261226_GRPBN:engineering_reverse": 35,
        "20261225_GRP:formal_native": 4012,
        "20261225_GRP:formal_encoder": 96096,
        "20261225_GRP:formal_reverse": 2003,
        "20261225_GRP:diagnostic_reverse": 10,
        "20261225_GRP:evaluation_native": 2892,
        "20261225_GRPB:formal_native": 4012,
        "20261225_GRPB:formal_encoder": 96096,
        "20261225_GRPB:formal_reverse": 2003,
        "20261225_GRPB:diagnostic_reverse": 10,
        "20261225_GRPB:evaluation_native": 2892,
        "20261225_GRPBN:formal_native": 4012,
        "20261225_GRPBN:formal_encoder": 96096,
        "20261225_GRPBN:formal_reverse": 2003,
        "20261225_GRPBN:diagnostic_reverse": 10,
        "20261225_GRPBN:evaluation_native": 2892,
        "20261226_GRP:formal_native": 4012,
        "20261226_GRP:formal_encoder": 96096,
        "20261226_GRP:formal_reverse": 2003,
        "20261226_GRP:diagnostic_reverse": 10,
        "20261226_GRP:evaluation_native": 2892,
        "20261226_GRPB:formal_native": 4012,
        "20261226_GRPB:formal_encoder": 96096,
        "20261226_GRPB:formal_reverse": 2003,
        "20261226_GRPB:diagnostic_reverse": 10,
        "20261226_GRPB:evaluation_native": 2892,
        "20261226_GRPBN:formal_native": 4012,
        "20261226_GRPBN:formal_encoder": 96096,
        "20261226_GRPBN:formal_reverse": 2003,
        "20261226_GRPBN:diagnostic_reverse": 10,
        "20261226_GRPBN:evaluation_native": 2892
      },
      "pending": null,
      "caps": {
        "detach_native": 160,
        "detach_encoder": 1632,
        "detach_reverse": 64,
        "engineering_native": 576,
        "engineering_encoder": 1152,
        "engineering_reverse": 240,
        "formal_native": 24300,
        "formal_encoder": 578000,
        "formal_reverse": 12024,
        "diagnostic_reverse": 60,
        "evaluation_native": 17352
      }
    },
    "polar": {
      "executed_steps": 12256,
      "executed_SVD": 98048,
      "pending": null,
      "cap_steps": 12256,
      "cap_SVD": 98048
    },
    "numerical_selfcheck_SVD": 10,
    "total_SVD": 98058,
    "workers_retired": true,
    "common_entry_empty_opt": true,
    "frozen_projection_constants": true,
    "finite_final_latest_opt_RNG_reads": true,
    "patient_EDES_exact": true,
    "logit_norm_matching": true,
    "synthetic_reverse_calls": 4,
    "audit_model_forwards": 0,
    "allocated_bytes_including_metadata": 224064640,
    "managed_elapsed_seconds": 5810.73499417305
  }
}
```

能量只匹配同一状态下普通SGD提议；不同轨迹不保证相同能量。保留普通momentum；P只变换encoder矩阵，B组仅将监督CE改成逐图现有类别平均，decoder仍正常更新。逐图类别平均CE可能损害新域适应；不将平均增益解释为患者安全。输出梯度范数匹配不保证参数梯度范数或clip系数匹配。部分恢复不等于超过原CE。所有旧失败保留，下一问题另立冻结协议。
