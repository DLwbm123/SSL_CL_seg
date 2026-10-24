# B0 层次模型 × 监督风险缺角实验

**H-Std 新基座联合继续门：FAILED。** 本报告在两个新增臂各完成RUNMC1600＋BMC800后封存四格；旧 U-Std 与 H-Bal 结果只读复用。源/目标每格为4名开发患者等权三维Dice（%）。

| 臂 | 源RUNMC | 源BMC | 目标RUNMC | 目标BMC |
|---|---:|---:|---:|---:|
| U_Std | 63.261 | 30.997 | 69.334 | 62.107 |
| U_Bal | 56.598 | 17.129 | 22.944 | 51.595 |
| H_Bal | 57.073 | 32.850 | 63.593 | 57.362 |
| H_Std | 58.565 | 37.275 | 64.249 | 60.407 |

| 冻结继续门 | 判定 |
|---|---|
| source_RUNMC_within_1pp_of_U | FAILED |
| target_BMC_within_2pp_of_U | PASS |
| target_RUNMC_within_2pp_of_U | FAILED |
| old_mean_change_ge_minus_0_5pp | PASS |
| old_each_change_ge_minus_5pp | PASS |
| BMC_at_least_3_of_4_within_5pp_of_U | FAILED |
| BMC_each_within_10pp_of_U | FAILED |
| no_empty_foreground_collapse | PASS |

## 风险配置的家族内作用

正值表示将监督风险从Bal改为Std的Dice提高；最后一列是H和U两家族风险效应之差，不能解释为单个 Bayesian 元件的因果贡献。

| 位置 | H-Std−H-Bal pp | U-Std−U-Bal pp | 差中之差 pp |
|---|---:|---:|---:|
| source_RUNMC | +1.492 | +6.663 | -5.171 |
| source_BMC | +4.425 | +13.869 | -9.444 |
| target_RUNMC | +0.655 | +46.390 | -45.735 |
| target_BMC | +3.045 | +10.512 | -7.467 |

H-Std目标BMC匿名患者结果：

| 索引 | H-Std−U-Std Dice pp | H-Std−H-Bal Dice pp | precision % | recall % | 体积比 |
|---|---:|---:|---:|---:|---:|
| 1 | +14.861 | +1.117 | 77.91 | 52.53 | 0.674 |
| 2 | -5.902 | -1.288 | 47.71 | 73.23 | 1.535 |
| 3 | -4.506 | +10.694 | 44.25 | 91.97 | 2.079 |
| 4 | -11.253 | +1.656 | 46.67 | 89.53 | 1.919 |

## 判读

H-Std目标BMC均值60.407%，比H-Bal高3.045pp、比U-Std低1.700pp，均值距离门槛已经足够近；但源RUNMC比U-Std低4.696pp，目标RUNMC低5.085pp。目标BMC只有2/4名患者达到相对U-Std的−5pp界限，最差患者低11.253pp，因此联合继续门只通过4/8项。H-Std自身旧域源→目标4名患者均提高，说明本轮失败不是这一定义下的旧域遗忘；它也未证明对旧域的独立泛化。

两家族换成标准风险后的目标BMC均值都提高（H +3.045pp，U +10.512pp），但同一监督风险在两个家族的效应不同；U-Bal目标RUNMC仅22.944%，提示平衡风险本身可造成严重跨域损失。四格不能分离架构、层次变分项和推断策略的单独贡献，也不能把目标BMC均值接近U-Std误作患者尾部达标。下一步遵守冻结失败分支：关闭该B0配置，另立q(Z|S,I)结构与同结构无变分项对照的独立协议。

H-Std目标BMC全训练患者Dice 79.114%；空真值切片每片FP 287.46（79片），非空真值切片每片FN 346.13（55片）。U-Bal训练读出和两臂详细NLL/Brier/HD95、匿名逐患者计数见[分析](ANALYSIS.json)。

H-Std与旧H-Bal初始张量、初始后验输出及变分项在同随机流下逐项一致；U-Bal和旧U-Std使用相同源码及确定性构造程序，但历史U-Std没有保存初始状态文件，无法回溯逐项核对。该差别限制U家族的精确初值配对声明。BAL与STD同时改变CE权重及空片Dice约定，不能把效应全部归给CE。固定患者反复用于开发，单训练种子和MC抽样不构成独立泛化；本轮L-only不能证明半监督或持续学习。结果与实际资源记录见[执行摘要](EXECUTION_SUMMARY.json)，冻结定义见[协议](PREREG_ZH.md)。

H-Std未过门；本B0配置关闭，需先独立冻结下一结构假设与预算，不按开发结果回选权重、阈值、种子或中途检查点。
