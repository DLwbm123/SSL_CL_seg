# S1：源患者梯度锥保护目标生成反馈

**冻结判定：PATIENT_GUARD_FAILED。** Full 与 PatientGuard 从同一 C1 Cut 源1600状态开始，在 BMC L4患者134片各做800次更新；两臂保留完整 BayeSeg 形状/外观图像生成，主要预测为同一 U 双logit。Full第800步逐张量等于封存R1 SwitchJoint；PatientGuard只对额外生成梯度施加四名RUNMC源训练患者风险梯度的欧氏半空间约束，目标期间不重放旧图/标签。Utarget为同更新/标注曝光的v45 U_B800；其 BMC训练读出取R1逐张量一致的RemainCut。

| 目标臂 | RUNMC开发Dice% | BMC开发Dice% | BMC完整L训练Dice% |
|---|---:|---:|---:|
| Utarget | 69.334 | 62.107 | 85.252 |
| Full | 67.091 | 65.184 | 87.243 |
| PatientGuard | 66.895 | 63.100 | 80.325 |

| 类型 | 预注册门 | 判定 |
|---|---|---|
| 机制 | Full_exact_R1 | PASS |
| 机制 | at_least_one_projection | PASS |
| 机制 | all_post_projection_constraints | PASS |
| 机制 | source_images_not_replayed | PASS |
| 旧域 | RUNMC_mean_vs_Utarget_ge_minus_1pp | FAILED |
| 旧域 | RUNMC_each_vs_Utarget_ge_minus_5pp | FAILED |
| 旧域 | RUNMC_HD95_vs_Utarget_le_1_10 | PASS |
| 旧域 | RUNMC_mean_vs_Usource_ge_minus_0_5pp | PASS |
| 旧域 | RUNMC_each_vs_Usource_ge_minus_5pp | PASS |
| 旧域 | no_RUNMC_patient_empty | PASS |
| 目标 | BMC_mean_vs_Utarget_ge_plus_2pp | FAILED |
| 目标 | BMC_at_least_3_of_4_nonnegative_vs_Utarget | FAILED |
| 目标 | BMC_each_vs_Utarget_ge_minus_5pp | FAILED |
| 目标 | BMC_HD95_vs_Utarget_le_1_10 | FAILED |
| 目标 | BMC_train_mean_vs_Utarget_ge_minus_5pp | PASS |
| 目标 | no_BMC_patient_empty | PASS |

匿名逐患者Dice（%）：

| 域 | 索引 | Utarget | Full | PatientGuard | Guard−U pp |
|---|---:|---:|---:|---:|---:|
| RUNMC | 1 | 68.390 | 68.958 | 68.225 | -0.166 |
| RUNMC | 2 | 80.852 | 70.327 | 68.215 | -12.637 |
| RUNMC | 3 | 82.765 | 77.886 | 75.295 | -7.470 |
| RUNMC | 4 | 45.330 | 51.193 | 55.845 | +10.515 |
| BMC | 1 | 47.889 | 43.915 | 45.524 | -2.365 |
| BMC | 2 | 63.678 | 71.806 | 67.599 | +3.921 |
| BMC | 3 | 64.254 | 74.081 | 75.329 | +11.075 |
| BMC | 4 | 72.606 | 70.934 | 63.948 | -8.658 |

与Utarget相比，PatientGuard旧/新域均值变化 -2.439/+0.993 pp；相对Full为 -0.196/-2.084 pp。旧域相对Utarget的配对最差 -12.637 pp、下降3/4名；新域相对Utarget的配对最差 -8.658 pp、获益2/4名。RUNMC绝对最差 Dice U/Full/Guard为 45.330/51.193/55.845%，四人插值p10为 52.248/56.522/59.556%。对4名患者配对均值的全部4⁴有放回重采样95%经验描述区间，Guard−U旧/新域为 [-10.053,+6.018]/[-5.513,+7.715] pp；并非独立患者推断区间，不进入门槛。

机制审计：正式800步中有 765 步投影改变了生成梯度；四源患者各自出现负内积的步数 [700, 557, 757, 665]，投影后最大残留负内积 1.63e-17。这仅是源状态梯度的一阶欧氏保证，Adam实际步与固定终点没有患者安全定理。

最后100批训练读数（生成bound每像素，和分割Dice不是同一指标）：

| 臂 | 标签风险 | 生成bound/像素 | qS均值logsd | qZ均值logsd |
|---|---:|---:|---:|---:|
| Full | 0.11221 | -2.47320 | -2.50202 | -1.06439 |
| PatientGuard | 0.10817 | -2.43923 | -2.50217 | -1.06148 |

任一冻结门失败即关闭本固定患者梯度锥策略，不调锚、权重、种子、阈值或回选checkpoint。方法新成本另含80次源切片梯度反向与目标双反向；C1 Cut源1600更新为继承历史成本。四名开发患者反复用于路线决策，不能宣称独立泛化、统计非劣、长序列CL或绝对患者安全；隐藏U标签与test未访问。v52另一投影负结果与R1安全失败仍保留。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[成本](EXECUTION_SUMMARY.json)。
