# 原GYRO Adam配置检验：全native目标训练

native优化器改为Adam，固定lr1e-4/betas(.9,.999)/eps1e-8/amsgrad=False/foreach=False，矩阵coupled L2 wd.0005/vector0，clip1。相对于原SGD .001，这是optimizer+LR配置包，不单独归因于二阶矩或声称实际步长相等。回到source_coordinate原有Cutout的视图，原GYRO公式和模块不变；G beta1/3/gamma.45、M beta0/gamma.3，原native头、同S† native/frame+cal1000 phi/BN复合入口/空Adam状态，phi固定函数而目标native所有原允许参数训练。数据流/增强/256图bank/历史8/V2/P24不变。

同期G/M公平比较，SGD参照来自source_coordinate相同入口和数据流、非同期只读已有结果。与BayeSeg/CARE独立，无KD/历史CE/新模块/损失/伪类参照/读出绑定。无noNewU，不证明U语义因果贡献。固定最后2000学生，EMA只诊断，反复开发验证属探索非独立泛化/TMI创新。

**预设判断：NO_JOINT_BENEFIT_DEMONSTRATED**。优化器配置门槛False，frame门槛False，对Dense两域实用门槛False。

## 两域与训练口径

| 配方/组 | Siemens% | Philips% | 均值% | 源10L训练% | 目标13L训练% |
|---|---:|---:|---:|---:|---:|
| 本轮Adam1e-4/G_U | 70.6083 | 79.2221 | 74.9152 | 76.1747 | 86.6373 |
| 本轮Adam1e-4/M_U | 69.7458 | 79.8361 | 74.7909 | 76.0286 | 86.9768 |
| 同入口SGD参照/G_U | 78.8272 | 81.8802 | 80.3537 | 88.6240 | 84.6930 |
| 同入口SGD参照/M_U | 78.6423 | 82.0544 | 80.3484 | 88.4385 | 85.0609 |
| 原五组实用参照/dense_fixmatch | 78.7117 | 82.5778 | 80.6448 | 87.0113 | 86.8492 |

共同入口Siemens79.4337143%、Philips73.2043174%。原80.3674%至S†的-.9337pp是历史源接入变化，不计入本轮遗忘。

## 冻结比较

- G_U_optimizer_change：Siemens-8.218824 pp，Philips-2.658081 pp。
- M_U_optimizer_change：Siemens-8.896480 pp，Philips-2.218359 pp。
- G_minus_M：Siemens+0.862523 pp，Philips-0.613915 pp。
- G_minus_Dense：Siemens-8.103347 pp，Philips-3.355673 pp。

optimizer门槛为G Adam相对G SGD目标13L训练至少+.5pp、Philips至少+.3pp且Siemens不下降超过.5pp；frame门槛为同期G-M旧域至少+.5pp且新域不低超过.5pp；实用门槛为相对原Dense两域均不低超过.5pp、目标训练不低超过1pp。

## Siemens类别、ED/ES及绝对变化

| 组 | LV% | MYO% | RV% | ED% | ES% | 较入口pp |
|---|---:|---:|---:|---:|---:|---:|
| G_U | 81.7243 | 68.1299 | 61.9709 | 73.8280 | 67.3887 | -8.8254 |
| M_U | 81.2246 | 68.8543 | 59.1586 | 73.0803 | 66.4114 | -9.6879 |

## Philips类别、ED/ES及绝对变化

| 组 | LV% | MYO% | RV% | ED% | ES% | 较入口pp |
|---|---:|---:|---:|---:|---:|---:|
| G_U | 84.5227 | 77.1086 | 76.0351 | 81.0092 | 77.4351 | 6.0178 |
| M_U | 84.9073 | 78.0782 | 76.5227 | 82.0682 | 77.6040 | 6.6317 |

## 患者配对

| 比较/域 | 改善 | 下降 | 不变 | 中位数pp |
|---|---:|---:|---:|---:|
| 组间/入口/G_U_minus_entry/Siemens | 2 | 21 | 0 | -9.2376 |
| 组间/入口/G_U_minus_entry/Philips | 29 | 2 | 0 | +8.4300 |
| 组间/入口/M_U_minus_entry/Siemens | 1 | 22 | 0 | -9.1063 |
| 组间/入口/M_U_minus_entry/Philips | 29 | 2 | 0 | +9.4606 |
| 组间/入口/G_U_minus_M_U/Siemens | 14 | 9 | 0 | +0.9014 |
| 组间/入口/G_U_minus_M_U/Philips | 11 | 20 | 0 | -0.5202 |
| Adam减SGD/G_U/Siemens | 0 | 23 | 0 | -8.2915 |
| Adam减SGD/G_U/Philips | 8 | 23 | 0 | -0.7624 |
| Adam减SGD/M_U/Siemens | 0 | 23 | 0 | -9.4175 |
| Adam减SGD/M_U/Philips | 15 | 16 | 0 | -0.0152 |

类/EDES详细配对见SUMMARY.json。新域与旧域分开判断，不用均值掩盖代价。

## 库与梯度诊断

复用库256图覆盖48/48名合法源训练患者，源L图50、源U图206，无标签读取。患者覆盖不能等同器官或语义覆盖。本轮未重建reservoir，未重复读取源图像准备库，未重复入口评估。

| 组 | 加权谱/CE比中位数 | 比值最大值 | native更新路径 | projector更新路径 |
|---|---:|---:|---:|---:|
| G_U | 6.563292 | 699.789773 | 25.876192 | 0.000000 |
| M_U | 6.267314 | 620.232172 | 26.375394 | 0.000000 |

比值只诊断，不自适应调权或触发旧比值停止门槛。每步实际native LR1e-4及optimizer=Adam核对，所有Adam step/exp_avg/exp_avg_sq在实际非空恢复重放中逐张量一致，最终已更新参数step2000。

## 一次完成审计、成本和边界

两组目标各2000正式更新且EXIT0；工程各32含实际恢复重放和干净入口完整目标，全部丢弃。总4064=正式4000+工程64，预留/执行/阶段账本一致，无在途。两目标入口逐张量对应S† native/frame及只读既有source_coordinate phi/BN、双动量空，frame和原常量不变，final/latest模型/EMA/optimizer/RNG/reads有限且一致，全部worker退休。Siemens23患者/46EDES、Philips31/62及源10L/20、目标13L/26训练身份精确对应。一次审计只读既有记录与本轮checkpoint，没有额外患者前向或优化器调用，没有重复审计旧轮权重。

所有新产物在12小时父目录内，SpaceGuard限制新增20GB、18GB提前保护、2GB原子保存余量，每50次更新及写前检查。所有正负结果保留，不改旧终点；不读test或U隐藏GT，不改原数据和源参考。公开源码/协议/匿名聚合报告，不公开患者、图像、权重、库/frame/特征、原始日志或服务器配置。
