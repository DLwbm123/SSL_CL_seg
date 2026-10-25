# 全监督前列腺域持续学习首批结果

模型：`B`；训练标签范围：`all_train`；固定 seed 单次探索性运行。
评测仅使用冻结 val，test 未访问；患者级原始结果、模型和重放图像仅存 NAS。

训练病例数：RUNMC 18、BMC 18、I2CVB 11、UCL 8、BIDMC 7、HK 7。

| 训练阶段 | 评价中心 | 患者 | Dice 均值 % | Dice p10 % | 惩罚 HD95 均值 mm | 零重叠 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| source | RUNMC | 4 | 88.278 | 86.181 | 4.956 | 0 |
| finetune_BMC | RUNMC | 4 | 89.017 | 87.562 | 6.656 | 0 |
| finetune_BMC | BMC | 4 | 87.235 | 85.441 | 4.922 | 0 |
| replay_BMC | RUNMC | 4 | 88.268 | 87.497 | 6.976 | 0 |
| replay_BMC | BMC | 4 | 84.627 | 83.563 | 8.220 | 0 |
| finetune_I2CVB | RUNMC | 4 | 22.908 | 2.830 | 87.667 | 1 |
| finetune_I2CVB | BMC | 4 | 1.857 | 0.000 | 190.752 | 3 |
| finetune_I2CVB | I2CVB | 3 | 86.694 | 85.282 | 4.451 | 0 |
| replay_I2CVB | RUNMC | 4 | 83.594 | 79.076 | 6.829 | 0 |
| replay_I2CVB | BMC | 4 | 78.609 | 70.042 | 10.795 | 0 |
| replay_I2CVB | I2CVB | 3 | 84.962 | 82.504 | 5.301 | 0 |
| finetune_UCL | RUNMC | 4 | 69.779 | 57.070 | 11.907 | 0 |
| finetune_UCL | BMC | 4 | 65.027 | 43.627 | 12.217 | 0 |
| finetune_UCL | I2CVB | 3 | 76.131 | 74.022 | 7.975 | 0 |
| finetune_UCL | UCL | 2 | 88.355 | 86.203 | 4.244 | 0 |
| replay_UCL | RUNMC | 4 | 79.959 | 67.636 | 10.102 | 0 |
| replay_UCL | BMC | 4 | 74.992 | 62.432 | 12.658 | 0 |
| replay_UCL | I2CVB | 3 | 79.762 | 78.456 | 7.737 | 0 |
| replay_UCL | UCL | 2 | 90.139 | 89.541 | 3.315 | 0 |
| finetune_BIDMC | RUNMC | 4 | 85.511 | 81.624 | 7.199 | 0 |
| finetune_BIDMC | BMC | 4 | 68.287 | 48.286 | 9.475 | 0 |
| finetune_BIDMC | I2CVB | 3 | 75.926 | 75.444 | 30.532 | 0 |
| finetune_BIDMC | UCL | 2 | 83.295 | 78.048 | 5.079 | 0 |
| finetune_BIDMC | BIDMC | 2 | 85.569 | 83.978 | 5.405 | 0 |
| replay_BIDMC | RUNMC | 4 | 89.241 | 86.573 | 6.359 | 0 |
| replay_BIDMC | BMC | 4 | 79.761 | 75.089 | 20.598 | 0 |
| replay_BIDMC | I2CVB | 3 | 82.380 | 78.499 | 6.508 | 0 |
| replay_BIDMC | UCL | 2 | 88.494 | 87.737 | 3.729 | 0 |
| replay_BIDMC | BIDMC | 2 | 85.265 | 84.872 | 5.172 | 0 |
| finetune_HK | RUNMC | 4 | 81.727 | 76.100 | 10.020 | 0 |
| finetune_HK | BMC | 4 | 53.489 | 36.265 | 15.210 | 0 |
| finetune_HK | I2CVB | 3 | 75.258 | 71.614 | 26.602 | 0 |
| finetune_HK | UCL | 2 | 84.040 | 80.064 | 5.079 | 0 |
| finetune_HK | BIDMC | 2 | 61.697 | 60.612 | 9.396 | 0 |
| finetune_HK | HK | 2 | 90.537 | 88.509 | 3.600 | 0 |
| replay_HK | RUNMC | 4 | 88.080 | 85.038 | 7.423 | 0 |
| replay_HK | BMC | 4 | 69.014 | 51.839 | 14.628 | 0 |
| replay_HK | I2CVB | 3 | 80.626 | 77.223 | 12.213 | 0 |
| replay_HK | UCL | 2 | 87.316 | 86.945 | 3.691 | 0 |
| replay_HK | BIDMC | 2 | 84.695 | 82.617 | 6.777 | 0 |
| replay_HK | HK | 2 | 90.997 | 89.269 | 3.600 | 0 |
| joint | RUNMC | 4 | 91.005 | 88.476 | 6.702 | 0 |
| joint | BMC | 4 | 86.262 | 85.032 | 7.466 | 0 |
| joint | I2CVB | 3 | 85.237 | 83.539 | 5.570 | 0 |
| joint | UCL | 2 | 89.900 | 89.576 | 3.315 | 0 |
| joint | BIDMC | 2 | 88.247 | 87.953 | 5.810 | 0 |
| joint | HK | 2 | 90.941 | 89.273 | 3.600 | 0 |

## 当前训练域拟合

| 阶段 | 训练中心 | 患者 | 训练 Dice 均值 % |
| --- | --- | ---: | ---: |
| source | RUNMC | 18 | 94.175 |
| finetune_BMC | BMC | 18 | 92.319 |
| replay_BMC | BMC | 18 | 90.508 |
| finetune_I2CVB | I2CVB | 11 | 90.940 |
| replay_I2CVB | I2CVB | 11 | 87.774 |
| finetune_UCL | UCL | 8 | 95.556 |
| replay_UCL | UCL | 8 | 94.685 |
| finetune_BIDMC | BIDMC | 7 | 95.443 |
| replay_BIDMC | BIDMC | 7 | 94.390 |
| finetune_HK | HK | 7 | 94.973 |
| replay_HK | HK | 7 | 94.200 |
| joint | RUNMC | 18 | 92.654 |
| joint | BMC | 18 | 90.451 |
| joint | I2CVB | 11 | 87.343 |
| joint | UCL | 8 | 93.921 |
| joint | BIDMC | 7 | 93.539 |
| joint | HK | 7 | 92.080 |

## 旧域遗忘（相对该模型学习该域后的首次评价）

| 路线 | 对比 | 平均 Dice 变化 pp | 最差患者变化 pp | 下降患者/总数 |
| --- | --- | ---: | ---: | ---: |
| finetune | RUNMC_after_BMC | +0.739 | -1.745 | 1/4 |
| finetune | RUNMC_after_I2CVB | -65.370 | -86.606 | 4/4 |
| finetune | BMC_after_I2CVB | -85.378 | -88.915 | 4/4 |
| finetune | RUNMC_after_UCL | -18.500 | -29.551 | 4/4 |
| finetune | BMC_after_UCL | -22.208 | -54.991 | 3/4 |
| finetune | I2CVB_after_UCL | -10.563 | -11.619 | 3/3 |
| finetune | RUNMC_after_BIDMC | -2.767 | -8.708 | 2/4 |
| finetune | BMC_after_BIDMC | -18.947 | -49.881 | 4/4 |
| finetune | I2CVB_after_BIDMC | -10.769 | -13.217 | 3/3 |
| finetune | UCL_after_BIDMC | -5.060 | -14.309 | 1/2 |
| finetune | RUNMC_after_HK | -6.551 | -11.587 | 4/4 |
| finetune | BMC_after_HK | -33.746 | -58.186 | 4/4 |
| finetune | I2CVB_after_HK | -11.437 | -14.271 | 3/3 |
| finetune | UCL_after_HK | -4.315 | -11.974 | 1/2 |
| finetune | BIDMC_after_HK | -23.872 | -27.216 | 2/2 |
| replay | RUNMC_after_BMC | -0.010 | -3.614 | 1/4 |
| replay | RUNMC_after_I2CVB | -4.684 | -10.548 | 3/4 |
| replay | BMC_after_I2CVB | -6.019 | -18.542 | 4/4 |
| replay | RUNMC_after_UCL | -8.320 | -24.265 | 3/4 |
| replay | BMC_after_UCL | -9.635 | -28.722 | 4/4 |
| replay | I2CVB_after_UCL | -5.200 | -7.890 | 3/3 |
| replay | RUNMC_after_BIDMC | +0.963 | -2.457 | 2/4 |
| replay | BMC_after_BIDMC | -4.866 | -10.997 | 4/4 |
| replay | I2CVB_after_BIDMC | -2.582 | -4.369 | 3/3 |
| replay | UCL_after_BIDMC | -1.645 | -1.843 | 2/2 |
| replay | RUNMC_after_HK | -0.199 | -3.554 | 2/4 |
| replay | BMC_after_HK | -15.613 | -39.625 | 4/4 |
| replay | I2CVB_after_HK | -4.336 | -12.685 | 2/3 |
| replay | UCL_after_HK | -2.823 | -3.107 | 2/2 |
| replay | BIDMC_after_HK | -0.570 | -2.676 | 1/2 |

## 最终阶段：固定容量重放相对顺序微调

| 中心 | 平均 Dice 变化 pp | 最差患者变化 pp | 下降患者/总数 |
| --- | ---: | ---: | ---: |
| RUNMC | +6.352 | +1.384 | 0/4 |
| BMC | +15.525 | +11.508 | 0/4 |
| I2CVB | +5.368 | +1.431 | 0/3 |
| UCL | +3.276 | -2.157 | 1/2 |
| BIDMC | +22.999 | +19.045 | 0/2 |
| HK | +0.460 | +0.085 | 0/2 |

联合训练使用全部六域训练标签，是离线参照，不满足逐域到来的访问约束。
本批只有一个训练种子，且 val 在既往项目中反复用于开发；结果只能用于判断后续方法问题是否存在，不能构成独立确认。
本轮共 106000 次优化更新，耗时 8519.9 秒，新增目录 591536451 字节。
