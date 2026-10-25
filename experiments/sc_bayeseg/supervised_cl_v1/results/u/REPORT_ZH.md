# 全监督前列腺域持续学习首批结果

模型：`U`；训练标签范围：`all_train`；固定 seed 单次探索性运行。
评测仅使用冻结 val，test 未访问；患者级原始结果、模型和重放图像仅存 NAS。

训练病例数：RUNMC 18、BMC 18、I2CVB 11、UCL 8、BIDMC 7、HK 7。

| 训练阶段 | 评价中心 | 患者 | Dice 均值 % | Dice p10 % | 惩罚 HD95 均值 mm | 零重叠 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| source | RUNMC | 4 | 88.446 | 86.228 | 5.541 | 0 |
| finetune_BMC | RUNMC | 4 | 88.554 | 87.254 | 7.700 | 0 |
| finetune_BMC | BMC | 4 | 86.953 | 85.364 | 4.801 | 0 |
| replay_BMC | RUNMC | 4 | 88.475 | 87.543 | 5.688 | 0 |
| replay_BMC | BMC | 4 | 86.566 | 85.366 | 5.243 | 0 |
| finetune_I2CVB | RUNMC | 4 | 26.390 | 7.043 | 24.430 | 0 |
| finetune_I2CVB | BMC | 4 | 2.598 | 0.000 | 190.734 | 3 |
| finetune_I2CVB | I2CVB | 3 | 86.740 | 85.665 | 4.809 | 0 |
| replay_I2CVB | RUNMC | 4 | 80.347 | 72.451 | 7.606 | 0 |
| replay_I2CVB | BMC | 4 | 75.507 | 59.963 | 10.714 | 0 |
| replay_I2CVB | I2CVB | 3 | 85.599 | 83.412 | 5.274 | 0 |
| finetune_UCL | RUNMC | 4 | 64.977 | 47.995 | 11.814 | 0 |
| finetune_UCL | BMC | 4 | 57.589 | 30.760 | 20.064 | 0 |
| finetune_UCL | I2CVB | 3 | 75.733 | 73.845 | 7.204 | 0 |
| finetune_UCL | UCL | 2 | 89.524 | 87.721 | 3.457 | 0 |
| replay_UCL | RUNMC | 4 | 79.041 | 64.201 | 9.888 | 0 |
| replay_UCL | BMC | 4 | 74.608 | 61.431 | 20.927 | 0 |
| replay_UCL | I2CVB | 3 | 80.599 | 79.522 | 6.963 | 0 |
| replay_UCL | UCL | 2 | 90.282 | 90.254 | 3.315 | 0 |
| finetune_BIDMC | RUNMC | 4 | 84.350 | 81.673 | 10.887 | 0 |
| finetune_BIDMC | BMC | 4 | 69.816 | 51.352 | 10.876 | 0 |
| finetune_BIDMC | I2CVB | 3 | 75.841 | 73.486 | 10.440 | 0 |
| finetune_BIDMC | UCL | 2 | 79.131 | 71.216 | 7.500 | 0 |
| finetune_BIDMC | BIDMC | 2 | 87.309 | 86.334 | 11.735 | 0 |
| replay_BIDMC | RUNMC | 4 | 88.191 | 85.435 | 7.929 | 0 |
| replay_BIDMC | BMC | 4 | 80.883 | 74.947 | 11.661 | 0 |
| replay_BIDMC | I2CVB | 3 | 80.425 | 76.534 | 8.467 | 0 |
| replay_BIDMC | UCL | 2 | 89.401 | 87.641 | 3.491 | 0 |
| replay_BIDMC | BIDMC | 2 | 89.056 | 88.922 | 3.955 | 0 |
| finetune_HK | RUNMC | 4 | 84.187 | 79.946 | 7.792 | 0 |
| finetune_HK | BMC | 4 | 59.647 | 40.363 | 10.830 | 0 |
| finetune_HK | I2CVB | 3 | 74.444 | 69.041 | 26.693 | 0 |
| finetune_HK | UCL | 2 | 74.685 | 63.986 | 10.274 | 0 |
| finetune_HK | BIDMC | 2 | 39.029 | 23.734 | 14.477 | 0 |
| finetune_HK | HK | 2 | 90.627 | 88.981 | 3.600 | 0 |
| replay_HK | RUNMC | 4 | 88.774 | 87.114 | 7.337 | 0 |
| replay_HK | BMC | 4 | 75.898 | 63.466 | 10.660 | 0 |
| replay_HK | I2CVB | 3 | 79.354 | 76.999 | 13.140 | 0 |
| replay_HK | UCL | 2 | 88.317 | 87.298 | 4.009 | 0 |
| replay_HK | BIDMC | 2 | 84.813 | 82.504 | 7.498 | 0 |
| replay_HK | HK | 2 | 89.647 | 87.518 | 3.600 | 0 |
| joint | RUNMC | 4 | 90.349 | 88.779 | 4.827 | 0 |
| joint | BMC | 4 | 87.727 | 85.395 | 5.932 | 0 |
| joint | I2CVB | 3 | 84.985 | 84.847 | 5.754 | 0 |
| joint | UCL | 2 | 89.838 | 89.686 | 3.491 | 0 |
| joint | BIDMC | 2 | 88.964 | 88.218 | 5.198 | 0 |
| joint | HK | 2 | 91.476 | 89.719 | 3.208 | 0 |

## 当前训练域拟合

| 阶段 | 训练中心 | 患者 | 训练 Dice 均值 % |
| --- | --- | ---: | ---: |
| source | RUNMC | 18 | 94.358 |
| finetune_BMC | BMC | 18 | 92.355 |
| replay_BMC | BMC | 18 | 91.250 |
| finetune_I2CVB | I2CVB | 11 | 91.364 |
| replay_I2CVB | I2CVB | 11 | 88.265 |
| finetune_UCL | UCL | 8 | 95.457 |
| replay_UCL | UCL | 8 | 94.275 |
| finetune_BIDMC | BIDMC | 7 | 95.328 |
| replay_BIDMC | BIDMC | 7 | 94.087 |
| finetune_HK | HK | 7 | 94.950 |
| replay_HK | HK | 7 | 94.265 |
| joint | RUNMC | 18 | 92.736 |
| joint | BMC | 18 | 91.092 |
| joint | I2CVB | 11 | 88.161 |
| joint | UCL | 8 | 93.645 |
| joint | BIDMC | 7 | 93.449 |
| joint | HK | 7 | 92.290 |

## 旧域遗忘（相对该模型学习该域后的首次评价）

| 路线 | 对比 | 平均 Dice 变化 pp | 最差患者变化 pp | 下降患者/总数 |
| --- | --- | ---: | ---: | ---: |
| finetune | RUNMC_after_BMC | +0.108 | -1.145 | 2/4 |
| finetune | RUNMC_after_I2CVB | -62.056 | -89.364 | 4/4 |
| finetune | BMC_after_I2CVB | -84.355 | -90.239 | 4/4 |
| finetune | RUNMC_after_UCL | -23.468 | -43.908 | 4/4 |
| finetune | BMC_after_UCL | -29.364 | -66.633 | 3/4 |
| finetune | I2CVB_after_UCL | -11.007 | -14.702 | 3/3 |
| finetune | RUNMC_after_BIDMC | -4.096 | -7.512 | 4/4 |
| finetune | BMC_after_BIDMC | -17.137 | -41.182 | 3/4 |
| finetune | I2CVB_after_BIDMC | -10.899 | -13.524 | 3/3 |
| finetune | UCL_after_BIDMC | -10.393 | -22.541 | 1/2 |
| finetune | RUNMC_after_HK | -4.258 | -7.571 | 4/4 |
| finetune | BMC_after_HK | -27.306 | -52.401 | 4/4 |
| finetune | I2CVB_after_HK | -12.295 | -20.037 | 3/3 |
| finetune | UCL_after_HK | -14.839 | -30.468 | 1/2 |
| finetune | BIDMC_after_HK | -48.280 | -68.618 | 2/2 |
| replay | RUNMC_after_BMC | +0.029 | -3.297 | 2/4 |
| replay | RUNMC_after_I2CVB | -8.099 | -15.239 | 3/4 |
| replay | BMC_after_I2CVB | -11.059 | -34.936 | 4/4 |
| replay | RUNMC_after_UCL | -9.404 | -27.253 | 2/4 |
| replay | BMC_after_UCL | -11.958 | -32.454 | 4/4 |
| replay | I2CVB_after_UCL | -5.001 | -10.452 | 3/3 |
| replay | RUNMC_after_BIDMC | -0.254 | -1.890 | 3/4 |
| replay | BMC_after_BIDMC | -5.683 | -13.868 | 4/4 |
| replay | I2CVB_after_BIDMC | -5.174 | -7.234 | 3/3 |
| replay | UCL_after_BIDMC | -0.881 | -3.045 | 1/2 |
| replay | RUNMC_after_HK | +0.328 | -2.192 | 1/4 |
| replay | BMC_after_HK | -10.669 | -29.279 | 4/4 |
| replay | I2CVB_after_HK | -6.246 | -12.822 | 3/3 |
| replay | UCL_after_HK | -1.965 | -3.274 | 2/2 |
| replay | BIDMC_after_HK | -4.243 | -6.962 | 2/2 |

## 最终阶段：固定容量重放相对顺序微调

| 中心 | 平均 Dice 变化 pp | 最差患者变化 pp | 下降患者/总数 |
| --- | ---: | ---: | ---: |
| RUNMC | +4.587 | +1.973 | 0/4 |
| BMC | +16.250 | -2.175 | 1/4 |
| I2CVB | +4.909 | -1.639 | 1/3 |
| UCL | +13.632 | +1.531 | 0/2 |
| BIDMC | +45.784 | +23.778 | 0/2 |
| HK | -0.979 | -1.584 | 2/2 |

联合训练使用全部六域训练标签，是离线参照，不满足逐域到来的访问约束。
本批只有一个训练种子，且 val 在既往项目中反复用于开发；结果只能用于判断后续方法问题是否存在，不能构成独立确认。
本轮共 106000 次优化更新，耗时 7349.3 秒，新增目录 581944422 字节。
