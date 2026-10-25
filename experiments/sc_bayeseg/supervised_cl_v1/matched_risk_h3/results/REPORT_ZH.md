# H3：前景平衡风险与 Fisher 配对

同一新种子、同源初始化；U-FT 为标准对照，risk_control 与 matched_risk 都使用同一前景平衡 Fisher、无过程噪声在线 Gaussian。matched_risk 仅将部署 CE 替换为当前训练批的前景／背景等权 CE；其余完整条件目标保持一致。所有优化仅用当前中心 train；test 封存。

| I2CVB 终点 | U-FT | risk_control | matched_risk |
| --- | ---: | ---: | ---: |
| 三中心等权 Dice % | 38.184 | 51.095 | 47.640 |
| 三中心等权惩罚 HD95 mm | 58.796 | 31.265 | 37.441 |

冻结积极门：未通过；旧域平均到达后遗忘 57.520 pp，最差旧患者变化 -87.379 pp。

完整固定读数、患者尾部、训练拟合和成本见 ANALYSIS.json；逐患者结果与权重仅私有保存。
