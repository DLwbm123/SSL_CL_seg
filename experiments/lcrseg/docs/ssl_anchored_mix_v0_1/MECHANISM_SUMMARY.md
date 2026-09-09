# 机制摘要

混合上下文对照自身有明显收益；增加U软目标提供较小增量，其患者区间仍含0。完整训练配方差不能解释为单像素因果证明。CE+Dice单独未稳定提高基础质量，普通MT_CED未获准复核。

P2 epoch100原图最终学生的两优化seed均值（每模型先患者平均）：

| 域 | 配方 | NLL | Brier | 错误前景confidence |
| --- | --- | --- | --- | --- |
| RIM_ONE_r3 | SUP_CE | 0.252028 | 0.067170 | 0.918310 |
| RIM_ONE_r3 | SUP_CED | 0.421143 | 0.079380 | 0.952993 |
| RIM_ONE_r3 | MIX_CTX | 0.161420 | 0.061204 | 0.874006 |
| RIM_ONE_r3 | MIX_CED | 0.138520 | 0.058227 | 0.857191 |
| Drishti_GS | SUP_CE | 0.335066 | 0.108682 | 0.905872 |
| Drishti_GS | SUP_CED | 0.486932 | 0.116007 | 0.937724 |
| Drishti_GS | MIX_CTX | 0.180702 | 0.080099 | 0.865511 |
| Drishti_GS | MIX_CED | 0.171109 | 0.078113 | 0.858111 |

P2 MIX_CED教师筛选的val前景质量（training-like draw先在患者内平均，再两个seed等权）：

| 域 | 类1/rim 2/cup | precision | accepted-correct recall |
| --- | --- | --- | --- |
| RIM_ONE_r3 | 1 | 0.755882 | 0.731419 |
| RIM_ONE_r3 | 2 | 0.820480 | 0.701192 |
| Drishti_GS | 1 | 0.773055 | 0.678488 |
| Drishti_GS | 2 | 0.793112 | 0.803014 |

这些是隔离val描述，不是隐藏train-U真实错误率；高confidence仍不等于正确。CE、GT-Dice、soft-CE、目标熵和去熵KL实际训练项及逐类来源曝光见SUPERVISION_SOURCE_ACCOUNTING.csv；同表全局loss在class行重复，除以steps求epoch均值，不跨类累加全局loss。

互补M及1−M来源计分、odd-U重复权重、q/mask停止梯度、上下文臂U目标梯度为零、共同warm-up及部署已在冻结源码CPU/CUDA资格验证。未额外计算真实参数梯度网格。
