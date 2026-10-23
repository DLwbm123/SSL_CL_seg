# v17 同4L标准U-Net单域对照（运行前冻结）

用户明确要求同样4名有标注患者标准U-Net公平对照。复用项目UNet2D base16 GroupNorm二分类，仅移除不参与分割的projection计算（Identity）；无坐标/概率/EMA/U/增强，普通CE+前景Dice（GT无前景切片Dice项0）沿用项目标准监督定义，不为baseline搜索损失或超参数。

与v11坐标纯语义主线同RUNMC冻结seed0的4L、4开发val，同H5强度预处理、均匀切片采样、相同data_seed+2*step序列、1600更新、batch1、Adam3e-4、float64、128双线性图像/nearest标签，输出恢复stored256后患者级whole-prostate Dice/NLL/Brier/HD95惩罚。初始化seed沿用主线cfg；不选checkpoint，仅最终1600步。标准U-Net不做主线1200步Adam清矩，且CE非主线类别平衡CE；这是同数据/曝光量参照而非仅架构差别消融，也非相同FLOP预算。训练步数匹配不证明U-Net充分收敛，禁止据结果挑改训练长度。

主比较v11coords纯语义（同4L）K0/K3所有三MC重复与确定性U-Net单值差距；v12/v14半监督另列并明确额外U。MC不算独立训练seed，不以一个seed/4开发val声称显著或普遍优越。先报告绝对差距，不事后定义接近阈值。

预检合成3次更新检查优化器恢复完全一致，真实批次有限梯度0更新与耗时准入；之后1600真实。7200秒/512MiB（预计checkpoint<100MiB）、外层7230秒，全局原起点24h/累计30GB不变。每100步原子替换当前恢复checkpoint，无原图概率输出。所有大产物NAS，test/U标签不读。失败计入同clock，公开源码协议匿名指标。
