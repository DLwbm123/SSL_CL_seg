# v40 启动回执：识别特征上的条件图像证据

已实际启动，L进入TRAIN、LU随后；尚无终点科学结果。协议 `conditional_feature_20261024_48`，GPU1，父/子955811/955812；完整argv与GPU可见名称为中性入口，启动显存约878MiB。这里只是启动快照，后续读取实际STATUS/ledger/EXIT。

首次实际起点1792799218.101388，绝对deadline1792813518.101388；硬上限14300秒，内部12000秒/800正式更新、预检600、每臂3600、每eval900，全根3GB停止线，独立timeout/rootguard。NAS挂载、容量、写读探针与GPU准入通过；实际campaign基线16,181,362,791字节加全额3GB低于29GB停止线；12小时窗口增长5,081,819,586字节加3GB也通过。历史和原始clock保留，无重试、无home回退，所有cache/tmp/大文件NAS。

一次预检PASS：小fixture对稠密Gaussian logprob与联合Gaussian Schur补一致，特征有限差分梯度、零设计单位噪声和隐藏值不入输入；1fixture autograd。真实独立aux1+两完整目标2模型反向，0Adam；辅助骨干梯度非零、q/域/cR直接梯度None、SbNone、同Ltask/anchor/globalRNG、sourcepayload/Adam/RNG恢复通过。真实预检3辅助骨干前向/6次17维Cholesky，另有原L前向和fixture小矩阵计算，不称零计算。

L/LU预检条件NLL为0.9839730/0.9940016，预测MSE为0.1314135/0.1598990；仅用于实现与有限性检查，不作为科学可行性或调整噪声/权重的依据。

共同v23L1600 ACTIVE，各400到2000，新L/LU同直接骨干条件风险、辅助池不同；原L目标/Adam/Sb保持。无新持久网络参数、额外标签或教师预测。复用v34固定遮挡规则，改用原phi16维特征+截距的解析局部线性读出；可见像素拟合后积分预测隐藏块，辅助不经过语义Pi/G/F。条件Gaussian解析不代表原后验精确或纹理等于语义。

后续按[冻结协议](PREREG_ZH.md)收口2session/4cell/800formal、固定终点原gate、三点审计与全部成本。只读封存结果，不旧重评分；失败不扩seed/特征层维度/先验噪声/遮挡/权重/增强，不自动retry。唯一runmc30分钟健康静默，12小时总结时点不变。
