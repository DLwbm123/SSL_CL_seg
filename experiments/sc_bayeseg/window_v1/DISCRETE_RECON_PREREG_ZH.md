# 标签条件类别重构：编码固定/可训练配对

从离散L训练S0600完整状态cursor8075重新分叉两臂，不挑交叉格，不重跑旧阶段。仅原两训练L和原四翻转组成batch8，不读取保留/U/目标，无新seed/增广/类别权重。目的是分离固定标签编码下生成均值拟合与编码适配；不是图像预测，也不是新完整概率目标。

fixed仅anatomy全部mean_w/mean_b训练；code臂同样生成均值加完整code_encoder参数训练，其余参数及Adam冻结。每步显式BayesianLayer取均值，qc取条件均值，最小化普通像素平均categorical CE；不加入KL、其余生成因素、类别平衡或额外识别CE。因此结果仅解释确定性条件重构，不能与原ELBO绝对值比较或归因于单一被删除因素。code_encoder含共享均值/方差输出参数，未使用的方差行可能由既有Adam历史移动，不声称其方差冻结；本阶段无随机code采样，不把该方差视为已校准后验。

两臂保留完整Adam/scheduler/RNG/lr3e-4/clip1/FP32；每臂600主+首末各2golden，共1208实际=1200主8工程，额外诊断反向0。两臂600主步前RNG精确配对。每100完整checkpoint，每0/300/600记录所有8训练视图逐类NLL、比例和24前景Dice，不择优/早停/调参。仅作为预定义描述性拟合门限：终态24Dice均>=既有.8；失败不盲延长，成功也不自动回写旧ELBO或启动U。

全部识别网络、structure、物理外观、anatomy logsd和所有prior/initial不变；冻结Adam不变，活跃参数Adam逐主步+600；真实cursor8075→8675，每段首末golden完整model/Adam/scheduler/RNG核验。所有实际cost连续入账；失败保留完整partial和已消耗，不盲重跑。公开仅源码协议匿名聚合中文正负报告，私有checkpoint仅数据盘；20GB硬限/18GB软阈值，截止UTC03:30。
