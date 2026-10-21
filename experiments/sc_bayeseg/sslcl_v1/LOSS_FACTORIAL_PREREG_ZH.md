# 固定JL的B模型目标损失分解（启动前冻结）

普通U-Net完整2×2已完成，历史与目标U组合的收益不需要B主体。因此不能将B11改进归因于Bayesian独有效果。原B目标U开关同时改变JL系数并加入JU与cons，需隔离。

固定source_B、完整历史（L/U两路）、原target数据/增强/5000终点与fresh Adam，target seed164/165。每seed四臂J0C0/J1C0/J0C1/J1C1，共8×5000=40000正式更新。损失为Lsup + .005 JL + j*.005 JU + c*ramp*cons，其中JL是L弱/强图像目标均值，JU为U强视图图像目标，ramp=min(step/1000,1)，j,c∈{0,1}。JL在全部臂固定.005；不与旧无U臂的.01 JL混同。两seed全部明确重置训练RNG，schedule/view同步。

所有臂均抽取同样U图像、执行teacher和student U forward，以保持随机消耗和计算路径匹配；零系数仅屏蔽该项对梯度贡献。J0C0不是“不访问U”对照，不能与原B01视为同一训练配置。teacher全程EMA，评价student，与原协议一致。J1C1重新跑，避免算术重排和复用旧endpoint造成不匹配；默认legacy训练分支不改。对完整分支执行同RNG loss/gradient近似等价检查，对4臂做真实数据反向及权重代数检查，无optimizer更新。

预先报告两seed所有结果，JU效应J1C0-J0C0及J1C1-J0C1、cons效应J0C1-J0C0及J1C1-J1C0、交互J1C1-J1C0-J0C1+J0C0；旧/新Dice、forgetting/BWT、HD95/NLL/Brier/空预测与配对患者描述性CI。仅两target seed探索，source固定；不由结果调整权重或选终点。先判断收益是否主要来自一致性、图像目标是否带来稳定增益/伤害，再决定简化或跨数据集。

独立root=/data/bmw/SSL_CL_seg/runs/sslcl_loss_factorial_20261022，ops=/data/bmw/.ops/x20261022_l1；GPU2345最多4并行。只读原data/checkpoint、不访问test/U隐藏GT。新root+ops预计<1GB，49GB硬停；同一首次集成开始时钟含所有子任务/重试，23h45控制器与23h55外层timeout。新目录创建、挂载写读/显存准入，保留所有历史。跨数据集仍需独立准入审计与适配，不将本轮称为跨数据完成。
