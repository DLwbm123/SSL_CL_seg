# 普通U-Net目标阶段2×2补齐（启动前冻结）

动机：历史机制确认显示HL在165不稳定，完整历史保旧域更稳定；此前U11相对U10旧域也改善。补齐普通U-Net，避免将模型无关增强/一致性效果归因于Bayesian主体。

新增U00/U01 × target seed163/164/165，共6臂、每臂5000更新，总30000；U00无目标U/无历史，U01无目标U/有历史。引用既有同seed U10/U11（有目标U/无或有历史），不重跑或择优替换。固定原source_U（含源SSL），继承student/teacher，fresh Adam；163恢复source RNG不传training_seed，164/165明确重置匹配既有同seed。L抽样/增强/预算/评价均沿原协议，历史只来自源训练L的两维强度统计。

普通U模型image objective恒零；无目标U去掉一致性，但有目标U增加一次forward及EMA KL。此2×2估计目标U包和历史的效应及交互，不宣称纯计算量匹配，更不证明B模型的JU与cons各自作用。仅有L时history_l默认True，history_u无作用。不能称无SSL，因为source_U已用源U。

终点固定5000，报告全部三个seed的U01-U00、U11-U10、U10-U00、U11-U01和交互(U11-U10)-(U01-U00)；旧/新patient macro Dice、forgetting/BWT、HD95/NLL/Brier/空预测，以及描述性患者配对CI。既有source_U到达点为遗忘参照。源/目标域Siemens→Philips，原患者L/U/val划分与图像预测先于val GT读取不变，不读test或U隐藏GT。

独立create-only root=/data/bmw/SSL_CL_seg/runs/sslcl_unet_factorial_20261022；ops=/data/bmw/.ops/x20261022_u1。共享只读数据/初始checkpoint，预计新增<1GB；GPU2345最多4并行，启动需4GiB余量。控制器23h45、外层23h55硬超时，root+ops49GB预防停；所有子任务/重试共享首次时钟。旧历史保持。无需修改模型/训练代码；对六任务唯一性、seed匹配、family/use_u/history配置作断言，继承原U集成通过记录。
