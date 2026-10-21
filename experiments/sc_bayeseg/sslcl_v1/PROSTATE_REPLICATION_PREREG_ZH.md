# Prostate完整源→目标流程的两个独立种子（启动前冻结）

初始163 pilot全部6阶段30000更新完成，source_B/U旧域70.43/76.99且无空预测。B历史效应旧+4.02新+1.32pp，U历史效应旧−1.85新−1.65pp；B旧4患者均改善但新域仅2/4，U效应留一可反号。不能根据一次运行、4val/域选择赢家或称B优越。

保持PROSTATE_PILOT_PREREG_ZH.md所有数据/患者/模型/损失/增强/训练和评价设置，仅增加完整初始化与抽样种子164、165。每seed各自从scratch训练source_B/source_U RUNMC5000；再从该seed对应source student/teacher与RNG进行B10/B11/U10/U11 BMC5000，fresh Adam。12阶段共60000更新，不复用163 source，不改变seed0数据划分，不看HK/test/U隐藏GT。

每seed B/U共享初始backbone数值；不同seed backbone不同。initialize使用initialization_seed，Schedule/view使用schedule_seed；training_seeds为空，source起点继承该初始化RNG，target恢复对应source RNG，避免把新seed强制重置target与163恢复逻辑混成另一协议。source_tasks逐target映射同family同seed，控制器等对应source完成才启动。

主要报告三完整流程seed每seed及均值/SD的B11-B10、U11-U10旧新Dice、forgetting/BWT/arrival gain、HD95/NLL/Brier/空预测与患者配对描述性CI/留一。163为探索，164/165为后续冻结确认；同一固定4val/域被重复评价，不能把12次预测当12位患者或将patient CI当seed CI。附B11-U11但不作结构单因素比较。保留全部seed/失败，不按表现挑源checkpoint、最佳迭代或改变历史强度。

新root=/data/bmw/SSL_CL_seg/runs/sslcl_prostate_replication_20261022，ops=/data/bmw/.ops/x20261022_p2；GPU2345最多4并行。共享只读数据，无新下载，预计新增<1GB，总root+ops49GB预防停；原始首次时钟含初始化检查，控制器23h45/外层23h55总限，重试不重置。启动检查12任务唯一/source映射/同seedbackbone相同与跨seed不同；原二分类真实数据集成和冻结审计沿用，新检查0optimizer。训练数值问题按原钟保存修复，不因性能低重跑选优。
