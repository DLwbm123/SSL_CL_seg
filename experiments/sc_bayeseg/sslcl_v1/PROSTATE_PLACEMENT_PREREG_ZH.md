# Prostate历史作用分支的固定幅度干预

依据：原完整历史在U三seed旧新域均负；半幅无法稳定修复，B半幅新seed方向条件失败。图像单独诊断显示HALF通常比FULL扰动小，U10三个seed HALF KL甚至低于NONE，却不保证训练Dice改善。停止幅度/seed搜索，现测试历史作用于L监督分支或U一致性分支的区别，不预设哪个获益。

只固定原完整alpha=1、历史概率.5。新增BHL/BHU/UHL/UHU×seed163/164/165，共12目标臂各5000=60000更新。HL仅history_l=True，HU仅history_u=True，另一分支仍原current变换。所有臂仍use_u=True、原JL/JU/cons权重/EMA/ramp/Adam，几何与五个随机数不变。现有B10/U10为NONE、B11/U11为BOTH，组成每family/seed的2×2位置因素；不是“是否使用U数据”的因素。

各自原source S5000 student/teacher/RNG恢复、fresh Adam，163来自pilot，164/165来自全流程replication；initialization_seed/schedule_seed对应seed，training_seeds为空，不重训/串source。冻结split0，RUNMC→BMC，各4L14U4val，风格仅trainL，禁止HK/test/U隐藏GT。B为MAP G/F+局部q，不能称全Bayesian。

主要报告两域每seed及均值/SD的HL−NONE、HU−NONE、BOTH−HL、BOTH−HU、交互BOTH−HL−HU+NONE，同时给全端点/source/BWT/arrivalgain、HD95/NLL/Brier/空预测与4patient配对CI/留一。不选一域或一seed赢家；不再增seed/alpha。若同一路三seed两域同向优于NONE且优于BOTH，仅称该固定设置下方向支持；否则如实保留不稳定/交互。该探索基于已查看val提出，4患者/域不支持广泛临床结论。心脏已测HL/HU只作不同数据集背景，不能当本轮复现。

create-only root=/data/bmw/SSL_CL_seg/runs/sslcl_prostate_placement_20261022，ops=/data/bmw/.ops/x20261022_p5；独立新假设原钟含检查/所有12任务，GPU2345最多4并行。预计<1GB，root+ops49GB硬停、controller85500/外层86100秒，不以重试重置时钟。初始化检查12源映射/固定幅度/分支配置并在真实B/U数据对HL/HU各一次finite backward、teacher detached、0optimizer；无运行时模型代码改动。
