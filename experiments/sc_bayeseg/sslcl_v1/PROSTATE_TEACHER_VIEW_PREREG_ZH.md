# 无标签教师视图配对：固定六臂探索

分支干预结果：U的HU−NONE旧三seed均负，均−1.089pp；新三seed均负，均−2.843pp。BOTH−HL在U旧新三seed也均负。HL−NONE新三seed均正（均+2.203）但旧不稳定（均−.389）。B的HL新正但旧164负；没有一路满足三seed两域同时胜NONE/BOTH。保留失败，不继续幅度或seed网格。

新假设是U一致性中原图教师与历史风格学生的输入差异可能影响效果。固定U模型、alpha1、history_l=False、原EMA/cons权重/ramp/5000终点，新增USAME0（无历史）与USAMEH（仅U历史）各seed163/164/165共6臂30000步。teacher_on_strong=True：教师与学生读取完全相同的确定性strong_u图像，教师仍EMA/no_grad，学生训练更新。对照既有U10（无历史、原图教师）和UHU（仅U历史、原图教师）。四格分解教师输入选择×U历史；不是简单拿USAMEH与旧UHU比较。

同视图教师减少了增强一致性部分，只保留EMA时序约束，可能强化错误预测；因此必须同时有USAME0，不能把一般训练目标变化称历史特异修复。输入相同不等于师生预测相同。默认teacher_on_strong=False保持旧路径。source student/teacher/RNG恢复fresh Adam，init/schedule对应163164165，training_seeds为空。无重训源、无新增seed或幅度。

固定split0/各域4val，禁HK/test/U隐藏GT；历史仅trainL统计。主要报告USAMEH−USAME0、UHU−U10及交互(USAMEH−USAME0)−(UHU−U10)，另报USAME0−U10、USAMEH−UHU；全seed旧新Dice、source/BWT/arrivalgain、辅助指标/4patient CI/留一。只有同视图下历史效应三seed两域均正且交互三seed两域正，才满足该设置的方向支持；否则不称稳定修复，不进一步调权重/幅度/补seed。val-informed探索，非患者外部确认。

create-only root=/data/bmw/SSL_CL_seg/runs/sslcl_prostate_teacher_view_20261022，ops=/data/bmw/.ops/x20261022_p6，GPU2345最多4并行，新独立原钟含检查。6×5000，预计<1GB，root+ops49GB预防停、85500控制器/86100外层，重试共享原钟。启动前核6源映射与四格定义、真实U两视图条件finite backward/teacher detached、默认路径与既有公式相同，零optimizer更新。
