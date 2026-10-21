# 半幅历史B模型的新训练种子有限复查

此前seed163/164/165为已看过val后提出alpha=.5的探索：BHALF−B10旧均+2.437、新+1.086pp且三seed两域同向；UHALF−U10旧均−.833、新−.638pp，不能稳定修复。现固定B模型、alpha=.5、原损失与5000终点，不做更多幅度网格，不扩展U历史主张。仅增加此前未用过的训练seed166/167，用于训练随机性复查，仍同4val/域，不能称独立患者验证；family选择本身依据已有结果，不能据此比较B/U架构。

每seed独立scratch source_B在RUNMC以L+U训练5000，再恢复其student/teacher/RNG到BMC的B10与BHALF，两者均fresh Adam、各5000，源训练不使用history。initialization_seed/schedule_seed=166或167，training_seeds为空。B10无历史、BHALF L/U同时半幅历史且概率.5，其余完全相同。共2源+4目标=6阶段30000更新。固定数据split0，各域4L14U4val，仅trainL统计风格，禁止HK/test/U隐藏GT；评价620163与原空间/4draw不变。

完整报告每seed BHALF−B10旧/新Dice、源端点、BWT/遗忘/arrival gain、HD95/NLL/Brier/空预测，4patient配对CI与留一，不把两新增seed合并成更多患者。预设方向复查条件：两个新增seed的旧新域差值均>0；否则如实判定方向复查不通过，不追加seed直至成功。该条件不是统计显著性。历史163–165与新增166–167分开列示，不隐藏不利结果，不改变终点。

新create-only root=/data/bmw/SSL_CL_seg/runs/sslcl_prostate_halfconfirm_20261022，ops=/data/bmw/.ops/x20261022_p4。只GPU2345最多4并行，source依赖完成才排对应target。原钟含初始化检查，控制器85500秒、外层86100秒硬限，root+ops49GB预防停，预计<1GB；重试沿原钟。代码沿用已通过真实binary/半幅集成版本，新增仅配置；启动前检查6任务/source映射/不重复与166/167初始化差异，0更新。完成后不继续幅度/seed网格，依据完整跨数据集证据收缩或转向不同可证伪机制。
