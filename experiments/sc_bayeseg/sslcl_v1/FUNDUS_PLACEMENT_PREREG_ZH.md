# Fundus 历史风格放置消融：预注册

本轮继三臂稳定性/历史特异性双FAIL后提出独立机制问题：同时改变L监督视图和U一致性视图，是否掩盖了单分支收益？不以消融改写已失败的FH结论。

新增HL（仅L历史，U current）和HU（L current，仅U历史），各seed163/164/165×5000更新，共30000。其他参数、源模型、RNG恢复/freshAdam、RGB/FOV六维统计、style50%概率及幅度、teacher原始U、384分辨率、损失、数据顺序与原三臂逐项相同。使用原source_U_s<seed>/S5000.pt；原F0=00、FH=11作为既有开发性对照，不重复源训练。只改history_l/history_u布尔开关，复用已验证训练引擎，不开放任何新数据、test/隐藏GT，不启用B或旧教师，无强度/权重网格。

描述量：每seed每域HL−F0、HU−F0、FH−HL−HU+F0的主Dice交互，以及cup/disc、像素HD95、空预测、10%尾部、扰动量。正负结果完整保留。两个单分支分别按旧域均值≥+1pp、新域均值≥−.5pp、任一seed/域cup或disc下降不得>2pp、HD95恶化不得>10%、无额外空预测、尾部下降不得>2pp进行资格判定，并额外要求旧域三个seed各自>0以排除单seed驱动。全部约束在新增训练前冻结。交互是解释性，不把任意符号直接称因果证明。

即使某放置通过，也只支持进一步做对应位置的generic对照，不宣称历史特异性；不得择优隐藏另一臂失败。若两放置都失败，停止把此历史风格方案当作可晋级候选，保留失败并优先分析目标域少量标注/样本尾部而非继续调风格强度。

新建pro5000 /data/bmw/SSL_CL_seg/runs/sslcl_fundus_placement_20261022和ops /data/bmw/.ops/x20261022_f2。为保守计费，与前轮共享original_started_unix=1792638157.570414及24h上限；磁盘统计含前轮root/ops和data副本，新增严格<50GB，controller49GB/85500s与外超时86100s。前轮工程INTEGRATION PASS作为同引擎依据，不伪造新的优化检查；本轮仅配置开关不同。启动前核GPU显存和挂载探针，进程中性argv。唯一30分钟监测，完成汇总公开source-only结果。
