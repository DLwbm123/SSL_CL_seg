# Fundus 三臂开发性机制实验：冻结协议（2026-10-22）

目标：在 RGB U-Net 的 REFUGE→RIM_ONE_r3 顺序适配中，区分历史风格和一般强扰动。保留此前全部负结果；Prostate P0 失败，不启动其四臂。本轮不启用 B、不使用旧教师或旧域图像回放。

## 数据与准入

冻结 seed0 的 primary_20pct_split：REFUGE L/U/val=40/160/100，RIM_ONE_r3=16/63/40。只复制上述图像和 L/val 标签；U 标签、test 文件与 Drishti 均不访问。清单患者键可能只是图像键，跨眼身份未核实，不声称患者独立的外部验证。Fundus 已被历史开发使用，本轮仍是开发性证据。

既有图像中心裁剪 800、缩放 384，RGB/255；原标签 255→背景0、128→rim1、0→cup2。实际本轮419张原图均800×800，196张L/val标签无黑像素伪cup；保留历史文件，不重预处理。训练和评估固定384，不额外resize。监督CE+rim/cup Dice沿用既有损失；评估cup=(y==2)，disc=(y>0)，rim单列，主Dice=(cup+disc)/2。

每seed源模型固定5000更新，三seed163/164/165全部完成且REFUGE val各自cup Dice≥0.65、disc Dice≥0.85、cup/disc空预测均0才放行所有target臂。阈值在训练前冻结；源模型学生作准入评价。失败不得靠换seed、延长源训练、挑checkpoint放行，不把5000步称为收敛。

## 实验

3个独立scratch源U，每个5000步，源阶段L+U一致性与current-only strong。三target臂各3seed×5000，总60000更新。每seed三臂恢复同一个源S5000的student、EMA和RNG，fresh Adam；固定数据顺序与随机增强种子，不在恢复后重新覆盖RNG。Adam lr3e-4，gradclip1，EMA .99，KL权重1及1000步ramp沿用原实现，batch L2+U2。无超参网格。

F0：current-only常规风格。FG：current-only generic style，50%分支以当前域均值±0.75sd、sd×exp(uniform[-log1.5,log1.5])采样。FH：50%分支真实历史六维统计。L和U都应用对应style；教师固定读取原始weak U。三臂forward数量相同，generic不能读取历史幅度，不声称实际扰动强度相同。固定训练前抽样和训练步报告三臂实际L1/RMS变化及边界饱和比例（输出为0或1，区别于真实越界裁剪率）。

六维统计是每通道RGB均值/标准差；仅各域trainL图像，按清单患者/病例等权，无GT、val/U/test参与统计。FOV=原图max(RGB)>5/255，由图像决定且变换过程中固定，FOV外保持原像素；每张图按自身FOV moments标准化再着色。所有臂同样采用每通道sd×[.8,1.2]、均值±.05sd、固定余弦bias±.05sd，最终clip[0,1]。不试Lab/FDA/其他阈值。

## 结果判定

单位是清单patient_key宏平均，每个key内先平均图片；无跨眼身份保证。主指标cup/disc均值，rim单列。HD95按384网格像素，空预测以图像对角线惩罚，GT空类单列并从该类有效平均排除，不用MRI毫米。尾部固定为patient_key主Dice第10百分位（linear）。NLL/Brier辅助。

FH历史收益门：三seed均值old FH−F0≥1pp，new≥−0.5pp；任一seed任一域cup或disc相对F0下降>2pp则否决；相对F0每seed每域cup/disc惩罚HD95不得恶化>10%，空预测不得增加，尾部Dice不得下降>2pp。历史特异性门另要求old FH−FG三seed均值≥0.5pp且三seed各自>0；若只赢F0而未赢FG，只支持强扰动解释。报告各seed、均值、全部负门，不能择域择类。三seed不足以证明临床非劣。

## 工程与预算

pro5000新建/data/bmw/SSL_CL_seg/data_fundus_20261022及runs/sslcl_fundus_20261022，ops中性入口；旧服务器源只读。所有新增数据/副本/日志/checkpoint计入<50,000,000,000B，预期<3GB。首次真实GPU适配检查开始计统一24h，所有重试共享，controller85500s/49GB、独立timeout86100s，worker85200s。唯一30分钟监测；全源PASS控制器自动排三臂；任一失败不放行。本轮与历史输出隔离。
