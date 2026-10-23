# 12小时自主实验汇总：2026-10-22 23:59 至 10-23 11:59

**结论：已完成4轮新实验、54组正式训练与评估，并完成1轮继承EMA结果的审计交付；尚无完整候选通过全部科学门槛。** 可靠margin有可重复的旧域均值收益，但患者尾部没有解决。旧域均值改善不能代替新域非劣和患者尾部门槛。

报告窗口为北京时间2026-10-22 23:59:44.865245—2026-10-23 11:59:44.865245。当前运行快照采于首次到期心跳约12:07，晚于截止约7分钟；以下在跑进度不冒充11:59精确切片。

## 已完成的新轮次

|完整轮次|正式组数|主候选−固定头CE旧域均值 pp|新域均值 pp|结论与失败点|
|---|---:|---:|---:|---|
|SOURCE_DECISION|12|+0.938738|-0.945366|FAIL；DS旧域收益明显，但新域损失约0.95pp，旧患者>2pp退化仍各1；选择性优于全特征的证据也不足。|
|RELIABLE_MARGIN|15|+0.263438|-0.026978|FAIL；新域非劣通过；旧域增量不足0.30pp，旧源CI与尾部失败（>1pp各5人，>2pp各2人）。|
|GROUP_RISK_MARGIN|12|+0.410188|-0.153847|FAIL；旧域增量通过0.30pp，但新增机制不优于原R；新域非劣CI与旧尾部失败。|
|COMPATIBLE_DECISION|15|+0.135863|-0.238172|FAIL；旧域增量不足0.30pp；新域三流均下降超过0.20pp；两域均不优于原R，尾部失败。|

各轮新目标seed不同、共享原源seed163；不同轮次数字不能直接当作严格配对排名。表内差值均是该轮三流配对固定头CE，主终点固定final EMA .999。SOURCE_DECISION原冻结新域门槛另含相对A的比较，完整报告保留；此总览统一列B仅为便于展示，不替代各轮门槛。

可靠margin原轮R−B旧域 +.293084/+.259562/+.237670pp、患者CI均正，新域 −.040442/+.000407/−.040898pp且非劣通过。这是正向效果，但不足以称完整方法成功。GROUP_RISK的T−B旧均+.410188pp也成立，但T−R旧均−.039818pp，不能归因于最大组机制。COMPATIBLE的C−R旧均−.057095pp，新域也三流负，已停止该组合。

继承的固定EMA确认：训练实际在10月22日23:40:54结束，早于本窗口；本窗口完成一次审计和公开交付。EMA旧均值80.950047/80.978531/81.034085%高于原源80.367407%，新域83.293866/83.278634/83.315949%，但尾部及对student新域非劣CI失败。其6096更新不计入下表新轮次总数，避免把跨窗口成本混记。

## 成本与故障留证

|新轮次|optimizer|native前向样本|reverse调用|完整运行秒|
|---|---:|---:|---:|---:|
|SOURCE_DECISION|24544|132386|24728|6535.680|
|RELIABLE_MARGIN|30832|226918|31171|11973.590|
|GROUP_RISK_MARGIN|24736|181972|24972|6249.299|
|COMPATIBLE_DECISION|30832|226918|91699|10511.986|
|合计|110944|768194|172570|35270.555|

新完成轮次管理历时合计9.797小时，非GPU内核计时；启动间隔、审计和写报告不包含其中。native指按样本记账的原生模型前向，reverse包括训练、校准和诊断反向，不等于optimizer步数。几何源SVD与合成自检另记，完整账见各轮AUDIT。当前REGIONAL截至快照额外13893更新，尚未做最终审计，不并入已完成成本。

两项实际故障均保留：SOURCE_DECISION首次入口依赖遗漏，零模型/零更新后修复，原钟未重置；GROUP_RISK首B完成训练及2892评估前向后汇总缺少preparation链接，用完整保存行重建汇总，零额外模型调用。原失败、退出码及修复回执保留，终态审计已核对。其余已完成轮次审计PASS，工程通过不等于科学成功。

## 存储与当前进度

累计账保留已完成轮次实际占用的保守上界、本地副本余量，以及活跃REGIONAL完整峰值预留，合计 **6,126,264,768字节（6.126GB）**，低于27GB准入线，另保留3GB恢复余量，严格总上限30GB。这个数是占用与预留合计，不是已写入量；不因12小时结束而清零。当前REGIONAL根快照占用317,755,392字节。未删除历史数据腾空间。

REGIONAL_MARGIN完整方法继续运行：P0全部通过、B/T零辅助64步全状态等价通过、12组资格通过；已完成6/12组训练评估，第7组20271002_W训练1157/2000，无FAIL、无未结更新。未读取局部验证性能，也未改冻结协议。

P0实际可靠前景边界覆盖67.54%/30.97%/45.65%，覆盖患者10/11/7；T边界损失份额94.14%，T−W归一梯度距.9757；工程上有可辨识信号，但不能据此预测患者性能。

下一明确动作是完成剩余6组及预注册全部门槛，唯一零前向终态审计后完整交付。T与W使用相同分层样本，区分边界/内部权重与单纯采样收益；无DS/投影、无旧图/U/新增测试访问。每轮原8h/<4GB硬限制和累计30GB继续，唯一30分钟监测继续。

## 可公开证据

- [SOURCE_DECISION完整报告](https://github.com/DLwbm123/SSL_CL_seg/blob/194742b5737109e09bf8c0244e091752de0ea028/experiments/lcrseg/gyro_spatial_memory_v1/SOURCE_DECISION_REPORT_ZH.md)
- [RELIABLE_MARGIN完整报告](https://github.com/DLwbm123/SSL_CL_seg/blob/44979d92f03f157d4f195cd2323698ed24def447/experiments/lcrseg/gyro_spatial_memory_v1/RELIABLE_MARGIN_REPORT_ZH.md)
- [GROUP_RISK_MARGIN完整报告](https://github.com/DLwbm123/SSL_CL_seg/blob/1a9829bcee76ba240faad84407bd4fa520eb9edf/experiments/lcrseg/gyro_spatial_memory_v1/GROUP_RISK_MARGIN_REPORT_ZH.md)
- [COMPATIBLE_DECISION完整报告](https://github.com/DLwbm123/SSL_CL_seg/blob/eaad7888b4817a6cff91a4a4853d426738bd163c/experiments/lcrseg/gyro_spatial_memory_v1/COMPATIBLE_DECISION_REPORT_ZH.md)
- [当前REGIONAL冻结协议](https://github.com/DLwbm123/SSL_CL_seg/blob/a06d9fa3f9e355f512d88af747cab91317766867/experiments/lcrseg/gyro_spatial_memory_v1/REGIONAL_MARGIN_PROTOCOL_V1_ZH.md)

所有数字是重复开发患者上的研究证据：旧23、新31患者在不同流之间共享，不能增加独立样本量；bootstrap不覆盖源随机、多重比较或反复开发。不宣称TMI充分证据、临床安全或无遗忘。当前没有达到全部门槛的成功方法，已保留全部负结果。
