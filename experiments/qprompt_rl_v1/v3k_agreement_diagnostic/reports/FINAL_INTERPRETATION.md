# V3K 最终解读：师生共同错误与分歧筛选的有限依据

## 完成与成本

2026-10-02 18:11:45北京时间完成，退出码0；1650个唯一attempt=success，无失败/重放。650L读取、2600U读取、6500图像推理、3300features/readout、3900HDF5打开；400行/100cell，50完整状态不变，50L cell精确复现J的teacher/student全量混淆及teacher接纳混淆。0optimizer、VJP、U标签、val、test。执行1980aebbb9fa87851e0fa8ac084c1bda7f5cea10。运行82.885秒，含准备465.274秒，峰值allocated193419776字节。累计optimizer仍237200，既有源40000和G只读VJP800单列。

## 预定描述量结果

FINE的U接纳区保留率：RIM99.9229%、Drishti99.9590%；对应训练L像素错误率仅降低0.0263/0.0097个百分点，共同支持类平衡错误降低0.1105/0.0321个百分点。默认快EMA与student高度一致，筛选几乎无空间；不能因被拒像素错误率很高便夸大实际去错量。

SLOW的U保留率98.2271%/97.7250%；L像素错误降低0.7108/1.0476个百分点，类平衡错误降低1.7989/3.8523个百分点。该诊断不推翻I的预定性能联合条件失败，也不证明训练时筛选能够改善新旧域Dice。

预定参照FIXED中分歧较多：U保留93.9296%/90.8460%；L像素错误降低4.1440/8.9148个百分点，类平衡错误降低11.0459/26.4056个百分点。这一较大效应是探索性的机制依据，不是本轮性能胜者。ENTRY师生100%一致且仍有错误，直接展示共同错误无法被一致性识别。

## 全部状态逐域统计

下表为五种子均值，覆盖单位%，错误差单位百分点。原始全部逐种子值与SD见SUMMARY，真实类支持和逐类值见CSV/STRATA。空值表示无真值或拒绝区为空。

|状态|角色|域|接纳区保留率|一致区像素错误差|共同类平衡错误差|拒绝区错误富集|
|---|---|---|---:|---:|---:|---:|
|ENTRY|L|RIM_ONE_r3|100.0000|0.0000|0.0000|—|
|ENTRY|L|Drishti_GS|100.0000|0.0000|0.0000|—|
|ENTRY|U|RIM_ONE_r3|100.0000|—|—|—|
|ENTRY|U|Drishti_GS|100.0000|—|—|—|
|ORIGINAL|L|RIM_ONE_r3|99.8711|-0.0377|-0.1585|26.2117|
|ORIGINAL|L|Drishti_GS|99.9674|-0.0121|-0.0262|33.3572|
|ORIGINAL|U|RIM_ONE_r3|99.8344|—|—|—|
|ORIGINAL|U|Drishti_GS|99.9559|—|—|—|
|FINE_05|L|RIM_ONE_r3|99.9312|-0.0263|-0.1105|36.3929|
|FINE_05|L|Drishti_GS|99.9607|-0.0097|-0.0321|45.2025|
|FINE_05|U|RIM_ONE_r3|99.9229|—|—|—|
|FINE_05|U|Drishti_GS|99.9590|—|—|—|
|FIXED_05|L|RIM_ONE_r3|94.8744|-4.1440|-11.0459|76.5438|
|FIXED_05|L|Drishti_GS|89.7970|-8.9148|-26.4056|78.3135|
|FIXED_05|U|RIM_ONE_r3|93.9296|—|—|—|
|FIXED_05|U|Drishti_GS|90.8460|—|—|—|
|SLOW_05|L|RIM_ONE_r3|98.6929|-0.7108|-1.7989|53.3329|
|SLOW_05|L|Drishti_GS|98.1620|-1.0476|-3.8523|55.4793|
|SLOW_05|U|RIM_ONE_r3|98.2271|—|—|—|
|SLOW_05|U|Drishti_GS|97.7250|—|—|—|

## 主要状态与探索参照的逐种子方向/离散性

|状态|域|指标|均值pp|SD pp|负差种子数/有效种子数|
|---|---|---|---:|---:|---|
|FINE_05|RIM_ONE_r3|teacher_error_delta|-0.02630|0.02081|5/5|
|FINE_05|RIM_ONE_r3|teacher_common_balanced_delta|-0.11055|0.08622|5/5|
|FINE_05|Drishti_GS|teacher_error_delta|-0.00972|0.00746|5/5|
|FINE_05|Drishti_GS|teacher_common_balanced_delta|-0.03209|0.03942|5/5|
|FIXED_05|RIM_ONE_r3|teacher_error_delta|-4.14402|0.68052|5/5|
|FIXED_05|RIM_ONE_r3|teacher_common_balanced_delta|-11.04585|1.74191|5/5|
|FIXED_05|Drishti_GS|teacher_error_delta|-8.91480|1.03192|5/5|
|FIXED_05|Drishti_GS|teacher_common_balanced_delta|-26.40563|4.26962|5/5|
|SLOW_05|RIM_ONE_r3|teacher_error_delta|-0.71079|0.24061|5/5|
|SLOW_05|RIM_ONE_r3|teacher_common_balanced_delta|-1.79887|0.25796|5/5|
|SLOW_05|Drishti_GS|teacher_error_delta|-1.04761|0.25580|5/5|
|SLOW_05|Drishti_GS|teacher_common_balanced_delta|-3.85234|1.29293|5/5|

## 预测类别覆盖与共同错误

|状态|域|U类0保留%|U类1保留%|U类2保留%|L一致区剩余teacher错误占原错误%|
|---|---|---:|---:|---:|---:|
|ENTRY|RIM_ONE_r3|100.0000|100.0000|100.0000|100.0000|
|ENTRY|Drishti_GS|100.0000|100.0000|100.0000|100.0000|
|ORIGINAL|RIM_ONE_r3|99.9295|99.0472|98.1767|97.6608|
|ORIGINAL|Drishti_GS|99.9845|99.7161|99.8781|99.4097|
|FINE_05|RIM_ONE_r3|99.9720|99.5233|99.1084|98.3413|
|FINE_05|Drishti_GS|99.9791|99.7515|99.9524|99.5236|
|FIXED_05|RIM_ONE_r3|98.3054|60.7058|55.6029|26.1292|
|FIXED_05|Drishti_GS|94.2245|54.2432|86.6930|16.4195|
|SLOW_05|RIM_ONE_r3|99.4701|88.7576|80.8728|67.4830|
|SLOW_05|Drishti_GS|98.8903|86.3010|97.4754|67.2308|

## 解释边界与下一轮

训练L已用于优化，师生共享源与训练历史，不能把L错误降低外推为U精度或独立泛化。当前仅clean视角，训练LCTX扰动并未在诊断中重现。所有U错误列为空；test封存，未产生任何新验证Dice。随机同覆盖期望只针对错误数量，不能代替训练对照。

固定teacher+student一致区降低错误的方向为两域5/5种子一致，且U保留量足够大，支持进行一轮新的、预注册的探索性干预；选择FIXED基于诊断而非宣称其原有验证性能成功。H固定教师在Drishti适应变差的反证仍保留。下一轮应同时运行原始L-only、默认FINE、FIXED、FIXED+一致筛选、FIXED+同数量随机筛选。随机对照逐步/逐图像/逐teacher预测类别使用一致筛选轨迹的保留数量，以区分空间选点与减少类别像素剂量；它匹配数量与类别构成，不保证梯度范数相同。固定教师、teacher阈值、lambda、LR、数据流、总步数保持一致。主要按新/旧域二维配对比较筛选与随机对照，再对FIXED/FINE检验实际收益，不能挑单域或单指标。完整方案需在启动前另行冻结。

这不是满意候选确认，继续使用既有优化种子以获得配对机制证据；只有预定联合条件通过才另列至少5个新种子做确认。若仅权衡或效应很小，不密集扫阈值。

## 交付

AGREEMENT.csv、STRATA.json、SUMMARY.json、ALL_COSTS.json、COMPLETION_AUDIT.json与本报告仅含匿名聚合；原始病例、标签、像素、checkpoint与日志保留NAS。
