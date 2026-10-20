# 源类别比例KL约束：语义覆盖修复结果

本轮完成、一次审计PASS。固定同一P/G、32码及初始参数，仅在原C0凝缩效用+.1maskKL上新增固定1.0 KL(源train-L类别像素比例||本批生成soft mask类别比例)。工程30次调用通过且丢弃，正式256outer＋256functional完成；总542=269optimizer+273functional，无toy、无在途、EXIT0/worker退休。不是扩大几何权重或增加码量。

## 固定语义覆盖门槛通过

|类别|源train-L像素比例%|生成soft比例%|hard非空码数/32|
|---|---:|---:|---:|
|背景|96.778992|97.372785|32|
|LV|1.152126|0.780830|19|
|MYO|0.987572|0.888321|26|
|RV|1.081310|0.958064|21|

三器官soft质量均落在预先冻结的[.25p,4p]内，hard非空码均≥8/32。对比旧C0三器官0/0/0，本轮19/26/21说明前景覆盖不足得到缓解。该检查不衡量形状、正确空间对应或真实旧域功能；类比例约束本身不保证生成mask有正确解剖，不把PASS写成旧域保持成功。

源统计仅合法源train-L10患者/20EDES/230切片，四类匿名像素累计，无U GT/val/test。原始输入和逐图特征未保存。形成后质量检查只32个generator输出，无新增native或患者前向。持久方法仍1,781,652字节，未增加码量或新模块。

## 计算与恢复审计

与原C0同成本口径：正式256optimizer+256functional；工程13optimizer+17functional，合计542，工程golden/真实image和mask超梯度/非空恢复/固定query下降通过。代码只更新码，native/P/G/frame逐值保持，初始模型与旧C0相同、空优化器；final/latest含optimizer/RNG/reads完全一致且有限。新增KL每次objective另生成8样本，不额外调用native；273次objective对应额外2184个generator样本，另32个终态质量样本。不能把同更新数称为完全相同计算量。

统一原始历时约523秒（8.72分钟）；含2MB小元数据预留约24.47MB，低于1GB硬预算。独立超时和过程guard生效，worker已退出，旧轮数据/结果未改变。源轮结束时target_training_started=false。

## 后继独立目标检验

覆盖资格通过后另立SEMANTIC_MASS_TARGET_PROTOCOL_ZH.md：固定R0，只替换这份记忆，工程32＋正式2000最后学生，EMA仅诊断；其他目标配置/数据合同沿原A。复用旧A作为历史匹配参照，不重跑既有患者评估；声明非同期和开发验证限制。相对A旧≥+.5pp且新≥−.5pp为成功解释，绝对保持和新域学习分别报告。目标结果尚未产生，不能将源轮覆盖PASS当作功能证据。

完整匿名收据见SEMANTIC_MASS_REPAIR_RESULTS.json与SEMANTIC_MASS_REPAIR_AUDIT.json。源码/协议公开；图像/生成mask/空间码/权重/患者记录/原日志/服务器配置不发布。
