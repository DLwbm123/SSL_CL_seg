# V3K-AGREEMENT-DIAGNOSTIC：师生分歧与高置信错误的只读诊断

## 证据与可区分假设

V3J已完成并发布faa2b339927dd0df2e1aff2cb1ee46d63a5731cf。teacher SLOW−FINE在训练L上的真实类平衡接纳错误率+2.1674pp，5/5同向；RIM−1.0224pp、Drishti+5.3573pp。置信和接纳同时增加，不能直接提高置信阈值解决。原有E/F/H/I性能联合条件均失败，不继续lambda/窗口/beta微网格。

下一条可区分假设：在teacher置信>0.7的像素中，teacher/student argmax分歧区可能富集teacher错误，故一致性可能提供置信以外的信息。反解释包括：师生共同高置信错误，student训练L后的偏乐观纠错，以及L/U分歧分布不同。J的单模型汇总无法恢复同像素联合分布；必须读取成对预测。先用有限只读诊断判断有无机制依据，避免直接投入完整训练矩阵。没有第三教师、辅助损失或新训练。

## 冻结矩阵与数据

run_id=v3k_agreement_diagnostic_20261002T100400Z。复用V3I执行2acdec5fd217bf31585f343befa525198be8f71a的ENTRY、ORIGINAL、FINE_05、FIXED_05、SLOW_05完整状态，种子168..172，两域RIM_ONE_r3/Drishti_GS，共50状态。复用既有五源，0新源更新。

每状态遍历当前域完整L/U：RIM 16/63，Drishti 10/41，规范排序、无增强、batch2，U最后一批保留单图不补样。325个L批次+1325个U批次=1650，L读取650、U读取2600，总3250次图像读取和6500次图像推理；teacher/student各一次，3300次features/readout，3900次HDF5打开（L图像+标签，U仅图像）。各状态/种子复用相同患者，不能称3250独立样本。

仅允许CurrentData的train_labeled/train_unlabeled，U项必须只有image/geometry，禁止隐藏标签；val/test拒绝。标签用途与既有数据审计不变，不重新划分。teacher冻结eval预测，student现有clean确定性readout；诊断不模拟训练中的LCTX随机扰动，因此结果只支持该clean视角。

## 指标、对照与判读

固定阈值>0.7，不调阈值。四个像素组：ALL有效区域；ADMITTED高置信；AGREE高置信且argmax一致；DISAGREE高置信且argmax不一致。L忽略y=255；U仅用geometry有效域。每个seed/domain/state/role保存四组，共400行和100个聚合cell，均不含病例ID或逐像素数组。

每组保存像素数、对ALL和ADMITTED比例、teacher预测类计数、3×3师生预测联合矩阵。仅L保存teacher/student各自真实类混淆、像素错误率、每真实类支持/错误及支持类等权错误。无支持类用空值并报告支持数，不补零；U所有真值错误字段为空。

主要描述量预先固定为FINE_05、SLOW_05逐域逐种子的AGREE−ADMITTED teacher像素错误率与共同有支持真实类的平均错误差；同时报告DISAGREE−ADMITTED错误富集、U一致区保留比例以及每预测类覆盖。ENTRY/ORIGINAL/FIXED完整保留作参照。先cell汇总全部像素再种子等权，逐域给5个值、均值、样本SD，不以跨域总均值掩盖方向。

随机同覆盖对照只计算解析期望：若从ADMITTED中均匀选取与AGREE相同数量像素，期望保留teacher错误数=ADMITTED错误数×AGREE数/ADMITTED数。无需新随机mask或GPU运行。该式只针对错误数量；不声称有限样本的类平衡比率也等于其期望，不将此替代训练中的剂量匹配对照。

本轮无性能成功阈值或方法胜者。若一致区错误未降低、只一域降低、共同支持不足或U保留比例很低，应保留反证并重新考虑假设。即使L两域一致降低，也只构成开发机制候选；后续训练仍需冻结方法和剂量对照，并可能失败。不能用L结果估计U准确率/因果收益；师生共有错误不可由一致性识别。不得从本轮选择新阈值或宣称独立泛化。

## 工程与预算

准备起点2026-10-02T10:04:00Z；optimizer预算严格0，14小时optimizer截止2026-10-03T00:04Z仅元数据，诊断硬截止2026-10-02T11:04Z，含准备1小时。所有source/main/smoke/development/controller/replay均0，恢复reserve0，无自动重试；失败立刻停止。NativeOperations(update_cap=0)和零预算Ledger阻止训练。累计optimizer仍237200；既有源40000、G只读VJP800另列，本轮0VJP。

CPU自检手工概率，覆盖共同/分歧错误、255忽略、缺失类、零接纳、U无真值及随机期望。每状态完整trainer前后相等；50组L的teacher/student全量混淆及teacher接纳混淆必须精确复现V3J。分歧+一致矩阵必须严格等于接纳矩阵。1650唯一attempt=success，无失败/重放，0optimizer/grad/backward/EMA/val/test/U标签。完整状态检查涵盖student/teacher/Adam/scheduler/scaler/RNG/mode/grad/step/cursor等。最小诊断CPU检查已足够覆盖新增逻辑，不新增无训练任务的backward/过拟合步骤。

GPU4可用显存满足时使用，NAS挂载/容量/读写探针通过后create-only目录，经with_nas_storage.sh和既有guardian中性argv运行，ps/nvidia-smi确认后结束唤醒。每小时监测继续，不等待本轮结束。完整完成后发布AGREEMENT.csv、STRATA.json、SUMMARY.json、ALL_COSTS.json、COMPLETION_AUDIT.json及FINAL_INTERPRETATION.md到本目录reports和Downloads/V3K_AGREEMENT_DIAGNOSTIC_20261002，GitHub代理推送、SHA和匿名访问验证。原始病例信息、像素、checkpoint和日志只留NAS。
