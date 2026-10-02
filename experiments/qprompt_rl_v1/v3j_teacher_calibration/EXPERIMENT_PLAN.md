# V3J-TEACHER-CALIBRATION：训练L置信校准的只读诊断

## 决策依据与边界

V3E/F/H/I预定联合改善条件连续未通过。最新I的SLOW−FINE新域−0.1310/旧域+0.4654个百分点，2/5joint，30历史控制精确复现。不能再用更细beta/λ/窗口网格追逐正分。G观察到U置信很高，但没有量化高置信错误；H/I表现出教师更新与域依赖权衡。缺失证据是允许访问的当前训练标签上，教师置信能否区分正确/错误，且其类别/域分布是否不同。

本轮不新增参数训练，不读取隐藏U标签、val或test。全部当前训练L被student用过，因此这是训练内代理诊断，不能给出真实U错误率、独立泛化或因果结论。跨域原始标签和难度不同，差异只能描述，不自动推出域专用超参数。预设指标不能用于本轮选择新置信阈值。

## 冻结覆盖

run_id=v3j_teacher_calibration_20261002T085800Z；V3I执行2acdec5fd217bf31585f343befa525198be8f71a的完整ENTRY和ORIGINAL/FINE_05/FIXED_05/SLOW_05 step1200状态，种子168..172，两域RIM_ONE_r3/Drishti_GS。50训练状态，每个状态遍历其完整当前域训练L：RIM16张、Drishti10张，固定规范排序、batch2、不翻转/旋转/扰动。共325批次、650带标签图像读取，每图分别做teacher与student一次预测，共1300图像预测（650 feature/readout调用），100个匿名聚合行。病例自然跨种子/状态复用，不能把650视为独立患者。

复用V3E五源创建同一trainer，再恢复I完整状态；teacher使用冻结eval模型，student用现有clean()确定性原生readout。统一无增强推理不涉及step0与1200的训练LCTX阶段差异。每状态前后完整student/teacher/Adam/scheduler/scaler/RNG/grad/mode等一致；10个FIXED教师的全部预测汇总与ENTRY严格相同作为内部校验。不还原缺失的中间EMA状态。

## 指标与汇总

只在y!=255像素计算，先汇总同一seed/domain/state/model的全部训练L像素。存3×3混淆矩阵与>0.7接纳混淆矩阵；像素准确率、真实类平均recall、平均置信、接纳比例、接纳像素错误率、每真实类接纳错误率及有支持类等权平均、NLL与真实类等权NLL。接纳类没有支持时该类指标为空并报告支持类数，不补0。10个固定等宽置信区间[0,.1),...,[.9,1]存count/confidence_sum/correct，并计算ECE=sum_bin|sum_conf−correct|/N。

主要描述量是teacher的真实类平衡接纳错误率：SLOW−FINE逐种子两域均值，以及ENTRY Drishti−RIM逐种子域差。没有性能胜出阈值；报告全部数值、5种子SD、缺失支持与方向，辅助ECE/NLL/准确率/接纳等完整展示，student作为训练内参照，不以表现最好的模型挑指标。定义不同于验证集Macro Dice，不冒称同一评分。

置信错误若存在，只说明在该训练L代理上置信过滤有局限；不存在也不能证明U伪标签正确或排除训练中错误。不会自动据结果换阈值、增加第三教师或采用事后域策略。下一步必须将已有负结果、诊断局限和新可区分假设一并考虑。

## 资源、安全门槛与交付

准备2026-10-02T08:58Z开始；optimizer预算严格0，默认optimizer窗口截止22:58Z仅作元数据，诊断硬截止09:58Z（含准备1小时）。source/main/smoke/development/controller/replay均0，恢复reserve0，无自动重试。上限325唯一成功批次，0autograd/VJP/backward/EMA更新；NativeOperations(update_cap=0)+零预算Ledger双重约束。数据getitem额外拒绝非train_labeled角色，构造元数据可以包含L/U，但U影像/标签访问计数必须0。测试封存，不改数据、旧模型、NAS历史文件。

CPU手构概率验证混淆、255忽略、真实类平衡接纳错误、ECE和空接纳；GPU每状态完整不变检查。预期L sample650、features/readout650、50状态、100汇总行，0U/val/test/optimizer/grad/EMA。逐批attempt/success与操作计数全部保存，错误立即停，不盲目重启。累计optimizer仍237200，既有源40000和G800VJP单列，本轮0VJP。

GPU4满足显存时使用，NAS新create-only目录和挂载/空间/读写探针，通过with_nas_storage.sh/既有guardian中性argv启动，短检查后继续每小时监测。源码/计划先冻结再运行。完成后发布CALIBRATION.csv、RELIABILITY.json、SUMMARY.json、ALL_COSTS.json、COMPLETION_AUDIT.json和实质判读报告，保存本分支reports与Downloads/V3J_TEACHER_CALIBRATION_20261002，经GitHub代理推送、SHA与匿名访问验证；不导出病例ID、病例级分数、标签或像素数组。
