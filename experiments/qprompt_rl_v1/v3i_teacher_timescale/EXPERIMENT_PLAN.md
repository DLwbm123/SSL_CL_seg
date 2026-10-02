# V3I-TEACHER-TIMESCALE：一档慢EMA的开发机制实验

## 证据与决策

V3H固定入口教师相对原EMA：新域−0.5283个百分点、旧域+1.3334，仅1/5种子joint，三个预定条件全失败。RIM新域5/5提高、Drishti5/5下降；Drishti的rim/cup/disc union同时下降，不足以把损失只归因于细粒度内部边界。固定教师是权衡而非稳定优胜。V3G只提供漂移关联，不能把固定干预失败改写成机制成功。H结果已经独立发布，不修改其标准。

本轮只检验EMA0.999这个预定值，比较原0.99和完全固定两端，保持其他设置不变。约1/(1−beta)记忆尺度从100步变为1000步，与1200步适应长度同量级；0.999取整数量级，不根据分数优化或微扫。假设慢但非零的教师适应可能缓解固定教师的新域损失，并比原EMA更保留旧域。实验也可能只移动权衡；若失败，不追加beta密集网格，先重新总结剩余假设与缺失证据。

## 冻结设计

run_id=v3i_teacher_timescale_20261002T045400Z；复用V3H执行2d96276f008d2f77471fae3463bb7a76c4cf977b的完整入口，V3E五源，五种子168..172，两域RIM_ONE_r3/Drishti_GS，H1200。四组×两域×五种子=40轨迹。

|组|U权重/窗口|U调用|教师更新|
|---|---|---:|---|
|ORIGINAL|0|0|原0.99 EMA|
|FINE_05|0.5全程|1200|原0.99 EMA|
|FIXED_05|0.5全程|1200|入口完整教师始终不变|
|SLOW_05|0.5全程|1200|0.999 EMA|

唯一新增科学因素是U教师EMA的时间尺度。共享NativeLRParent.update_dense_ema增加可选decay，默认0.99的浮点运算与alpha=.01保持原样；实验实例门控将0.999同时传给实际更新与validate_only预提交校验。非浮点和grad_update缓冲沿用原复制规则。FIXED仍跳过全部实际教师更新，不能把decay=1时的部分缓冲复制冒充完整冻结；无第三教师/额外loss/回放。主student、原生LCTX、学习率时间轴、Adam、rank8、head、A侧约束、置信0.7、valid归一化、数据流与增强全部不变。其他框架与已完成轮次仍使用默认调用；历史NAS代码和checkpoint不变。

同入口每5步轮换，100步完整状态，入口/300/600/1200部署快照。每条计数：ORIGINAL/FINE/SLOW各1200教师updates/0skip，FIXED0updates/1200skip。最终10固定teacher与入口全状态一致，30动态teacher变化；三类历史控制共30条的新旧域终点评分须精确复现V3H。复制门控元数据在RUN_LOCK和TEACHER_AUDIT，完整状态含telemetry；恢复必须根据锁定方法选项，不能跨策略误恢复。

## 数据、主比较和失败规则

仅原冻结L/U训练、val开发评估；不改HDF5/manifest/split，不读test。种子和val都已使用，本轮明确是机制开发，非独立确认。底座NativeLRParent/LR_SRC_A保留，原KI身份未核实。全部训练完成后评估固定1200步合并student，不挑best/EMA、不删负seed，不看中间val决定训练。40终点+10入口=50评估文件100域评分。

主要SLOW−FINE：新旧域平均都>0；至少4/5种子域均值两项>=0且一项>0；两个域各自五种子均值两项>=0。三项全过才记录timescale_intervention_supported，否则未支持联合改善。次要SLOW−FIXED和SLOW−ORIGINAL展示完整二维权衡，不能隐瞒旧域回落或原生对照退化。报告逐cell、逐域/种子配对差、SD、描述性t95区间和均值Pareto，不用主观加权总分或事后容忍阈值取代规则。若有候选，再冻结至少五个新优化种子确认；若失败，不自动增加.995等邻近档。

## 预算与工程检查

准备从2026-10-02T04:54:00Z开始，optimizer截止18:54Z、硬截止20:54Z，包含本次分析准备，不延期。main48000，smoke计划80上限100，source/development/controller0，恢复reserve2405=ceil(.05×48100)。计划新增48080；前V3D/E/F/H累计189120，完成累计237200；G另800只读VJP，既有源40000单列。不自动重试，同错误两次停止受影响路径，所有失败重放计费。

CPU检查真实共享denseEMA默认/0.999方程、grad_update与整型缓冲、只读validate、不合法decay拒绝、实例固定/动态门控。两域四组各5次真实Adam/预演/只读/精确恢复+5次计时=80次smoke，并检查teacher是否变化/计数。默认路径最终30控制精确复现是必要门槛。通过实测串行估计×1.25+1800秒评估余量确认完整矩阵能在原截止前结束，不缩矩阵换数据。

GPU4显存核验；NAS create-only目录、真实挂载/容量/读写探针，with_nas_storage.sh与既有guardian中性argv，ps/nvidia-smi短检查后交还每小时监测。源码和计划先冻结提交，再做新轮GPU实验。完成后审计40端点、100评分、48000唯一main+80smoke、U次数、全部40teacher计数/10固定状态、30历史控制复现以及源/入口绑定。发布ENDPOINTS/SUMMARY/TIMESCALE_DECISION/TEACHER_AUDIT/CONTROL_REPLICATION/ALL_COSTS/COMPLETION_AUDIT和实质报告，经GitHub代理推送与SHA/匿名访问验证。患者、checkpoint、病例分数、原日志仅NAS。
