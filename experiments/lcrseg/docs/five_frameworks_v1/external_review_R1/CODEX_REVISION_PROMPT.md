# Codex｜五框架 V1 外部审阅 R1 修订任务

你在仓库 DLwbm123/SSL_CL_seg 工作。固定审阅基线：
- branch: codex/sslcl-five-frameworks-v1-review
- reviewed commit: b909a9131dc7137232ac8eb3a6922fe66520b932
- original base: 8a6e93a8bc39554c878cffe0dedf7fbfefe7d16d

请读取附件 REVIEW_REPORT.md、TARGETED_REPRO_RESULTS.json、test_external_review_R1.py，再检查当前工作区是否包含固定提交。若有后续用户改动，不覆盖；报告实际 base 并在其上最小化修复。独立分支保留，可追加提交；禁止 amend/force-push 抹掉原审阅证据，禁止合并 main。

## 当前权限与目的

当前仍为 CODE_ONLY。外部审阅结论是 CHANGES_REQUESTED，不是实验审批。本轮只修订代码、补 CPU 合成测试、完成可解析的元数据/桥接代码、更新审阅材料并推送，然后停止。

不得读取真实图像/标签/checkpoint tensor，不得占 GPU、做真实 smoke、训练源模型、开始 B/C/D/E。不得将测试通过、模板 approval 或本文当作用户启动确认。合成测试张量允许，本轮实际测试/优化/VJP成本如实记录。

保留五个框架、各8配置开发、全部进入三种子复核，以及原计划212轨迹/424目标阶段的上界。仅在真实父参数解析造成配置alias时按原规则去重。不得因为这次审阅删除某个框架或增加科研“全种子/全类别必须改善”门槛。

## 必修 R01：恢复构造与新阶段初始化分离

目前 deploy.parent 冻结全部参数；initialize=False 跳过 stage_entry，load_state_dict 不恢复 requires_grad。请增加安全的 resume 构造接口，根据已绑定阶段的合法参数白名单恢复 requires_grad 与模块模式，正确构建优化器后加载其状态。

不得重跑 F2/F4 校准/方向探针；不得重置已学 A/B、R、EMA、原型或数据位置；不得在加载非零B后盲目调用会拒绝该状态的 stage_entry；不得全模型 requires_grad_(True)。

增加真实调用链的第二阶段中途、独立进程恢复测试（五框架与基线），验证 flags、参数组、学生、EMA、optimizer/scheduler、原型、F/Q/R、探针状态、随机流和数步轨迹。第一阶段恢复对照仍须通过。

## 必修 R02：恢复语义绑定

保存完整 resolved semantic config 及摘要，并绑定实际 provider seed/order/stage、manifest/split、随机流定义和游标、父/阶段来源身份。恢复前从实际对象推导并对照，不只比较调用者传入 identity。

lambda_U/JML/structure/shape/SWD、total_steps/warmup、family/rank、seed/order/stage/manifest 改变须拒绝，不能继续一条不同的轨迹。明确非语义变化白名单；校验在任何 trainer 状态变更前完成。不要把代码/hash/数据检查写成不可执行的文书要求，只绑定真正决定运行的内容。

## 必修 R03：数值验证覆盖 loss 与提交状态

现状只验证梯度有限。增加活跃 loss/组合值有限检查；optimizer/父约束之后、EMA/原型/科学游标提交之前，检查有效模型权重、必要optimizer状态、feature有效矩阵与候选EMA状态。

loss不必下降；合法空集合graph-connected零允许。AMP skip 不推进科学轨迹。故障已执行optimizer时保留physical耗费，标为未提交，依旧checkpoint恢复；不要悄悄跳过、清除账本、污染EMA后继续。

加入正常可微损失+inf常数（梯度仍有限）、非有限更新/constraint、AMP skip/fault-tail 的回归。

## 必修 R04：F2 fallback 的异常类型

只允许无支持/零探针、数值秩不足等已规定情况保留原合法A；错误维度、非有限G、P非对称/非幂等、无效参数和求解器异常必须传播。

不要 broad except ValueError 把它们统称 fallback。建议专用 InsufficientProbeSupport/RankDeficientProbe 等类型；不要根据错误字符串做脆弱分支。测试合法秩不足仍不触发性能淘汰，P=2I/错维/非有限探针则不能设置complete=True。测试替身只用于异常路径，CWMI作者后端资格测试仍保留。

## 修复 R05：显式桥接模式

EMA features 不应继续传 mode='student'。给Model.parts/forward和所有teacher/student/eval调用显式、一致的语义；同时检查训练模式与requires_grad，不把eval()当作mode字符串的替代。

增加teacher/student行为确有区别的合成bridge，验证原型、原U目标、F4修正、学生混合与部署全部走正确路径。不得借此改真实随机分类头或强制替换为确定性线性头。

## 修复 R06：评价的 ignore 支持

segmentation_metrics 当前把255处预测算入假阳性。按native评价协议显式处理valid mask、全ignore/双空/单空、形状与类别定义。

要求有效区域完全相同、仅255位置预测不同，不影响Dice。边界/拓扑与ignore的交互需明确：不把ignore边缘制造为真实解剖边缘。也可明确拒绝带ignore的辅助形态指标并交native evaluator处理，但必须同步调用契约和测试；不可静默算错。历史已发布结果不重算、不改终态。

## 方法定义澄清：F2探针不是本轮新算法修改

外部审阅已撤回“structure-only必为实现错误”的临时判断。框架正文和早期规格存在歧义；本提交契约明确structure-only，R1允许保留该定义。

统一活跃方法公式/注释，写明 G_i=grad(lambda_structure*scale*CWMI_struct)，不是CE+Dice组合梯度；真实训练L仍为CE+Dice+结构项。不要为了应付审阅加一组combined-gradient新方法或增加搜索预算；保留原delivery文件作为历史输入，修订放活跃文档并注明解释。

## 小型一致性修复

Model每阶段目前两次channel_basis/eigh，planner声称一次。选择一次完整谱分解后切片复用，或如实记录两次；更新相应测试、计数，不作为科研 gate。

本轮不把合法近重特征值、低秩退化、某方向没有提升当作错误。未绑定真实父方法时不能声称低秩toy已经验证原KI；也不能让PARENT_BINDING_REQUIRED阻止上述模块修复。

## 已披露真实集成待办的处理

继续用明确接口实现能在CODE_ONLY内完成的真实ParentBridge/数据adapter/有限DAG启动/隔离评价代码；元数据能绑定的则绑定。

若原始父入口仍不可定位，只报告一个最小缺项，不重新遍历所有历史、不请求历史SOTA表。不得自行选F_CONV、源权重SVD或新O-LoRA替代。

CWMI目前显式拒绝CUDA：可以在本轮实现设备本地化代码并做CPU结构测试；不得只删除拒绝，不得静默CPU/GPU往返。CUDA执行资格仍NOT_RUN，等待之后明确B阶段授权。

真正run入口以后必须核验外部审批、用户启动确认、实际代码/父/依赖/计划与预算绑定后才能打开真实payload；当前保持fail-closed，不能靠手工改lock开跑。生产分支未完备时明确标记，不伪装完成。

## 测试要求

附带9个回归期望在旧提交是1通过/8失败，环境为reviewer's torch2.10.0+cpu，并非原104套件。请先复现相关缺陷，再修复并将必要测试正式并入仓库。

可适应经审阅的新API/专用异常名称，但不能删除验收语义、放宽核心断言、改成xfail/skip或伪造报告。修复后复跑原CPU套件和新增回归；后端缺失应报告而不是返回零loss。精确CPU源码、依赖哈希与本次成本单独记录。

## 提交与回传

更新 review/REVIEW_RESPONSE_R1.md，对R01–R06逐项记录：修复位置、测试、剩余限制。同步实现偏离、梯度契约、生命周期、代码清单、测试覆盖、计划/预算摘要和CODE_ONLY lock。旧104通过报告保留可追踪版本，不声称新测试在旧提交也通过。

推送同一审阅分支或明确的后继分支，不合并main；核对远端真实SHA，随后停止：

status = STOP_AWAITING_EXTERNAL_CODE_REVIEW
review_round = R2
reviewed_previous_commit = b909a9131dc7137232ac8eb3a6922fe66520b932
actual_base = ...
new_review_commit = ...
review_response_path = ...
issues = R01...R06 的 RESOLVED/OPEN 及依据
CPU_tests = 实测数量
CUDA = NOT_RUN（除非另有明确的新授权）
parent_binding = 实际状态
real_integration = 实际状态
real_image_label_checkpoint_reads = 0
real_smoke = 0
formal_updates = 0
push_verified = 实测结果

不要自填外部APPROVED，也不要启动真实实验。
