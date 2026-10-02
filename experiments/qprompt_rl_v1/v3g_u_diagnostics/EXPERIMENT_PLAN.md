# V3G-U-DIAGNOSTICS：现有终点的只读机制诊断

## 证据与问题

V3F已完整交付。LATE−EARLY新域+0.10154个百分点、旧域−2.78164个百分点，0/5种子同时改善，原时间位置假设失败。ORIGINAL/FINE_05同入口控制均精确复现V3E；控制复现排除了该轮实现漂移这一解释。EARLY的旧域均值较高仍只是开发集上的权衡，不能宣布方法胜出。下一步先检查已有模型，不微调窗口或λ追逐分数。

预设问题：相同训练探针上，LATE终点是否比EARLY有更负的监督/U梯度余弦、以及更大的起点教师至终点教师KL？这两项为主要描述量；没有定义性能成功门槛，也不据单个探针或指标选择新参数。全部配对值、域均值、五种子均值与标准差及方向计数都报告。两项模式即使出现也仅提示关联，不构成遗忘的因果证明；不出现也保留。

## 冻结数据和状态

run_id=v3g_u_diagnostics_20261002T005000Z。五种子168/169/170/171/172，两域RIM_ONE_r3与Drishti_GS，五状态ENTRY/ORIGINAL/FINE_05/EARLY_05/LATE_05；每状态八个固定cursor=0/5/10/15/20/25/30/35，共50状态、400批次、800次autograd.grad。批次沿用原生L/U各2个样本的确定性流与几何增强，跨条件同种子同域同cursor匹配；允许自然样本重叠，不当作400个独立患者。源复用V3E五源，完整状态只读V3F执行baafffbac4c56495af85164c0d4265a6473e9996。ENTRY保留step0原生阶段，四终点保留step1200阶段，只临时改探针cursor。主比较仅四终点中LATE−EARLY，ENTRY梯度不能当作同阶段变化。300/600部署快照缺完整EMA/Adam，不重建或冒充完整时间序列。

仅训练L影像/标签和U影像/有效几何，拒绝val/test访问；不读取U隐藏标签。既有val已跨轮用于研发，本轮不是独立患者确认。NativeLRParent/LR_SRC_A及原KI身份未核实的限制不变。

## 测量、状态保护与判读

监督项是现有supervised+constraint，U项是原有FINE三类KL、置信>0.7、按全部valid像素归一化。autograd.grad分别计算两项，报告余弦、负余弦比例、范数和统一0.5倍U/L范数比；零范数导致余弦未定义，保留空值和数量，不改为0。窗口开关不影响诊断：这里测量相同反事实U项，不执行U更新。

同时报告U教师置信/接纳比例、教师学生KL(q_teacher||p_student)、argmax分歧、三类教师像素比例、起点教师至当前教师KL(q_entry||q_current)。起点教师仅缓存在内存。L教师像素准确率、当前批次出现类别等权准确率只是训练标签探针，不是验证Dice或无偏质量估计。所有探针值先按种子/域/状态平均；配对统计种子内两域等权，不把批次当独立种子。次要指标全部展示，不能取代预定主要两项。

每探针恢复student buffers/modes/RNG/telemetry；每状态前后核对student/EMA/Adam/scheduler/scaler/RNG/grad/step/cursor等完整状态一致。禁止optimizer.step、EMA更新、新checkpoint写入；独立操作计数器update_cap=0和零额Ledger双重约束。主副数据访问器拒绝非训练角色。每探针记attempt/success，首次异常停队列，不自动重试。

## 成本与交付

准备起点2026-10-02T00:50:00Z，默认optimizer窗口截止14:50Z但optimizer预算严格0；本只读队列另取1小时总硬上限01:50Z，含准备，不延期。source/main/smoke/development/controller/replay均0，400探针上限，恢复reserve0；CPU数学检查不计入GPU探针但记录检查结果。V3D/E/F累计新增优化器调用153060，既有源历史40000单列，本轮不增加。预计分钟级、GPU4显存余量充足，复用NAS历史状态，仅新建代码、账本及匿名诊断表；不用新的依赖。

冻结提交后才运行任何新诊断。NAS挂载/空间/读写探针、with_nas_storage.sh和既有guardian中性argv启动；短检查后交还每小时监测。终结核验400唯一成功探针、800 VJP、50状态不变、0 optimizer/EMA/val/test。记录成本、失败账本及最终判读；发布安全源码、匿名DIAGNOSTICS.csv/CELL_SUMMARY.csv/SUMMARY.json/COMPLETION_AUDIT.json/ALL_COSTS.json和报告。原状态、患者数据、原始日志仅NAS；GitHub代理推送并验证远端和公开访问。诊断完成后依据完整证据再冻结下一项有实质区分力的实验，不能自动把某一诊断模式映射成调参规则。
