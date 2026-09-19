# 给 Codex：准备 OBS0，不启动实验

请按照同包 `01_SCIENTIFIC_SOLUTION.md`，准备 `AGMS_OBSERVER_V1`。
用户转发本段只授权代码/公开聚合信息读取、独立新包实现、下述有限CPU生成测试、提交推送。
不授权真实模型/患者读取、CUDA、smoke、正式训练、模型重评估或新监测。
实现完成后必须交给外部审阅；获得真实新批准且用户另行确认，才可实验。

## 固定证据

HALF结果：6e65a9b987f8818f7b10fa11a2cc205d11283779
HALF实现：44c1da8a5a8a75021a0298d42bd8f9088e9b059a
原AGMS实现：89dc7657c4f0b7c6f51de339d0f6bc2a621da55f
原AGMS结果：ef6888dc81519ce9e9ca13ea497bbf6397747d70
B2前缀：3034b2199aa7d6e54ea67499f391a0dd4ea3d21e

先读适用AGENTS及真实接口。保护所有历史工作区、代码、结果、账本和审批。
在独立分支/新包 `experiments/lcrseg/agms_observer_v1/` 实施，不改旧AGMS/HALF训练文件或放宽旧权限。

## 唯一科学改变

参照原A5 DS=.25，而非把HALF继续减半。
两个辅助头的输入改为只对辅助分支 `h.detach()`，前向数值不变；主网络原features不得全局detach。
头仍接受 .25 * mean(two heads CE+Dice)，原aux lr=.001、weight decay4e-5、EMA=.99。
保持头随机初始化namespace、aux层/形状、主B2 loss/PAS/KL、原H=.5、R公式/滞后及全部数据几何不变。

主A/B继续接受原L、KL和H梯度。DS对主A/B的梯度必须为None/0，对head仍合法非零；teacher/mask/risk不反传。
禁止恢复BA双端正交、增加R/F、旧数据、历史teacher或跨域风险/原型库。
保持模型主头冻结规则，不把DS=0写成头也不训练。

H关闭的B2等价shadow仅限生成资格，非新增真实实验臂。证明main/EMA/AB optimizer/数据读取/RNG不受观测头影响。
同阶段恢复heads/EMA/R/optimizer/diagnostics；阶段出口只保存合并主网络，aux/R丢弃。
恢复对象必须是新研究trainer；旧study、错prefix或非detach语义checkpoint拒绝。

## 唯一未来真实矩阵

seed163，O1/O2，OBS0仅第二目标阶段：
O1：原B2已学RIM前缀→Drishti2100；
O2：原B2已学Drishti前缀→RIM3200。
两节点5300；source和第一目标新增0。
历史B2、A1 H-only、原A5=.25、HALF=.125各两条导入，正确匹配prefix/timeline/provenance。
不从NKA/A5/HALF终点初始化，不共用别臂warmup或优化状态。
不实现ROUTE_HALF，不搜索其他系数，不生成SCNP、条件降级反馈、更多seed或完整轨迹节点。

## 新诊断

复用已有teacher-L/U及混合前向，不额外跑整网。
全部active U步保存聚合的同状态main-only/equal/risk coarse选择差异、候选数、XOR/Jaccard和拒绝原因。
只对当前L离线计数正确率，不能读取U标签或把U风险值写成实测正确率。
8个固定点同原时点，四项VJP共32；DS梯度None应合法，不能为取得非零而接回主网络。
新增父类与disc内部条件分类的分开诊断；对条件分歧和主头entropy做同覆盖率/同边界区域描述性比较。
真值边界只能用于L诊断分层，不参与U gate；分歧数据/标签不反馈loss，不在线改阈值。
不要将训练L统计称为独立校准或像素级显著性。

可实现KL parent/conditional分解的纯数值核及生成测试/影子日志；生产训练不得使用新的conditional权重。
纯数值核要求w=1恒等恢复原KL，处理极端概率/ignore/空支持；不能额外重复加一份KL。
正式监督仍只为原B2+旧H，唯一干预是辅助特征detach。

## 预算与测试

旧CPU235（134+80+21）保留。新CPU最多32 optimizer调用、2次尝试、每次预声明<=16，失败计账，不准清空或跑旧suite。
先做0-step数学/权限/报告测试，再进行必要的原生生成更新；不为凑次数添加测试。
必须覆盖：head梯度与EMA、主DS路径为0、原主L/KL/H路径仍通、B2等价shadow、原生恢复、错身份拒绝、部署、失败原子提交、报告、诊断不扰动RNG。

未来生产提案：CUDA10=continuation5+baseline/shadow4+预设失败1；smoke每顺序4共8；formal5300，真实总5308。
实现前展开并自校验上述调用数量；不能通过改正式步数适配测试预算。
所有生产调用必须有新批准+独立用户启动确认，当前全部PENDING。

## 交付

完整METHOD/PLAN/EXECUTION_PLAN/前缀与历史导入/环境/风险诊断/恢复合同/调用预算。
只读复算旧结果，检查HALF恢复主要来自source而非first_target；不改写原门槛或阴性结论。
新报告预计10终点/30域行/12主要配对/8梯度点（历史不可用diagnostics为NA）。
恢复检查按新提案：每序Final>=B2-.001，每个旧域macro>=B2-.002，各序Incoming>=B2-.005；不是非劣统计检验。
恢复到基线只表示去除负作用；是否有额外证据价值另作科研审阅。不能自动进入细类降级。

一次性交付受限CLI、验收与报告，不把production adapter留为未实现。
不创建APPROVED，不复用旧review/launch，不启动P0、CUDA、smoke或训练。
push后报告完整SHA、科学diff、测试/累计调用、待执行项并停在：
STOP_AWAITING_EXTERNAL_CODE_REVIEW。
