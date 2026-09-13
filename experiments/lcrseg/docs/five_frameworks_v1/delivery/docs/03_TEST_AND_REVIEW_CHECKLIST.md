# 03｜合成验收、真实资格与外部审阅

## 1. 审阅前允许与禁止

允许源码/配置/manifest元数据读取、文献与作者源码核对、CPU合成数据、无模型训练的计划展开、本轮独立分支push。默认不允许真实图像/标签/真实模型tensor读取；不允许真实smoke/GPU训练/test或hidden GT。`checkpoint exists/stat`元数据不等于读取tensor，可以计为metadata。

审阅前可以将所有五个框架编码完成并以toy ParentBridge验证；不得把toy结果声称真实父集成成功。父绑定缺失不阻止其余代码完成并push；真实跑分接口必须仍然拒绝。

## 2. 所有方法共享的测试

### 2.1 来源、损失、教师

- 互补mask与来源恢复：取回原位置，非均值；每个L锚像素只计分一次。
- U分支的L donor没有额外GT损失；无U基线0 U payload访问。
- geometry/255 ignore统一，KL空支持可微0，JML双空类0、匹配软标签0、硬标签与Jaccard对应。
- 相同mask/T/归约下KL和soft CE的学生梯度相等；严禁把此作为不同科学候选。
- 教师q、prototype、support、可靠性、F3 D没有梯度。
- PAS missing class显式fallback，teacher-only无学生confidence门槛；strict >语义一致。
- 同一feature坐标中做原型；L背景、rim、cup支持分别记录，不用伪标签更新current-L原型。
- teacher模型行为（eval、BN/GN、随机头）与父契约一致；同source重复前向语义可复现。

### 2.2 参数与投影

- ParentBridge关闭全部新模块时还原原父行为；启用LCTX是新recipe，不能拿native关闭测试冒充二者同配方。
- A/B依据实际乘法顺序命名；软/硬正交不混用。
- Q所在维度与feature节点匹配；C_eff=C F_prev的矩阵次序测试。
- 偏置、3×3 padding0或其他native几何、output interpolate/align_corners完全保留。
- 不对完整读出梯度做恒等投影冒充新方法；对照验证完整row-space投影可能无变化。
- rank=1、rank接近d、重复特征值、零谱/极小谱正确报告；重根比投影矩阵而不是基向量逐值。
- 新G初始化I；F_next=F_prev G而不是G F_prev；阶段合并在完整模型上等价。
- 累计F仅一份d²参数，不逐任务堆栈；存储旧checkpoint仅供审计/eval，不是训练可访问记忆。
- frozen父参数、F_prev、readout无非法梯度；当前R有非零U梯度；L对父与R都有梯度。
- U梯度隔离与总optimizer后参数变化分开测试（Adam动量/weight decay会产生变化）。
- 参照dtype FP64代数、FP32实际误差预算分开；不以微小update为分母无限放大舍入误差。

### 2.3 恢复与计数

- 连续N步 vs K步checkpoint+重启+(N−K)步：所有训练态/EMA/优化器/原型/采样/RNG一致或预定数值容差内一致。
- 在正式step之前/之后、EMA之前/之后、checkpoint写入时的故障注入有明确恢复语义。
- 不将日志中已执行但无checkpoint状态的tail当成可跳过；重跑计physical成本。
- AMP skip不算科学optimizer step；EMA/原型提交/data游标与实际成功语义一致。
- 第二目标域从自己第一目标域checkpoint开始；拒绝交叉方法、种子或顺序的checkpoint。
- 一个训练阶段结束时student封存，evaluator只加载那个student，不使用EMA/最佳epoch。
- independent-process deploy无需teacher、内层repair、当前prototype和Q缓存。
- 多线程/DAG任务无跨run全局prototype/monkeypatch污染；随机流按namespace隔离。

## 3. 每框架必须有的专门测试

### F1

JML在0/1与soft targets、全背景、无视杯、empty mask、teacher requires_grad情况下正确。F1与loss-only消融只差预定梯度权限；不是一边改mask一边改loss。非零R仍保持U只更新R。任意hard feature投影与累积F不能造成模型输出维度/几何变化。

### F2

CWMI结构项与外部锁定后端在合法toy输入上数值和梯度一致；没有重复CE。常量预测、空前景、all-ignore、all-valid crop边界测试。所有依赖文件与kwargs映射写出。

current-L探针不更新model/optimizer/EMA，zero-init B保证重设当前A后入口函数不变。禁止重置旧adapter。stacked-G thin-SVD与小维GᵀG eigenspace一致；P_free右乘与shape正确；秩不足fallback被记录而不是称PASS。不同lambda_structure真的改变所用probe objective，scale校准不读取val。

### F3

identity-forward exact equality；custom backward与手算D*g一致；L分支不被过滤；同一个U backward不重复乘D。κ=0退回同覆盖的普通KL；u=0全强度；missing proto用熵；s小不NaN。

关键：禁止对custom-gradient identity做常规gradcheck并要求有限差分通过，这个规则有意不是identity的真导数。应测试显式规定的梯度。D必须来自映射后的该mixed-view feature位置，mask方向反了的单测必须失败。

### F4

调用真正D-Convexity后端的loss-only测试，不把toy penalty替代正式验收。disc=p1+p2/cup=p2；rim环形样例不按凸目标约束。原q所定义PAS集合不因target repair改变。

λshape=0返回q；steps=0不修正。d-only求导，teacher/student梯度无污染；同q输入固定steps/seed有确定输出；inner d detach每步不积二阶图。readout权重/偏置/卷积/resize实际一致。低分辨率d必须通过固定插值进入feature，而不是直接把输出概率当feature。

修正范数在信任半径；finite objective/no NaN；不要求每一步shape和真实Dice都改善——这不是工程正确性的定义。返回target正、和为1且detach；元数据正确记录真实full/readout-only/VJP计数。

### F5

同集合或其排列SWD=0；equal-cardinality的一维排序W2与手工例子一致；学生有梯度、teacher/directions无梯度。missing class/少样本明确从valid-class denominator排除；不能填背景凑样本。

L GT与U pseudo类别未混淆；采样随机流不消耗augment stream；当前batch数据不进入永久queue。规范化/投影/teacher feature坐标一致，extra clean-U前向确实存在且计成本。关闭SWD回到相同输入同梯度路径的KL对照。

## 4. 审阅前不做真实资格；审批后B阶段做什么

B在确认代码commit/计划hash/parent绑定/依赖lock后才访问真实数据。允许一次至多24个真实current-L smoke成功optimizer更新，覆盖关键父桥接与部署；具体各framework分配在审阅前固定，全部discard。source真实checkpoint读权限只在B以后启用。

synthetic CUDA资格与实际formal dtype/optimizer组合一致；优先验证每个框架至少1个实际call graph而非只测数学子函数。真U路径的测试先用合成U，禁止为了方便读取hiddenU标签。

资格CPU/GPU正向、反向、恢复、显存profile，给计划生成实际显存需求/并发准入阈值。共享GPU只启动本研究进程，不terminate别人的任务。不会从已有低利用率截图推断当前空闲。

性能指标不进入B准入规则。若B发现纯工程bug，修复并报告commit差异；涉及loss/更新/几何/随机语义需重新外部审阅，不能边跑边悄悄改科学定义。

## 5. Code review包

Codex必须commit以下公开小文件：

```text
review/REVIEW_INDEX.md
review/PARENT_BINDING.json             # sanitized; private exact paths retained elsewhere
review/ARCHITECTURE_AND_COORDINATES.md
review/IMPLEMENTATION_DEVIATIONS.md
review/FRAMEWORK_COMPLETENESS.csv
review/TRAINABLE_PARAMETER_MANIFEST.json
review/STATE_LIFECYCLE.md
review/LOSS_AND_GRADIENT_CONTRACT.md
review/DATA_ACCESS_CONTRACT.md
review/DEPENDENCY_LOCK.json
review/SOURCE_MAP.md
review/TEST_REPORT.json
review/TEST_LOG_SUMMARY.txt
review/RESOLVED_PROTOCOL.json
review/SEARCH_CANDIDATES.json
review/BASELINE_CANDIDATES.json
review/TASK_MATRIX_PREVIEW.csv
review/BUDGET_PLAN.json
review/REVIEW_LOCK.json                 # remains false / CODE_ONLY
review/CODE_MANIFEST.json
```

FRAMEWORK_COMPLETENESS逐项列backend、parent integration、forward、backward、merge、resume、synthetic evidence，不允许用一个总PASS掩盖CWMI/DConv仍stub。测试不可运行明确NOT_RUN原因；不要把skip当pass。

CODE_MANIFEST包含source-relative path、sha256、变更原因。reviewed_code_tree hash仅覆盖执行代码/测试/依赖科学配置，不包含将来approval/report，避免审批文件修改造成循环hash。review_commit与训练execution_commit可相同；报告追加commit可以不同，但执行代码指纹必须匹配。不得把包含私有token的remote URL提交。

## 6. 审阅流程与判定

审阅先确认：五框架是否是文献所支持的目标+明确本研究变体；梯度路径是否真的不一样；readout Q是否有意义；父KI是否未被替换；数据与时间谱系是否正确；hyperparameter比较是否公平；运行锁是否有效。

审阅判定可以是：
- `APPROVED_FOR_EXPERIMENTS`：绑定具体code commit/tree、parent、plan、dependency与B–D预算。
- `CHANGES_REQUESTED`：代码或科学定义存在实质问题，列修复位置。
- `PARENT_BINDING_REQUIRED`：模块可审阅，但真实父identity尚未给足。

严格要求工程真实性，不要求“理论上保证比基线好”。不因为没有所有非关键文档或某个历史别名，就无限阻塞有用工作；但源模型、数据泄漏、损失是否生效等关键问题必须解决。
