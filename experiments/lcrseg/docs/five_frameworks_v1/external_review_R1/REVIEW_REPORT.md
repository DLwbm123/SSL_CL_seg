# SSLCL Five Frameworks V1｜外部代码审阅 R1

## 审阅结论

- 审阅提交：`b909a9131dc7137232ac8eb3a6922fe66520b932`
- Base：`8a6e93a8bc39554c878cffe0dedf7fbfefe7d16d`
- 分支：`codex/sslcl-five-frameworks-v1-review`
- 决定：**CHANGES_REQUESTED（代码需修订）**。
- 真实实验授权：**NOT_GRANTED**。本文不是 `APPROVED_FOR_EXPERIMENTS` 回执。
- 五框架的研究方向及全部进入比较的计划：保留。
- 发现：4 项 P1 正确性问题；2 项 P2 接口/评价问题；另有已公开的真实集成待办。

P1 指会改变训练定义、破坏恢复等价或把无效更新计为成功的问题；不是算法效果判定。P2 指当前 toy 模型没有暴露、接入真实模型/评价时需要解决的具体接口问题。修复不增加“所有种子/类别必须为正”的性能 gate。

## 一、范围与证据强度

读取了固定提交的审阅入口、偏离说明、代码指纹、主要数学/模型/损失/训练/恢复/调度/统计/评价实现及对应测试。GitHub compare 返回 78 个 added 文件、1 个新提交，没有修改或删除旧文件。未宣称逐字审阅全部历史仓库。

仓库记录原 CPU 套件 `104 passed`，CUDA 未运行。**本次没有完整重跑这 104 项，也没有重新执行 CWMI/JDT/D-Convexity 作者源码对照套件。**

本地复现使用从连接器读取并重建的 9 个源码文件；每个 SHA256 都与固定提交的 `CODE_MANIFEST.json` 一致。环境是 `torch 2.10.0+cpu`，不是原报告的 torch 2.6.0。运行了新增的定向合成反例与 9 项回归期望：旧版本 **1 passed / 8 failed**。这些 8 个失败是新增测试暴露旧缺陷，不是声称原 104 项测试有 8 项失败。

未读真实图像、真实标签或真实 checkpoint；生成/读取的张量均为此次合成测试。未运行 CUDA、真实 smoke、正式训练，未推送或修改远端仓库。

随附：`TARGETED_REPRO_RESULTS.json`、`REGRESSION_BASELINE_LOG.txt`、`test_external_review_R1.py`。

## 二、必须修复的问题

### R01｜P1：第二阶段恢复时 A/B 可训练状态丢失

位置：`model.py:61–68`；`train_stage.py:40–55`；`checkpoint.py:40–52`。

`deploy()` 返回全体 requires_grad=False 的部署模型。正常新阶段进入时，`stage_entry()` 会重新启用 A/B；但 `StageTrainer(..., initialize=False)` 将这个调用与昂贵的探针初始化一起跳过。随后 `load_state_dict()` 只恢复张量数值，不恢复 requires_grad 或各模块的训练模式。

复现：从同一第一阶段封存学生创建两条第二阶段执行，一条正常初始化，另一条通过 initialize=False 恢复第二阶段 checkpoint。连续执行的 6 个 A/B 张量可训练，恢复后的 6 个全被冻结；再执行 1 步，最大模型状态差 `0.01001700572669506`。第一阶段从新模型恢复的对照仍是误差 0，所以不是“所有恢复都失败”。

修复：提供独立的恢复构造路径。在不重新初始化 A/B、不重复探针、不重置原型的前提下，根据经验证的阶段参数白名单恢复训练标志/模块模式并构建优化器，再加载其状态。不要在加载非零 B 后简单调用当前 stage_entry；该函数会拒绝未封存增量。也不要对整个网络无差别 requires_grad_(True)。

验收：五框架、五基线分别测试第二阶段中途 checkpoint 的独立进程恢复；比较参数 flags、参数组、学生/EMA、优化器、阶段游标、探针状态及后续数步轨迹。F2 恢复不得重复执行方向探针。

### R02｜P1：恢复没有绑定生效配置和 provider 身份

位置：`checkpoint.py:27–52`；`train_stage.py:31,112–173,176–178`。

保存/校验的 identity 并未保证构造出的 trainer 与其一致。checkpoint 不保存/核验完整 options，也不核验 provider 的 seed/order/stage/random-stream 身份。比较调用者传入的 identity 与 checkpoint identity 不足以保证本次恢复真实使用的是同一配置。

复现 A：相同 identity 下将 lambda_U 从 0.5 改为 0.0，restore 不报错；下一步最大模型状态差 `0.0037487302906811237`。

复现 B：提供第一阶段 identity，却使用 provider.stage=2，restore 仍成功；之后读取的是不同的 stateless 随机流。

修复：将实际解析后的语义配置、父/阶段来源、provider/manifest/随机流/数据游标定义绑定到 checkpoint；与本次实际对象比较，在改变其状态前拒绝不匹配。允许的非语义变更（输出路径、经声明的设备迁移）单独列白名单。校验不能只相信调用方传入的一串 ID。

验收：lambda_U/JML/structure/shape/SWD、warmup/total_steps、family/rank、seed/order/stage/manifest 任一实质改变须拒绝；完全相同的配置可以恢复。

### R03｜P1：只检查梯度有限，未检查 loss 和更新后的状态

位置：`train_stage.py:180–206`。

update 在求导后只检查 `.grad`，直接执行 optimizer、父约束、EMA、原型提交和计数。有限梯度不蕴含有限目标值；optimizer/约束也可能在此后产生非有限状态。

复现 A：合成父监督函数返回正常可微损失加 `+inf` 常数。梯度仍有限，更新被计为成功，step=1，记录 labeled_loss=inf。

复现 B：合成父约束在 optimizer 后写入一个 inf。当前程序把非有限学生传播进 EMA，仍将 step/cursor 推进到 1。这是故障注入，不是断言真实父约束必然产生 inf。

修复：梯度前检查各活跃损失及组合值；optimizer/约束后、EMA/原型/科学游标提交前检查有效参数、必要优化器状态、有效特征变换及候选 EMA 状态。零支持有限损失继续允许。不要把数值检查变成损失必须下降的 gate。

已经执行的 optimizer 算作 physical cost；状态验证失败则标记未提交并按恢复协议处理，不能隐藏消耗，也不能假装没有执行。不能污染 EMA 后再继续训练。

验收：有限梯度+inf loss、非有限 optimizer/constraint 结果、AMP skip 与 fault-tail 的交叉测试。无效步骤不推进科学计数/游标/EMA/原型。

### R04｜P1：F2 的宽泛 ValueError 捕获把工程错误当成合法退化

位置：`train_stage.py:100–106`；`kernels.py:215–240`。

方向初始化仅应对无支持/零探针或数值秩不足做明确 fallback。目前捕获所有 ValueError；而 `gradient_seed_basis()` 对非有限探针、错误维度、非对称/非幂等投影同样抛 ValueError。

复现：给所选 toy adapter 一个 P_free=2I（不是正交投影），F2 不报错退出，而是记录两个 `not an orthogonal projector` fallback，并将 probe.complete=True。

该反例使用明确标注的 ToyStructure 测试替身，仅测试异常分类；不是用它替代 CWMI 做算法验证。

修复：使用细分异常类型或显式返回状态，仅对许可的无支持/秩不足进行 fallback；错误形状、非法投影、非有限值、求解器异常应传播为工程错误，不能转成成功的探针完成记录。保留合法退化继续训练的政策。

验收：合法秩不足可回退；P=2I、错维 P、非有限 G、不对称 P 均应失败，且不得改动 A/提交探针完成状态。

## 三、接入前修复的 P2 项

### R05｜Teacher 的桥接 mode 被写死为 student

位置：`model.py:36–45`；`reliability.py:12–18`；`train_stage.py:134–138`。

`teacher().eval()` 不会改变显式传给桥接层的字符串。EMA 的原型/U特征调用仍走 `parent.features(...,'student')`，而 U 的最终读出用 `'teacher'`。在 teacher/student 分支行为不同的真实随机分类头/骨干上，可能产生混合语义。

本地记录型 toy bridge 确认教师原型初始化收到 `student`。真实父方法是否因此产生预测差异尚未验证，故不虚称已经发生真实错误。

修复：显式传递 feature/readout 的 student/teacher/eval 模式；独立检查 nn.Module.training、requires_grad 与 mode，三者不能互相替代。使用行为确实不同的 conformance fixture，而不只用忽略 mode 的 toy 网络。

### R06｜评价 helper 没有处理 ignore=255

位置：`evaluate.py:32–35`。

`segmentation_metrics()` 直接比较 pred==class 和 target==class，忽略像素在目标中相当于非前景，但同位置前景预测仍算假阳性。

复现：target=[[1,255],[0,0]]，两张预测只在255位置不同，其余都正确，rim Dice 却从1.0变为0.6666666667。

修复：按父评价协议显式传递 valid mask；完全忽略图像标记无有效支持，不能当双空背景给满分。边界/拓扑指标不能把忽略边界伪造为真实器官边界；原协议无法定义时记录该指标不可评估，或明确本 helper 只支持全有效图像并拒绝含255输入，交真实 native evaluator处理。是否采用后者需要同步测试/调用契约，不要隐式忽略问题。

这不证明现有 Fundus 历史分数错误；真实 evaluator 仍未接入，且历史报告可能无忽略像素。该问题是当前新 helper 的适用范围错误。

## 四、不作为实现错误或性能 gate 的事项

### C01：F2 “结构单项探针”是定义澄清，不强行改成新变体

此前框架正文写 G=∇(CE+Dice+结构项)，后续交付规格写“以该结构损失”获取探针；措辞有歧义。本次 `LOSS_AND_GRADIENT_CONTRACT.md` 与代码一致明确采用 lambda_structure*scale*CWMI-only。

因此撤回审阅过程中的“已确定实现错误”判断。R1 接受当前结构单项探针作为此版本的明确语义；请只统一本轮活跃公式/文档，并保留历史原始交付，不增加组合目标搜索臂。正标量缩放不会改变精确 SVD 的方向，这一点本次契约已披露；lambda_structure 仍影响正式 L 损失，因此不能据此把两个训练配置判成重复。

### C02：Q 的一次/阶段成本口径

Model 初始化调用两次 channel_basis（截断谱、全谱），而成本计划写每阶段一次 eigensolve。可以一次求完整特征分解后切片，或者按实际两次计数。这是小型成本一致性修复，不是因性能否定方法的理由。

### C03：保留已正确的设计

保留 F_prev(I+QRQᵀ) 的坐标和合并顺序；F1/F3/F4/F5 的 R-only U 权限；F2 的 B-only 权限；F3 自定义反向而非伪标量 loss；F4 只修原来源目标并保留原 PAS 接受集合；F5 的额外 U 特征前向和 SWD；原型仅当前阶段；E 消融只注册不执行。现有 author parity 测试有价值，但仍需在正式运行环境资格验证，不把它们当成真实父方法集成证明。

## 五、真实实验仍有的公开待办

这些在用户和提交中已经明确披露，不当作隐藏缺陷：

1. 原始父实现、真实 optimizer/约束、几何/随机读出和阶段转移尚未绑定；不自行换成 F_CONV/SVD/O-LoRA。
2. 生产数据桥接、隔离 evaluator、有限 DAG 实际派发、源模型谱系、源/目标预算仍待实现/解析。
3. CWMI 当前显式拒绝 CUDA；需要实现正式设备路径并按协议做 CUDA 资格。不能只删除拒绝语句或每步偷偷CPU往返。
4. `cli run` 当前总是抛 CODE_ONLY；外部审批 helper 尚未接入真实启动。不能“改 lock 就开跑”。

先修 R01–R06，推进能在 CODE_ONLY 内完成的真实桥接/设备代码与元数据绑定，再推送新提交。原父入口仍缺时，模块修复照常完成，不重新无限遍历历史档案。

**未授权 GPU、真实 smoke、数据读取或 B–D 实验。** 后续审批可以一次绑定 B→C→D 的有限执行计划，B 的数值/隔离资格完成后继续；不必为每个性能波动新增人工 gate。

## 六、固定来源

以下均来自同一不可变提交：

- https://github.com/DLwbm123/SSL_CL_seg/blob/b909a9131dc7137232ac8eb3a6922fe66520b932/experiments/lcrseg/docs/five_frameworks_v1/review/REVIEW_INDEX.md
- https://github.com/DLwbm123/SSL_CL_seg/blob/b909a9131dc7137232ac8eb3a6922fe66520b932/experiments/lcrseg/five_frameworks_v1/train_stage.py
- https://github.com/DLwbm123/SSL_CL_seg/blob/b909a9131dc7137232ac8eb3a6922fe66520b932/experiments/lcrseg/five_frameworks_v1/checkpoint.py
- https://github.com/DLwbm123/SSL_CL_seg/blob/b909a9131dc7137232ac8eb3a6922fe66520b932/experiments/lcrseg/five_frameworks_v1/model.py
- https://github.com/DLwbm123/SSL_CL_seg/blob/b909a9131dc7137232ac8eb3a6922fe66520b932/experiments/lcrseg/five_frameworks_v1/kernels.py
- https://github.com/DLwbm123/SSL_CL_seg/blob/b909a9131dc7137232ac8eb3a6922fe66520b932/experiments/lcrseg/five_frameworks_v1/evaluate.py
- https://github.com/DLwbm123/SSL_CL_seg/blob/b909a9131dc7137232ac8eb3a6922fe66520b932/experiments/lcrseg/docs/five_frameworks_v1/review/LOSS_AND_GRADIENT_CONTRACT.md
- https://github.com/DLwbm123/SSL_CL_seg/blob/b909a9131dc7137232ac8eb3a6922fe66520b932/experiments/lcrseg/docs/five_frameworks_v1/review/IMPLEMENTATION_DEVIATIONS.md
- https://github.com/DLwbm123/SSL_CL_seg/blob/b909a9131dc7137232ac8eb3a6922fe66520b932/experiments/lcrseg/tests/five_frameworks_v1/test_integration.py

完整反例数值、环境和9个源码哈希见同目录 JSON。没有新增任何真实模型性能数据。
