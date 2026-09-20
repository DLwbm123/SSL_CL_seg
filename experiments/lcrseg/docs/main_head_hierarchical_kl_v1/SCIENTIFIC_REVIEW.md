# AGMS_OBSERVER_V1：科研审阅与下一轮实验提案

**状态：PROPOSAL_ONLY / NOT_EXECUTION_AUTHORIZATION**

审阅结果提交：`f7d28f4b855a3ab04d2003bcde2f5b2b9a960c3c`。原执行提交：`f4a1088efe7ea570de0c64d67a1a9d7051ddaa83`。
报告记载完成时间：2026-09-19 22:52:21 Asia/Shanghai，即纽约时间同日 10:52:21。`results_20260920` 是归档目录名。

本审阅依据公开报告、终点 CSV、相关诊断片段及实现代码；没有读取私有模型权重、患者数据或重跑训练。工程通过状态引用已发布验收记录，不等于本审阅独立重做模型验收。

## 一、科研裁决

1. OBS0 的描述性恢复门槛通过；不是跨种子的统计非劣证明。
2. 相对 A5 的收益主要是在撤销负干预；尚未建立超过 B2 / A1 的新增收益。
3. 本实现的多尺度条件分歧尚未表现出值得保留辅助头的独立信息价值。关闭其默认生产接入，不自动进入旧方案中的“观测器风险定位条件降权”矩阵。
4. 保留用户既定的 history-free、已训练参数作为 memory、A 侧隔离/保护主线。不重启 BA 双侧正交，不引入旧数据回放、旧预测库、跨域特征缓存或推理时辅助头。
5. 下一轮只提出一个不同的新假设：在 B2 已接受的 fine-PAS 像素内，保留可靠的 disc/background 软监督，仅降低不确定 rim/cup 条件监督，是否产生独立收益。主头条件熵只是预先固定的候选分数，并非已验证有效的模块。

## 二、结果重算

所有 Final / Old / Incoming 都沿用冻结的 rim/cup macro Dice；disc_union 单独报告，不替代主指标。

| 配方 | 平均 Final | 平均 Old | 平均 Incoming | Final 相对 B2，百分点 |
|---|---:|---:|---:|---:|
| B2 | 0.657465897 | 0.626901857 | 0.718593977 | 0 |
| A1：H-only | 0.657486208 | 0.626884929 | 0.718688765 | +0.002031 |
| 原 A5 | 0.639210407 | 0.599443930 | 0.718743360 | -1.825549 |
| HALF | 0.640116635 | 0.603546471 | 0.713256963 | -1.734926 |
| OBS0 | 0.657407804 | 0.626753003 | 0.718717407 | -0.005809 |

OBS0-A5：Final +1.8197398 点，Old +2.7309073 点，Incoming -0.0025954 点。
Final=(2*Old+Incoming)/3，因此平均 Final 回升几乎全部来自旧域恢复。

| OBS0-A5，百分点 | Final | Old | Incoming |
|---|---:|---:|---:|
| O1：REFUGE→RIM→Drishti | +2.911671 | +5.022328 | -1.309642 |
| O2：REFUGE→Drishti→RIM | +0.727808 | +0.439487 | +1.304451 |

Incoming 的近零均值是两种顺序相反变化的抵消，不是逐顺序等效。
OBS0 相对 B2 的首目标域宏 Dice：O1 -0.047659 点，O2 -0.001924 点，均在原描述性恢复门槛内。

O2 中 OBS0 相对 A5：已学 Drishti 的 cup +8.062041 点、rim -3.287596 点，宏 Dice +2.387222 点，而 disc_union -2.972661 点。这再次说明 parent 指标与 rim/cup 主指标并不等价。

旧 AGMS 消融中，M-only 的平均 Final 已比 B2 低约 1.86 点，H-only 接近 B2；HALF 未恢复，而 detach 恢复。这条证据链支持“本实现中 DS 上游路径参与了退化”，但不能推广成所有深监督都损害持续学习。

## 三、观测器证据的边界

### 3.1 梯度路径

八个固定诊断点显示 DS 对主 A/B、辅助头上游 A/B 梯度为零，辅助头 DS 梯度非零。非零梯度证明头在学习，不证明头提供了可靠的新信息。真实 OBS0 保留 H，因而不是整条训练轨迹与 B2 的严格等价证明。

### 3.2 风险权重确实改变了选择，但不等于有效

同一 teacher 状态下，equal 与 risk 的 XOR 占主头候选集的比例为 O1 1.7299%、O2 0.4246%。不能把此前训练 L 上的零分歧外推成 U 上永远不改变 mask。

这些分母是“主头候选集”，不是全部合法像素。O1 当前导出显示 risk coarse 只占全部合法像素约 2.0163%，equal-risk XOR 只占约 0.04246%。选择变化、错误识别能力和最终收益是三件不同的事。

### 3.3 条件分歧没有通过信息价值筛选

| 顺序 | 两种排序各选像素 | 条件分歧选中错误 | 主头熵选中错误 | 条件分歧选区错误比例 | 主头熵选区错误比例 |
|---|---:|---:|---:|---:|---:|
| O1 | 51,366 | 2,910 | 11,097 | 5.6652% | 21.6038% |
| O2 | 29,735 | 3,736 | 4,743 | 12.5643% | 15.9509% |

这里是“错误捕获排序”，同样覆盖率下选中更多错误更好，不应将较低选区错误率误读成风险检测更好。
现有比较按真实类别、边界区域和覆盖率匹配，但来自相关的训练 L 像素，不是独立患者显著性检验。

重要实现细节：`labeled_diagnostics` 的 `main_entropy` 是主头三分类熵；错误是 disc 内 rim/cup 条件错误。该排序未限定在实际 fine-PAS 接受集合。不能直接据此声称条件熵已被验证，更不能声称它能改善训练。

未测试组合预测器，所以不能证明多尺度分歧在任何组合下均无价值；但当前证据不足以为保留它或继续救活它分配训练预算。

## 四、冻结底座与新假设

建议研究标识：`MAIN_HEAD_HIERARCHICAL_KL_V1`。

全部实验直接从已验收 B2 起步，不从 A5、HALF 或 OBS0 的训练终点续训。冻结 B2 的参数保护、A/B 更新语义、优化器、EMA、数据顺序、增强、PAS、LCTX 几何、warmup、U ramp、分母、学习率和部署方式。

新臂不保留辅助头、DS、lagged multiscale risk，也不附加原 H。主网络的允许更新路径保持不变；不得通过冻结 B 或将 U 限制到辅助头实现“安全”。

### 4.1 层级分解

设 teacher 分布 q=(q_bg,q_rim,q_cup)，student 为 p，且 teacher stop-gradient。

- d_q=q_rim+q_cup，d_p=p_rim+p_cup；
- c_q=q_cup/d_q，c_p=p_cup/d_p；
- P_i=KL(Bern(d_q)||Bern(d_p))；
- C_i=d_q*KL(Bern(c_q)||Bern(c_p))。

则 KL(q||p)=P_i+C_i。该分解是已有概率恒等式，不是新理论贡献。

只替换 B2 原 fine KL：

    L_U_new = original_reduce(M_PAS * (P + w*C))

不是在原 KL 之外再增加一项。w=1 必须走原损失代码路径，或达到预注册的数值/梯度等价要求；禁止更改 mask 或按 sum(w) 重新归一化。稳定实现使用 logsumexp 及正确的零质量极限，不直接对极小概率粗暴相除。

### 4.2 唯一预注册候选

    E = original_fine_PAS_mask AND (teacher_disc_probability >= 0.9)
    s = binary_entropy(c_q) / log(2)
    w_target = 1 - 0.5*s on E; otherwise 1

范围 0.5≤w≤1。背景及低父类可信区域不降级。门槛 0.9 沿用现有父类可信语义；0.5 是这轮预注册的最大衰减幅度，不是数据证明的最优值。不得在本研究内搜索该系数、阈值、温度或边界宽度。

仅使用当前 teacher 主头与当前图像，无真实边界输入 U，无旧域校准器。所有权重 stop-gradient。

软 KL 已经包含 teacher 的软不确定性，额外熵降权可能没有收益，甚至抑制有用监督。这正是本轮需检验的假设，而不是可以省略的对照。

## 五、阶段 P0：零 optimizer 的可行性诊断

### P0-A：只读既有结果

重算均值、逐域/逐类差异、八点分层排序和同状态选择量。不得由聚合 JSON 反推并不存在的逐像素联合分布、conditional entropy 排序或患者置信区间。

### P0-B：经另行授权的无更新前向

公开聚合文件不足以完成新增排序。因此，需要在执行环境中用已验收 B2 checkpoint，对当前域训练 L 做无梯度评估；U 仅计算无标签聚合统计。零 optimizer 不等于零数据读取、零前向或自动获得权限。

优先使用每种顺序的 stage-2 起点与 B2 终点，共四个已有冻结状态。不要为取得未保存的中间状态重跑训练。对应当前 L 每状态一次完整固定评估；U 使用每状态预先绑定的有限 batch 清单。数据量和前向次数在运行前由 manifest 展开冻结。

必须同时报告：

- 原 fine-PAS 接受集和 E 的覆盖，按图像、teacher 预测 rim/cup 分开；不能被大量背景像素稀释。
- 三分类熵与条件熵，对条件错误的 top-25% 捕获能力，按类别/区域匹配；GT 仅用于当前 L 的事后评估。
- parent 误判：不能只在 GT disc 内评估而掩盖将背景误认为 disc 的情况。
- 候选 w 的分布、U 上被衰减的 teacher disc 概率质量、P/C 损失占比；这些 U 统计不叫准确率。
- 以独立图像/患者为报告单位；若只有聚合像素计数，明确不能计算患者级置信区间。

建议投入门槛，需在新数据读取前冻结：在两种顺序的 B2 终点，条件熵 top-25% 的图像平衡错误捕获率至少达到匹配随机排序期望的 1.25 倍；相对三分类熵没有超过 5% 的相对捕获率下降。没有条件错误的图像不硬凑比值，单独列支持量；样本支持不足则记为证据不足。上述数字只是开发投入门槛，不是统计定理。

若分数在实际 fine-PAS 接受集合中不具备错误定位能力，不进入训练，也不通过更换阈值/加辅助头补救。若覆盖几乎为空，记录无实用作用范围，不靠增加系数补偿。

## 六、阶段 P1：先区分定位收益与普遍减弱监督

仅在 P0 值得继续且完成新代码外部审阅后，提出以下矩阵。

| 臂 | 定义 | 相对 B2 新增 stage-2 节点 | 正式更新 |
|---|---|---:|---:|
| C0 | 原 B2 | 优先历史导入 2 个终点 | 0 |
| C1 | 同作用范围、同 teacher 前景概率质量衰减的均匀 conditional 降权 | 2 | 5,300 |
| C2 | 固定主头 conditional-entropy 定位降权 | 2 | 5,300 |

seed=163；O1 从原 B2 已学 RIM 的 stage1 前缀开始，在 Drishti 做 2,100 次更新；O2 从原 B2 已学 Drishti 的 stage1 前缀开始，在 RIM 做 3,200 次更新。
新增 4 个节点、10,600 次正式更新，source/stage1 新增 0。

C0 历史复用有条件：checkpoint、数据 split、环境、优化器语义、无操作代码路径及资格等价均需验证。若不能严格复用，先申请重跑 C0 两节点，正式预算变为 15,900；不能静默补基线。

### C1 的匹配公式

每张图像、每个 teacher 预测细类组成分组 E_g：

    w_uniform_g = sum_{i in E_g}(d_q_i*w_target_i) / sum_{i in E_g}(d_q_i)

空组置 1，E 之外仍为 1。这匹配被衰减的 teacher disc 概率质量，不匹配实际 loss 值、梯度范数或 Adam 参数位移。论文不得混称为“完全同等优化剂量”。

每个臂使用本臂当前 teacher 状态计算预算，禁止回放另一臂的 mask 或权重。各臂相同图像/初始随机状态不意味着训练后 teacher 状态相同。

### 开发晋级门槛

以下门槛针对新研究，须运行前冻结，不改写任何旧门槛：

- C2-B2 两顺序平均 Final≥0.003，即 +0.3 个百分点；每个顺序 Final≥B2。
- C2-C1 两顺序平均 Final≥0.001，即 +0.1 个百分点。
- 每顺序 Incoming≥B2-0.005；每个旧域单独 macro Dice≥B2-0.002。
- 逐域 rim/cup 全部列出；任何旧域 cup 下降超过 0.005 触发科学失败审查，不允许 source 或 disc_union 掩盖。

这不是显著性/非劣检验。不要把 +0.0001 量级的开发波动称为新机制成功。

若 C1、C2 均改善但彼此持平，只支持“降低条件监督压力”，不支持“风险定位”。若 C2 胜 C1 但不胜 B2，不算新增收益。若失败，关闭该固定配方，不自动补 seed、改系数或挑最好中途 checkpoint。

### 输出与诊断

输出 6 个终点、18 个域级结果以及预注册 contrasts。报告 source / first-target / incoming 和各自 rim / cup / disc_union，全部采用固定最终 checkpoint。沿用对应每节点四个固定诊断时点，观测 w、覆盖和 parent/conditional 损失；任何额外 VJP 在新预算中单列，不伪装为免费的既有诊断。

共前缀时 DeltaForget=-DeltaOld；两者不是两项独立证据。

## 七、后续条件分支，不自动执行

### P2-A：排除普通不确定性降权解释

只有 C2 在 P1 通过后，才考虑在同开发 seed 的两种顺序补 C3：对整个 fine KL 做主头熵降权。新增 2 节点、5,300 正式更新。故 P1+此项为 15,900 次新增正式更新（历史 B2 可复用时）。

C3 应匹配被移除的总 loss 值，而不只是同一系数。一个可审阅的确定性实现：令 r_i=1-w_target_i，在每个同样的 E_g 中，

    gamma_g = sum(r_i*C_i) / sum(r_i*(P_i+C_i))
    L_C3_i = (1-gamma_g*r_i)*(P_i+C_i)

分母为零时 gamma=0；比值和权重全部 stop-gradient，数值限制在 [0,1]。这在同一模型状态下匹配移除的标量 loss，不意味着匹配梯度或 Adam 更新。每臂用本臂状态计算。

此控制用于区分“保留 parent、降低 conditional”与一般整 KL 减弱；不胜 C3 时，不声称层级处理有独立优势。

### P2-B：独立训练种子确认

只有完成上述区分后，再提出两个预先固定的新训练种子，至少比较 B2、C1、C2；若论文核心包含相对整 KL 降权的优势，C3 也应进入确认。

只做 stage-2 条件确认时：三个臂×两个新种子×(2,100+3,200)=31,800 正式更新；四臂为 42,400。前提是各 seed 自己的 B2 前缀已验收可复用，不能将 seed163 前缀伪装成新 seed 的完整训练复现。缺失前缀须另列预算，不自动补。

每 seed 先平均两个顺序，再统计 seed 间差异；两个顺序和不同域不是独立 seeds。报告逐 seed 差异、均值、标准差，不以海量像素制造显著性。

### 完整持续学习与最终确认

stage-2 干预成功只证明对 B2 既有前缀有效。完整方法必须从首目标域起启用同一机制，并在两种顺序的完整持续学习序列上验证。开发 seed163 的 stage-2 结果不能与两个新 seed 的完整序列混成“三种子完整方法”。

当前固定域/划分已被多轮开发结果用于决策，增加随机种子只验证训练随机性，不会恢复测试集独立性。最终论文确认需要锁定后的未参与选型评估划分或外部域；完整矩阵和额外数据预算单独提交，不在本提案中自动授权。

## 八、工程、权限与历史记录

本文件不授权真实 optimizer、source/stage1 重训、监测恢复、私有数据读取或 GitHub 写入。
准备阶段应交付冻结公式与边界条件、CPU 单测、same-state/no-op 等价测试、checkpoint/RNG/EMA/optimizer 恢复测试、历史导入绑定、独立预算及 REVIEW_REQUEST，然后停止在 `STOP_AWAITING_EXTERNAL_CODE_REVIEW`。

10,600 / 15,900 等数字是正式更新预算，不是全部物理调用。CPU/CUDA 资格、smoke、失败注入、失败尝试和额外 VJP 必须另册逐项预声明；不得将本轮已完成的 CUDA 10、smoke 8 或其授权当作新研究可复用调用额度。

原 AGMS、HALF、OBS0 的负结果与冻结门槛继续保留，不以改名覆盖。即使新机制有效，也不能将 stop-gradient、熵筛选或 KL 恒等式本身称为首创。

## 证据位置

以下仓库文件均以本文件开头的结果提交为 ref：

- `experiments/lcrseg/docs/agms_observer_v1/results_20260920/FINAL_REPORT.md`
- `experiments/lcrseg/docs/agms_observer_v1/results_20260920/FINAL_METRICS.csv`
- `experiments/lcrseg/docs/agms_observer_v1/results_20260920/DOMAIN_METRICS.csv`
- `experiments/lcrseg/docs/agms_observer_v1/results_20260920/COVERAGE_AND_RISK.json`（本次核对了公开汇总及部分诊断片段）
- `experiments/lcrseg/docs/agms_observer_v1/results_20260920/GRADIENT_DIAGNOSTICS.json`（八点整体结论引用 FINAL_REPORT；本次直接检查了部分逐点导出）
- `experiments/lcrseg/docs/agms_observer_v1/01_SCIENTIFIC_SOLUTION.md`
- `experiments/lcrseg/docs/agms_cl_v0_1/results_20260919/FINAL_REPORT.md`
- `experiments/lcrseg/agms_cl_v0_1/core.py`
- `experiments/lcrseg/agms_observer_v1/diagnostics.py`
- `experiments/lcrseg/docs/agms_cl_v0_1/inputs/BASE_FROZEN_OPTIONS.json`

关于创新边界：Wang et al., *Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels*, CVPR 2022 / arXiv:2203.03884，已使用预测熵区分可靠/不可靠像素；其具体队列机制不是本提案，也不应引入本项目的 history-free 约束中。
