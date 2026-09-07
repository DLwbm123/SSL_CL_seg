# SSL_CL_seg — SHOR-UV V0.8：冻结专家的反事实收益学习与安全否决开发实验

## 0. 执行任务与证据边界

用户转交本文件并要求执行时，授权一个**新的开发实验**，不是 CARe-HR R2 恢复，也不是重新裁定 R2。不要只返回计划：完成固定的工程检查后，实际拟合下述小型收益模型，生成患者分组折外预测、对照和报告，然后停止。

旧 `FAIL_FROZEN_ACTION_SPACE_CAPACITY`、所有旧失败、BLOCKED、INCOMPLETE、locks、reservation、源码、测试与报告不变。不得通过降低旧门槛、将 oracle 当成部署策略、识别并硬编码失败病例、改主方法名字掩盖失败来获得 PASS。

这是基于已经开发暴露的 Fundus 数据设计的探索性实验，不是独立验证。新增真实监督拟合是**本文件明确允许的范围变化**；不通过修改旧 REVIEW_LOCK 实现。它利用历史 train-labeled GT 和冻结快照，不能声称已经符合不回访历史标签的严格在线／无回放协议。本轮要验证能否学习有用的拒绝决策，不是完成论文全部主张。

## 1. 起点、分支与不变资产

- 仓库：DLwbm123/SSL_CL_seg。
- 起点：67d38914085cfa94094b297ff539a38417976e49（R2 publication verification）。核验它与 c0ff70c1f36e9836206dabc0eb471abca61dce69 的血缘；不得用旧 main 代替。
- 新分支：`codex/shor-uv-v0-8-utility-pilot`。
- 新实现／测试／文档：`experiments/lcrseg/shor_uv_v0_8/`、`experiments/lcrseg/tests/shor_uv_v0_8/`、`experiments/lcrseg/docs/shor_uv_v0_8/`。
- R1 动作源码：9bbbacd25f3c3abf885205009eb14d698b36f32b。
- R2 evaluator 源码：271fc261380d2ef0abe5ea33088e8df72553eefe。
- 使用新 worktree、新 NAS 输出根，不修改用户未提交工作，不 force push、不合并 main。

先读 AGENTS、R2 报告、发布核验、数据/快照/cache/role/row-order 清单、scoring contract 和 loader adapter。复用已经通过实际部署的 R2 路径与解码实现；不要重写路径拼接、HDF5 格式或基础 Dice。

固定人群：198 个 own-seed train_labeled seed-case 行，177 名患者，seed 0/1/2 各 66 行；当前域为 Drishti_GS，历史域为 REFUGE、RIM_ONE_r3。9 份冻结 expert probability cache、冻结 current/Ridge/SHOR/PPC predictions、路由和 alpha 均需逐项绑定已有哈希和行顺序。不能因为研究新 gate 而替换 checkpoint、病例或 source-role。

新分割网络 forward、模型训练、EMA/GAS/prototype 更新为 0。不得读取旧 SHOR V0.4 formal_03、own-seed val/test/hidden-unlabeled GT。允许在本次训练服务/评估服务中读取规定人群的 own-seed train_labeled GT，以及旧 R2 已授权的开发评价产物；所有复用计数与新读取计数区分记录。

## 2. 研究问题与为什么改变方法

R2 严格 O_CAP 的整体/历史增益约 0.019588/0.029381；去掉两项面积限制后为 0.206080/0.309120。相同严格动作空间内的任何选择器都不可能超过已枚举 oracle。本轮不再拟合这个空间，不扩大 proposal 数，不做 2%→4%→8% 的预算试到通过。

优先利用已经有较强开发收益的冻结 SHOR：整体 Dice 0.8052579174291238、历史 gain 0.31303484788919705；其当前域 drop 为 0.015315363792476487。本轮问题是：

> 能否在不输入真实域身份或 GT 的情况下，预测当前与历史专家在整个病例上的收益/损害，对 SHOR 的危险覆盖进行否决，同时保留足够历史收益？

不使用 GT 选最优专家作为部署策略。学习目标是专家预测的分割损失差，而不是来源域是否识别正确。这里的“反事实”指在同一已标注开发病例上对多个冻结专家的预测作离线配对评价，不作因果识别声明。

## 3. 部署动作只有两个，不重新设计区域动作

记 current=B0/seed/stage2。对给定病例读取冻结 SHOR route：
- route=2：始终返回 current，收益 gate 不得新增历史调用。
- route=h，h∈{0,1}：仅可选择该冻结历史专家的整图预测，或否决并返回 current。

主方法：`SHOR_UV_CF`。输出精确复用 corresponding frozen hard mask/probability，不做空间裁剪、概率混合、形态学、oracle 局部修正或另选专家。没有新区域面积上限。所有保存模型均为轻量收益 heads；分割网络不更新。

该动作空间明确不同于旧 CARe-HR，故需独立新协议。风险来源从“像素改得太多”改为“该病例上历史专家预计会降低 Dice”。不承诺这一定能成功。

## 4. 用未被路由器选择的专家构造训练监督

对每个允许的 seed-case 行 i，使用缓存中的 current p_i2 和两个历史 p_i0、p_i1。分别 h=0,1 计算三项目标：

- d_rim(i,h) = Dice_rim(argmax p_ih, y_i) - Dice_rim(argmax p_i2, y_i)
- d_cup(i,h) = Dice_cup(argmax p_ih, y_i) - Dice_cup(argmax p_i2, y_i)
- H(i,h) = max(0, -d_rim(i,h), -d_cup(i,h))
- g(i,h) = (d_rim+d_cup)/2，仅由前两者导出，不用 union foreground。

使用冻结评分 contract：GT=255 不参与支持，empty class 与 all-ignore 约定不变；全-ignore样本如发生则有明确标记，不给新 heads 当作有监督证据。R2 已发现本批无 ignore，仍需实际核验。

最多 396 个 seed-case-expert 配对，仍只有177个患者，绝非396个独立样本。对同一患者跨 seed 和 h 的所有配对给予总拟合权重1。所有配对跟患者一起划分，不能按动作或专家随机拆折。

**不能只用原 SHOR 接受的配对训练主方法。**那会丢失大量潜在危险历史调用。原路由未选中历史专家的当前病例，仍可提供两种历史专家的有标签误差差值。另设“只用原路由接受配对”的消融，检验这一设计，而不是宣称补充配对必然有效。

## 5. 固定18维、无GT的特征

U = current hard 或 historical hard 属于{1,2}的像素并集；U为空时使用全图。所有函数输入只允许两份预测概率和冻结alpha，不允许 GT、true domain、patient_id、case_id、seed、文件路径、fold、oracle action ID 或过去评价误差。

定义 area 为全图 hard-mask 像素数、N为图像像素数；entropy 使用自然对数和数值安全的0log0=0；softDice采用 2*sum(p*q)/(sum(p)+sum(q))，分母为0时1。特征顺序固定为：
1–4. current rim/cup area/N，historical rim/cup area/N。
5–6. log((historical class area+1)/(current class area+1))，rim、cup。
7–8. U内 current/historical 三类entropy均值。
9–10. U内 current/historical max probability均值。
11–12. U内 current/historical top1-top2概率margin均值。
13. U内 current/historical JS divergence均值。
14. U内 hard disagreement比例。
15–16. 全图 current/historical softDice，rim、cup。
17. log(alpha_h+1e-12)-log(alpha_stage2+1e-12)。
18. 同一冻结alpha的top1-top2 margin。

使用float64提取与拟合，缓存输入和预测复用原dtype，不改变原argmax约定。非有限/非法概率不允许静默填补。alpha来自已冻结描述子/路由管线，不重拟合；明确其本身不是本轮重新cross-fit的上游模型。

特征提取、病例动作和fold方案先冻结哈希。拟合的标准化均值/尺度只能来自相应fit split，零方差尺度设1；不得用全体198行预标准化。

## 6. 小模型与固定网格

使用一个共享设计矩阵的3输出加权 Ridge，分别回归 d_rim、d_cup、H。截距不惩罚。加权平方损失加 λ||W||²，patient total weight按上节执行；明确权重总和为训练患者数，避免λ随样本复制变化。

λ固定为[0.1,1,10,100]。内层OOF三个目标的等权、患者加权MSE最小者入选，精确tie选较大λ。不增加树模型、MLP、非线性扩展或事后网格。

推断时 d_hat_rim/d_hat_cup裁剪到[-1,1]，H_hat裁剪到[0,1]；g_hat为两项d_hat均值。接受冻结SHOR历史覆盖的条件：

`g_hat > epsilon_gain AND H_hat <= harm_limit`

固定8个阈值候选：epsilon_gain∈{0,0.005}，harm_limit∈{0.01,0.025,0.05,0.10}。

这里的H_hat是回归预测，不是置信上界或conformal保证。本轮不拟合conformal、不做200次bootstrap再校准；不得把point prediction命名为UCB。

## 7. 患者分组嵌套开发评价

沿用冻结PPC 5个外层patient folds，并核验177名患者的所有seed/h重复严格同折。只有本轮新增heads/标准化/阈值的外层患者排除可以声称通过，上游分割和旧SHOR已看过这些开发患者，不能声称全管线无泄漏或独立测试。

每个外层fold：
1. 外层评估患者GT不进入该fold heads、λ选择、阈值选择。训练服务只能获取允许的inner patients配对目标。
2. 外层训练患者按固定salt `shor-uv-v0.8-inner` 的sha256排序后index%4，构造4个inner folds，不按case或专家分折。
3. 对4个λ分别得到内层OOF pair predictions，并按第6节选λ。
4. 在该λ的内层OOF预测上，按实际冻结SHOR route评价8个阈值；source/domain字段只能由inner selection evaluator用于开发安全聚合，不能进入特征或fit API。
5. 候选内层安全条件：当前域mean drop<=0.01，当前域最大class drop<=0.015，最大seed-domain mean drop<=0.02，无当前域case macro delta<-0.10；需要的组无支持则候选不得称为安全合格。
6. 安全候选中优先历史gain较大，再整体gain较大，再当前drop较小，再harm_limit较小，再epsilon_gain较大，再固定candidate ID。浮点比较用未舍入数值，不引入看结果决定的tolerance。
7. 若没有合格候选，预定义该fold `CURRENT_FALLBACK_NO_INNER_SAFE_CANDIDATE`，如实输出current并纳入外层评价。它不是成功，不准丢弃该fold，也不提前放弃其余可执行对照。
8. 用全部外层训练患者重拟合选中的λ，冻结模型/标准化/阈值，输出该fold全部外层患者的GT-free部署动作。先写预测seal，再由独立evaluator评价。

所有fold的预测在查看本轮outer汇总前完成并密封。GT先前已被开发使用是事实，seal只能约束本轮执行流程，不会恢复数据的新颖性。一次固定实验矩阵不按outer结果迭代重选特征、λ或threshold。

## 8. 必须保留的对照与消融

同样198行上完整报告：
- `CURRENT`、`RIDGE_HARD_FROZEN`、`SHOR_FROZEN`、`PPC_FROZEN`：复用并重验R2基线。
- `SHOR_CONFIDENCE_VETO`：简单置信差对照，U内historical max probability均值减current均值。候选接受条件difference>=t，t∈{-0.1,0,0.1}，另有ALWAYS_ACCEPT。内层OOF选择使用同样安全条件与排序，不用outer调参。
- `SHOR_UV_CF`：第4–7节，唯一预指定主方法。
- `SHOR_UV_ROUTED_ONLY`：相同特征/模型/流程，但fit配对只取原SHOR历史route接受的那个h。仍按患者归一化；如某训练split无足够可求解数据，显式fallback和记录，不捏造训练成功。
- `SHOR_UV_GAIN_ONLY`：使用完整CF pairs及同一heads拟合，但接受仅检查g_hat>epsilon_gain，epsilon∈{0,0.005}，仍在内层选择。检验harm head是否有增量价值。

不能把消融或置信对照胜出写成主方法胜出；但所有结果必须交付，让研究者知道简单对照是否已经足够。即使主方法不满足内层选择条件，也完成其他对照和分析，不将失败候选隐藏。

## 9. 本轮分析输出：重点看真实收益与稀有大损害

统一使用R2 case→seed/domain均值→组等权聚合；主指标不改分母、macro定义或patient权重。训练patient equal不等于评价可更换主aggregation。

报告overall/history/current、每seed、每domain、rim/cup、paired差、历史收益保留率、实际接受/否决数、beneficial/harmful pair数量、实际当前域错误覆盖数、最差病例/患者delta、delta<-0.10数量，以及每个outer fold候选/fit/FALLBACK状态。

私有诊断：将历史错误覆盖来源问题与分割损害分开。核验已知来源错误是否集中于一条当前域seed-case，不能只凭154/155自动推断其身份。复核所有当前域和历史域的大损害，不只检查已知坏病例。明确seed-row与patient是否重复。

将无GT得出的outer接受决定与事后GT比较，报告实际纠错precision/收益损失，不引用oracle151/154等比率冒充模型结果。

新增head的可预测性同时报告：外层pair患者加权MAE、收益符号准确率（no-op/zero-target单列）、harmful pair召回和worst harm案例；不能只报RMSE而不报实际策略Dice。报告CF pairs中危险样本的来源和类别比例。

可以对已固定的outer预测做2000次患者聚类bootstrap，salt/seed在freeze中固定为2026090801。每次给唯一患者抽样权重，同一患者跨seed/policy保持配对，重新按原seed/domain aggregation计算；空组replicate标记不可用，报告有效数，不悄悄重抽。CI只代表固定折外策略的病例重采样，不是重拟合稳定性或泛化保证。

不用bootstrap p90替代旧的router-refit p90。旧重拟合p90/p10门槛记NOT_EVALUATED。本轮当前域只有28名独立患者，不能基于少量零事件给出强尾部安全保证。

## 10. 不降低数值目标，但区分开发信号与正式确认

`PASS_DEVELOPMENT_UTILITY_SIGNAL` 仅当主方法outer结果同时满足：
- overall gain>=0.17、historical gain>=0.27；
- REFUGE、RIM_ONE_r3各自gain>0，3个seed的overall gain都>0；
- 当前域mean drop<=0.01，最大当前class drop<=0.015，最大seed-domain mean drop<=0.02；
- 当前域没有case macro delta<-0.10；
- 对冻结SHOR overall差>=-0.01、history差>=-0.02；
- 对冻结SHOR current drop至少减少0.005。

这不是旧R2通过，不是最终安全保证，也不是新数据的确认。若主方法不满足，输出`DEVELOPMENT_UTILITY_SIGNAL_NOT_ESTABLISHED`，并逐条标注VALUE/SAFETY/BASELINE/INCREMENTAL问题，不采用只显示一个失败字段的报告方式。

无候选fallback是明确策略行为，不能让全current凭安全通过；无真实fit、未完成全部outer患者或基线不一致时不能给上述科学终态，应输出准确工程BLOCKED/INCOMPLETE。

若简单置信对照同样好或更好，明确`NO_ADDED_VALUE_OVER_SIMPLE_VETO`，不粉饰新增heads的贡献。若新gate有开发信号但可能过拟合，只能建议冻结后新的独立确认；不得启用旧formal_03或把现有开发集当成重新未见。

## 11. 工程执行：复用已恢复的系统，避免新的审批循环

先纯函数与合成端到端测试，再冻结本轮源码、18维特征、pair target定义、fold、网格、候选选择与报告脚本；记录精确source SHA，local/server回归与真实进程退出。继承原189项相关回归和路径/metric/密封保护，新增patient leakage、标准化只fit训练、CF pair分组、threshold只inner、no-op和fallback、部署API拒绝GT/domain、prediction seal、实例完整流程测试。测试数量由实际收集决定，不预写PASS数量。

允许真实监督Ridge拟合，但日志必须给出实际fit counts、患者数、pair数、失败/退化状态，不以optimizer=0宣称没有训练。基础网络forward/更新、实际新增图片读取均应为0，允许按清单读取概率与label。GPU不需为了利用率而开工。

本轮不再要求新增一遍容量oracle，也不重做所有缓存。基线zero-error parity、资产/接口检查与外层隔离是必要门槛；普通开发负面结果不是中途抛异常理由，应完成固定矩阵并保存证据。

若遇到纯工程错误：未查看本轮outer汇总时，允许至多一次有完整diff及新source测试的工程恢复，只能修I/O、schema或报告错误，不改本协议科学定义；旧attempt和暴露计数保留、新目录执行。已查看outer结果后不得自动改特征/超参数/选择规则重跑；报告准确原因并停止。不是无限调试直到获得PASS。

所有历史保护清单及R2公开/私有结果保持不变，新增本轮保护manifest。发布聚合指标、协议、哈希、版本和拟合统计；私有病例、GT、概率、checkpoint和病例身份不公开。推送新分支且验证远端SHA，不合并main。

## 12. 必须交付

- `PROTOCOL.md/json`：本文件的操作性冻结；科学规则不能事后改动。
- `INPUT_AND_EXPOSURE_LINEAGE.json`、`FOLD_ASSIGNMENT_MANIFEST.json`（患者身份私有，只公开hash/count）。
- `FEATURE_CONTRACT.json`、`COUNTERFACTUAL_TARGET_AUDIT.json`、`TEST_REPORT.json`。
- `FIT_ACCOUNTING.json`：每fold/λ/head实际fit及数据范围。
- `INNER_SELECTION.json`、每fold模型与动作seal，私有保存。
- `OUTER_POLICY_METRICS.csv`、`PAIRED_COMPARISONS.csv`、`SAFETY_AND_TAIL.csv`、`ABLATION.csv`。
- `FAILURE_ATTRIBUTION.md`：所有机制结论的证据、样本量和限制。
- `FINAL_REPORT.md`：真实部署规则的折外结果、主方法是否满足开发信号、与简单对照相比的增量、可否继续。
- `PUBLICATION_VERIFICATION.json`：source/report/head明确区分。

本轮结束后停止。下一阶段只能草拟：固定验证集/新患者确认、在线stage-wise训练协议、存储与推理代价、必要的论文新颖性比较。不要自动启动新网络训练、外部测试或新方法。

## 13. 结果来源与方法定位

本计划基于 R2 c0ff70c1f36e9836206dabc0eb471abca61dce69 下的 CAPACITY_FINAL_REPORT.md、CAPACITY_METRICS.csv，及其继承的 PPC_SHOR_V0_6B_ROUTING.csv。原始终态与所有oracle仅作为设计依据。

“学习接受/拒绝专家”的一般思想已有 learning-to-defer 文献，不能声称首次提出。可对照 Mozannar & Sontag (ICML 2020, Consistent Estimators for Learning to Defer to an Expert) 和 Verma et al. (AISTATS 2023, Learning to Defer to Multiple Experts)。这些论文不为本次Ridge gate提供自动的理论保证。潜在贡献必须由分割损失差、持续快照、当前能力保护、数据权限协议和实证结果共同支持，而不是名字或模块数量。

## Operational freeze adopted before new target access

- Weighted standardization uses each fit split's patient-normalized pair weights for both mean and population variance; exact zero variance becomes scale 1.
- Inner lambda MSE evaluates deployment-clipped three-output predictions. CF uses all evaluable pairs; ROUTED_ONLY uses routed evaluable pairs for fitting and lambda validation. Zero eligible fit rows are an explicit no-fit fallback; positive lambda permits rank-deficient nonempty designs.
- Patient hashing is SHA256 of UTF-8 salt + "|" + patient_id, lexicographically sorted digest, with patient_id as deterministic hash-tie fallback, then index modulo 4.
- Confidence IDs are fixed in order ALWAYS_ACCEPT, -0.1, 0, 0.1. Its absent harm/epsilon coordinates are identical across candidates; final exact ties use candidate ID. GAIN_ONLY uses the identical CF heads and selected lambda; only inner threshold selection differs.
- The target-materialization service opens the 198 authorized labels once and checks frozen control parity. It writes five distinct outer-training packages and a separate evaluator-owned target file. Each trainer opens only its own training package; individual fit calls receive only their inner training array slice. Deployment reads no target package, GT or domain. The final evaluator reads cached target-service metrics only after all five outer deployment masks are sealed.
- Probability and hard-mask byte identities are preserved. Deployment saves exact cached-expert argmax masks, with no learned segmentation or new forwarding.
- Each safe candidate needs evaluable support in all nine seed/domain groups. All-ignore rows remain in compatibility evaluation but are excluded from head supervision.
- All policy comparisons and gates use unrounded primary seed/domain-equal metrics. Current maximum class drop is the maximum of aggregated current rim/cup drops, consistent with inherited PPC.
- Primary vs simple-veto dominance is declared NO_ADDED_VALUE_OVER_SIMPLE_VETO if confidence weakly exceeds primary overall/history gains and weakly improves current mean/class drop, maximum seed-domain drop and current severe-row count. Other cases remain an explicit tradeoff; no superiority claim follows automatically.
- Pair sign accuracy excludes exactly zero macro targets; these counts are separate. Harmful-pair recall uses actual H>0 and predicted H>0, solely a descriptive diagnostic, not the deployed threshold or a calibration claim.
- Bootstrap: 2000 paired multinomial patient draws, numpy default_rng seed 2026090801; same patient multiplicity across seed/policy; original group weights, no redraw of empty-group draws; percentile 95% intervals, no refits.
- Private model/selection/prediction files and raw counters are create-only. Protected history hashes and all original R1/R2 seals/results remain unchanged.
- Pure synthetic fixtures may substitute external roots/identities and in-memory cached arrays; no real CLI population overrides exist. They execute actual production target, fit, deployment, seal and report functions. The inherited 189 regression items remain required.
- Publication is explicitly authorized for this new experiment and overrides the legacy AGENTS no-auto-push default. No segmentation milestone backward/overfit test is applicable to a study that explicitly forbids network training; inherited synthetic regressions plus new utility tests are required.
