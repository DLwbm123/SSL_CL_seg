# CIST Plasticity + Add-on V0.2：可塑性与强基线增量核验

## 0. 本轮决策、身份和预算

这是基于 CIST V0.1 已公开负结果提出的新实验，不是旧实验的续跑、复现或数值修复。未声称服务器训练已启动。

- 唯一主候选：`CONV_CIST_L`。
- 两个诊断对照：`GN_L`、`GN_CIST_L`。
- 常规新增预算：3臂 × 3种子 × 2顺序 = 18个目标训练任务，47,700次正式更新。
- 不重训六份 SRC_CE，不重训已有36个CIST任务，不寻找旧KI，不启动插值/回拉/SVD保护。
- 本计划替代尚未执行的“两臂GN检查”计划，不是与之相加再执行一遍。工作代号 `CIST_PLASTICITY_ADDON_V0_2`，建议分支 `codex/cist-plasticity-addon-v0-2`。
- 如果精确对应的GN两臂任务实际上已经完成，只核查其已知任务账本与源码/输入/初始化/训练语义。完全等价者复用并披露；不能重复执行后择优。若确有协议差异，列出差异，不把不等价结果冒充匹配对照。
- GPU4/5/6/7既有共享授权沿用，实际启动前复查显存，不终止无关进程。本文不自动启动其他阶段。

旧 `NO_PRIMARY_ACCURACY_GAIN` 和旧的所有科学/工程终态均保持不变。主要指标仅为等权 Final，不要求新域、旧域、每类、每个种子同时改善。

## 1. 源证据与问题边界

仓库 `DLwbm123/SSL_CL_seg`。
固定报告提交：`4c0a0435d14762e27405f2f365e5c100a3a8bc07`。

必须读取：

```
experiments/lcrseg/docs/cist_v0_1/results/FINAL_REPORT.md
experiments/lcrseg/docs/cist_v0_1/results/TRAJECTORIES.csv
experiments/lcrseg/docs/cist_v0_1/results/SOURCE_BINDING.json
experiments/lcrseg/docs/cist_v0_1/results/BASELINE_BINDING.json
experiments/lcrseg/cist_v0_1/core.py
experiments/lcrseg/cist_v0_1/engine.py
```

已知事实：

- ISO_COND_LU Final=0.563604；F_CONV=0.606879；主差=-0.043275。
- 条件患者区间[-0.064430,-0.022930]，三个优化种子的顺序平均差均为负。
- 主方法对F_CONV的平均Incoming/Old差分别约-0.109656/+0.023106。
- O1（RIM→Drishti）顺序均差：Final -0.02027135、Incoming -0.10942977、Old +0.06888707。
- O2（Drishti→RIM）顺序均差：Final -0.06627812、Incoming -0.10988106、Old -0.02267518。
- ISO_COND_L Final=0.563747；增加当前U目标后的Final差=-0.000143，尚无实用U增量证据。
- AFFINE_LU的Incoming=0.623181，仍低于密集基线，故不能只把失败归于正交约束。
- 整个core（含GN）在旧实现中冻结，features()使用no_grad。这符合旧设计，不是执行器忘记解冻。

附加数学解释，不是患者实验结论：

对同一图像，CIST等价于 h'_p=T(x)h_p+t(x)，其中 T=I+V(Q-I)V^T，t=Vb。固定读出前的所有像素共享这个变换。对padding0卷积，有图像条件核 K_delta T(x) 与偏置 sum_delta K_delta t(x)。这说明它不能任意独立修正每个空间位置，但不证明其函数类一定无法达到高准确率。

待检验假设：末端图像级修正的适配空间可能不足；允许中间归一化或已有效的卷积更新后，条件输运是否仍有增量？该假设不能预先称为已证实根因。

## 2. GOLD与U目标的解释补充

用户附件《The Golden Subspace》C.1第12—13页明确说明更新归一化层weight/bias。该文是CTTA，其源原型、AGOP及训练目标不直接等同本项目。不得把当前CIST失败称为完整GOLD复现失败，也不得把GN解冻本身称为原创贡献。

旧全局等距臂存在下述解析不变性：

    ||(Q S_a+b)-(Q S_b+b)||_F^2 = ||S_a-S_b||_F^2,  Q^T Q=I.

在冻结特征、分母仅依赖冻结特征时，原U损失对全局Q/b没有有效实数梯度。条件臂两个视图的Q/b可以不同，不受这条零梯度结论覆盖。

这限制原GLOBAL与条件臂的归因解释；不改变旧评分，不重跑GLOBAL。本轮所有新臂均不使用U和旧view loss，避免同时改变适配空间和监督来源。未来再加入U属于独立试验。

## 3. 三个新臂与既有参照

### 主候选 CONV_CIST_L

从原SRC_CE开始；采用F_CONV相同的14层普通卷积训练集合；安装原ISO_COND_L的条件等距控制器。冻结全部GN、转置卷积、原分类头及惰性参数。学习14层卷积权重和2,920个控制器参数。

主问题：在保留已验证的F_CONV可塑性后，加上CIST模块是否提高Final？

### 对照 GN_L

从相同SRC_CE开始；只训练14个GroupNorm层的weight/bias；不安装CIST控制器。所有卷积、转置卷积、原分类头等均冻结。

### 对照 GN_CIST_L

GN集合与GN_L相同；另安装并训练原ISO_COND_L控制器；其他核心权重冻结。

诊断问题：轻量中间表示更新本身提供多少收益？在这一更新空间中CIST是否还有额外收益？

### 复用参照

F_CONV（主要效能参照）、F_FULL、ISO_COND_L、ISO_COND_LU、STATIC_SOURCE。保留原来源与评价，不声称这些结果在新代码重训。已有源/基线绑定如实核验，不能因为旧权重没有另一个名字而启动全目录溯源。

主候选不得按GN臂结果事后切换。GN臂更好时可以作为控制发现单独报告，但不改写主候选终态。

## 4. 精确参数集合与规模

七个double-conv块：

```
enc1
enc2
enc3
bottleneck
decoder.dec3.merge
decoder.dec2.merge
decoder.dec1.merge
```

CONV_CIST_L允许各块的 `.block.0.weight` 与 `.block.3.weight`，共14个Conv2d kernel。GN保持源值。

GN两臂允许各块的 `.block.1.weight/.bias` 与 `.block.4.weight/.bias`，共14个GroupNorm模块、28个affine参数张量。

已知模型的预计计数：

- 14个Conv2d kernel：438,192个标量；以真实枚举核验。
- 14个GN affine：4*(16+32+64+128+64+32+16)=1,408个标量。
- 原控制器16→32→72 MLP：2,920个标量。
- GN_L：1,408；GN_CIST_L：4,328；CONV_CIST_L：441,112。

这些臂不是等参数量消融。模块整体增量不等于等距几何的独立因果贡献。若新可塑性空间有效，不能直接套用全冻结核心上的旧消融证明新空间中各组件都不可缺少。

## 5. 控制器、基底与源初始化

- 源：每个seed/order已绑定的SRC_CE学生；不能加载已训练的目标CIST控制器或目标F_CONV作为初始化。
- CIST基底：复用同一源分类核导出的V/N，rank8，固定、停止梯度，不重做目标驱动AGOP或SVD。
- 控制器：原16→32→72、GELU、末层全零；与原CIST同源域/seed的keyed初始化。
- Cayley：FP64可微8×8 solve，FP32特征应用；不改成参数插值或一般仿射。
- 所有新臂初始化预测应等于共同源学生；两个带控制器臂初始化为恒等作用。
- 只继承源学生参数，目标Adam/LR/RNG按原目标协议重建；不继承历史EMA或优化器。
- 不要求新臂epoch20状态一致。可训练集合不同，此时分叉是实验干预的一部分。

## 6. 必须同时修改的三个工程位置

### 6.1 梯度图与优化器

移除新训练features路径上的整体no_grad/inference_mode，不在当前H、context或读出前detach。冻结卷积/头仍须允许对输入反传；参数冻结用requires_grad=False和优化器白名单。

不能继续 `optimizer_for(m.transport)`，否则GN或卷积不会更新。显式从所选arm的全部白名单参数构建Adam，并验证ID与数量。

不能在解冻后再次执行core.requires_grad_(False)。不要把eval()当作冻结梯度操作。只允许规定参数有.grad，其他固定参数无梯度且值不变。不要要求每个张量每一步梯度都非零；零初始化使控制器第一层首步梯度可能为零。

### 6.2 checkpoint与恢复

旧CIST payload只保存controller并引用immutable source core。该模式不适用于新臂：GN或卷积会变化。

恢复快照必须保存全部当前可训练核心参数、完整控制器状态（若有）、Adam、RNG、position、数据顺序/计数、固定参数/基底引用及哈希、pending诊断状态。可保存完整student.state_dict，或完整动态状态加不可变源引用；必须通过完整恢复等价测试。

禁止只恢复controller后让GN/卷积回到源值。禁止继续以“整个core等于source哈希”为不变量；替换为“该arm的冻结子集不变”。边界的源哈希仍永久保留，最终学生另有新哈希。

沿用先保存post-update状态、再运行可能失败的诊断。pending checkpoint恢复后先做未完成诊断，不重复已提交optimizer更新。保留失败、重放和所有实际更新成本。

部署必须包含最终更新后的GN或卷积、控制器以及固定基底；不能评价由原源core加新controller重建的错误学生。

### 6.3 U与诊断访问

新任务名字不能直接套入旧 `arm != ISO_COND_L` 条件，否则会错误创建U loader。新循环显式L-only，不实例化U图像accessor，不调用appearance/view_loss，不保留EMA。

仍可使用已冻结manifest元数据确定原步数；不把没有U图像读取误记成没有读任何元数据。

诊断等距性只能比较“同一次forward的当前H”和transport(H)，不能比较源模型H与经过GN/卷积更新后的H，不声称源功能不变。

## 7. 数据、损失与训练配置

```
split: 0
seeds: [61,62,63]
O1: RIM_ONE_r3 -> Drishti_GS
O2: Drishti_GS -> RIM_ONE_r3
target_epochs: 100
batch_size_L: 2
steps_per_epoch_Drishti: 21
steps_per_epoch_RIM: 32
optimizer: Adam
lr0: 0.001
betas: [0.9,0.999]
eps: 1e-8
weight_decay: 4e-5
lr: lr0 * (1 - zero_based_step/total_steps)^0.9
U_images: 0
teacher_or_EMA: none
```

optimizer字段与父 `optimizer_for` 及冻结配置交叉核对，不静默换AdamW或新weight_decay。这里按项目父设置，不因为GOLD正文用不同weight_decay而同步替换。

epoch1—20干净L的CE+前景Dice；epoch21—100原LCTX互补标注图像混合、donor只提供图像、锚定L像素一次计分。保留原噪声、几何增强、mask、ignore支持、source collection与浮点loss实现；不追加干净L损失。

源域数据、原型、旧特征和历史预测不进入目标训练。每次前向的context/Q/b用完释放，不缓存跨样本历史语境。旧域评价仅最终学生封存后的隔离evaluator执行。

## 8. 预算及工程资格

| 顺序 | 新目标任务 | 更新数 |
|---|---:|---:|
| O1 | 3臂×3种子=9 | 18,900 |
| O2 | 3臂×3种子=9 | 28,800 |
| 合计 | 18 | 47,700 |

资格成本单列。必要合成测试覆盖：初始输出一致、参数白名单、真实梯度路径、LCTX loss/来源等价、身份控制器旁路时与F_CONV原生训练内核的一步学生/Adam等价（EMA不参与梯度故分开记账）、当前H几何、完整动态状态恢复、部署重新加载、数据隔离。

控制器旁路回归必须真正固定为identity并排除优化器；不能让controller更新后再要求与无模块模型一致。

最多12次丢弃真实L smoke更新（3臂×2域×2步）。不以smoke分数选配置。不新增大资格网格。

正交诊断沿用现有可工作数值合同，不引入源权重左右泄漏门槛。真正NaN、状态丢失或违规数据访问暂停；类别退化和分数不佳不是工程失败。

GPU显存实测：中间特征需要反传，不能借用原纯冻结CIST的显存峰值。只保存一个当前学生、优化器及必要参数状态，不加载源教师。冻结参数、可训练参数、梯度、Adam、基底、临时计算及GPU峰值分别记录。

## 9. 少量有解释力的观测，不转成新门槛

复用每epoch训练日志报告CE、Dice、总loss均值。若聚合字段是epoch累计值，应除以相应steps后比较，不把sum写成mean。

记录GN偏移或卷积相对源变化量、控制器更新、Q/b变化、当前H的图内等距残差。均为描述量，不规定它们必须向预期方向变化。

保留epoch20和100 checkpoint以支持本次协议的恢复与解释；不读取epoch20验证GT，不按中间成绩选checkpoint。

全部新训练终态权重封存后，执行固定epoch100患者级最终评价。每任务评价旧域与当前域，共18个目标评价任务；源与旧基线评分复用，公开其来源。不得称这些重复开发患者为新患者确认。

## 10. 主要指标与决策

对每个seed先平均两个顺序，再三个seed等权：

    Final=(Old+Incoming)/2
    Forget=matched_source_score-Old

唯一主效能比较：`CONV_CIST_L - F_CONV` 的Final。

辅助比较：
- CONV_CIST_L - F_FULL；
- GN_CIST_L - GN_L（中间归一化条件下的模块增量）；
- 两个GN臂相对F_CONV和旧ISO_COND_L；
- 所有新臂的Old/Incoming/类别/seed/顺序/计算成本。

不要求所有比较为正，也不要求每个seed/类别赢。科学结论不得串联回旧“全面占优”规则。

- 主差>=0.005：实用幅度的正向开发信号，优先考虑固定方案复核。区间跨零则仍标记患者条件不确定性。
- 0<主差<0.005：小正向信号，不强行说无效，也不自动追加seed追显著。
- 主差<=0：无主准确率增益，不以只战胜旧弱CIST、参数少或某类改善替代主结果。

Old、Incoming或类别下降如实报告并解释，不设单项-0.05自动否决。本轮无临床安全认证目标；允许研究权衡不等于证明临床可接受。

组件比较决定主张：主臂赢只支持“CIST整体模块在F_CONV更新空间中有增量”；不直接证明等距、条件化、读出基底分别是独立原因。GN控制赢而主臂不赢，不重命名主候选；可将控制作为后续新问题，但本次不自动新训。

患者配对bootstrap：2000次，analysis_seed=2026091202，按物理域统一患者权重并在所有方法/seed/order共享。三个优化seed的配对差另报，不把六个顺序×seed单元当六个独立seed。区间条件于已训练模型和开发队列。

完成后不自动扩大decoder、rank、层数、U损失或源权重搜索。若主臂没有增量，结束这份CIST附加模块的固定尝试，而不是再逐层解冻形成连环试错。

## 11. 交付与终止

输出源码/配置freeze、精确任务矩阵、源/基线绑定、可训练参数清单、资格记录、完整RUN_LEDGER、FINAL_SITE_CLASS_METRICS、PAIRED_EFFECTS、PATIENT_INTERVALS、DOMAIN_CLASS_COSTS、MEMORY_AND_COMPUTE、FINAL_REPORT和公开可访问性核验。

全程保留旧终态、旧失败成本和新失败/恢复账本。图像/GT、患者ID、逐病例输出、权重及原始NAS路径不公开。不得把生成协议说成已启动训练，也不得把调度已提交说成36/36已完成。

建议状态字段独立：ENGINEERING_COMPLETE、PRIMARY_ACCURACY_EFFECT、COMPONENT_EFFECT、PATIENT_UNCERTAINTY。用具体差值而不是只有PASS/FAIL讲结论。

## 12. Codex执行入口

请执行本文件定义的CIST_PLASTICITY_ADDON_V0_2。

旧链接4c0a0435对应的仍是CIST V0.1六臂结果，不是GN可塑性检查结果。不要以同一报告重复计算新增训练。

本轮只新增GN_L、GN_CIST_L、CONV_CIST_L。最后一臂为唯一主候选。标准18目标任务、47,700步；已存在精确等价GN任务时只按不可变身份复用并减少新增预算，不重复训练择优。

从已绑定六个SRC_CE开始，只用原LCTX，不读U、不用EMA、不调rank/LR、不使用目标已适配权重初始化。保持CIST基底、控制器与Cayley定义。

修改必须同时覆盖feature反传、优化器参数白名单、core冻结子集检查、GN/卷积动态状态checkpoint及部署。禁止只保存controller又恢复源GN/卷积。保留pending-diagnostics精确恢复。

必要资格通过后一次完成固定矩阵，不根据首seed/单类别中途删臂或增加候选。主指标是CONV_CIST_L相对F_CONV的Final，披露权衡但不要求各方向全优。完成发表聚合报告和NAS归档后停止，不自动启动后续实验。

## 13. 参考定位

项目源码/报告以上述固定SHA及相对路径为准。GOLD：用户附件第12—13页C.1（归一化affine更新）；附录B（固定线性分类头的最小范数可实现修正）。论文没有保证本项目三臂会有效。

实现语义的官方文档：
```
https://docs.pytorch.org/docs/stable/notes/autograd.html
```
