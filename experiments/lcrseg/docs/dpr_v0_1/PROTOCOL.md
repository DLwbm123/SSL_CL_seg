# DPR V0.1：保留监督一阶进展的响应约束更新

## 0. 定位与本轮范围

本文件是新实验建议与可交给 Codex 的执行契约，不是已完成的训练报告。

停止 CIST 的扩容、换插入位置、调 Cayley/rank、再加 view loss 等后续搜索。保留 CIST V0.1 与 plasticity V0.2 的 `NO_PRIMARY_ACCURACY_GAIN`。不重新寻找原 KI，不将本方法冒充原 KI 或 GOLD 完整复现。

新问题：在保留 F_CONV 稠密更新空间和 LCTX 监督目标的条件下，能否用当前图像上的输出响应导数，减少一次参数更新中不必要的函数变化，同时不降低该步相对原 Adam 提案的监督一阶进展？

本方案是有待验证的研究假设。它不是旧域不遗忘定理。当前域图像不是历史图像的替代品；若旧、新分布在相关响应方向上不重叠，即使本地响应变化减少，也可能不改善旧域。

采用三个新臂、三种子、两个顺序；正常18个目标训练任务、47,700次正式更新。不要再铺开六臂95,400步的新结构搜索。通过必要工程资格后完成本固定矩阵；不设置类别/域/种子全面占优条件。

## 1. 证据起点及主次

当前报告：
- 仓库 `DLwbm123/SSL_CL_seg`
- 提交 `8610343c08de61f639844aa069ceecaec0de2b49`
- `experiments/lcrseg/docs/cist_plasticity_v0_2/results/FINAL_REPORT.md`
- 同目录 `TRAJECTORIES.csv`、`PATIENT_INTERVALS.csv`、`BASELINE_BINDING.json`。

既有 F_CONV、F_FULL、六份 SRC_CE 的绑定沿用报告及其父运行，最终确认学生/输入/训练配置哈希。不要仅依靠文件名或四舍五入分数认定等价。

已知点估计：F_CONV Final 0.606879、Incoming 0.719677、Old 0.494081；CONV_CIST_L Final 0.600493、Incoming 0.723969、Old 0.477018。后者对前者 Final −0.006385，Old −0.017063。GN_CIST_L 对 GN_L Final +0.006963，但自身仍低于 F_CONV 约0.029984。不能以弱参照上的局部增量替换主比较。

新实验不需要恢复 CIST、SVD 或低秩适配器。只保留已核实的普通模型、F_CONV 参数白名单、LCTX 数学和数据/评价边界。

## 2. 方法角色与历史状态

阶段入口与训练中保留的知识载体仍是已训练模型参数。当前 L 提供真实监督；当前 U 只用于查询当前参数所实现的读出函数的局部敏感方向，不产生伪标签、不产生 teacher target、不加入类别自训练损失。

每一步计算的输出、VJP/响应行、Gram 矩阵均为当步临时量，用完释放。禁止跨步/跨域维护响应方向库、历史特征、原型、Fisher/AGOP累计矩阵或旧样本。允许沿用父 F_CONV 工程中的当前 EMA，仅作对称状态/诊断，所有新臂一致，且对损失和响应查询均无贡献。不得为本方法新增独立旧教师。

不声称“只保留两模型对象就是只有两倍参数内存”：参数快照、响应行、Adam、梯度、激活和FP64求解临时量必须另计。部署是单一普通学生，无 CIST 控制器、无额外输入视图、无响应求解、无域路由。

## 3. 固定模型与训练参数

训练以下14个普通 Conv2d 的 weight，其他所有参数保持父 F_CONV 的冻结状态：

```text
enc1.block.0.weight
enc1.block.3.weight
enc2.block.0.weight
enc2.block.3.weight
enc3.block.0.weight
enc3.block.3.weight
bottleneck.block.0.weight
bottleneck.block.3.weight
decoder.dec3.merge.block.0.weight
decoder.dec3.merge.block.3.weight
decoder.dec2.merge.block.0.weight
decoder.dec2.merge.block.3.weight
decoder.dec1.merge.block.0.weight
decoder.dec1.merge.block.3.weight
```

预计438,192个可训练标量，以冻结结构和真实枚举核验。不存在A/B、低秩rank或Cayley矩阵。保留3×384×384输入、三类0/1/2、原3×3 padding0读出及align_corners=True输出插值。不更换骨干、损失、标签预算或归一化。

原 Adam：lr0=0.001、betas=(0.9,0.999)、eps=1e-8、weight_decay=4e-5、原多项式学习率。不要切换 AdamW 或另加源回拉。每步的监督梯度为纯 LCTX CE+前景Dice 的梯度；原 Adam 提案包含其既有 weight decay/动量作用。

前20 epoch干净 L CE+Dice；epoch21后原 LCTX 互补混合与来源计分，每个锚定 L 像素只计分一次，无额外干净 L 监督。

## 4. 响应查询（response probes）

### 4.1 固定输入和时间点

所有响应导数都在当步更新前的同一参数 theta_k 上计算。不能在原 Adam 已修改参数后才计算 J，也不能使用上一批/上一epoch的旧 J。

U臂在epoch21后建立当前域 train_unlabeled accessor。沿用父单域的U顺序和几何增强独立流，获得当前真正的U batch（B=2，尾批允许B=1），只含image/geometry。不要为凑两张而复制尾批U。

L臂复用当前已读取、已做几何增强的锚定L图像（未做LCTX混合版本），只传image/geometry给查询函数，明确剥离label。没有新增标签损失，没有额外L文件读取。

查询不使用 CIST 外观增强，不采用 max-confidence 阈值，不按当前预测类别挑像素/患者。

### 4.2 对比logit场

以当前普通学生最终384×384 logits z_theta(x)为基础。采用固定正交类别对比矩阵：

```text
C[0] = (-1, 1, 0) / sqrt(2)
C[1] = (-1, -1, 2) / sqrt(6)
```

C的两行张成三个logit去掉共同偏移后的空间。这里没有宣称类别对比矩阵是新的语义先验；它只是避免把共同logit平移当成预测变化。

使用几何有效掩码做32×32 masked adaptive average pooling：
- z_pool = adaptive_avg_pool(z*valid,32) / clamp_min(adaptive_avg_pool(valid,32),1e-12)。
- pooled_valid = pooled_valid_fraction > 0；无有效像素的单元不计。
- chi = C @ z_pool。
- 对每个原图先独立pool，不能把患者拼接后pool。

每批固定m=2个响应探针。每个探针生成与chi形状一致的独立Rademacher符号xi，使用新独立keyed stream：
`('DPR_V0_1', seed, target_domain, epoch, step, probe_index)`。
不得含arm名，不消耗已有标签顺序/几何/噪声RNG。

phi_j(theta) = sum(xi_j * chi_theta * pooled_valid) / sqrt(2 * pooled_valid.sum())。

对14层可训练权重计算r_j = grad_theta phi_j，使用`autograd.grad(create_graph=False)`，最后detach。这不是目标损失的第二次backward，不得把r_j累加进监督.grad。

R由两行r_j组成；令J=R/||R||_F。若||R||_F<=1e-30，按明示退化策略直接保留原Adam提案，记录zero_probe。不为不同臂/seed调归一化规则。

这一J是两个随机响应线性函数的归一化Jacobian sketch，不是完整Fisher、旧域Jacobian或全像素保真证据。m=2是计算近似，不是模型低秩rank。

## 5. 参数增量的闭式修正

令g=grad_theta L_LCTX，d0为原Adam一步得到的实际参数增量theta_Adam−theta_k。必须修正实际增量d0，不能只处理原始梯度后直接声称Adam也遵守同一关系。

主候选与L查询臂求解：

min_d 1/2 ||d-d0||² + lambda/2 ||Jd||²
subject to g^T d = g^T d0。

设e=g/||g||，A=J−(Je)e^T（即JP，P=I−ee^T）。实现时不显式构造参数维方阵P。

```text
S = I_m + lambda * (A @ A.T)
v = solve(S, J @ d0)
d_star = d0 - lambda * A.T @ v
```

无进展约束臂设A=J，其他完全相同。

lambda(epoch)=min(1,max(0,(epoch-20)/20))。epoch20及之前必须走真正恒等分支：不读U、不算J、不重复回写权重，保持原Adam/F_CONV行为。无lambda网格。

若||g||<=1e-30，三臂均保留d0并记录zero_task_gradient；这是预定执行策略。NaN/Inf不属于退化策略，必须工程停止。

在实数算术下主候选有：
1. g^T d_star = g^T d0；若原提案有一阶下降，则保留相同的监督一阶下降量。
2. 因d0是可行解，最优目标不大于d0，故||Jd_star||<=||Jd0||。
3. 仅说明所采样当前批响应的一阶变化不增大，不保证真实有限步L损失下降、旧域不忘、所有像素不变或新旧域指标全面改善。
4. g^T d0可能因Adam动量/weight decay为非负；如实记录比例，不能把此时的等式说成保证下降。
5. 因||J||_F=1且0<=lambda<=1，目标比较还给出||d_star-d0||<=sqrt(lambda)*||Jd0||<=||d0||；这是一阶优化子问题的校正量界，不是整网安全保证。

所有内积、m×m Gram/solve及矫正公式使用FP64临时量，最终训练参数仍为FP32。不改前向或原Adam精度。不通过求解过程反传。

保留原Adam更新后的m/v和step计数，不把它们投影、重置或缩放。此方法应描述为“Adam提案后的约束修正”，不是声称与Adam或自然梯度严格等价。下一步在实际修正参数上重新计算监督梯度。

沿用当前EMA时，在实际参数修正完成后才更新EMA；EMA不得读取raw-Adam临时参数。各新臂规则一致。

## 6. 三个新臂与必要比较

| arm | 响应输入 | 监督一阶进展等式 | 作用 |
|---|---|---|---|
| DPR_U | 当前U | 有 | 唯一主候选 |
| RESPONSE_U | 当前U | 无 | 检验显式保留监督进展的作用 |
| DPR_L | 当前L图像，查询不读标签 | 有 | 检验额外U图像而非算法形式的作用 |

主比较 DPR_U−F_CONV；F_FULL作第二参照。
组件比较 DPR_U−RESPONSE_U 与 DPR_U−DPR_L。
不加入CIST、软伪标签、源教师、额外原型、源回拉、另一个保护矩阵或随机基底臂。若完整方法有价值后需要更细机制消融，另行固定，当前不扩大矩阵。

## 7. 任务与预算

split0；优化种子61/62/63。源入口均为已绑定的SRC_CE，不从目标适配模型初始化。

- O1: RIM_ONE_r3 -> Drishti_GS；21步/epoch，100epoch，2100步。
- O2: Drishti_GS -> RIM_ONE_r3；32步/epoch，100epoch，3200步。
- 每臂3*(2100+3200)=15900步。
- 三臂18个新目标任务、47700次正式优化器更新。
- 六个源模型与旧F_CONV/F_FULL成绩复用，无正常源重训成本。

正常活动响应步骤共0.8*47700=38160。每步两个VJP，因此38160*2=76320次额外响应VJP；另有47700次监督backward。报告VJP、监督backward和optimizer updates为不同计数，不把76320次VJP算成76320次训练更新。

U文件读取：在冻结的标准池大小/每epoch一次U遍历成立时，每个U臂3*80*(41+63)=24960张图像访问，两U臂合计49920；以真实accessor计数为准，不把预计计数冒充实际完成。DPR_L的额外U读取为0，查询复用L张量但前向与VJP仍要计成本。

相同optimizer steps不等于相同FLOPs或时间。此方法比普通F_CONV多当前数据前向和两次VJP，训练未必更快；部署没有附加模块开销。共享GPU4/5/6/7授权沿用，实时检查显存，不杀其他进程。

## 8. 一次性工程资格与计算可验证性

先做必要资格，禁止反复来源搜索或大规模资格训练。本附件附带CPU NumPy FP64公式检查，但它不能代替真实PyTorch/CUDA集成验证。

需覆盖：
- 闭式解与小维稠密KKT解一致，g等式与目标非增条件成立。
- J=0/lambda=0精确恒等分支；g=0按预定策略处理；J仅平行g时主方法不能虚构可用修正。
- response VJP确实非零，模型.grad在VJP前后与监督g一致，冻结参数仍无梯度更新。
- 同一输入同一key时L/U查询实现一致；U标签路径被拒绝。
- lambda=0时训练学生/Adam/EMA及原随机流与父F_CONV逐步一致。
- 真正FP32参数增量与FP64代数解分开记录，不用对小更新相对残差的单个1e-5检查重演旧问题。约束残差是否在FP32回写误差包络内，以独立计算的舍入预算判定。
- checkpoint保存14层实际参数、Adam、当前EMA（若保留）、RNG、步数、源身份和输入顺序状态。梯度/J及当前U图像不进入历史库。
- 精确恢复中必须知道当前步是raw proposal还是corrected committed；不得在raw Adam状态上继续训练。先保存post-correction/pending checkpoint，再运行可能失败的诊断。
- 仅最多12次丢弃的真实L smoke；资格/失败额外更新单列。不读取新评价GT调工程超参。

完成必要资格后完成三臂全矩阵。科学主结果不作为删臂或延迟任务的门槛。数学错误、数据访问错误、NaN或恢复错误仍可工程停止。

## 9. 机制记录：必须区分一阶代理和实际输出

每步记录：loss及CE/Dice、lambda、||g||、||d0||、||d_star-d0||/||d0||、g·d0、g·d_star、||Jd0||、||Jd_star||、小矩阵条件数、零退化计数和计算操作计数。

每个目标任务在lambda首次达到1、以及之后每500步（预先确定索引）的当前训练批上做一次只读诊断：比较raw proposal和corrected proposal相对theta_k的实际对比logit变化、同批监督损失变化与一阶预测。只用已读当前L/U张量，不新增旧域/val/test读取，不用结果选择接受/拒绝或调lambda。

诊断临时权重切换必须恢复最终corrected参数，保护optimizer/EMA/RNG不发生额外更新。所有额外forward计入实际成本。如果工程上采用独立同源临时执行策略，必须明确模型/权重副本峰值，不能隐瞒。

不要将||Jd||减少当作“旧域保持成功”。最终性能只能由预定阶段末学生评价决定。

## 10. 效能与推断

唯一主指标仍为Final=(Incoming+Old)/2，先按患者在各物理域内平均、再两个顺序在每seed内等权、最后三seed等权。主比较DPR_U−F_CONV。

平均Final>=+0.005：优先复核信号；0<mean<0.005：小幅正面信号；mean<=0：NO_PRIMARY_ACCURACY_GAIN。不要求Old/Incoming/所有类别/所有seed全部同时提高。

患者配对区间与训练seed差异分别报告。bootstrap沿用物理域患者共享权重、所有方法/阶段/seed/order配对，2000次，分析seed2026091204。区间条件于已训练模型和重复开发患者；不称为独立患者验证。仅3优化seed的区间/SD不要过度推断总体随机性。

组件决定解释，不强迫主效能与所有组件比较一起PASS：
- DPR_U不超过DPR_L：不能主张额外U具有已建立的净价值。
- DPR_U不超过RESPONSE_U：不能主张监督进展等式具有已建立的实证增量。
- 任一对照更好：完整报告，不把对照重命名为预指定主候选成功。
- 一阶代理减少但实际输出/评价不改善：说明局部代理不足，不能只展示代数性质宣布成功。

保持所有old/current/class/seed代价、实际训练成本和部署参数量。若整体正面，下一阶段才用新优化seed和新的合法确认资源复核；本阶段不自动扩展。

## 11. 研究定位和局限

GOLD附件第3–5页及附录B讨论读出相关方向和AGOP；它不证明本方案，也不证明当前U可以代表旧域。它的原型记忆、动态矩阵等与本方案不同。

OGD、GPM、SGP等已经研究输出/梯度空间约束；因此不能称“梯度投影是新创新”。本候选待检验的差异是：不存历史梯度/激活基，用当前无标注响应查询当步参数，实现软函数扰动控制，并显式保留监督一阶方向进展，而非固定禁止一大块参数空间。完整原创性尚需与相关原始工作系统比较，本阶段不声称SOTA/首创。

最大不确定性：查询仅覆盖当前训练分布、线性近似有限、m=2只提供低成本随机sketch、Adam非欧氏状态与修正未做全局优化分析、每步局部稳定不能推出长序列稳定。潜在收益可能是目标泛化而非历史保持。应让Final与分项结果回答，而非先给方法贴上“记忆保护成功”的标签。

## 12. Codex执行摘要

读取最新CIST plasticity报告与本协议。停止CIST扩展，不改旧终态。实现DPR_U、RESPONSE_U、DPR_L三个普通F_CONV臂，复用真实六份SRC_CE和已有基线。不再找旧KI。

关键：在theta_k计算监督g和当前输入响应J；原Adam产生实际d0；通过2×2系统求d_star；把实际参数设为corrected值后再更新当前EMA。不要只投影梯度。不要让U VJP污染监督.grad，不储存历史方向。

先完成公式/PyTorch/CUDA与恢复资格，再完整运行18任务47700步；将76320次预计额外VJP与正式更新分开计数。只按预指定主要Final判断总体效果，允许新旧域权衡。

交付至少：PROTOCOL、SOURCE_AND_BASELINE_BINDING、TASK_MATRIX、QUALIFICATION、RUN_LEDGER、TRAJECTORIES、PAIRED_EFFECTS、PATIENT_INTERVALS、ACTUAL_STEP_RESPONSE、RESOURCE_ACCOUNTING、FINAL_REPORT及NAS归档证据。任务完成后停止，不自动加CIST、伪标签、第三域、seed或超参网格。

## 13. 可核查参考

项目文件均以第1节不可变提交为准。外部原始来源仅作方法背景，不能替代项目结果：

```text
GOLD: user-provided Lai et al. (2026), The Golden Subspace..., §3–4 and Appendix B.
OGD: https://proceedings.mlr.press/v108/farajtabar20a.html
GPM: https://arxiv.org/abs/2103.09762
SGP: https://ojs.aaai.org/index.php/AAAI/article/view/26157
```
