# SSL_CL 论文收敛与最终对照包 V1

日期：2026-09-13
状态：PROPOSED_NOT_EXECUTED。本文是新的实验与写作方案，不是训练完成报告。
建议执行代号：LCTX_PAPER_CLOSEOUT_V1。

## 1. 决策与研究范围

停止 DPR/SIGN、CIST 和源权重 SVD 保护的新增调参或变体，不修改它们的历史终态。SIGN 新优化种子 Final 差为 -0.0001835866，条件患者区间跨零，不将其宣布为有效、无效已被证明、等效或非劣效。

转向已有 LCTX 正面开发发现的贡献归属、标注敏感性与论文初稿。保留 F_CONV 的可训练范围，不把 F_CONV 或 LCTX 冒充未定位的原 KI，也不把仅使用 L 的方法称为建立了额外 U 增益。

这不是给 LCTX 改名后宣布新方法成功。下一组比较用于决定稿件应该主张“来源计分设计有额外价值”，还是主张“强混合基线与局部稳定性代理的实证分析”。不预设 LCTX 必须胜过新的混合对照。是否达到投稿期刊的原创性与验证广度要求，不能由训练任务完成或 gate 放宽自动推出。

本轮交付不是新的性能 PASS，而是：固定比较表、条件不确定性、成本表、证据—主张对应表和可继续编辑的论文初稿。无论分数方向如何，完成这些写作交付。

## 2. 已核实的依据

A. SIGN 复核：
- 提交 e2ac3943a61447572489b8772dee71cedaf65d6b。
- experiments/lcrseg/docs/dpr_sign_replication_v0_1/COMPLETION_REPORT.md。
- public_results/TRAJECTORIES.csv、MECHANISM_EQUAL_TASK.csv、ARM_SUMMARY.csv。
- 六个新源学生与 F_CONV 目标学生已存在。科学更新 47,700，迁移额外 15，实际物理更新 47,715；不得与下一轮重计。
- SIGN 额外 25,440 VJP、38,160 响应前向，不具有已建立的训练效率优势。

B. 原序列混合对照：
- 提交 671d0e57c8a2b56b5162933be4973ea0f538f58c。
- experiments/lcrseg/docs/ams_seq_transfer_v0_1/FINAL_REPORT.md。
- 全参数 T_LCTX 相对 T_CE / T_CED 的 Final 开发差为 +0.051042 / +0.055500。
- 这是全参数、旧种子、相同开发队列的结果，不是新的 F_CONV 空间内 LCTX 增量复核。

C. 数学实现：
- 在 e2ac3943... 的 experiments/lcrseg/ams_seq_transfer_v0_1/core.py 中，LCTX 使用当前 L batch 反转作为 donor，仅提供 donor 图像。
- experiments/lcrseg/ssl_anchored_mix_v0_1/core.py 定义互补混合、collect_sources、逐来源图像/前景类别 Dice、CE 和当前 EMA 更新。
- 不更换这些已存在的 LCTX 定义。

## 3. Gate 改为“有效性边界 + 报告规则”

### 必须停止并修复
- 错误源学生、未声明的额外目标训练、使用评价/隐藏 GT 指导训练。
- 非有限损失/梯度/权重、丢失动态参数的恢复、重复更新不计数。
- 不符合实际运行的指标或文件声明。

### 不再作为自动否决条件
- Final 未达到 +0.005。
- 某个种子/顺序/类别为负。
- 患者区间跨零或未达到某个 p 值。
- 所有消融没有全部为正。
- 日志的非关键字段缺失、历史报告别名不齐、旧源码目录没有 .git 等可明确记录的问题。

效应大小、区间、种子差、类别代价和成本分别报告。区间跨零不等于“没有效应”，也不等于等效。没有复现正增益就写没有复现，不能降低 gate 后改写为增益。

旧 gate 和旧终态原样保留。可以对旧 AMS 单列“平均有益但原合取条件未满足”的叙述，不能把旧结论修改成重新预注册的成功。

本轮不调参。以后允许正常、有界、同预算的开发集调参，不要求所有新设计只试一个任意超参；但必须将开发选择与确认评价分开。本条不是给 SIGN 追加搜索的授权。

## 4. 固定主比较：当前约20%目标域标注

沿用 split0，seeds71/72/73，六份本次新 SRC_CE。O1=RIM_ONE_r3→Drishti_GS，O2=Drishti_GS→RIM_ONE_r3。

四个配方均采用同一 F_CONV 更新范围：
1. C_CE：原图真实标签 CE，新训练。
2. C_CED：原图真实标签 CE+Dice，新训练。
3. C_FULLMIX：匹配输入的全来源监督 CutMix 式分割对照，新训练。
4. C_LCTX：原 F_CONV+LCTX，复用 SIGN 复核中的六份 F_CONV 目标结果。

C_CE/C_CED 必须是相同源学生、相同 F_CONV 参数集合的目标训练。不得拿旧全参数 CE 的分数代替。

所有新任务从共同 SRC_CE 开始，不能从已经适配的 SIGN、LCTX 或 CIST 权重开始。源分数不作为准入标准，不更换不利种子。

C_LCTX 是本次预指定待解释配方，不代表已经优于 C_FULLMIX。主要来源计分比较为 C_LCTX-C_FULLMIX，C_LCTX-C_CE/C_CED 为训练配方实用差。所有比较完整保留。

## 5. C_FULLMIX 的精确定义

它是匹配本项目输入与训练预算的全来源监督 CutMix 式适配，不是原分类 CutMix 或 BCP 的逐项复现。不能在论文中写成“官方 CutMix/BCP 实现”或作为它们的全面排名结论。

### 5.1 相同输入

当前 L batch 恰为2名不同患者。设锚定图像为 x_i，donor 是同 batch 的另一位患者 x_d；复制原 LCTX 的 donor noisy()、矩形随机流和两张互补视图：

x_i^a = M_i*x_i + (1-M_i)*noisy(x_d)
x_i^b = (1-M_i)*x_i + M_i*noisy(x_d)

384×384、原矩形规则、原 augmentation/key 不变。训练前20 epochs 为干净 L CE+Dice；第21起采用混合。图像读取和实际学生输入数量与 LCTX 相同。

### 5.2 全来源标签

y_i^a = where(M_i, y_i, y_d)
y_i^b = where(M_i, y_d, y_i)

使用标签整型 where，保留原255忽略语义；不要插值离散标签。调用一次原 supervised_parts，输入为两批 log-softmax 的拼接与两批混合标签拼接，得到 CE+Dice。这样以合成图像为 Dice 汇总单位，所有有效来源区域参与监督。

不叠加额外 clean-L loss，不再加来源重组 loss。

互补对使每个锚定/供体来源在两视图中完整出现；由于每个患者同时作为锚和供体，原始监督出现次数不同于 LCTX 的锚定一次。记录 raw multiplicity=2，合成批次平均的有效系数，不通过双倍加总把梯度尺度人为放大。存在忽略像素时应按原监督函数的全体有效像素分母和逐图 Dice 定义精确记录，不声称每个像素完全等权匹配。

### 5.3 LCTX 仍保持原定义

logp_i^source = where(M_i, logp_i^a, logp_i^b)
L_LCTX = supervised_parts(logp^source, y).CE + supervised_parts(...).Dice

LCTX 仅对锚来源计分，在两种语境中恢复同一来源的预测后算 Dice；不启用 donor 标签监督。

C_FULLMIX vs LCTX 是“全来源合成监督 vs 来源一致锚监督”的配方比较，同时涉及供体监督和 Dice 分组，不能将差异完全归因于某一个环节。Dice 的比值非线性说明两种分组一般不等价，但不证明某一组必然更好。

## 6. 固定低标注敏感性块：只比较两种混合

仍沿用同一源学生，不重新训练源域。只将目标域可用 L 减半：
- 当前 RIM L16 → L8。
- 当前 Drishti L10 → L5。

先核验 manifest 实际人数，按固定“取原 L 的一半，向下取整、至少2名”的规则建立嵌套子集。患者以 SHA256('LCTX_PAPER_CLOSEOUT_V1|domain|patient_id') 排序；每域所有种子与方法使用同一子集。选择不能使用任何患者分数、类别面积、模型表现或评价标签。公开子集 manifest 的哈希；患者身份留 NAS。

新训练 C_LCTX_LOW 与 C_FULLMIX_LOW，各六个目标任务。子集外标签不进入训练器，U 图像访问为0。可以保留在数据盘上，但不能由 loader 读取后再丢弃标签伪装隔离。

目标步数仍为 Drishti2100 / RIM3200；不按“完整U遍历一轮”增加新步数。报告 L 重复曝光量。

这项结果只能称为“固定源模型下的目标标注预算敏感性”，不能声称源与目标全序列均使用10%标注。没有低标注纯 CE/CED 新结果，不推测其分数。

两块参数与矩阵在第一次新增评价前冻结，不看主块结果后给低标注改mask或阈值。

## 7. 训练、部署、最小资格

- 优化器为已有 Adam：lr0.001，betas(0.9,0.999)，eps1e-8，weight_decay4e-5；poly power0.9；不要切换AdamW。
- 架构、分类头、padding与输出resize均原样；输入384×384，L batch2。
- F_CONV 14层权重：enc1/2/3.block.0和.3，bottleneck.block.0和.3，decoder.dec3/2/1.merge.block.0和.3。最终以真实参数枚举核验。冻结GN、转置卷积与读出等其余参数。
- 无响应VJP、DPR、Cayley、SVD保护、旧教师或历史样本；目标阶段只访问当前L。
- 保留父工程当前EMA更新以匹配目标学生训练语义，EMA不提供目标、不参与部署。不要声称训练时没有EMA。
- 源Adam/EMA不继承；各目标从相同源学生重置目标优化器。
- 源与目标权重、optimizer、RNG、计数和数据位置全部保存；沿用先保存post-update精确状态再诊断的恢复语义。
- 六份历史C_LCTX结果按已记录合法身份复用，不重新hash所有模型或重训同一prefix来证明存在。源绑定/部署不一致等实质问题才处理，不进行旧KI溯源。
- 必要测试：来源掩码/标签互补、CE/Dice归一化、忽略标签、冻结参数白名单、恢复一致性、同一输入生成器、来源隔离与零U访问。
- 允许至多128次合成资格更新、12次丢弃真实L smoke更新；与科学预算分开，不因合成测试没有模拟出每种现象另起大任务。
- 模型阶段末只部署epoch100学生；不增加epoch选择/验证驱动早停。

## 8. 预算

主比较：3个新配方×3seeds×(2100+3200)=47,700，18个目标任务。
低标注：2个新配方×3seeds×(2100+3200)=31,800，12个目标任务。
合计：30个新目标任务，79,500次正式optimizer更新，源训练0，基线重训0，额外响应VJP0。

复用六份20% LCTX目标学生不记作新任务。旧SIGN的47,700和迁移15步不重计。前向、EMA、资格、失败/恢复额外工作分别计数。不能根据更新数承诺墙钟时间。

GPU4/5/6/7按既有共享授权、当前显存调度；GPU6/7已释放不是占用独占权。不得停止他人进程。没有新的可用GPU时清楚说明队列状态，不写已经完成。

## 9. 评价与解释

主指标：Final=(Incoming+Old)/2；种子内等权平均两个顺序，再跨三种子平均。两个预算单列，不能平均后隐藏预算差。

新主比较为 C_LCTX-C_FULLMIX；同时完整报告 C_LCTX-C_CE、C_LCTX-C_CED、C_FULLMIX-C_CED、低标注两臂差。

保留每域rim/cup、绝对遗忘、配对seed SD、患者配对区间及实际成本。患者bootstrap按物理域共享权重跨所有方法/种子/顺序/阶段；建议沿用2000次。区间条件于已训练模型和反复开发患者，不包含独立患者或完整训练随机性。

不作“所有方向全优”筛选，不以+0.005或区间排零作为能否写论文的门槛；所有结果无论方向都进入报告。若有多重探索/新主张，明确标为开发分析。旧结果选择LCTX和本次选题的时间线单列。

- LCTX胜过无混合但不胜过FULLMIX：主要证据是强混合配方，不声称来源锚定独特优势。
- LCTX胜过FULLMIX：支持当前来源计分设计的局部实用增量；仍不直接证明超越全部混合/持续学习方法。
- 低标注结果不一致：明确限定预算依赖，不追加5%、15%网格救结果。
- 整体没有增益：完成经验分析稿，不假装方法成功，不再自动开启新模块搜索。

## 10. 与训练并行的写作交付

现在就写，不等待新均值变正。建议输出 Markdown + 可用LaTeX源（非必须联网模板），不需要先做完整排版PDF。

1. EVIDENCE_CLAIM_MATRIX：主张、支持文件、适用设置、反例、不能主张的内容。
2. PAPER_DRAFT：Introduction、Problem/Protocol、LCTX既有定义、Methods being compared、Results、Discussion、Limitations。
3. TABLES：历史训练配方表、匹配FULLMIX新比较、标注敏感性、SIG探索/复核分离表、资源表。
4. EXPERIMENT_INDEX：每行的源提交、模型初始化、训练空间、L/U、数据split、seed、用途，标明复用。

不要把不同源初始化、不同训练空间或单域SSL与CL的数值混成一个“所有方法排行榜”。不要将F_CONV命名为原KI。没有未标注增量证据，不将“label-efficient”偷偷写成“validated novel semi-supervised learning”。

可选工作标题（不是最终主张）：
- 方法差异成立：Source-Consistent Context Supervision for Label-Efficient Continual Segmentation。
- 差异未成立：Strong Mixing Baselines and the Limits of Local Stability Proxies in Replay-Free Segmentation。

所有创新和跨数据集推广语句在有相应证据前标记待定。TMI方法论文的原创性与影响力不能仅由这个包保证。

## 11. 还缺的投稿证据，不阻塞已知任务与初稿

本包主要解决归因和目标标注敏感性，尚不是完整的通用多域TMI验证。

优先检查有无真正未参与训练、未标注输入、初始化或模型选择的患者评价集。若存在，可另冻结评价范围，以封存后的模型评价；不要用它再选择配方。若没有，就不称为独立患者验证，也不无限寻找而阻塞本包。

若论文声称三域以上的持续保持，应新增真实第三训练域的完整序列，不能用“第三域零样本测试”替代。若只做两域迁移，就按两域范围写结论。

若论文宣称优于持续学习SOTA，应增加同训练配方下的已发表方法对照并完整列历史状态成本；不以当前自设消融替代全部文献比较。本包不启动任何身份不明的方法恢复。

此类缺口进入 SUBMISSION_SCOPE.md：清楚区分“已有初稿”和“已经具备某期刊接收条件”。不虚构患者、第三域结果或比较方法成绩。

## 12. 停止条件与完整输出

完成30个新训练与固定评价后停止。保留负面结果，不调超参、不追加SIG种子、不自动启动新模块或第三域训练。报告错误修复不重训、不重新评价，保留原错误证据。

非关键NAS归档快照的旧字段差异可以补一份明确的新receipt，不覆盖旧receipt，也不据此重新跑科学矩阵。

最终回答：训练实际完成多少、每个比較真实结果、来源计分贡献是否支持、预算依赖、论文初稿路径、仍未支持的主张、剩余投稿证据。不能只返回一个PASS/FAIL，也不能宣称稿件已经发表或必然录用。

## 参考来源

项目报告：
https://github.com/DLwbm123/SSL_CL_seg/blob/e2ac3943a61447572489b8772dee71cedaf65d6b/experiments/lcrseg/docs/dpr_sign_replication_v0_1/COMPLETION_REPORT.md
https://github.com/DLwbm123/SSL_CL_seg/blob/671d0e57c8a2b56b5162933be4973ea0f538f58c/experiments/lcrseg/docs/ams_seq_transfer_v0_1/FINAL_REPORT.md

外部方法/评价背景（不是本项目实验结果）：
Yun et al. CutMix. ICCV 2019.
https://openaccess.thecvf.com/content_ICCV_2019/html/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.html
Bai et al. Bidirectional Copy-Paste. CVPR 2023.
https://openaccess.thecvf.com/content/CVPR2023/html/Bai_Bidirectional_Copy-Paste_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html
Cawley & Talbot. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. JMLR 2010.
https://www.jmlr.org/beta/papers/v11/cawley10a.html
IEEE TMI, Key Criteria for Publication（查询日期2026-09-13）.
https://ieeetmi.org/key-criteria-for-publication/
