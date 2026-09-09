# Anchored Mix Stability V0.1 最终报告

**MIX_CED 在固定患者划分上取得开发收益，并在两个预定新优化种子上复核通过。** P2 相对 SUP_CE 平均 Dice +0.036492，相对 SUP_CED +0.046829；两种子均正。相对同输入 MIX_CTX 的伪目标增量为 +0.006228，达到预定 +0.003 幅度，但患者 bootstrap 区间包含 0。主要收益已经出现在混合上下文对照中，不能把全部提高归因于伪标签。

**终态：VALUE_REPRODUCED；DIRECTION_REPRODUCED_ON_TWO_NEW_SEEDS；ENGINEERING_COMPLETE。** P1 24任务/63600步，按冻结规则选择 MIX_CED 后，P2 16任务/42400步全部完成。共40任务、106000次正式更新、200个固定诊断、40次单学生部署、80个成功子进程退出。存在 seed41/Drishti cup 相对 SUP_CE −0.021771 的 amber 类别代价。未启动额外实验或 CL。

## 五个比较与发现/复核分离

Q 为域内每患者 rim/cup 宏平均 Dice，再将两域等权平均；固定 epoch100 学生。SUP_CE 的 P1 数值来自冻结旧 SUP 评分，未重训；P2 的 SUP_CE 为新训练的匹配对照。

| 配方 | P1三seed平均Q | P1 Q sample SD | P2两seed平均Q | P2 Q sample SD |
| --- | --- | --- | --- | --- |
| SUP_CE | 0.689790 | 0.016348 | 0.697217 | 0.005503 |
| SUP_CED | 0.693116 | 0.003860 | 0.686880 | 0.020539 |
| MT_CED | 0.687631 | 0.009092 | 未准入/未训练 | 未准入/未训练 |
| MIX_CTX | 0.723997 | 0.005237 | 0.727481 | 0.007733 |
| MIX_CED | 0.729633 | 0.003494 | 0.733709 | 0.011782 |

| 阶段 | 比较 | 平均配对差 | median | 配对sample SD | 范围 | 正/负seed数 |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | SUP_CED − SUP_CE | +0.003326 | +0.002934 | 0.018555 | [-0.015030, +0.022075] | 2/1 |
| P1 | MT_CED − SUP_CED | -0.005485 | -0.008056 | 0.005485 | [-0.009213, +0.000813] | 1/2 |
| P1 | MIX_CTX − SUP_CED | +0.030880 | +0.032474 | 0.009098 | [+0.021091, +0.039076] | 3/0 |
| P1 | MIX_CED − MIX_CTX | +0.005636 | +0.003463 | 0.006264 | [+0.000748, +0.012697] | 3/0 |
| P1 | MIX_CED − MT_CED | +0.042002 | +0.042435 | 0.008818 | [+0.032976, +0.050596] | 3/0 |
| P1 | MIX_CED − SUP_CED | +0.036517 | +0.033788 | 0.005223 | [+0.033222, +0.042539] | 3/0 |
| P1 | MIX_CED − SUP_CE | +0.039843 | +0.045473 | 0.019456 | [+0.018192, +0.055863] | 3/0 |
| P2 | SUP_CED − SUP_CE | -0.010337 | -0.010337 | 0.015036 | [-0.020970, +0.000295] | 1/1 |
| P2 | MIX_CED − SUP_CE | +0.036492 | +0.036492 | 0.006279 | [+0.032051, +0.040932] | 2/0 |
| P2 | MIX_CED − SUP_CED | +0.046829 | +0.046829 | 0.008757 | [+0.040637, +0.053021] | 2/0 |
| P2 | MIX_CED − MIX_CTX | +0.006228 | +0.006228 | 0.004049 | [+0.003365, +0.009091] | 2/0 |

**真实监督基础：**CE+Dice 相对 CE，P1 +0.003326（2/3正），P2 −0.010337（1/2正），五seed描述性均值 −0.002139。因此不能声称 Dice 单独稳定改善监督基础或降低方差。P1 SUP_CED 的 Q SD 较低，但 P2 升到0.020539。

**普通 MT：**MT_CED 相对 SUP_CED 为 −0.005485（仅1/3正），且 seed33/Drishti rim 下降0.072761，未准入。**混合上下文：**MIX_CTX 相对 SUP_CED 的 P1 增益 +0.030880，已覆盖 MIX_CED 总增益中的大部分。**伪目标增量：**MIX_CED−MIX_CTX 在发现三seed及复核两seed均为正，但增量幅度小于上下文对照收益，且患者不确定性尚未消除。

## 每个种子的主候选配对差

| 阶段 | seed | MIX_CED Q | ΔSUP_CE | ΔSUP_CED | ΔMIX_CTX |
| --- | --- | --- | --- | --- | --- |
| P1 | 31 | 0.725651 | +0.018192 | +0.033222 | +0.000748 |
| P1 | 32 | 0.731063 | +0.055863 | +0.033788 | +0.012697 |
| P1 | 33 | 0.732186 | +0.045473 | +0.042539 | +0.003463 |
| P2 | 41 | 0.725377 | +0.032051 | +0.053021 | +0.003365 |
| P2 | 42 | 0.742040 | +0.040932 | +0.040637 | +0.009091 |

全部五seed为经过P1选择后的描述性汇总，不代替独立列出的新seed P2结果。五seed MIX_CED−SUP_CE 平均 +0.038502，配对SD0.014230；−SUP_CED 平均 +0.040642，配对SD0.008045；−MIX_CTX 平均 +0.005873，配对SD0.004881。完整 Q 的 median/SD/range 与配对差分别见 RECIPE_SEED_SUMMARY.csv 和 PAIRED_EFFECTS.csv。

## 冻结门槛与分项状态

P1：MIX_CED 的平均 ΔSUP_CED +0.036517、3/3正，超过较强监督参照 +0.036517，每域/类别均值与单次严重退化保护线通过，ΔMIX_CTX +0.005636≥0.003。MT_CED不合格，因此按规则选择MIX_CED。全部候选原始门槛布尔值和逐行成本保存在 P1_SELECTION.json。

P2：相对 SUP_CE/SUP_CED 的均值分别 +0.036492/+0.046829，均超过 +0.010；相对两种基线的每域均值≥−0.005，每域/类别均值≥−0.020，单seed类别差均≥−0.050。ΔMIX_CTX +0.006228≥0.003。两个新seed相对SUP_CED均正。门槛按原协议判定，没有用bootstrap或五seed混合结果修改。

| 分项 | 状态 | 边界 |
| --- | --- | --- |
| MEAN_VALUE | VALUE_REPRODUCED | 新seed分别战胜两种监督基线 |
| PAIRED_SEED_VARIATION | DIRECTION_REPRODUCED_ON_TWO_NEW_SEEDS | 仅两个新优化seed，不证明普遍低方差 |
| DOMAIN_CLASS_COST | MEAN_GUARDS_PASSED_WITH_AMBER | 保留seed41 Drishti cup下降 |
| PSEUDO_TARGET_INCREMENT | PRACTICAL_MARGIN_MET_COHORT_UNCERTAIN | P1/P2患者区间均跨0 |
| REPLICATION_STATUS | FIXED_SPLIT_DEVELOPMENT_REPLICATION | 不是独立患者、临床安全或CL确认 |

## 类别代价

| seed | 域 | 基线 | macro差 | rim差 | cup差 |
| --- | --- | --- | --- | --- | --- |
| 41 | RIM_ONE_r3 | SUP_CE | +0.026743 | +0.009780 | +0.043706 |
| 41 | RIM_ONE_r3 | SUP_CED | +0.045715 | +0.045487 | +0.045942 |
| 41 | Drishti_GS | SUP_CE | +0.037360 | +0.096491 | -0.021771 |
| 41 | Drishti_GS | SUP_CED | +0.060328 | +0.110182 | +0.010473 |
| 42 | RIM_ONE_r3 | SUP_CE | +0.026918 | +0.001358 | +0.052479 |
| 42 | RIM_ONE_r3 | SUP_CED | +0.037500 | +0.021973 | +0.053028 |
| 42 | Drishti_GS | SUP_CE | +0.054945 | +0.109952 | -0.000061 |
| 42 | Drishti_GS | SUP_CED | +0.043774 | +0.076401 | +0.011147 |

P2 相对 SUP_CE 的 Drishti cup 平均下降 −0.010916；seed41 为 −0.021771，标记 amber；seed42 为 −0.000061。平均保护线通过不等于所有类别都改善或临床安全。P1主候选相对SUP_CED的最差类别为seed31 Drishti cup −0.003280。

| 阶段 | 配方 | 基线 | seed | 域 | 类别 | 所有amber差值 |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | SUP_CED | SUP_CE | 31 | RIM_ONE_r3 | cup_dice | -0.033092 |
| P1 | SUP_CED | SUP_CE | 33 | Drishti_GS | cup_dice | -0.025672 |
| P1 | MT_CED | SUP_CED | 33 | Drishti_GS | rim_dice | -0.072761 |
| P2 | SUP_CED | SUP_CE | 41 | RIM_ONE_r3 | rim_dice | -0.035707 |
| P2 | SUP_CED | SUP_CE | 41 | Drishti_GS | cup_dice | -0.032245 |
| P2 | SUP_CED | SUP_CE | 42 | RIM_ONE_r3 | rim_dice | -0.020615 |
| P2 | MIX_CED | SUP_CE | 41 | Drishti_GS | cup_dice | -0.021771 |

## 不确定性与晚期波动

下表为预定2000次配对bootstrap、分析seed2026090807的95%百分位区间。患者在每域内重采样，所有配方/训练seed共享患者权重；训练seed则整块重采样配对轨迹。两个域固定等权，不把噪声draw或像素当独立样本。

| 阶段 | 比较 | 重采样 | 95%区间 | 含0 |
| --- | --- | --- | --- | --- |
| P1 | MIX_CED − MIX_CTX | patient | [-0.001647, +0.013900] | True |
| P1 | MIX_CED − MT_CED | patient | [+0.015923, +0.068498] | False |
| P1 | MIX_CED − SUP_CED | patient | [+0.010653, +0.061821] | False |
| P1 | MIX_CED − SUP_CE | patient | [+0.018249, +0.063092] | False |
| P1 | MIX_CED − MIX_CTX | training_seed | [+0.000748, +0.012697] | False |
| P1 | MIX_CED − MT_CED | training_seed | [+0.032976, +0.050596] | False |
| P1 | MIX_CED − SUP_CED | training_seed | [+0.033222, +0.042539] | False |
| P1 | MIX_CED − SUP_CE | training_seed | [+0.018192, +0.055863] | False |
| P2 | MIX_CED − SUP_CE | patient | [+0.008087, +0.067306] | False |
| P2 | MIX_CED − SUP_CED | patient | [+0.020241, +0.076805] | False |
| P2 | MIX_CED − MIX_CTX | patient | [-0.001576, +0.014222] | True |
| P2 | MIX_CED − SUP_CE | training_seed | [+0.032051, +0.040932] | False |
| P2 | MIX_CED − SUP_CED | training_seed | [+0.040637, +0.053021] | False |
| P2 | MIX_CED − MIX_CTX | training_seed | [+0.003365, +0.009091] | False |

P2 相对两种监督基线的患者区间均在0以上；相对CTX的区间 [−0.001576,+0.014222] 含0。因此伪目标组件具有预定幅度和方向信号，但不能宣称其独立患者效应已经确定。仅2/3个训练seed的bootstrap是粗略敏感性，不能声称精确覆盖或显著性。

LATE_EPOCH_VARIATION.csv 独立列出每条轨迹的60/80/100阶段波动；旧参考见P0_LATE_VARIATION.csv。这些epoch不构成独立重训练，不挑最佳阶段。P2 MIX_CED Q的seed SD0.011782高于SUP_CE的0.005503，即使配对增益两次都正，也不称为全面降低原始分数方差。

## 机制与计算边界

详见 MECHANISM_SUMMARY.md 和 MECHANISM_METRICS.csv。MIX_CTX与MIX_CED的学生输入及L来源计分相同；CTX仍使用U图像上下文，不能称纯监督。相同原L像素在互补两视图中恰好监督一次；没有额外干净L损失。单张末尾U在两个上下文中的损失分别权重0.5，再按唯一U接受支持归一化；不平均logits，不把重复U计成独立患者。GT-Dice仅作用于真实标签，未给伪标签加Dice。

## 执行、内存与来源

冻结源码 `800cabe5e5dd69612738aaba45ad35121d63c0f2`，本次只读重验服务器checkout干净，40任务/200诊断/40部署通过。GPU3/4/5后台总耗时 160.96 分钟。正式更新106000；历史正式累计354108。426次合成更新及8次smoke单列，本轮包含资格/smoke的成功更新合计106434。旧资格计数缺项没有填0或重跑。

| 实际操作 | 正式矩阵(含诊断/部署) | 资格+smoke后本轮全部 |
| --- | --- | --- |
| optimizer_steps | 106000 | 106434 |
| backward | 106000 | 106434 |
| ema_updates | 106000 | 106434 |
| model_forward | 261340 | 262342 |
| image_forward | 454620 | 456096 |
| sample_access | 334460 | 335404 |
| hdf5_open | 552960 | 554830 |

正式训练L样本访问212000次，U图像访问108160次；诊断image-only6500次及含GT的val样本6500次；部署原图1300次且新增GT读取0。上述是记录张次/loader调用，不能称为同等数量的唯一患者。每域唯一训练人群保持RIM L16/U63、Drishti L10/U41，val40/25。SUP不读训练U；CTX不做teacher-U前向。SUPERVISION_SOURCE_ACCOUNTING.csv逐epoch逐类记录真实曝光和U接受数；global loss/record数在三类行重复，不能跨类别重复求和。

每任务最多student+当前EMA两份完整模型，最终仅部署学生；无历史教师、原型或CPU full shadow。每模型参数1936064 bytes、Adam3865412 bytes，惰性sigma/grad_update各1728 bytes不入Adam。实际峰值Torch allocated 891.082 MiB，reserved 1020.000 MiB；不把两模型数解释为全部显存仅两倍参数。所有200诊断保持学生/EMA/Adam/RNG状态，真实诊断额外backward=0。操作计数含尝试与成功两列；合成中有故意缺路径断言，不是正式训练失败。

## 停止与交付

配方按本次冻结定义保留。仅准备 NEXT_STAGE_DRAFT.md，不开展新患者划分、标签预算、更多seed、CL或外部测试。同一split多轮暴露，优化seed复核不是独立患者验证；此结果不自动构成新颖性、SOTA、BCP完整复现或临床安全主张。

源代码、协议、全部聚合指标、区间、成本、测试及退出证据公开；图像、GT、患者ID、逐病例预测和权重留NAS。EXECUTOR_FINAL_REPORT.md保留运行器原报告；原P1/P2选择与终态不改。发布提交、匿名可读性和NAS归档见PUBLICATION_VERIFICATION.json。运行本目录build_final_report.py仅重建聚合叙述并交叉检查计数及配对差，无数据/模型访问。
