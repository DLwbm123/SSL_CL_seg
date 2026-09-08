# SSL_TARGET_PATH_V0_1 最终报告

**最终学生没有建立稳定的 SSL 增益。** T_SCE 相对匹配 SUP 的三种子两域平均 Dice 差为 -0.001439，sample SD=0.019952；仅 seed32 改善，seed31/33 下降。最大类别代价为 seed33/Drishti cup −0.042400。仅教师筛选确实保留更多纠错机会，也接受更多教师错误；机制改善不能替代最终模型收益。

**科学终态：TARGET_PATH_SSL_NOT_ESTABLISHED。工程矩阵：ENGINEERING_COMPLETE。** 30 个完整任务、150 个固定诊断、30 个单学生部署及 60 个成功子进程退出已交叉核验。无追加训练、阈值调整、候选替换或 CL。本报告保留运行器原终态；合成资格的细粒度计数限制见后文。

## 最终学生结果

指标为各患者 rim/cup 宏平均 Dice，先域内患者平均，再两域等权；三种子预先指定。全部取 epoch100 学生，不使用 EMA 或最佳 epoch。表中 SD 为三次优化种子的 sample SD。

| 配方 | seed31 Q | seed32 Q | seed33 Q | 三种子平均 Q | Q sample SD | 平均 ΔSUP |
| --- | --- | --- | --- | --- | --- | --- |
| SUP | 0.707459 | 0.675200 | 0.686713 | 0.689790 | 0.016348 | +0.000000 |
| J_MSE | 0.701772 | 0.688955 | 0.672135 | 0.687621 | 0.014863 | -0.002170 |
| T_MSE | 0.701559 | 0.682390 | 0.676245 | 0.686731 | 0.013204 | -0.003059 |
| J_SCE | 0.703366 | 0.689511 | 0.671621 | 0.688166 | 0.015915 | -0.001624 |
| T_SCE | 0.704039 | 0.694631 | 0.666386 | 0.688352 | 0.019596 | -0.001439 |

完整 30 行、各域 rim/cup 见 [FINAL_METRICS.csv](FINAL_METRICS.csv)；所有阶段学生/EMA 见 [EPOCH_METRICS.csv](EPOCH_METRICS.csv)。

## 预定门槛逐项判定

| 门槛 | 实际 | 判定 |
| --- | --- | --- |
| 平均 ΔSUP ≥ +0.010 | -0.001439 | 未通过 |
| 每个 seed ΔSUP > 0 | 31: -0.003420, 32: +0.019430, 33: -0.020327 | 未通过 |
| 平均 ΔJ_MSE ≥ +0.005 | +0.000731 | 未通过 |
| 每 seed/domain Δmacro ≥ −0.005 | 最小 -0.032537 | 未通过 |
| 每 seed/domain/class ΔDice ≥ −0.020 | 最小 -0.042400 | 未通过 |
| 完整矩阵与隔离、部署检查 | 30/30；150/150；30/30 | 通过 |

T_SCE−SUP 的范围为 [−0.020327, +0.019430]；T_SCE−J_MSE 的平均差 +0.000731，sample SD=0.005865，范围 [−0.005749, +0.005676]。平均差不足，且跨种子不稳定。

## 域和类别代价

| seed | 域 | T_SCE−SUP macro | rim | cup |
| --- | --- | --- | --- | --- |
| 31 | RIM_ONE_r3 | +0.001699 | +0.000424 | +0.002973 |
| 31 | Drishti_GS | -0.008539 | -0.027973 | +0.010896 |
| 32 | RIM_ONE_r3 | +0.014289 | +0.010591 | +0.017988 |
| 32 | Drishti_GS | +0.024571 | +0.018316 | +0.030826 |
| 33 | RIM_ONE_r3 | -0.008116 | -0.012765 | -0.003468 |
| 33 | Drishti_GS | -0.032537 | -0.022675 | -0.042400 |

seed32 两域及两类均提高，不能表述为各处均无效；但 seed31 Drishti rim、seed33 Drishti rim/cup 超过预定类别损害上限。

## 两因素效应

下表为三种子平均的完整训练配方差，不是单像素因果效应。相同名义系数下，SCE 替换同时改变梯度形状与尺度，不能完全分离二者。

| 效应 | 定义 | RIM 平均 Δmacro | Drishti 平均 Δmacro | 两域平均 |
| --- | --- | --- | --- | --- |
| mask_effect_MSE | T_MSE−J_MSE | +0.002645 | -0.004424 | -0.000890 |
| mask_effect_SCE | T_SCE−J_SCE | +0.002025 | -0.001654 | +0.000185 |
| loss_effect_joint | J_SCE−J_MSE | +0.006554 | -0.005463 | +0.000546 |
| loss_effect_teacher | T_SCE−T_MSE | +0.005934 | -0.002693 | +0.001621 |
| interaction | (T_SCE−J_SCE)−(T_MSE−J_MSE) | -0.000620 | +0.002770 | +0.001075 |

在 RIM 上，MSE→SCE 在 J 和 T 掩码下三个种子均改善；Drishti 的 J 掩码下三个种子均下降。仅教师掩码的收益也依赖域和种子。所有消融的三种子两域平均 Q 均低于 SUP；不将消融替换为主候选。完整逐 seed/domain/rim/cup 的效应、sample SD 和范围见 [FACTORIAL_CONTRASTS.csv](FACTORIAL_CONTRASTS.csv)。

## 纠错机会与错误传播

下表固定查看最终 T_SCE 轨迹的同一组输出，同时计算 raw/T/J；没有为比较掩码重新前向。单位是平均像素/患者：先将四个 training-like draw 在患者内平均，再患者平均、最后三个种子等权平均。仅汇总互斥的 teacher_predicted_class 三类，不重复加上 true_class 分组。表内比率是平均计数之比，不是患者比率的平均；原始逐类患者比率与 undefined 支持另见 CSV。

| 视图 | 域 | 分层 | raw | T接受 | J接受 | T/raw | J/raw |
| --- | --- | --- | --- | --- | --- | --- | --- |
| posterior_mean_clean | RIM_ONE_r3 | teacher_correct_student_wrong | 49.625 | 0.192 | 0.008 | 0.39% | 0.02% |
| posterior_mean_clean | RIM_ONE_r3 | teacher_wrong_student_correct | 36.067 | 0.050 | 0.000 | 0.14% | 0.00% |
| posterior_mean_clean | RIM_ONE_r3 | both_wrong_same_class | 5897.492 | 5315.575 | 5282.258 | 90.13% | 89.57% |
| posterior_mean_clean | Drishti_GS | teacher_correct_student_wrong | 158.760 | 6.840 | 0.573 | 4.31% | 0.36% |
| posterior_mean_clean | Drishti_GS | teacher_wrong_student_correct | 81.920 | 0.240 | 0.000 | 0.29% | 0.00% |
| posterior_mean_clean | Drishti_GS | both_wrong_same_class | 8792.333 | 7537.400 | 7455.813 | 85.73% | 84.80% |
| training_like | RIM_ONE_r3 | teacher_correct_student_wrong | 422.990 | 206.071 | 102.306 | 48.72% | 24.19% |
| training_like | RIM_ONE_r3 | teacher_wrong_student_correct | 301.506 | 111.248 | 50.396 | 36.90% | 16.71% |
| training_like | RIM_ONE_r3 | both_wrong_same_class | 5595.585 | 5174.873 | 5001.581 | 92.48% | 89.38% |
| training_like | Drishti_GS | teacher_correct_student_wrong | 473.053 | 158.173 | 57.973 | 33.44% | 12.26% |
| training_like | Drishti_GS | teacher_wrong_student_correct | 719.990 | 271.847 | 126.340 | 37.76% | 17.55% |
| training_like | Drishti_GS | both_wrong_same_class | 8098.510 | 7237.290 | 6814.533 | 89.37% | 84.15% |

training-like 纠错机会保留率，RIM 从 J 的约 24.19% 增至 T 的 48.72%，Drishti 从 12.26% 增至 33.44%。教师置信度仍先拒绝约 51.28% / 66.56% 的原始纠错机会；学生置信度又拒绝 T 准入机会的约 50.35% / 63.35%。同时，“教师错、学生对”被接受数从 50.40→111.25（RIM）、126.34→271.85（Drishti）。扩大准入包含潜在有益和有害方向，不是正确性保证。

clean 纠错机会极少被 T 接受：RIM 0.192/49.625，Drishti 6.840/158.760。training-like 下大量共同错误仍被接受；共同错误不等于零梯度：RIM 同类共同错误中约 5580.58/5595.59、Drishti 8049.73/8098.51 的 p/q 分布不同。只有 p≈q（冻结 L1≤1e−6）才接近一致性固定点，不能把相同 argmax 当成相同分布。

## 局部梯度、实际训练贡献与概率质量

四种解析候选均在同一输出上计算。下表固定 T_SCE 轨迹、epoch100、training-like、true_class=rim；为三种子各自患者平均的等权平均。未加权范数只在相应候选接受像素上统计；weighted dot 是该分层逐像素与隔离 GT-CE 方向的内积和，再作患者平均。正内积表示局部一阶方向一致，负内积表示冲突，不代表全网络参数梯度或实际 Dice 因果。

| 域 | rim 分层 | 候选 | raw norm | weighted norm | weighted dot GT |
| --- | --- | --- | --- | --- | --- |
| RIM_ONE_r3 | teacher_correct_student_wrong | J_MSE | 0.507604 | 4.48241e-06 | 4.81645e-05 |
| RIM_ONE_r3 | teacher_correct_student_wrong | T_MSE | 0.458396 | 9.38256e-06 | 9.95442e-05 |
| RIM_ONE_r3 | teacher_correct_student_wrong | J_SCE | 0.780362 | 8.17078e-06 | 0.000113661 |
| RIM_ONE_r3 | teacher_correct_student_wrong | T_SCE | 0.536356 | 1.24927e-05 | 0.000167491 |
| RIM_ONE_r3 | teacher_wrong_student_correct | J_MSE | 0.502276 | 4.72382e-06 | -1.09299e-05 |
| RIM_ONE_r3 | teacher_wrong_student_correct | T_MSE | 0.457342 | 9.44229e-06 | -4.53562e-05 |
| RIM_ONE_r3 | teacher_wrong_student_correct | J_SCE | 0.772556 | 8.57685e-06 | -1.7716e-05 |
| RIM_ONE_r3 | teacher_wrong_student_correct | T_SCE | 0.541317 | 1.26508e-05 | -5.3708e-05 |
| Drishti_GS | teacher_correct_student_wrong | J_MSE | 0.505791 | 4.77791e-06 | 4.22867e-05 |
| Drishti_GS | teacher_correct_student_wrong | T_MSE | 0.458573 | 1.11637e-05 | 0.000108146 |
| Drishti_GS | teacher_correct_student_wrong | J_SCE | 0.756668 | 8.94928e-06 | 0.000145614 |
| Drishti_GS | teacher_correct_student_wrong | T_SCE | 0.540952 | 1.47523e-05 | 0.000214877 |
| Drishti_GS | teacher_wrong_student_correct | J_MSE | 0.4929 | 1.01818e-05 | -3.5822e-05 |
| Drishti_GS | teacher_wrong_student_correct | T_MSE | 0.462236 | 1.86117e-05 | -0.000138005 |
| Drishti_GS | teacher_wrong_student_correct | J_SCE | 0.815118 | 2.00442e-05 | -6.04294e-05 |
| Drishti_GS | teacher_wrong_student_correct | T_SCE | 0.590436 | 2.69811e-05 | -0.000168223 |

实际训练按教师预测类统计 accepted、raw/weighted loss、SCE/entropy/KL 和 local logit norm，逐 epoch 原值均在 CLASS_LOSS_AND_GRADIENT.csv（kind=actual_training）。下表汇总 T_SCE 的 epoch21–100，三种子合计；loss 为逐 batch 类贡献之和，norm 为各 batch 类局部范数之和，不是合并后的参数梯度。

| 域 | 教师类 0/bg 1/rim 2/cup | accepted | raw loss | weighted loss | SCE | entropy | KL | local norm sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RIM_ONE_r3 | 0 | 1.99794e+09 | 68.9081 | 28.1679 | 68.9081 | 56.9828 | 11.9254 | 0.143026 |
| RIM_ONE_r3 | 1 | 1.5682e+08 | 93.2475 | 38.0012 | 93.2475 | 77.2331 | 16.0144 | 0.181389 |
| RIM_ONE_r3 | 2 | 4.24056e+07 | 30.0291 | 12.5443 | 30.0291 | 25.9109 | 4.11818 | 0.0961633 |
| Drishti_GS | 0 | 1.19683e+09 | 59.9521 | 25.0026 | 59.9521 | 53.9159 | 6.03619 | 0.0804629 |
| Drishti_GS | 1 | 1.07471e+08 | 93.0432 | 38.7393 | 93.0432 | 83.336 | 9.70715 | 0.116838 |
| Drishti_GS | 2 | 1.12099e+08 | 47.8416 | 19.5285 | 47.8416 | 41.4406 | 6.40107 | 0.0885898 |

背景 accepted 数量不能用于断言背景参数梯度支配。原始 SCE 含目标熵；比较损失优化时需同时查看去熵 KL。

| 域 | 类 | mask | teacher precision | accepted-correct recall |
| --- | --- | --- | --- | --- |
| RIM_ONE_r3 | 0 | teacher | 0.987913 | 0.986036 |
| RIM_ONE_r3 | 0 | joint | 0.988450 | 0.984767 |
| RIM_ONE_r3 | 1 | teacher | 0.719294 | 0.738072 |
| RIM_ONE_r3 | 1 | joint | 0.722946 | 0.726574 |
| RIM_ONE_r3 | 2 | teacher | 0.780075 | 0.626613 |
| RIM_ONE_r3 | 2 | joint | 0.785490 | 0.610649 |
| Drishti_GS | 0 | teacher | 0.985837 | 0.978460 |
| Drishti_GS | 0 | joint | 0.987107 | 0.977110 |
| Drishti_GS | 1 | teacher | 0.722741 | 0.570881 |
| Drishti_GS | 1 | joint | 0.733113 | 0.555734 |
| Drishti_GS | 2 | teacher | 0.735480 | 0.847794 |
| Drishti_GS | 2 | joint | 0.744376 | 0.830634 |

以上为隔离 val 的患者平均质量，不是隐藏 train-U GT 上的真实标签质量。T-mask 扩大覆盖，但前景精度约 0.719–0.780，显著低于背景约 0.986–0.988，不能用总体高置信替代前景正确性。

| 域 | 最终学生 | NLL | multiclass Brier |
| --- | --- | --- | --- |
| RIM_ONE_r3 | SUP | 0.269221 | 0.074609 |
| RIM_ONE_r3 | T_SCE | 0.226528 | 0.072039 |
| Drishti_GS | SUP | 0.308807 | 0.105867 |
| Drishti_GS | T_SCE | 0.273515 | 0.104002 |

T_SCE 在两域平均 NLL/Brier 均改善，但未建立 Dice 实用增益；概率指标与任务指标必须分开。完整 confidence 分位数、pmax>0.99、错误前景 confidence、所有阶段、学生/教师及逐类质量均保存在 [PROBABILITY_AND_PSEUDO_QUALITY.csv](PROBABILITY_AND_PSEUDO_QUALITY.csv)。

## 工程执行与计数边界

冻结执行源码 `098c9c73dc1b7dd1b723363e45cbe34e169fe354`；服务器执行 checkout 干净。任务耗时 106.13 分钟，GPU4/5/6 三条独立队列。公共输入、五臂初始化、20epoch warm-up、标签顺序与模型状态匹配；SUP 和 warm-up 无 U 图像读取；150 次诊断前后学生/EMA/优化器/RNG 状态保持，真实诊断额外 backward=0。

| 阶段 | optimizer / backward | 模型 forward | image forward | EMA |
| --- | --- | --- | --- | --- |
| 正式训练 | 79500 / 79500 | 181260 | 358680 | 79500 |
| 固定真实诊断 | 0 / 0 | 48750 | 48750 | 0 |
| 独立单学生部署 | 0 / 0 | 975 | 975 | 0 |
| 真实 labeled-only smoke | 10 / 10 | 26 | 52 | 10 |
| 四次合成资格 | 616 / 616 | 1456 | 2184 | 未单独持久化 |
| 本轮合计 | 80126 / 80126 | 232467 | 410641 | 已持久化真实合计 79510；另有合成 EMA |

正式训练 L 样本打开 159000 次、U 图像打开 99840 次；固定诊断 image-only 打开 4875 次，隔离 val 样本（含图像/GT）打开 4875 次，预测密封 24375 份；部署 image-only 打开 975 次，额外 GT=0。这些是 loader 级计数，不是操作系统 read 系统调用数。smoke 源码一次加载每域前两例 labeled，共 4 个样本打开后供五臂复用（源码推导，未单独 I/O 仪表计数）；不访问真实 U 图像。

四次合成资格均通过，各 154 次成功更新，合计 616≤1000；冻结执行源码本地/服务器资格均通过，真实 smoke 10≤10。合成 autograd.grad 共 12 次，仅合成算术验证。**计数限制：资格仪表未单独持久化合成 EMA、fixture I/O 与合成诊断的细分小计；不能把这些标为 0 或声称计数无缺项。** 合成总 forward/backward/optimizer 已实测保留。本次不为补细项重跑资格、不更改运行器原 ENGINEERING_COMPLETE；它表示固定真实矩阵及隔离部署核验完成，细粒度资格审计仍有上述限制。

历史正式更新 168608 + 本轮 79500 = 248108；该数不含合成和 smoke。本轮包含合成/smoke 的成功更新总数 80126。历史全部非正式 forward/I/O/EMA 未在本轮重新构建，不能与正式历史总数混称。

实测训练最多 student+EMA 两份完整模型，部署一份；每份参数 1936064 bytes，Adam 3865412 bytes。峰值 Torch allocated=867.922 MiB、reserved=996.000 MiB。sigma/grad_update 各 1728 bytes 为惰性状态，未入 Adam；无原型、GAS 或第三份完整模型。

## 结论和停止边界

纠错机会被学生置信度排除的机制得到描述性支持，但教师置信度同样排除了大量机会；扩大通路会同时引入更多错误监督。SCE 的局部梯度性质与概率质量改善未转化为稳定的 Dice 增益。因此结束本轮微型骨干、同视图轻扰动 EMA 下的监督路径检验，不追加参数、seed、PAS、分类头或投影，也不进入 CL。这不构成“半监督学习普遍无效”的结论。

固定 split0 已经历多轮研究者暴露；三个优化种子衡量训练随机性，不是独立患者确认。患者 bootstrap 按预先声明未执行；不作事后显著性检验，不将像素或四个 draw 当独立患者。

## 交付与复现

原冻结协议和 SOURCE_FREEZE 不变。`EXECUTOR_FINAL_REPORT.md` 保留运行器原报告；`DECISION.json` 保留原终态。`CLOSEOUT_VERIFICATION.json` 保存只读重验及部署/退出凭据；`ACCOUNTING_SUMMARY.json` 提供明确计数和不可用细项。运行 `python3 build_final_report.py` 只读取这些公开汇总表，重建本报告并核对 30 行结果、79500 步及原判定；不加载模型、不读 GT、不训练。

全部逐类/逐阶段汇总和资格证据公开。图像、GT、患者 ID、逐病例像素、checkpoint 和模型权重保留 NAS，不进入 GitHub。具体发布提交、匿名访问结果及 NAS 归档以 PUBLICATION_VERIFICATION.json 为准。
