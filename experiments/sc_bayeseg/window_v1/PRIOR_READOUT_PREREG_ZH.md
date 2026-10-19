# 原始生成先验负对照：零更新诊断

coordinate与code_closure已确认GT固定code存在不匹配混杂；重编码c后原J初态也正，因此尚未证明后验学到了语义。原始生成先验是围绕保存的随机初始化均值的高斯分布，而非零均值分布。

保持已有两局部坐标（原J初态、robust_quarter终态）、GT/自身ell模板及fixed/reencoded/swapped三种code处理；只将anatomy/structure的mean_w/mean_b/logsd_w/logsd_b替换为原J checkpoint同层prior_*缓冲。先断言prior_*与initial_*完全相等，防止误用递推先验。其余参数与局部坐标保持一致。仍用原8个共同draw、六模板和shape/logits项，0标签似然。

2局部×2模板×3code×6图案×8draw=576次批量生成评分，0optimizer/0反向。与已交付code_closure两实际后验进行配对差值，不重新计算其1152评分。不调参、不选择模板，不加U/目标域/训练seed。若先验也正，说明匹配指标可能包含架构、SAR以及推断器对随机生成映射的适配；后验与先验差才是增量证据，但也非泛化证明。报告全部正负和逐图。

20GB窗口上限，真实存储/探针先检查，保存私有分数与退出回执，后继再根据证据登记，不把零更新读出改善冒充训练性能。
