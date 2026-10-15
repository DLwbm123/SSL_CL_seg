# V2 工程验收边界

本轮授权实现/冻结，不启动训练。使用 `test_checks.py`，全部图像/标签为生成tensor；复用实际锁定 upstream UNet，C=2/3/4。Adam/SGD step由fixture拦截，工程/正式 optimizer calls都必须0。

检查包括：

1. 固定30 DAG节点、511600名义曝光更新、130400源复用、381200新增目标调用预算；无test/ablation；没有authority时worker在创建输出前拒绝。
2. 接近奇异G、单独合法G但非法累计F、超界inheritedF拒绝；有限位移与非有限提案拒绝；Adam式各向异性提案违反参考时能被实际位移约束。
3. 有界EMA插值、source/target native折叠输出一致；所有teacher无梯度。
4. confidence class2/3/4 entry前景冲突保护、独立geometry、q/可靠性/evidence detach；历史支持缺类失败、负importance拒绝且状态不变、proto软证据范围。
5. 各native模型source L观察、target完整U反向、A/B/R图覆盖、MRI首层冻结且不在optimizer；source/target观察不得改模型。零B入口A梯度为零是预期，非零A路径由固定合成提案检查。
6. 64x64完整U golden gradient/loss恢复；实际384/256 L反向geometry；两域摘要携带且旧摘要不可改。
7. student/EMA/optimizer/观察/可靠性/guard/RNG/cursor/provider_reads恢复；被篡改history与非法F恢复拒绝；写入故障保留原完整checkpoint。

合成 phase cursor 与一次固定合成参数提案只用于覆盖图和guard，不是执行过的科学更新、训练数据拟合或主实验进度。

本轮不执行：真实患者训练/验证、GPU优化器更新、两病例迭代overfit、性能比较/消融、真实已有source的32批bootstrap。全局milestone中的迭代overfit因用户“先不要启动训练”明确限制而未运行，不能标成PASS；source复用/真实数据导入与GPU训练资格在未来明确授权后单独检查，不能将本报告自动当作启动许可。

保留每次检查源码、退出码、JUnit/log、成功/失败回执及setup错误。不得覆盖失败尝试，不以修复后PASS删除原FAIL。

最终attempt_07：16项PASS；另6个真实source checkpoint只读导入/生成输入零增量入口对照PASS，最大logit差0、argmax相同、source F恒等。attempt_06曾在真实Prostate164导入上暴露原满秩依赖，已用已认证修复的独立依赖快照解决；原失败与之前3个成功源对照均保留。所有尝试实际optimizer calls=0。
