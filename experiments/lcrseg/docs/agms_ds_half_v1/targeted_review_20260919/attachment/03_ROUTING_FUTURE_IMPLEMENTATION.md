# 备用实现约束：AGMS_DS_ROUTE_DIAG_V1（PLAN_ONLY）

此文件只定义后续候选，当前不授权实现或实验。HALF结果审阅后另行决定是否使用。

核心：保留原A5头训练系数.25，只在学生L辅助头输入上设置kappa=0或.5。不更改主网络forward输出，不对主监督、KL、H截断梯度，不重新引入F5式仅训练附加R的通道。

同状态合成测试应验证：
- h_k与h前向完全相同；kappa0主DS梯度为零且头DS梯度不为零。
- kappa.5时上游DS梯度为kappa1的一半，同状态头梯度相同。
- kappa.5/lambda.25与kappa1/lambda.125具有相同的同状态主DS梯度，头梯度前者为后者两倍；这不是Adam参数更新倍数承诺。
- 禁用H且kappa0时，主网络与B2的原生关闭态一致，新增头、风险计算不能改变主数据/RNG和优化组。资格对照不产生第三个真实实验臂。
- 教师及risk仍stop-gradient；头只接L梯度，学生U仍沿原白名单进入A/B。
- 多尺度头初始化与原方法一致，study/run/arm不意外进入基础随机流。
- 原LCTX同mask源收集、readout几何、warmup、EMA和polynomial schedule保持。
- kappa需要进入semantic record、canonical node、checkpoint/resume及权限；错kappa/旧study恢复拒绝。
- 主模型从原B2前缀开始，辅助头/EMA/risk/optimizer当前阶段初始化，阶段结束丢弃。

诊断新增同状态U门控对照：共享同一次teacher输出，计算原主头门、等权门和风险门，记录数量/并交集/拒绝原因/阈值margin，不读取U真值，不保存患者级原图与mask到公开结果。记录all-valid和候选子集不同归约，空候选记NA。

下一份完整实现prompt必须另外固定实际文件边界、生成资格次数、CUDA/smoke成本、两个顺序的4节点10600更新、历史控制导入、阶段完整性和最终报告。代码完成提交后先外部审阅，未经新审批不得实验。不得借用HALF或原AGMS权限、预算与CPU报告。
