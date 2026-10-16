# M&Ms 单一下一版协议

2026年10月15日。状态为方案与核心工程检查，无训练授权。方法、detach与生命周期分别以METHOD_SPEC、IMPLEMENTATION_CONTRACT、ACCEPTANCE及protocol.py为准；没有第二个候选或超参扫描。

固定原manifest、split seed0、optimizer seeds163/164、background/LV/MYO/RV标签、患者先划分后phase/slice、canonical320/aux25排除规则。源→目标顺序为Siemens→Philips→GE→Canon；复用已有两源各43600，源新增optimizer0。每seed目标Philips53300、GE25100、Canon26500，共104900；两个seed新增209800、继承87200、名义297000。

原paired batch2、FP32、AMP/TF32关闭、Adam/lr/weight decay/poly、20%warmup和20%U-ramp、EMA.99、LCTX/style.2及V1主目标权重保持。新增历史项.1、Cayley正交G/F、有限强度elastic=1及MRI首层冻结是固定方法设计值，不是经验证集挑出的最优参数。

每个source和每个target都在固定final模型上采集32批当前训练L记忆；两seed四域共256批、512图像曝光，新增源current-bank初始化另外4图像曝光；目标原bank初始化仍按原训练流程单列，不重复计作本新增摘要。真实摘要读数/额外VJP/CPU和GPU时间另记，不能称与V1完全相同数据曝光或计算量。正式优化器预算不变，额外资格调用单独授权和记账。

每阶段固定final在原seen-domain验证集报告全域/全类，按患者内ED/ES平均再聚合；报告Final、Old、Incoming、signed acquisition-to-final Forgetting、各阶段旧域跌落、Infinity/缺失/失败及全部负结果。重点公开Philips/GE后Siemens，不用Canon后的恢复掩盖中途崩塌。既有V1为同源描述性参照；本轮不产生新性能比较。

未来每目标只允许一次正式尝试，失败后代BLOCKED，不自动重试、加seed、延长预算、补零位移调用、挑best或改顺序。每个历史域类与当前学习的冲突、代理损害、实际/提案位移比例、FP32零位移、谱误差和内存/时间均保留。没有独立test、验证梯度、ablation或隐藏label能力。

本轮工程通过不等于学习有效或遗忘改善，也不授予启动权限。完整训练绑定、真实数据摘要/GPU恢复资格仍未实现或未运行；须在后续明确授权范围内完成并留下独立回执。

## 训练就绪工程补齐（2026-10-15）

用户追加授权补齐trainer/controller、真实历史摘要采集和GPU恢复资格，仍不启动正式训练。本次按相同单一方法和正式209800新调用预算做必要工程检查，不调整方法常数，不做ablation或独立test。旧核心资格保留在core_engineering_history。GPU工程预算与图像曝光单列，允许独立短端点验证三阶段携带，但engineering_only标记不能冒充完成的正式前代；成功资格预定110次Adam。两病例拟合仅工程，不产生科学比较或选checkpoint。最终TRAIN_READY只绑定源码和工程能力，execution_authorized仍false。
