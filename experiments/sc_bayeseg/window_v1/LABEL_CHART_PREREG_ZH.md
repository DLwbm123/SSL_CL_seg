# 显式标签条件变分提议：零更新数学与数值审计

动机：G的安全joint改进仍强化own模板，matched-confidence oracle评分未改善。先验证一个只改变L变分族、不改生成模型/损失权重的机制，而非立即训练新模型。

令未约束参考类logit样本u∈R³~qφ(u|I)，背景logit固定0。对观测Y=0令z=−u，ell_j=−softplus(z_j)。对Y=k>0，令z_k=u_k、z_j=u_k−u_j(j≠k)，ell_k=softplus(z_k)、ell_j=ell_k−softplus(z_j)。线性A_Y的|det|=1；非线性T_Y在实数域双射到参考类为0且argmax=Y的开放区域，|det ∂ell/∂z|=∏sigmoid(z_j)。因此log q_L(ell|I,Y)=log qφ(u|I)−Σ_j log sigmoid(z_j)。x/m提议不变，c在新ell下重新编码与采样，全部原q因子及logq必须完整保留。q_U仍原未约束提议，Y未知时不使用变换。

原p(I,x,m,ell,c,Y|Theta)与KL完全不变，这仍是合法有标签ELBO的变分族选择；但q_L支持只覆盖一类区域，真实categorical标签后验仍可能对错误argmax区域赋质量，因此这是有偏的受限近似，不声称精确后验或扩大了变分族。它保证的是精确算术中L提议样本处于观测类别区域，不保证q_U图像单独预测正确，也存在“标签替推断器纠错”的梯度绕路风险。新提议不是教师/回放/路由，不加入额外CE或语义权重。

本阶段0optimizer/0参数应用/0局部优化：四类各一个固定FP64三维合成样本，显式3行Jacobian共12反向，检查解析logdet、inverse及log-density变换；另外K/G两完整固定状态×U/L提议×8共同draw在原两L四翻转batch8读取ELBO/各项/提议类正确率，32次bound。两状态U首draw另与原拆分模型同RNG逐项复现，共34bound；每个状态/提议首draw分别做完整ELBO及标签项梯度，8反向，总20诊断反向。统计FP32 argmax不符/类别并列，不能以精确算术保证掩盖浮点区域挤压。所有输出有限并完整模型不变；原G物理账本不变。

不使用heldout图像或标签，不启动生成参数训练。合成变换、自身proposal/ELBO调用与反向分别记账，原始局部样本不公开。只公开公式、源码、匿名聚合、局限与中文报告。检查通过只表示变分密度实现可审计，不表示性能成功；若有FP32区域问题或ELBO/梯度异常，先解释数学/数值原因，不通过修改阈值或概率尺度掩盖。后继训练需另立协议；用户窗口自主范围内推进，原20GB空间/UTC03:30截止持续有效。
