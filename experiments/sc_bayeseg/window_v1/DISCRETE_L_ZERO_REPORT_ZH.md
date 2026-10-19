# 离散语义、标注钳制生成模型的零更新报告

新模型的L边界数学与实现审计通过，但尚未训练、没有性能结论。它用观测Z=Y的类别生成p(Y|c)取代连续ell语义层；图像、外观、SAR形状以及Gaussian生成权重后验保留。不是只删旧损失的消融，ELBO数值不能与旧模型比较。

两像素16状态的合成模型精确logevidence−.9593657335，因子化q的ELBO−1.6483115808，两者差.6889458473等于KL，误差1.11e-16；完全枚举与score-function梯度最大差1.11e-16。以精确联合posterior作q时ELBO紧性误差2.22e-16。该检查只证明所写离散概率恒等式与玩具梯度，未运行真实U，也不证明高维离散推断方差可控。

原两L四翻转8draw的原始新目标均值2.33012，全部归一化/有限值通过。首draw类别项NLL .90532，完整梯度范数1.57645；该数值比旧目标小，**不能称优化或性能提升**。L下界对40个图像语义识别参数梯度全None，与Z已被标签观测相符；单独识别CE .00637927、语义梯度范数2.60099，有效且只作用于识别参数。这说明后续需清楚区分生成拟合与识别拟合，不能宣称L生成ELBO已训练图像分类器。

生成模型为c→Z→x→I、m→I，类别Y=Z无噪声观测；qc输入onehotY三前景通道，qx/qm依图像。类别概率归一化误差2.38e-7，五个joint因素code/appearance/image/shape/categories各计一次，qell不存在，logq只有qx/qm/qc。代码拒绝真实U bound和旧连续corrected推断，避免混用模型。

成功版本7诊断反向/8bound/0optimizer；先前默认CE归约因确定性kernel失败6反向/1bound/0optimizer，等价改逐像素后mean且保持确定性，新目录重做零审计。累计13反向/9bound，失败现场保留，没有清零。所有模型tensor与grad缓存/source104物理账本核验，未访问heldout/真实U/目标。公式与范围见[DISCRETE_L_ZERO_PREREG_ZH.md](DISCRETE_L_ZERO_PREREG_ZH.md)，完整匿名数据见[DISCRETE_L_ZERO.json](results/DISCRETE_L_ZERO.json)。

下一项仅对新模型做一次固定L生成校准：40图像语义识别参数及Adam冻结，60生成/物理/外观/code参数学习600步新L下界，保留原Adam/RNG初始化，prior仍p0；旧后验只是初始化，不是跨模型后验递推。识别预测保持不变不是成功指标。随后需在新模型内部检查正确标签与原错误图的完整L下界读出，不用旧ell点密度混比，不凭钳制训练语义称SSL/CL。
