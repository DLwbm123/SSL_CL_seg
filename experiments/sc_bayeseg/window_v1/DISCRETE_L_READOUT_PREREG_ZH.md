# 离散模型内完整局部下界的反事实读出

固定新模型，初/终推断网络psi×初/终生成后验Theta全四格。识别40参数原本冻结，用唯一原识别均值argmax生成硬own标签；不用旧连续ell或Gaussian四采样。每格原两L+两预选heldout同批，GT六模板（原样/三空间滚动/两前景循环）+own六模板+全背景，共13模板×8共同draw×4格=416次批量bound，0optimizer/0反向/0局部优化。

每次调用新模型bound(I,Zcandidate)，q(x|I)q(m|I)q(c|Zcandidate)重新计算并计全部logq，五个生成因素各一次。读出完整局部L下界、类别/shape/image/code/appearance/负logq；不只看旧固定x/c点密度。同draw Theta与qx/qm/qc的基础噪声在模板间精确配对。qTheta在本读出视为固定预测混合，局部下界含全部局部熵，不再将全局KL/2称单图证据项；KL另外报告，它在同后验模板比较中为常数。

报告GT/own各自空间/类别差、GT−own、GT−全背景、own−全背景，全部train/heldout/每图/draw正比例；分别报告固定psi下Theta更新效应、固定Theta下psi效应及两者交互。Monte Carlo局部变分下界只在期望意义提供预测混合联合证据的下界，变分间隙会影响候选排序，不能把分数当精确p(Y|I)、分割输出或全局可识别性证明。GT/own类比例不同，空间和类循环对照用于补充检查；不能只挑其中一个正项称成功。

heldout标签仅只读模板，不进训练/局部梯度/验收/调参/选臂。初末全部模型不变、32组Theta draw内模板基础RNG相同、416连续bound、原604物理账本不变、退出及有限值核验。源码协议先提交，原分数/标签/权重私有数据盘，公开匿名聚合及中文正负报告；写前四根空间/挂载容量/探针、20GB硬限18GB软阈值，中性create-only入口日志，UTC03:30截止。
