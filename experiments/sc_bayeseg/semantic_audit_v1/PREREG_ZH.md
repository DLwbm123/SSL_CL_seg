# R/J终态语义分布零更新诊断

用户仅授权两项：均值×方差四格交叉替换及第二类混淆/margin；独立输出坐标与共享参数分项梯度。optimizer更新严格为0，不构造优化器、不训练、不做候选专属code适配、不改checkpoint、不加U/目标/seed或调参。既有监测保持暂停。

## 四格

同原两张可见Siemens训练L，128²。提取split_v1已完成R/J终态的语义mu_ell和s_ell=-4+3tanh(raw/3)，构成RR、RJ、JR、JJ四Gaussian，前字母为均值来源，后字母为logsd来源。

沿用seed50163的4次共同epsilon_ell，复现原proposal对x/m/ell/c的随机数消耗次序。每格四抽样softmax概率平均，不做refinement/SNIS、密度评分或logq替代。RR/JJ必须复现历史主指标；固定两套均值链同时报告。4×4×2=32个图像级logits抽样，不是32次完整网络推断。

报告4类聚合混淆矩阵、逐类Dice/召回/precision、CE/背景占比；当前可见GT=2区域的预测去向。mean margin定义为mu_GT-max(mu_other)，背景参考0；报告均值、5/25/50/75/95分位数、负比例及最大竞争类别。另报告四抽样平均概率的概率margin，单位不同不得混用。记录R正margin→J负margin的GT2像素数量及去向，只在原训练可见标签诊断，不作筛选规则。

## 梯度

仅R终态，冻结生成后验和所有非语义网络参数，但保留F(P)及code_encoder对ell的导数。4次共同MC流95163..95166（仅诊断抽样，不是新增训练seed）；每次重建实际随机x/m/ell/c及Theta抽样和原密度。项分别为label/image/shape/logits/code/appearance的NLL，加x/m/ell/c各节点正logq，最后完整负ELBO，全部采用原每像素归一化。

对同一个计算图的mu_ell和s_ell分别求偏导，然后计算两条VJP到语义参数：J_mu^T g_mu及J_s^T g_s，核对两者和等于直接参数梯度。参数组为语义前端shape_encoder、logit共享trunk、最终mu头、最终raw-sd头。输出logsd坐标与raw-sd头不能混称。

报告输出梯度范数及与label项的cosine、GT2 mean margin沿负梯度的一阶变化；参数梯度及mu/s两路径分组范数、与label梯度关系和margin方向。独立mu的Gaussian自身logq梯度应为0；code logq可以经由code方差依赖ell产生均值梯度。记录ds/draw=1-tanh²(raw/3)的分位数及小于1e-6比例。

所有方向读数是未经Adam预条件/clip的局部负梯度或其单位范数方向，不是实际训练更新，不以其推断完整600步因果。先对4次梯度求均值再计算主要方向，同时保留各次范围。最多100次autograd.grad调用；0次optimizer调用。没有执行真实参数扰动或局部梯度下降。

状态不变、历史指标复现、VJP恒等式及分项和核对属于工程证据。有限MC、两训练图、固定R状态及模型近似限制保留。完成后中文Markdown报告、源码协议匿名聚合结果按全局规则交付；原图/身份/权重/原始日志/机器路径不公开。任何失败保留回执，不能扩大到第三项。
