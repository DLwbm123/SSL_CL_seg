# BayeSeg 官方实现与本方法的对应

官方来源：https://github.com/obiyoag/BayeSeg ，MedIA实现，核对提交 ed6a45ba5e8c9f9d017469463a1fb8926f8e68e3，Apache-2.0。已阅读 README、models/BayeSeg.py、models/ResNet.py、models/Basic_module.py 和 train.py。

官方代码的shape/appearance残差推断、由shape推断分割的顺序为本实现的架构参考。本实现独立编写，没有直接复制第三方代码，不将作者已训练模型或原始数据收入仓库。官方网络宽64、shape10块、appearance6块，分割使用EfficientUNet-B2；本轮采用宽16的小型推断器和新的生成器，是新模型开发验证，不能声称复现作者成绩。

官方实现训练时用Gumbel-softmax得到概率，多个精度与相关权重detach，变分项按残差精度总和归一化，另结合Dice/CE。这些数值目标不等价于本次正规化联合证据。SC实现明确区分实数ell与softmax概率，使用固定可逆SAR和精确Normal–Gamma积分，不沿用原精度近似、经验归一化、Dice或Gumbel过程。

共享生成函数G(c)、F(P)在官方模型中没有可直接复用的对应模块。这里为它们的全部权重和bias建立Gaussian参数后验，并按上一阶段后验或初始prior构造固定阶段目标。推断网络参数是确定性的proposal，不伪称为生成参数posterior。测试局部refinement和联合证据权重用于连接预测与生成模型，不承诺有限样本推断完全消除推断器遗忘。

核心数学定义见公开执行前协议。设计来源为用户确认的Pro方案；用户提供的私有文档不原样公开。全背景、外观吞噬、概率场藏纹理、重要性权重退化和后验过尖锐均保留为可观测风险，不以Bayesian名称代替验证。
