# Structure方向诊断：条件项、均值KL与Adam

2026-10-20。只读components已完成：新mean端点交换损害两初始化四相对指标，新sd端点对BG/DELETE2反有小幅益处；mean KL下降.00640092而variance KL略升。不能据原92%加法份额认定variance塌缩，不能盲做mean-only训练。本新诊断只回答训练梯度是否与类别证据冲突、KL还是数据项或Adam历史参与了冲突，不改模型/损失/预算。

## 固定来源和数据

原两canonical训练L、128²，不翻转、不读开发保留/U/目标/隐藏test，无新训练seed。固定P2/P1 init0五候选GT/BG/CROSS/SHIFT/DELETE2的旧200步qxm，**不优化q**。读取structure_only/S0000(cursor8675)和S0200(cursor8875)两份完整model/Adam/scheduler/RNG；只允许structure12参数参与autograd，其他88无梯度；所有100模型、prior、Adam/来源payload严格只读。

每个状态从该保存RNG起，固定读取两个连续2Theta×4local批量流。每个Theta共享对应局部e1/e2给五候选；第一状态第一流的GT B及完整coordinate objective必须复现原正式第1步loss误差≤1e-6。第二状态没有旧第201实际更新，不能虚构有限更新复现。读出不借最终评价流或换seed挑符号。

## 目标和反向预算

每状态每流得到七个标量与梯度：

1. J=−mean B_GT+global KL/(2N)，与原训练坐标目标一致；
2. data=−mean B_GT；
3. KL=global KL/(2N)；
4. Gamma=mean(B_GT−B_CROSS)；
5. mean(B_GT−B_BG)；
6. mean(B_GT−B_SHIFT)；
7. mean(B_GT−B_DELETE2)。

按原概率密度定义B，局部q冻结，绝不对这些类别对比构造新训练损失。每组七次autograd.grad，合计**28批量诊断反向**（两图工作量56）。验证gJ=gData+gKL相对误差<1e-5。按mean6/sd6/all12三个块报告梯度norm、data-KL余弦及方向导数，不公开向量。

方向固定为负J、负data、负KL、原继承Adam的一次纯张量候选。Adam复用已有candidate函数、完整12F梯度clip1与原lr/eps/betas/历史；4次纯张量Adam，不构建Optimizer或调用step、不应用候选、不做候选前向。分别报告全F12以及其mean/sd块分量，不把mean块分量叫做“另一次只训均值的Adam试验”。各方向报告L2范数、目标及四对比的raw/unit-L2方向导数。现有梯度或Adam若损害类别，只是一阶证据，不能当有限步或长期训练结论。

固定4组，每组2Theta×5候选×4local：共**160批量B密度、8Theta draw、40 F批量前向**；0optimizer/0参数更新/0候选前向/0局部更新。所有反向、纯Adam与密度独立连续记账。私有梯度存既定数据盘，公开仅匿名聚合及标量Gram/方向导数。

## 预定解释及运行约束

- 负data有利而负KL不利：支持正规正则项与拟合/类别支持的局部冲突，后继需正规概率解释，不能直接删KL或调类权重。
- 负data本身也不利：不能把缺口全归KL，优先检查固定局部解释/条件目标与类别支持。
- 负J与Adam方向不同：保留历史动量/预条件因素，不能把结果全归原目标或盲重置Adam。
- 状态/流间不一致记敏感，不增流或反向直到出现期望结论。

入口与输出create-only、中性argv，先查ps/GPU、挂载容量/du及真实读写探针；预计新增<20MB。独立原始时钟守护23h45保存/23h59硬停，新增软45GB/硬49GB，保留旧四根18GB保守门槛；所有重试共享起始时钟与实际成本，失败保留现场不盲重跑。模型/加载payload/RNG/gradNone、源历史账本/q起终不变，资格/成本/EXIT共同核验。只读结果中文正负报告、源码/协议匿名标量push并核验远端和公开200，再自主选择单一后继，不延长旧all12训练、不加U/目标/seed，不声称分割/SSLCL/泛化收益。唯一sc-bayeseg心跳统一跟进。
