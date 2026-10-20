# 正规空间残差潜变量：零更新检查与冻结模型局部VI

## 新生成模型与ELBO

这是离散模型的版本化扩展，不是旧自由残差的后验回写。c~N(0,I)，R~N(0,I)独立、R维数16×16×16。base=upsample(ReLU(Wc+b)4×4,128)，f=ReLU(base+upsample(R,128))，Z逐像素Categorical(softmax[0,Conv2(ReLU(Conv1(f)))])；x|Z、m、I|x,m沿原正规密度，L标签Y=Z。R先验为标准化单位高斯，不扫方差或损失权重。

L变分族qx(I)qm(I)qc(Y)qR，qR=diagGaussian(mu,logsd)，本阶段每个原训练视图独立局部状态；其余100模型参数冻结。新L下界=E[原五个joint因素(类别项用R修改)−logqx−logqm−logqc]−KL(qR||N0I)。KL=1/2 sum(mu²+exp(2logsd)−1−2logsd)解析计一次，禁止再计pR/logqR。该式是对R精确积分KL后的合法ELBO估计，不是加权惩罚；qTheta原全局KL常数按2N计入训练摘要，单图模板评分不混入全局KL。

唯一模型源为原discrete_l_train/S0600完整cursor8075，不使用诊断recon权重或oracle残差初始化；旧qTheta仅新模型变分初始化，所有prior/initial保持原p0，不声称跨模型持续后验更新。局部qR初mu=0/logsd=0=pR，不由标签复制oracle特征。U未实现，bound(None)继续拒绝；不给图像识别器加额外CE，40识别参数冻结。生成标签使用Y不是预测。

## 零更新准入

同一旧proposal和Theta抽样下，R=0的5joint因素/类别概率逐项精确复现旧离散模型；qR=pR解析KL严格0；非零mu/logsd的解析KL与标准torch.distributions.kl_divergence一致；8次实际随机qR下界/归一化有限。额外2次诊断反向检查qR mu/logsd有限非零、全部模型gradNone；0optimizer/0模型或局部应用。模型/RNG/源604物理账本不变，零检查独立ZERO_RESULT；失败不进入局部VI，保留成本。

## 条件性局部VI

准入后固定原两L四翻转8视图，仅优化qR的mu/logsd，各500局部尝试（总500反向，不是每视图500独立反向），最多500接受/4000候选SAA。梯度与同一步验收使用同一2Theta×4局部样本；每步从源完整RNG继续新流，不反复拟合同一SAA。每次候选恢复本步起始RNG，整步后只前进一次8draw所消耗流。固定50163/80163流只在0/100/200/300/400/500读出，评价恢复训练RNG、不参与验收/选择。

loss=−sum每图ELBO/N，空间L2梯度mu/logsd均乘16²，通道不另放大，lr.05及原八值alpha；逐图ELBO不降1e-6且均值升>1e-5接受，不按标签Dice验收。KL加入后logsd不钳制，非有限候选仅拒绝，全部cost保留。每25保存mu/logsd+step/RNG/cost；500局部反向、最多4000候选、含零检查及两流评价总bound上限36200，另标签条件生成均值读出6次。0optimizer构建调用/0模型参数应用。

评价全部8view标签条件均值解码Dice/类别NLL、两流ELBO及每项、KL_R/N、mu位移/logsd分位数，不择优/早停/调参。原.8门限描述性，成功也不称图像识别/泛化/SSLCL有效，失败不盲延长。此次局部VI不是可部署摊销器或持续更新；其有效性只决定是否值得再实现正规摊销推断。四根20GB硬/18GB软、数据盘中性create-only、UTC03:30截止。
