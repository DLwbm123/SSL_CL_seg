# 90%能量支持与谱均衡 V1：运行前冻结协议

2026-10-22 新独立问题；旧更新谱轮16320更新已完成、AUDIT PASS，其失败门槛与原时钟永久保留。未修改旧配方或重跑旧终点。

## 根据与可证伪机制

旧轮GRP−GR旧域平均+.222730pp，低于+.3阈值；入口旧患者>1pp损失只减少1/0人，复合门槛失败。SP−S旧域+1.337999/+1.108463pp明显大于GRP−GR，旧域差中差为负，不支持GR特有协同。提议更新平均stable rank约2.37–2.62，数值rank平均59.125；全谱均衡使方向cosine约.58。

这些诊断不证明弱方向有害。本轮检验：保留原位移承载90%能量的主方向，再均衡其奇异值，是否比全数值支持均衡更少伤害患者尾部？若只截断即可解释收益，则不能归因于支持内均衡。

已读取MuLoRA arXiv 2610.02283v1，2026-10-01，Junkang Liu，方法3.1/3.2：https://arxiv.org/html/2610.02283v1 。题名摘要对应ICLR27目录nu3KHzX29K；匿名目录无作者，作者一致性未核实。该文采用固定正交LoRA基底及动量极分解，并指出单步均衡不保证累计谱均衡。本轮不采用其历史白化/LoRA训练，是独立定义的能量支持干预，不称论文复现或新颖性已证明。

## 四组及唯一差异

两新目标流20261121/20261122；每流GR、GRP、GRT、GRTP。共同源163/S†69入口、空opt；每组最后student2000主，EMA.999统一辅助。全部同GR损失、原P/Q/W_R/bias、lambda_R=.1144953842725508，不重新校准/抽Q；两个G/GR辅助标量和48encoder前向均保持一致。GR是标准SGD，另外三组在encoder八个卷积矩阵上干预SGD提议位移D（含momentum、decay、clip）。decoder/bias/归一化参数不变，SGD动量buffer保留未变换值。

float32薄SVD D=U diag(s) V^T，CUDA固定gesvd；数值支持r_num=count(s>max(m,n)*eps32*smax)。GRP在全部数值支持上均衡，D'=||D||F/sqrt(r_num)*U_rV_r^T。GRT/GRTP选择最小k使sum_{i<=k}s_i²>=.9*sum_i s_i²（上界r_num）。GRT保留这些奇异值的相对比例，归一回原||D||F；GRTP将这k个奇异值设为||D||F/sqrt(k)。零矩阵保持零。每层均保留本次提议F范数；三组都用相同实现的SVD、数值支持与应用流程。90%为运行前固定机制选择，无阈值/系数搜索，不用验证数据选rank。k按当前更新本身计算，不使用患者标识或标签之外的信息。

这构成支持（全/90%）×形状（原/均衡）的四格对照；GR对应完整原位移且不执行无用SVD。主比较GRTP−GRP；GRTP−GRT分离均衡与截断；GRT−GR刻画截断单独作用；GRP−GR复核全谱作用。交互=(GRTP−GRT)−(GRP−GR)。均不推断不同训练轨迹逐步能量相同，也不把权重空间能量等同函数风险。

## 数据、检查与终点

沿用原Philips train-L白名单：2监督图+14 L池image-only图，三view48encoder、64位置行、原32维谱损失。无U accessor/隐藏GT/旧源训练图/在线teacher/历史bank/生成器。保护原HDF5、splits、患者身份、环境、所有旧权重。SGD.001 Nesterov.9 matrixwd.0005/vector0 globalclip1；sigma/grad_update/P/rotatedbuffer冻结。

CPU/GPU三个变换数值自检共30小矩阵SVD、0模型前向、0更新：已知能量支持、范数、零矩阵、重复、转置、卷积布局。64更新detach用MINIMAL+energy_polar对S+energy_polar，逐步model/opt/EMA/RNG exact并丢弃。8组资格各32包含实际非空恢复重放、golden、两样本过拟合，全通过后才正式训练。失败成本记原账原钟，不盲重放、改公式、加预算。

每组独立训练2000步后退出再评价；student旧val544/新val660/源train230/目标train254及EMA旧544/新660，共2892 native，8组23136。只读固定final，不选best/中途val。40固定诊断step0/499/999/1499/1999，共80额外reverse，不用诊断调训练。SVD逐步记录rank/numerical_rank/保留能量/stable_rank/cosine/提议和实际范数；数值变换相对能量误差<=2e−5，float32应用误差<=1e−7+.005proposal_norm；保留能量>=.9−1e−6。

## 预声明成功条件

GRTP−GRP旧域两流均值>=+.3pp且每流>0、新域每流>=−.3pp；GRTP相对入口旧域每流>=−.5pp。相对入口旧患者损失>1pp每流比GRP至少少2人，>2pp不增加，p10不得低于GRP−.3pp。另要求GRTP−GRT旧均>=+.2pp且每流>0、新每流>=−.3pp，才将收益归于支持内均衡而非单纯截断。以上全部满足才判复合成功，任何失败均保留完整证据。

交互只在旧每流>0、均>=+.2pp、新每流>=−.3pp时称一致点估计交互。所有比较完整报告患者均值/中位/p10/改善下降、>.5/1/2pp人数、LV/MYO/RV、ED/ES、配对bootstrap10000 seed163、源/目标train与EMA。CI不覆盖训练随机性、多重比较、反复开发；两流共享源和验证患者，不是独立源泛化。原源→入口−.933692pp接入损失继续计入，不称临床安全或非劣保证。

## 硬预算

新create-only root，原钟8小时，7h不新stage，7h45m主动stop，独立timeout在8h前KILL全部所属进程。新增实际存储<4GB，3.5GB写保护+.5GB退出余量，预计<700MB，2MB中性操作目录计入。所有新日志/临时/ckpt/code在指定/data/bmw；旧环境和权重只读不复制。仅既定pro5000 GPU3 UUID，主/子argv中性，不干扰其他进程。

16320 optimizer=16000formal+256qual+64detach；functional/toy0。变换步12256=12000formal+192qual+64detach；每步8SVD=98048，再加30数值自检，总98078。每次reserve/executed持久化，pending异常停，不盲重放。

Calls caps：detach native160/encoder1632/reverse64；资格native768/encoder1536/reverse320；正式native32400/encoder770000/reverse16032；diagreverse80；evalnative23136。合计native56464/encoder773168/reverse16496，teacher/generator0。预期实际detach128/1536/64、资格640/1152/280、正式32096/768768/16024、diag80、eval23136。

全8组结束且所有worker退休后仅一次energy_support_audit，0模型forward：查EXIT0、16320账本/calls、12256变换步/98048训练SVD、数值自检30、共同入口空opt、final=latest finite optimizer/RNG/reads[2000,14000]、40诊断、固定P/Q/bias、患者EDES和资源。审计后科学解释并公开代码/协议/匿名聚合报告；禁止患者ID、图mask、权重/投影/逐图特征/原日志/服务器配置。继续自主分析下一独立问题，不能为正结果修改本轮门槛。
