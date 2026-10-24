# APOS V3 源函数保持结构漂移 单种子冻结执行协议

2026-10-24用户明确授权本轮。独立于已完成V1/V2，禁止重跑旧轮、参数/种子/秩/迭代/EMA扫描。主R，单seed20271611，N/Q/R三完整配置；本协议将V3_PROPOSED绑定到同一引擎及物理调用账。

## 方法与数据

原FixMatch43600 EMA163、真实3×3读出/padding0/原插值完整保持，主体/末头永久冻结eval。学生h'=h+B tanh(Ah)，16→8→16无bias，复用V2已保存源末EMA（B为0）。源参考q0只对同一当前输入做冻结native前向；学生qθ经共同适配器再走相同native读出。每输入两条完整native路径，即使N也同算，不把调用公平称相同FLOPs。

只读复用V2 prior.private.pt和source/CAL/final.pt；源新优化0、先验新拟合0、新源校准0。源门禁只既定Siemens验证23患者/46EDES体积544切片，N源原始保存行复用不重评。适配只Philips既定13患者26体积254切片train-L，每步2不同患者，沿V2 Current读取规则及seed；拒绝任何旧图/旧标签HDF5/U/test。适配阶段源参考仅访问同一Philips当前输入，不访问Siemens。目标评估为固定Siemens旧23/Philips新31开发验证，不进入训练或选择。

原先验8模式各rank8，占据格32×32×3，复用230源训练标签拟合的既有模型，不重读源训练标签。原校准头和模式头全程冻结，用原生h0而非学生特征；pi和W=s²/(s²+b²)固定在一次前向中。pi作为混合矩权重，不再优化潜在模式rho。mu_bar=Σpi mu，Fx拼接sqrt(pi_k)Fk和sqrt(pi_k)(mu_k−mu_bar)，至多72列，不能平均任意PCA基。dx=Σpi sigma²，数值下限1e−6。

A为384→32非重叠平均，a0=Aq0_FG，r=u−a0−Fxz。Q的H=.5Σr²/dx+.5||z||²；R的H=Σ(sqrt(1+r²/dx)−1)+.5||z||²（用等价有理式避免小残差抵消）。二者零点曲率相同1/dx，R影响函数不超过Q，无Student-t中心收缩混杂。lambda=beta=1、标准化pseudo-Huber尺度1固定，不扫描。

E=mean KL(P||qθ)+1/(2M)||AP_FG−u||²_W+H/M；P四类单纯形，u三前景非负且sum<=1，M1024。源入口qθ=q0取P=q0,u=a0,z0使所有非负项为零，唯一P=q0。固定x的Fx/dx/W下目标凸，不能由此声称适配后旧域无遗忘或新域必然达标；原源在新域错误也可能受此保护。

## 求解及资格

固定8次外层：16次像素固定点，z的<=72维SPD Cholesky，u逐格精确加权单纯形投影；最后额外16次像素刷新。R用D=1/(dx sqrt(1+r²/dx))作相切majorizer，Q用D=1/dx。每次源/目标预测都执行同一推断，N直接native。全展开求导，禁止用源恒等提前return来删除结构梯度。

每轮及最后刷新能量不得增加超过1e−6 max(1,|Eprevious|)。像素固定点残差<=1e−4。最后以当前r重新计算D，检查z条件解差/(1+||z||∞)、u预条件投影梯度映射差、最后像素残差三者max<=1e−4；这是明确的预条件驻点判据，不冒称未缩放梯度范数。失败STOP，不增加迭代/容差或换seed。

先零医疗CPU/CUDA合成：源恒等及非零漂移、有限梯度、影响函数对照、独立中心差分、真实native384尺寸反向及速度/显存。数值资格cap128native样本/64reverse；计划14native/17reverse（CPU/CUDA各6目标backward含方差下限压力测试+1导数grad，真实尺寸N/Q/R各1backward），失败调用以阶段进度保留，0opt。预计总耗时含余量>=20000秒则不进入医疗。

随后Q/R源门禁各一次评估、每次544输入且两native路径；记录每张输入max概率误差<=5e−7、argmax完全一致。完整23患者相对原始N均值、CI两端、p10差绝对值<=1e−10，>1/>2人数为0。任一函数或能力失败SOURCE_STOP，禁止任何目标资格/优化；不得靠放宽源阈值继续。

源PASS后先N32资格，包含两golden backward、两额外entry参考/EMA函数检查、30次固定2样本过拟合+一次精确回放+恢复初态完整1步，共32真实优化。沿已有engine做空/非空optimizer/EMA/RNG恢复、finite、冻结清单及optimizer清单检查。N过拟合终态另用同一固定两图的一次前向复用qθ/q0，算Q/R结构引起的任务梯度（LQ−LN/LR−LN），两次autograd.grad计账；能量各>=1e−12且归一方向距离>=.05才P0 PASS。此为一次可辨识门禁，不追加probe/扩大样本。源入口结构项为零符合设计，不错误要求入口非零。P0失败正常P0_STOP，不做Q/R资格或正式。

P0通过后Q/R各32同资格；全部资格通过才三组各2000正式。三组目标均原像素GTCE+按图存在前景类Dice，空图Dice0且CE处罚假阳性；无辅助校准损失/可训练模式头。SGD.001/momentum.9/Nesterov/matrixWD.0005、adapter clipnorm1、EMA.999固定，资格丢弃且正式从共同源重新初始化。每组0/999/1999从已有图求CE/Dice两个梯度诊断，6次额外reverse/组，无额外native。Q/R记录W/修正幅度/类flip/标准化残差/影响函数比、能量及驻点，评估只保存匿名聚合和私有患者行，不保存新旧图。

## 硬账与时钟

总optimizer6096=3×32资格+3×2000正式；源optimizer0。医疗native上限34000=源2176+全部资格600+正式24000+终评7224。源Q/R各1088native；资格预计N156/Q152/R152，总460native（含N末辨识4），golden及额外读出全部入账；正式每步4native×6000=24000；终评每组(544+660)×2=2408，三组7224。

医疗reverse cap6168=资格150+正式6018。资格预计104=三组34+N末辨识2；正式6000+三组各6梯度诊断=6018。源/终评0reverse。无额外probeopt。像素固定点sample更新<=2500000、latentblocksample更新<=200000，包含P0同图Q/R推断及最后刷新；不能把求解步骤当native或漏计失败调用。native/backward在调用前记账，optimizer pre预留与post实际提交分开。已完成调用不重做/借bucket。

单轮首次数值运行起原8h钟；25200秒不新stage，27900停止，独立timeout28600TERM/30KILL。严格新增<4000000000B含ops/缓存/原子副本，3500000000B写保护；remote+local峰值预留4050000000B，累计30GB不清零、启动前占用+活跃预留<27GB留3GB恢复余量。不删除历史。本项目/data/bmw既定例外，仅pro5000 GPU3 UUID GPU-9ea1d4ae-e403-cb3b-29b3-c63b083bf8ff，实际挂载容量写读及显存准入。中性父子argv，启动后ps/nvidia核验。

## 最终判定与交付

主R final EMA.999。旧绝对>=80.36740663770262、新>=83；own-source及原始source旧患者CI下界>=−.30、>1<=2/>2=0/p10>=−.50、人数不劣N。由于本轮源门禁要求N/Q/R相同，两种源参照数值相同但仍明示。R−N新均值/CI下界>=−.20；R−N旧>=.30；R−Q旧>=.15；R对Q两域均值/CI下界>=−.20。全部同时才综合PASS。

旧23新31患者EDES共同bootstrap10000seed163，单seed及重复开发，不声称独立测试/跨seed稳定/临床安全。正式全部完成才读取目标性能。源/P0 STOP不是目标性能阴性。终态全worker/controller退休、AUDIT不存在，只一次本轮audit.py零forward，公开全部协议/源码/匿名结果/报告/审计/状态，代理推送核remoteSHAeq/匿名HTTP200、localremoteDELIVERY/results_public齐。只本唯一30min监测，健康静默；交付后等待用户，不自动后继。
