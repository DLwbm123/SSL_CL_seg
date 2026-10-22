# 历史decoder功能参照：阶段P0冻结协议

用户2026-10-22明确同意按分享分析的修订意见执行，窄范围授权增加冻结入口decoder/同期EMA decoder与功能损失；不授权历史图、U隐藏标签、真实U阶段或自动扩展序列。正在运行的center_spectrum不改动。本实验直接服务主线，不做baseline优化。

## 本阶段与条件正式阶段

先从共同源163/S†69空opt按原GRP训练64步，seed20270301，随后丢弃。复用同一训练engine，原CE/strong/lambda_R=.1144953842725508、P/Q/W_R固定、encoderpolar、SGD/EMA.999完全不变。仅当前L2+14Limage-only，64步终点不做验证。首次执行开始8h时钟，所有后续共享该钟和总12320optimizer硬cap（64探针+64工程+6*32资格+12000正式）；不得重置。P0仅有PROBE:formal=64相位许可，未提前许可正式相位。P0未通过不可辨识门槛则停止，不临时改门槛或放大权重。

P0通过后，才完善并冻结B0原GRP/B-EMA/B-HIST×两新流×2000的实现与分阶段调用账，在当前原钟预算内推进。当前不宣称正式阶段已实现或已启动。三组同状态共同正确mask定义，跨轨迹支持不保证相同。固定最后student2000主、EMA辅助，不改入口、不中途val/best。

## 实现语义与校准

原parent.features包含decoder，禁止把该输出误当encoder特征。显式复用encoder计算b/e3/e2/e1，detach完整多尺度特征，分别原decoder三层dec3/2/1及conv_logit(stochastic=False)、原bilinear align_corners=True读出。冻结D0来自共同入口完整decoder，EMA来自64步probe EMA完整decoder。当前decoder仍训练。历史encoder不用于生成参照；当前图为合法L2，image-only14标签不读取。

固定校准流20270303、16批L2；共同mask要求D0和EMA均argmax正确。H/E各按所有有效像素平均KL(q||p)，不宏类别平均、不温度或权重扫描。当前CE decoder梯度与H/E梯度能量求和，一次lambda=.25 sqrt(sum||gCE||²/sum||gref||²)，两正式流共用；0.25只是预声明强度，不声称最优。

记录每类GT像素/共同正确覆盖及有有效像素患者数、各类KL贡献、各类梯度平方范数（不把这些非正交范数相加声称总梯度归因）、H/E输出KL和拼接16批梯度余弦/归一化距离。仅输出匿名聚合。当前feature支持不是旧域支持，D0仍受encoder漂移影响，无历史保持保证。

## 预声明可辨识性门槛

每前景类覆盖>=其GT像素5%，且至少3位当前标注患者有有效像素。H/E各前景合计KL贡献>=总KL的5%。三种梯度总平方能量均>=1e-12；lambda有限且在[1e-4,10]。H/E共同mask上平均KL>=1e-8，拼接梯度归一化距离sqrt(2-2cos)>=.05。这些为可辨识性工程/开发下限，不是临床标准；门槛在看本轮测量前冻结，不达则NOT_IDENTIFIABLE，不看val挑阈值。

第一校准批手工多尺度decoder与原native forward逐元素相等；所有新增项encoder直接梯度None，decoder梯度有限，模型及两参照参数/buffer/RNG全程不变。注意正式阶段globalclip可能间接改变encoder实际更新，不能宣称其位移不变。

## P0独立调用账与硬约束

probe复用原preflight，64真实optimizer/512polar矩阵SVD，另原CPU/CUDA10小矩阵SVD自检。probe医疗cap native300/encoder3200/reverse70/diagnosticreverse2；预期native140、encoder3168、reverse67、diagnostic2，检查实际账不擅自追加。校准单列持久measurement_calls：encoder32、decoder96、native2（第一批重构核验）、reverse176（16批×CE/H/E及8逐类项共11）。每调用前pending、完成后executed，异常不盲重放。无校准optimizer、无验证调用。

仅pro5000GPU3，/data/bmw明确例外create-only gyro_decoder_function_v1_20261022，中性入口；实际mount/write-read/free/GPU余量通过才启动，可与旧轮共存但不干扰。预计全轮<1GB，3.5GB保护+.5余量<4GB含2MBops，8h原钟7h不新stage7h45stop，timeout27899TERM899KILL。若P0通过，正式额外decoder与求梯度预算须在启动前单列，不能只用12320更新宣称同算力。

正式科学门槛预声明：HIST−B0旧均>=+.3且各>0，新各>=−.3；HIST−EMA旧均>=+.2且各>0，新各>=−.3；HIST入口旧各>=−.5，旧入口>1pp人数比B0各至少少2，>2不增，p10>=B0−.3，全满足才综合通过。保留原源→入口−.933692及原源79.867407%参考线；患者CI不覆盖训练随机/重复开发。前景支持或参照不可辨识记NA，不能当作方案性能阴性。正式失败停止本配方，不扫温度/lambda/EMA，不自动进入真实U阶段。
