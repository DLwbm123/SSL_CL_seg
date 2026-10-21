# 等能量更新谱干预 V1：冻结协议

冻结于 2026-10-21；首次实际执行前保存至新 create-only run root。既有方向对照已完成且 AUDIT PASS，不修改旧终点。

## 证据、问题与归因边界

方向对照两流 G−GR 旧域 −0.630/−0.790 pp，19/23、20/23 患者下降，方向优势门槛失败。GR 相对入口旧域仍有 8/9 人下降 >1 pp、6/4 人下降 >2 pp。问题是更新的谱集中是否是可干预因素；当前并没有观测证明它造成患者损失。

MuLoRA（OpenReview nu3KHzX29K，https://openreview.net/forum?id=nu3KHzX29K）本地 2026-10-18 摘要提到历史白化和动量更新的近似极分解。在线全文受验证阻塞，未核对其公式或代码。本轮不复现 MuLoRA/Muon、不加入 LoRA/历史白化；采用独立定义、可检查的全模型 encoder 位移干预，不能把论文的定理或性能归于本实现。

## 单模块与四组

两条新目标流 20261111/20261112；每流 S、SP、GR、GRP，从相同源163/S†69步入口、空 optimizer 开始，各最后 student2000 为主。S/GR 是新流匹配参照，不能拿旧流结果代替。SP/GRP 在各自损失下仅开启以下 encoder 矩阵位移干预。

保留 SGD(.001, Nesterov .9)、matrix decay .0005/vector0、全局 clip1 和 EMA .999。先生成正常 SGD 位移 D=theta_proposed−theta_before（包括 momentum、decay、clip 的作用），对八个 encoder/bottleneck 卷积权重逐层 reshape 为 out×(in*kh*kw)。float32 薄 SVD D=U diag(s) V^T，CUDA 固定 gesvd。数值支持阈值 tau=max(m,n)*eps32*s_max，r=count(s>tau)。D'=||D||F/sqrt(r)*U_r V_r^T；零矩阵保持零。最终 theta=theta_before+D'。无混合系数、无调参；momentum buffer 保持普通 SGD 的值，encoder bias/归一化参数与 decoder 不变。该操作均衡数值支持内的模式并保留每层提议 Frobenius 能量，不能保证全程任务损失不增加。

S/GR 不做 SVD 或重写参数，沿用原优化器。所有组计算原 G/GR 两个辅助标量以匹配 encoder 前向；S/SP 均断开辅助损失，GR/GRP 使用已冻结 lambda_R=.1144953842725508。既有 W_R、P、偏置只读共享，不重抽 Q、不重新拟合或校准。只匹配当前状态下本层提议能量，不声称不同训练轨迹实际步长完全相同。变换包含 decay 贡献，不能归因为纯梯度谱作用。

## 数据和终点

继续原白名单 Philips train-L：2张监督图+14张 L池 image-only 图、3view48 encoder、64位置行；无U accessor/隐藏GT/旧源训练图/teacher/历史bank/生成器。原 HDF5、患者划分、模型及 P 均只读。投影损失、增广及固定读出沿用 projection_orientation.py，不修改公式。

八组资格各32更新（含实际恢复重放）全部通过才正式训练；新增64更新 detach 对照比较 MINIMAL+polar 与 S+polar，逐步 model/optimizer/EMA/RNG exact。CPU/GPU 数值自检共10次小矩阵 SVD、0模型前向、0更新：能量、零矩阵、数值支持、重复与转置、卷积reshape。资格检查包括 golden、完整非空 optimizer/RNG 恢复和两样本过拟合；失败保留全部已用账本与时钟，不自动改公式、补步或加预算。

每组最后 student2000；EMA统一辅助，不选best、不做中途验证。每组训练结束进程退出后评价：student旧val544/新val660/源train230/目标train254，加EMA旧544/新660，共2892 native；总23136。每组固定step0/499/999/1499/1999诊断，共40点、80额外autograd，不用于调参。

## 冻结成功/否决门槛

主比较 GRP−GR：旧域两流均值 >=+.3pp、每流>0，新域每流>=−.3pp；GRP 相对入口旧域每流>=−.5pp。同时相对入口旧患者损失>1pp人数每流至少减少2人，>2pp人数不增加，患者差值p10相对GR不降低超过.3pp。所有条件才称本轮复合门槛通过。若均值收益与患者代价冲突，按失败报告，保留双方证据。阈值是研究决策阈值，不是临床有效性或安全界限。

SP−S 同样报告均值/患者代价以判断单独更新干预。交互 (GRP−GR)−(SP−S) 仅在旧域两流均正、均值>=+.2pp且新域每流>=−.3pp时称有一致的点估计交互；否则不得声称 GR 特有协同。每个比较完整报告每域患者均值/中位/p10、改善/下降、>.5/1/2pp人数、LV/MYO/RV、ED/ES、配对患者 bootstrap10000 seed163；入口和原源−.933692pp接入损失继续计入。配对患者CI不覆盖训练随机性、反复开发或多重比较，两流共享源与验证患者，不是独立源重复。对统计不确定结果不做临床非劣推断。

每次polar更新记录8层提议/均衡/实际范数、数值rank、stable rank和方向cosine；精确变换能量误差<=2e−5相对，float32应用误差<=1e−7+.005*proposal_norm。原始诊断非患者数据，但只公开聚合摘要。数值失败、非finite、未清pending或预算超限均停止；不得盲重放。

## 硬预算与实际计费

独立原钟上限8小时：7小时不启新stage，7h45m停止主动工作，原8h前KILL全部所属进程。保留退出/保存余量。新增存储严格<4,000,000,000字节，3.5GB写保护+.5GB退出余量，预计<650MB；包括新代码、日志、临时、ckpt、数值检查和2MB中性操作目录。原数据、环境、投影只读共享不复制。GPU仅pro5000 GPU3指定UUID；中性父子argv。

optimizer总16320=16000正式+256资格+64detach；functional/toy更新0，恢复重放计入。polar更新上限8192=8000正式+128资格+64detach；每次8矩阵，更新SVD65536，加数值自检10，总65546 SVD。所有polar reserve/executed持久化，pending非空禁止重入。

|阶段|native上限|额外encoder上限|reverse上限|
|---|---:|---:|---:|
|detach|160|1632|64|
|八资格|768|1536|320|
|正式|32400|770000|16032|
|固定诊断|0|0|80|
|评价|23136|0|0|

总上限native56464、encoder773168、reverse16496；teacher/generator0。预期成功实际detach128/1536/64，资格640/1152/280，正式32096/768768/16024，诊断80，评价23136。此前方向对照成本独立完整保留，不重置或隐去。

完成后仅一次零模型前向审计：全部worker退休、所有EXIT0、16320计数与calls、8192polar步及65536SVD、共同入口空opt、冻结P/rotation/常量、final=latest finite optimizer/RNG/reads[2000,14000]、40诊断点/患者EDES、资源限制。发布源码/协议/匿名聚合报告，禁止患者标识/图mask/权重P/Q/W_R/原始私有日志。然后依据结果继续自主选择新的有限问题，不以单轮失败结束整个研究。
