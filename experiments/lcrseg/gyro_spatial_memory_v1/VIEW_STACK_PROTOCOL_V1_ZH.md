# 加权视图堆叠谱：冻结协议 V1

## 主线问题

center_spectrum完整两流主增量和患者尾部均FAIL，停止中心化及补偿扫描。另立独立假设：先对三视图特征求均值再求谱可能抵消视图差异；保持各视图二阶矩可能改善主线表示与旧域尾部，也可能保留增强噪声、损失隐含对齐而变差。历史结果不证明此因果。

两新流20270401/20270402，各GRP原方案与GRPV候选，共4组。原64行三视图z_w,z_s0,z_s1及w=(.5,.25,.25)。GRP C=sum w_i z_i。GRPV V=concat(sqrt(3w_i) z_i)，形状192×32。两者原谱函数1-||(2/3)X||*/sqrt(rows*32)，lambda_R=.1144953842725508保持。V^T V/3=sum w_i z_i^T z_i；其与C^T C差为加权视图离差二阶矩。三视图相同时值与各视图梯度匹配；一般情况下不保证梯度能量相同，不做能量补偿/重校准。sqrt3只是使原rows分母与三视图权重相容，不是调参。

不中心化、不detach、不白化、不新增teacher、对齐损失或历史数据。不改strong（包括cutout）、P/Q/W_R/bias、16图4位置、CE、源163/S†69共同入口空opt、SGD lr.001 Nesterov.9 matrixwd.0005vector0clip1/EMA.999。两组全支持8encoderpolar，普通momentum不投影，decoder自由（sigma/grad_update固定）。只Philips L2监督+14L image-only，48encoder样本；原G/GR均计算，只有GR参与损失。GRP谱64行，GRPV192行。相同图像调用不等于相同计算量：候选几何SVD矩阵行数三倍，实际耗时另报。

固定student2000主、EMA辅助；无中途val/best，测试封存。当前13标注患者、23旧31新验证患者及ED/ES定义不变。无Uaccessor隐藏GT/旧图/replay/bank/generator，reads[2000,14000]后者为L image-only。

## 工程与预算

先CPU/CUDA数值检查：原polar10 SVD，加权堆叠每设备4 SVD，共18；6次合成反向。检查相同视图值/梯度匹配、二阶矩恒等式、权重能量、全视图有限非零梯度、与centroid不同。零医疗调用、零optimizer。64 detach精确检查MINIMAL与只计算断开辅项，model/opt/EMA/RNG逐步一致；再4×32资格，含golden/单批反向/非空恢复重放/两例overfit，全通过才正式。复用唯一原engine。

总8192真实optimizer=64detach+128资格+8000正式；8192polar65536矩阵SVD+18数值=65554，另16104几何谱，合计81658 SVD；候选矩阵192×32，对照64×32。20固定诊断0/499/999/1499/1999，4×2892评测native样本。调用cap detach160/1632/64，资格384/768/160，正式16200/385000/8016，diagreverse40，evalnative11568；预期128/1536/64、320/576/140、16048/384384/8012、40、11568。持久pending/reserved/executed，异常不盲重放。

stack_events 16104条=64detach+4×6资格+4×4004正式G/GR，各条记录行数、second/centroid/within energy与恒等式相对误差<1e-6，不新增模型/SVD，均为匿名聚合。两流都报告，不能只保留通过流。

仅pro5000GPU3，create-only /data/bmw/SSL_CL_seg/runs/gyro_view_stack_v1_20261022；实际挂载/写读/容量/空余>=18000MiB才启动，其他进程不动。预计<500MB，保护3.5GB+0.5GB余量<4GB（含2MBops），首执行起8h原钟，重试共享，7h不新stage7h45stop，timeout27899TERM+899KILL。每轮全体结束才一次CPU零forward最终审计，不重复旧实验。

## 主科学门槛

GRPV−GRP旧两流均值>=+.3pp且每流>0，新每流>=−.3pp；候选入口旧每流>=−.5pp；入口旧下降>1pp人数每流比GRP至少少2，>2pp不增加，p10>=GRP−.3pp，全部满足才综合PASS。不估交互，不换EMA主终点。完整类/患者/EDES/trainval/梯度/二阶矩统计，患者bootstrap10000seed163。原源→入口−.933692pp仍计，CI不覆盖训练随机、反复开发、多重比较，不宣称临床安全。

失败停止本堆叠配方及视图权重/行数/谱尺度扫描，不复活中心化。若未达到门槛，保留负结果后再选择有独立机制依据的主线问题。
