# centroid中心化谱目标：冻结协议 V1

2026-10-22首次执行前冻结。用途：直接改进主线GRP的谱辅助目标，非优化论文baseline；独立设计、非文献复现、不声称新颖性。

## 依据、反证与预测

原谱作用于未中心化的64×32 centroid矩阵C，既依赖行间差异，也依赖共同均值分量。历史P0只做过中心化统计，没有训练中心化目标；当时G00未中心化r*约27.18、中心化r*约26.91、协方差trace约.807，不支持“G00已严重坍缩”作为本轮前提，也不是当前GRP诊断。已失败的弱视图detach不再继续。

本轮仅检验去均值后的谱目标能否改善旧域患者尾部：H C=C−mean_rows(C)，L_C=1−||(2/3)HC||*/sqrt(64*32)。共同平移在谱输入层面不改变该目标，梯度行均值为零；不等价于原始图像平移不变，也不证明原共同分量有害。去均值可能丢失任务相关信息，是明确失败风险。

均值参与反向，没有detach、白化、方差/协方差新损失、行重归一化、能量补偿或混合系数。每图4位置、16图64行一并去均值，不按患者/类别重采样。分母仍32=min(63,32)，无新阈值。损失尺度会变化，固定lambda是该单一方案的一部分；不以结果重标定lambda追正。

## 两组与共享训练

两新流20270201/20270202×GRP/GRPC；原GRP对照，GRPC只替换谱输入为HC。所有组原CE+lambda_R=.1144953842725508谱项、原strong含cutout、三视图centroid .5/.25/.25全梯度。相同源163/S†69空opt、P/Q/W_R/bias只读不重抽校准，decoder自由、sigma/grad_update固定。原两SGD组lr.001 Nesterov.9 matrixwd.0005vector0clip1，普通momentum不投影，EMA.999仅辅助。

全组8encoder矩阵polar，float32thinSVD CUDAgesvd，全数值支持tau=maxshape eps32smax，Dprime=||D||F/sqrt(r)UrVrT零→零；理论rtol2e-5、实际<=1e-7+.005proposalnorm；不匹配跨轨迹能量。没有宏CE/归一化/decoder约束/weakdetach。

仅Philips train-L2监督+14Limage-only，48encoder样本/步、64行P32谱。无Uaccessor/隐藏GT/旧图teacher/replay/bank/generator，不改HDF5/manifests/splits/env。reads[2000,14000]后者是Limage-only。fixedstudent2000主，无中途val/best；4组结束各eval2892native=student旧544新660源train230目标train254+EMA旧544新660。开发验证结果用于方案淘汰，测试封存。

## 工程检查与硬账

CPU/CUDA各5polar小矩阵检查共10SVD；每设备4次合成谱SVD（中心化、平移后、常量、原矩阵）共8，合计18数值SVD、2次合成autograd，0医疗模型/optimizer。检查共同向量平移不变、常量输入loss1、梯度有限非零且行均值近0、能量分解、中心化确实改变非中心化目标。常量不做反向，不把秩退化点的任意SVD梯度当数学唯一导数。

64detach MINIMAL+P+C vsS+P+C模型/opt/EMA/RNG逐步exact丢弃；4资格各32含golden/单批/非空恢复重放/两例overfit，全PASS才正式。总8192optimizer=8000formal128qual64detach，functional0；全8192polar×8=65536矩阵SVD，加18数值=65554（此账不包含既有几何目标SVD）。每次几何谱调用记录事件，16104=64detach+4*6qual+4*4004formal；对应16104次既有几何SVD，因此这两类加数值合计81658，不把polar账误称全部算子调用。额外合成反向2另列。

医疗调用cap detach160/1632/64 qual384/768/160 formal16200/385000/8016 diagreverse40 evalnative11568；预期actual128/1536/64、320/576/140、16048/384384/8012、40、11568。20固定诊断step0/499/999/1499/1999复用forward。每谱调用记录group/phase/64rows/是否center/total-common-residual能量和分解误差/输入行均值最大值，0额外模型或SVD，不存图/特征/ID。误差<1e-6、中心化输入行均值max<1e-6。

仅pro5000 GPU3 UUID GPU-9ea1d4ae-e403-cb3b-29b3-c63b083bf8ff，create-only root /data/bmw/SSL_CL_seg/runs/gyro_center_spectrum_v1_20261022，明确/data/bmw例外。启动实际mount/free/write-read/GPU检查，存储包装器，中性父子argv。预计400MB，3.5GB写保护+.5余量<4GB含2MBops；8h原钟7h不新stage7h45stop，timeout27899TERM899KILL。失败/恢复共享账时钟，pending异常不盲重放，不删历史。

## 门槛、分析与停止

GRPC−GRP旧两流均>=+.3pp且每流>0、新每流>=−.3；GRPC入口旧每流>=−.5。旧入口>1pp损失人数相对GRP每流至少少2，>2pp不增，p10不低于GRP超过.3。全部才综合PASS；两组不估计polar交互。主门槛失败停止中心化及其均值混合/补偿扫描，不换EMA终点。

完整报告class/patientEDES/trainval/EMA/20诊断及center能量统计；共用23旧31新患者、源163/S†69，原源→入口−.933692继续计。bootstrap10000seed163不覆盖训练随机/多重比较/重复开发，不称独立最终泛化或患者安全。不根据验证患者损失选择训练样本。

全COMPLETE/EXIT0/workers退休/AUDIT不存在后仅一次center_spectrum_audit，CPU4中性0forward；核8192opt/8192polar/65554数值加polarSVD/16104几何事件/20diag/共同空opt/固定P/final=latest有限optRNG/readsEDES预算。异常停止保存证据。源码协议与匿名聚合经代理公开，remoteSHA/匿名HTTP/local+remote回执齐才称交付。唯一ssl30min健康静默，结果后自主分析主线后继，所有既有负证据和停止边界保留。
