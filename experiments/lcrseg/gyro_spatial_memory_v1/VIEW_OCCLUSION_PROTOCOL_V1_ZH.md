# 去遮挡光度增强 × encoder均衡：冻结协议 V1

2026-10-22首次实际执行前冻结。独立增强机制消融，非论文复现、不声明新颖性。停止宏CE及其范数匹配支线；本轮全部回到原CE，不扫描类别权重或增强强度。

## 依据与可证伪假设

宏CE及输出梯度范数匹配均失败；后者减少了固定诊断点的频繁global clipping，但新域相对原CE仍均−3.903691pp。故不继续这一损失族。现有prior.strong在2次固定随机光度操作后，用128灰度覆盖边长min(H,W)/2、边界截断的矩形；原几何仍对weak/strong0/strong1相同空间位置形成centroid。代码事实提示遮挡位置的局部语义可能不对应，但尚未证明它损害旧域。

新预测：只移除这个最终cutout，保留全部光度操作和随机数序列，可在GR谱约束下保留更多局部内容，改善旧域患者尾部且不损害新域适应。若不满足预声明主收益/尾部/新非劣，则拒绝这个设计，不扫描遮挡尺寸、概率或增强强度。光度变换仍可能强烈改变像素，不声称去遮挡就保证医学语义不变。

## 四组两新流

seed20270111/20270112，每流GR、GRP、GRV、GRPV。GR原strong与普通SGD；GRP原strong+encoder全支持polar；GRV去最终cutout、普通SGD；GRPV去cutout+polar。所有组原监督pixelCE.mean+lambda_R=.1144953842725508固定旋转P32谱，入口源163/S†69、空SGD，P/Q/W_R/bias只读，不重抽校准。

V只作用于几何分支两张强视图，监督2张L输入不变。完全相同的uint8量化、每slot局部generator seed与2次光度op/level/sign，包括identity/autocontrast/equalize/brightness/contrast/sharpness/posterize/solarize；仍消耗原cutout坐标随机数，只不覆盖像素。4个空间位置/slot及三视图centroid权重不变，位置不重抽、行不删除、不作mask加权；不添加teacher、损失或labelaccess。原组直接调用原strong实现。V生成的mask仅用于代码一致性验证，训练不使用其标签/权重或保存mask。

所有原允许参数训练，decoder自由，sigma/grad_update固定。原SGD矩阵/向量两组lr.001 Nesterov.9 matrixwd.0005/vector0、clip1、普通momentum不投影；EMA.999不换主终点。P原8encoder矩阵float32薄SVD CUDAgesvd，全数值支持tau=maxshape eps32 smax，Dprime=||D||F/sqrt(r)UrVrT，零→零；D含mom/decay/clip。理论energy rtol2e-5，实际误差<=1e-7+.005proposalnorm。不同轨迹不匹配能量。

## 数据、工程与终点

只Philips train-L白名单：2监督+14Limage-only，3view48encoder、64行P32谱，无Uaccessor/隐藏标签/旧源图/teacher/replay/bank/generator。U_reads14000仍是Limage-only。HDF5/splits/env不改。fixedstudent2000主、EMA辅助，无中途验证/选best，固定训练结束后评估。

CPU/CUDA各5小矩阵polar自检共10SVD；两设备×两seed×两view的固定合成图，验证V光度图重新施加同一mask能逐元素重构原strong、mask外严格相同、slot前缀一致、有限且[-1,1]，并验证mask内确有变化。0医疗forward/反向/optimizer。64detach MINIMAL+P+V vs S+P+V逐步model/opt/EMA/RNGexact discarded。8资格各32实际更新含onebatch/golden/非空恢复重放/两样本overfit，全PASS才formal；原CE无变化。

每次strong调用记录group/phase/step/view/images/occlusion_removed，不记图mask/患者ID。预期detach64事件，8资格各6事件，8formal各4004事件，总32144，均16images，每view0或1；事件是既有encoder调用的操作回执，不新增模型计算。40固定梯度诊断step0/499/999/1499/1999。

## 硬账、资源与停止

16320opt=16000formal+256qual+64detach，functional0；8192polar=8000formal+128qual+64detach，65536SVD+10数值=65546。新增医疗forward/encoder/reverse0，合成反向0。调用cap detach160/1632/64 qual768/1536/320 formal32400/770000/16032 diagreverse80 evalnative23136。预期actual detach128/1536/64 qual640/1152/280 formal32096/768768/16024 diag80 eval23136。8组每组固定eval2892=student旧544新660源train230目标train254+EMA旧544新660。

只pro5000 GPU3 UUID GPU-9ea1d4ae-e403-cb3b-29b3-c63b083bf8ff，create-only root /data/bmw/SSL_CL_seg/runs/gyro_view_occlusion_v1_20261022，明确/data/bmw例外并经存储包装器。启动前实际mount/free/write-read/GPU检查。预计700MB，3.5GB写保护+.5GB余量总<4GB含2MBops。首次执行8h原钟，7h不新stage、7h45stop；timeout27899TERM899KILL确保8h前结束本轮子进程。所有父子argv中性，不影响他人。失败/恢复同原钟账，pending异常不盲重放。

数值/视图一致性/资格/固定投影/能量检查失败，或预算到限立即停止并保留证据，不临时改协议追正结果。

## 科学门槛

主GRPV−GRP旧两流均>=+.3pp且各>0、新各>=−.3；GRPV相对入口旧均每流>=−.5；入口旧>1pp人数比GRP各至少少2、>2不增加、p10不比GRP低>.3。另GRPV−GRV旧均>=+.2各>0、新各>=−.3，检验polar超越去遮挡本身。全部满足compositePASS。

交互(GRPV−GRV)−(GRP−GR)：旧每流>0且均>=+.2、新各>=−.3仅称两流点交互；不代表普适因果或临床安全。完整分列GRV−GR、GRP−GR、GRPV−GR，患者中位/p10/改善下降/>.5/1/2pp、LV/MYO/RV、ED/ES、train-val、EMA及40诊断。bootstrap10000seed163，原源→入口−.933692pp仍计。两流共享23旧/31新患者及源入口；CI不含训练随机、多重比较、反复开发，不是独立源重复。

## 完成与交付

全COMPLETE/EXIT0/workers退休/AUDIT不存在时仅一次view_occlusion_audit，CPU4中性entry0forward。核16320opt/8192polar65546总SVD/32144视图回执/40诊断/共同入口空opt/final=latest finite optRNG/reads[2000,14000]/固定P/EDES/资源。不得运行旧三组auditor。源码、协议、匿名聚合报告公开，排除图mask/患者ID/权重/原日志与私有配置。GitHub显式代理，remoteSHA/匿名HTTP/local+remoteDELIVERY齐才称交付完成。唯一ssl30min持续、健康静默；结果后自主后继。
