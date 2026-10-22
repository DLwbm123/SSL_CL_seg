# 当前弱视图停止梯度参照：冻结协议 V1

2026-10-22首次实际运行前冻结。研究用途是直接改进主线GRP谱约束的梯度路径；GRP为同期主线参照，不是新增优化论文对比方法。本轮不增加baseline搜索或通用评测建设，非论文复现，不声明新颖性。

## 问题与唯一干预

view_occlusion完整结果主旧均−0.311763pp、新均−0.156298pp，四个患者CI均负，故停止去遮挡/遮挡强度扫描，恢复原增强。代码中谱输入为归一化投影行的centroid C=.5z_w+.25z_s0+.25z_s1，谱损失通过三个分支共同更新encoder。原弱图可能随强增强共同移动；这是结构性可能，不是本轮负结果证明的因果。

新假设：C_A=.5stopgrad(z_w)+.25z_s0+.25z_s1，固定当前弱视图分支作为两强视图参照，可以减少谱约束对原图表示的漂移并改善旧域尾部。两者同状态前向值完全相同，仅弱视图分支梯度为零，强视图局部导数不变。参照每步重算，模型共享参数仍经监督及强视图更新；不等于冻结弱图表示、不含EMA teacher、历史图或新监督。停止梯度产生的是所声明的梯度路由，不声称对未停止的同一标量函数求完整导数。

只两组GRP/GRPA×新流20270121/20270122。GRP原谱+polar，GRPA上述弱停止梯度+同polar。无去遮挡、宏CE、参数/输出梯度归一化、decoder锁定或回缩。两组相同源163/S†69、空SGD、固定P/Q/W_R/bias，原CE+lambda_R=.1144953842725508*(1-||(2/3)C||*/sqrt(64*32))，不重校准权重、不扫学习率/梯度补偿倍数/centroid权重。

## 冻结训练与数据

仅Philips train-L白名单2监督+14Limage-only；原strong两强视图，原4位置/图、64行P32谱、每步48encoder样本。无Uaccessor/隐藏GT/旧源图/replay/bank/generator，reads[2000,14000]中后者仍是Limage-only。HDF5/manifests/splits/env只读。源入口和所有历史失败保留。

所有既有允许参数训练，decoder自由，sigma/grad_update固定；原矩阵/向量两SGD组lr.001、Nesterov.9、矩阵wd.0005/向量0、clip1、普通momentum不投影；EMA.999仅辅助。原8encoder矩阵polar全数值支持tau=maxshape eps32 smax，float32thinSVD CUDAgesvd，Dprime=||D||F/sqrt(r)UrVrT，零→零。理论能量rtol2e-5、实际<=1e-7+.005proposalnorm，不匹配跨轨迹能量。

fixedstudent2000唯一主终点，无中途val/best。每组结束固定评估student旧544/新660/源训练230/目标训练254及EMA旧544/新660，共2892native样本；4组总11568。开发验证结果用于淘汰机制，不使用测试集调参；共享验证集反复开发的偏差明确披露，不声称独立最终泛化测试。

## 验证、账本与资源

CPU/CUDA原5polar自检共10SVD；每设备两次合成autograd共4合成反向，验证centroid前向逐元素相等、weak梯度None、两strong梯度与原式逐元素相同、解析导数正确，0医疗forward/optimizer。64detach MINIMAL+P+A vs S+P+A逐步model/opt/EMA/RNGexact，discarded。4资格各32更新含golden/onebatch/非空恢复重放/两例overfit，全部PASS才正式；所有资格和重放计入原账。

总8192optimizer=8000formal+128qual+64detach，functional0；全部8192polar，65536SVD+10数值=65546。额外合成反向4单列。医疗调用cap：detach native160/encoder1632/reverse64；qual384/768/160；formal16200/385000/8016；diagreverse40；evalnative11568。预期actual detach128/1536/64；qual320/576/140；formal16048/384384/8012；diag40；eval11568。20固定诊断step0/499/999/1499/1999，0额外医疗调用。每次centroid记录group/phase/64rows/weak_anchor，G及GR各一次；预期64detach+4*6qual+4*4004formal=16104事件，不存图/患者ID/特征。

只pro5000 GPU3 UUID GPU-9ea1d4ae-e403-cb3b-29b3-c63b083bf8ff、/data/bmw明确例外，create-only root gyro_weak_anchor_v1_20261022。首次执行前mount/free/write-read/GPU检查。预计400MB，3.5GB写保护+.5GB余量，总<4GB含2MBops。8h原钟，7h不新stage、7h45stop，timeout27899TERM899KILL确保8h内退出；所有重试共享时钟账，不重启避预算，pending异常不盲重放。父子中性argv，经既有存储包装器。不动他人/历史资产。

## 预声明门槛与停止

主GRPA−GRP旧均>=+.3pp且每流>0、新每流>=−.3；GRPA相对入口旧每流>=−.5。旧入口下降>1pp人数相对GRP每流至少减少2，>2pp不增加，p10不低于GRP超过.3pp。全部满足才综合PASS；两组不估计polar交互。

报告主/EMA、LV/MYO/RV、ED/ES、患者中位/p10/改善下降/>.5/1/2pp、train-val和20梯度诊断。10000患者bootstrap seed163；共享23旧31新患者、同源入口；原源→入口−.933692pp继续计入。CI不涵盖训练随机、多重比较、反复开发，不是患者安全保证或独立源复现。若主门槛失败停止弱停止梯度及其缩放/混合扫描，不换EMA主终点。

数值/资格/固定常量/能量/恢复一致性或硬预算失败停止本轮保存证据。完成后仅一次weak_anchor_audit，CPU4中性entry0forward；需全COMPLETE/EXIT0/workers退休/AUDIT不存在，核8192opt65546SVD/16104事件/20diag/有限final=latest optRNG/空入口/固定投影/EDES/资源。源码、协议、匿名聚合完整报告经代理公开，local+remote回执及匿名HTTP和remoteSHA齐才称完成。唯一ssl30min健康静默、结果自主分析推进主线。
