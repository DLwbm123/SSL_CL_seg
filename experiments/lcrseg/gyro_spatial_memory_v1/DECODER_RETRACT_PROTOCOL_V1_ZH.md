# 解码器输出滤波器范数约束 × 全支持encoder均衡：冻结协议 V1

2026-10-22，首次实际运行前冻结。独立问题，不修改能量支持轮和此前所有终点、失败与预算。

## 依据与可证伪假设

上一轮固定decoder复合失败：GRPF−GRP旧均+.163911pp不足+.3，新−.986123/−1.083015pp两患者CI均负；旧>1pp损失人数虽下降，p10两流变差且第二流>2pp增加。固定decoder下polar仍有旧增益及点交互，但不能抵消新域代价。故不扫描decoder学习率。

已有终点零forward诊断显示正常更新decoder存储张量首轴范数平均下降约.2%–1.2%，单行最大6.3%；转置卷积首轴是输入方向，不能把此观察叫输出滤波器尺度。这只是尺度漂移线索，不是有害因果。新假设：保持decoder输出滤波器范数到入口，同时放开方向和bias，能避免完全固定的适应损失并改善旧尾部。若旧增量、尾部或新非劣任一失败，则拒绝本设计，不扫描半径/混合系数。

本轮独立设计，非MuLoRA复现；该文固定正交LoRA/历史白化与此不同。不声明新颖性。新的目标范数仅由已有共同入口权重计算，不访问新源数据。

## 四组和唯一干预

新流20261211/20261212，每流GR、GRP、GRR、GRPR。相同源163/S†69入口、空SGD和固定旋转P32谱，所有组CE+lambda_R .1144953842725508*GR。原P/Q/W_R/bias只读，不重抽或校准。

GR普通SGD；GRP仅encoder全数值支持等F极分解；GRR仅decoder输出滤波器范数约束；GRPR同时二者。所有组原矩阵/向量两optimizer组、学习率.001、Nesterov.9、矩阵wd.0005/向量0、globalclip1；所有原生允许参数均训练，sigma.weight/grad_update固定。原普通momentum不投影。bias及norm参数完全自由，不固定decoder。

R在普通SGD后对10个decoder可训练卷积weight逐输出滤波器重标定。普通Conv2d按首轴输出；groups=1的ConvTranspose2d先交换0/1轴，使输出通道为行，再flatten。rho_j=入口对应行L2范数，Wtilde_j为SGD后行，Wnew_j=rho_j*Wtilde_j/||Wtilde_j||。rho=0行设零；rho>0但当前行范数为零/非有限立即失败。此约束只锚定范数，方向自由；不承诺保留全模型位移能量、层函数尺度或抵消normalization自由度。允许新任务通过方向/bias适应是待检验假设。

10个weight的存储形状依次(128,64,2,2),(64,128,3,3),(64,64,3,3),(64,32,2,2),(32,64,3,3),(32,32,3,3),(32,16,2,2),(16,32,3,3),(16,16,3,3),(4,16,3,3)。三个up为转置卷积；其余普通conv。固定8个encoder/bottleneck polar矩阵与上轮相同：D为含momentum/decay/clip的SGD提议，float32薄SVD CUDAgesvd，tau=maxshape*eps32*smax，全数值支持Dprime=||D||F/sqrt(r)UrVrT，零→零。理论energy rtol2e-5、实际误差<=1e-7+.005proposalnorm。

R每物理step记录10矩阵correction norm、转轴标记、行数、最大相对norm误差（≤2e-5），新增均记8192步/81920矩阵，0额外SVD。约束在encoder后置诊断写入前完成，诊断实际位移包含两类干预。EMA统一原.999，不作R重投影；EMA不声称固定norm。输出范数只要求student。

## 数据与终点

严格复用Philips train-L白名单：2监督+14 L池image-only，3view48encoder、64位置行P32谱；无U accessor/隐藏标签/旧源图/teacher/replay/bank/generator/额外统计。内部U_reads14000仍是L池image-only计数。冻结HDF5/splits/旧环境不改。最终student2000唯一主终点，EMA仅辅助；无中途验证/选best，退出训练后固定评估。共享源与23旧/31新验证患者，双流不是独立源重复。

## 工程和硬计费

CPU/CUDA各5小矩阵polar自检共10 SVD，0医疗forward0optimizer；另各设备普通/转置布局的R正尺度不变、近似幂等、零参考和坍缩拒绝检查，0SVD0optimizer。64 detach为MINIMAL+P+R vs S+P+R逐步model/opt/EMA/RNG exact discarded；8组资格各32实际更新含onebatch/golden/非空恢复重放/两样本overfit，全PASS才正式。

总16320opt=16000formal+256qual+64detach，functional/toy0。8192polar步=8000formal+128qual+64detach→65536 SVD+10数值=65546。8192R步同样分解，每步10矩阵→81920矩阵retraction，无额外forward/SVD。失败、恢复重放记原账原钟，pending异常不盲重放。每步norm/finite异常立刻失败。

Calls cap：detach native160/encoder1632/reverse64；qualification768/1536/320；formal32400/770000/16032；diagnostic reverse80；evaluation native23136。总native56464 encoder773168 reverse16496。预期成功实际detach128/1536/64、资格640/1152/280、formal32096/768768/16024、diag80、eval23136。40固定step0/499/999/1499/1999诊断，复用forward不选参数。8组每组eval2892=student旧544新660源train230目标train254+EMA544660。

## 预声明科学门槛

主比较GRPR−GRP：旧两流均值≥+.3pp且每流>0；新每流≥−.3pp。GRPR相对入口旧每流均值≥−.5pp；旧入口损失>1pp人数比GRP每流至少少2人，>2pp人数不得增，入口p10不得比GRP低超过.3pp。另GRPR−GRR旧两流均值≥+.2pp且每流>0、新每流≥−.3pp，用于检验全支持均衡超越单纯decoder范数约束。上述全满足才复合PASS。

交互(GRPR−GRR)−(GRP−GR)：旧每流>0且均≥+.2、新每流≥−.3时仅称固定两流点估计交互，不作因果普适或临床安全声明。分列GRR−GR、GRP−GR、GRPR−GR。完整报告患者均值/中位/p10/改善下降/>.5/1/2pp/LV-MYO-RV/ED-ES、配对bootstrap10000 seed163、train-val、EMA与位移。原源→入口−.93369236648307pp仍计入。患者CI不涵盖训练随机性、多重比较和反复开发。

## 资源、停止、交付

仅pro5000 GPU3 UUID GPU-9ea1d4ae-e403-cb3b-29b3-c63b083bf8ff。新create-only root /data/bmw/SSL_CL_seg/runs/gyro_decoder_retract_v1_20261022，明确/data/bmw例外。启动前实际mount/capacity/write-read/GPU显存核查，旧数据权重只读共享。预计新增<700MB；3.5GB写保护+.5GB余量，总严格<4GB（含2MB中性ops），不删除旧结果。

首次实际执行原钟8h，7h不新stage、7h45m主动停止；独立timeout27899秒TERM+899秒KILL，8h前结束全部本轮子进程。失败/续跑不重置原钟、预算；不结束他人进程。所有父子argv中性。资格失败、非有限数值、decoder范数约束失效/固定投影改变、能量超差、pending异常或预算到限即停止并保留证据。

全部COMPLETE/EXIT0/worker退休后仅一次decoder_retract_audit，CPU4、0模型forward，审16320计费/8192R81920矩阵/8192polar65536+10SVD/40诊断/共同入口空opt/final=latest finite optRNG/reads[2000,14000]/固定投影/最终student输出范数/EDES及资源。只发布代码、协议、匿名聚合，不公开ID/图mask/权重/私有日志配置；GitHub显式代理、remoteSHA与匿名访问、本地远程回执齐后称交付完成。唯一ssl30分钟监测，健康静默；结果后自主后继，不改本轮门槛。
