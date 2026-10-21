# 解码器零学习率 × 全支持encoder均衡：冻结协议 V1

2026-10-22，首次实际运行前冻结。独立问题，不修改能量支持轮和此前所有终点、失败与预算。

## 依据与可证伪假设

90%支持轮GRTP−GRP旧两流−0.187032/−0.432412pp，尾部失败；GRT−GR旧新两域两流均负，患者CI均负。保留全部数值支持的GRP较GR旧均改善，但新域中位负、患者尾部仍损失。停止截断阈值路线，不扫描阈值。已有诊断显示辅助谱梯度不直接作用decoder，但decoder仍有CE位移；这不是decoder损伤因果证据。

本轮假设：固定入口decoder参数能降低旧域患者损失，同时保留全支持encoder均衡的增量效果。反证：新域非劣失败、旧增量不足、尾部不降，或极分解在固定decoder下不能超过固定decoder基线。没有额外文献复现主张；MuLoRA全文仅作为既往谱均衡动机，其LoRA、历史白化和动量变换与本设计不同。新颖性未证明。

## 四组与唯一干预

两条新随机流20261201/20261202，每流GR、GRP、GRF、GRPF。相同源163/S†69入口、空SGD和固定旋转P32谱：所有组CE + lambda_R 0.1144953842725508 * GR。既有P/Q/W_R/bias只读共享，不重新抽样/校准。

- GR：普通SGD，decoder学习率.001。
- GRP：8个encoder/bottleneck卷积矩阵的普通SGD提议位移作全数值支持等F范数极分解；decoder学习率.001。
- GRF：普通encoder SGD，所有native.decoder参数学习率固定0。
- GRPF：同GRP encoder变换，所有native.decoder参数学习率固定0。

F是参数零学习率控制，仍计算decoder梯度和momentum，参与同一globalclip；不声称消除decoder梯度计算。所有组均把optimizer原矩阵/向量两组拆成encoder/decoder四组，参数顺序各自保持、weight decay不变。F的decoder数值state每次物理step前后与入口exact比较，不允许运行缓冲变化；EMA decoder直接复制固定student，避免相同数值混合带来的浮点漂移。未锁定组使用原EMA.999。encoder各层bias/normalization仍正常更新。所有可训练参数集合保持原生，sigma.weight/grad_update原固定。除F decoder学习率外，原SGD .001/Nesterov .9/矩阵wd .0005/向量wd0/globalclip1不变。保留普通momentum，不把投影位移写回momentum。

P变换与已完成更新谱轮完全相同：D=普通SGD后theta−更新前theta，包含momentum/decay/clip；按out×inKH KW float32薄SVD CUDA gesvd，保留s>max(m,n)*eps32*smax的全部数值支持r；D'=||D||F/sqrt(r) U_r V_r^T；零→零。八矩阵形状(16,1,3,3)/(16,16,3,3)/(32,16,3,3)/(32,32,3,3)/(64,32,3,3)/(64,64,3,3)/(128,64,3,3)/(128,128,3,3)。每步记录rank/stable_rank/cosine/提议与实际norm；理论能量rtol2e-5、实际float32误差≤1e-7+.005*proposal_norm。不同轨迹能量不保证一致，F去掉decoder位移故不声称全模型等能量。

## 数据与终点

严格复用Philips train-L白名单：2监督+14 L池image-only，3view48encoder、64位置行P32谱；无U accessor/隐藏标签/旧源图/teacher/replay/bank/generator/额外统计。内部U_reads14000仍是L池image-only计数。冻结HDF5/splits/旧环境不改。最终student2000唯一主终点，EMA仅辅助；无中途验证/选best，退出训练后固定评估。共享源与23旧/31新验证患者，双流不是独立源重复。

## 工程和硬计费

CPU与CUDA各5小矩阵SVD检查共10，0医疗forward、0optimizer。64 detach实际更新：MINIMAL+polar+F 与 S+polar+F逐步model/opt/EMA/RNG exact，discarded。8组资格每组32实际更新，含one-batch、golden、非空optimizer恢复重放、两样本overfit，全部通过后才正式。F参数每步exact检查覆盖detach/资格/正式，预计8192条；失败立即停止，不增加重试cap。

总optimizer16320=16000正式+256资格+64detach；functional/toy0。全支持P变换8192步=8000正式+128资格+64detach，每步8 SVD=65536，另10数值SVD，总65546。reserve/executed/pending持久化，失败/恢复重放计原账原钟，不盲目重复。40固定诊断step0/499/999/1499/1999，每组5，复用forward；F decoder实际位移必须0，不用诊断选参数。

Calls上限：detach native160/encoder1632/reverse64；qualification native768/encoder1536/reverse320；formal native32400/encoder770000/reverse16032；diagnostic reverse80；evaluation native23136。合计native56464/encoder773168/reverse16496。预期成功实际detach128/1536/64、资格640/1152/280、正式32096/768768/16024、diag80、eval23136。所有组匹配辅助G/GR计算，S仅用于detach断开对照。

每组评估2892=student旧544/新660/源train230/目标train254 + EMA旧544/新660，8组。无需重评旧模型。

## 预声明科学门槛

主比较GRPF−GRP：旧两流均值≥+.3pp且每流>0；新每流≥−.3pp。GRPF相对入口旧每流均值≥−.5pp；旧入口损失>1pp人数比GRP每流至少少2人，>2pp人数不得增，入口p10不得比GRP低超过.3pp。另GRPF−GRF旧两流均值≥+.2pp且每流>0、新每流≥−.3pp，用于检验全支持均衡超越单纯decoder零学习率。上述全满足才复合PASS。

交互(GRPF−GRF)−(GRP−GR)：旧每流>0且均≥+.2、新每流≥−.3时仅称固定两流点估计交互，不作因果普适或临床安全声明。分列GRF−GR、GRP−GR、GRPF−GR。完整报告患者均值/中位/p10/改善下降/>.5/1/2pp/LV-MYO-RV/ED-ES、配对bootstrap10000 seed163、train-val、EMA与位移。原源→入口−.93369236648307pp仍计入。患者CI不涵盖训练随机性、多重比较和反复开发。

## 资源、停止、交付

仅pro5000 GPU3 UUID GPU-9ea1d4ae-e403-cb3b-29b3-c63b083bf8ff。新create-only root /data/bmw/SSL_CL_seg/runs/gyro_decoder_lock_v1_20261022，明确/data/bmw例外。启动前实际mount/capacity/write-read/GPU显存核查，旧数据权重只读共享。预计新增<700MB；3.5GB写保护+.5GB余量，总严格<4GB（含2MB中性ops），不删除旧结果。

首次实际执行原钟8h，7h不新stage、7h45m主动停止；独立timeout27899秒TERM+899秒KILL，8h前结束全部本轮子进程。失败/续跑不重置原钟、预算；不结束他人进程。所有父子argv中性。资格失败、非有限数值、固定decoder/投影改变、能量超差、pending异常或预算到限即停止并保留证据。

全部COMPLETE/EXIT0/worker退休后仅一次decoder_lock_audit，CPU4、0医疗forward，审全部计费/8192固定decoder检查/8192变换65536 SVD+10数值/固定40诊断/共同入口空opt/final=latest finite optRNG/数据及EDES。只发布代码、协议、匿名聚合结果；不公开私有行/ID/图像/权重P/Q/W_R/原日志/配置。GitHub显式代理、remoteSHA与匿名访问及本地远程回执齐后称交付完成。唯一ssl30分钟监测，健康常规静默；结果后自主分析独立后继问题，不修改本轮门槛。
